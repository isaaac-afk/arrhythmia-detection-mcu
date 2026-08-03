#include <stdio.h>
#include <stdint.h>
#include "detector.h"
#include "ecg_data.h"
#include "app_ecg.h"

/* Core clock in MHz. Set to match your CubeMX clock configuration
 * (100 for a 100 MHz SYSCLK). Used only to convert cycles -> ns. */
#ifndef CPU_MHZ
#define CPU_MHZ 84u
#endif

/* --- cycle counter -------------------------------------------------------
 * On the STM32 we use the Cortex-M DWT cycle counter (counts core clocks).
 * On the host (HOST_TEST) we stub it so the logic can be unit-tested. */
#ifdef HOST_TEST
static void     cyc_init(void)      { }
static uint32_t cyc_now(void)       { return 0; }
#else
#include "stm32f4xx.h"
static void cyc_init(void) {
	CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}
static uint32_t cyc_now(void) { return DWT->CYCCNT; }
#endif
/* --- BPM from R-peak spacing (shared: replay + live) -------------------
 * Each fired peak gives a sample index. RR = gap to previous peak.
 * BPM = 60 * fs / RR. We smooth over the last few RR intervals. */
typedef struct {
    long     last_idx;      /* sample index of previous peak, -1 = none yet */
    uint32_t rr[4];         /* recent RR intervals (samples) */
    int      n;             /* how many RRs collected (caps at 4) */
} bpm_state;

static void bpm_init(bpm_state *b) { b->last_idx = -1; b->n = 0; }

/* Call on every fired peak. Returns smoothed BPM, or 0 if not enough data yet. */
static uint32_t bpm_update(bpm_state *b, long idx, int fs)
{
    if (b->last_idx < 0) { b->last_idx = idx; return 0; }  /* first peak: no interval */

    uint32_t rr = (uint32_t)(idx - b->last_idx);
    b->last_idx = idx;
    if (rr == 0) return 0;

    /* shift the last 4 intervals */
    for (int i = 3; i > 0; i--) b->rr[i] = b->rr[i - 1];
    b->rr[0] = rr;
    if (b->n < 4) b->n++;

    uint32_t sum = 0;
    for (int i = 0; i < b->n; i++) sum += b->rr[i];
    uint32_t rr_avg = sum / (uint32_t)b->n;

    return (60u * (uint32_t)fs) / rr_avg;   /* smoothed BPM */
}
/* --- Stage 1.3c: timer-driven feed + ring buffer (MCU only) ----------- */
#ifndef HOST_TEST
#include "main.h"                  /* HAL types */
extern TIM_HandleTypeDef htim2;
extern UART_HandleTypeDef huart2;
extern ADC_HandleTypeDef hadc1;
#define RB_SIZE 64                 /* power of two */
static volatile double   rb[RB_SIZE];
static volatile uint32_t rb_head, rb_tail, rb_overflow;
static volatile uint32_t feed_idx;
static volatile uint8_t  feed_done;
#endif

void app_ecg_run(void) {
    setvbuf(stdout, NULL, _IONBF, 0);   /* <-- add this */
    pt_detector d;
    pt_init(&d);
    cyc_init();

    long ridx;
    uint32_t total_cyc = 0, worst_cyc = 0;
    int n_peaks = 0;

    printf("\r\n=== Stage 1.3: replay %d samples @ %d Hz ===\r\n", ECG_N, ECG_FS);

    for (int i = 0; i < ECG_N; i++) {
        double sample = ecg_samples[i];
        uint32_t t0 = cyc_now();
        int fired = pt_process(&d, sample, &ridx);
        uint32_t dt = cyc_now() - t0;
        total_cyc += dt;
        if (dt > worst_cyc) worst_cyc = dt;
        if (fired) {
            printf("R-peak @ %ld\r\n", ridx);
            n_peaks++;
        }
    }

    uint32_t avg_cyc  = (ECG_N > 0) ? total_cyc / (uint32_t)ECG_N : 0;
    uint32_t avg_ns   = (avg_cyc  * 1000u) / CPU_MHZ;
    uint32_t worst_ns = (worst_cyc * 1000u) / CPU_MHZ;
    uint32_t budget_us = 1000000u / (uint32_t)ECG_FS;

    printf("--- done: %d R-peaks ---\r\n", n_peaks);
    printf("compute/sample: avg %lu cyc (%lu ns), worst %lu cyc (%lu ns)\r\n",
           (unsigned long)avg_cyc, (unsigned long)avg_ns,
           (unsigned long)worst_cyc, (unsigned long)worst_ns);
    printf("real-time budget @ %d Hz = %lu us/sample -> %s\r\n",
           ECG_FS, (unsigned long)budget_us,
           (worst_ns / 1000u < budget_us) ? "PASS (fits)" : "FAIL (too slow)");
}
#ifndef HOST_TEST
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance != TIM2) return;
    if (feed_idx >= ECG_N) { feed_done = 1; return; }

    uint32_t next = (rb_head + 1u) & (RB_SIZE - 1u);
    if (next == rb_tail) { rb_overflow++; return; }   /* full: drop, count it */

    rb[rb_head] = ecg_samples[feed_idx++];
    rb_head = next;
}

void app_ecg_run_timed(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    pt_detector d;
    pt_init(&d);
    bpm_state bpm;
    bpm_init(&bpm);
    cyc_init();

    rb_head = rb_tail = feed_idx = rb_overflow = 0;
    feed_done = 0;

    long ridx;
    uint32_t n_peaks = 0, processed = 0, total_cyc = 0, worst_cyc = 0;

    printf("\r\n=== Stage 1.3c: timer-driven replay, %d samples @ %d Hz ===\r\n",
           ECG_N, ECG_FS);

    uint32_t t0 = HAL_GetTick();
    HAL_TIM_Base_Start_IT(&htim2);

    while (!feed_done || rb_tail != rb_head) {
        if (rb_tail == rb_head) continue;          /* empty: wait for next tick */

        double s = rb[rb_tail];
        rb_tail = (rb_tail + 1u) & (RB_SIZE - 1u);

        uint32_t c0 = cyc_now();
        int fired = pt_process(&d, s, &ridx);
        uint32_t dc = cyc_now() - c0;

        total_cyc += dc;
        if (dc > worst_cyc) worst_cyc = dc;
        processed++;

        if (fired) {
                    uint32_t bpm_val = bpm_update(&bpm, ridx, ECG_FS);
                    if (bpm_val) printf("R-peak @ %ld   inst BPM %lu\r\n",
                                        ridx, (unsigned long)bpm_val);
                    else         printf("R-peak @ %ld   (first)\r\n", ridx);
                    n_peaks++;
                }
    }

    HAL_TIM_Base_Stop_IT(&htim2);
    uint32_t ms = HAL_GetTick() - t0;

    printf("--- done: %lu R-peaks ---\r\n", (unsigned long)n_peaks);
    printf("processed %lu / %d, ring overflows: %lu\r\n",
           (unsigned long)processed, ECG_N, (unsigned long)rb_overflow);
    printf("wall clock: %lu ms (expected %lu) -> %s\r\n",
           (unsigned long)ms, (unsigned long)(1000UL * ECG_N / ECG_FS),
           (rb_overflow == 0) ? "REAL-TIME OK" : "OVERFLOW");
    printf("compute/sample: avg %lu cyc, worst %lu cyc\r\n",
           (unsigned long)(total_cyc / processed), (unsigned long)worst_cyc);
}

/* --- harness self-test: dump ecg_samples[] as raw IEEE-754 hex --------
 * Hex, not %.17g: newlib-nano strips float printf by default, and hex
 * proves bit-identity rather than round-trip-close-enough. ~35 s @ 115200. */
void app_ecg_dump(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("\r\n=== DUMP BEGIN n=%d fs=%d fmt=hex64be ===\r\n", ECG_N, ECG_FS);
    for (uint32_t i = 0; i < ECG_N; i++) {
        union { double d; uint64_t u; } cv;
        cv.d = ecg_samples[i];
        printf("%08lX%08lX\r\n",
               (unsigned long)(uint32_t)(cv.u >> 32),
               (unsigned long)(uint32_t)(cv.u & 0xFFFFFFFFu));
    }
    printf("=== DUMP END ===\r\n");
}

/* --- mode select ------------------------------------------------------ */
void app_ecg_menu(void)
{
    uint8_t c = 0;
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("\r\n=== ecg-mcu ===\r\n");
    printf("  [1] replay, free-running        (Stage 1.3b)\r\n");
    printf("  [2] replay, TIM2-driven 360 Hz  (Stage 1.3c)\r\n");
    printf("  [3] dump samples as hex         (harness self-test)\r\n");
    printf("select within 5 s [default 2]: ");

    if (HAL_UART_Receive(&huart2, &c, 1, 5000) != HAL_OK) c = '2';
    printf("%c\r\n", (char)c);

    switch (c) {
        case '1': app_ecg_run();  break;
        case '3': app_ecg_dump(); break;
        default:  app_ecg_run_timed(); break;
    }
}
#endif
#ifndef HOST_TEST
#define ADC_BUF 8
static volatile uint16_t adc_buf[ADC_BUF];

void app_ecg_adc_raw(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("\r\n=== 1.4a: raw ADC on PA0 @ 360 Hz ===\r\n");
    printf("jumper PA0 -> 3V3 (expect ~4095), PA0 -> GND (expect ~0)\r\n");

    if (HAL_ADC_Start_DMA(&hadc1, (uint32_t *)adc_buf, ADC_BUF) != HAL_OK) {
        printf("HAL_ADC_Start_DMA FAILED\r\n");
        return;
    }
    HAL_TIM_Base_Start(&htim2);   /* TRGO paces the ADC; no IRQ needed here */

    while (1) {
            /* newest sample the DMA has written */
            uint16_t v = adc_buf[0];

            /* print value + a bar so you can SEE it move (0..4095 -> 0..60 chars) */
            int bars = v / 68;
            char line[80];
            int n = 0;
            for (; n < bars && n < 60; n++) line[n] = '#';
            line[n] = '\0';
            printf("%4u %s\r\n", v, line);

            HAL_Delay(20);   /* 50 prints/sec — fast enough to see a beat */
        }
}
#endif
