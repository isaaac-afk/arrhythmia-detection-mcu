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

/* --- Stage 1.3c: timer-driven feed + ring buffer (MCU only) ----------- */
#ifndef HOST_TEST
#include "main.h"                  /* HAL types */
extern TIM_HandleTypeDef htim2;

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

        if (fired) { printf("R-peak @ %ld\r\n", ridx); n_peaks++; }
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
#endif
