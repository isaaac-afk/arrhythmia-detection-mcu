#include <stdio.h>
#include <stdint.h>
#include "detector.h"
#include "ecg_data.h"
#include "app_ecg.h"

/* Core clock in MHz. Set to match your CubeMX clock configuration
 * (100 for a 100 MHz SYSCLK). Used only to convert cycles -> ns. */
#ifndef CPU_MHZ
#define CPU_MHZ 100u
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

void app_ecg_run(void) {
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
