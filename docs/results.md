# Results

## Stage 1.3 — recorded data on hardware

**Hardware:** ST Nucleo-F411RE, Cortex-M4F @ 84 MHz (SYSCLK 84, PCLK1 42).
**Toolchain:** STM32CubeIDE 2.2.0 / CubeMX 6.18, arm-none-eabi-gcc.
**Build:** `-O0`, DEBUG. **Not optimized** — see Deferred optimizations.
**Numerics:** detector uses `double`. The F411 FPU is single-precision
(`-mfpu=fpv4-sp-d16`), so double math is software-emulated.

**Data:** MIT-BIH record 100, samples 0-21599 (60 s @ 360 Hz), channel 0
(MLII), physical mV via wfdb `p_signal`, stored as `const double` in flash.

### Correctness

72/72 R-peaks match the desktop C reference exactly (667 ... 21428).
Verified by `Compare-Object` against a captured serial log, not by
inspection. Artifacts: `firmware/board_peaks_rec100.txt` (device output),
`stage13/expected_rpeaks.txt` (golden reference).

### Latency

| Metric | Value |
|---|---|
| avg compute/sample | 3627 cyc = 43.2 us |
| worst compute/sample | 14004 cyc = 166.7 us |
| real-time budget @ 360 Hz | 2777 us |
| worst-case utilization | **6.0%** |
| roadmap target | < 50 ms -> **300x under** |

Measured with the DWT cycle counter.

### Footprint

| Region | Used | Capacity |
|---|---|---|
| Flash (text) | 190,788 B | 512 KB (36%) |
| RAM (data+bss) | 2,096 B | 128 KB (1.6%) |

Most of the flash is the 21600-sample ECG array (172.8 KB); the firmware
itself is ~18 KB. The array is `const` and correctly placed in `.rodata`.

### Record-length independence

Per-sample cost on the 3600-sample synthetic run was 3631 avg / 13968 worst
cycles; on the 21600-sample real record, 3627 / 14004. Six times the data,
same per-sample cost. Pan-Tompkins does fixed work per sample.

### Compute latency vs detection latency

The numbers above are **compute latency** — time to process one sample.
Distinct from **algorithmic detection latency**: the delay between the
physical R-peak and the detector reporting it, set by Pan-Tompkins'
~150 ms integration window plus filter group delay (~100-200 ms total).
That is inherent to the algorithm, not the hardware, and would not improve
with a faster MCU.

### Deferred optimizations

1. `double` -> `float`, using the FPU natively. Expected to be several times
   faster, but breaks bit-exactness with the Python reference — a
   precision/speed tradeoff to measure, not assume.
2. `-O0` -> `-O2`.

Both were deliberately deferred until after the bit-match was established,
so any subsequent change in output is attributable to the optimization
rather than to a port bug.

### Known artifacts

First detected peak is at sample 667 (1.85 s), not near 0: Pan-Tompkins
spends its first ~2 s learning adaptive thresholds. Identical on desktop
and device. Roughly 2 beats at the head and 1-2 at the tail fall outside
the detected window — 72 detected against ~76 expected at 76 BPM.

---

**Not a medical device.** Engineering portfolio project. Not for diagnosis
or clinical use.
