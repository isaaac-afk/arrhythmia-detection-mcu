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

## Scope: detection on-device, classification in Python (Phase 1)

Stage 1.3 deploys the **R-peak detector** to hardware. The classical AAMI
beat classifier from Stage 1.1 remains in Python and is deliberately not
ported to C or to the MCU during Phase 1.

**Why.** Stage 1.1 established the ceiling of hand-crafted RR + morphology
features: N and V separate well, S and F do not. That result is the reason
Phase 2 exists — a quantized 1D-CNN replaces the classical classifier
outright. Porting a RandomForest to embedded C (tree export, feature
extraction, float bit-matching against the Python reference) is several days
of work, and in Phase 1 it has nothing to be compared against.

**Deferred, not cancelled.** The port happens in **Stage 2.3**, where it
becomes the baseline arm of the classical-vs-NN head-to-head. That
comparison — accuracy, latency, flash/RAM, power, all measured on the same
device — is only meaningful if both classifiers run on the same hardware. So
the C/MCU port is scheduled for the stage where it earns its keep rather
than done on spec here.

**Consequence for this stage.** On-device output is R-peak indices, verified
72/72 against the desktop C reference on record 100. Checkpoint 1.3's
"on-device classifications" is read as on-device *detections*; end-to-end
classification latency is a Phase 2 measurement.

This also closes a gap inherited from Checkpoint 1.2, whose wording asked for
R-peaks *and labels* from the C pipeline. `c-reference/` emits peaks only.
The same reasoning applies: labels arrive with the Stage 2.3 port.

---

**Not a medical device.** Engineering portfolio project. Not for diagnosis
or clinical use.

## Stage 1.3c — timer-driven real-time playback

TIM2 @ 84 MHz, PSC=0, ARR=233332. ARR+1 = 233333 gives **360.0000857 Hz** —
exact 360 Hz is unreachable from an 84 MHz clock (84e6/360 = 233333.33).

ISR pushes one sample into a 64-slot ring buffer; the main loop drains it and
calls `pt_process`. Acquisition and processing never touch the same slot.

| Metric | Value |
|---|---|
| R-peaks | **72/72**, identical to the free-running build |
| samples processed | 21600 / 21600 |
| ring buffer overflows | **0** |
| wall clock | 60003 ms (expected 60000) — **0.005% error** |
| avg compute/sample | 3649 cyc |
| worst compute/sample | 14047 cyc |

**Clock accuracy.** The 3 ms drift comes from the 360.0000857 Hz quantization,
`HAL_GetTick` granularity, and the HSI internal oscillator (spec ±1%). A real
device would use an external crystal (HSE).

**ISR cost, measured indirectly.** Per-sample compute rose from 3627/14004 cyc
(free-running) to 3649/14047 cyc under the timer. The DWT-measured region is
now preemptible: at 3627 cyc per sample against a 233333-cyc tick, the ISR
lands inside the measurement ~1.55% of the time. A +22 cyc average implies
`HAL_TIM_IRQHandler` costs roughly 1400 cycles — consistent with HAL's known
overhead. A register-level handler would be ~20 cycles.

**Why this matters for Stage 1.4.** The free-running loop consumed 60 s of ECG
in ~1 s: it proved compute, not scheduling. The AD8232 will deliver samples at
360 Hz regardless of readiness. Zero overflows at 6% CPU utilization means the
sample path holds.

Artifact: `firmware/board_peaks_rec100_timed.txt`.
