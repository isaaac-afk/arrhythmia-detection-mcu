# Stage 1.3 — Get the detector running on the STM32 (fed recorded data)

**Goal:** run your verified C detector on the Nucleo-F411RE, feeding it a
recorded MIT-BIH slice stored in flash, and prove two things:
1. The on-device R-peaks match the golden reference **exactly** (correctness).
2. The per-sample compute time is far under the real-time budget (latency).

You are NOT wiring up a live ECG yet — that's Stage 1.4. Feeding *recorded*
data first lets you validate timing and correctness with a known-correct
answer, before real-world noise complicates things.

> The whole reason this stage is easy: `detector.c` is already portable C with
> no platform headers. Porting = copy the file in and call it. That was the
> payoff of doing Stage 1.2 properly.

---

## What you can do NOW (before the board arrives)

Almost everything. Only the final flash-and-run needs the physical board.

1. **Install STM32CubeIDE** (free, needs a free ST account).
   https://www.st.com/en/development-tools/stm32cubeide.html
   ~2–3 GB download; start it early.
2. **Generate the test data** on your PC (you have PhysioNet access):
   ```
   cd stage13
   python export_to_c.py 100 --seconds 10
   ```
   -> `ecg_data.h` (10 s of record 100) + `expected_rpeaks.txt` (the answer).
   A synthetic `ecg_data.h` is already included so you can build immediately
   even before running this.
3. **Create the CubeIDE project + integrate the code** (guide: INTEGRATION.md).
   This all compiles without the board attached.

## What needs the board (when it arrives, ~this week)

4. Flash (one click), open a serial terminal at 115200 baud, watch it print
   the R-peaks and the timing report.
5. Compare printed R-peaks to `expected_rpeaks.txt`.

---

## Milestones & checkpoints

**1.3a — Blinky + UART hello (first hour with the board).**
Prove the toolchain, flashing, and serial link work before anything else.
- ✅ Onboard LED (LD2) blinks; a "hello" string appears in your serial terminal.

**1.3b — Detector on recorded data (the real checkpoint).**
Replay `ecg_data.h` through `pt_process`, print each R-peak, and print the
per-sample compute time (measured with the Cortex-M cycle counter).
- ✅ **Correctness:** printed R-peaks == `expected_rpeaks.txt`, exactly.
- ✅ **Latency:** worst-case compute/sample << 2777 µs (the 360 Hz budget).
  You'll see microseconds, not milliseconds — enormous headroom.

**1.3c — True real-time playback (optional polish).**
Add a hardware timer firing at 360 Hz to feed one sample per tick, and blink
the LED on each detected beat, so it runs at genuine wall-clock speed.

---

## Two different latencies — know the difference (interview gold)

- **Compute latency** (what 1.3b measures): time to process one sample —
  microseconds on a 100 MHz M4. This is the number that proves the MCU keeps
  up in real time. Budget at 360 Hz is 2777 µs/sample; you'll use a tiny
  fraction.
- **Algorithmic detection latency:** the delay between the physical R-peak and
  the detector *reporting* it, set by Pan-Tompkins' ~150 ms integration window
  plus filter group delay (~100–200 ms total). This is inherent to the
  algorithm, not the hardware. A shorter integration window or a lighter
  detector would cut it. Being able to explain this distinction is exactly the
  kind of embedded-DSP nuance that reads as real engineering.

## Note on `double` vs `float`
The detector uses `double` to bit-match the Python reference. The F411's FPU is
single-precision, so `double` math is software-emulated (slower, still far
within budget). Switching to `float` later would speed it up but break exact
bit-match — a documented precision/speed tradeoff, good future work.
