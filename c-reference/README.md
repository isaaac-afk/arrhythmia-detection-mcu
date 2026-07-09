# Stage 1.2 — Portable C detector

Streaming, **causal** Pan-Tompkins R-peak detector in portable C99 (no platform
headers), plus a Python golden reference implementing the identical algorithm
for bit-comparison. This is the code that ports to the STM32.

## Why a new "golden reference" and not pipeline/pan_tompkins.py?
The Python pipeline uses `filtfilt` — zero-phase, non-causal, uses future
samples. A microcontroller can't do that on a live stream. So the C detector is
causal (forward-only), and `golden.py` is a causal Python twin it can bit-match.

## Files
| File | Role |
|---|---|
| `detector.h` / `detector.c` | streaming causal detector (`pt_process` one sample at a time) |
| `main.c` | runs the detector over a CSV of samples |
| `golden.py` | identical algorithm in Python (bit-match reference) |
| `compare.py` | exports a MIT-BIH record, runs both, bit-compares + scores Se/PPV |

## Run
```
make
python compare.py 100
```

## Checkpoints
- **Bit-match:** integrated max abs diff < 1e-9 AND identical R-peak lists.
- **Accuracy:** causal Se/PPV vs annotations, compared to the filtfilt numbers
  (a small drop is expected and worth documenting).
