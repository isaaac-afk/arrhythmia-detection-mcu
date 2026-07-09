# Stage 1.1 — Offline ECG pipeline (Python)

Filters MIT-BIH ECG, detects R-peaks with Pan-Tompkins, classifies beats into
AAMI classes, and validates both with a proper **inter-patient** split.

## Setup

```
pip install -r requirements.txt
```

The first run downloads MIT-BIH from PhysioNet automatically (via `wfdb`,
`pn_dir='mitdb'`) — no manual download, but you need internet access.

## Run order + checkpoints

```
# 1. R-peak detection on a single clean record
python run_stage1.py detect 100
#    CHECKPOINT: Se and PPV should both be > 99%.

# 2. Visual sanity check (saves detect_100.png)
python run_stage1.py plot 100
#    CHECKPOINT: red markers land on every R-peak in the first 10 s.

# 3. Detection across all 22 held-out (DS2) records
python run_stage1.py detect
#    CHECKPOINT: aggregate Se and PPV in the high 99s.

# 4. Inter-patient AAMI classification (train DS1, test DS2)
python run_stage1.py classify
#    CHECKPOINT: a confusion matrix + per-class precision/recall over
#    patients the model never saw. N and V will look good; S is hard
#    inter-patient (that's expected and honest — don't fake it).
```

## Files

| File | Role |
|---|---|
| `pipeline/data_loader.py` | MIT-BIH load, AAMI mapping, de Chazal DS1/DS2 lists |
| `pipeline/filters.py` | Bandpass (5–15 Hz), 60 Hz notch, baseline removal |
| `pipeline/pan_tompkins.py` | R-peak detection from scratch |
| `pipeline/evaluate_detection.py` | Se / PPV against annotations (±150 ms) |
| `pipeline/features.py` | RR-interval + QRS morphology features |
| `pipeline/classify.py` | Inter-patient train/test + confusion matrix |
| `run_stage1.py` | CLI entry point |

## Notes

- Detection is scored against annotated R-peaks; classification uses annotated
  R-peak locations for features, so the two problems are evaluated separately
  (standard practice).
- Paced records (102, 104, 107, 217) are excluded per AAMI.
- **Not a medical device.** Research/portfolio use only.
