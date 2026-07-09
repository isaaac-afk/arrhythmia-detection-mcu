"""
run_stage1.py — Stage 1.1 end-to-end runner.

Usage:
    python run_stage1.py detect         # R-peak detection + Se/PPV on DS2
    python run_stage1.py detect 100     # single record
    python run_stage1.py classify       # inter-patient AAMI classification
    python run_stage1.py plot 100       # sanity plot: detections over waveform

Checkpoints:
    - detect: Se and PPV should be >99% on clean records (e.g. 100).
    - classify: a 5x5 (or smaller) confusion matrix over held-out patients.
"""

import sys
import numpy as np

from pipeline.data_loader import load_record, DS2_TEST
from pipeline.pan_tompkins import detect_rpeaks
from pipeline.evaluate_detection import match_peaks, print_report
from pipeline.classify import run_classification


def run_detection(record_ids):
    agg = {"TP": 0, "FP": 0, "FN": 0}
    for rid in record_ids:
        signal, fs, ref, _ = load_record(rid)
        det = detect_rpeaks(signal, fs)
        m = match_peaks(det, ref, fs)
        print_report(str(rid), m)
        for k in agg:
            agg[k] += m[k]
    tp, fp, fn = agg["TP"], agg["FP"], agg["FN"]
    se = tp / (tp + fn) if (tp + fn) else 0
    ppv = tp / (tp + fp) if (tp + fp) else 0
    print(f"\nAGGREGATE  Se={se*100:.2f}%  PPV={ppv*100:.2f}%  "
          f"(TP={tp} FP={fp} FN={fn})")


def plot_record(rid):
    import matplotlib.pyplot as plt
    signal, fs, ref, _ = load_record(rid)
    det = detect_rpeaks(signal, fs)
    t = np.arange(len(signal)) / fs
    n = 10 * fs  # first 10 s
    plt.figure(figsize=(14, 4))
    plt.plot(t[:n], signal[:n], lw=0.8, label="ECG")
    d = det[det < n]
    plt.plot(d / fs, signal[d], "rv", label="detected R")
    r = ref[ref < n]
    plt.plot(r / fs, signal[r] + 0.15, "g|", ms=14, label="annotated R")
    plt.legend(); plt.xlabel("time (s)"); plt.title(f"Record {rid}")
    plt.tight_layout(); plt.savefig(f"detect_{rid}.png", dpi=120)
    print(f"saved detect_{rid}.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "detect"
    if cmd == "detect":
        ids = [int(sys.argv[2])] if len(sys.argv) > 2 else DS2_TEST
        run_detection(ids)
    elif cmd == "classify":
        run_classification()
    elif cmd == "plot":
        plot_record(int(sys.argv[2]))
    else:
        print(__doc__)
