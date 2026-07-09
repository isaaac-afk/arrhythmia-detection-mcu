"""
features.py — per-beat features for classification.

Kept deliberately simple and interpretable for the first classifier:
  RR features   — previous RR, next RR, and ratio to local average RR.
                  (Ventricular ectopics are premature => short prev-RR,
                   long next-RR. This is the strongest simple signal.)
  Morphology    — a small window of the bandpassed signal around each R-peak,
                  which captures QRS shape/width.
"""

import numpy as np
from .filters import bandpass


def extract_features(signal, fs, rpeaks, window_s=0.28):
    """
    Build a feature matrix, one row per R-peak.

    Returns
    -------
    X       : (n_beats, n_features) float array
    valid   : bool mask of beats that had enough neighbours to feature-ise
    """
    filtered = bandpass(signal, fs)
    rpeaks = np.sort(np.asarray(rpeaks))
    n = len(rpeaks)
    half = int(window_s * fs / 2)

    rr = np.diff(rpeaks)                       # length n-1
    avg_rr = np.median(rr) if len(rr) else fs  # fallback: 1 s

    rows, valid = [], []
    for i in range(n):
        lo, hi = rpeaks[i] - half, rpeaks[i] + half
        if lo < 0 or hi >= len(filtered):
            valid.append(False)
            continue
        prev_rr = rr[i - 1] if i > 0 else avg_rr
        next_rr = rr[i] if i < n - 1 else avg_rr

        seg = filtered[lo:hi]
        morph = [seg.max(), seg.min(), seg.max() - seg.min(),
                 np.sqrt(np.mean(seg ** 2)),          # energy
                 float(np.argmax(seg) - np.argmin(seg))]  # peak-to-trough offset

        rows.append([prev_rr / fs, next_rr / fs,
                     prev_rr / avg_rr, next_rr / avg_rr] + morph)
        valid.append(True)

    return np.array(rows, dtype=float), np.array(valid, dtype=bool)


FEATURE_NAMES = ["prev_RR_s", "next_RR_s", "prev_RR_ratio", "next_RR_ratio",
                 "seg_max", "seg_min", "seg_ptp", "seg_rms", "peak_trough_off"]
