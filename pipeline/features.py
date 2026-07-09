"""
features.py — per-beat features for classification.

Design rationale (this is the interview-relevant part):
  Timing is what separates the AAMI classes:
    - V (ventricular ectopic): premature AND followed by a full compensatory
      pause (long next-RR), with a WIDE QRS.
    - S (supraventricular ectopic): premature but with a NEAR-NORMAL width QRS
      and usually a less-than-full compensatory pause. The discriminator is
      prematurity *relative to the recent local rhythm*, not the global mean.
  So we compute RR ratios against a LOCAL running average (last ~10 beats),
  not just the global average, plus a QRS-width proxy from the waveform.
"""

import numpy as np
from .filters import bandpass


def _local_rr(rr, i, k=10):
    """Mean RR over the ~k intervals surrounding position i (local rhythm)."""
    lo = max(0, i - k)
    hi = min(len(rr), i + k)
    seg = rr[lo:hi]
    return float(np.mean(seg)) if len(seg) else float(np.median(rr))


def _qrs_width(seg, fs):
    """Proxy for QRS duration: span of samples with |signal| above half-max."""
    a = np.abs(seg)
    thr = 0.5 * a.max() if a.max() > 0 else 0
    above = np.where(a > thr)[0]
    return (above[-1] - above[0]) / fs if len(above) > 1 else 0.0


def extract_features(signal, fs, rpeaks, window_s=0.28):
    """
    Build a feature matrix, one row per R-peak.

    Returns
    -------
    X     : (n_beats, n_features) float array
    valid : bool mask of beats that had enough neighbours to feature-ise
    """
    filtered = bandpass(signal, fs)
    rpeaks = np.sort(np.asarray(rpeaks))
    n = len(rpeaks)
    half = int(window_s * fs / 2)

    rr = np.diff(rpeaks)                          # length n-1
    global_rr = np.median(rr) if len(rr) else fs

    rows, valid = [], []
    for i in range(n):
        lo, hi = rpeaks[i] - half, rpeaks[i] + half
        if lo < 0 or hi >= len(filtered):
            valid.append(False)
            continue

        prev_rr = rr[i - 1] if i > 0 else global_rr
        next_rr = rr[i] if i < n - 1 else global_rr
        local_rr = _local_rr(rr, min(i, len(rr) - 1)) if len(rr) else global_rr
        local_rr = local_rr if local_rr > 0 else global_rr

        seg = filtered[lo:hi]
        width = _qrs_width(seg, fs)

        rows.append([
            prev_rr / fs,                    # absolute prev RR (s)
            next_rr / fs,                    # absolute next RR (s)
            prev_rr / local_rr,              # prematurity vs LOCAL rhythm  <- S
            next_rr / local_rr,              # compensatory pause vs local  <- V
            prev_rr / next_rr,               # RR asymmetry
            local_rr / global_rr,            # local vs global rhythm context
            abs(prev_rr - next_rr) / local_rr,  # RR irregularity
            width,                           # QRS width (s)                <- V
            seg.max() - seg.min(),           # peak-to-peak amplitude
            np.sqrt(np.mean(seg ** 2)),      # segment energy
            float(np.argmax(seg) - np.argmin(seg)),  # peak-trough offset (shape)
        ])
        valid.append(True)

    return np.array(rows, dtype=float), np.array(valid, dtype=bool)


FEATURE_NAMES = [
    "prev_RR_s", "next_RR_s", "prev_RR_local_ratio", "next_RR_local_ratio",
    "prev_next_ratio", "local_global_ratio", "RR_irregularity",
    "qrs_width_s", "seg_ptp", "seg_rms", "peak_trough_off",
]