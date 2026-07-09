"""
pan_tompkins.py — R-peak detection, implemented from scratch.

Pipeline (Pan & Tompkins, 1985):
    bandpass 5-15 Hz  ->  derivative  ->  square  ->  moving-window integrate
    ->  adaptive dual thresholds with search-back.

The integrated signal has broad bumps where QRS complexes are. We find peaks
in that signal, adapt a signal/noise threshold pair to the running levels, and
map each accepted bump back to the true R-peak (max of the bandpassed signal in
a small neighbourhood).
"""

import numpy as np
from scipy.signal import find_peaks
from .filters import bandpass


def _derivative(x, fs):
    # 5-point derivative from the original paper, scaled by fs.
    d = np.zeros_like(x)
    d[2:-2] = (2 * x[3:-1] + x[4:] - x[:-4] - 2 * x[1:-3]) * (fs / 8.0)
    return d


def _moving_window_integrate(x, fs, window_s=0.150):
    w = max(1, int(round(window_s * fs)))
    return np.convolve(x, np.ones(w) / w, mode="same")


def detect_rpeaks(signal, fs, refractory_s=0.20, search_neighbourhood_s=0.05):
    """
    Detect R-peaks in a raw ECG signal.

    Returns
    -------
    rpeaks : int array of sample indices of detected R-peaks.
    """
    # --- Stage 1: transform the signal so QRS stands out ------------------
    filtered = bandpass(signal, fs)
    deriv = _derivative(filtered, fs)
    squared = deriv ** 2
    integrated = _moving_window_integrate(squared, fs)

    # --- Stage 2: candidate peaks in the integrated signal ----------------
    min_dist = int(refractory_s * fs)          # 200 ms physiological refractory
    cand, _ = find_peaks(integrated, distance=min_dist)
    if len(cand) == 0:
        return np.array([], dtype=int)

    # --- Stage 3: adaptive dual thresholding with search-back -------------
    # Initialise from the first ~2 s of the integrated signal.
    init = integrated[: min(len(integrated), 2 * fs)]
    spki = np.max(init) * 0.25     # running estimate of signal peak level
    npki = np.mean(init) * 0.5     # running estimate of noise peak level

    rpeaks = []
    rr = []                        # recent RR intervals (samples) for search-back
    last_qrs = -min_dist

    def threshold():
        return npki + 0.25 * (spki - npki)

    i = 0
    while i < len(cand):
        idx = cand[i]
        peak_val = integrated[idx]
        thr = threshold()

        is_qrs = False
        if peak_val > thr and (idx - last_qrs) > min_dist:
            is_qrs = True
        else:
            # search-back: if we've gone >1.66x the average RR without a beat,
            # relax to half-threshold and look for a missed QRS among candidates.
            if rr and (idx - last_qrs) > 1.66 * np.mean(rr[-8:]):
                lowered = 0.5 * thr
                window = [c for c in cand if last_qrs < c < idx
                          and integrated[c] > lowered]
                if window:
                    best = max(window, key=lambda c: integrated[c])
                    r = _refine(filtered, best, fs, search_neighbourhood_s)
                    rpeaks.append(r)
                    if last_qrs > 0:
                        rr.append(best - last_qrs)
                    last_qrs = best
                    spki = 0.25 * integrated[best] + 0.75 * spki

        if is_qrs:
            r = _refine(filtered, idx, fs, search_neighbourhood_s)
            rpeaks.append(r)
            if last_qrs > 0:
                rr.append(idx - last_qrs)
            last_qrs = idx
            spki = 0.125 * peak_val + 0.875 * spki      # adapt signal level
        else:
            npki = 0.125 * peak_val + 0.875 * npki      # adapt noise level
        i += 1

    return np.unique(np.array(rpeaks, dtype=int))


def _refine(filtered, idx, fs, neighbourhood_s):
    """Snap a detection to the local max of the bandpassed signal nearby."""
    half = int(neighbourhood_s * fs)
    lo, hi = max(0, idx - half), min(len(filtered), idx + half + 1)
    return lo + int(np.argmax(filtered[lo:hi]))
