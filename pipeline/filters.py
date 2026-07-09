"""
filters.py — signal conditioning.

The 5–15 Hz bandpass is the key one: it concentrates energy on the QRS
complex while suppressing P/T waves, baseline wander, and high-freq noise.
The notch/baseline helpers matter later for the live AD8232 signal, not for
the clean MIT-BIH data.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass(signal, fs, low=5.0, high=15.0, order=2):
    """Zero-phase Butterworth bandpass. filtfilt => no phase distortion,
    so detected peaks line up with the true R-peak locations."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def notch_60hz(signal, fs, freq=60.0, q=30.0):
    """Remove mains hum (60 Hz in North America). For the live signal."""
    b, a = iirnotch(freq / (0.5 * fs), q)
    return filtfilt(b, a, signal)


def remove_baseline(signal, fs, cutoff=0.5):
    """Highpass to kill baseline wander (breathing). For the live signal."""
    nyq = 0.5 * fs
    b, a = butter(2, cutoff / nyq, btype="high")
    return filtfilt(b, a, signal)
