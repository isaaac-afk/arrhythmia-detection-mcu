"""
data_loader.py — MIT-BIH loading, AAMI class mapping, and the de Chazal split.

MIT-BIH Arrhythmia Database: 48 half-hour records, 360 Hz, two leads,
expert beat-by-beat annotations. Hosted on PhysioNet (pn_dir='mitdb').
"""

import numpy as np
import wfdb

# ---------------------------------------------------------------------------
# AAMI class mapping (de Chazal et al., 2004).
# MIT-BIH uses many beat symbols; AAMI collapses them into 5 classes.
# ---------------------------------------------------------------------------
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

_SYMBOL_TO_AAMI = {
    # N — normal / bundle branch / escape
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # S — supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # V — ventricular ectopic
    "V": "V", "E": "V",
    # F — fusion
    "F": "F",
    # Q — unknown / paced
    "/": "Q", "f": "Q", "Q": "Q",
}

# Beat symbols only (non-beat annotations like rhythm markers are ignored).
BEAT_SYMBOLS = set(_SYMBOL_TO_AAMI.keys())

# de Chazal inter-patient split. Paced records (102,104,107,217) are excluded
# per AAMI recommendation, leaving 44 records: 22 train, 22 test.
DS1_TRAIN = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
             124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_TEST = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
            212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

FS = 360  # MIT-BIH sampling rate (Hz)


def symbol_to_aami(symbol):
    """Map a MIT-BIH beat symbol to its AAMI class, or None if not a beat."""
    return _SYMBOL_TO_AAMI.get(symbol)


def load_record(record_id, pn_dir="mitdb", channel=0):
    """
    Load one MIT-BIH record and its beat annotations from PhysioNet.

    Returns
    -------
    signal : 1-D float array   (the chosen ECG channel)
    fs     : int               (sampling rate, 360)
    r_locs : int array         (annotated R-peak sample indices, beats only)
    labels : list[str]         (AAMI class per beat, same length as r_locs)
    """
    rec = wfdb.rdrecord(str(record_id), pn_dir=pn_dir)
    ann = wfdb.rdann(str(record_id), "atr", pn_dir=pn_dir)
    signal = rec.p_signal[:, channel].astype(np.float64)

    r_locs, labels = [], []
    for sample, symbol in zip(ann.sample, ann.symbol):
        aami = symbol_to_aami(symbol)
        if aami is not None:              # keep beats only
            r_locs.append(int(sample))
            labels.append(aami)
    return signal, int(rec.fs), np.array(r_locs, dtype=int), labels
