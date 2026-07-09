"""
evaluate_detection.py — score R-peak detection against reference annotations.

Standard tolerance: a detection counts as a true positive if it falls within
±150 ms of a reference beat. Reports sensitivity and positive predictive value,
the two numbers the literature uses (accuracy is misleading here — there are no
"true negatives" in a continuous stream).
"""

import numpy as np


def match_peaks(detected, reference, fs, tolerance_s=0.150):
    """
    Greedy one-to-one matching within tolerance.

    Returns dict with TP, FP, FN, sensitivity, ppv, f1.
    """
    tol = tolerance_s * fs
    detected = np.sort(np.asarray(detected))
    reference = np.sort(np.asarray(reference))

    matched_ref = np.zeros(len(reference), dtype=bool)
    tp = 0
    for d in detected:
        if len(reference) == 0:
            break
        j = np.argmin(np.abs(reference - d))
        if not matched_ref[j] and abs(reference[j] - d) <= tol:
            matched_ref[j] = True
            tp += 1
    fp = len(detected) - tp
    fn = len(reference) - int(matched_ref.sum())

    sens = tp / (tp + fn) if (tp + fn) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn,
            "sensitivity": sens, "ppv": ppv, "f1": f1}


def print_report(name, m):
    print(f"[{name}] TP={m['TP']} FP={m['FP']} FN={m['FN']}  "
          f"Se={m['sensitivity']*100:.2f}%  PPV={m['ppv']*100:.2f}%  "
          f"F1={m['f1']*100:.2f}%")
