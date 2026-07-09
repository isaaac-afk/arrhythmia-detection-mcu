"""
classify.py — AAMI beat classification with a proper inter-patient split.

Train on DS1 patients, test on DS2 patients. No patient appears in both.
Running run_classification() saves the Stage 1.1 deliverables: a
confusion-matrix figure, a metrics block in docs/results.md, and the model.
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import joblib

from .data_loader import load_record, AAMI_CLASSES, DS1_TRAIN, DS2_TEST
from .features import extract_features


def _gentle_weights(y):
    """
    sqrt-inverse-frequency class weights, capped.

    Full 'balanced' weighting over-weights ultra-rare classes so hard the model
    scatters false positives everywhere (the F-class leak in v1). sqrt-inverse
    boosts minority classes enough to be learned without letting a 388-beat
    class dominate a 44,000-beat one; the cap stops the 7-beat Q class leaking.
    """
    classes, counts = np.unique(y, return_counts=True)
    w = np.sqrt(counts.sum() / counts)
    w = w / w.min()
    w = np.clip(w, 1.0, 12.0)
    return {c: float(wi) for c, wi in zip(classes, w)}


def build_dataset(record_ids, pn_dir="mitdb"):
    """Concatenate per-beat features + AAMI labels across a set of records."""
    X_all, y_all = [], []
    for rid in record_ids:
        signal, fs, r_locs, labels = load_record(rid, pn_dir=pn_dir)
        X, valid = extract_features(signal, fs, r_locs)
        y = np.array(labels)[valid]
        X_all.append(X)
        y_all.append(y)
        print(f"  record {rid}: {len(y)} beats")
    return np.vstack(X_all), np.concatenate(y_all)


def run_classification(pn_dir="mitdb", out_dir="docs", model_dir="models"):
    print("Building DS1 (train) ...")
    X_tr, y_tr = build_dataset(DS1_TRAIN, pn_dir)
    print("Building DS2 (test) ...")
    X_te, y_te = build_dataset(DS2_TEST, pn_dir)

    scaler = StandardScaler().fit(X_tr)
    clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5,
        class_weight=_gentle_weights(y_tr),
        n_jobs=-1, random_state=0)
    clf.fit(scaler.transform(X_tr), y_tr)

    y_pred = clf.predict(scaler.transform(X_te))
    cm, report_txt = report_classification(y_te, y_pred, out_dir=out_dir)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler},
                os.path.join(model_dir, "beat_classifier.joblib"))
    print(f"\nSaved model -> {model_dir}/beat_classifier.joblib")

    _write_results_md(out_dir, y_te, y_pred, report_txt,
                      n_train=len(y_tr), n_test=len(y_te))
    return clf, scaler


def report_classification(y_true, y_pred, labels=AAMI_CLASSES, out_dir=None):
    present = [c for c in labels if c in set(y_true) or c in set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=present)

    print("\nConfusion matrix (rows=true, cols=pred):")
    print("     " + "  ".join(f"{c:>6}" for c in present))
    for c, row in zip(present, cm):
        print(f"{c:>4} " + "  ".join(f"{v:>6}" for v in row))

    report_txt = classification_report(y_true, y_pred, labels=present,
                                       zero_division=0, digits=3)
    print("\nPer-class metrics:")
    print(report_txt)

    if out_dir:
        _save_cm_figure(cm, present, out_dir)
    return cm, report_txt


def _save_cm_figure(cm, labels, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("AAMI beat classification (inter-patient, DS2)")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"Saved confusion matrix -> {path}")


def _write_results_md(out_dir, y_true, y_pred, report_txt, n_train, n_test):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.md")
    section = (
        "\n## Classification (Stage 1.1)\n\n"
        "Inter-patient split (de Chazal): trained on DS1 (22 patients), "
        "tested on DS2 (22 different patients). Features from annotated "
        "R-peaks; RandomForest, sqrt-inverse-frequency class weights.\n\n"
        f"- Train beats: {n_train}\n"
        f"- Test beats: {n_test}\n\n"
        "![Confusion matrix](confusion_matrix.png)\n\n"
        "```\n" + report_txt + "```\n"
    )
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode) as f:
        if mode == "w":
            f.write("# Results\n")
        f.write(section)
    print(f"Wrote results -> {path}")