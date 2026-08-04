"""
train_cnn.py — Phase 2.1: a compact 1-D CNN on RAW beat windows.

Where Stage 1.1 hand-crafted RR + morphology features and fed a RandomForest,
this learns straight from the raw waveform: segment a fixed window around each
annotated R-peak, per-beat normalise, and let a small Conv1D stack learn the
AAMI class. Same de Chazal inter-patient split (DS1 train / DS2 test) and the
same sqrt-inverse-frequency class weighting as classify.py, so the numbers are
directly comparable to the RandomForest — that comparison is the point of the
Stage 2.3 head-to-head, and it only means anything if the methodology matches.

The model is deliberately tiny (~8k params) so it int8-quantizes to tens of KB
for the F411 in Stage 2.2.

Run from the repo root:
    python -m pipeline.train_cnn
    python -m pipeline.train_cnn --epochs 60
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quiet TF's info spam

import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow import keras
from tensorflow.keras import layers

from .data_loader import load_record, AAMI_CLASSES, DS1_TRAIN, DS2_TEST, FS

CLS_TO_IDX = {c: i for i, c in enumerate(AAMI_CLASSES)}


# ------------------------------------------------------------ beat windows --
def segment_beats(signal, r_locs, labels, pre, post):
    """Fixed window [r-pre, r+post) around each R-peak, per-beat z-normalised.
    Beats whose window runs off either end of the record are dropped."""
    n = len(signal)
    X, y = [], []
    for r, lab in zip(r_locs, labels):
        a, b = r - pre, r + post
        if a < 0 or b > n:
            continue                       # edge beat, no full window
        beat = signal[a:b].astype(np.float32)
        sd = beat.std()
        beat = (beat - beat.mean()) / sd if sd > 1e-6 else beat - beat.mean()
        X.append(beat)
        y.append(lab)
    return X, y


def build_dataset(record_ids, pre, post, pn_dir="mitdb"):
    """Stack per-beat windows + AAMI labels across a set of records."""
    X_all, y_all = [], []
    for rid in record_ids:
        signal, fs, r_locs, labels = load_record(rid, pn_dir=pn_dir)
        X, y = segment_beats(signal, r_locs, labels, pre, post)
        X_all.extend(X)
        y_all.extend(y)
        print(f"  record {rid}: {len(y)} beats")
    X = np.asarray(X_all, dtype=np.float32)[..., np.newaxis]   # (n, L, 1)
    y = np.array([CLS_TO_IDX[c] for c in y_all], dtype=np.int64)
    return X, y


# ------------------------------------------------------------ class weights --
def gentle_weights(y_idx, n_classes):
    """sqrt-inverse-frequency, capped [1, 12] — same scheme as classify.py,
    keyed by integer class index for Keras. Absent classes default to 1.0."""
    classes, counts = np.unique(y_idx, return_counts=True)
    w = np.sqrt(counts.sum() / counts)
    w = w / w.min()
    w = np.clip(w, 1.0, 12.0)
    d = {int(c): 1.0 for c in range(n_classes)}
    d.update({int(c): float(wi) for c, wi in zip(classes, w)})
    return d


# ------------------------------------------------------------------ model ----
def build_model(win_len, n_classes=5):
    """Compact 1-D CNN (~8k params). BatchNorm folds into the convs at TFLite
    conversion time, so it doesn't cost anything on-device."""
    inp = keras.Input(shape=(win_len, 1))
    x = layers.Conv1D(16, 7, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = keras.Model(inp, out, name="ecg_beat_cnn")
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


# --------------------------------------------------------------- reporting ---
def report(y_true_idx, y_pred_idx, out_dir):
    y_true = [AAMI_CLASSES[i] for i in y_true_idx]
    y_pred = [AAMI_CLASSES[i] for i in y_pred_idx]
    present = [c for c in AAMI_CLASSES if c in set(y_true) or c in set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=present)

    print("\nConfusion matrix (rows=true, cols=pred):")
    print("     " + "  ".join(f"{c:>6}" for c in present))
    for c, row in zip(present, cm):
        print(f"{c:>4} " + "  ".join(f"{v:>6}" for v in row))
    report_txt = classification_report(y_true, y_pred, labels=present,
                                       zero_division=0, digits=3)
    print("\nPer-class metrics:\n" + report_txt)

    _save_cm(cm, present, out_dir)
    return report_txt


def _save_cm(cm, labels, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("AAMI beat classification — 1-D CNN (inter-patient, DS2)")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix_cnn.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"Saved confusion matrix -> {path}")


def write_results_md(out_dir, report_txt, n_train, n_test, win_len, n_params):
    path = os.path.join(out_dir, "results.md")
    section = (
        "\n## Phase 2.1 — 1-D CNN\n\n"
        "Same de Chazal inter-patient split (DS1 train / DS2 test) and "
        "sqrt-inverse-frequency class weights as Stage 1.1, but the classifier "
        f"is a compact 1-D CNN ({n_params} params) fed RAW {win_len}-sample beat "
        "windows around each R-peak — no hand-crafted features.\n\n"
        f"- Train beats: {n_train}\n"
        f"- Test beats: {n_test}\n\n"
        "![CNN confusion matrix](confusion_matrix_cnn.png)\n\n"
        "```\n" + report_txt + "```\n"
    )
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode) as f:
        if mode == "w":
            f.write("# Results\n")
        f.write(section)
    print(f"Wrote results -> {path}")


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--pre-s", type=float, default=0.25, help="window before R (s)")
    ap.add_argument("--post-s", type=float, default=0.45, help="window after R (s)")
    ap.add_argument("--pn-dir", default="mitdb")
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()

    np.random.seed(0)
    keras.utils.set_random_seed(0)

    pre, post = int(args.pre_s * FS), int(args.post_s * FS)
    win = pre + post
    print(f"Beat window: {win} samples ({args.pre_s+args.post_s:.2f}s) @ {FS} Hz")

    # Patient-wise validation carved from DS1 (never touch DS2 for tuning).
    ds1_val = DS1_TRAIN[-4:]
    ds1_fit = DS1_TRAIN[:-4]

    print("Building DS1-fit (train) ...")
    X_tr, y_tr = build_dataset(ds1_fit, pre, post, args.pn_dir)
    print("Building DS1-val ...")
    X_va, y_va = build_dataset(ds1_val, pre, post, args.pn_dir)
    print("Building DS2 (test) ...")
    X_te, y_te = build_dataset(DS2_TEST, pre, post, args.pn_dir)
    print(f"\nshapes  train {X_tr.shape}  val {X_va.shape}  test {X_te.shape}")

    cw = gentle_weights(y_tr, len(AAMI_CLASSES))
    print("class weights:", {AAMI_CLASSES[k]: round(v, 2) for k, v in cw.items()})

    model = build_model(win)
    model.summary()
    n_params = model.count_params()

    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                        restore_best_weights=True)]
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=args.epochs, batch_size=args.batch,
              class_weight=cw, callbacks=cb, verbose=2)

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    report_txt = report(y_te, y_pred, args.out_dir)
    write_results_md(args.out_dir, report_txt, len(y_tr), len(y_te), win, n_params)

    os.makedirs(args.model_dir, exist_ok=True)
    mpath = os.path.join(args.model_dir, "beat_cnn.keras")
    model.save(mpath)
    approx_kb = n_params / 1024.0  # ~1 byte/param after int8
    print(f"\nSaved model -> {mpath}")
    print(f"~{n_params} params  (~{approx_kb:.1f} KB int8-quantized — Stage 2.2)")


if __name__ == "__main__":
    main()
