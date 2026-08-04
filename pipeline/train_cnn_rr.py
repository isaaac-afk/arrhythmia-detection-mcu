"""
train_cnn_rr.py — Phase 2.1b: two-input CNN = raw beat window + RR timing.

The 2.1 baseline beat V well but collapsed on S, because a window centred on
the R-peak throws away the ONE thing that defines a supraventricular beat: its
TIMING — a premature beat has a short RR relative to its neighbours. This model
gives that information back. Two inputs:

  1. the raw beat window  -> the same compact Conv1D stack as 2.1
  2. two timing scalars   -> prev_RR (s) and RR_ratio (prev_RR / local-avg RR)
                             through a tiny dense branch

...concatenated before the classifier head. RR_ratio < 1 means "this beat came
early" — the S signature. Same de Chazal split, same class weights, so it stays
comparable to both the RandomForest and the 2.1 baseline.

Run from the repo root:
    python -m pipeline.train_cnn_rr
    python -m pipeline.train_cnn_rr --epochs 60
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow import keras
from tensorflow.keras import layers

from .data_loader import load_record, AAMI_CLASSES, DS1_TRAIN, DS2_TEST, FS

CLS_TO_IDX = {c: i for i, c in enumerate(AAMI_CLASSES)}


# ------------------------------------------------------ RR timing features --
def rr_features(r_locs, fs, local_n=10):
    """Per-beat (prev_RR seconds, RR_ratio). RR_ratio = prev_RR / mean of the
    surrounding ~local_n RR intervals — <1 flags a premature (early) beat."""
    r = np.asarray(r_locs, dtype=np.float64)
    rr = np.diff(r) / fs                       # rr[i] is between beat i and i+1
    prev_rr = np.empty(len(r))
    prev_rr[0] = rr[0] if len(rr) else 1.0     # no prior beat for the first
    prev_rr[1:] = rr
    # local average RR around each beat (centered window over the rr series)
    ratio = np.ones(len(r))
    if len(rr) >= 1:
        pad = local_n // 2
        rr_pad = np.pad(rr, (pad, pad), mode="edge")
        local = np.convolve(rr_pad, np.ones(local_n) / local_n, mode="valid")
        local = local[: len(rr)]
        loc_full = np.empty(len(r)); loc_full[0] = local[0]; loc_full[1:] = local
        ratio = prev_rr / np.where(loc_full > 1e-6, loc_full, 1.0)
    return prev_rr, ratio


# ------------------------------------------------------------ beat windows --
def segment_beats(signal, r_locs, labels, pre, post, fs):
    """Raw window + RR features per beat; drop beats whose window runs off end."""
    n = len(signal)
    prev_rr, ratio = rr_features(r_locs, fs)
    X, R, y = [], [], []
    for i, (r, lab) in enumerate(zip(r_locs, labels)):
        a, b = r - pre, r + post
        if a < 0 or b > n:
            continue
        beat = signal[a:b].astype(np.float32)
        sd = beat.std()
        beat = (beat - beat.mean()) / sd if sd > 1e-6 else beat - beat.mean()
        X.append(beat)
        R.append([prev_rr[i], ratio[i]])
        y.append(lab)
    return X, R, y


def build_dataset(record_ids, pre, post, pn_dir="mitdb"):
    X_all, R_all, y_all = [], [], []
    for rid in record_ids:
        signal, fs, r_locs, labels = load_record(rid, pn_dir=pn_dir)
        X, R, y = segment_beats(signal, r_locs, labels, pre, post, fs)
        X_all.extend(X); R_all.extend(R); y_all.extend(y)
        print(f"  record {rid}: {len(y)} beats")
    X = np.asarray(X_all, dtype=np.float32)[..., np.newaxis]
    R = np.asarray(R_all, dtype=np.float32)
    y = np.array([CLS_TO_IDX[c] for c in y_all], dtype=np.int64)
    return X, R, y


def normalize_rr(R_tr, *others):
    """Standardize the 2 RR scalars using TRAIN stats only (no test leakage)."""
    mu, sd = R_tr.mean(axis=0), R_tr.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    out = [(R_tr - mu) / sd] + [(R - mu) / sd for R in others]
    return out, (mu, sd)


# ------------------------------------------------------------ class weights --
def gentle_weights(y_idx, n_classes, cap=12.0):
    classes, counts = np.unique(y_idx, return_counts=True)
    w = np.sqrt(counts.sum() / counts)
    w = w / w.min()
    w = np.clip(w, 1.0, cap)
    d = {int(c): 1.0 for c in range(n_classes)}
    d.update({int(c): float(wi) for c, wi in zip(classes, w)})
    return d


# ------------------------------------------------------------------ model ----
def build_model(win_len, n_classes=5):
    """Two-input CNN: beat window (Conv1D stack) + RR scalars (dense branch)."""
    beat_in = keras.Input(shape=(win_len, 1), name="beat")
    x = layers.Conv1D(16, 7, padding="same", activation="relu")(beat_in)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    rr_in = keras.Input(shape=(2,), name="rr")
    r = layers.Dense(16, activation="relu")(rr_in)

    z = layers.Concatenate()([x, r])
    z = layers.Dense(32, activation="relu")(z)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(n_classes, activation="softmax")(z)

    m = keras.Model([beat_in, rr_in], out, name="ecg_beat_cnn_rr")
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("AAMI — 1-D CNN + RR (inter-patient, DS2)")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix_cnn_rr.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"Saved confusion matrix -> {path}")


def write_results_md(out_dir, report_txt, n_train, n_test, win_len, n_params):
    path = os.path.join(out_dir, "results.md")
    section = (
        "\n## Phase 2.1b — 1-D CNN + RR timing\n\n"
        "The 2.1 baseline was blind to beat timing (window centred on R). This "
        "two-input model adds prev_RR and RR_ratio (prev_RR / local-average RR) "
        f"through a small dense branch, merged before the head. {n_params} params, "
        f"raw {win_len}-sample window; same de Chazal split + class weights.\n\n"
        f"- Train beats: {n_train}\n"
        f"- Test beats: {n_test}\n\n"
        "![CNN+RR confusion matrix](confusion_matrix_cnn_rr.png)\n\n"
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
    ap.add_argument("--pre-s", type=float, default=0.25)
    ap.add_argument("--post-s", type=float, default=0.45)
    ap.add_argument("--cap", type=float, default=12.0, help="class-weight cap")
    ap.add_argument("--pn-dir", default="mitdb")
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()

    np.random.seed(0)
    keras.utils.set_random_seed(0)

    pre, post = int(args.pre_s * FS), int(args.post_s * FS)
    win = pre + post
    print(f"Beat window: {win} samples ({args.pre_s+args.post_s:.2f}s) @ {FS} Hz")

    ds1_val = DS1_TRAIN[-4:]
    ds1_fit = DS1_TRAIN[:-4]

    print("Building DS1-fit (train) ...")
    Xtr, Rtr, ytr = build_dataset(ds1_fit, pre, post, args.pn_dir)
    print("Building DS1-val ...")
    Xva, Rva, yva = build_dataset(ds1_val, pre, post, args.pn_dir)
    print("Building DS2 (test) ...")
    Xte, Rte, yte = build_dataset(DS2_TEST, pre, post, args.pn_dir)

    (Rtr, Rva, Rte), _ = normalize_rr(Rtr, Rva, Rte)
    print(f"\nshapes  train {Xtr.shape}+{Rtr.shape}  val {Xva.shape}  test {Xte.shape}")

    cw = gentle_weights(ytr, len(AAMI_CLASSES), cap=args.cap)
    print("class weights:", {AAMI_CLASSES[k]: round(v, 2) for k, v in cw.items()})

    model = build_model(win)
    model.summary()
    n_params = model.count_params()

    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                        restore_best_weights=True)]
    model.fit({"beat": Xtr, "rr": Rtr}, ytr,
              validation_data=({"beat": Xva, "rr": Rva}, yva),
              epochs=args.epochs, batch_size=args.batch,
              class_weight=cw, callbacks=cb, verbose=2)

    y_pred = np.argmax(model.predict({"beat": Xte, "rr": Rte}, verbose=0), axis=1)
    report_txt = report(yte, y_pred, args.out_dir)
    write_results_md(args.out_dir, report_txt, len(ytr), len(yte), win, n_params)

    os.makedirs(args.model_dir, exist_ok=True)
    mpath = os.path.join(args.model_dir, "beat_cnn_rr.keras")
    model.save(mpath)
    print(f"\nSaved model -> {mpath}")
    print(f"~{n_params} params  (~{n_params/1024.0:.1f} KB int8-quantized)")


if __name__ == "__main__":
    main()
