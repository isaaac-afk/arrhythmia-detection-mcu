"""
quantize_cnn.py — Stage 2.2: int8-quantize the RR-augmented CNN for the F411.

Post-training full-integer quantization via the LiteRT/TFLite converter:
float32 Keras model -> int8 weights AND activations -> .tflite -> C byte array
(model_data.cc/.h) for LiteRT-for-Microcontrollers + CMSIS-NN on the STM32.

Two things this does carefully:
  1. Representative calibration for a TWO-INPUT model — the converter needs
     sample inputs to measure activation ranges, and it must get BOTH the beat
     window and the RR scalars, in the model's input order [beat, rr]. Built
     from DS1 (train) so calibration matches the training distribution, with
     the SAME RR standardization stats used in training (no test leakage).
  2. Honest float-vs-int8 comparison on DS2 — reports per-class Se/PPV for
     both, so the quantization accuracy drop is measured, not assumed.

Run from the repo root (after training the RR model):
    python -m pipeline.quantize_cnn
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

from .data_loader import AAMI_CLASSES, DS1_TRAIN, DS2_TEST, FS
from .train_cnn_rr import build_dataset, normalize_rr


def per_class(tag, y_true_idx, y_pred_idx):
    y_true = [AAMI_CLASSES[i] for i in y_true_idx]
    y_pred = [AAMI_CLASSES[i] for i in y_pred_idx]
    present = [c for c in AAMI_CLASSES if c in set(y_true) or c in set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=present)
    txt = classification_report(y_true, y_pred, labels=present,
                                zero_division=0, digits=3)
    acc = float(np.mean(np.asarray(y_true_idx) == np.asarray(y_pred_idx)))
    print(f"\n=== {tag} === (accuracy {acc*100:.2f}%)")
    print("     " + "  ".join(f"{c:>6}" for c in present))
    for c, row in zip(present, cm):
        print(f"{c:>4} " + "  ".join(f"{v:>6}" for v in row))
    print(txt)
    return acc, txt


def tflite_predict_int8(tflite_bytes, X_beat, X_rr):
    """Batch int8 inference through the TFLite interpreter."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    ins = interp.get_input_details()
    out = interp.get_output_details()[0]
    n = len(X_beat)

    # match each input tensor to beat vs rr by size (beat window >> 2 scalars)
    beat_d = max(ins, key=lambda d: int(np.prod(d["shape"])))
    rr_d = min(ins, key=lambda d: int(np.prod(d["shape"])))

    def q(x, d):
        s, z = d["quantization"]
        xq = np.round(x / s + z) if s else x
        return np.clip(xq, -128, 127).astype(np.int8)

    interp.resize_tensor_input(beat_d["index"], [n, X_beat.shape[1], 1])
    interp.resize_tensor_input(rr_d["index"], [n, X_rr.shape[1]])
    interp.allocate_tensors()
    interp.set_tensor(beat_d["index"], q(X_beat, beat_d))
    interp.set_tensor(rr_d["index"], q(X_rr, rr_d))
    interp.invoke()
    o = interp.get_tensor(out["index"]).astype(np.float32)
    s, z = out["quantization"]
    if s:
        o = (o - z) * s
    return np.argmax(o, axis=1)


def emit_c_array(tflite_bytes, cc_path, hdr_path, var="g_model"):
    """Write model_data.cc/.h — 16-byte-aligned byte array for TFLM."""
    b = tflite_bytes
    lines = []
    for i in range(0, len(b), 12):
        chunk = b[i:i + 12]
        lines.append("  " + " ".join(f"0x{c:02x}," for c in chunk))
    body = "\n".join(lines)
    with open(cc_path, "w") as f:
        f.write(f'#include "model_data.h"\n\n')
        f.write(f"// int8 TFLite model for LiteRT-for-Microcontrollers.\n")
        f.write(f"// 16-byte aligned as TFLM requires for the model buffer.\n")
        f.write(f"alignas(16) const unsigned char {var}[] = {{\n{body}\n}};\n")
        f.write(f"const unsigned int {var}_len = {len(b)};\n")
    with open(hdr_path, "w") as f:
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        f.write(f"extern const unsigned char {var}[];\n")
        f.write(f"extern const unsigned int {var}_len;\n\n")
        f.write("#endif  // MODEL_DATA_H\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/beat_cnn_rr.keras")
    ap.add_argument("--pn-dir", default="mitdb")
    ap.add_argument("--pre-s", type=float, default=0.25)
    ap.add_argument("--post-s", type=float, default=0.45)
    ap.add_argument("--calib", type=int, default=300, help="calibration samples")
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()

    np.random.seed(0)
    pre, post = int(args.pre_s * FS), int(args.post_s * FS)

    print("Building DS1 (calibration + RR stats) ...")
    Xtr, Rtr, ytr = build_dataset(DS1_TRAIN, pre, post, args.pn_dir)
    print("Building DS2 (test) ...")
    Xte, Rte, yte = build_dataset(DS2_TEST, pre, post, args.pn_dir)
    (Rtr, Rte), _ = normalize_rr(Rtr, Rte)   # train-only stats, same as training

    model = keras.models.load_model(args.model)

    # ---- float baseline on DS2 ----
    yf = np.argmax(model.predict({"beat": Xte, "rr": Rte}, verbose=0), axis=1)
    acc_f, txt_f = per_class("float32 (Keras)", yte, yf)

    # ---- int8 full-integer quantization ----
    idx = np.random.permutation(len(Xtr))[: args.calib]

    def representative():
        # Yield a NAME-KEYED dict, not a list: the converter reorders multi-input
        # models internally (it puts rr before beat), so a positional list feeds
        # the beat window into the rr slot and calibration fails. A dict is
        # order-independent and robust.
        for i in idx:
            yield {"beat": Xtr[i:i + 1].astype(np.float32),
                   "rr":   Rtr[i:i + 1].astype(np.float32)}

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite_bytes = conv.convert()

    os.makedirs(args.model_dir, exist_ok=True)
    tfl_path = os.path.join(args.model_dir, "beat_cnn_rr_int8.tflite")
    with open(tfl_path, "wb") as f:
        f.write(tflite_bytes)

    # ---- int8 accuracy on DS2 ----
    yq = tflite_predict_int8(tflite_bytes, Xte, Rte)
    acc_q, txt_q = per_class("int8 (TFLite)", yte, yq)

    # ---- C byte array ----
    cc = os.path.join(args.model_dir, "model_data.cc")
    hh = os.path.join(args.model_dir, "model_data.h")
    emit_c_array(tflite_bytes, cc, hh)

    # ---- summary + results.md ----
    keras_kb = os.path.getsize(args.model) / 1024.0
    tfl_kb = len(tflite_bytes) / 1024.0
    print("\n----------------------------------------")
    print(f"float32 accuracy : {acc_f*100:.2f}%")
    print(f"int8   accuracy  : {acc_q*100:.2f}%   (drop {(acc_f-acc_q)*100:+.2f} pts)")
    print(f"Keras model  : {keras_kb:6.1f} KB")
    print(f"int8 .tflite : {tfl_kb:6.1f} KB   -> {tfl_path}")
    print(f"C array      : {cc}, {hh}")
    print("Arena: start the TFLM tensor arena around 24 KB and shrink to the "
          "value the interpreter reports it actually used.")

    section = (
        "\n## Phase 2.2 — int8 quantization\n\n"
        f"Full-integer post-training quantization of the RR-augmented CNN via the "
        f"LiteRT converter ({args.calib} DS1 calibration samples). DS2 accuracy: "
        f"float32 {acc_f*100:.2f}% -> int8 {acc_q*100:.2f}% "
        f"({(acc_f-acc_q)*100:+.2f} pts). Model size {keras_kb:.1f} KB (Keras) -> "
        f"{tfl_kb:.1f} KB (int8 .tflite).\n\n"
        "int8 per-class metrics:\n```\n" + txt_q + "```\n"
    )
    rp = os.path.join(args.out_dir, "results.md")
    mode = "a" if os.path.exists(rp) else "w"
    with open(rp, mode) as f:
        if mode == "w":
            f.write("# Results\n")
        f.write(section)
    print(f"Wrote results -> {rp}")


if __name__ == "__main__":
    main()
