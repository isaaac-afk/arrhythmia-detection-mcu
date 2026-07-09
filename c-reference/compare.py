"""
compare.py — Stage 1.2 verification against a real MIT-BIH record.

Does three things:
  1. Exports the record's ECG to CSV.
  2. Runs the C detector and the Python golden reference on it.
  3. Bit-compares them, and measures the causal detector's Se/PPV against
     the expert annotations (so you can compare to your filtfilt numbers).

Usage:  python compare.py [record_id]   (default 100)
Requires the compiled ./detector (run `make` first) and wfdb.
"""
import sys, subprocess
import numpy as np
import wfdb
import golden

TOL = int(0.150 * golden.FS)   # ±150 ms matching tolerance (54 samples)


def match(detected, reference, tol=TOL):
    detected = np.sort(np.asarray(detected))
    reference = np.sort(np.asarray(reference))
    used = np.zeros(len(reference), bool)
    tp = 0
    for d in detected:
        if len(reference) == 0:
            break
        j = int(np.argmin(np.abs(reference - d)))
        if not used[j] and abs(reference[j] - d) <= tol:
            used[j] = True; tp += 1
    fp = len(detected) - tp
    fn = len(reference) - int(used.sum())
    se = tp / (tp + fn) if (tp + fn) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    return tp, fp, fn, se, ppv


def main(rid="100"):
    print(f"Loading record {rid} ...")
    rec = wfdb.rdrecord(rid, pn_dir="mitdb")
    ann = wfdb.rdann(rid, "atr", pn_dir="mitdb")
    sig = rec.p_signal[:, 0].astype(float)

    beat_syms = set("NLRejAaJSVEF/fQ")
    ref = np.array([s for s, sym in zip(ann.sample, ann.symbol)
                    if sym in beat_syms])

    np.savetxt("record.csv", sig, fmt="%.17g")

    subprocess.run(["./detector", "record.csv", "rpeaks_c.txt", "integ_c.txt"],
                   check=True)
    rp_c = np.loadtxt("rpeaks_c.txt", dtype=int)
    integ_c = np.loadtxt("integ_c.txt")

    rp_py, integ_py = golden.run(sig)
    rp_py = np.array(rp_py, dtype=int)

    print("\n=== BIT-COMPARISON (C vs Python golden) ===")
    dmax = float(np.max(np.abs(integ_c - integ_py)))
    same = len(rp_c) == len(rp_py) and np.array_equal(rp_c, rp_py)
    print(f"integrated signal max abs diff : {dmax:.3e}")
    print(f"R-peak lists identical         : {same}  "
          f"(C={len(rp_c)}, golden={len(rp_py)})")
    ok = dmax < 1e-9 and same
    print(f"CHECKPOINT 1.2 (bit-match)     : {'PASS' if ok else 'FAIL'}")

    print("\n=== ACCURACY (causal C detector vs annotations) ===")
    tp, fp, fn, se, ppv = match(rp_c, ref)
    print(f"beats={len(ref)}  TP={tp} FP={fp} FN={fn}  "
          f"Se={se*100:.2f}%  PPV={ppv*100:.2f}%")
    print("(Compare to your filtfilt Se/PPV — causal is usually a bit lower.)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "100")
