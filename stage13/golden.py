"""
golden.py — Python golden reference for the C detector.

Implements the IDENTICAL streaming causal Pan-Tompkins algorithm as
detector.c, sample-by-sample in the same arithmetic order, so the two can be
bit-compared. This is the reference the C port must reproduce. It is NOT the
same as pipeline/pan_tompkins.py (that one is batch + non-causal filtfilt).
"""
import numpy as np

FS = 360
MWI_WIN = 54
REFRACTORY = 72
LEARN = 720
REFINE_WIN = 90

SOS = [
    [0.0067654132571125444, 0.013530826514225089, 0.0067654132571125444,
     1.0, -1.7919338331533654, 0.84122205760775526],
    [1.0, -2.0, 1.0, 1.0, -1.9193726170801984, 0.92874464520573263],
]


def run(samples):
    """Return (rpeak_indices list, integrated signal array)."""
    z1 = [0.0, 0.0]; z2 = [0.0, 0.0]
    dbuf = [0.0]*5
    mwi_buf = [0.0]*MWI_WIN; mwi_sum = 0.0; mwi_pos = 0
    bp_hist = [0.0]*128
    i_prev1 = 0.0; i_prev2 = 0.0
    spki = 0.0; npki = 0.0
    learn_max = 0.0; learn_sum = 0.0; learned = False
    last_qrs = -REFRACTORY - 1
    rpeaks = []
    integrated = np.empty(len(samples))

    for n, x in enumerate(samples):
        # 1. bandpass (transposed DF-II cascade)
        for s in range(2):
            b0, b1, b2, _, a1, a2 = SOS[s]
            y = b0 * x + z1[s]
            z1[s] = b1 * x - a1 * y + z2[s]
            z2[s] = b2 * x - a2 * y
            x = y
        bp = x
        bp_hist[n % 128] = bp

        # 2. causal derivative
        dbuf[4] = dbuf[3]; dbuf[3] = dbuf[2]; dbuf[2] = dbuf[1]
        dbuf[1] = dbuf[0]; dbuf[0] = bp
        deriv = (2.0*dbuf[0] + dbuf[1] - dbuf[3] - 2.0*dbuf[4]) / 8.0

        # 3. square
        sq = deriv * deriv

        # 4. moving-window integrate
        mwi_sum -= mwi_buf[mwi_pos]
        mwi_buf[mwi_pos] = sq
        mwi_sum += sq
        mwi_pos = (mwi_pos + 1) % MWI_WIN
        integ = mwi_sum / MWI_WIN
        integrated[n] = integ

        # 5. learning
        if n < LEARN:
            if integ > learn_max: learn_max = integ
            learn_sum += integ
            if n == LEARN - 1:
                spki = 0.25 * learn_max
                npki = 0.5 * (learn_sum / LEARN)
                learned = True
        else:
            # 6. local-max detection at index n-1
            p = n - 1
            if i_prev1 > integ and i_prev1 >= i_prev2 and p >= LEARN:
                v = i_prev1
                thr = npki + 0.25 * (spki - npki)
                if v > thr and (p - last_qrs) > REFRACTORY:
                    lo = max(0, p - REFINE_WIN)
                    best = lo; bestv = bp_hist[lo % 128]
                    for k in range(lo, p + 1):
                        val = bp_hist[k % 128]
                        if val > bestv: bestv = val; best = k
                    rpeaks.append(best)
                    last_qrs = p
                    spki = 0.125 * v + 0.875 * spki
                else:
                    npki = 0.125 * v + 0.875 * npki

        i_prev2 = i_prev1
        i_prev1 = integ

    return rpeaks, integrated
