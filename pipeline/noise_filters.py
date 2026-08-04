#!/usr/bin/env python3
"""
Stage 1.4d — noise-filter design and validation (software-first, pre-hardware).

Designs the two real-time filters the MCU will run on the live AD8232 signal:
  1. a baseline-wander high-pass (~0.5 Hz)   — kills breathing / electrode drift
  2. a 60 Hz mains notch                     — kills Canadian wall-power hum

...then proves them by taking a CLEAN record, injecting synthetic wander +
60 Hz hum (the noise the recorded data never had), running the filters
CAUSALLY (the way the MCU must — no looking into the future), and showing the
noise is gone in both the waveform and the spectrum.

It also prints the biquad coefficients ready to paste into C for the firmware,
and does an optional R-peak detection sanity check (clean vs noisy vs filtered)
with a small reference Pan-Tompkins.

Run from the repo root (it imports the pipeline package):
    python -m pipeline.noise_filters                 # uses MIT-BIH record 100
    python -m pipeline.noise_filters --synthetic     # no data download; synthetic ECG
    python -m pipeline.noise_filters --hum-mv 0.25   # heavier mains hum

Real-time note: validation uses scipy.signal.sosfilt (CAUSAL), never filtfilt.
filtfilt is zero-phase but runs the data backwards too — impossible on a live
MCU. Testing causally is the whole point.
"""

import argparse
import numpy as np
from scipy import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, sys
# Import Isaac's real Stage 1.1 detector + matcher. They live in the `pipeline`
# package and use package-relative imports, so put the repo root on sys.path and
# import by full package path — works whether this is run as a script
# (python pipeline/noise_filters.py) or a module (python -m pipeline.noise_filters).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from pipeline.pan_tompkins import detect_rpeaks      # real adaptive detector
from pipeline.evaluate_detection import match_peaks  # real matcher (150 ms tol)

FS = 360.0            # Hz — matches the firmware sample rate
HP_CUT = 0.5          # Hz — baseline-wander high-pass corner
NOTCH_F = 60.0        # Hz — mains frequency (Canada / North America)
NOTCH_Q = 30.0        # notch quality factor (~2 Hz wide at 60 Hz)


# ---------------------------------------------------------------- filters ----
def design_filters(fs=FS, hp_cut=HP_CUT, notch_f=NOTCH_F, notch_q=NOTCH_Q):
    """Return (sos_hp, sos_notch, sos_all). All second-order sections."""
    sos_hp = signal.butter(2, hp_cut, btype="high", fs=fs, output="sos")
    b, a = signal.iirnotch(notch_f, notch_q, fs)
    sos_notch = signal.tf2sos(b, a)
    sos_all = np.vstack([sos_hp, sos_notch])
    return sos_hp, sos_notch, sos_all


def apply_causal(sos, x):
    """Causal, real-time-equivalent filtering (single forward pass)."""
    return signal.sosfilt(sos, x)


def print_c_coeffs(sos_hp, sos_notch):
    """Emit biquad coefficients ready to drop into firmware."""
    def one(name, sos):
        b0, b1, b2, a0, a1, a2 = sos[0]
        # normalise so a0 == 1 (direct-form II transposed convention)
        b0, b1, b2, a1, a2 = (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
        print(f"  /* {name} @ {FS:.0f} Hz */")
        print(f"  {{ .b0 = {b0: .8f}f, .b1 = {b1: .8f}f, .b2 = {b2: .8f}f,")
        print(f"    .a1 = {a1: .8f}f, .a2 = {a2: .8f}f }},")

    print("\n// ---- biquad coefficients for firmware (float, DF2T) ----")
    print("static const biquad_t ecg_filters[2] = {")
    one("baseline-wander high-pass, 0.5 Hz, 2nd-order Butterworth", sos_hp)
    one(f"mains notch, {NOTCH_F:.0f} Hz, Q={NOTCH_Q:.0f}", sos_notch)
    print("};")


# ---------------------------------------------------------------- signal -----
def synthetic_ecg(seconds=60.0, fs=FS, bpm=75.0):
    """A plausible clean ECG in mV: baseline ~0, R ~1.2 mV, ~75 bpm."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    rr = 60.0 / bpm
    x = np.zeros(n)

    def bump(tc, width, amp):
        return amp * np.exp(-0.5 * ((t - tc) / width) ** 2)

    beat = 0.0
    while beat < seconds:
        x += bump(beat + 0.18 * rr, 0.020, 0.12)    # P
        x -= bump(beat + 0.44 * rr, 0.008, 0.15)    # Q
        x += bump(beat + 0.46 * rr, 0.008, 1.20)    # R
        x -= bump(beat + 0.49 * rr, 0.010, 0.30)    # S
        x += bump(beat + 0.66 * rr, 0.040, 0.30)    # T
        beat += rr
    return t, x


def load_record_100(seconds=60.0, fs=FS):
    """MIT-BIH record 100, channel 0 (MLII), physical mV."""
    import wfdb
    rec = wfdb.rdrecord("100", pn_dir="mitdb")           # streams from PhysioNet
    sig = rec.p_signal[:, 0].astype(float)
    assert abs(rec.fs - fs) < 1e-6, f"record fs {rec.fs} != {fs}"
    n = int(seconds * fs)
    sig = sig[:n]
    t = np.arange(len(sig)) / fs
    return t, sig


def add_noise(x, t, fs=FS, hum_mv=0.15, wander_mv=0.30, gauss_mv=0.01, seed=0):
    """Inject the noise the recorded data never had."""
    rng = np.random.default_rng(seed)
    wander = (wander_mv * np.sin(2 * np.pi * 0.15 * t)
              + 0.5 * wander_mv * np.sin(2 * np.pi * 0.33 * t + 1.0))  # breathing/drift
    mains = hum_mv * np.sin(2 * np.pi * NOTCH_F * t)                    # 60 Hz hum
    white = gauss_mv * rng.standard_normal(len(x))
    return x + wander + mains + white






# ------------------------------------------------------------ spectrum -------
def band_power(x, fs, f_lo, f_hi):
    f, pxx = signal.welch(x, fs=fs, nperseg=min(4096, len(x)))
    m = (f >= f_lo) & (f <= f_hi)
    integ = getattr(np, "trapezoid", getattr(np, "trapz", None))  # NumPy 2.x renamed trapz
    return float(integ(pxx[m], f[m])) if m.any() else 0.0


# --------------------------------------------------------------- plots -------
def plot_waveforms(t, clean, noisy, filt, path, secs=4.0):
    n = int(secs * FS)
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for a, y, c, lbl in [(ax[0], clean, "#1a1a1a", "clean (reference)"),
                         (ax[1], noisy, "#c02020", "noisy: +wander +60 Hz hum"),
                         (ax[2], filt, "#1668b0", "filtered: HP 0.5 Hz + 60 Hz notch (causal)")]:
        a.plot(t[:n], y[:n], c, lw=0.9)
        a.set_ylabel("mV")
        a.text(0.01, 0.9, lbl, transform=a.transAxes, va="top",
               fontsize=10, family="monospace")
        a.grid(alpha=0.25)
    ax[2].set_xlabel("time (s)")
    fig.suptitle("Stage 1.4d — noise injection and causal filtering", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_spectrum(noisy, filt, path, fs=FS):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for y, c, lbl in [(noisy, "#c02020", "noisy"), (filt, "#1668b0", "filtered")]:
        f, pxx = signal.welch(y, fs=fs, nperseg=min(4096, len(y)))
        ax.semilogy(f, pxx, c, lw=1.1, label=lbl)
    ax.axvline(NOTCH_F, color="#888", ls="--", lw=1, label="60 Hz")
    ax.set_xlim(0, 80)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (mV²/Hz)")
    ax.set_title("Spectrum — the 60 Hz spike and sub-0.5 Hz drift are removed")
    ax.legend()
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main -------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="use a synthetic ECG instead of downloading record 100")
    ap.add_argument("--hum-mv", type=float, default=0.15, help="60 Hz hum amplitude (mV)")
    ap.add_argument("--wander-mv", type=float, default=0.30, help="baseline wander amplitude (mV)")
    ap.add_argument("--out-prefix", default="stage1_4d", help="output PNG prefix")
    args = ap.parse_args()

    if args.synthetic:
        t, clean = synthetic_ecg()
        src = "synthetic ECG"
    else:
        try:
            t, clean = load_record_100()
            src = "MIT-BIH record 100"
        except Exception as e:
            print(f"[!] Could not load record 100 ({e}); falling back to synthetic.")
            t, clean = synthetic_ecg()
            src = "synthetic ECG (fallback)"

    print(f"Signal: {src}, {len(clean)} samples @ {FS:.0f} Hz ({len(clean)/FS:.1f} s)")

    noisy = add_noise(clean, t, hum_mv=args.hum_mv, wander_mv=args.wander_mv)
    sos_hp, sos_notch, sos_all = design_filters()
    filt = apply_causal(sos_all, noisy)

    # --- spectral proof ---
    p60_noisy = band_power(noisy, FS, 59, 61)
    p60_filt = band_power(filt, FS, 59, 61)
    plo_noisy = band_power(noisy, FS, 0.01, 0.5)
    plo_filt = band_power(filt, FS, 0.01, 0.5)
    db = lambda a, b: 10 * np.log10(b / a) if a > 0 and b > 0 else float("nan")
    print("\nNoise suppression (causal filter):")
    print(f"  60 Hz band power   : {p60_noisy:.4e} -> {p60_filt:.4e}  ({db(p60_noisy, p60_filt):+.1f} dB)")
    print(f"  <0.5 Hz drift power: {plo_noisy:.4e} -> {plo_filt:.4e}  ({db(plo_noisy, plo_filt):+.1f} dB)")

    # --- detection check with YOUR real detector + matcher ---
    # Reference = the detector's own peaks on the CLEAN signal, so this isolates
    # the filters' effect (does noise/filtering change what the detector finds?).
    pk_clean = detect_rpeaks(clean, FS)
    pk_noisy = detect_rpeaks(noisy, FS)
    pk_filt  = detect_rpeaks(filt, FS)
    m_noisy = match_peaks(pk_noisy, pk_clean, FS)   # (detected, reference, fs)
    m_filt  = match_peaks(pk_filt,  pk_clean, FS)
    print("\nR-peak detection vs the clean-signal peaks (your detector, 150 ms tolerance):")
    print(f"  clean peaks : {len(pk_clean)}   noisy peaks : {len(pk_noisy)}   filtered peaks : {len(pk_filt)}")
    print(f"  noisy    : Se {m_noisy['sensitivity']*100:5.1f}%   PPV {m_noisy['ppv']*100:5.1f}%   F1 {m_noisy['f1']*100:5.1f}%")
    print(f"  filtered : Se {m_filt['sensitivity']*100:5.1f}%   PPV {m_filt['ppv']*100:5.1f}%   F1 {m_filt['f1']*100:5.1f}%")

    # --- plots ---
    wpath = f"{args.out_prefix}_waveforms.png"
    spath = f"{args.out_prefix}_spectrum.png"
    plot_waveforms(t, clean, noisy, filt, wpath)
    plot_spectrum(noisy, filt, spath)
    print(f"\nSaved: {wpath}, {spath}")

    # --- coefficients for the firmware port ---
    print_c_coeffs(sos_hp, sos_notch)


if __name__ == "__main__":
    main()
