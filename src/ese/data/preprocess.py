"""Common preprocessing to the 2nd-study spec (docs/03 §1.1, docs/06 §3).

  1. IIR Butterworth order-4 zero-phase band-pass 0.5-45 Hz  (+ notch)
  2. polyphase resample -> 100 Hz
  3. per-subject/session per-channel robust z-score (median / IQR), fit on the
     WHOLE session (not per window), then clamp to ±5
  4. cut 10 s contexts with 5 s hop

Steps 1-2 operate on a continuous trial; step 3 needs session statistics, so
`session_stats` is computed once per (subject, session) over all its trials
after filtering/resampling, and applied by `normalize`.
"""
from math import gcd
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly, iirnotch, filtfilt

BAND = (0.5, 45.0)
TARGET_SFREQ = 100
CONTEXT_S = 10.0
HOP_S = 5.0
CLAMP = 5.0


def filter_resample(x, sfreq_in, band=BAND, notch_hz=None, target_sfreq=TARGET_SFREQ):
    """x: [C, T] continuous trial at sfreq_in -> float32 [C, T'] at target_sfreq."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = band
    hi = min(hi, sfreq_in / 2.0 * 0.95)
    sos = butter(4, [lo, hi], btype='band', fs=sfreq_in, output='sos')
    y = sosfiltfilt(sos, x, axis=1)
    if notch_hz is not None and notch_hz < sfreq_in / 2.0:
        b, a = iirnotch(notch_hz, Q=30.0, fs=sfreq_in)
        y = filtfilt(b, a, y, axis=1)
    if abs(target_sfreq - sfreq_in) > 1e-6:
        up, down = int(round(target_sfreq)), int(round(sfreq_in))
        g = gcd(up, down)
        y = resample_poly(y, up // g, down // g, axis=1)
    return y.astype(np.float32)


def session_stats(trials):
    """trials: list of [C, T_i] arrays from ONE (subject, session) -> (median [C,1], iqr [C,1])."""
    cat = np.concatenate(trials, axis=1)
    med = np.median(cat, axis=1, keepdims=True)
    q75, q25 = np.percentile(cat, [75, 25], axis=1, keepdims=True)
    scale = q75 - q25
    scale[scale == 0] = 1.0
    return med.astype(np.float32), scale.astype(np.float32)


def normalize(x, med, scale, clamp=CLAMP):
    return np.clip((x - med) / scale, -clamp, clamp).astype(np.float32)


def cut_contexts(x, sfreq=TARGET_SFREQ, context_s=CONTEXT_S, hop_s=HOP_S):
    """x: [C, T] -> (windows [N, C, L], onsets_s [N]). Drops the incomplete tail."""
    L, H = int(round(context_s * sfreq)), int(round(hop_s * sfreq))
    T = x.shape[1]
    if T < L:
        return np.empty((0, x.shape[0], L), np.float32), np.empty((0,), np.float32)
    starts = np.arange(0, T - L + 1, H)
    win = np.stack([x[:, s:s + L] for s in starts]).astype(np.float32)
    return win, (starts / sfreq).astype(np.float32)
