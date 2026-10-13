"""d'Ascoli-spec preprocessing for raw EEG affect windows.

Pipeline (matches Stage-1 content-encoder preprocessing, adapted for continuous
emotion EEG rather than word-locked epochs):
  1. bandpass [0.1, 40] Hz  (IIR Butterworth order-4, zero-phase filtfilt;
     robust for the short 3 s epochs that torcheeg yields)
  2. resample native -> 50 Hz (polyphase, anti-aliased)
  3. RobustScaler per channel (median/IQR), fit on the window
  4. clamp to [-clamp, +clamp]  (default 5)
  (baseline correction is OFF: SEED emotion trials are not event-locked, so there
   is no pre-stimulus baseline interval as in the reading data.)

Output: float32 [n_channels, n_times_50hz].
Also provides SEED 62-ch and DEAP 32-ch normalized 2D scalp coordinates for the
ChannelMerger positional embedding (FourierEmb over [0,1]^2).
"""
import numpy as np
from scipy.stats import iqr
from scipy.signal import butter, sosfiltfilt, resample_poly

DASCOLI_BAND = (0.1, 40.0)
TARGET_SFREQ = 50.0

def preprocess_window(x, sfreq_in, band=DASCOLI_BAND, target_sfreq=TARGET_SFREQ,
                      clamp=5.0, robust=True):
    """x: [n_channels, n_times] float. Returns [n_channels, n_times_target] float32."""
    x = np.asarray(x, dtype=np.float64)
    C, T = x.shape
    # 1. bandpass — IIR Butterworth zero-phase (filtfilt). torcheeg hands us short
    #    (3 s) epochs; a 0.1 Hz FIR highpass would need thousands of taps (> window)
    #    and error/distort. IIR is recursive (no length requirement) and is the
    #    standard choice for short EEG epochs; the low edge is soft here anyway
    #    (0.1 Hz = 10 s period > window) and DC/drift is removed by the RobustScaler.
    lo, hi = band
    hi = min(hi, sfreq_in / 2.0 * 0.99)           # keep below Nyquist
    sos = butter(4, [lo, hi], btype="band", fs=sfreq_in, output="sos")
    padlen = min(3 * (sos.shape[0]), T - 1)
    y = sosfiltfilt(sos, x, axis=1, padlen=padlen)
    # 2. resample to target_sfreq (poly resample, anti-aliased)
    if abs(target_sfreq - sfreq_in) > 1e-6:
        from math import gcd
        up = int(round(target_sfreq)); down = int(round(sfreq_in))
        g = gcd(up, down); up //= g; down //= g
        y = resample_poly(y, up, down, axis=1)
    # 3. RobustScaler per channel
    if robust:
        med = np.median(y, axis=1, keepdims=True)
        sc = iqr(y, axis=1, keepdims=True); sc[sc == 0] = 1.0
        y = (y - med) / sc
    # 4. clamp
    if clamp is not None:
        y = np.clip(y, -clamp, clamp)
    return y.astype(np.float32)


def seed62_positions():
    """Normalized 2D coords [62,2] in [0,1] for the SEED 62-ch ESI/10-20 montage.
    Uses MNE standard_1020; falls back gracefully for the 4 non-1020 SEED names."""
    seed_ch = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
               'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ',
               'C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
               'P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4',
               'PO6','PO8','CB1','O1','OZ','O2','CB2']
    mont = mne.channels.make_standard_montage('standard_1020')
    pos = mont.get_positions()['ch_pos']
    # map by uppercase; CB1/CB2 (SEED-specific) approximate to nearby occipital
    alias = {'CB1':'O1','CB2':'O2'}
    xy = []
    for ch in seed_ch:
        key = ch.title() if ch.title() in pos else ch
        # MNE keys are like 'Fp1','Fpz'... normalize
        found = None
        for k in pos:
            if k.upper() == ch.upper():
                found = k; break
        if found is None and ch in alias:
            for k in pos:
                if k.upper() == alias[ch].upper():
                    found = k; break
        if found is None:
            xy.append([np.nan, np.nan])
        else:
            xy.append([pos[found][0], pos[found][1]])  # x (L-R), y (post-ant)
    xy = np.array(xy)
    # normalize to [0,1]^2 (invalid -> -0.1 sentinel handled by ChannelMerger)
    valid = ~np.isnan(xy).any(axis=1)
    mn = xy[valid].min(0); mx = xy[valid].max(0)
    norm = (xy - mn) / (mx - mn)
    norm[~valid] = -0.1
    return norm.astype(np.float32), seed_ch


DEAP_CHANNELS = ['FP1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7',
                 'PO3','O1','OZ','PZ','FP2','AF4','FZ','F4','F8','FC6','FC2','CZ',
                 'C4','T8','CP6','CP2','P4','P8','PO4','O2']

def deap32_positions():
    """Normalized 2D coords [32,2] in [0,1] for the DEAP 32-ch 10-20 montage,
    on the SAME normalization frame convention as seed62_positions so the two
    montages live in a shared ChannelMerger coordinate space."""
    import numpy as np, mne
    mont = mne.channels.make_standard_montage('standard_1020')
    pos = mont.get_positions()['ch_pos']
    xy = []
    for ch in DEAP_CHANNELS:
        found = None
        for k in pos:
            if k.upper() == ch.upper():
                found = k; break
        xy.append([pos[found][0], pos[found][1]] if found else [np.nan, np.nan])
    xy = np.array(xy)
    valid = ~np.isnan(xy).any(axis=1)
    mn = xy[valid].min(0); mx = xy[valid].max(0)
    norm = (xy - mn) / (mx - mn)
    norm[~valid] = -0.1
    return norm.astype(np.float32), DEAP_CHANNELS
