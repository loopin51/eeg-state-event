"""Hand-crafted baseline features from cached windows (docs/05 §2)."""
import numpy as np
from scipy.signal import welch

BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}


def de_features(windows, sfreq=100, bands=BANDS, nperseg=200, batch=512):
    """Differential entropy per (channel, band): DE = 0.5*log(2*pi*e*P_band).
    windows: memmap/array [N, C, L] (float16 ok). Returns float32 [N, C*len(bands)]."""
    N = windows.shape[0]; out = []
    for i in range(0, N, batch):
        x = np.asarray(windows[i:i + batch], dtype=np.float32)
        f, P = welch(x, fs=sfreq, nperseg=nperseg, axis=-1)               # [b, C, F]
        feats = []
        for lo, hi in bands.values():
            m = (f >= lo) & (f < hi)
            p = P[..., m].mean(-1) + 1e-12
            feats.append(0.5 * np.log(2 * np.pi * np.e * p))
        out.append(np.stack(feats, -1).reshape(x.shape[0], -1))          # [b, C*B]
    return np.concatenate(out).astype(np.float32)
