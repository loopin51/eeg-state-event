"""THINGS-EEG2 preprocessing of one session file (D-111, F-V): low-pass -> 250 Hz -> epochs -> baseline -> catch removal -> MVNN.

Order and memory (D-111 debate section 3.2): the continuous 1000-Hz recording is filtered one channel at a time with a
linear-phase FIR low-pass (centred, zero delay), and each epoch is taken directly from the filtered channel on a 4-ms grid
anchored at its own event sample (decimation by 4 inside the epoch, no resampling jitter). No 1000-Hz epoch array is ever
built: the only large arrays are the raw dict (float64 as stored) and the float32 epoch array [n, C, T250].

MVNN (Guggenmos 2018, Gifford 2022 "epochs" variant): for every training epoch the Ledoit-Wolf covariance over its time samples
(sklearn _cov(shrinkage='auto') formula: standardise, ledoit_wolf, rescale), averaged over the epochs of each image condition,
then over conditions; W = sigma^(-1/2) (symmetric). Unlike Gifford's mvnn(), sigma comes from the training runs of the
session only (no test covariance), and the test file of the same session is whitened with that W inside the verdict job.

Functions take the config dict of configs/fv_preprocess.yaml (keys `filter`, `epoch`, `events`, `mvnn`).
"""
import time, numpy as np
from scipy import signal
from sklearn.covariance import ledoit_wolf


def fir_lowpass(cfg, fs):
    f = cfg['filter']; nyq = fs / 2; width = f['stop_hz'] - f['pass_hz']
    n, beta = signal.kaiserord(f['atten_db'], width / nyq); n |= 1                               # odd: centred, zero delay
    return signal.firwin(n, (f['pass_hz'] + f['stop_hz']) / 2, window=('kaiser', beta), fs=fs)


def find_events(stim, fs):
    """events as mne.find_events(raw, stim_channel='stim') with its defaults (as Gifford's epoching): [sample, previous, code]"""
    import mne
    raw = mne.io.RawArray(np.asarray(stim, np.float64)[None], mne.create_info(['stim'], fs, ['stim']), verbose='error')
    return mne.find_events(raw, stim_channel='stim', verbose='error')


def grid(cfg, fs):
    """epoch sample offsets at the input rate (ints) and the epoch times at the output rate"""
    e = cfg['epoch']; step = int(round(fs / e['sfreq'])); assert step * e['sfreq'] == fs, f'{fs} Hz is not a multiple of {e["sfreq"]} Hz'
    n = int(round((e['tmax'] - e['tmin']) * e['sfreq'])); off = int(round(e['tmin'] * fs)) + step * np.arange(n)
    return off, off / fs


def epochs_from_raw(d, cfg, log=print):
    """raw dict (rawio.load_dict) -> dict(X float32 [n, C, T], code, sample, ch_names, times, fs, counts)"""
    t0 = time.time(); fs = float(d['sfreq']); x = d['raw_eeg_data']; types = list(d['ch_types']); names = list(d['ch_names'])
    si = [i for i, t in enumerate(types) if t == 'stim']; assert len(si) == 1, f'stim channels {si}'
    eeg = [i for i, t in enumerate(types) if t == 'eeg']; assert len(eeg) + 1 == len(types), f'channel types {sorted(set(types))}'
    ev = find_events(x[si[0]], fs); code = ev[:, 2].astype(np.int64); samp = ev[:, 0].astype(np.int64)
    off, times = grid(cfg, fs); N = x.shape[1]
    inside = (samp + off[0] >= 0) & (samp + off[-1] < N); catch = np.isin(code, cfg['events']['catch_codes'])
    keep = inside & ~catch; counts = dict(events=int(len(code)), catch=int(catch.sum()), outside=int((~inside & ~catch).sum()), kept=int(keep.sum()))
    code, samp = code[keep], samp[keep]; idx = samp[:, None] + off[None]
    h = fir_lowpass(cfg, fs); X = np.empty((len(samp), len(eeg), len(off)), np.float32)
    for j, c in enumerate(eeg):
        y = signal.oaconvolve(np.asarray(x[c], np.float64), h, mode='same'); X[:, j] = y[idx]
    nb = int((times < 0).sum()); X -= X[..., :nb].mean(-1, keepdims=True)                         # baseline: mean of tmin..0 s
    log(f'epochs {X.shape} (fir {len(h)} taps, baseline {nb} samples, {counts}) {time.time() - t0:.0f}s')
    return dict(X=X, code=code, sample=samp, ch_names=[names[c] for c in eeg], times=times.astype(np.float32), fs=fs,
                counts=counts, fir_taps=int(len(h)))


def lw_cov(Z):
    """sklearn.discriminant_analysis._cov(Z, shrinkage='auto'): Z [samples, channels]"""
    sd = Z.std(0); sd[sd == 0] = 1.0; S = ledoit_wolf((Z - Z.mean(0)) / sd)[0]
    return sd[:, None] * S * sd[None]


def mvnn_fit(X, code, times, cfg):
    """sigma and W = sigma^(-1/2) from training epochs X [n, C, T] (epoch covariance over time, condition mean, grand mean)"""
    m = cfg['mvnn']; tw = (times >= m['tmin'] - 1e-9) & (times < m['tmax'] - 1e-9); C = X.shape[1]
    conds = np.unique(code); S = np.zeros((C, C));
    for c in conds:
        ii = np.where(code == c)[0]; S += np.mean([lw_cov(X[i][:, tw].T.astype(np.float64)) for i in ii], 0)
    S /= len(conds); w, V = np.linalg.eigh(S); assert w.min() > 0, f'covariance not positive definite (min eigenvalue {w.min()})'
    return S, (V * w ** -0.5) @ V.T


def whiten(X, W):
    """W [C, C] applied to every epoch, in place on a float32 array"""
    for a in range(0, len(X), 2048): X[a:a + 2048] = np.einsum('dc,nct->ndt', W.astype(np.float32), X[a:a + 2048])
    return X


def test_epochs(path, cfg, mvnn_npz, seal_table, unseal):
    """verdict job only: the sealed test file of one session -> whitened epochs with that session's TRAINING-run W.
    rawio.guard stops unless unseal is True and the file's sha256 matches the seal table."""
    from ese.things import rawio
    d = rawio.load_dict(path, 'raw', unseal=unseal, seal_table=seal_table); E = epochs_from_raw(d, cfg); del d
    m = np.load(mvnn_npz); assert list(m['ch_names']) == E['ch_names'], 'test and training channel orders differ'
    whiten(E['X'], m['W']); return E
