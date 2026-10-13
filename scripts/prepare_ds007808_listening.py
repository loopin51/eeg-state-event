"""ds007808 listening task: per-segment features and targets for the day-held-out identification test (D-057).

Segments = events of task-listening with duration >= 2.0 s; window 0-2.0 s after onset (events onset, seconds).
Channels (eeg.json Notes): 0-127 EEG, 128-129 listening-audio monitor, 130-131 speaking microphone, 132-133 EOG,
134-137 upper/lower-lip EMG, 139 trigger.
  LF   [n, 128, 40]  EEG 1-8 Hz band-pass (zero-phase), 20 Hz, 0-2 s           (primary feature)
  PRE  [n, 128, 10]  same, -0.5-0 s                                           (leak meter)
  EXG  [n, 6, 40]    EOG + lip EMG channels, same as LF                        (muscle / ocular control)
  DE   [n, 768]      EEG 1-100 Hz + 50/100 Hz notch, 6-band log power over 0-2 s (secondary feature)
  AUD  [n, 40]       audio-monitor envelope: mean of ch 128-129, 80-550 Hz band, |hilbert|, 1-8 Hz, 20 Hz, 0-2 s
  meta: day, run, onset (s, in run), duration, n_chars, text
One npz per subject in data/cache/ds007808_listening/.

usage: python scripts/prepare_ds007808_listening.py [--subjects 1 2]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, re, time, argparse, numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, resample_poly, welch, iirnotch, filtfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data/raw/ds007808'); OUT = os.path.join(ROOT, 'data/cache/ds007808_listening')
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45), (45, 100)]
EEG, AUD_CH, EXG_CH = list(range(128)), [128, 129], [132, 133, 134, 135, 136, 137]
MIN_DUR, WIN, PRE_WIN, FS_LF = 2.0, 2.0, 0.5, 20


def bp(x, fs, lo, hi):
    return sosfiltfilt(butter(4, [lo, hi], btype='band', fs=fs, output='sos'), x, axis=-1)


def run_features(edf, ev):
    import mne
    raw = mne.io.read_raw_edf(edf, preload=True, verbose=False); fs = raw.info['sfreq']; X = raw.get_data()
    assert X.shape[0] >= 140, f'{edf}: {X.shape[0]} channels'
    seg = ev[(ev.trial_type == 'listening') & (ev.duration >= MIN_DUR)].reset_index(drop=True)
    seg = seg[(seg.onset - PRE_WIN >= 0) & ((seg.onset + WIN) * fs < X.shape[1])].reset_index(drop=True)
    dec = int(round(fs / FS_LF))
    lf = resample_poly(bp(X[EEG + EXG_CH], fs, 1, 8), 1, dec, axis=-1)
    i0 = np.round(seg.onset.values * FS_LF).astype(int); nw, npre = int(WIN * FS_LF), int(PRE_WIN * FS_LF)
    LF = np.stack([lf[:128, i:i + nw] for i in i0]); PRE = np.stack([lf[:128, i - npre:i] for i in i0]); EXG = np.stack([lf[128:, i:i + nw] for i in i0])
    wide = bp(X[EEG], fs, 1, 100)
    for f0 in (50, 100):
        b, a = iirnotch(f0, 30, fs); wide = filtfilt(b, a, wide, axis=-1)
    s0 = np.round(seg.onset.values * fs).astype(int); nwf = int(WIN * fs)
    w = np.stack([wide[:, s:s + nwf] for s in s0]); f, P = welch(w, fs=fs, nperseg=int(fs), axis=-1)
    DE = np.log(np.stack([P[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in BANDS], -1) + 1e-30).reshape(len(seg), -1)
    aud = np.abs(hilbert(bp(X[AUD_CH].mean(0), fs, 80, min(550, fs / 2 - 10))))
    aud = resample_poly(bp(aud, fs, 1, 8), 1, dec); AUD = np.stack([aud[i:i + nw] for i in i0])
    return seg, dict(LF=LF, PRE=PRE, EXG=EXG, DE=DE, AUD=AUD)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--subjects', type=int, nargs='+', default=[1, 2]); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    for sub in a.subjects:
        edfs = sorted(glob.glob(os.path.join(RAW, f'sub-{sub:02d}', 'ses-*', 'eeg', '*task-listening_*_eeg.edf')))
        days = sorted({re.search(r'ses-(\d+)', e).group(1) for e in edfs}); parts = {}; meta = []
        for e in edfs:
            ses, run = re.search(r'ses-(\d+)', e).group(1), int(re.search(r'run-(\d+)', e).group(1))
            ev = pd.read_csv(e.replace('_eeg.edf', '_events.tsv'), sep='\t')
            seg, F = run_features(e, ev)
            for k, v in F.items(): parts.setdefault(k, []).append(v.astype(np.float32))
            meta.append(pd.DataFrame(dict(day=days.index(ses), run=run, onset=seg.onset.values, duration=seg.duration.values,
                                          text=seg.value.astype(str).values, n_chars=seg.value.astype(str).str.len().values)))
            print(f'sub-{sub:02d} ses-{ses} run-{run:02d}: {len(seg)} segments ({time.time() - t0:.0f}s)', flush=True)
        M = pd.concat(meta, ignore_index=True); out = {k: np.concatenate(v) for k, v in parts.items()}
        assert all(np.isfinite(v).all() for v in out.values()), 'non-finite feature values'
        np.savez(os.path.join(OUT, f'sub-{sub:02d}.npz'), day=M.day.values, run=M.run.values, onset=M.onset.values,
                 duration=M.duration.values, n_chars=M.n_chars.values, text=M.text.values, **out)
        print(f'sub-{sub:02d}: {len(M)} segments, days {np.bincount(M.day.values).tolist()}, ' +
              ' '.join(f'{k}{list(v.shape[1:])}' for k, v in out.items()), flush=True)


if __name__ == '__main__':
    main()
