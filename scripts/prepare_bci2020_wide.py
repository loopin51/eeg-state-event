"""Stage 0-a (D-039): BCI2020 Track 3 trial arrays under the wide-band pipelines.

  P1: 256 Hz raw -> 0.5 Hz high-pass (Butterworth 4, zero-phase), 60 Hz notch, polyphase resample to 200 Hz,
      no low-pass beyond Nyquist (100 Hz)
  P2: same as P1 plus a 45 Hz low-pass (separates the band effect from the sampling-rate effect)
Both: per-subject robust z-score over all its Training+Validation trials, clamp +-5 (same as the P0 cache).
Output: data/cache/bci2020_<p>/x.npy (float16 [N, 64, T]), meta.parquet (subject, session, trial, label),
        header.json. session 0 = official Training set (300/subject), 1 = official Validation set (50/subject).
The trial keeps the full epoch (-0.5 s to +2.6 s around the cue); header records the onset sample.
"""
import os, sys, json, argparse, numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.preprocess import session_stats, normalize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_bci2020 import load_split, CLASSES

FS_OUT = 200


def prep(x, fs, lowpass):
    """x: [C, T] at fs -> float32 [C, T'] at FS_OUT."""
    sos = butter(4, [0.5, lowpass], btype='band', fs=fs, output='sos') if lowpass else butter(4, 0.5, btype='high', fs=fs, output='sos')
    y = sosfiltfilt(sos, np.asarray(x, np.float64), axis=1)
    b, a = iirnotch(60.0, Q=30.0, fs=fs); y = filtfilt(b, a, y, axis=1)
    return resample_poly(y, 25, 32, axis=1).astype(np.float32)             # 256 -> 200 Hz


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('pipeline', choices=['p1', 'p2'])
    ap.add_argument('--raw', default=os.path.join(ROOT, 'data/raw/bci2020/BCI2020 EEG Signal for Words')); a = ap.parse_args()
    lowpass = None if a.pipeline == 'p1' else 45.0
    out = os.path.join(ROOT, 'data/cache', f'bci2020_{a.pipeline}'); os.makedirs(out, exist_ok=True)
    X, rows = [], []
    for sid in range(1, 16):
        xt, yt, names, fs, classes = load_split(os.path.join(a.raw, 'Training set', f'Data_Sample{sid:02d}.mat'), 'epo_train')
        xv, yv, _, _, _ = load_split(os.path.join(a.raw, 'Validation set', f'Data_Sample{sid:02d}.mat'), 'epo_validation')
        assert classes == CLASSES and fs == 256.0
        trials = [prep(t, fs, lowpass) for t in np.concatenate([xt, xv])]
        med, scale = session_stats(trials)
        X.append(np.stack([normalize(t, med, scale) for t in trials]).astype(np.float16))
        for i, (s, l) in enumerate(zip(np.r_[np.zeros(len(yt), int), np.ones(len(yv), int)], np.r_[yt, yv])):
            rows.append(dict(subject=sid - 1, session=int(s), trial=i, label=int(l)))
        print(f'[{a.pipeline}] subject {sid:02d}: {len(trials)} trials, T={trials[0].shape[1]}', flush=True)
    X = np.concatenate(X); np.save(os.path.join(out, 'x.npy'), X)
    pd.DataFrame(rows).to_parquet(os.path.join(out, 'meta.parquet'))
    json.dump(dict(pipeline=a.pipeline, sfreq=FS_OUT, shape=list(X.shape), channel_names=names, classes=CLASSES,
                   epoch_start_s=-0.5, onset_sample=int(round(0.5 * FS_OUT)), highpass=0.5, lowpass=lowpass, notch=60.0,
                   norm='per-subject robust z-score over Training+Validation, clamp 5'),
              open(os.path.join(out, 'header.json'), 'w'), indent=1)
    print(out, X.shape, f'{X.nbytes / 1e6:.0f} MB')


if __name__ == '__main__':
    main()
