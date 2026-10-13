"""Build the BCI2020 Track 3 (imagined speech, 5 phrases) trial cache (Phase 3.0, D-035).

Source: Kaggle abdulkareembageri/imagined-speech-eeg-signal-bci2020, 'Training set' (epo_train, 300 trials) and
'Validation set' (epo_validation, 50 trials) per subject; x is [time, channel, trial] at 256 Hz, y one-hot [5, trial].
The Test set is MATLAB v7.3 with uncertain labels and is not used.
Common spec: 0.5-45 Hz IIR zero-phase, polyphase resample to 100 Hz, per-subject robust z-score over all its trials,
clamp +-5. One window per trial (no context cutting). session = 0 train split, 1 validation split.
"""
import os, sys, glob, argparse, numpy as np
from scipy.io import loadmat
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.montage import positions
from ese.data.preprocess import filter_resample, session_stats, normalize
from ese.data.cache import CacheWriter

CLASSES = ['Hello', 'Helpme', 'Stop', 'Thankyou', 'Yes']


def load_split(path, key):
    e = loadmat(path)[key][0, 0]
    x = np.transpose(e['x'], (2, 1, 0)).astype(np.float64)          # [trial, ch, time]
    y = e['y'].argmax(0).astype(int)
    names = [str(c[0]) for c in e['clab'][0]]; fs = float(e['fs'][0, 0])
    classes = [str(c[0]) for c in e['className'][0]]
    return x, y, names, fs, classes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', default=os.path.join(ROOT, 'data/raw/bci2020/BCI2020 EEG Signal for Words'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/cache')); a = ap.parse_args()
    w = None
    for sid in range(1, 16):
        xt, yt, names, fs, classes = load_split(os.path.join(a.raw, 'Training set', f'Data_Sample{sid:02d}.mat'), 'epo_train')
        xv, yv, names_v, fs_v, _ = load_split(os.path.join(a.raw, 'Validation set', f'Data_Sample{sid:02d}.mat'), 'epo_validation')
        assert names == names_v and fs == fs_v and classes == CLASSES, (classes, fs, fs_v)
        trials = [filter_resample(t, fs, notch_hz=None) for t in np.concatenate([xt, xv])]
        med, scale = session_stats(trials); T = min(t.shape[1] for t in trials)
        win = np.stack([normalize(t, med, scale)[:, :T] for t in trials])
        if w is None:
            pos = positions(names)
            w = CacheWriter(a.out, 'bci2020', len(names), T, pos,
                            extra=dict(source='Kaggle abdulkareembageri/imagined-speech-eeg-signal-bci2020 (Training+Validation sets)',
                                       sfreq_in=fs, channel_names=names, classes=CLASSES,
                                       invalid_positions=[n for n, p in zip(names, pos) if p[0] < 0]))
        labels = np.concatenate([yt, yv]); sess = np.r_[np.zeros(len(yt), int), np.ones(len(yv), int)]
        w.add(win, [dict(subject=sid - 1, session=int(s), trial=i, clip_id=-1, t_onset=0.0, valence=0, label=int(l))
                    for i, (s, l) in enumerate(zip(sess, labels))])
        print(f'[bci2020] subj {sid:02d}: {len(labels)} trials, T={T}', flush=True)
    print(w.close())


if __name__ == '__main__':
    main()
