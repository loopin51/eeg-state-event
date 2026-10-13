"""Build the EAV window cache (Stage 0-c, D-039/D-041).

Source: data/raw/eav/subjectN/EEG/subjectN_eeg.mat (server: data/raw/eav -> /mnt/data/eeg_datasets/eav, D-044)
('seg' or 'seg1', [10000, 30, 200] = 20 s at 500 Hz x 30 ch x
200 trials; distributed data already high-passed 0.5 Hz and low-passed 50 Hz) and subjectN_eeg_label.mat
('label', one-hot [10, 200]).
Label index (D-041): 0/1 neutral, 2/3 sadness, 4/5 anger, 6/7 happiness, 8/9 calmness; even = listening,
odd = speaking. valence +1 happiness/calmness, -1 sadness/anger, 0 neutral. No shared stimulus -> clip_id = -1.
Common spec: 0.5-45 Hz IIR zero-phase, 100 Hz, per-subject robust z-score over all 200 trials, 10 s contexts hop 5 s.
Extra meta columns: cond (0 listen, 1 speak), emotion (0..4), order (trial position 0..199 in the session).
"""
import os, sys, argparse, numpy as np
from scipy.io import loadmat
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.montage import positions
from ese.data.preprocess import filter_resample, session_stats, normalize, cut_contexts, TARGET_SFREQ, CONTEXT_S
from ese.data.cache import CacheWriter

CHANNELS = 'FP1 FP2 F7 F3 FZ F4 F8 FC5 FC1 FC2 FC6 T7 C3 CZ C4 T8 CP5 CP1 CP2 CP6 P7 P3 PZ P4 P8 PO9 O1 OZ O2 PO10'.split()
EMOTION = ['neutral', 'sadness', 'anger', 'happiness', 'calmness']
VALENCE = [0, -1, -1, 1, 1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', default=os.path.join(ROOT, 'data/raw/eav'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/cache')); a = ap.parse_args()
    w = CacheWriter(a.out, 'eav', len(CHANNELS), int(CONTEXT_S * TARGET_SFREQ), positions(CHANNELS),
                    extra=dict(source='EAV (Lee et al., Sci. Data 2024), data/raw/eav', sfreq_in=500, channel_names=CHANNELS,
                               label_map='0/1 neutral, 2/3 sadness, 4/5 anger, 6/7 happiness, 8/9 calmness; even listen, odd speak'))
    for sid in range(1, 43):
        d = loadmat(os.path.join(a.raw, f'subject{sid}', 'EEG', f'subject{sid}_eeg.mat'))
        x = d['seg'] if 'seg' in d else d['seg1']                                      # [10000, 30, 200]
        lab = loadmat(os.path.join(a.raw, f'subject{sid}', 'EEG', f'subject{sid}_eeg_label.mat'))['label'].argmax(0)
        trials = [filter_resample(x[:, :, k].T, 500, notch_hz=None) for k in range(x.shape[2])]
        med, scale = session_stats(trials)
        for k, tr in enumerate(trials):
            win, on = cut_contexts(normalize(tr, med, scale)); e = int(lab[k]) // 2
            w.add(win, [dict(subject=sid, session=0, trial=k, clip_id=-1, t_onset=float(t), valence=VALENCE[e], label=int(lab[k]),
                             cond=int(lab[k]) % 2, emotion=e, order=k) for t in on])
        print(f'[eav] subject {sid}: n={w.n}', flush=True)
    print(w.close())


if __name__ == '__main__':
    main()
