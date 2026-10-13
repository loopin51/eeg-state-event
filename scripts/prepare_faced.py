"""Build the FACED window cache from Processed_data/subXXX.pkl (official preprocessing:
0.05-47 Hz, ICA-cleaned, re-referenced, 250 Hz, LAST 30 s of each of 28 clips, µV, [28, 32, 7500],
clips ordered by video index 1..28, channel order = second-cohort order for all 123 subjects).

t_onset is relative to (clip end - 30 s), identical across subjects, so stim-align pairs are exact.
Labels: 9 emotions (0 anger,1 disgust,2 fear,3 sadness,4 neutral,5 amusement,6 inspiration,7 joy,
8 tenderness); valence -1 (clips 1-12) / 0 (13-16) / +1 (17-28).
"""
import os, sys, glob, pickle, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.montage import positions_for
from ese.data.preprocess import filter_resample, session_stats, normalize, cut_contexts, TARGET_SFREQ, CONTEXT_S
from ese.data.cache import CacheWriter

SFREQ_IN = 250
EMO = [0]*3 + [1]*3 + [2]*3 + [3]*3 + [4]*4 + [5]*3 + [6]*3 + [7]*3 + [8]*3
VAL = [-1]*12 + [0]*4 + [1]*12

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=os.path.join(ROOT, 'data/raw/faced/FACED/Processed_data'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/cache'))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.raw, 'sub*.pkl')))
    pos, names = positions_for('faced')
    w = CacheWriter(a.out, 'faced', 32, int(CONTEXT_S * TARGET_SFREQ), pos,
                    extra=dict(source='Kaggle saifkhancse/faced-eeg Processed_data (official FACED preprocessing)', sfreq_in=SFREQ_IN,
                               note='last 30 s of each clip; t_onset relative to clip_end-30s', channel_names=names))
    for f in files:
        sid = int(os.path.basename(f)[3:6])
        d = np.asarray(pickle.load(open(f, 'rb')), dtype=np.float64)          # [28,32,7500]
        trials = [filter_resample(d[k], SFREQ_IN, notch_hz=None) for k in range(28)]   # already 0.05-47 Hz
        med, scale = session_stats(trials)
        for k, y in enumerate(trials):
            win, on = cut_contexts(normalize(y, med, scale))
            w.add(win, [dict(subject=sid, session=0, trial=k, clip_id=k + 1, t_onset=float(t), valence=VAL[k], label=EMO[k]) for t in on])
        if sid % 20 == 0: print(f'[faced] sub{sid:03d}: cache n={w.n}', flush=True)
    print(w.close())

if __name__ == '__main__':
    main()
