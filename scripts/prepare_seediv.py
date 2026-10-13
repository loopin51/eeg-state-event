"""Phase 0.1: build the SEED-IV window cache from eeg_raw_data/{1,2,3}/<subj>_<date>.mat.

Each .mat holds 24 trials '<name>_eeg1'..'<name>_eeg24', 62 ch x T at 200 Hz.
Labels per session are fixed (ReadMe.txt): 0 neutral, 1 sad, 2 fear, 3 happy.
clip_id = (session-1)*24 + trial  (72 distinct clips; every subject of a session
saw the same 24 clips) -> stim-align pairs are available within each session.
Robust z-score statistics are per (subject, session).
"""
import os, sys, re, glob, argparse, numpy as np
from scipy.io import loadmat
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.montage import positions_for
from ese.data.preprocess import filter_resample, session_stats, normalize, cut_contexts, TARGET_SFREQ, CONTEXT_S
from ese.data.cache import CacheWriter

SFREQ_IN = 200
LABELS = {1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
          2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
          3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]}
VALENCE = {0: 0, 1: -1, 2: -1, 3: 1}      # neutral excluded from binary valence

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=os.path.join(ROOT, 'data/raw/seediv'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/cache'))
    ap.add_argument('--subjects', default='all')
    a = ap.parse_args()
    raw = sorted(glob.glob(os.path.join(a.raw, '**', 'eeg_raw_data'), recursive=True), key=len)[0]   # top-level copy; seed_iv/ is a duplicate
    pos, names = positions_for('seediv')
    w = CacheWriter(a.out, 'seediv', 62, int(CONTEXT_S * TARGET_SFREQ), pos,
                    extra=dict(source='Kaggle phhasian0710/seed-iv eeg_raw_data', sfreq_in=SFREQ_IN, notch_hz=50, channel_names=names))
    for sess in (1, 2, 3):
        files = sorted(glob.glob(os.path.join(raw, str(sess), '*.mat')), key=lambda p: int(os.path.basename(p).split('_')[0]))
        for f in files:
            sid = int(os.path.basename(f).split('_')[0])   # 1..15
            if a.subjects != 'all' and sid not in set(int(s) for s in a.subjects.split(',')): continue
            m = loadmat(f)
            keys = {int(re.search(r'eeg(\d+)$', k).group(1)): k for k in m if re.search(r'_eeg\d+$', k)}
            trials = [filter_resample(m[keys[k]].astype(np.float64), SFREQ_IN, notch_hz=50.0) for k in range(1, 25)]
            med, scale = session_stats(trials)
            for k, y in enumerate(trials):
                win, on = cut_contexts(normalize(y, med, scale))
                lab = LABELS[sess][k]
                rows = [dict(subject=sid - 1, session=sess - 1, trial=k, clip_id=(sess - 1) * 24 + k, t_onset=float(t),
                             valence=VALENCE[lab], label=lab) for t in on]
                w.add(win, rows)
            print(f'[seediv] session {sess} subj {sid:02d}: cache n={w.n}', flush=True)
    h = w.close(); print(h)

if __name__ == '__main__':
    main()
