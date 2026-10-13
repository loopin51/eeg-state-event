"""Phase 0.1: build the DEAP window cache from data_preprocessed_python/s01..s32.dat.

DEAP preprocessed: 32 EEG ch (Geneva order == montage.DEAP32) + 8 peripheral,
128 Hz, 63 s per trial = 3 s pre-trial baseline + 60 s video; already 4-45 Hz
band-passed, EOG-removed, common-average referenced. We drop the baseline,
apply the common spec (filter/resample/robust z-score per subject), and cut
10 s contexts. clip_id = Experiment_id (1-40) from participant_ratings.xls so
that stim-align can pair subjects who watched the same video.
"""
import os, sys, glob, pickle, argparse, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.montage import positions_for
from ese.data.preprocess import filter_resample, session_stats, normalize, cut_contexts, TARGET_SFREQ, CONTEXT_S
from ese.data.cache import CacheWriter

SFREQ_IN, BASELINE_S, EEG_CH = 128, 3, 32

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=os.path.join(ROOT, 'data/raw/deap/deap-dataset'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/cache'))
    ap.add_argument('--subjects', default='all')
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.raw, 'data_preprocessed_python', 's*.dat')))
    if a.subjects != 'all':
        keep = set(int(s) for s in a.subjects.split(',')); files = [f for f in files if int(os.path.basename(f)[1:3]) in keep]
    ratings = pd.read_excel(os.path.join(a.raw, 'Metadata', 'participant_ratings.xls'))
    ratings.columns = [c.strip().lower() for c in ratings.columns]
    pos, names = positions_for('deap')
    w = CacheWriter(a.out, 'deap', EEG_CH, int(CONTEXT_S * TARGET_SFREQ), pos,
                    extra=dict(source='Kaggle manh123df/deap-dataset data_preprocessed_python', sfreq_in=SFREQ_IN,
                               note='source already 4-45 Hz band-passed; delta band absent', channel_names=names))
    for f in files:
        sid = int(os.path.basename(f)[1:3])          # 1..32
        d = pickle.load(open(f, 'rb'), encoding='latin1')
        data, labels = d['data'], d['labels']           # [40,40,8064], [40,4]
        r = ratings[ratings['participant_id'] == sid].set_index('trial')
        med_all = float(np.median(labels[:, 0]))       # per-subject median valence (1st-study convention)
        trials = []
        for k in range(40):
            x = data[k, :EEG_CH, BASELINE_S * SFREQ_IN:]                    # [32, 7680]
            trials.append(filter_resample(x, SFREQ_IN, notch_hz=None))       # already notch-free at 4-45
        med, scale = session_stats(trials)
        for k, y in enumerate(trials):
            win, on = cut_contexts(normalize(y, med, scale))
            val, aro = float(labels[k, 0]), float(labels[k, 1])
            clip = int(r.loc[k + 1, 'experiment_id']) if (k + 1) in r.index else -1
            rows = [dict(subject=sid - 1, session=0, trial=k, clip_id=clip, t_onset=float(t),
                         valence=(1 if val > med_all else -1), label=int(val > 5), valence_raw=val, arousal_raw=aro,
                         arousal=(1 if aro > 5 else -1)) for t in on]
            w.add(win, rows)
        print(f'[deap] s{sid:02d}: {len(trials)} trials -> cache n={w.n}', flush=True)
    h = w.close(); print(h)

if __name__ == '__main__':
    main()
