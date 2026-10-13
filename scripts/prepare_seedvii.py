"""Build the SEED-VII window cache from seed_vii/EEG_preprocessed/<subj>.mat.

Each file: 80 variables '1'..'80' (video index), [62, T] at 200 Hz (officially preprocessed). 20 videos per
session x 4 sessions; every subject watched all 80 clips -> clip_id = video index, session = (v-1)//20.
Emotion per video from emotion_label_and_stimuli_order.xlsx. Codes follow SEED-VII_stimulation.xlsx:
0 disgust, 1 fear, 2 sad, 3 neutral, 4 happy, 5 anger, 6 surprise.
valence: happy +1; disgust/fear/sad/anger -1; neutral & surprise 0 (excluded from binary valence).
continuous_labels/<subj>.mat: per video a [1, n] self-reported intensity curve, one value per 4 s.
For each 10 s window we store the mean of the overlapping 4 s label bins as `cont_label` (NaN if none).
"""
import os, sys, argparse, numpy as np, pandas as pd
from scipy.io import loadmat
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.montage import positions_for
from ese.data.preprocess import filter_resample, session_stats, normalize, cut_contexts, TARGET_SFREQ, CONTEXT_S
from ese.data.cache import CacheWriter

SFREQ_IN, LABEL_BIN_S = 200, 4.0
CODE = {'disgust': 0, 'fear': 1, 'sad': 2, 'neutral': 3, 'happy': 4, 'anger': 5, 'surprise': 6}
VAL = {0: -1, 1: -1, 2: -1, 3: 0, 4: 1, 5: -1, 6: 0}

def video_emotions(path):
    x = pd.read_excel(path, header=None); emo = {}
    for r in range(0, len(x), 2):
        for v, e in zip(x.iloc[r, 1:], x.iloc[r + 1, 1:]):
            if pd.notna(v): emo[int(v)] = CODE[str(e).strip().lower()]
    return emo

def cont_for_windows(curve, onsets, win_s=CONTEXT_S):
    curve = np.asarray(curve, dtype=np.float64).ravel(); out = np.full(len(onsets), np.nan, np.float32)
    starts = np.arange(len(curve)) * LABEL_BIN_S
    for i, t in enumerate(onsets):
        m = (starts < t + win_s) & (starts + LABEL_BIN_S > t)
        if m.any(): out[i] = curve[m].mean()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=os.path.join(ROOT, 'data/raw/seed_series_full/seed_vii'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/cache'))
    a = ap.parse_args()
    emo = video_emotions(os.path.join(a.raw, 'emotion_label_and_stimuli_order.xlsx')); assert len(emo) == 80, len(emo)
    pos, names = positions_for('seedvii')
    w = CacheWriter(a.out, 'seedvii', 62, int(CONTEXT_S * TARGET_SFREQ), pos,
                    extra=dict(source='Kaggle blackfox20092006/seed-series-multimodal-eeg-eye-full seed_vii/EEG_preprocessed', sfreq_in=SFREQ_IN,
                               notch_hz=50, channel_names=names, label_codes=CODE))
    for sid in range(1, 21):
        m = loadmat(os.path.join(a.raw, 'EEG_preprocessed', f'{sid}.mat')); cl = loadmat(os.path.join(a.raw, 'continuous_labels', f'{sid}.mat'))
        for sess in range(4):
            vids = list(range(sess * 20 + 1, sess * 20 + 21))
            trials = [filter_resample(np.asarray(m[str(v)], dtype=np.float64), SFREQ_IN, notch_hz=50.0) for v in vids]
            med, scale = session_stats(trials)
            for v, y in zip(vids, trials):
                win, on = cut_contexts(normalize(y, med, scale)); cont = cont_for_windows(cl[str(v)], on) if str(v) in cl else np.full(len(on), np.nan)
                w.add(win, [dict(subject=sid - 1, session=sess, trial=v - 1, clip_id=v, t_onset=float(t), valence=VAL[emo[v]], label=emo[v], cont_label=float(c)) for t, c in zip(on, cont)])
        del m; print(f'[seedvii] subj {sid:02d}: cache n={w.n}', flush=True)
    print(w.close())

if __name__ == '__main__':
    main()
