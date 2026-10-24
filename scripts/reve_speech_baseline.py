"""D-125 arm A0: DE + logistic regression baseline on the same tasks and splits as the neural arms (server CPU, gpuq).

usage: python scripts/reve_speech_baseline.py configs/reve_speech.yaml [--tasks mode,cat,word,bci_within,bci_loso]

Features: log variance of each channel in delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-75 Hz (Butterworth order 4,
zero-phase, per epoch), standardised on the training subjects, LogisticRegression(C=1, lbfgs, max_iter=2000).
Writes results/baselines/reve_speech_de.json with the same per-subject metrics as scripts/reve_speech_train.py.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, argparse
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from reve_speech_train import load_task, folds, scores   # noqa: E402

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 75)]


def de(x, fs=200):
    from scipy.signal import butter, sosfiltfilt
    out = [np.log(sosfiltfilt(butter(4, b, btype='band', fs=fs, output='sos'), x, axis=-1).var(-1) + 1e-12) for b in BANDS]
    return np.concatenate(out, 1).astype(np.float32)


def main():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--tasks', default='mode,cat,word,bci_within,bci_loso'); a = ap.parse_args()
    c = yaml.safe_load(open(os.path.join(ROOT, a.config))); out = os.path.join(ROOT, c['verdict']['baseline'])
    res = json.load(open(out)) if os.path.exists(out) else {}
    for task in a.tasks.split(','):
        subs = load_task(c, task)
        for s in subs: s['x'] = de(s['x'])
        per = {}
        for sub, (xtr, ytr, _), (xte, yte) in folds(subs, c['tasks'][task]['protocol']):
            sc = StandardScaler().fit(xtr); m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(xtr), ytr)
            per[sub] = dict(scores(yte, m.predict(sc.transform(xte))), n_train=int(len(ytr)), n_test=int(len(yte)))
            print(json.dumps({task: {sub: per[sub]}}), flush=True)
        res[task] = dict(subjects=per, mean={k: float(np.mean([v[k] for v in per.values()])) for k in ('acc', 'bal_acc', 'wf1', 'kappa')})
        os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
