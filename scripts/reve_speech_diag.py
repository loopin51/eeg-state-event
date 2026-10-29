"""D-125 post-hoc diagnosis (report only, written after the verdict): why the REVE label-shuffle arm sits above chance.

usage: python scripts/reve_speech_diag.py configs/reve_speech.yaml

The protocol stops on the held-out subject's score and reports the best epoch, so even a model that learned nothing gets the
maximum of several noisy test scores. For every arm x task this reads history.json / preds.npz and writes
results/reve_speech/diag.json:
  - mean per-subject score at the selected (best) epoch, at the first epoch, at the last epoch run, and averaged over epochs;
  - mean number of epochs run;
  - majority-class share of the test labels (accuracy of a constant prediction);
  - a null simulation: per subject, draw uniform random predictions for as many epochs as the run had and keep the best
    (same metric, same test labels), verdict.bootstrap repeats -> mean of the selected score with no learning at all.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, glob, argparse
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ['eegnet', 'labram', 'reve', 'reve_shuffle', 'reve_lr1e4']


def score(y, p, key, k):
    if key == 'acc': return float((y == p).mean())
    return float(np.mean([(p[y == c] == c).mean() for c in range(k) if (y == c).any()]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    c = yaml.safe_load(open(os.path.join(ROOT, a.config))); rng = np.random.default_rng(c['seed']); out = {}
    for task, t in c['tasks'].items():
        key = 'bal_acc' if t['data'] == 'bci2020' else 'acc'; k = t['n_classes']
        for arm in ARMS:
            fs = glob.glob(os.path.join(ROOT, c['runs_dir'], f'*_rs_{arm}_{task}'))
            if not fs: continue
            assert len(fs) == 1, fs
            h = json.load(open(os.path.join(fs[0], 'history.json'))); P = np.load(os.path.join(fs[0], 'preds.npz'))
            rows = dict(best=[], first=[], last=[], mean_ep=[], epochs=[], majority=[], null_best=[])
            for sub, eps in h.items():
                v = np.array([e[key] for e in eps]); y = P[f'{sub}_true']
                rows['best'].append(v.max()); rows['first'].append(v[0]); rows['last'].append(v[-1]); rows['mean_ep'].append(v.mean())
                rows['epochs'].append(len(v)); rows['majority'].append(np.bincount(y, minlength=k).max() / len(y))
                sims = [max(score(y, rng.integers(0, k, len(y)), key, k) for _ in range(len(v))) for _ in range(c['verdict']['bootstrap'])]
                rows['null_best'].append(float(np.mean(sims)))
            out[f'{arm}/{task}'] = dict(chance=1.0 / k, metric=key, n=len(h), **{q: float(np.mean(r)) for q, r in rows.items()})
            print(f'{arm:13s} {task:10s} ' + ' '.join(f'{q}={out[f"{arm}/{task}"][q]:.4f}' for q in rows), flush=True)
    path = os.path.join(ROOT, 'results/reve_speech/diag.json'); json.dump(out, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()
