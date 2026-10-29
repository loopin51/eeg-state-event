"""D-128 DE+LR baseline (CLAUDE.md: before any new-model number) on the splits of scripts/nc_train.py.

usage: env/bin/python scripts/nc_baseline_de.py configs/nc_train.yaml --task <task> [--sb <sb name>] [--shuffle-seeds 5]

Features: differential entropy 0.5 log(2 pi e var) of every channel in the N1 bands (FACED: 5 bands) of the task window;
standardised on the training windows; multinomial logistic regression (L2, C from configs `baseline.C`). Training uses the
training and inner-validation subjects together (no selection is made). SB: shortcuts are added to the training and
validation subjects of each fold exactly as in nc_train.py. Label-shuffle control: labels permuted within each training
subject, shuffle_seeds times. Writes results/baselines/nc_<task>[_<sb>]_de.json with per-subject scores and pooled scores.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse
import numpy as np, yaml, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.reve_nc import band_split
from nc_train import TaskData, splits, scores


def de(X, bands, chunk=500):
    out = []
    for i in range(0, len(X), chunk):
        xb = band_split(torch.from_numpy(X[i:i + chunk]).double(), 200, bands); v = xb.var(-1)
        out.append((0.5 * torch.log(2 * np.pi * np.e * v + 1e-12)).reshape(len(xb), -1).float().numpy())
    return np.concatenate(out)


def fit_predict(Ftr, ytr, Fte, C):
    from sklearn.linear_model import LogisticRegression
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    m = LogisticRegression(C=C, max_iter=2000).fit((Ftr - mu) / sd, ytr); return m.predict((Fte - mu) / sd)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--task', required=True); ap.add_argument('--sb')
    ap.add_argument('--shuffle-seeds', type=int, default=5); a = ap.parse_args()
    c = yaml.safe_load(open(os.path.join(ROOT, a.config))); C = c['baseline']['C']; t0 = time.time()
    c['tasks'][a.task]['null'] = False                      # task windows only
    sb = c['sb'][a.sb] if a.sb else None; d = TaskData(c, a.task, torch.device('cpu'), sb=sb)
    F0 = de(d.x, d.bands); res = dict(task=a.task, sb=a.sb, C=C, bands=d.bands, n_windows=int(len(d.y)), folds={}, shuffle={})
    P = {'real': np.zeros(len(d.y), int), **{f'shuf{s}': np.zeros(len(d.y), int) for s in range(a.shuffle_seeds)}}; tested = np.zeros(len(d.y), bool)
    for f, tr, va, te in splits(d, c, a):
        tv = np.concatenate([tr, va]); F = F0
        if getattr(d, 'sc_x', None) is not None:              # SB: shortcuts in the training/validation subjects only
            Xs = d.x[tv] + d.sc_x[tv]; F = F0.copy(); F[tv] = de(Xs, d.bands); del Xs
        P['real'][te] = fit_predict(F[tv], d.y[tv], F[te], C); tested[te] = True
        for s in range(a.shuffle_seeds):
            rng = np.random.default_rng([c['seed'], 77, f, s]); ys = d.y.copy()
            for g in np.unique(d.subj[tv]): j = tv[d.subj[tv] == g]; ys[j] = ys[rng.permutation(j)]
            P[f'shuf{s}'][te] = fit_predict(F[tv], ys[tv], F[te], C)
        res['folds'][f] = scores(d.y[te], P['real'][te]); print(json.dumps({f'fold{f}': res['folds'][f]}), flush=True)
    for k, p in P.items():
        per = {d.subjects[s]: scores(d.y[(d.subj == s) & tested], p[(d.subj == s) & tested]) for s in np.unique(d.subj[tested])}
        row = dict(pooled=scores(d.y[tested], p[tested]), per_subject=per)
        if k == 'real': res['real'] = row
        else: res['shuffle'][k] = row
    res['seconds'] = round(time.time() - t0, 1)
    out = os.path.join(ROOT, 'results', 'baselines', f"nc_{a.task}{'_' + a.sb if a.sb else ''}_de.json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1); print(json.dumps(dict(written=out, real=res['real']['pooled'], seconds=res['seconds'])), flush=True)


if __name__ == '__main__':
    main()
