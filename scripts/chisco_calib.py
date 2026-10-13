"""Chisco day-held-out per-user calibration curves and the D-047 verdict (39 sentence categories).

Reads data/cache/chisco_feats/sub-XX.npz (scripts/prepare_chisco_feats.py). Per subject, each recording day is held out
once; calibration trials come from all other days. Readout = shrinkage LDA with uniform priors over the classes present
in the calibration set; metric = balanced accuracy over the classes present on the test day.
k = calibration trials per class (k values above the available count use all available trials of that class).
Chance threshold: calibration labels permuted (k = all), subject mean, 95th percentile.
All feature sets use the common trial set raw_ok (raw EDF window available, D-047 addendum) so that every comparison is paired;
DE_IMAG_ALL is DE_IMAG on every derivative trial (secondary, not used in the verdict).

usage: python scripts/chisco_calib.py OUT_JSON [--perms 100] [--no-center] [--sets DE_IMAG EMG ...]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.eval.readout import bootstrap_ci
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

FEATS = os.path.join(ROOT, 'data/cache/chisco_feats')
SETS = ['DE_IMAG', 'DE_IMAG_ALL', 'DE_IMAG_RAW', 'DE_READ', 'EMG', 'EOG', 'PRE', 'DC', 'ORDER', 'LEN']
METERS = ['ORDER', 'DC', 'PRE']; KS = [5, 10, 20, 40, 'all']; N_DRAWS = 5; SUBJECTS = [1, 2, 3, 4, 5]


def center(X, groups):
    X = X.astype(np.float64).copy()
    for g in np.unique(groups): X[groups == g] -= X[groups == g].mean(0)
    return X


def fit_score(Xtr, ytr, Xte, yte):
    cls = np.unique(ytr)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=np.ones(len(cls)) / len(cls)).fit(Xtr, ytr)
    return balanced_accuracy_score(yte, clf.predict(Xte))


def draw(pool, y, k, rng):
    if k == 'all': return pool
    return np.concatenate([rng.choice(pool[y[pool] == q], min(k, (y[pool] == q).sum()), replace=False) for q in np.unique(y[pool])])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out_json'); ap.add_argument('--perms', type=int, default=100)
    ap.add_argument('--no-center', action='store_true'); ap.add_argument('--sets', nargs='+', default=SETS); a = ap.parse_args()
    t0 = time.time(); res = dict(ks=[str(k) for k in KS], perms=a.perms, centered=not a.no_center, sets=a.sets, subjects={})
    null = {f: np.zeros((len(SUBJECTS), a.perms)) for f in a.sets}
    for si, sub in enumerate(SUBJECTS):
        Z = np.load(os.path.join(FEATS, f'sub-{sub:02d}.npz')); r = {}
        for f in a.sets:
            m = np.ones(len(Z['label']), bool) if f == 'DE_IMAG_ALL' else Z['raw_ok']
            y, day = Z['label'][m], Z['day'][m]; days = np.unique(day); Xf = Z['DE_IMAG' if f == 'DE_IMAG_ALL' else f][m]
            assert np.isfinite(Xf).all(), f'non-finite values in {f}'
            X = Xf.astype(np.float64) if a.no_center else center(Xf, day); curve = {}
            for k in KS:
                accs = []
                for td in days:
                    pool, te = np.where(day != td)[0], np.where(day == td)[0]
                    for d in range(1 if k == 'all' else N_DRAWS):
                        cal = draw(pool, y, k, np.random.default_rng(1000 * sub + 10 * int(td) + d))
                        accs.append(fit_score(X[cal], y[cal], X[te], y[te]))
                curve[str(k)] = float(np.mean(accs))
            rng = np.random.default_rng(99 + sub)
            for p in range(a.perms):
                accs = []
                for td in days:
                    pool, te = np.where(day != td)[0], np.where(day == td)[0]
                    accs.append(fit_score(X[pool], rng.permutation(y[pool]), X[te], y[te]))
                null[f][si, p] = np.mean(accs)
            r[f] = dict(curve=curve, null_p95=float(np.quantile(null[f][si], 0.95)), n_trials=int(m.sum()), n_per_day=np.bincount(day).tolist())
            print(f'sub-{sub:02d} {f:11s} ' + ' '.join(f'k={k}: {v:.4f}' for k, v in curve.items()) +
                  f' | own null p95 {r[f]["null_p95"]:.4f} ({time.time() - t0:.0f}s)', flush=True)
        res['subjects'][sub] = r
        json.dump(res, open(a.out_json, 'w'), indent=1)
    summary = {}
    for f in a.sets:
        per = {k: np.array([res['subjects'][s][f]['curve'][k] for s in SUBJECTS]) for k in res['ks']}
        summary[f] = dict(curve={k: dict(mean=float(v.mean()), ci95=bootstrap_ci(v), per_subject=v.round(4).tolist()) for k, v in per.items()},
                          chance_p95=float(np.quantile(null[f].mean(0), 0.95)), null_mean=float(null[f].mean()))
    res['summary'] = summary
    if 'DE_IMAG' in a.sets and all(m in a.sets for m in METERS + ['EMG', 'EOG']):
        prim = summary['DE_IMAG']['curve']['all']; thr = summary['DE_IMAG']['chance_p95']
        meters = {m: dict(mean=summary[m]['curve']['all']['mean'], chance_p95=summary[m]['chance_p95']) for m in METERS}
        pde = np.array(prim['per_subject'])
        diffs = {c: dict(mean=float((pde - np.array(summary[c]['curve']['all']['per_subject'])).mean()),
                         ci95=bootstrap_ci(pde - np.array(summary[c]['curve']['all']['per_subject']))) for c in ['EMG', 'EOG']}
        if prim['ci95'][0] <= thr: result = 'not detected'
        elif any(v['mean'] > v['chance_p95'] for v in meters.values()): result = 'withheld (leak meter above chance)'
        elif any(v['ci95'][0] <= 0 for v in diffs.values()): result = 'detected but not separable from muscle/ocular activity'
        else: result = 'detected'
        res['verdict'] = dict(primary=prim, chance_p95=thr, meters=meters, de_minus_control=diffs, result=result)
    json.dump(res, open(a.out_json, 'w'), indent=1)
    for f, v in summary.items():
        print(f'{f:11s} ' + ' '.join(f'k={k}: {v["curve"][k]["mean"]:.4f}' for k in v['curve']) +
              f' | ci(all) {np.round(v["curve"]["all"]["ci95"], 4)} | per-subject {v["curve"]["all"]["per_subject"]} | chance p95 {v["chance_p95"]:.4f}', flush=True)
    if 'verdict' in res: print('VERDICT', res['verdict'], flush=True)


if __name__ == '__main__':
    main()
