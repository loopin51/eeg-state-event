"""D-128 H1 report-only: SB shortcut runs on the reversed-shortcut test (outside the verdict family).

usage: env/bin/python scripts/nc_sb_reversed.py configs/nc_sb_reversed.yaml

The training script tests every SB run with shortcuts twice on the same clean test subjects (background + true signal of
class y; shortcuts are added to the training and inner-validation subjects only):
  normal    (probs)          the test windows as they are, without shortcuts
  reversed  (probs_reversed) the same windows plus every shortcut of the condition rendered for class (y + 1) mod K with
                             the training amplitude (generate.reversed_shortcuts); the true signal still says y
A model that follows the shortcuts predicts (y + 1) mod K on the reversed test. Per SB condition x arm x seed this
writes the pooled kappa on both tests, their difference (normal - reversed), the fraction of predictions equal to
(y + 1) mod K on both tests, per-fold values, and kappa on the clean test of the same arm and seed trained without
shortcuts ({sig}_clean) and of its label-shuffle run. Predictions are argmax over the K task columns (the null column
of VERA / EEGNet+N3 is ignored, as in the verdict). Subject bootstrap of the seed mean as in scripts/nc_verdict.py.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, glob, argparse
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from nc_grid import runs as grid_runs, folder
from nc_verdict import metrics as cm_metrics

KEYS = ('kappa', 'kappa_rev', 'kappa_diff', 'shift_frac', 'shift_frac_rev')


def conf(s, y, p, S, K):
    C = np.zeros((S, K, K)); np.add.at(C, (s, y, p), 1); return C


def load(rd, r, S):
    f = sorted(glob.glob(os.path.join(rd, '*' + folder(r), 'preds.npz')))
    if not f: return None
    z = np.load(f[-1]); m = json.load(open(os.path.join(os.path.dirname(f[-1]), 'metrics.json')))
    K = int(m['info']['n_classes']); y, s = z['y'].astype(int), z['subject'].astype(int)
    out = dict(run=os.path.basename(os.path.dirname(f[-1])), K=K, C=conf(s, y, z['probs'][:, :K].argmax(1), S, K))
    if 'probs_reversed' in z.files:
        p, pr = z['probs'][:, :K].argmax(1), z['probs_reversed'][:, :K].argmax(1); out['Cr'] = conf(s, y, pr, S, K)
        folds, i = {}, 0
        for fk, fm in m['folds'].items():            # preds are concatenated in fold order
            n = fm['n_test']; j = slice(i, i + n); i += n
            Cf, Crf = conf(s[j], y[j], p[j], S, K)[None].sum(1), conf(s[j], y[j], pr[j], S, K)[None].sum(1)
            kf, kr = float(cm_metrics(Cf)[0][0]), float(cm_metrics(Crf)[0][0])
            assert abs(kf - fm['test']['kappa']) < 1e-6 and abs(kr - fm['test_reversed']['kappa']) < 1e-6, (out['run'], fk)
            folds[fk] = dict(n_test=n, kappa=kf, kappa_rev=kr, kappa_diff=kf - kr,
                             shift_frac=float(np.mean(p[j] == (y[j] + 1) % K)), shift_frac_rev=float(np.mean(pr[j] == (y[j] + 1) % K)))
        assert i == len(y); out['folds'] = folds
    return out


def shift(C):
    K = C.shape[-1]; return C[..., np.arange(K), (np.arange(K) + 1) % K].sum(-1) / C.sum((-1, -2))


def boot_stats(run, W):
    A = np.einsum('bs,skl->bkl', W, run['C']); out = dict(kappa=cm_metrics(A)[0], shift_frac=shift(A))
    if 'Cr' in run:
        Ar = np.einsum('bs,skl->bkl', W, run['Cr']); out.update(kappa_rev=cm_metrics(Ar)[0], shift_frac_rev=shift(Ar))
        out['kappa_diff'] = out['kappa'] - out['kappa_rev']
    return out


def summarise(L, W, level, keys):
    per = [boot_stats(r, W) for r in L]; a = (1 - level) / 2; out = dict(seeds_runs=[r['run'] for r in L], n_seeds=len(L))
    for k in keys:
        mean = np.mean([p[k] for p in per], axis=0)
        out[k] = float(mean[0]); out[k + '_ci'] = [float(np.quantile(mean[1:], a)), float(np.quantile(mean[1:], 1 - a))]
    return out, per


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    v = yaml.safe_load(open(os.path.join(ROOT, a.config))); g = yaml.safe_load(open(os.path.join(ROOT, v['grid'])))
    rd = os.path.join(ROOT, v['runs_dir']); R = [r for r in grid_runs(g) if r['task'] == v['task'] and r['sb']]
    S = None
    for r in R:
        f = sorted(glob.glob(os.path.join(rd, '*' + folder(r), 'preds.npz')))
        if f: S = int(np.load(f[-1])['subject'].max()) + 1; break
    rng = np.random.default_rng([v['boot_seed'], sum(map(ord, v['task']))])
    W = np.vstack([np.ones(S), rng.multinomial(S, np.full(S, 1 / S), size=v['n_boot'])])
    res = dict(test=__doc__.split('\n\n')[2].strip(), n_boot=v['n_boot'], boot_seed=v['boot_seed'], level=v['level'],
               n_subjects=S, conditions={}, missing=[])
    for sig in v['signals']:
        for sc in v['shortcuts']:
            cond = f'{sig}_{sc}'; C = {}
            for arm in sorted({r['arm'] for r in R if r['sb'] == cond}, key=[r['arm'] for r in R].index):
                rr = sorted([r for r in R if r['sb'] == cond and r['arm'] == arm and not r['shuffle']], key=lambda r: r['seed'])
                L = [load(rd, r, S) for r in rr]; res['missing'] += [folder(r)[1:] for r, l in zip(rr, L) if l is None]
                L = [l for l in L if l is not None]
                assert all('Cr' in l for l in L), (cond, arm)
                out, per = summarise(L, W, v['level'], KEYS)
                out['per_seed'] = {}
                for r, l, p in zip(rr, L, per):
                    cl = load(rd, dict(r, sb=f'{sig}_clean'), S)
                    out['per_seed'][l['run']] = dict(seed=r['seed'], **{k: float(p[k][0]) for k in KEYS}, folds=l['folds'],
                                                     kappa_clean=float(boot_stats(cl, W[:1])['kappa'][0]) if cl else None,
                                                     clean_run=cl['run'] if cl else None)
                cl = [load(rd, dict(r, sb=f'{sig}_clean'), S) for r in rr]; cl = [c for c in cl if c is not None]
                if len(cl) == len(rr):                 # clean-trained runs of the same seeds, same draw
                    o2, _ = summarise(cl, W, v['level'], ('kappa',)); out['kappa_clean'] = o2['kappa']; out['kappa_clean_ci'] = o2['kappa_ci']
                sh = load(rd, dict(task=v['task'], sb=f'{sig}_clean', arm=arm, shuffle=True, seed=0), S)
                if sh:
                    o3, _ = summarise([sh], W, v['level'], ('kappa',)); out['kappa_clean_shuffle'] = o3['kappa']
                    out['kappa_clean_shuffle_ci'] = o3['kappa_ci']; out['clean_shuffle_run'] = sh['run']
                C[arm] = out; print(cond, arm, len(L), flush=True)
            res['conditions'][cond] = C
    out = os.path.join(ROOT, v['out']); json.dump(res, open(out, 'w'), indent=1); print(out, 'missing', len(res['missing']))


if __name__ == '__main__':
    main()
