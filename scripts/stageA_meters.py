"""Stage A linear references on the stage A split (CPU): DE + ridge baseline and the ORDER leak meter.

Both predict the time-pooled (window mean) speech feature of each window and rank every test segment of a subject
against all test segments by cosine, exactly like the decoder evaluation (top-1/5/10, mean rank percentile).
  DE: log band power (delta, theta, alpha, beta, gamma-low) of each channel in the window (after the same clamp,
      reference and baseline as the decoder), ridge per subject, penalty by 5-fold CV over training blocks.
  ORDER: no EEG; features = run one-hot + polynomial / Fourier terms of the relative position inside the run, ridge.
         Above chance means the split lets position alone identify segments.

usage: python scripts/stageA_meters.py configs/stageA_train.yaml --feature w2v|mel
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, argparse, yaml, numpy as np
from scipy.signal import welch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageA_train import Data

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]; ALPHAS = np.logspace(-1, 6, 15)


def ridge(X, Y, a):
    U, s, Vt = np.linalg.svd(X, full_matrices=False); return Vt.T @ ((s / (s ** 2 + a))[:, None] * (U.T @ Y))


def unit(A): return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def rank_eval(P, Yte, run):
    S = unit(P) @ unit(Yte).T; true = np.diag(S); above = S > true[:, None]; rk = above.sum(1); n = len(S)
    same = run[:, None] == run[None]; rr = (above & same).sum(1); nr = same.sum(1)       # within-run candidates (D-077)
    return {'top1': float((rk < 1).mean()), 'top5': float((rk < 5).mean()), 'top10': float((rk < 10).mean()), 'rank_pct': float((1 - rk / (n - 1)).mean()),
            'run_top1': float((rr < 1).mean()), 'run_top10': float((rr < 10).mean()), 'run_rank_pct': float((1 - rr / (nr - 1)).mean()),
            'run_chance_top10': float(np.mean(np.minimum(10, nr) / nr))}


def fit_predict(Xtr, Ytr, Xte, groups):
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8; ym = Ytr.mean(0); Xtr = (Xtr - xm) / xs; Xte = (Xte - xm) / xs; sc = np.zeros(len(ALPHAS))
    for g in np.unique(groups):
        tr, va = groups != g, groups == g
        for j, a in enumerate(ALPHAS):
            P = Xtr[va] @ ridge(Xtr[tr], Ytr[tr] - ym, a); sc[j] += np.mean(np.sum(unit(P) * unit(Ytr[va] - ym), 1))
    a = ALPHAS[int(np.argmax(sc))]; return Xte @ ridge(Xtr, Ytr - ym, a), float(a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--feature', default='w2v'); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); D = Data(c, a.feature); fs = D.fs
    pooled = {sp: np.stack([D.feat(r, t).mean(1) for r, t in D.items[sp]]) for sp in ('train', 'test')}
    grp = np.array([hash((r, int(t // 30))) % 5 for r, t in D.items['train']])         # 30-s chunks of training time as CV groups
    pos = {}
    for sp in ('train', 'test'):
        rel = np.array([t / (D.lens[r] / fs) for r, t in D.items[sp]]); run = np.array([r for r, _ in D.items[sp]])
        pos[sp] = np.c_[np.eye(len(D.lens))[run], rel, rel ** 2, rel ** 3, *[f(2 * np.pi * k * rel) for k in (1, 2, 4, 8) for f in (np.sin, np.cos)]]
    run_te = np.array([r for r, _ in D.items['test']])
    P, al = fit_predict(pos['train'], pooled['train'], pos['test'], grp); order = rank_eval(P, pooled['test'], run_te)
    res = {'feature': a.feature, 'n_test_segments': len(D.items['test']), 'chance_top10': 10 / len(D.items['test']),
           'ORDER': {**order, 'alpha': al}, 'DE': {}}
    print('ORDER', order, flush=True)

    def de(s, sp):
        out = []
        for r, t in D.items[sp]:
            x, _ = D.brain(s, r, t); f, p = welch(x, fs=fs, nperseg=min(fs, x.shape[1]), axis=1)
            out.append(np.log(np.stack([p[:, (f >= lo) & (f < hi)].mean(1) for lo, hi in BANDS], 1) + 1e-8).ravel())
        return np.array(out)
    for s in D.subjects:
        Xtr, Xte = de(s, 'train'), de(s, 'test'); P, al = fit_predict(Xtr, pooled['train'], Xte, grp)
        res['DE'][s] = {**rank_eval(P, pooled['test'], run_te), 'alpha': al}; print(f'DE sub-{s:03d}', res['DE'][s], flush=True)
    res['DE_mean'] = {k: float(np.mean([v[k] for v in res['DE'].values()])) for k in ('top1', 'top5', 'top10', 'rank_pct', 'run_top1', 'run_top10', 'run_rank_pct', 'run_chance_top10')}
    out = os.path.join(ROOT, 'results/baselines'); os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f"stageA_de_order_{a.feature}{c.get('meters_suffix', '')}.json"), 'w'), indent=1); print('DE mean', res['DE_mean'], flush=True)


if __name__ == '__main__':
    main()
