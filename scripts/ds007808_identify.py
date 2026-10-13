"""ds007808 listening: day-held-out identification of the heard segment, step 1 (D-057).

Per subject, each recording day is the test set once. Ridge regression maps features to a target representation; the
penalty is picked by leave-one-day-out CV inside the calibration days (mean cosine between prediction and target).
Identification: for every test segment the candidates are the segment itself plus its 19 nearest segments in time in the
same run (chance 0.05); segments in runs with fewer than 20 segments are skipped. Secondary: 99 random same-day
distractors (chance 0.01). Chance threshold: calibration pairs shuffled 100 times (penalty fixed), 95th percentile of the
pooled top-1. Intervals: bootstrap over test days (1,000). Features are centred and scaled per day and column (label-free).

usage: python scripts/ds007808_identify.py OUT_JSON [--perms 100] [--subjects 1 2]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import json, time, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CACHE = os.environ.get('ESE_DS7808_CACHE', os.path.join(ROOT, 'data/cache/ds007808_listening'))   # override only for synthetic tests
ALPHAS = np.logspace(0, 7, 15); N_LOCAL = 20; N_RAND = 100
PAIRS = [('LF', 'TXT'), ('LF', 'AUD'), ('DE', 'TXT'), ('PRE', 'TXT'), ('EXG', 'TXT'), ('ORDER', 'TXT'), ('LOWONLY', 'TXT')]


def day_standardise(X, day):
    X = X.reshape(len(X), -1).astype(np.float64).copy()
    for g in np.unique(day):
        m = day == g; X[m] -= X[m].mean(0); X[m] /= X[m].std(0) + 1e-8
    return X


def ridge_fit(U, s, Vt, Y, alpha):
    return Vt.T @ ((s / (s ** 2 + alpha))[:, None] * (U.T @ Y))


def svd(X):
    U, s, Vt = np.linalg.svd(X, full_matrices=False); return U, s, Vt


def cos_rows(P, Y):
    return (P * Y).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12)


def pick_alpha(X, Y, day):
    score = np.zeros(len(ALPHAS))
    for g in np.unique(day):
        tr, va = day != g, day == g; mu = Y[tr].mean(0); U, s, Vt = svd(X[tr] - X[tr].mean(0))
        for j, a in enumerate(ALPHAS):
            P = (X[va] - X[tr].mean(0)) @ ridge_fit(U, s, Vt, Y[tr] - mu, a); score[j] += cos_rows(P, Y[va] - mu).mean()
    return float(ALPHAS[np.argmax(score)])


def candidates(run, onset, idx, rng):
    """Candidate matrices over global segment indices, column 0 = the segment itself.
    local [m, 20]: the 19 nearest in time within the same run (runs with < 20 test segments are skipped);
    rand  [n, 100]: 99 random same-day segments."""
    loc, rnd = [], []
    for i in idx:
        same = idx[(run[idx] == run[i]) & (idx != i)]
        if len(same) >= N_LOCAL - 1: loc.append(np.r_[i, same[np.argsort(np.abs(onset[same] - onset[i]))[:N_LOCAL - 1]]])
        others = idx[idx != i]; rnd.append(np.r_[i, rng.choice(others, min(N_RAND - 1, len(others)), replace=False)])
    return np.array(loc), np.array(rnd)


def hits(P, Yn, C, pos):
    """P: predictions for the test rows; Yn: L2-normalised targets (global); C: candidate matrix (column 0 = self).
    Returns top-1 hits and rank percentiles per candidate row."""
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    sim = np.einsum('md,mkd->mk', Pn[pos[C[:, 0]]], Yn[C])
    return (sim.argmax(1) == 0).astype(float), (sim[:, 1:] > sim[:, :1]).mean(1)


def boot_days(per_day, n=1000, seed=0):
    rng = np.random.default_rng(seed); keys = list(per_day); vals = []
    for _ in range(n):
        pick = rng.choice(keys, len(keys)); v = np.concatenate([per_day[k] for k in pick]); vals.append(v.mean())
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out_json'); ap.add_argument('--perms', type=int, default=100)
    ap.add_argument('--subjects', type=int, nargs='+', default=[1, 2]); a = ap.parse_args(); t0 = time.time(); res = dict(perms=a.perms, subjects={})
    for sub in a.subjects:
        Z = np.load(os.path.join(CACHE, f'sub-{sub:02d}.npz'), allow_pickle=True); day, run, onset = Z['day'], Z['run'], Z['onset']
        T = {'TXT': np.load(os.path.join(CACHE, f'sub-{sub:02d}_txt.npy')).astype(np.float64), 'AUD': Z['AUD'].astype(np.float64)}
        F = {k: day_standardise(Z[k], day) for k in ['LF', 'DE', 'PRE', 'EXG']}
        F['ORDER'] = np.c_[onset, onset ** 2] / np.array([3600.0, 3600.0 ** 2]); F['LOWONLY'] = np.c_[Z['duration'], Z['n_chars']].astype(np.float64)
        rng = np.random.default_rng(sub); R = {}; hit_store = {}
        for fname, tname in PAIRS:
            X, Y = F[fname], T[tname]; per_day, per_day_rand, rk_all, alphas = {}, {}, [], {}; null = np.zeros(a.perms); n_null = np.zeros(a.perms)
            for g in np.unique(day):
                tr, te = np.where(day != g)[0], np.where(day == g)[0]
                Xtr = X[tr]; xm = Xtr.mean(0); sd = Xtr.std(0) + 1e-8 if fname in ('ORDER', 'LOWONLY') else 1.0
                Xtr = (Xtr - xm) / sd; Xte = (X[te] - xm) / sd; ym = Y[tr].mean(0); ys = Y[tr].std(0) + 1e-8 if tname == 'AUD' else 1.0
                Ytr = (Y[tr] - ym) / ys; Yall = (Y - ym) / ys
                alpha = pick_alpha(Xtr, Ytr, day[tr]); alphas[int(g)] = alpha; U, s, Vt = svd(Xtr)
                P = Xte @ ridge_fit(U, s, Vt, Ytr, alpha); pos = np.full(len(day), -1); pos[te] = np.arange(len(te))
                Yn = Yall / (np.linalg.norm(Yall, axis=1, keepdims=True) + 1e-12)
                loc, rnd = candidates(run, onset, te, np.random.default_rng(1000 * sub + int(g)))
                h, rk = hits(P, Yn, loc, pos); hr, _ = hits(P, Yn, rnd, pos)
                per_day[int(g)] = h; per_day_rand[int(g)] = hr; rk_all += list(rk); hit_store[(fname, tname, int(g))] = h
                for q in range(a.perms):
                    hq, _ = hits(Xte @ ridge_fit(U, s, Vt, Ytr[rng.permutation(len(tr))], alpha), Yn, loc, pos)
                    null[q] += hq.sum(); n_null[q] += len(hq)
            pooled = np.concatenate(list(per_day.values()))
            R[f'{fname}->{tname}'] = dict(top1=float(pooled.mean()), ci95=boot_days(per_day), n=int(len(pooled)),
                                          per_day={k: float(v.mean()) for k, v in per_day.items()}, rank_pct=float(np.mean(rk_all)),
                                          rand100_top1=float(np.concatenate(list(per_day_rand.values())).mean()),
                                          chance_p95=float(np.quantile(null / np.maximum(n_null, 1), 0.95)), alphas=alphas)
            r = R[f'{fname}->{tname}']
            print(f'sub-{sub:02d} {fname:7s}->{tname}: top1 {r["top1"]:.4f} {np.round(r["ci95"], 4)} (n={r["n"]}) rank {r["rank_pct"]:.3f} '
                  f'rand100 {r["rand100_top1"]:.4f} | null p95 {r["chance_p95"]:.4f} ({time.time() - t0:.0f}s)', flush=True)
        # paired LF - EXG on the same segments and candidates
        dd = {}
        for g in np.unique(day):
            dd[int(g)] = hit_store[('LF', 'TXT', int(g))] - hit_store[('EXG', 'TXT', int(g))]   # same candidate rows, same order
        R['LF-EXG'] = dict(mean=float(np.concatenate(list(dd.values())).mean()), ci95=boot_days(dd))
        p = R['LF->TXT']
        if p['ci95'][0] <= p['chance_p95']: verdict = 'not detected'
        elif any(R[m]['top1'] > R[m]['chance_p95'] for m in ('PRE->TXT', 'ORDER->TXT')): verdict = 'withheld (leak meter above chance)'
        elif R['LF-EXG']['ci95'][0] <= 0: verdict = 'detected but not separable from ocular / lip-muscle activity'
        else: verdict = 'detected'
        R['verdict'] = verdict; res['subjects'][sub] = R; json.dump(res, open(a.out_json, 'w'), indent=1)
        print(f'sub-{sub:02d} LF-EXG {R["LF-EXG"]["mean"]:+.4f} {np.round(R["LF-EXG"]["ci95"], 4)} | VERDICT {verdict}', flush=True)


if __name__ == '__main__':
    main()
