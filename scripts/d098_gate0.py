"""D-098 gate 0 (stimulus only, no EEG): can the new L1 target be identified among the new pairs by acoustic or position meters?

Target (per run fold f): 20-Hz series of 5 channels - Zipf, GPT-2 surprisal, entropy at word onsets and phoneme surprisal,
cohort entropy at phone onsets - standardised with the fold's training-run words / phones, Gaussian-smoothed (sigma 0.05 s),
then linearly residualised (ridge, 7-group CV over the 14 training runs) on the lagged acoustic design (20 regressors of
d098_features.py, lags -0.25..+0.25 s). Window target = residual series in [t - 0.5, t + 2.5) (5 x 60 = 300-d).
Window acoustic features (380-d): B2 primary pairing features (mel 120, env 30, log2 F0 30, voiced fraction 30) + env edge
(30 bins of 0.1 s) + band edges (8 x 10 bins of 0.3 s) + word onset counts (30) + phone onset counts (30).
New pairs: per test query, the 19 same-run queries >= 30 s away nearest in cosine distance of the RFF-ridge (seed 20261030)
prediction of the window target. Meters (fit on the fold's training-run grid windows with words): ACOUSTIC (linear ridge),
ACOUSTIC_NL (RFF ridge, 5 seeds, cell value = seed mean; per-seed values in gate0_nl_seeds.json, report only), ACOUSTIC_KNN (50 nearest training windows), ORDER (position terms).
Score = cosine(prediction_i, target_j); pair MRP = 1 - #(pairs scoring above the true) / 19. Null: B2 amendment-3 circular
shift (row fixed, truth and pairs -> shift partner j and pairs(j)), k_min = median onsets per 30 s, same k_r for the run,
1,000 draws. Cells = test runs (20); value = query-weighted mean of (MRP - own null mean); 1,000 run bootstrap, 90% upper < 0.02.

usage: python scripts/d098_gate0.py configs/d098_gates.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ds004408_p0 import lagged, ridge, corr
from stageB_b1_meters import unit, ALPHAS


def fit_predict(Xtr, Ytr, Xte, groups):
    """same computation as stageB_b1_meters.fit_predict (standardise, grouped-CV cosine over ALPHAS, ridge via SVD), with one
    SVD per CV group instead of one per (group, penalty)"""
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8; ym = Ytr.mean(0); Xtr = (Xtr - xm) / xs; Xte = (Xte - xm) / xs; sc = np.zeros(len(ALPHAS))
    for g in np.unique(groups):
        tr, va = groups != g, groups == g; U, s, Vt = np.linalg.svd(Xtr[tr], full_matrices=False); UtY = U.T @ (Ytr[tr] - ym); Xv = Xtr[va] @ Vt.T
        for j, a in enumerate(ALPHAS):
            P = Xv @ ((s / (s ** 2 + a))[:, None] * UtY); sc[j] += np.mean(np.sum(unit(P) * unit(Ytr[va] - ym), 1))
    a = ALPHAS[int(np.argmax(sc))]; U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    return Xte @ (Vt.T @ ((s / (s ** 2 + a))[:, None] * (U.T @ (Ytr - ym)))), float(a)
LAMS = 10.0 ** np.arange(-3, 5)


def rff(Xtr_std, X_std, n, seed):
    rng = np.random.default_rng(seed); d = X_std.shape[1]; Wm = rng.normal(0, 1 / np.sqrt(d), (d, n)); b = rng.uniform(0, 2 * np.pi, n)
    return np.sqrt(2 / n) * np.cos(Xtr_std @ Wm + b), np.sqrt(2 / n) * np.cos(X_std @ Wm + b)


def shift_maps(t_by_run, n_shift, min_shift, rng):
    """B1 amendment-1 segment-order circular shift: per run one k per draw in [k_min, n - k_min]"""
    maps = {}
    for r, on in t_by_run.items():
        order = np.argsort(on); n = len(on); so = on[order]
        kmin = int(np.ceil(np.median(np.searchsorted(so, so + min_shift) - np.arange(n))))
        pos = np.empty(n, int); pos[order] = np.arange(n); ks = rng.integers(kmin, n - kmin + 1, n_shift)
        maps[r] = order[(pos[None] + ks[:, None]) % n]
    return maps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); t0 = time.time()
    c = yaml.safe_load(open(a.config)); g = c['gate_0']; fs = c['fs']; b1 = yaml.safe_load(open(os.path.join(ROOT, c['b1_config']))); sp = b1['split']
    I = np.load(os.path.join(ROOT, 'data/cache/stageB_b2/items.npz'), allow_pickle=True); Pz = np.load(os.path.join(ROOT, 'data/cache/stageB_b2/pairs.npz'))
    Fz = np.load(os.path.join(ROOT, c['features_cache'], 'features.npz'), allow_pickle=True); W = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True)
    S = np.load(os.path.join(ROOT, 'data/cache/ds004408/stim.npz'), allow_pickle=True)
    reg = list(Fz['reg']); T = [len(x) for x in reg]; run, t, isq, has = I['run'], I['t'], I['is_query'], I['has_text']; n = len(run)
    s0 = np.round((t - 0.5) * fs).astype(int); L = int(round(3.0 * fs))
    ok_win = (s0 >= 0) & (s0 + L <= np.array(T)[run]); assert ok_win[isq].all(), 'query window outside the 20-Hz series'
    # window acoustic features (fold independent)
    names = list(Fz['reg_names']); ie = names.index('env_edge'); ib = [names.index(f'band_edge{b}') for b in range(8)]
    iw, ip = names.index('word_onset'), names.index('phone_onset')
    def binned(x, nb):                                                                  # x [L, k] -> [nb * k] bin means
        return x.reshape(nb, L // nb, -1).mean(1).T.ravel()
    EXTRA = np.zeros((n, 30 + 80 + 30 + 30), np.float32)
    for i in np.flatnonzero(ok_win):
        R = reg[run[i]][s0[i]:s0[i] + L]
        EXTRA[i] = np.r_[binned(R[:, [ie]], 30), binned(R[:, ib], 10), binned(R[:, [iw]], 30) * (L // 30), binned(R[:, [ip]], 30) * (L // 30)]
    XA = np.c_[I['mel'], I['env'], I['f0'], I['vf'], EXTRA].astype(np.float64); assert XA.shape[1] == 380
    rel = t / np.asarray(S['dur'], float)[run]
    POS = np.c_[rel, rel ** 2, rel ** 3, *[f(2 * np.pi * k * rel) for k in (1, 2, 4, 8) for f in (np.sin, np.cos)]]
    acoustic_cols = [names.index(x) for x in names]                                      # all 20 regressors of the acoustic design
    rl = int(round(g['resid_lags_s'][0] * fs)), int(round(g['resid_lags_s'][1] * fs)); rlags = list(range(rl[0], rl[1] + 1))
    ph_run, ph_on = Fz['ph_run'], Fz['ph_onset']; ph_val = {'ph_surprisal': Fz['ph_surprisal'], 'cohort_entropy': Fz['cohort_entropy']}
    rng_null = np.random.default_rng(g['shift_null']['seed']); meters = list(g['meters']); res = {'folds': {}, 'cells': {}}
    cell_obs = {m: {} for m in meters}; cell_null = {m: {} for m in meters}; cell_raw_null = {}; cell_n = {}
    b2_obs = {m: {} for m in meters}; b2_null = {m: {} for m in meters}
    nl_seeds = list(g['meters']['ACOUSTIC_NL']['seeds']); seed_obs = {pn: [{} for _ in nl_seeds] for pn in ('new', 'b2')}; seed_null = {pn: [{} for _ in nl_seeds] for pn in ('new', 'b2')}
    for f in range(5):
        test = [r - 1 for r in sp['run_folds'][f]]; valid = [r - 1 for r in sp['valid_runs'][f]]; train = [r for r in range(20) if r not in test + valid]
        # target series
        wtr = np.isin(W['run'], train); ptr = np.isin(ph_run, train); Y = []
        vals = {k: np.asarray(W[k], float) for k in ('zipf', 'surprisal', 'entropy')}
        for r in range(20):
            M = np.zeros((T[r], 5))
            for ch, k in enumerate(('zipf', 'surprisal', 'entropy')):
                v = (vals[k] - vals[k][wtr].mean()) / vals[k][wtr].std(); m = W['run'] == r; p = np.round(W['onset'][m] * fs).astype(int); kk = p < T[r]
                np.add.at(M[:, ch], p[kk], v[m][kk])
            for ch, k in ((3, 'ph_surprisal'), (4, 'cohort_entropy')):
                v = (ph_val[k] - ph_val[k][ptr].mean()) / ph_val[k][ptr].std(); m = ph_run == r; p = np.round(ph_on[m] * fs).astype(int); kk = p < T[r]
                np.add.at(M[:, ch], p[kk], v[m][kk])
            Y.append(gaussian_filter1d(M, g['smooth_sigma_s'] * fs, axis=0))
        X = [lagged(reg[r][:, acoustic_cols].astype(np.float64), rlags) for r in range(20)]
        groups = [train[i::7] for i in range(7)]; Cr = {r: X[r].T @ X[r] for r in train}; Br = {r: X[r].T @ Y[r] for r in train}; sc = np.zeros(len(LAMS))
        for gr in groups:
            tr = [r for r in train if r not in gr]; Ct = sum(Cr[r] for r in tr); Bt = sum(Br[r] for r in tr)
            for j, lam in enumerate(LAMS):
                Wr = ridge(Ct, Bt, lam); sc[j] += np.mean([corr(X[r] @ Wr, Y[r]).mean() for r in gr])
        lam = float(LAMS[int(np.argmax(sc))]); Wr = ridge(sum(Cr.values()), sum(Br.values()), lam); E = [Y[r] - X[r] @ Wr for r in range(20)]
        def explained(rs): return float(1 - sum(((E[r] - E[r].mean(0)) ** 2).sum() for r in rs) / sum(((Y[r] - Y[r].mean(0)) ** 2).sum() for r in rs))
        TGT = np.zeros((n, 300))
        for i in np.flatnonzero(ok_win): TGT[i] = E[run[i]][s0[i]:s0[i] + L].T.ravel()
        tr_i = np.flatnonzero(~isq & has & np.isin(run, train) & ok_win); grp = np.array([train.index(r) // 2 for r in run[tr_i]])
        pool = Pz[f'pool_test_f{f}']; assert isq[pool].all()
        xm, xs = XA[tr_i].mean(0), XA[tr_i].std(0) + 1e-8; Xs_tr, Xs_pool = (XA[tr_i] - xm) / xs, (XA[pool] - xm) / xs
        Ytr = TGT[tr_i]; Ypool = TGT[pool]
        # new pairs
        Ztr, Zp = rff(Xs_tr, Xs_pool, g['pair_model']['n_features'], g['pair_model']['seed']); Phat, al_pair = fit_predict(Ztr, Ytr, Zp, grp)
        D = 1 - unit(Phat) @ unit(Phat).T; rp, tp = run[pool], t[pool]
        okm = (rp[:, None] == rp[None]) & (np.abs(tp[:, None] - tp[None]) >= g['min_same_run_s']); np.fill_diagonal(okm, False)
        Dm = np.where(okm, D, np.inf); NP = np.argsort(Dm, 1)[:, :g['n_pairs']]; assert np.isfinite(np.take_along_axis(Dm, NP, 1)).all()
        loc = {int(x): k for k, x in enumerate(pool)}; B2P = np.vectorize(loc.get)(Pz[f'pairs_within_test_f{f}']); assert (B2P >= 0).all()
        # meters
        preds = {}
        preds['ACOUSTIC'] = [fit_predict(XA[tr_i], Ytr, XA[pool], grp)[0]]
        preds['ACOUSTIC_NL'] = []
        for sd in g['meters']['ACOUSTIC_NL']['seeds']:
            Ztr2, Zp2 = rff(Xs_tr, Xs_pool, g['meters']['ACOUSTIC_NL']['n_features'], sd); preds['ACOUSTIC_NL'].append(fit_predict(Ztr2, Ytr, Zp2, grp)[0])
        nn = NearestNeighbors(n_neighbors=g['meters']['ACOUSTIC_KNN']['k']).fit(Xs_tr); idx = nn.kneighbors(Xs_pool, return_distance=False)
        preds['ACOUSTIC_KNN'] = [Ytr[idx].mean(1)]
        preds['ORDER'] = [fit_predict(POS[tr_i], Ytr, POS[pool], grp)[0]]
        maps = shift_maps({r: tp[rp == r] for r in test}, g['shift_null']['n_shift'], g['shift_null']['min_shift_s'], rng_null)
        Yn = unit(Ypool); Yn = np.nan_to_num(Yn)
        for pair_name, PR, ob, nl in (('new', NP, cell_obs, cell_null), ('b2', B2P, b2_obs, b2_null)):
            for m in meters:
                for r in test:
                    ir = np.flatnonzero(rp == r); o_runs, n_runs = [], []
                    for Pm in preds[m]:
                        Sm = unit(Pm[ir]) @ Yn.T                                            # rows: queries of run r, columns: all pool
                        tru = Sm[np.arange(len(ir)), ir]; o_runs.append(np.mean(1 - (Sm[np.arange(len(ir))[:, None], PR[ir]] > tru[:, None]).sum(1) / g['n_pairs']))
                        J = ir[maps[r]]                                                     # [n_shift, n_r] shift partners (pool index)
                        tj = Sm[np.arange(len(ir))[None], J]; pj = Sm[np.arange(len(ir))[None, :, None], PR[J]]
                        n_runs.append(np.mean(1 - (pj > tj[..., None]).sum(-1) / g['n_pairs'], 1))   # [n_shift]
                    ob[m][r] = float(np.mean(o_runs)); nl[m][r] = np.mean(n_runs, 0); cell_n[r] = len(ir)
                    if m == 'ACOUSTIC_NL':
                        for k in range(len(nl_seeds)): seed_obs[pair_name][k][r] = float(o_runs[k]); seed_null[pair_name][k][r] = n_runs[k]
        res['folds'][str(f)] = dict(resid_lambda=lam, explained_train_series=explained(train), explained_test_series=explained(test), pair_alpha=al_pair,
                                    pair_cos_dist=float(np.take_along_axis(D, NP, 1).mean()), random_cos_dist=float(D[okm].mean()),
                                    target_cos_pairs=float(np.take_along_axis(Yn @ Yn.T, NP, 1).mean()), target_cos_random=float((Yn @ Yn.T)[okm].mean()),
                                    acoustic_sqdist_new_pairs=float(np.mean([((Xs_pool[i] - Xs_pool[NP[i]]) ** 2).mean() for i in range(len(pool))])),
                                    acoustic_sqdist_b2_pairs=float(np.mean([((Xs_pool[i] - Xs_pool[B2P[i]]) ** 2).mean() for i in range(len(pool))])),
                                    acoustic_sqdist_random=float(np.mean([((Xs_pool[i] - Xs_pool[np.flatnonzero(okm[i])]) ** 2).mean() for i in range(len(pool))])))
        print(f'fold {f}: resid lambda {lam:g}, explained train {explained(train):.4f} test {explained(test):.4f}; '
              + ' '.join(f'{m} {np.mean([cell_obs[m][r] - cell_null[m][r].mean() for r in test]):+.4f}' for m in meters) + f' ({time.time() - t0:.0f}s)', flush=True)
    runs = sorted(cell_n); w = np.array([cell_n[r] for r in runs], float); rng = np.random.default_rng(g['bootstrap']['seed'])
    BI = np.stack([rng.integers(0, len(runs), len(runs)) for _ in range(g['bootstrap']['n_boot'])])
    def summ(ob, nl):
        exc = np.array([ob[r] - nl[r].mean() for r in runs]); raw = np.array([ob[r] for r in runs])
        def wm(v, ix): return float((v[ix] * w[ix]).sum() / w[ix].sum())
        be = np.array([wm(exc, ix) for ix in BI]); br = np.array([wm(raw, ix) for ix in BI])
        return dict(excess=wm(exc, np.arange(len(runs))), excess_ci90=[float(np.percentile(be, 5)), float(np.percentile(be, 95))],
                    raw=wm(raw, np.arange(len(runs))), raw_ci90=[float(np.percentile(br, 5)), float(np.percentile(br, 95))],
                    null_mean=wm(np.array([nl[r].mean() for r in runs]), np.arange(len(runs))), per_run_excess=dict(zip([str(r + 1) for r in runs], exc.tolist())))
    res['meters_new_pairs'] = {m: summ(cell_obs[m], cell_null[m]) for m in meters}
    res['meters_b2_pairs_report'] = {m: summ(b2_obs[m], b2_null[m]) for m in meters}
    for m, v in res['meters_new_pairs'].items(): v['pass'] = bool(v['excess_ci90'][1] < g['margin']); v['raw_pass_052'] = bool(v['raw_ci90'][1] < 0.5 + g['margin'])
    res['verdict'] = 'pass' if all(v['pass'] for v in res['meters_new_pairs'].values()) else 'fail'
    res['verdict_raw_052_report'] = 'pass' if all(v['raw_pass_052'] for v in res['meters_new_pairs'].values()) else 'fail'
    res['n_queries'] = int(w.sum()); res['seconds'] = time.time() - t0
    # report only (D-098): ACOUSTIC_NL per RFF seed, same cells, null draws and bootstrap; written to a separate file
    seeds_res = {pn: {str(sd): summ(seed_obs[pn][k], seed_null[pn][k]) for k, sd in enumerate(nl_seeds)} for pn in ('new', 'b2')}
    for v in seeds_res['new'].values(): v['pass'] = bool(v['excess_ci90'][1] < g['margin'])
    o = os.path.join(ROOT, c['out']); os.makedirs(o, exist_ok=True); json.dump(res, open(os.path.join(o, 'gate0.json'), 'w'), indent=1)
    json.dump(dict(meter='ACOUSTIC_NL', seeds=nl_seeds, new_pairs=seeds_res['new'], b2_pairs_report=seeds_res['b2']), open(os.path.join(o, 'gate0_nl_seeds.json'), 'w'), indent=1)
    print('ACOUSTIC_NL per seed (new pairs): ' + ', '.join(f"{sd} {v['excess']:+.4f} [{v['excess_ci90'][0]:+.4f}, {v['excess_ci90'][1]:+.4f}]" for sd, v in seeds_res['new'].items()), flush=True)
    print(json.dumps(dict(verdict=res['verdict'], verdict_raw=res['verdict_raw_052_report'],
                          new={m: dict(excess=round(v['excess'], 4), ci90=[round(x, 4) for x in v['excess_ci90']], raw=round(v['raw'], 4)) for m, v in res['meters_new_pairs'].items()},
                          b2={m: dict(excess=round(v['excess'], 4), ci90=[round(x, 4) for x in v['excess_ci90']]) for m, v in res['meters_b2_pairs_report'].items()})), flush=True)


if __name__ == '__main__':
    main()
