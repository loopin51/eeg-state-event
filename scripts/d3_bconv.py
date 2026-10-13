"""D-083 b conversion, steps 2-3, and its precision check. Computed before p* exists; D-089 reads the curve at c*.

1. Residual covariance Sigma of the D-068 pooled EEG->TXT ridge (pool 17): per run fold, the residual of each subject's
   held-out prediction after removing its least-squares slope on the centred target; fold covariances weighted by test
   size, trace normalised to 1. Saved (with sha256) for the R error structure of D-087/D-089.
2. Oracle curve on the D-073 design: 3-s non-overlapping windows at 20 Hz (same lengths as trf_nested), text = words whose
   onset lies in the window (lower case, as D-061), e5 "query: " mean-pooled embedding, centred on the training-run mean
   of each D-069 run fold. Prediction = target + g * noise (Sigma scaled to the mean target energy, or isotropic);
   candidates = query + 19 nearest windows of the same run; rank_pct with ties 0.5 (trf_nested). x = mean cosine.
3. b(c*) = Phi(sqrt((d_B^2 + d_s^2) / 2)) - MRP_B, d = sqrt(2) Phi^-1(MRP), MRP_s interpolated in log cosine; b = 0 if
   MRP_s <= 0.5 (function b_from_cstar).
Precision check (reported only): forward envelope-only TRF MRP on the D-073 windows (pool 17, pooled ridge, run 5-fold)
vs the observed backward EEG->ENV MRP of the D-068 predictions on the D-061 local design.

usage: python scripts/d3_bconv.py configs/d3_bconv.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, hashlib, yaml, numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ds004408_p0 import lagged, ridge, corr, z
from ds004408_trf_nested import windows, chunk_mat, rank_pct, FOLDS, N_RUNS, CACHE
from ds004408_identify import embed, local_candidates
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def b_from_cstar(c_star, curve, mrp_B):
    """curve: list of {'cosine', 'mrp'}; returns (MRP_s, b). Raises if c* is outside the curve (D-089 extends the grid)."""
    if c_star is None: return None, None
    if np.isinf(c_star): return 1.0, float('inf')
    xs = np.log([r['cosine'] for r in curve]); ys = np.array([r['mrp'] for r in curve]); o = np.argsort(xs)
    if not xs.min() <= np.log(c_star) <= xs.max(): raise ValueError(f'c* {c_star} outside curve [{np.exp(xs.min())}, {np.exp(xs.max())}]')
    mrp_s = float(np.interp(np.log(c_star), xs[o], ys[o]))
    if mrp_s <= 0.5: return mrp_s, 0.0
    dB, ds = np.sqrt(2) * norm.ppf(mrp_B), np.sqrt(2) * norm.ppf(mrp_s)
    return mrp_s, float(norm.cdf(np.sqrt((dB ** 2 + ds ** 2) / 2)) - mrp_B)


def sigma_resid(P, run, pool_idx):
    Y = P['target_TXT'].astype(np.float64); S, w = 0, []
    for f in FOLDS:
        trc, tec = np.where(~np.isin(run, f))[0], np.where(np.isin(run, f))[0]; Yc = Y[tec] - Y[trc].mean(0)
        Pa = P['pred_TXT'][pool_idx][:, tec].astype(np.float64); b = (Pa * Yc).sum((1, 2)) / (Yc ** 2).sum()
        R = (Pa - b[:, None, None] * Yc).reshape(-1, Yc.shape[1]); R -= R.mean(0); Sig = R.T @ R / len(R)
        S = S + len(tec) * Sig / np.trace(Sig); w.append(len(tec))
    S = S / sum(w); return S / np.trace(S)


def main(cfg_path):
    c = yaml.safe_load(open(cfg_path)); tcfg = yaml.safe_load(open(os.path.join(ROOT, c['trf_config']))); tc = tcfg['trf']
    out = os.path.join(ROOT, c['out_dir']); os.makedirs(out, exist_ok=True); rng = np.random.default_rng(c['seed']); t0 = time.time()
    subjects = tcfg['subjects']; pool = [s for s in subjects if s not in tcfg['pool_exclude']]; fs = tc['fs']; L = int(tc['chunk_s'] * fs)
    P = np.load(os.path.join(ROOT, c['pooled_predictions'])); prun = P['run']; psub = list(P['subjects']); pidx = [psub.index(s) for s in pool]
    # 1. Sigma
    Sig = sigma_resid(P, prun, pidx); sp = os.path.join(ROOT, c['sigma_out']); os.makedirs(os.path.dirname(sp), exist_ok=True)
    np.savez(sp, sigma=Sig); sig_sha = hashlib.sha256(open(sp, 'rb').read()).hexdigest()
    ev = np.linalg.eigvalsh(Sig)[::-1]; sig_info = dict(sha256=sig_sha, dim=int(len(Sig)), eff_rank=float(1 / np.sum(ev ** 2)), top10_share=float(ev[:10].sum()))
    print(f'Sigma: {sig_info} ({time.time() - t0:.0f}s)', flush=True)
    # 2. D-073 windows and texts
    St = np.load(os.path.join(CACHE, 'stim.npz'), allow_pickle=True)
    E = {s: np.load(os.path.join(CACHE, f'sub-{s:03d}.npz'), allow_pickle=True)['eeg20'] for s in subjects}
    T = [min(len(St['env20'][r]), St['bands20'][r].shape[1], *(E[s][r].shape[1] for s in subjects)) for r in range(N_RUNS)]
    win = {r: windows(T[r], L, None, tc['n_candidates']) for r in range(N_RUNS)}
    texts, wrun = [], []
    for r in range(N_RUNS):
        words = list(St['words'][r])
        for s0 in win[r][0]:
            a, b = s0 / fs, (s0 + L) / fs; texts.append(' '.join(x[2] for x in words if a <= x[0] < b).lower()); wrun.append(r)
    wrun = np.array(wrun); Y = embed(texts); n_empty = int(sum(t == '' for t in texts)); off = np.r_[0, np.cumsum([len(win[r][0]) for r in range(N_RUNS)])]
    print(f'{len(texts)} windows, {n_empty} without words ({time.time() - t0:.0f}s)', flush=True)
    Lc = np.linalg.cholesky(Sig + 1e-8 * np.eye(len(Sig)))
    grid = 10 ** np.linspace(**c['noise']['scale_grid']); curves = {nt: [] for nt in c['noise']['types']}
    acc = {nt: np.zeros((len(grid), 3)) for nt in c['noise']['types']}
    for f in FOLDS:
        te = np.isin(wrun, f); Yc = Y[te] - Y[~te].mean(0); Yn = unit(Yc); en = (Yc ** 2).sum(1).mean(); n = len(Yc)
        loc = {r: (off[r] - off[f[0]], win[r][1]) for r in f}       # position of run r's windows inside the fold block
        for nt in c['noise']['types']:
            for gi, g in enumerate(grid):
                for _ in range(c['noise']['draws']):
                    Z = rng.standard_normal(Yc.shape); N = Z @ Lc.T * np.sqrt(en) if nt == 'residual' else Z * np.sqrt(en / Yc.shape[1])
                    Pr = Yc + g * N; Pn = unit(Pr); rp = np.concatenate([rank_pct(Pn[o:o + len(NB)], Yn[o:o + len(NB)], NB)[0] for o, NB in loc.values()])
                    cs = float(((Pn * Yn).sum(1)).mean()); acc[nt][gi] += np.array([cs, rp.mean(), 1]) * n
    for nt in curves:
        for gi, g in enumerate(grid): curves[nt].append(dict(scale=float(g), cosine=float(acc[nt][gi, 0] / acc[nt][gi, 2]), mrp=float(acc[nt][gi, 1] / acc[nt][gi, 2])))
    print(f'oracle curves done ({time.time() - t0:.0f}s)', flush=True)
    # precision check: forward envelope-only TRF (pool 17) on the D-073 windows
    calib = {}
    if c['calibration']['env_forward']:
        lags = list(range(int(round(tc['lags_ms'][0] / 1000 * fs)), int(round(tc['lags_ms'][1] / 1000 * fs)) + 1))
        lams = 10.0 ** np.arange(tcfg['lambdas_log10'][0], tcfg['lambdas_log10'][1] + 1)
        Yeeg = {s: [z(np.asarray(E[s][r][:, :T[r]], np.float64)).T.astype(np.float32) for r in range(N_RUNS)] for s in pool}; del E
        Ybar = [np.mean([Yeeg[s][r] for s in pool], 0) for r in range(N_RUNS)]
        X = [lagged(z(np.asarray(St['env20'][r][:T[r]], np.float64))[:, None], lags).astype(np.float64) for r in range(N_RUNS)]
        Cr = [x.T @ x for x in X]; Br = [X[r].T @ Ybar[r] for r in range(N_RUNS)]; rps = {s: [] for s in pool}; lam_f = []
        for test in FOLDS:
            train = [r for r in range(N_RUNS) if r not in test]; groups = [train[i::4] for i in range(4)]; score = np.zeros(len(lams))
            for g in groups:
                tr = [r for r in train if r not in g]
                for j, lam in enumerate(lams):
                    W = ridge(sum(Cr[r] for r in tr), sum(Br[r] for r in tr), lam)
                    score[j] += np.mean([corr(X[r] @ W, Yeeg[s][r]).mean() for r in g for s in pool])
            lam = lams[int(np.argmax(score))]; lam_f.append(float(lam)); W = ridge(sum(Cr[r] for r in train), sum(Br[r] for r in train), lam)
            for r in test:
                Pm = chunk_mat(X[r] @ W, win[r][0], L)
                for s in pool: rps[s].append(rank_pct(chunk_mat(Yeeg[s][r], win[r][0], L), Pm, win[r][1])[0])
        fwd = float(np.mean([np.concatenate(rps[s]).mean() for s in pool]))
        Yt = P['target_ENV'].astype(np.float64); tst = P['tstart']; bk = []
        for si in pidx:
            v = []
            for f in FOLDS:
                trc, tec = np.where(~np.isin(prun, f))[0], np.where(np.isin(prun, f))[0]; Yc = Yt[tec] - Yt[trc].mean(0)
                pos = np.full(len(prun), -1); pos[tec] = np.arange(len(tec)); C = pos[local_candidates(prun, tst, tec)]
                S = unit(P['pred_ENV'][si][tec].astype(np.float64)) @ unit(Yc).T; d = S[C[:, 0], C[:, 0]]; cand = S[C[:, 0][:, None], C]
                v.append(((cand < d[:, None]).sum(1) + 0.5 * ((cand == d[:, None]).sum(1) - 1)) / (C.shape[1] - 1))
            bk.append(np.concatenate(v).mean())
        calib = dict(forward_env_only_mrp=fwd, backward_env_observed_mrp=float(np.mean(bk)), lambdas=lam_f,
                     note='different chunk sets (D-073 windows vs D-061 chunks); reported only (D-083)')
        print(f'calibration: {calib} ({time.time() - t0:.0f}s)', flush=True)
    examples = {nt: {str(cs): b_from_cstar(cs, curves[nt], c['mrp_B']) for cs in (0.02, 0.05, 0.1, 0.2)} for nt in curves}
    res = dict(config=c, sigma=sig_info, n_windows=len(texts), n_empty=n_empty, curves=curves, calibration=calib,
               b_examples_not_results=examples, seconds=time.time() - t0)
    json.dump(res, open(os.path.join(out, 'curve.json'), 'w'), indent=1); print(json.dumps(dict(calibration=calib, examples=examples), indent=1), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
