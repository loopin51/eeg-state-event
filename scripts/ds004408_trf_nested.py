"""ds004408 nested forward TRF judged by forward-model identification (D-069, registered before results).

Models (20 Hz, lags -100..700 ms, 128 channels; word-level regressors are impulses at word onset):
  A      = envelope + 8 log-band energies + word onset
  B      = A + Zipf frequency, GPT-2 surprisal, entropy
  Cs     = B + 16 PCs of GloVe (static)            C      = B + 16 PCs of GPT-2 layer-6 (contextual)
  Cp     = B + S0 semantic dissimilarity           A_ctx  = A + 16 PCs of layer-6 (free attribution)
  C_shuf = B + 16 PCs of layer-6 from the sentence-shuffled story (context-shuffle control)
Semantic vectors are residualised on 6 word variables (letters, syllables, duration, Zipf, surprisal, entropy) and reduced
by PCA inside each run fold (training-run words only). Residualisation gate (EEG-free): run-grouped CV R^2 of the
residual PCs predicting each variable < 0.01 for ridge and gradient boosting; else degree-2 residualisation; else the
feature is flagged and comparisons using it are "undecidable (gate)".
Pooled model (primary): training pool 17 subjects (sub-001/015 excluded), run 5-fold; held-out runs are never trained on.
Per-subject models (secondary). Identification (Kay et al. 2008): 3-s chunks, 20 candidates = query + 19 nearest chunks
of the same run, candidate score = correlation of predicted (128 x 60) and observed EEG; rank percentile (0.5 chance).
Intervals: refit bootstrap over subjects (penalty fixed per fold) x pigeonhole resampling of query chunks.
Null: within-run circular shifts (>= 10 s) of the added semantic regressors, refit (pooled; per-subject for H1).
Verdict rules: docs/DECISIONS.md D-069.
Supplement (D-070, registered after the word-level gate proved unattainable): nonlinear lexical control at model level.
  B_nl = B + cubic B-spline bases (3 quantile knots, 4 columns each) of the 6 word variables;
  C_nl = B_nl + 16 PCs of layer-6, Cs_nl = B_nl + 16 PCs of GloVe, with the vectors residualised on [6 variables, splines].
  S1 = C_nl - B_nl, S2 = C_nl - Cs_nl; one-sided alpha 0.05 Holm over the 2, outside F1.

usage: python scripts/ds004408_trf_nested.py configs/ds004408_s0_trf.yaml [--quick] [--pca-k 32] [--end-key sent_end_pause] [--tag k32]
  --end-key only changes the sentence ends of the S0 dissimilarity (Cp); --pca-k only changes the PC models, so one run with
  both serves the two registered sensitivity analyses (H3 with pause ends, H1/H2/S1/S2 with 32 PCs).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ds004408_p0 import lagged, ridge, corr, z
from ds004408_s0 import dissimilarity, sentences, load_words
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CACHE = os.path.join(ROOT, 'data/cache/ds004408')
N_RUNS = 20; FOLDS = [list(range(i, i + 4)) for i in range(0, 20, 4)]
B_VARS = ['n_letters', 'n_syl', 'duration', 'zipf', 'surprisal', 'entropy']
SEM = {'glove': 'glove', 'ctx': 'ctx_l6', 'shuf': 'ctx_l6_shuf'}
SEM_NL = {'glove_nl': 'glove', 'ctx_nl': 'ctx_l6'}
PARTS = {'A': [], 'B': ['B3'], 'Cs': ['B3', 'glove'], 'C': ['B3', 'ctx'], 'Cp': ['B3', 'dis'], 'A_ctx': ['ctx'], 'C_shuf': ['B3', 'shuf'],
         'B_nl': ['B3', 'spl'], 'C_nl': ['B3', 'spl', 'ctx_nl'], 'Cs_nl': ['B3', 'spl', 'glove_nl']}
COMPARE = {'H1': ('C', 'B'), 'H2': ('C', 'Cs'), 'H3': ('Cp', 'B'), 'free': ('A_ctx', 'A'), 'C_vs_shuf': ('C', 'C_shuf'), 'shuf_vs_B': ('C_shuf', 'B'),
           'S1': ('C_nl', 'B_nl'), 'S2': ('C_nl', 'Cs_nl'), 'nl_vs_B': ('B_nl', 'B')}
NULL = {'H1': ('C', 'ctx', 'B'), 'H2': ('C', 'ctx', 'Cs'), 'H3': ('Cp', 'dis', 'B'), 'S1': ('C_nl', 'ctx_nl', 'B_nl'), 'S2': ('C_nl', 'ctx_nl', 'Cs_nl')}
USES = {'H1': ['ctx'], 'H2': ['ctx', 'glove'], 'H3': [], 'free': ['ctx'], 'C_vs_shuf': ['ctx', 'shuf'], 'shuf_vs_B': ['shuf'], 'S1': [], 'S2': [], 'nl_vs_B': []}


def poly(Z, level):
    if level == 1: return Z
    k = Z.shape[1]; return np.hstack([Z] + [Z[:, i:i + 1] * Z[:, j:j + 1] for i in range(k) for j in range(i, k)])


def fold_features(F, wtr, k, levels, dis):
    bv = np.stack([F[v].astype(np.float64) for v in B_VARS], 1); Zb = (bv - bv[wtr].mean(0)) / (bv[wtr].std(0) + 1e-12)
    out = {'B3': Zb[:, 3:6]}
    for name, key in SEM.items():
        V = F[key].astype(np.float64); Z = np.hstack([np.ones((len(V), 1)), poly(Zb, levels.get(name, 1))])
        beta = np.linalg.lstsq(Z[wtr], V[wtr], rcond=None)[0]; R = V - Z @ beta; m = R[wtr].mean(0)
        Vt = np.linalg.svd(R[wtr] - m, full_matrices=False)[2]; P = (R - m) @ Vt[:k].T; out[name] = P / (P[wtr].std(0) + 1e-12)
    from sklearn.preprocessing import SplineTransformer
    spl = SplineTransformer(n_knots=3, degree=3, knots='quantile', include_bias=False).fit(Zb[wtr]).transform(Zb)
    out['spl'] = (spl - spl[wtr].mean(0)) / (spl[wtr].std(0) + 1e-12); Znl = np.hstack([np.ones((len(Zb), 1)), Zb, out['spl']])
    for name, key in SEM_NL.items():
        V = F[key].astype(np.float64); beta = np.linalg.lstsq(Znl[wtr], V[wtr], rcond=None)[0]; R = V - Znl @ beta; m = R[wtr].mean(0)
        Vt = np.linalg.svd(R[wtr] - m, full_matrices=False)[2]; P = (R - m) @ Vt[:k].T; out[name] = P / (P[wtr].std(0) + 1e-12)
    c = F['content']; out['dis'] = (dis / (dis[wtr & c].std() + 1e-12))[:, None]
    return out


def resid_gate(F, k, thr, quick):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    run = F['run']; res, levels = {}, {}; zero = np.zeros(len(run))
    for name in SEM:
        res[name] = {}
        for level in (1, 2):
            pred = {kind: np.zeros((len(B_VARS), len(run))) for kind in ('linear', 'nonlinear')}
            for test in FOLDS:
                wtr = ~np.isin(run, test); X = fold_features(F, wtr, k, {name: level}, zero)[name]
                for vi, v in enumerate(B_VARS):
                    y = F[v].astype(np.float64)
                    for kind, mdl in (('linear', Ridge(alpha=1.0)), ('nonlinear', HistGradientBoostingRegressor(max_iter=50 if quick else 200, random_state=0))):
                        mdl.fit(X[wtr], y[wtr]); pred[kind][vi, ~wtr] = mdl.predict(X[~wtr])
            r2 = {kind: {v: float(1 - np.sum((F[v] - pred[kind][vi]) ** 2) / np.sum((F[v] - F[v].mean()) ** 2)) for vi, v in enumerate(B_VARS)} for kind in pred}
            ok = all(x < thr for d in r2.values() for x in d.values()); res[name][f'level{level}'] = dict(r2=r2, passed=ok)
            print(f'gate {name} level {level}: max R2 linear {max(r2["linear"].values()):.4f} nonlinear {max(r2["nonlinear"].values()):.4f} -> {ok}', flush=True)
            if ok: break
        levels[name] = level; res[name]['used_level'] = level; res[name]['passed'] = ok
    return res, levels


def windows(T, L, starts=None, n_cand=20):
    st = np.arange(T // L) * L if starts is None else np.asarray(starts)
    st = st[st + L <= T]; n = len(st); idx = np.arange(n)
    NB = np.stack([np.array(sorted(idx, key=lambda j: (abs(j - i), j))[:min(n_cand, n)]) for i in idx]) if n else np.zeros((0, 0), int)
    return st, NB


def chunk_mat(M, st, L):
    X = np.stack([M[s:s + L].ravel() for s in st]).astype(np.float32); X -= X.mean(1, keepdims=True)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def rank_pct(O, P, NB):
    S = O @ P.T; d = np.diag(S); cand = S[np.arange(len(S))[:, None], NB]
    less = (cand < d[:, None]).sum(1); eq = (cand == d[:, None]).sum(1) - 1; top1 = (less == NB.shape[1] - 1)
    return (less + 0.5 * eq) / (NB.shape[1] - 1), top1


def holm(p, alpha):
    order = np.argsort(p); m = len(p); rej = np.zeros(m, bool); lev = np.zeros(m)
    for rank, i in enumerate(order):
        lev[i] = alpha / (m - rank)
        if p[i] <= lev[i] and (rank == 0 or rej[order[rank - 1]]): rej[i] = True
    return rej, lev


def prevalence(obs, null, alpha=0.05, n_comb=10000, seed=0):
    """Allefeld et al. (2016) minimum-statistic prevalence lower bound. obs [N], null [N, K]."""
    rng = np.random.default_rng(seed); N, K = null.shape; m_obs = obs.min()
    m_null = null[np.arange(N)[:, None], rng.integers(0, K, (N, n_comb))].min(0)
    p = (1 + np.sum(m_null >= m_obs)) / (1 + n_comb); q = p ** (1 / N)
    g = (alpha ** (1 / N) - q) / (1 - q) if p <= alpha else None
    return dict(p_global=float(p), gamma0_lower=None if g is None else float(g))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--pca-k', type=int); ap.add_argument('--tag', default='')
    ap.add_argument('--end-key', default='sent_end_punct', choices=['sent_end_punct', 'sent_end_pause'])
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); tc = cfg['trf']; t0 = time.time(); rng = np.random.default_rng(cfg['seed'])
    k = a.pca_k or tc['pca_k']; fs = tc['fs']; n_boot = 5 if a.quick else tc['boot']; n_shift = 3 if a.quick else tc['shifts']
    subjects = [2, 3, 4, 5] if a.quick else cfg['subjects']; pool = [s for s in subjects if s not in cfg['pool_exclude']]
    pi = [subjects.index(s) for s in pool]; npool = len(pool)
    out_dir = os.path.join(ROOT, cfg['out_dir']); os.makedirs(out_dir, exist_ok=True)
    lams = 10.0 ** np.arange(cfg['lambdas_log10'][0], cfg['lambdas_log10'][1] + 1)
    lags = list(range(int(round(tc['lags_ms'][0] / 1000 * fs)), int(round(tc['lags_ms'][1] / 1000 * fs)) + 1))
    F = load_words(); S = np.load(os.path.join(CACHE, 'stim.npz'), allow_pickle=True)
    assert sum(len(w) for w in S['words']) == len(F['run'])
    dis = dissimilarity(F, a.end_key, 'sentence', cfg['seed'])[0]
    gate, levels = resid_gate(F, k, tc['resid_gate_r2'], a.quick); print(f'gate done ({time.time() - t0:.0f}s)', flush=True)
    # --- data
    E = {s: np.load(os.path.join(CACHE, f'sub-{s:03d}.npz'), allow_pickle=True)['eeg20'] for s in subjects}
    T = [min(len(S['env20'][r]), S['bands20'][r].shape[1], *(E[s][r].shape[1] for s in subjects)) for r in range(N_RUNS)]
    Y = {s: [z(np.asarray(E[s][r][:, :T[r]], np.float64)).T.astype(np.float32) for r in range(N_RUNS)] for s in subjects}; del E
    Ybar = [np.mean([Y[s][r] for s in pool], 0) for r in range(N_RUNS)]
    wi = [np.flatnonzero(F['run'] == r) for r in range(N_RUNS)]
    wpos = [np.round(F['onset'][w] * fs).astype(int) for w in wi]; wok = [p < T[r] for r, p in enumerate(wpos)]
    def impulse(r, vals):
        M = np.zeros((T[r], vals.shape[1])); np.add.at(M, wpos[r][wok[r]], vals[wi[r][wok[r]]]); return M
    base = []
    for r in range(N_RUNS):
        on = np.zeros((T[r], 1)); np.add.at(on[:, 0], wpos[r][wok[r]], 1.0)
        base.append(np.hstack([z(np.asarray(S['env20'][r][:T[r]], np.float64))[:, None], z(np.asarray(S['bands20'][r][:, :T[r]], np.float64), axis=1).T, on]))
    L = int(tc['chunk_s'] * fs); win = {r: windows(T[r], L, None, tc['n_candidates']) for r in range(N_RUNS)}
    Ls = int(tc['sentence_min_s'] * fs); win_s = {}
    for r, sents in enumerate(sentences(F['run'], F['sent_end_punct'])):
        st = [int(round(F['onset'][s[0]] * fs)) for s in sents if F['offset'][s[-1]] - F['onset'][s[0]] >= tc['sentence_min_s']]
        win_s[r] = windows(T[r], Ls, st, tc['n_candidates'])
    O = {s: {r: chunk_mat(Y[s][r], win[r][0], L) for r in range(N_RUNS)} for s in subjects}
    Os = {s: {r: chunk_mat(Y[s][r], win_s[r][0], Ls) for r in range(N_RUNS)} for s in subjects}
    nq = [len(win[r][0]) for r in range(N_RUNS)]; qoff = np.r_[0, np.cumsum(nq)]; NQ = int(qoff[-1])
    print(f'{NQ} chunks, {sum(len(win_s[r][0]) for r in range(N_RUNS))} sentence windows ({time.time() - t0:.0f}s)', flush=True)
    def design(r, feats, m, shift_part=None, shift=0):
        cols = [base[r]] + [np.roll(impulse(r, feats[p]), shift, 0) if p == shift_part else impulse(r, feats[p]) for p in PARTS[m]]
        return lagged(np.hstack(cols), lags)
    def evaluate(P, s, r):
        rp, t1 = rank_pct(O[s][r], chunk_mat(P, win[r][0], L), win[r][1]); return rp, t1
    # --- observed fits
    models = list(PARTS); feats_f = []; lam_pool = {m: [0.0] * 5 for m in models}; lam_sub = {m: [[0.0] * len(subjects) for _ in range(5)] for m in models}
    Csum = {m: [None] * 5 for m in models}; G = {m: [None] * 5 for m in models}; Xtest = {m: [None] * 5 for m in models}
    rp_pool = {m: np.zeros((len(subjects), NQ)) for m in models}; t1_pool = {m: np.zeros((len(subjects), NQ)) for m in models}
    rp_sub = {m: np.zeros((len(subjects), NQ)) for m in models}; r_pool = {m: np.zeros((len(subjects), N_RUNS)) for m in models}
    rp_sent = {m: {s: [] for s in subjects} for m in models}; P_A = {}
    minutes = tc['learning_curve_min']; rp_lc = {mm: {m: np.zeros((len(subjects), NQ)) for m in ('B', 'C')} for mm in minutes}; lc_min = {mm: [] for mm in minutes}
    for fi, test in enumerate(FOLDS):
        train = [r for r in range(N_RUNS) if r not in test]; feats = fold_features(F, np.isin(F['run'], train), k, levels, dis); feats_f.append(feats)
        groups = [train[i::4] for i in range(4)]
        for m in models:
            Xr = [design(r, feats, m) for r in range(N_RUNS)]; Cr = [X.T.astype(np.float64) @ X for X in Xr]
            XtY = np.stack([[Xr[r].T @ Y[s][r] for r in range(N_RUNS)] for s in subjects]).astype(np.float64)   # [S, R, p, 128]
            XtYbar = XtY[pi].mean(0)
            score = np.zeros(len(lams)); score_s = np.zeros((len(subjects), len(lams)))
            for g in groups:
                tr = [r for r in train if r not in g]; Ct = sum(Cr[r] for r in tr)
                for j, lam in enumerate(lams):
                    W = ridge(Ct, XtYbar[tr].sum(0), lam)
                    for r in g:
                        P = Xr[r] @ W; score[j] += np.mean([corr(P, Y[s][r]).mean() for s in pool])
                    for si, s in enumerate(subjects):
                        Ws = ridge(Ct, XtY[si, tr].sum(0), lam); score_s[si, j] += np.mean([corr(Xr[r] @ Ws, Y[s][r]).mean() for r in g])
            lam_pool[m][fi] = float(lams[int(np.argmax(score))]); Ct = sum(Cr[r] for r in train); Csum[m][fi] = Ct
            G[m][fi] = XtY[:, train].sum(1); Xtest[m][fi] = {r: Xr[r] for r in test}
            W = ridge(Ct, XtYbar[train].sum(0), lam_pool[m][fi])
            for r in test:
                P = Xr[r] @ W; sl = slice(qoff[r], qoff[r + 1])
                if m == 'A': P_A[r] = P
                for si, s in enumerate(subjects):
                    rp_pool[m][si, sl], t1_pool[m][si, sl] = evaluate(P, s, r); r_pool[m][si, r] = corr(P, Y[s][r]).mean()
                    if len(win_s[r][0]) > 1: rp_sent[m][s].append(rank_pct(Os[s][r], chunk_mat(P, win_s[r][0], Ls), win_s[r][1])[0])
            for si, s in enumerate(subjects):
                lam_sub[m][fi][si] = float(lams[int(np.argmax(score_s[si]))]); Ws = ridge(Ct, G[m][fi][si], lam_sub[m][fi][si])
                for r in test: rp_sub[m][si, qoff[r]:qoff[r + 1]] = evaluate(Xr[r] @ Ws, s, r)[0]
                if m in ('B', 'C'):
                    for mm in minutes:
                        cum = np.cumsum([T[r] / fs / 60 for r in train]); nr = int(np.searchsorted(cum, mm) + 1); runs = train[:nr]
                        if si == 0 and m == 'B': lc_min[mm].append(float(cum[min(nr, len(cum)) - 1]))
                        Wl = ridge(sum(Cr[r] for r in runs), XtY[si, runs].sum(0), lam_sub[m][fi][si])
                        for r in test: rp_lc[mm][m][si, qoff[r]:qoff[r + 1]] = evaluate(Xr[r] @ Wl, s, r)[0]
            print(f'fold {fi} {m}: lam pool {lam_pool[m][fi]:g}, p={Xr[0].shape[1]} ({time.time() - t0:.0f}s)', flush=True)
            del Xr, Cr, XtY
    # --- observed summaries
    def dmrp(rp, big, small, rows): return float((rp[big][rows] - rp[small][rows]).mean())
    obs = {c: dmrp(rp_pool, b, s_, pi) for c, (b, s_) in COMPARE.items()}
    obs_all = {c: dmrp(rp_pool, b, s_, list(range(len(subjects)))) for c, (b, s_) in COMPARE.items()}
    sent = {m: np.array([np.concatenate(rp_sent[m][s]).mean() for s in subjects]) for m in models}
    # --- gate A: pooled A r vs prediction-shift null
    rA = r_pool['A'][pi].mean(); nullA = []
    for _ in range(n_shift):
        sh = [int(rng.integers(int(tc['min_shift_s'] * fs), T[r] - int(tc['min_shift_s'] * fs))) for r in range(N_RUNS)]
        nullA.append(np.mean([[corr(np.roll(P_A[r], sh[r], 0), Y[s][r]).mean() for r in range(N_RUNS)] for s in pool]))
    print(f'gate A: r {rA:.4f}, null p95 {np.quantile(nullA, .95):.4f} ({time.time() - t0:.0f}s)', flush=True)
    # --- circular-shift nulls (pooled; per-subject for H1)
    null = {c: [] for c in NULL}; null_sub = np.zeros((len(subjects), n_shift))
    for it in range(n_shift):
        sh = [int(rng.integers(int(tc['min_shift_s'] * fs), T[r] - int(tc['min_shift_s'] * fs))) for r in range(N_RUNS)]
        null_models = sorted({(m, part) for m, part, _ in NULL.values()}); rp_n = {m: np.zeros((len(subjects), NQ)) for m, _ in null_models}; rp_ns = np.zeros((len(subjects), NQ))
        for fi, test in enumerate(FOLDS):
            train = [r for r in range(N_RUNS) if r not in test]
            for m, part in null_models:
                Xr = {r: design(r, feats_f[fi], m, part, sh[r]) for r in range(N_RUNS)}
                Ct = sum(Xr[r].T.astype(np.float64) @ Xr[r] for r in train); W = ridge(Ct, sum(Xr[r].T @ Ybar[r] for r in train), lam_pool[m][fi])
                for r in test:
                    P = Xr[r] @ W
                    for si in pi: rp_n[m][si, qoff[r]:qoff[r + 1]] = evaluate(P, subjects[si], r)[0]
                if m == 'C':
                    for si, s in enumerate(subjects):
                        Ws = ridge(Ct, sum(Xr[r].T @ Y[s][r] for r in train), lam_sub[m][fi][si])
                        for r in test: rp_ns[si, qoff[r]:qoff[r + 1]] = evaluate(Xr[r] @ Ws, s, r)[0]
        for c, (m, _, small) in NULL.items(): null[c].append(float((rp_n[m][pi] - rp_pool[small][pi]).mean()))
        null_sub[:, it] = (rp_ns - rp_sub['B']).mean(1)
        print(f'shift {it}: ' + ' '.join(f'{c} {null[c][-1]:+.4f}' for c in NULL) + f' ({time.time() - t0:.0f}s)', flush=True)
    # --- refit bootstrap x pigeonhole
    boot = {c: [] for c in COMPARE}; bootA = []
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, npool, npool), minlength=npool).astype(float); cq = np.bincount(rng.integers(0, NQ, NQ), minlength=NQ).astype(float)
        drawn = [pi[j] for j in np.flatnonzero(w)]; ww = w[w > 0]; rpb = {m: np.zeros((len(drawn), NQ)) for m in models}; rAb = np.zeros(len(drawn))
        for fi, test in enumerate(FOLDS):
            for m in models:
                W = ridge(Csum[m][fi], np.tensordot(w, G[m][fi][pi], 1) / npool, lam_pool[m][fi])
                for r in test:
                    P = Xtest[m][fi][r] @ W
                    for di, si in enumerate(drawn):
                        rpb[m][di, qoff[r]:qoff[r + 1]] = evaluate(P, subjects[si], r)[0]
                        if m == 'A': rAb[di] += corr(P, Y[subjects[si]][r]).mean() / N_RUNS
        tot = ww.sum() * cq.sum()
        for c, (bg, sm) in COMPARE.items(): boot[c].append(float(ww @ (rpb[bg] - rpb[sm]) @ cq / tot))
        bootA.append(float(ww @ rAb / ww.sum()))
        if b % 50 == 0: print(f'boot {b}: ' + ' '.join(f'{c} {boot[c][-1]:+.4f}' for c in COMPARE) + f' ({time.time() - t0:.0f}s)', flush=True)
    # --- verdicts (D-069)
    alpha, at = tc['alpha_f1'], tc['alpha_tost']; Hs = ['H1', 'H2', 'H3']
    pboot = np.array([(1 + np.sum(np.array(boot[c]) <= 0)) / (1 + n_boot) for c in Hs]); rej, lev = holm(pboot, alpha)
    gateA = dict(r=float(rA), null_p95=float(np.quantile(nullA, .95)), boot_lower=float(np.quantile(bootA, .05)),
                 passed=bool(rA > np.quantile(nullA, .95) and np.quantile(bootA, .05) > 0))
    comp = {}
    for c in COMPARE:
        bt = np.array(boot[c]); d = dict(obs=obs[c], obs_all19=obs_all[c], boot_se=float(bt.std(ddof=1)), ci90=[float(np.quantile(bt, .05)), float(np.quantile(bt, .95))],
                                       lower95_one_sided=float(np.quantile(bt, alpha)), gate_ok=all(gate[u]['passed'] for u in USES[c]))
        if c in NULL: d['null_p95'] = float(np.quantile(null[c], .95)); d['null'] = null[c]
        if c in Hs:
            i = Hs.index(c); d.update(p_boot=float(pboot[i]), holm_level=float(lev[i]), holm_lower=float(np.quantile(bt, lev[i])), holm_rejected=bool(rej[i]))
            d['exists'] = bool(rej[i] and obs[c] > d['null_p95'])
            d['verdict'] = ('보류(관문 A)' if not gateA['passed'] else '결정 불가(잔차화 관문)' if not d['gate_ok'] else
                            ('있음' if d['exists'] else '존재 미확인'))
        comp[c] = d
    Ss = ['S1', 'S2']; ps = np.array([(1 + np.sum(np.array(boot[c]) <= 0)) / (1 + n_boot) for c in Ss]); rej_s, lev_s = holm(ps, alpha)
    for i, c in enumerate(Ss):
        d = comp[c]; bt = np.array(boot[c]); d.update(p_boot=float(ps[i]), holm_level=float(lev_s[i]), holm_lower=float(np.quantile(bt, lev_s[i])), holm_rejected=bool(rej_s[i]))
        d['exists'] = bool(rej_s[i] and obs[c] > d['null_p95']); d['verdict'] = '보류(관문 A)' if not gateA['passed'] else ('있음(보완)' if d['exists'] else '존재 미확인(보완)')
    a_s1 = float((norm.ppf(1 - at) + norm.ppf(.8)) * comp['S1']['boot_se'])
    stop_s1 = dict(a=a_s1, upper90_unique=comp['S1']['ci90'][1], upper90_free=comp['free']['ci90'][1],
                   both_below_a=bool(comp['S1']['ci90'][1] < a_s1 and comp['free']['ci90'][1] < a_s1), verdict='보류(b 미정)')
    a_gain = float((norm.ppf(1 - at) + norm.ppf(.8)) * comp['H1']['boot_se']); up_u, up_f = comp['H1']['ci90'][1], comp['free']['ci90'][1]
    stop = dict(a=a_gain, b=None, upper90_unique=up_u, upper90_free=up_f, both_below_a=bool(up_u < a_gain and up_f < a_gain),
                verdict='보류(b 미정)', statement=(f'고유·자유 귀속 모두 a={a_gain:.4f} 이상의 이득을 배제한다' if up_u < a_gain and up_f < a_gain else 'a 이상의 이득을 배제하지 못한다'))
    h1 = comp['H1']
    if gateA['passed'] and h1['gate_ok'] and not h1['exists']:
        if comp['H1']['lower95_one_sided'] <= 0 and comp['free']['lower95_one_sided'] > 0: h1['note'] = '언어 정보는 있으나 의미와 예측 정보를 분리할 수 없음'
    if h1.get('exists') and comp['C_vs_shuf']['lower95_one_sided'] <= 0: h1['note_shuffle'] = '증분이 문맥보다 단어 정체성에서 왔을 가능성'
    sub = dict(dmrp_H1=(rp_sub['C'] - rp_sub['B']).mean(1).tolist(), null_p95=np.quantile(null_sub, .95, axis=1).tolist())
    sub['share_above_own_null'] = float(np.mean(np.array(sub['dmrp_H1'])[pi] > np.array(sub['null_p95'])[pi]))
    sub['prevalence_pool'] = prevalence(np.array(sub['dmrp_H1'])[pi], null_sub[pi])
    sub['mrp'] = {m: rp_sub[m].mean(1).tolist() for m in models}
    lc = {str(mm): dict(minutes_mean=float(np.mean(lc_min[mm])), dmrp_per_subject=(rp_lc[mm]['C'] - rp_lc[mm]['B']).mean(1).tolist(),
                        dmrp_pool_mean=float((rp_lc[mm]['C'] - rp_lc[mm]['B'])[pi].mean())) for mm in minutes}
    res = dict(config=tc, pca_k=k, end_key=a.end_key, quick=a.quick, subjects=subjects, pool=pool, n_chunks=NQ, gate_resid=gate, gate_A=gateA,
               lam_pool=lam_pool, lam_sub=lam_sub,
               mrp_pool={m: float(rp_pool[m][pi].mean()) for m in models}, mrp_all19={m: float(rp_pool[m].mean()) for m in models},
               top1_pool={m: float(t1_pool[m][pi].mean()) for m in models}, r_pool={m: float(r_pool[m][pi].mean()) for m in models},
               dr_pool={c: float((r_pool[b][pi] - r_pool[s_][pi]).mean()) for c, (b, s_) in COMPARE.items()},
               sentence_dmrp={c: float((sent[b][pi] - sent[s_][pi]).mean()) for c, (b, s_) in COMPARE.items()},
               sentence_mrp={m: float(sent[m][pi].mean()) for m in models},
               compare=comp, stop_H1=stop, stop_S1=stop_s1, per_subject=sub, learning_curve=lc, boot_A=bootA, boot={c: boot[c] for c in COMPARE})
    name = 'trf_nested' + (f'_{a.tag}' if a.tag else '') + ('_quick' if a.quick else '')
    json.dump(res, open(os.path.join(out_dir, name + '.json'), 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(dict(gate_A=gateA, mrp_pool=res['mrp_pool'], verdicts={c: comp[c].get('verdict') for c in Hs + Ss},
                          obs=obs, stop=stop, stop_S1=stop_s1), ensure_ascii=False, indent=1), f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
