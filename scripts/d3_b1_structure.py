"""D-093: error structure of the stage B B1 decoder transferred to the direction-3 device, and its p*.

Stage estimate (GPU for the segment-text embedding): for every pool subject s and test run r of the 25 B1 cross-fit jobs,
the within-run score matrix S (query x candidate, main kind). For query i the distractor set D_i holds the same-run segments
whose onset differs by >= min_sep_s (no shared audio). Scores are standardised over D_i: z_ij = (S_ij - mean) / sd.
  mu      = mean over queries of z_ii (target advantage in distractor-SD units)
  sd_t    = SD over queries of z_ii
  beta    = mean over queries of corr_j in D_i (z_ij, x_ij), x_ij = cosine of the D-068 TXT embeddings of the segment texts
            (words with onset in [t - 0.5, t + 2.5) s, the B1 window), centred on the mean embedding of the runs outside the
            test fold of r (semantic structure of the decoder's errors)
  p_meas  = mean over queries of P(target beats 19 distractors drawn without replacement from D_i)
            = C(m_i, 19) / C(|D_i|, 19), m_i = distractors with a lower score (measured 20-candidate accuracy)
Interval: pigeonhole subject x run bootstrap (D-085) over cell sums.

Stage pstar (CPU): Thurstone structure B on the device test trials (D-087 table). Trial i, candidate j:
  target       a * (mu + tau_t * eta_i) + e_ij,   tau_t^2 = max(sd_t^2 - (1 - beta^2), 0), eta ~ N(0, 1) per draw
  alternative  a * beta * xs_ij + e_ij,           xs = cosine to the target (TXT, centred as R) standardised over the 19
  e ~ N(0, 1 - beta^2) for every candidate (exchangeable at a = 0: uniform evidence, the D-088 U zero-information case).
  Swap (vector swap, D-087): another trial k's target in trial i's list: score a * beta * xs_ij(k) + e_ij, xs(k) = cosine to t_k
  standardised over the 20 candidates of trial i.
  a = 1 is the B1 decoder; a is set by bisection to reach each grid accuracy. p* follows D-089 (independent design, e5;
  bge and the shared design as sensitivity). Precondition: the bootstrap 5th percentile of mu > 0, else p*_B is undefined.

usage: python scripts/d3_b1_structure.py configs/d3_b1_structure.yaml --stage estimate|pstar [--smoke]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, time, argparse, yaml, numpy as np
from scipy.special import gammaln
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log_comb(n, k): return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def segment_texts(W, run, onset, tmin, tmax):
    out = []
    for r, t in zip(run, onset):
        m = (W['run'] == r) & (W['onset'] >= t + tmin) & (W['onset'] < t + tmax)
        out.append(' '.join(str(w) for w in W['word'][m]).lower())
    return out


def find_jobs(c, smoke):
    pat = os.path.join(ROOT, c['b1_runs'] + ('_smoke' if smoke else ''))
    return sorted(h for h in glob.glob(pat) if os.path.exists(os.path.join(h, 'scores.npz')) and (smoke or not h.endswith('_smoke')))


def estimate(c, a):
    from ds004408_identify import embed
    t0 = time.time(); bc = yaml.safe_load(open(os.path.join(ROOT, c['b1_config']))); w = bc['window']
    pool = sorted(s for f in bc['split']['subject_folds'] for s in f); folds = [[r - 1 for r in f] for f in bc['split']['run_folds']]
    W = np.load(os.path.join(ROOT, c['word_features']), allow_pickle=True); jobs = find_jobs(c, a.smoke)
    assert a.smoke or len(jobs) == 25, f'{len(jobs)} B1 jobs found'
    # segment texts and embeddings for every run that appears
    seg = {}
    for h in jobs:
        Z = np.load(os.path.join(h, 'scores.npz'))
        for r in np.unique(Z['run']): seg.setdefault(int(r), np.sort(Z['onset'][Z['run'] == r]))
    allw = {r: segment_texts(W, np.full(len(seg[r]), r), seg[r], w['tmin'], w['tmax']) for r in seg}
    # centring: mean embedding of all word windows (B1 windows at every word onset) of the runs outside the fold of r
    all_runs = np.unique(W['run']); ctx = {}
    for r in all_runs:
        on = np.unique(W['onset'][W['run'] == r]); ctx[int(r)] = segment_texts(W, np.full(len(on), r), on, w['tmin'], w['tmax'])
    E_ctx = {r: embed(ctx[r]) for r in ctx}; E_seg = {r: embed(allw[r]) for r in allw}
    fold_of = {r: i for i, f in enumerate(folds) for r in f}
    cells = []   # per (s, r): sums for the pigeonhole bootstrap
    for h in jobs:
        Z = np.load(os.path.join(h, 'scores.npz')); run = Z['run']; on = Z['onset']
        for key in [k for k in Z.files if k.startswith('main_s') and '_r' in k and 'rank' not in k]:
            s = int(key.split('_')[1][1:]); r = int(key.split('_r')[1])
            if s not in pool and not a.smoke: continue
            S = Z[key]; t = on[run == r]; o = np.argsort(t); assert np.allclose(t[o], seg[r]); S = S[np.ix_(o, o)]; t = t[o]
            out_runs = [q for q in ctx if fold_of.get(q) != fold_of.get(r)]
            mu_c = np.concatenate([E_ctx[q] for q in out_runs]).mean(0); X = E_seg[r] - mu_c; X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            C = X @ X.T; zt, cor, pm = [], [], []
            for i in range(len(t)):
                D = np.where(np.abs(t - t[i]) >= c['min_sep_s'])[0]
                if len(D) < c['k_device'] or np.isnan(S[i, i]): continue
                sd = S[i, D].std()
                if sd <= 0: continue
                z = (S[i, D] - S[i, D].mean()) / sd; zt.append((S[i, i] - S[i, D].mean()) / sd)
                x = C[i, D]; cor.append(float(np.corrcoef(z, x)[0, 1]) if x.std() > 0 else 0.0)
                m = int((S[i, D] < S[i, i]).sum()); k = c['k_device'] - 1
                pm.append(float(np.exp(log_comb(m, k) - log_comb(len(D), k))) if m >= k else 0.0)
            zt = np.array(zt); cells.append(dict(s=s, r=r, n=len(zt), sum_mu=float(zt.sum()), sum_mu2=float((zt ** 2).sum()),
                                                 sum_beta=float(np.sum(cor)), sum_p=float(np.sum(pm))))
        print(f'{os.path.basename(h)} ({time.time() - t0:.0f}s)', flush=True)
    def stats(cl):
        n = sum(x['n'] for x in cl); mu = sum(x['sum_mu'] for x in cl) / n; m2 = sum(x['sum_mu2'] for x in cl) / n
        return dict(n=n, mu=mu, sd_t=float(np.sqrt(max(m2 - mu ** 2, 0))), beta=sum(x['sum_beta'] for x in cl) / n, p_meas=sum(x['sum_p'] for x in cl) / n)
    point = stats(cells); rng = np.random.default_rng(c['seed']); subs = sorted({x['s'] for x in cells}); runs = sorted({x['r'] for x in cells}); B = []
    for _ in range(c['boot']):
        ss = rng.choice(subs, len(subs)); rr = rng.choice(runs, len(runs)); ws = {q: int((ss == q).sum()) for q in subs}; wr = {q: int((rr == q).sum()) for q in runs}
        cl = [dict(x, **{k: x[k] * ws[x['s']] * wr[x['r']] for k in ('n', 'sum_mu', 'sum_mu2', 'sum_beta', 'sum_p')}) for x in cells if ws[x['s']] * wr[x['r']]]
        B.append(stats(cl))
    ci = {k: np.quantile([b[k] for b in B], [0.05, 0.95]).tolist() for k in ('mu', 'sd_t', 'beta', 'p_meas')}
    res = dict(jobs=[os.path.basename(h) for h in jobs], n_cells=len(cells), point=point, ci90=ci, mu_p05=float(np.quantile([b['mu'] for b in B], 0.05)),
               boot_params=[{k: b[k] for k in ('mu', 'sd_t', 'beta')} for b in B], n_empty_texts=int(sum(x == '' for r in allw for x in allw[r])),
               smoke=bool(a.smoke), seconds=time.time() - t0)
    out = os.path.join(ROOT, c['results'] if not a.smoke else c['results'].replace('.json', '_smoke.json')); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1); print(json.dumps({k: res[k] for k in ('point', 'ci90', 'mu_p05', 'n_cells')}, indent=1), flush=True)


class BStructure:
    """Thurstone structure of D-093 on the device trials (see module docstring). Common random numbers over draws."""
    def __init__(self, emb, ti, mu, sd_t, beta, draws, seed):
        n, K, d = emb.shape; rng = np.random.default_rng(seed); self.n, self.K, self.ti = n, K, ti
        Cn = emb / (np.linalg.norm(emb, axis=2, keepdims=True) + 1e-12); t = Cn[np.arange(n), ti]
        cos = np.einsum('nd,nkd->nk', t, Cn); alt = np.ones((n, K), bool); alt[np.arange(n), ti] = False
        m = np.where(alt, cos, 0).sum(1, keepdims=True) / (K - 1); v = np.where(alt, (cos - m) ** 2, 0).sum(1, keepdims=True) / (K - 1)
        xs = np.where(alt, (cos - m) / np.sqrt(v + 1e-12), 0.0)
        k = (np.arange(n) + rng.integers(1, n, (draws, n))) % n                       # swap partner per draw
        cs = np.einsum('rnd,nkd->rnk', t[k], Cn); xsw = (cs - cs.mean(2, keepdims=True)) / (cs.std(2, keepdims=True) + 1e-12)
        tau = np.sqrt(max(sd_t ** 2 - (1 - beta ** 2), 0.0)); se = np.sqrt(max(1 - beta ** 2, 1e-6)); self.beta = beta
        eta = rng.standard_normal((draws, n, 1))
        self.sig = np.where(alt[None], beta * xs[None], (mu + tau * eta) * (~alt)[None]).astype(np.float32)
        self.sig_sw = (beta * xsw).astype(np.float32); self.E = (se * rng.standard_normal((draws, n, K))).astype(np.float32)
        self.E2 = (se * rng.standard_normal((draws, n, K))).astype(np.float32)

    def _hist(self, S): j = S.argmax(2); return (j[..., None] == np.arange(self.K)).mean(0)

    def q(self, a):
        if np.isinf(a): q = np.zeros((self.n, self.K)); q[np.arange(self.n), self.ti] = 1; return q   # p = 1: evidence = target (D-088)
        return self._hist(a * self.sig + self.E)

    def q_swap(self, a):
        if np.isinf(a): return self._hist(self.sig_sw) if self.beta > 0 else np.full((self.n, self.K), 1 / self.K)  # noise-free limit
        return self._hist(a * self.sig_sw + self.E2)

    def acc(self, a): return float(self.q(a)[np.arange(self.n), self.ti].mean())

    def scale_for(self, p, a_max, iters=40):
        """0 at p <= 1/K, inf (evidence = target) at p >= 1, None if p lies above the accuracy reached at a_max (saturation)"""
        if p <= 1 / self.K: return 0.0
        if p >= 1: return np.inf
        lo, hi = 0.0, a_max
        if self.acc(hi) < p: return None
        for _ in range(iters):
            mid = (lo + hi) / 2
            if self.acc(mid) < p: lo = mid
            else: hi = mid
        return (lo + hi) / 2


def pstar(c, a):
    from d3_device import load_device, moments, p_star, wolpaw
    t0 = time.time(); dc = yaml.safe_load(open(os.path.join(ROOT, c['device_config']))); sm = dc['simulate']
    est = json.load(open(os.path.join(ROOT, c['results'] if not a.smoke else c['results'].replace('.json', '_smoke.json'))))
    g = json.load(open(os.path.join(ROOT, sm['results_dir'], 'gate.json'))); P = est['point']
    res = dict(estimate=dict(point=P, ci90=est['ci90'], mu_p05=est['mu_p05']), precondition_mu_p05_gt_0=bool(est['mu_p05'] > 0))
    D = load_device(dc, a.tag); ti = D['target_index']; n = len(ti); K = D['rr_e5'].shape[1] - 1
    Bs = BStructure(D['emb'], ti, P['mu'], P['sd_t'], P['beta'], c['draws'], c['seed'])
    # location of the B1 decoder on the device (a = 1), with the parameter bootstrap (200 of the pigeonhole draws)
    loc = dict(acc=Bs.acc(1.0), delta_swap={}, delta_alone={})
    for m in ('e5', 'bge'):
        mo = moments(D[f'rr_{m}'], Bs.q(1.0), Bs.q_swap(1.0)); loc['delta_swap'][m] = mo['swap']['delta']; loc['delta_alone'][m] = mo['alone']['delta']
    accs = [BStructure(D['emb'], ti, b['mu'], b['sd_t'], b['beta'], 50, c['seed']).acc(1.0) for b in est['boot_params'][:200]]
    loc['acc_ci90'] = np.quantile(accs, [0.05, 0.95]).tolist(); res['b1_location'] = loc
    if not res['precondition_mu_p05_gt_0']:
        res['p_star_B'] = None; res['note'] = 'B1 shows no target advantage (mu 5th percentile <= 0): structure shape undefined, p*_B undefined (D-093)'
    else:
        grid = sm['p_grid']; A = [Bs.scale_for(p, c['a_max']) for p in grid]; res['saturation_acc'] = Bs.acc(c['a_max'])
        res['dropped_grid'] = [p for p, x in zip(grid, A) if x is None]; A = [x for x in A if x is not None]   # above saturation
        Q = [(Bs.q(x), Bs.q_swap(x)) for x in A]
        pg = np.array([float(q[np.arange(n), ti].mean()) for q, _ in Q]); pg = np.maximum.accumulate(pg)
        rng = np.random.default_rng([c['seed'], 89]); bs = rng.integers(0, n, (sm['boot'], n)); res['scale'] = A; res['p_coord'] = pg.tolist(); res['cells'] = {}
        for m in ('e5', 'bge'):
            RR = D[f'rr_{m}']; M = [moments(RR, q, qs) for q, qs in Q]
            mom = {k: tuple(np.array([x[k][y] for x in M]) for y in ('delta', 'sigma', 'omega')) for k in ('swap', 'alone')}
            bm = [[moments(RR, q, qs, bs[r])['swap'] for q, qs in Q] for r in range(sm['boot'])]
            bmom = {'swap': np.array([[x['delta'] for x in row] for row in bm])}; bsig = {'swap': np.array([[x['sigma'] for x in row] for row in bm])}
            for N in sm['N']:
                for S in sm['S']:
                    pf = lambda dl, sg, om=None: p_star(pg, dl, sg, N, S, sm['logit_sd_subject'], sm['z_power'], omega=om)
                    ps = {k: pf(*mom[k][:2]) for k in mom}; val = lambda r: r['p_star'] if r['reachable'] else np.inf
                    pst = max(val(ps[k]) for k in ps); boot = np.array([val(pf(bmom['swap'][r], bsig['swap'][r])) for r in range(sm['boot'])])
                    res['cells'][f'{m}|N{N}|S{S}'] = dict(p_star=None if np.isinf(pst) else pst, p_star_swap=ps['swap'], p_star_alone=ps['alone'],
                                                          ci90_swap=np.quantile(boot, [0.05, 0.95]).tolist(), boot_unreachable=float(np.isinf(boot).mean()),
                                                          shared_design_sensitivity=pf(*mom['swap'])['p_star'],
                                                          bits_per_selection=wolpaw(None if np.isinf(pst) else pst, K))
        N0, S0 = sm['b_cell']; res['p_star_B'] = res['cells'][f'e5|N{N0}|S{S0}']['p_star']
        res['alone_uninterpretable'] = g['verdict']['per_structure']['U']['single_warning'].count('alone') > 0
    res['seconds'] = time.time() - t0
    out = os.path.join(ROOT, c['pstar_results'] if not a.smoke else c['pstar_results'].replace('.json', '_smoke.json'))
    json.dump(res, open(out, 'w'), indent=1, default=float); print(json.dumps({k: res[k] for k in ('estimate', 'b1_location', 'p_star_B')}, indent=1, default=float), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', required=True, choices=['estimate', 'pstar'])
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--tag', default=''); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); dict(estimate=estimate, pstar=pstar)[a.stage](c, a)
