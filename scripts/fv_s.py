"""F-IX (D-119, amendments 1-2) method S: staged colour / category read-out of the frozen decoder output, with abstention.
Server CPU (gpuq).

Inputs are unit outputs z (decoder, ridge, simulation); every read-out sees z~ = unit(z - mu), mu = mean original CLIP over the
14,540 non-development training images (the same centring for every input, amendment 2).
Residual model (amendment 2, review 2 item 3): z = a c + m + r on unit vectors (c = unit original CLIP of the photo, or the unit
concept mean of them for concept rows), a by least squares with intercept m, r ~ N(0, Sigma), Sigma by Ledoit-Wolf shrinkage of the
pooled residuals (10 subjects pooled, no per-subject centring: subject offsets stay in Sigma). Photo level (photo rows, about 4
repetitions) and concept level (concept rows: the mean of the 10 photo outputs against the concept-mean c).
Heads (colour 11 classes / category 18 classes):
  (i)  learned: softmax linear layer on unit(unit(a c + m + r) - mu) of the non-development training images (2 noise copies),
       soft targets (colour pixel fractions, category 1/|L|); colour uses the photo-level model, category the concept-level model.
       L2 penalty by concept-grouped 5-fold CV over the training images (once, with the full development residual model)
  (ii) text prototype: cos(unit(z), T_k), fv_s_text.py prompt embeddings
Calibration p = softmax(l / tau + b). Colour: tau and 11 free b. Category: tau and b with an L2 penalty (cal_l2b), b = 0 for the
classes with fewer than 3 development concepts (review 2 item 4).
Development cross-fit (review 2 item 2): for each of the 5 development-concept folds f, the residual model, the head (i) and the
calibration are fitted on the other 4 folds and fold f is predicted; the test uses the versions fitted on every development row.
Abstention (review 2 item 8): thresholds are confidence quantile levels on the grid 0.00..0.90 (0.01). In fold f the lowest level
whose said rows reach precision >= 0.5, lift >= 1.5 is found on its own cross-fitted rows; the out-of-fold check applies to fold f
the mean level of the other folds (when >= 3 of them found one); the test threshold is the mean level of the folds (>= 3 of 5),
converted to a confidence value on the development rows under the full calibration.

  dev     : development EEG outputs (D-115 infer, ses123 seed mean), ridge files (fv_s_ridge.py). Heads, calibrations, choice,
            thresholds, gates, meter calibrations, simulation scales (a matched to the development name MRP).
            -> results/fv_s/dev.json, results/fv_s/readout.npz, data/cache/fv_s/noise.npz
  sim     : no EEG. Information-zero and signal simulations on the test structure, EEG-free colour gate, references.
            -> results/fv_s/sim.json
  c2prep  : C2 inputs (data/cache/fv_s/c2_inputs.npz) for fv_s_c2.py
  verdict : once. -> results/fv_s/verdict.json
usage: python scripts/fv_s.py configs/fv_s.yaml --stage {dev,sim,c2prep,verdict} [--smoke]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, csv, glob, json, time, hashlib, argparse, yaml, numpy as np
from scipy.optimize import minimize
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.things.data import Meta
from ese.things import lang_rows as LR
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
STAGES = ('col', 'cat')
METERS = ('de', 'time', 'eye', 'pre', 'direct_eye', 'direct_pre')


def unit(x): x = np.asarray(x, np.float64); return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def peak_gb():
    import resource; return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def salt_folds(folders, salt, n):
    order = sorted(range(len(folders)), key=lambda i: hashlib.sha256(f'{folders[i]}|{salt}'.encode()).hexdigest()); f = np.zeros(len(folders), int)
    for r, i in enumerate(order): f[i] = r % n
    return f


def cells_mrp(S, true):
    t = S[np.arange(len(S)), true]; return 1 - (S > t[:, None]).sum(1) / (S.shape[1] - 1)


# ---------------------------------------------------------------- setup (no EEG)
class Setup:
    def __init__(self, c):
        self.c = c; self.tc = yaml.safe_load(open(P(c['train_config']))); self.lc = yaml.safe_load(open(P(c['lang_config']))); self.pc = yaml.safe_load(open(P(c['pairs_config'])))
        meta = self.meta = Meta(P(self.tc['meta'])); self.cls = sorted(c['categories']['phrase']); self.cols = c['colour']['names']; par = c['categories']['parent']
        self.names = list(csv.DictReader(open(P(self.lc['names_table']), encoding='utf-8'), delimiter='\t'))
        self.catset = [sorted({self.cls.index(par.get(x.strip(), x.strip())) for x in r['category'].split(',') if x.strip()}) for r in self.names]
        tr = {r['folder']: i for i, r in enumerate(self.names) if r['split'] == 'train'}; te = {r['folder']: i for i, r in enumerate(self.names) if r['split'] == 'test'}
        ix = json.load(open(P(c['clip_index']))); assert ix['split'].count('train') == 16540 and ix['split'][16540:] == ['test'] * 200
        assert all(ix['paths'][16540 + k].split('/')[1] == f for k, f in enumerate(meta.test_img_concept)), 'clip index test order'
        ic = np.array(meta.train_img_concept); self.dev_c = meta.dev_concepts(self.tc['dev']['salt'], self.tc['dev']['n'])
        self.dev_rows = np.flatnonzero(np.isin(ic, self.dev_c)); self.fit_rows = np.flatnonzero(~np.isin(ic, self.dev_c)); assert len(self.dev_rows) == 2000 and len(self.fit_rows) == 14540
        self.img_name = np.array([tr[f] for f in ic] + [te[f] for f in meta.test_img_concept])
        X = np.load(P(self.pc['pairs']['clip_img'])).astype(np.float64); assert X.shape == (16740, 1024); self.Craw = unit(X); self.mu = self.Craw[self.fit_rows].mean(0)
        assert np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-3), 'clip_img rows are unit vectors'
        L = np.load(P(os.path.join(c['out_dir'], 'colour_labels.npz'))); self.frac = L['frac'].astype(np.float64); self.top1 = L['top1'].astype(int)
        self.colour_sha = fsha(P(os.path.join(c['out_dir'], 'colour_labels.npz')))
        self.Ycat_name = np.zeros((len(self.names), len(self.cls)))
        for i, s in enumerate(self.catset):
            if s: self.Ycat_name[i, s] = 1 / len(s)
        self.has_cat = np.array([bool(s) for s in self.catset])
        ci = {f: k for k, f in enumerate(self.dev_c)}; self.dev_concept = np.array([ci[ic[r]] for r in self.dev_rows]); self.dev_name = np.array([tr[f] for f in self.dev_c])
        self.dev_fold = salt_folds(self.dev_c, c['fold_salt'], c['n_folds']); self.dev_catk = np.flatnonzero(self.has_cat[self.dev_name])
        self.Ck_raw = unit(np.stack([self.Craw[self.dev_rows][self.dev_concept == k].mean(0) for k in range(len(self.dev_c))]))     # unit concept means (raw)
        cnt = np.zeros(len(self.cls))
        for n in self.dev_name[self.dev_catk]: cnt[self.catset[n]] += 1
        self.cat_dev_count = cnt; self.cat_free = cnt >= c['calibration']['min_dev_concepts']
        self.test_name = np.array([te[f] for f in meta.test_img_concept]); self.test_rows = np.arange(16540, 16740)
        self.subjects = self.lc['subjects']
        NT = unit(np.load(os.path.join(P(self.tc['targets']['dir']), self.tc['targets']['names_text']))); self.Tdev = NT[self.dev_name]; self.Ttest = NT[self.test_name]

    def member(self, names_rows):
        M = np.zeros((len(names_rows), len(self.cls)))
        for i, n in enumerate(names_rows): M[i, self.catset[n]] = 1
        return M


# ---------------------------------------------------------------- linear softmax heads and calibration
def softmax(L): L = L - L.max(1, keepdims=True); E = np.exp(L); return E / E.sum(1, keepdims=True)


def fit_softmax(X, Y, l2, maxiter):
    n, d = X.shape; K = Y.shape[1]
    def f(th):
        W = th[:d * K].reshape(d, K); b = th[d * K:]; L = X @ W + b; L -= L.max(1, keepdims=True); lp = L - np.log(np.exp(L).sum(1, keepdims=True))
        G = (np.exp(lp) * Y.sum(1, keepdims=True) - Y) / n; return float(-(Y * lp).sum() / n + l2 * (W * W).sum()), np.concatenate([(X.T @ G + 2 * l2 * W).ravel(), G.sum(0)])
    r = minimize(f, np.zeros(d * K + K), jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter)); return r.x[:d * K].reshape(d, K), r.x[d * K:], int(r.nit)


def ce(L, Y): L = L - L.max(1, keepdims=True); lp = L - np.log(np.exp(L).sum(1, keepdims=True)); return float(-(Y * lp).sum(1).mean())


def fit_cal(L, Y, free=None, l2b=0.0):
    """tau, bias minimising the soft cross entropy of softmax(L / tau + bias) + l2b |bias|^2; bias = 0 outside `free`"""
    n, K = L.shape; free = np.ones(K, bool) if free is None else np.asarray(free, bool); nf = int(free.sum())
    def f(th):
        t = np.exp(th[0]); b = np.zeros(K); b[free] = th[1:]; M = L / t + b; M = M - M.max(1, keepdims=True); lp = M - np.log(np.exp(M).sum(1, keepdims=True)); G = (np.exp(lp) - Y) / n
        return float(-(Y * lp).sum() / n + l2b * (b * b).sum()), np.concatenate([[float((G * (-L / t)).sum())], (G.sum(0) + 2 * l2b * b)[free]])
    r = minimize(f, np.zeros(nf + 1), jac=True, method='L-BFGS-B', options=dict(maxiter=500)); b = np.zeros(K); b[free] = r.x[1:]
    if l2b == 0 and free.all(): b = b - b.mean()
    return float(np.exp(r.x[0])), b


def cal_probs(L, tau, b): return softmax(L / tau + b)


# ---------------------------------------------------------------- residual model
def noise_model(Zu, Cu):
    from sklearn.covariance import LedoitWolf
    zb, cb = Zu.mean(0), Cu.mean(0); Zd, Cd = Zu - zb, Cu - cb; a = float((Zd * Cd).sum() / (Cd * Cd).sum()); m = zb - a * cb; R = Zu - a * Cu - m
    lw = LedoitWolf(assume_centered=True).fit(R); S = lw.covariance_; d = S.shape[0]
    return dict(a=a, m=m, S=S, L=np.linalg.cholesky(S + 1e-10 * np.eye(d)), iso=float(np.trace(S) / d), n=int(len(Zu)), shrinkage=float(lw.shrinkage_))


def sim_z(nm, Cu, rng, a=None, iso=False):
    g = rng.standard_normal(Cu.shape); r = np.sqrt(nm['iso']) * g if iso else g @ nm['L'].T
    return unit((nm['a'] if a is None else a) * Cu + nm['m'] + r)


# ---------------------------------------------------------------- truths and per-row values
class Truth:
    """one row set of ONE subject: colour (pixel fraction F [n, 11], first colour t1 [n]), category membership M [n, 18]; references
    over the other concepts (concept index g), computed once from the full data (review 2 item 7.1)"""
    def __init__(self, F, t1, M, g, hascat):
        self.F, self.t1, self.M, self.g, self.hc = F, t1, M, np.asarray(g), np.asarray(hascat, bool); n = len(F); G = np.unique(self.g)
        oh = np.eye(11)[t1]; Fs, Os = F.sum(0), oh.sum(0); Ms = M[self.hc].sum(0)
        gF = {k: F[self.g == k].sum(0) for k in G}; gO = {k: oh[self.g == k].sum(0) for k in G}; gn = {k: (self.g == k).sum() for k in G}
        gM = {k: M[(self.g == k) & self.hc].sum(0) for k in G}; gmn = {k: ((self.g == k) & self.hc).sum() for k in G}
        self.Fp = np.stack([(Fs - gF[k]) / (n - gn[k]) for k in self.g]); self.Op = np.stack([(Os - gO[k]) / (n - gn[k]) for k in self.g])
        self.Mp = np.stack([(Ms - gM[k]) / max(self.hc.sum() - gmn[k], 1) for k in self.g])

    def colour(self, pred):
        i = np.arange(len(pred)); return self.F[i, pred], self.Fp[i, pred], (self.t1 == pred).astype(float), self.Op[i, pred]

    def category(self, pred):
        i = np.arange(len(pred)); return self.M[i, pred], self.Mp[i, pred]


def row_values(T, stage, P_):
    """probabilities [n, K] of one subject's rows -> dict(value, ref (Q1 / Q2 excess parts), hit, hitref (binary, for precision), conf, pred)"""
    pred = P_.argmax(1); conf = P_.max(1)
    pv, pr = ((P_ * T.F).sum(1), (P_ * T.Fp).sum(1)) if stage == 'col' else ((P_ * T.M).sum(1), (P_ * T.Mp).sum(1))   # probability version (review 3 item 1)
    if stage == 'col': v, vp, h, hp = T.colour(pred); return dict(value=v, ref=vp, hit=h, hitref=hp, conf=conf, pred=pred, pvalue=pv, pref=pr)
    h, hp = T.category(pred); return dict(value=h, ref=hp, hit=h, hitref=hp, conf=conf, pred=pred, pvalue=pv, pref=pr)


def level_of(conf, hit, hitref, ab):
    """lowest quantile level q (grid) whose said rows (conf >= quantile_q) reach precision >= p and lift >= l; None when none"""
    for q in np.round(np.arange(0, ab['max_level'] + 1e-9, ab['step']), 2):
        t = np.quantile(conf, q); m = conf >= t
        if m.sum() and hit[m].mean() >= ab['precision'] and hit[m].mean() >= ab['lift'] * max(hitref[m].mean(), 1e-12): return float(q)
    return None


class Boot:
    """crossed subjects x concepts pigeonhole as multiplicity weights; one index set for every statistic"""
    def __init__(self, ns, nk, nb, seed):
        rng = np.random.default_rng(seed); self.Cs = np.stack([np.bincount(rng.integers(0, ns, ns), minlength=ns) for _ in range(nb)]).astype(float)
        self.Ck = np.stack([np.bincount(rng.integers(0, nk, nk), minlength=nk) for _ in range(nb)]).astype(float)

    def mean(self, X, m=None):
        m = np.ones(X.shape[1]) if m is None else m.astype(float)
        return np.einsum('bs,sk,bk->b', self.Cs, X, self.Ck * m) / (self.Cs.sum(1) * (self.Ck * m).sum(1))

    def ratio(self, A, B):
        return np.einsum('bs,sk,bk->b', self.Cs, A, self.Ck) / np.maximum(np.einsum('bs,sk,bk->b', self.Cs, B, self.Ck), 1e-12)


def tail(diff):
    """pigeonhole one-sided tail share (1 + #{diff <= 0}) / (1 + n): the resampling distribution, not a null distribution; minimum 1/(n+1)"""
    return float((1 + np.sum(diff <= 0)) / (1 + len(diff)))


def stats(cl, B, catmask):
    """cl: dict stage -> dict of [ns, nk] arrays (value, ref, hit, hitref, said) -> Q1, Q2, Q3"""
    r = {}
    for q, s_, m, vk, rk in (('q1', 'cat', catmask, 'value', 'ref'), ('q2', 'col', None, 'value', 'ref'), ('q1p', 'cat', catmask, 'pvalue', 'pref'), ('q2p', 'col', None, 'pvalue', 'pref')):
        X = cl[s_][vk] - cl[s_][rk]; pt = float(X[:, m].mean()) if m is not None else float(X.mean()); bt = B.mean(X, m)
        r[q] = dict(point=pt, p=tail(bt), lower=float(np.percentile(bt, 5)), upper95=float(np.percentile(bt, 95)), boot=bt)
    sc, sk = cl['col']['said'].astype(float), cl['cat']['said'].astype(float) * catmask[None]; n = sc + sk
    hit = cl['col']['hit'] * sc + cl['cat']['hit'] * sk; prm = cl['col']['hitref'] * sc + cl['cat']['hitref'] * sk; ns = n.sum(); cells_said = float((n > 0).mean())
    if ns > 0:
        pb, qb = B.ratio(hit, n), B.ratio(prm, n)
        r['q3'] = dict(precision=float(hit.sum() / ns), perm=float(prm.sum() / ns), point=float((hit.sum() - prm.sum()) / ns), p=tail(pb - qb), lower=float(np.percentile(pb - qb, 5)),
                       precision_p5=float(np.percentile(pb, 5)), n_statements=int(ns), n_colour=int(sc.sum()), n_category=int(sk.sum()), cells_said=cells_said, boot=pb - qb)
    else: r['q3'] = dict(n_statements=0, cells_said=0.0, point=0.0, p=1.0, lower=0.0, precision_p5=0.0, boot=np.zeros(B.Cs.shape[0]), note='no statement: excess 0')
    return r


def strip(d):
    if isinstance(d, dict): return {str(k): strip(v) for k, v in d.items() if k != 'boot' and not str(k).startswith('_')}
    if isinstance(d, (list, tuple)): return [strip(x) for x in d]
    if isinstance(d, np.ndarray): return d.tolist()
    if isinstance(d, np.floating): return float(d)
    if isinstance(d, np.integer): return int(d)
    if isinstance(d, np.bool_): return bool(d)
    return d


# ---------------------------------------------------------------- read-out
class Readout:
    """heads D[stage][head] (W, b or T) and calibrations; inputs are unit z rows"""
    def __init__(self, D, mu):
        self.D, self.mu = D, mu

    def logits(self, stage, head, Zu):
        d = self.D[stage][head]; return unit(Zu - self.mu) @ d['W'] + d['b'] if head == 'i' else unit(Zu) @ d['T'].T

    def probs(self, stage, head, Zu, cal=None):
        d = self.D[stage][head]; cal = cal or (d['tau'], d['bias']); return cal_probs(self.logits(stage, head, Zu), *cal)


def train_heads(c, st, nm, l2, seed):
    """head (i) of both stages with residual models nm[photo|concept] and penalties l2[stage]"""
    rd = c['readout']; D = {}; rng = np.random.default_rng(seed)
    for s_, lvl in (('col', 'photo'), ('cat', 'concept')):
        rows = st.fit_rows if s_ == 'col' else st.fit_rows[st.has_cat[st.img_name[st.fit_rows]]]
        Y = np.tile(st.frac[rows] if s_ == 'col' else st.Ycat_name[st.img_name[rows]], (rd['noise_copies'], 1))
        X = np.concatenate([unit(sim_z(nm[lvl], st.Craw[rows], rng) - st.mu) for _ in range(rd['noise_copies'])]); W, b, nit = fit_softmax(X, Y, l2[s_], rd['maxiter'])
        D[s_] = dict(W=W, b=b, nit=nit, n_rows=int(len(rows)))
    return D


def choose_l2(c, st, nm, t0):
    rd = c['readout']; tr_f = st.meta.train_img_concept; fc = sorted({tr_f[r] for r in st.fit_rows}); ff = dict(zip(fc, salt_folds(fc, c['fold_salt'], c['n_folds']))); out = {}; rng = np.random.default_rng([rd['noise_seed'], 7])
    for s_, lvl in (('col', 'photo'), ('cat', 'concept')):
        rows = st.fit_rows if s_ == 'col' else st.fit_rows[st.has_cat[st.img_name[st.fit_rows]]]; g = np.tile([ff[tr_f[r]] for r in rows], rd['noise_copies'])
        Y = np.tile(st.frac[rows] if s_ == 'col' else st.Ycat_name[st.img_name[rows]], (rd['noise_copies'], 1))
        X = np.concatenate([unit(sim_z(nm[lvl], st.Craw[rows], rng) - st.mu) for _ in range(rd['noise_copies'])]); cv = []
        for l2 in rd['l2_grid']:
            sc = []
            for f in range(c['n_folds']): W, b, _ = fit_softmax(X[g != f], Y[g != f], l2, rd['maxiter']); sc.append(ce(X[g == f] @ W + b, Y[g == f]))
            cv.append(float(np.mean(sc))); print(f'{s_} l2 {l2:g}: cv ce {cv[-1]:.4f} ({time.time() - t0:.0f}s)', flush=True)
        out[s_] = dict(l2=rd['l2_grid'][int(np.argmin(cv))], cv_ce=cv)
    return out


# ---------------------------------------------------------------- development rows
def dev_outputs(st, key='seed|ses123|photo'):
    if getattr(st, 'smoke', False):                                              # code check: synthetic outputs, no EEG
        rng = np.random.default_rng([7, len(key)]); return {t: unit(0.3 * st.Craw[st.dev_rows] + 0.5 * st.mu + rng.standard_normal((2000, 1024)) / 40) for t in st.subjects}
    od = P(st.pc['infer']['out_dir']); out = {}
    for t in st.subjects:
        Z = np.load(os.path.join(od, f'z_raw_t{t:02d}.npz')); assert np.array_equal(Z['rows'], st.dev_rows); arm, lay, kind = key.split('|')
        out[t] = unit(np.mean([unit(Z[f'seed{j}|{lay}|{kind}']) for j in range(len(st.pc['dev_runs']['seeds']))], 0)) if arm == 'seed' else unit(Z[f'shuffle|{lay}|{kind}'])
    return out


def smoke_ridge(c, st):
    """code check: synthetic ridge / direct-meter files with the real keys and shapes, and their manifest (smoke folder only)"""
    rng = np.random.default_rng(11); man = dict(stage='ridge', smoke=True, subjects={}); od = P(os.path.join(c['cache_dir'], 'smoke')); os.makedirs(od, exist_ok=True)
    for t in st.subjects:
        arr = dict(dev_rows=st.dev_rows)
        for k in ('de', 'time', 'eye', 'pre'):
            arr[f'dev_{k}'] = (0.1 * st.Craw[st.dev_rows] + 0.5 * st.mu + rng.standard_normal((2000, 1024)) / 30).astype(np.float32); arr[f'test_{k}'] = (0.5 * st.mu + rng.standard_normal((200, 1024)) / 30).astype(np.float32)
        for k in ('eye', 'pre'):
            for s_, K in (('col', 11), ('cat', len(st.cls))): arr[f'dev_direct_{k}_{s_}'] = rng.standard_normal((2000, K)).astype(np.float32) * 0.1; arr[f'test_direct_{k}_{s_}'] = rng.standard_normal((200, K)).astype(np.float32) * 0.1
        f = os.path.join(od, f'ridge_t{t:02d}.npz'); np.savez(f, **arr); man['subjects'][str(t)] = dict(file=os.path.relpath(f, ROOT), sha256=fsha(f))
    json.dump(man, open(P(os.path.join(c['out_dir'], 'ridge_manifest.json')), 'w'), indent=1)


def concept_rows(st, Z): return unit(np.stack([Z[st.dev_concept == k].mean(0) for k in range(len(st.dev_c))]))


class DevRows:
    """row sets of the two stages over the 10 subjects (subject-major); colour = photo rows, category = concept rows with a category"""
    def __init__(self, st, Zp, logits_cat=None):
        self.st = st; self.ns = len(st.subjects); kk = st.dev_catk; self.kk = kk
        self.T = dict(col=Truth(st.frac[st.dev_rows], st.top1[st.dev_rows], st.member(st.img_name[st.dev_rows]), st.dev_concept, st.has_cat[st.img_name[st.dev_rows]]),
                      cat=Truth(np.zeros((len(kk), 11)), np.zeros(len(kk), int), st.member(st.dev_name[kk]), kk, np.ones(len(kk), bool)))
        self.n = dict(col=2000, cat=len(kk)); self.fold = dict(col=np.tile(st.dev_fold[st.dev_concept], self.ns), cat=np.tile(st.dev_fold[kk], self.ns))
        self.concept = dict(col=np.tile(st.dev_concept, self.ns), cat=np.tile(kk, self.ns))
        self.Y = dict(col=np.tile(st.frac[st.dev_rows], (self.ns, 1)), cat=np.tile(st.Ycat_name[st.dev_name[kk]], (self.ns, 1)))
        if Zp is not None:
            self.Zp = Zp; self.Zk = {t: concept_rows(st, Zp[t]) for t in st.subjects}
            self.Z = dict(col=np.concatenate([Zp[t] for t in st.subjects]), cat=np.concatenate([self.Zk[t][kk] for t in st.subjects]))

    def values(self, stage, P_):
        n = self.n[stage]; per = [row_values(self.T[stage], stage, P_[i * n:(i + 1) * n]) for i in range(self.ns)]; return {k: np.concatenate([p[k] for p in per]) for k in per[0]}


def cal_rule(c, st, stage):
    return dict(free=None, l2b=0.0) if stage == 'col' else dict(free=st.cat_free, l2b=c['calibration']['cat_l2b'])


def crossfit_cal(c, st, dr, stage, Lg):
    """calibration only (fixed logits): 5-fold cross-fitted probabilities and the full calibration"""
    Pcf = np.zeros_like(Lg)
    for f in range(c['n_folds']):
        tr_, te_ = dr.fold[stage] != f, dr.fold[stage] == f; Pcf[te_] = cal_probs(Lg[te_], *fit_cal(Lg[tr_], dr.Y[stage][tr_], **cal_rule(c, st, stage)))
    return Pcf, fit_cal(Lg, dr.Y[stage], **cal_rule(c, st, stage))


def summary(v, K, names):
    share = np.bincount(v['pred'], minlength=K) / len(v['pred']); nz = share[share > 0]
    return dict(excess=float((v['value'] - v['ref']).mean()), hit=float(v['hit'].mean()), hitref=float(v['hitref'].mean()), n_classes_1pct=int((share >= 0.01).sum()),
                entropy_bits=float(-(nz * np.log2(nz)).sum()), class_share={names[k]: float(share[k]) for k in np.argsort(-share)[:8] if share[k] > 0})


def stage_dev(c, st, a):
    t0 = time.time(); rd, ab = c['readout'], c['abstain']; res = dict(stage='dev', smoke=bool(st.smoke), colour_labels_sha256=st.colour_sha); sub = st.subjects
    Zp = dev_outputs(st); dr = DevRows(st, Zp); ncls = dict(col=11, cat=len(st.cls)); nmz = dict(col=st.cols, cat=st.cls)
    Cp = np.tile(st.Craw[st.dev_rows], (len(sub), 1)); Ck = np.tile(st.Ck_raw, (len(sub), 1)); Zk_all = np.concatenate([dr.Zk[t] for t in sub])
    pf_photo = np.tile(st.dev_fold[st.dev_concept], len(sub)); pf_con = np.tile(st.dev_fold, len(sub))
    def nms(mask_p=None, mask_c=None):
        mp = slice(None) if mask_p is None else mask_p; mc = slice(None) if mask_c is None else mask_c
        return dict(photo=noise_model(dr.Z['col'][mp], Cp[mp]), concept=noise_model(Zk_all[mc], Ck[mc]))
    nm = nms(); res['noise_model'] = {k: dict(a=v['a'], iso_var=v['iso'], n=v['n'], m_norm=float(np.linalg.norm(v['m'])), shrinkage=v['shrinkage']) for k, v in nm.items()}
    l2 = choose_l2(c, st, nm, t0); res['l2'] = l2; ncat = len(st.cls)
    if st.smoke and not glob.glob(P(c['prompt_text_glob'])): T_ = unit(np.random.default_rng(3).standard_normal((ncat + 11, 1024))); pinfo = dict(prompts=[(0, k) for k in st.cls + st.cols]); psha = 'smoke'
    else: T_, pinfo, psha = prompt_text(c)
    assert [p[1] for p in pinfo['prompts']] == st.cls + st.cols, 'prompt order'; res['prompt_text_sha256'] = psha
    proto = dict(cat=dict(T=T_[:ncat]), col=dict(T=T_[ncat:]))
    # cross-fit over the development folds: residual model, head (i), calibration
    Pcf = {s_: {h: np.zeros((len(dr.fold[s_]), ncls[s_])) for h in ('i', 'ii')} for s_ in STAGES}
    for f in range(c['n_folds']):
        nm_f = nms(pf_photo != f, pf_con != f); Hf = train_heads(c, st, nm_f, {s_: l2[s_]['l2'] for s_ in STAGES}, [rd['noise_seed'], f])
        Rf = Readout({s_: dict(i=Hf[s_], ii=proto[s_]) for s_ in STAGES}, st.mu)
        for s_ in STAGES:
            for h in ('i', 'ii'):
                Lg = Rf.logits(s_, h, dr.Z[s_]); tr_, te_ = dr.fold[s_] != f, dr.fold[s_] == f
                Pcf[s_][h][te_] = cal_probs(Lg[te_], *fit_cal(Lg[tr_], dr.Y[s_][tr_], **cal_rule(c, st, s_)))
        print(f'fold {f} done ({time.time() - t0:.0f}s)', flush=True)
    H = train_heads(c, st, nm, {s_: l2[s_]['l2'] for s_ in STAGES}, [rd['noise_seed'], 99]); D = {s_: dict(i=H[s_], ii=dict(proto[s_])) for s_ in STAGES}; R = Readout(D, st.mu)
    for s_ in STAGES:
        for h in ('i', 'ii'): D[s_][h]['tau'], D[s_][h]['bias'] = fit_cal(R.logits(s_, h, dr.Z[s_]), dr.Y[s_], **cal_rule(c, st, s_))
    # head choice, thresholds, gates
    res['heads'] = {}; choice, levels, thr, oof = {}, {}, {}, {}
    for s_ in STAGES:
        res['heads'][s_] = {h: summary(dr.values(s_, Pcf[s_][h]), ncls[s_], nmz[s_]) for h in ('i', 'ii')}
        choice[s_] = 'i' if res['heads'][s_]['i']['excess'] - res['heads'][s_]['ii']['excess'] >= rd['choose_tie'] else 'ii'
        v = dr.values(s_, Pcf[s_][choice[s_]]); fo = dr.fold[s_]
        lv = [level_of(v['conf'][fo == f], v['hit'][fo == f], v['hitref'][fo == f], ab) for f in range(c['n_folds'])]; levels[s_] = lv
        said = np.zeros(len(fo), bool)
        for f in range(c['n_folds']):
            oth = [x for g, x in enumerate(lv) if g != f and x is not None]
            if len(oth) >= ab['min_folds']: m = fo == f; said[m] = v['conf'][m] >= np.quantile(v['conf'][~m], np.mean(oth))   # value from the other folds (review 3 item 2)
        oof[s_] = dict(said=said, hit=v['hit'], hitref=v['hitref'])
        found = [x for x in lv if x is not None]
        thr[s_] = dict(level=float(np.mean(found)), value=float(np.quantile(v['conf'], np.mean(found))), dev_said=float((v['conf'] >= np.quantile(v['conf'], np.mean(found))).mean())) if len(found) >= ab['min_folds'] else None
    rl = c['rules']; gate = {}
    for q, s_ in (('Q1', 'cat'), ('Q2', 'col')):
        hd = res['heads'][s_][choice[s_]]; gate[q] = dict(stage=s_, head=choice[s_], excess=hd['excess'], n_classes=hd['n_classes_1pct'], open=bool(hd['excess'] >= rl['dev_gate'] and hd['n_classes_1pct'] >= rl['dev_min_classes']))
    eff = {s_: (thr[s_]['value'] if thr[s_] and gate['Q1' if s_ == 'cat' else 'Q2']['open'] else None) for s_ in STAGES}
    sa = {s_: oof[s_]['said'] & (eff[s_] is not None) for s_ in STAGES}; n_st = sum(int(sa[s_].sum()) for s_ in STAGES)
    hs = sum(float(oof[s_]['hit'][sa[s_]].sum()) for s_ in STAGES); ps = sum(float(oof[s_]['hitref'][sa[s_]].sum()) for s_ in STAGES)
    q3 = dict(said_share={s_: float(sa[s_].mean()) for s_ in STAGES}, n_statements=n_st, precision=hs / n_st if n_st else None, perm=ps / n_st if n_st else None, excess=(hs - ps) / n_st if n_st else 0.0, by_stage={s_: int(sa[s_].sum()) for s_ in STAGES})
    gate['Q3'] = dict(out_of_fold=q3, thresholds=eff, open=bool(any(v is not None for v in eff.values()) and q3['excess'] >= rl['dev_gate']))
    res.update(choice=choice, levels=levels, thresholds_full=thr, thresholds=eff, gate=gate)
    # controls (decoder calibration) and meters / baselines (their own calibration, review 2 item 1)
    def dec_vals(Zphoto):
        d2 = DevRows(st, Zphoto); return {s_: summary(d2.values(s_, R.probs(s_, choice[s_], d2.Z[s_])), ncls[s_], nmz[s_]) for s_ in STAGES}
    res['controls_dev'] = dict(shuffle=dec_vals(dev_outputs(st, 'shuffle|ses123|photo')), noise=dec_vals(dev_outputs(st, 'seed|ses123|noise')))
    meter_cal = {}; res['meters_dev'] = {}
    if st.smoke: smoke_ridge(c, st)
    if True:
        rm = json.load(open(P(os.path.join(c['out_dir'], 'ridge_manifest.json'))))
        Rr = {}
        for t in sub:
            f_ = P(rm['subjects'][str(t)]['file']); assert fsha(f_) == rm['subjects'][str(t)]['sha256']; Rr[t] = np.load(f_); assert np.array_equal(Rr[t]['dev_rows'], st.dev_rows)
        for mk in METERS:
            res['meters_dev'][mk] = {}
            for s_ in STAGES:
                if mk.startswith('direct_'):
                    k_ = mk.split('_')[1]; per = []
                    for t in sub:
                        Lp = Rr[t][f'dev_direct_{k_}_{s_}'].astype(np.float64); per.append(Lp if s_ == 'col' else np.stack([Lp[st.dev_concept == k].mean(0) for k in range(len(st.dev_c))])[dr.kk])
                    Lg = np.concatenate(per)
                else:
                    Zr = {t: unit(Rr[t][f'dev_{mk}']) for t in sub}; d2 = DevRows(st, Zr); Lg = R.logits(s_, choice[s_], d2.Z[s_])
                Pm, calm = crossfit_cal(c, st, dr, s_, Lg); meter_cal[(mk, s_)] = calm; res['meters_dev'][mk][s_] = summary(dr.values(s_, Pm), ncls[s_], nmz[s_])
    # simulation scales (review 2 item 5): a matched to the development name MRP of each level, per signal and noise model
    Lb = unit(np.load(P(c['blur_file']))[c['blur_key']].astype(np.float64))
    dev_mrp = dict(photo=float(np.mean([cells_mrp(Zp[t] @ st.Tdev.T, st.dev_concept).mean() for t in sub])), concept=float(np.mean([cells_mrp(dr.Zk[t] @ st.Tdev.T, np.arange(200)).mean() for t in sub])))
    sig_dev = dict(semantic=dict(photo=st.Craw[st.dev_rows], concept=st.Ck_raw), blur=dict(photo=Lb[st.dev_rows], concept=unit(np.stack([Lb[st.dev_rows][st.dev_concept == k].mean(0) for k in range(200)]))))
    true = dict(photo=st.dev_concept, concept=np.arange(200)); sims = {}
    for sg, S in sig_dev.items():
        for lvl in ('photo', 'concept'):
            for iso in (False, True):
                rngs = [np.random.default_rng([c['seeds']['signal'], 900 + k]) for k in range(c['sim_calib_draws'])]
                G = [(r.standard_normal(S[lvl].shape)) for r in rngs]
                def mrp(av):
                    vals = []
                    for g in G:
                        r = np.sqrt(nm[lvl]['iso']) * g if iso else g @ nm[lvl]['L'].T; Z = unit(av * S[lvl] + nm[lvl]['m'] + r); vals.append(cells_mrp(Z @ st.Tdev.T, true[lvl]).mean())
                    return float(np.mean(vals))
                key = f'{sg}|{lvl}|{"iso" if iso else "resid"}'; m0 = float(cells_mrp(unit(S[lvl]) @ st.Tdev.T, true[lvl]).mean())
                if m0 <= dev_mrp[lvl]:                                           # the noiseless signal is below the target: deterministic simulation (as D-115)
                    sims[key] = dict(a=None, noiseless=True, name_mrp=m0, target=dev_mrp[lvl]); print(f'sim scale {key}: noiseless {m0:.4f} <= target {dev_mrp[lvl]:.4f}', flush=True); continue
                lo, hi = 0.0, max(1.0, 4 * abs(nm[lvl]['a']))
                while mrp(hi) < dev_mrp[lvl] and hi < 1e3: hi *= 2
                for _ in range(c['sim_bisect']):
                    mid = (lo + hi) / 2
                    if mrp(mid) < dev_mrp[lvl]: lo = mid
                    else: hi = mid
                av = (lo + hi) / 2
                Zs = [unit(av * S[lvl] + nm[lvl]['m'] + (np.sqrt(nm[lvl]['iso']) * G[0] if iso else G[0] @ nm[lvl]['L'].T))]
                img_true = st.Ck_raw if lvl == 'concept' else st.Ck_raw[st.dev_concept]
                sims[key] = dict(a=av, noiseless=False, noiseless_mrp=m0, name_mrp=mrp(av), target=dev_mrp[lvl], image_mrp=float(cells_mrp(Zs[0] @ st.Ck_raw.T, true[lvl]).mean()))
                print(f'sim scale {key}: a {av:.4f} mrp {sims[key]["name_mrp"]:.4f} target {dev_mrp[lvl]:.4f}', flush=True)
    res['dev_image_mrp'] = dict(photo=float(np.mean([cells_mrp(Zp[t] @ st.Ck_raw.T, st.dev_concept).mean() for t in sub])), concept=float(np.mean([cells_mrp(dr.Zk[t] @ st.Ck_raw.T, np.arange(200)).mean() for t in sub])))
    res['sim_scales'] = sims; res['dev_name_mrp'] = dev_mrp; res['readout_cv_l2'] = l2
    # save
    out = P(c['out_dir']); os.makedirs(out, exist_ok=True); os.makedirs(P(c['cache_dir']), exist_ok=True)
    nf = P(os.path.join(c['cache_dir'], 'noise_smoke.npz' if st.smoke else 'noise.npz')); np.savez(nf, **{f'{k}_{x}': v[x] for k, v in nm.items() for x in ('m', 'S')}, **{f'{k}_a': v['a'] for k, v in nm.items()})
    res['noise_file'] = dict(file=os.path.relpath(nf, ROOT), sha256=fsha(nf)); rf = os.path.join(out, 'readout.npz')
    arr = {f'{s_}_{h}_{k}': np.asarray(v) for s_ in STAGES for h in ('i', 'ii') for k, v in D[s_][h].items() if k in ('W', 'b', 'T', 'tau', 'bias')}
    arr.update({f'meter_{mk}_{s_}_tau': np.asarray(v[0]) for (mk, s_), v in meter_cal.items()}); arr.update({f'meter_{mk}_{s_}_bias': v[1] for (mk, s_), v in meter_cal.items()})
    np.savez(rf, **arr); res['readout_file'] = dict(file=os.path.relpath(rf, ROOT), sha256=fsha(rf)); res['seconds'] = time.time() - t0; res['peak_rss_gb'] = peak_gb()
    json.dump(strip(res), open(os.path.join(out, 'dev.json'), 'w'), indent=1)
    print(json.dumps(strip(dict(choice=choice, levels=levels, thresholds=eff, gate=gate, heads={s_: {h: v['excess'] for h, v in res['heads'][s_].items()} for s_ in STAGES}))), flush=True)


def prompt_text(c):
    d = sorted(glob.glob(P(c['prompt_text_glob']))); assert d, 'prompt_text.npy not fetched'; info = json.load(open(os.path.join(os.path.dirname(d[-1]), 'prompts.json')))
    return np.load(d[-1]).astype(np.float64), info, fsha(d[-1])


def load_readout(c, st):
    out = P(c['out_dir']); dv = json.load(open(os.path.join(out, 'dev.json'))); rf = os.path.join(out, 'readout.npz'); assert fsha(rf) == dv['readout_file']['sha256']
    Z = np.load(rf); D = {s_: {'i': {}, 'ii': {}} for s_ in STAGES}; mc = {}
    for k in Z.files:
        if k.startswith('meter_'):
            _, rest = k.split('_', 1); mk, s_, x = rest.rsplit('_', 2); mc.setdefault((mk, s_), {})[x] = Z[k] if Z[k].ndim else float(Z[k]); continue
        s_, h, x = k.split('_', 2); D[s_][h][x] = Z[k] if Z[k].ndim else float(Z[k])
    nz = np.load(P(dv['noise_file']['file'])); assert fsha(P(dv['noise_file']['file'])) == dv['noise_file']['sha256']; nm = {}
    for lvl in ('photo', 'concept'):
        S = nz[f'{lvl}_S']; d = S.shape[0]; nm[lvl] = dict(a=float(nz[f'{lvl}_a']), m=nz[f'{lvl}_m'], S=S, L=np.linalg.cholesky(S + 1e-10 * np.eye(d)), iso=float(np.trace(S) / d))
    return Readout(D, st.mu), dv, nm, {k: (v['tau'], v['bias']) for k, v in mc.items()}


# ---------------------------------------------------------------- test structure
def test_truth(st): return Truth(st.frac[st.test_rows], st.top1[st.test_rows], st.member(st.test_name), np.arange(200), st.has_cat[st.test_name])


def test_cells(T, Ps, thr):
    """Ps: list per subject of {stage: probabilities [200, K]} -> {stage: dict of [ns, 200] arrays incl. said}"""
    out = {}
    for s_ in STAGES:
        per = [row_values(T, s_, p[s_]) for p in Ps]; d = {k: np.stack([x[k] for x in per]) for k in per[0]}
        d['said'] = (d['conf'] >= thr[s_]) if thr[s_] is not None else np.zeros_like(d['conf'], bool); out[s_] = d
    return out


def probs_of(R, heads, Zus, cal=None):
    return [{s_: R.probs(s_, heads[s_], Zu, None if cal is None else cal[s_]) for s_ in STAGES} for Zu in Zus]


# ---------------------------------------------------------------- sim (no EEG)
def stage_sim(c, st, a):
    t0 = time.time(); R, dv, nm, _ = load_readout(c, st); thr, heads = dv['thresholds'], dv['choice']; T = test_truth(st); rl = c['rules']; ns = len(st.subjects)
    catmask = st.has_cat[st.test_name]; B = Boot(ns, 200, c['n_boot'], c['seeds']['boot']); res = dict(stage='sim', smoke=bool(st.smoke), dev_sha256=fsha(P(os.path.join(c['out_dir'], 'dev.json'))))
    def judge(r):
        return dict(Q1=bool(r['q1']['p'] <= rl['alpha'] and r['q1']['point'] >= rl['min_point']), Q2=bool(r['q2']['p'] <= rl['alpha'] and r['q2']['point'] >= rl['min_point']),
                    Q3=bool(r['q3']['cells_said'] >= rl['q3_min_cells'] and r['q3']['p'] <= rl['alpha'] and r['q3']['precision_p5'] >= rl['q3_floor']))
    def run(make, n, seed, tag):
        rec = []
        for b in range(n):
            rng = np.random.default_rng([seed, b]); cl = test_cells(T, probs_of(R, heads, make(rng)), thr); r = stats(cl, B, catmask)
            rec.append(dict(judge(r), q1=r['q1']['point'], q2=r['q2']['point'], q3=r['q3']['point'], q3_prec=r['q3'].get('precision', 0.0), said=r['q3']['cells_said']))
        o = dict(n=n, pass_rate={q: float(np.mean([x[q] for x in rec])) for q in ('Q1', 'Q2', 'Q3')})
        for q in ('q1', 'q2', 'q3', 'q3_prec', 'said'):
            v = np.array([x[q] for x in rec]); o[q] = dict(mean=float(v.mean()), p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)))
        print(f'{tag}: {o["pass_rate"]} q1 {o["q1"]["mean"]:.4f} q2 {o["q2"]["mean"]:.4f} ({time.time() - t0:.0f}s)', flush=True); return o
    d = st.Craw.shape[1]
    res['info0'] = dict(gauss=run(lambda rng: [unit(rng.standard_normal((200, d))) for _ in range(ns)], c['n_sim'], c['seeds']['info0'], 'info0 gauss'),
                        const=run(lambda rng: [np.tile(unit(rng.standard_normal((1, d))), (200, 1)) for _ in range(ns)], c['n_sim'], c['seeds']['info0'] + 1, 'info0 const'))
    res['info0_ok'] = bool(all(v <= rl['info0_max'] for x in res['info0'].values() for v in x['pass_rate'].values()))
    Lb = unit(np.load(P(c['blur_file']))[c['blur_key']].astype(np.float64)); sig = dict(semantic=st.Craw[st.test_rows], blur=Lb[st.test_rows]); res['signal'] = {}
    for j, (sg, S) in enumerate(sig.items()):
        for lvl in ('photo', 'concept'):
            for iso in (False, True):
                key = f'{sg}|{lvl}|{"iso" if iso else "resid"}'; sc = dv['sim_scales'][key]; av = sc['a']
                if sc['noiseless']: res['signal'][key] = dict(run(lambda rng, S=S: [unit(S) for _ in range(ns)], 1, 0, key), note='noiseless signal, judged once'); continue
                res['signal'][key] = run(lambda rng, S=S, lvl=lvl, iso=iso, av=av: [sim_z(nm[lvl], S, rng, a=av, iso=iso) for _ in range(ns)], c['n_sim'],
                                         c['seeds']['signal'] + 10 * j + 2 * ('photo', 'concept').index(lvl) + int(iso), key)
    res['blur_p95'] = {lvl: {q: res['signal'][f'blur|{lvl}|resid'][q]['p95'] for q in ('q1', 'q2', 'q3')} for lvl in ('photo', 'concept')}
    def single(Zu):
        cl = test_cells(T, probs_of(R, heads, [Zu]), thr); return dict(Q2=float((cl['col']['value'] - cl['col']['ref']).mean()), Q1=float((cl['cat']['value'] - cl['cat']['ref'])[:, catmask].mean()), col_top1_hit=float(cl['col']['hit'].mean()))
    res['eegfree'] = dict(original=single(st.Craw[st.test_rows]), blur=single(Lb[st.test_rows]))
    res['colour_gate'] = dict(value=res['eegfree']['original']['Q2'], min=rl['colour_gate'], open=bool(res['eegfree']['original']['Q2'] >= rl['colour_gate']))
    import re
    pool = list(csv.DictReader(open(P(os.path.join(st.lc['pool']['out_dir'], 'pool.tsv')), encoding='utf-8'), delimiter='\t')); words = {w: i for i, w in enumerate(st.cols)}; words['gray'] = st.cols.index('grey'); cnt = {}
    for r in pool:
        for w in re.findall(r'[a-z]+', r['sentence'].lower()):
            if w in words: cnt.setdefault(r['things_id'], np.zeros(11))[words[w]] += 1
    tf = [st.names[n]['things_id'] for n in st.test_name]; pred = np.array([int(np.argmax(cnt[f])) if f in cnt else -1 for f in tf]); ok = pred >= 0   # pool rows use the THINGS id
    v, vp, h, hp = T.colour(np.where(ok, pred, 0)); res['typical_colour'] = dict(n_with_word=int(ok.sum()), Q2_excess_on_these=float((v - vp)[ok].mean()), top1_hit=float(h[ok].mean()), top1_perm=float(hp[ok].mean()))
    res['seconds'] = time.time() - t0; res['peak_rss_gb'] = peak_gb(); json.dump(strip(res), open(P(os.path.join(c['out_dir'], 'sim.json')), 'w'), indent=1)
    print(json.dumps(strip(dict(info0_ok=res['info0_ok'], colour_gate=res['colour_gate'], blur_p95=res['blur_p95'], eegfree=res['eegfree'], typical=res['typical_colour']))), flush=True)


# ---------------------------------------------------------------- C2 inputs (server CPU, after dev)
def stage_c2prep(c, st, a):
    from fv_ridge import ridge_cv
    t0 = time.time(); R, dv, nm, _ = load_readout(c, st); cc = c['c2']; lc = st.lc
    pool = list(csv.DictReader(open(P(os.path.join(lc['pool']['out_dir'], 'pool.tsv')), encoding='utf-8'), delimiter='\t'))
    pt = sorted(glob.glob(P(cc['pool_text_glob']))); assert pt; E = np.load(pt[-1]).astype(np.float64); assert len(E) == len(pool)
    tid = {r['folder']: r['things_id'] for r in st.names if r['split'] == 'train'}; assert len(set(tid.values())) == len(tid)   # pool rows use the THINGS id, not the folder
    ic = np.array(st.meta.train_img_concept); dev = {tid[f] for f in st.dev_c}; cm = {tid[f]: unit(st.Craw[np.flatnonzero(ic == f)].mean(0)) for f in st.meta.train_concepts}
    fit = np.array([r['split'] == 'train' and r['things_id'] not in dev for r in pool]); Y = np.stack([cm[r['things_id']] for r, f in zip(pool, fit) if f])
    folders = sorted({r['things_id'] for r, f in zip(pool, fit) if f}); fo = dict(zip(folders, salt_folds(folders, c['fold_salt'], c['n_folds'])))
    pred, alpha, cv = ridge_cv(E[fit].astype(np.float32), Y.astype(np.float32), np.array([fo[r['things_id']] for r, f in zip(pool, fit) if f])); X = unit(pred(E.astype(np.float32)))
    Zp = dev_outputs(st); dev_z = np.stack([concept_rows(st, Zp[t]) for t in st.subjects]).astype(np.float32); test = {}
    for cn in cc['test_conditions']:
        rows = LR.rows(cn, lc, ROOT, st.subjects); test['test_' + cn.replace('|', '+')] = np.stack([unit(np.stack([x[2] for x in rows if x[0] == s])) for s in st.subjects]).astype(np.float32)
    m = nm['concept']; f = P(cc['inputs'])
    np.savez(f, sent_x=X.astype(np.float32), sentences=np.array([r['sentence'] for r in pool]), things_id=np.array([r['things_id'] for r in pool]), a=m['a'], m=m['m'].astype(np.float32),
             L=m['L'].astype(np.float32), mu=st.mu.astype(np.float32), dev_z=dev_z, subjects=np.array(st.subjects), **test)
    h = fsha(f); open(f + '.sha256', 'w').write(h + '\n'); te = np.array([r['split'] == 'test' for r in pool]); tm = {st.names[n]['things_id']: st.Craw[16540 + k] for k, n in enumerate(st.test_name)}
    res = dict(stage='c2prep', inputs=os.path.relpath(f, ROOT), sha256=h, ridge_alpha=alpha, ridge_cv=cv, n_fit=int(fit.sum()), pool_text=os.path.relpath(pt[-1], ROOT), pool_text_sha256=fsha(pt[-1]),
               test_cos=float(np.mean([X[i] @ tm[pool[i]['things_id']] for i in np.flatnonzero(te)])), conditions=list(test), seconds=time.time() - t0)
    json.dump(res, open(P(os.path.join(c['out_dir'], 'c2prep.json')), 'w'), indent=1); print(json.dumps(res), flush=True)


# ---------------------------------------------------------------- verdict (once)
def stage_verdict(c, st, a):
    out = P(c['out_dir']); vf = os.path.join(out, 'verdict.json'); assert not os.path.exists(vf), 'the verdict is computed once'
    t0 = time.time(); R, dv, nm, mcal = load_readout(c, st); sim = json.load(open(os.path.join(out, 'sim.json'))); assert sim['dev_sha256'] == fsha(os.path.join(out, 'dev.json'))
    assert sim['info0_ok'], 'information-zero simulation failed: stop'
    thr, heads = dv['thresholds'], dv['choice']; T = test_truth(st); rl = c['rules']; sub = st.subjects; ns = len(sub); catmask = st.has_cat[st.test_name]; B = Boot(ns, 200, c['n_boot'], c['seeds']['boot'])
    def zus(cn):
        if st.smoke:                                                             # code check: synthetic test outputs, the test z is never read
            rng = np.random.default_rng([5, len(cn)]); sig = 0.3 if cn.startswith('dec') else 0.0
            return [unit(sig * st.Craw[st.test_rows] + 0.5 * st.mu + rng.standard_normal((200, 1024)) / 40) for _ in sub]
        rows = LR.rows(cn, st.lc, ROOT, sub); Z = []
        for s in sub:
            r = [x for x in rows if x[0] == s]
            if len(r) == 1 and r[0][1] == -1: Z.append(np.tile(unit(r[0][2]), (200, 1)))
            else: assert [x[1] for x in r] == list(range(200)); Z.append(unit(np.stack([x[2] for x in r])))
        return Z
    if st.smoke:
        rng = np.random.default_rng(13); np.savez(os.path.join(out, 'c2_cells.npz'), dev_gate_open=True, step=1000, q4=rng.standard_normal((ns, 200)) * 0.1 + 0.05,
                                                  **{f'q4_{k}': rng.standard_normal((ns, 200)) * 0.1 for k in ('shuf_ses123_avg20', 'dec_ses123_noise', 'dec_ses123_phase', 'noeeg', 'dec_k0_avg20')})
    res = dict(stage='verdict', smoke=bool(st.smoke), dev_sha256=fsha(os.path.join(out, 'dev.json')), sim_sha256=fsha(os.path.join(out, 'sim.json')), choice=heads, thresholds=thr, gates=dv['gate'], colour_gate=sim['colour_gate'])
    # 1. family from label-free counts (review 2 item 7.4): statements of the main condition before any hit is computed
    thr = dict(thr)
    if not sim['colour_gate']['open']: thr['col'] = None                       # review 3 item 3: no colour statement when the EEG-free colour gate is closed
    Zmain = zus('dec|ses123|avg20'); Pm = probs_of(R, heads, Zmain)
    said = np.stack([sum(((p[s_].max(1) >= thr[s_]) & (catmask if s_ == 'cat' else True)) if thr[s_] is not None else np.zeros(200, bool) for s_ in STAGES) > 0 for p in Pm])
    open_q = dict(Q1=dv['gate']['Q1']['open'], Q2=dv['gate']['Q2']['open'] and sim['colour_gate']['open'], Q3=dv['gate']['Q3']['open'] and float(said.mean()) >= rl['q3_min_cells'])
    res['thresholds_applied'] = thr; res['q3_cells_said'] = float(said.mean()); res['said_share_test'] = {s_: float(np.mean([(p[s_].max(1) >= thr[s_]) for p in Pm])) if thr[s_] is not None else 0.0 for s_ in STAGES}
    res['said_share_dev'] = {s_: (dv['thresholds_full'][s_] or {}).get('dev_said') for s_ in STAGES}
    c2f = os.path.join(out, 'c2_cells.npz'); z4 = np.load(c2f) if os.path.exists(c2f) else None; open_q['Q4'] = bool(z4['dev_gate_open']) if z4 is not None else False; res['open'] = dict(open_q)
    # 2. baselines and meters (their own calibration), embedding and direct versions
    rm = json.load(open(os.path.join(out, 'ridge_manifest.json'))); Rr = {}
    for s in sub:
        f_ = P(rm['subjects'][str(s)]['file']); assert fsha(f_) == rm['subjects'][str(s)]['sha256']; Rr[s] = np.load(f_)
    meters = {}
    for mk in METERS:
        if mk.startswith('direct_'):
            k_ = mk.split('_')[1]; Pm_ = [{s_: cal_probs(Rr[s][f'test_direct_{k_}_{s_}'].astype(np.float64), *mcal[(mk, s_)]) for s_ in STAGES} for s in sub]
        else: Pm_ = probs_of(R, heads, [unit(Rr[s][f'test_{mk}']) for s in sub], {s_: mcal[(mk, s_)] for s_ in STAGES})
        cl = test_cells(T, Pm_, thr); meters[mk] = stats(cl, B, catmask); meters[mk]['class_entropy_bits'] = {s_: float(-(lambda p: (p[p > 0] * np.log2(p[p > 0])).sum())(np.bincount(cl[s_]['pred'].ravel(), minlength=11 if s_ == 'col' else 18) / cl[s_]['pred'].size)) for s_ in STAGES}
    # 3. decoder and controls
    main = stats(test_cells(T, Pm, thr), B, catmask); main_cells = test_cells(T, Pm, thr)
    ctl = {k: stats(test_cells(T, probs_of(R, heads, zus(k)), thr), B, catmask) for k in ('shuf|ses123|avg20', 'dec|ses123|noise', 'dec|ses123|phase', 'noeeg')}
    def ruleN(v, e): return dict(excess=v['point'], lower=v['lower'], bound=rl['control_ratio'] * e, passed=bool(v['lower'] <= 0 and v['point'] <= rl['control_ratio'] * e))
    prem = {q: {k: ruleN(v[q], main[q]['point']) for k, v in ctl.items()} for q in ('q1', 'q2', 'q3')}
    for q in ('q1', 'q2'):                                                      # argmax and probability versions (review 3 item 1)
        for mk in ('eye', 'pre', 'direct_eye', 'direct_pre'):
            prem[q][f'meter_{mk}'] = ruleN(meters[mk][q], main[q]['point']); prem[q][f'meter_{mk}_prob'] = ruleN(meters[mk][q + 'p'], main[q + 'p']['point'])
    res['meter_report'] = {mk: dict(tau={s_: float(mcal[(mk, s_)][0]) for s_ in STAGES}, entropy_bits=meters[mk]['class_entropy_bits']) for mk in METERS}
    q4 = None
    if z4 is not None:
        X4 = z4['q4']; bt = B.mean(X4); q4 = dict(point=float(X4.mean()), p=tail(bt), lower=float(np.percentile(bt, 5)), step=int(z4['step']))
        prem['q4'] = {k: ruleN(dict(point=float(z4[k].mean()), lower=float(np.percentile(B.mean(z4[k]), 5))), q4['point']) for k in z4.files if k.startswith('q4_') and k != 'q4_dec_k0_avg20'}
    res['premises'] = prem
    # 4. Holm over the open hypotheses (the Q3 precision floor is a further condition outside Holm)
    pv = {q: (main[q.lower()]['p'] if q != 'Q4' else q4['p']) for q, o in open_q.items() if o}; order = sorted(pv, key=lambda q: pv[q]); adj, run_max = {}, 0.0
    for i, q in enumerate(order): run_max = max(run_max, min(1.0, (len(order) - i) * pv[q])); adj[q] = run_max
    hyp = {}
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        if not open_q[q]: hyp[q] = 'not tested (development gate, colour gate, statement count or no C2 cells)'; continue
        r = main[q.lower()] if q != 'Q4' else q4; pr = all(v['passed'] for k, v in prem[q.lower()].items() if not k.startswith('_')); ok = adj[q] <= rl['alpha'] and pr
        if q in ('Q1', 'Q2'): ok = ok and r['point'] >= rl['min_point']
        if q == 'Q3': ok = ok and r['precision_p5'] >= rl['q3_floor']
        hyp[q] = dict(strip(r), p_holm=adj[q], premises_passed=bool(pr), adopted=bool(ok))
        if q != 'Q4':
            b = {lvl: sim['blur_p95'][lvl][q.lower()] for lvl in ('photo', 'concept')}; ab_ = [r['point'] > v for v in b.values()]
            hyp[q]['blur_rule'] = dict(blur_p95=b, reading='larger than the blur simulation' if all(ab_) else ('explainable by blur-level information' if not any(ab_) else 'undetermined'))
    res['hypotheses'] = hyp; res['main'] = strip(main); res['controls'] = strip(ctl); res['meters'] = strip(meters); res['q4'] = q4
    # 5. report only
    rep = {cn: strip(stats(test_cells(T, probs_of(R, heads, zus(cn)), thr), B, catmask)) for cn in ('dec|k0|avg20', 'dec|ses1|avg20', 'dec|ses12|avg20', 'dec|ses123|single1')}
    def dist(p, names):
        s = np.bincount(p, minlength=len(names)) / len(p); return dict(share={names[k]: float(s[k]) for k in np.argsort(-s) if s[k] > 0}, entropy_bits=float(-(s[s > 0] * np.log2(s[s > 0])).sum()))
    rep['class_distribution'] = dict(colour=dist(main_cells['col']['pred'].ravel(), st.cols), category=dist(main_cells['cat']['pred'][:, catmask].ravel(), st.cls))
    M = st.member(st.test_name); pk = main_cells['cat']['pred']; n_test = M[catmask].sum(0)
    rep['category_per_class'] = {st.cls[k]: dict(n_test_concepts=int(n_test[k]), few=bool(n_test[k] < 3), n_pred=int(((pk == k) & catmask).sum()),
                                                 precision=float(main_cells['cat']['hit'][(pk == k) & catmask].mean()) if ((pk == k) & catmask).any() else None, base_rate=float(M[catmask, k].mean()),
                                                 recall=float(((pk == k) & (M[:, k][None] > 0)).sum() / max(1, ns * n_test[k]))) for k in range(len(st.cls))}
    chrom = ~np.isin(st.top1[st.test_rows], [0, 3, 9]); q2c = main_cells['col']['value'] - main_cells['col']['ref']; rep['q2_split'] = dict(achromatic=float(q2c[:, ~chrom].mean()), chromatic=float(q2c[:, chrom].mean()))
    from math import comb
    rep['closed_name_choice'] = {}
    for N in (2, 5, 20, 200):
        acc = [np.mean([comb(199 - m, N - 1) / comb(199, N - 1) for m in (S > S[np.arange(200), np.arange(200)][:, None]).sum(1)]) for S in [Z @ st.Ttest.T for Z in Zmain]]
        rep['closed_name_choice'][str(N)] = dict(accuracy=float(np.mean(acc)), chance=1 / N)
    rep['notes'] = ['development outputs: 2 development-model seeds; test outputs: 3 final-model seeds', 'colour names: van de Weijer et al. 2009 w2c table, author-released research data, no license stated, not redistributed']
    res['report'] = rep; res['seconds'] = time.time() - t0; res['peak_rss_gb'] = peak_gb()
    json.dump(strip(res), open(vf, 'w'), indent=1); print(json.dumps(strip(dict(open=open_q, hypotheses=hyp))), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', choices=['dev', 'sim', 'c2prep', 'verdict'], required=True)
    ap.add_argument('--smoke', action='store_true', help='code check with synthetic development outputs, written to out_dir/smoke (never reported)'); a = ap.parse_args()
    c = yaml.safe_load(open(P(a.config))); st = Setup(c); st.smoke = a.smoke
    if a.smoke:
        assert a.stage in ('dev', 'sim', 'verdict'), 'smoke: dev, sim and verdict (synthetic inputs only)'
        if a.stage == 'verdict' and os.path.exists(P(os.path.join(c['out_dir'], 'smoke', 'verdict.json'))): os.remove(P(os.path.join(c['out_dir'], 'smoke', 'verdict.json')))
        import shutil; c['out_dir'] = os.path.join(c['out_dir'], 'smoke'); os.makedirs(P(c['out_dir']), exist_ok=True); shutil.copy(P('results/fv_s/colour_labels.npz'), P(c['out_dir']))
        c['n_sim'] = 3; c['readout']['l2_grid'] = c['readout']['l2_grid'][:2]
    {'dev': stage_dev, 'sim': stage_sim, 'c2prep': stage_c2prep, 'verdict': stage_verdict}[a.stage](c, st, a)


if __name__ == '__main__':
    main()
