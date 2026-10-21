"""F-XI (D-121): cross-modal transfer of concept information in ds004306. Server CPU (gpuq).

Within subject (sessions pooled; features z-scored inside session x condition without labels). For each test modality m, a reader is
trained on the perception trials of the other two modalities and tested on m:
  H1  perception m        amplitude features (124), shrinkage LDA                         (arm M)
  H2  imagination after m band log-variance (620) -> PCA 64 fitted on the training trials -> shrinkage LDA
Shrinkage LDA: class means, pooled within-class covariance with Ledoit-Wolf shrinkage, equal priors.
Excess = accuracy - mean accuracy of n_perm readers trained on labels permuted inside session x modality strata. Subject value =
mean over the three test modalities; group G = mean over subjects. Null: G_k from permutation k; p = (1 + #{G_k >= G}) / (K + 1).
Holm over H1, H2; adopt when Holm p <= alpha and G >= sesoi. Subject bootstrap 95% intervals.
Meters (E eog, P pre-stimulus, T onset-time kNN, G GFP), controls (label-shuffle model, noise input, phase randomised, no EEG), the DE+LR
baseline (written first), report-only items (pairs, within-modality CV, temporal generalisation, abstaining sentences, VVIQ).
Stages:
  smoke   synthetic sessions (no data): pipeline, planted-signal recovery, timing
  sim     real features with random labels: information-zero pass rate, power over planted shared patterns -> sim.json
  verdict once, true labels -> verdict.json (+ baseline file)
usage: python scripts/sem_verdict.py configs/sem_ds004306.yaml --stage {smoke,sim,verdict} [--workers 8]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'                                    # one BLAS thread per worker process; workers <= ESE_NUM_THREADS
import sys, csv, json, time, hashlib, argparse, yaml, numpy as np
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
NC = 3


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------- readers
def lw_cov(X):
    """Ledoit-Wolf shrinkage of the covariance of already centred rows X (n x d)."""
    n, d = X.shape; S = X.T @ X / n; mu = np.trace(S) / d
    X2 = X ** 2; b2 = ((X2.T @ X2) / n - S ** 2).sum() / n
    d2 = ((S - mu * np.eye(d)) ** 2).sum(); sh = min(b2, d2) / d2 if d2 > 0 else 1.0
    return (1 - sh) * S + sh * mu * np.eye(d)


def lda_fit(X, y):
    M = np.stack([X[y == k].mean(0) for k in range(NC)]); C = lw_cov(X - M[y])
    W = np.linalg.solve(C, M.T)                             # d x NC
    return W, -0.5 * (M * W.T).sum(1)


def lda_score(model, X):
    W, b = model; return X @ W + b


def lr_fit(X, y, C):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=2000).fit(X, y); return m.coef_.T, m.intercept_


def knn_time_pred(t_tr, y_tr, t_te, k=5):
    d = np.abs(t_te[:, None] - t_tr[None]); idx = np.argsort(d, 1)[:, :k]
    v = np.zeros((len(t_te), NC))
    for j in range(k):
        np.add.at(v, (np.arange(len(t_te)), y_tr[idx[:, j]]), 1.0 + 1e-3 * (k - j))   # ties -> nearer
    return v.argmax(1)


def perm_labels(y, strata, rng):
    yp = y.copy()
    for s in np.unique(strata):
        i = np.flatnonzero(strata == s); yp[i] = y[rng.permutation(i)]
    return yp


def pca_fit(X, k):
    mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False); return mu, Vt[:k].T


# ------------------------------------------------------------------------------------------------------------ data access
def sid(s):
    return int(s.split('-')[1])                             # 'sub-016' and 'sub-16' -> 16


def zs(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def load_subjects(c, smoke=False):
    """-> {sub: dict of arrays (trials pooled over sessions)}; perception rows and imagination rows share the trial index."""
    pj = json.load(open(os.path.join(P(c['cache_dir']), 'prep.json')))
    subs = {}
    for m in pj['sessions']:
        if m.get('missing'):
            continue
        f = P(m['file']); assert fsha(f) == m['sha256'], f
        z = dict(np.load(f)); n = len(z['concept'])
        keep = np.ones(n, bool)
        for cd in 'pi':
            for mi in range(3):
                if ((z['modality'] == mi) & z[f'{cd}_ok']).sum() < c['min_trials_per_condition']:
                    z[f'{cd}_ok'] = z[f'{cd}_ok'] & (z['modality'] != mi)
        rec = dict(concept=z['concept'], modality=z['modality'], block=z['block'], onset=z['onset'], p_ok=z['p_ok'], i_ok=z['i_ok'],
                   ses=np.full(n, len(subs.get(m['sub'], {}).get('ses_ids', []))))
        for k in ('p_amp', 'p_de', 'p_pre', 'p_gfp', 'p_eog_amp', 'p_eog_de', 'p_amp_ph', 'p_de_ph',
                  'i_amp', 'i_de', 'i_eog_amp', 'i_eog_de', 'i_amp_ph', 'i_de_ph'):
            X = z[k].astype(np.float64); ok = z[f'{k[0]}_ok']
            mu, sd = X[ok].mean(0), X[ok].std(0) + 1e-8        # session x condition z-score (no labels); phase versions use their own
            rec[k] = (X - mu) / sd
        rec['p_ep'] = z['p_ep']; rec['p_t'] = z['p_t']
        S = subs.setdefault(m['sub'], dict(ses_ids=[], parts=[])); S['ses_ids'].append(m['ses']); S['parts'].append(rec)
    out = {}
    for s, S in subs.items():
        out[s] = {k: np.concatenate([r[k] for r in S['parts']]) for k in S['parts'][0] if k not in ('p_t',)}
        out[s]['p_t'] = S['parts'][0]['p_t']; out[s]['block_id'] = out[s]['ses'] * 100 + out[s]['block']
    return out


# ------------------------------------------------------------------------------------------------------------- core runs
def transfer(D, y, feat_tr, feat_te, cond_te, reader, n_perm, rng, extra_te=(), pca=None, C=1.0):
    """For each test modality m: train on perception rows of the other modalities, test on cond_te rows of m.
    Returns acc[m], perm[m, k] for the main test features and for each extra test feature key."""
    res = {'main': (np.zeros(3), np.zeros((3, n_perm)))}
    for e in extra_te:
        res[e] = (np.zeros(3), np.zeros((3, n_perm)))
    for m in range(3):
        tr = np.flatnonzero(D['p_ok'] & (D['modality'] != m)); te = np.flatnonzero(D[f'{cond_te}_ok'] & (D['modality'] == m))
        Xtr = D[feat_tr][tr]; strata = D['ses'][tr] * 10 + D['modality'][tr]
        tests = {'main': D[feat_te][te]}
        for e in extra_te:
            tests[e] = e_feat(D, e, te, Xtr, rng)
        if pca is not None:
            mu, V = pca_fit(Xtr, pca); Xtr = (Xtr - mu) @ V; tests = {k: (v - mu) @ V for k, v in tests.items()}
        yt = D['concept'][te]
        for k in range(n_perm + 1):
            yy = y[tr] if k == 0 else perm_labels(y[tr], strata, rng)
            mdl = lda_fit(Xtr, yy) if reader == 'lda' else lr_fit(Xtr, yy, C)
            for key, Xte in tests.items():
                a = (lda_score(mdl, Xte).argmax(1) == yt).mean()
                if k == 0:
                    res[key][0][m] = a
                else:
                    res[key][1][m, k - 1] = a
    return res


def e_feat(D, key, te, Xtr, rng):
    """Extra test variants: ('noise', feat) Gaussian with the training mean / covariance; ('alt', feat) another feature key."""
    kind, f = key
    if kind == 'noise':
        mu = Xtr.mean(0); C = lw_cov(Xtr - mu); L = np.linalg.cholesky(C + 1e-9 * np.eye(len(mu)))
        return mu + rng.standard_normal((len(te), len(mu))) @ L.T
    return D[f][te]


def time_meter(D, y, n_perm, rng):
    acc, perm = np.zeros(3), np.zeros((3, n_perm))
    for m in range(3):
        tr = np.flatnonzero(D['p_ok'] & (D['modality'] != m)); te = np.flatnonzero(D['p_ok'] & (D['modality'] == m))
        strata = D['ses'][tr] * 10 + D['modality'][tr]; yt = D['concept'][te]
        for k in range(n_perm + 1):
            yy = y[tr] if k == 0 else perm_labels(y[tr], strata, rng)
            pr = np.zeros(len(te), int)
            for s in np.unique(D['ses'][te]):           # onset times are session specific
                a, b = te[D['ses'][te] == s], tr[D['ses'][tr] == s]
                pr[D['ses'][te] == s] = knn_time_pred(D['onset'][b], yy[D['ses'][tr] == s], D['onset'][a])
            v = (pr == yt).mean()
            if k == 0:
                acc[m] = v
            else:
                perm[m, k - 1] = v
    return {'main': (acc, perm)}


_DS = {}


def subj_job(args):
    """All verdict analyses of one subject (true labels unless y is given). Data come from the fork-shared _DS."""
    s, c, y, seed, n_perm, what = args
    D = _DS[s]; rng = np.random.default_rng(seed); out = {}
    y = D['concept'] if y is None else y
    D = dict(D); D['concept'] = y
    if 'H1' in what:
        r = transfer(D, y, 'p_amp', 'p_amp', 'p', 'lda', n_perm, rng, extra_te=[('noise', 'p_amp'), ('alt', 'p_amp_ph')])
        out['H1'] = r['main']; out['ctl_noise_H1'] = r[('noise', 'p_amp')]; out['ctl_phase_H1'] = r[('alt', 'p_amp_ph')]
    if 'H2' in what:
        r = transfer(D, y, 'p_de', 'i_de', 'i', 'lda', n_perm, rng, extra_te=[('noise', 'p_de'), ('alt', 'i_de_ph')], pca=64)
        out['H2'] = r['main']; out['ctl_noise_H2'] = r[('noise', 'p_de')]; out['ctl_phase_H2'] = r[('alt', 'i_de_ph')]
    if 'DE' in what:
        out['DE_H1'] = transfer(D, y, 'p_de', 'p_de', 'p', 'lr', n_perm, rng, C=c['classifier']['lr_C'])['main']
    if 'meters' in what:
        out['meter_E_H1'] = transfer(D, y, 'p_eog_amp', 'p_eog_amp', 'p', 'lda', n_perm, rng)['main']
        out['meter_E_H2'] = transfer(D, y, 'p_eog_de', 'i_eog_de', 'i', 'lda', n_perm, rng)['main']
        out['meter_P'] = transfer(D, y, 'p_pre', 'p_pre', 'p', 'lda', n_perm, rng)['main']
        out['meter_G'] = transfer(D, y, 'p_gfp', 'p_gfp', 'p', 'lda', n_perm, rng)['main']
        out['meter_T'] = time_meter(D, y, n_perm, rng)['main']
    if 'report' in what:
        out['report'] = report_subject(D, c, rng)
    return s, out


# ----------------------------------------------------------------------------------------------------------------- report
def report_subject(D, c, rng):
    y = D['concept']; R = {}
    # within-modality leave-one-block-out (perception, arm M), 200 permutations
    wm = []
    for m in range(3):
        rows = np.flatnonzero(D['p_ok'] & (D['modality'] == m)); acc, perms = [], np.zeros(200)
        pred = np.zeros(len(rows), int); predp = np.zeros((200, len(rows)), int)
        for b in np.unique(D['block_id'][rows]):
            te = rows[D['block_id'][rows] == b]; tr = rows[D['block_id'][rows] != b]; st = D['ses'][tr]
            Xtr, Xte = D['p_amp'][tr], D['p_amp'][te]
            pred[D['block_id'][rows] == b] = lda_score(lda_fit(Xtr, y[tr]), Xte).argmax(1)
            for k in range(200):
                predp[k, D['block_id'][rows] == b] = lda_score(lda_fit(Xtr, perm_labels(y[tr], st, rng)), Xte).argmax(1)
        a = (pred == y[rows]).mean(); pm = (predp == y[rows]).mean(1)
        wm.append(dict(acc=float(a), perm_mean=float(pm.mean()), excess=float(a - pm.mean())))
    R['within_modality'] = wm
    # pairs (train one modality -> test another), arm M, 200 permutations
    pr = {}
    for a_ in range(3):
        for b_ in range(3):
            if a_ == b_:
                continue
            tr = np.flatnonzero(D['p_ok'] & (D['modality'] == a_)); te = np.flatnonzero(D['p_ok'] & (D['modality'] == b_))
            st = D['ses'][tr]; acc = (lda_score(lda_fit(D['p_amp'][tr], y[tr]), D['p_amp'][te]).argmax(1) == y[te]).mean()
            pm = np.mean([(lda_score(lda_fit(D['p_amp'][tr], perm_labels(y[tr], st, rng)), D['p_amp'][te]).argmax(1) == y[te]).mean() for _ in range(200)])
            pr[f'{a_}->{b_}'] = dict(acc=float(acc), excess=float(acc - pm))
    R['pairs'] = pr
    # imagination variants (DE -> PCA 64 -> LDA), point values with 200 permutations
    def imag(tr, te, ktr, kte):
        Xtr = D[ktr][tr]; mu, V = pca_fit(Xtr, 64); Xtr = (Xtr - mu) @ V; Xte = (D[kte][te] - mu) @ V; st = D['ses'][tr] * 10 + D['modality'][tr]
        acc = (lda_score(lda_fit(Xtr, y[tr]), Xte).argmax(1) == y[te]).mean()
        pm = np.mean([(lda_score(lda_fit(Xtr, perm_labels(y[tr], st, rng)), Xte).argmax(1) == y[te]).mean() for _ in range(200)])
        return dict(acc=float(acc), excess=float(acc - pm))
    R['perc_to_same_imag'] = [imag(np.flatnonzero(D['p_ok'] & (D['modality'] == m)), np.flatnonzero(D['i_ok'] & (D['modality'] == m)), 'p_de', 'i_de') for m in range(3)]
    R['imag_cross'] = [imag(np.flatnonzero(D['i_ok'] & (D['modality'] != m)), np.flatnonzero(D['i_ok'] & (D['modality'] == m)), 'i_de', 'i_de') for m in range(3)]
    # temporal generalisation (pairs pooled as "other two -> m"), 20 ms steps on 0..1 s
    ep, t = D['p_ep'], D['p_t']; steps = [np.flatnonzero((t >= a) & (t < a + 0.02)) for a in np.arange(0.0, 1.0, 0.02)]
    Xt = np.stack([ep[..., s_].mean(-1) for s_ in steps], 1)          # trials x steps x ch
    tg = np.zeros((3, len(steps), len(steps)))
    for m in range(3):
        tr = np.flatnonzero(D['p_ok'] & (D['modality'] != m)); te = np.flatnonzero(D['p_ok'] & (D['modality'] == m))
        for i in range(len(steps)):
            Xi = zs(Xt[tr, i]); mu_, sd_ = Xt[tr, i].mean(0), Xt[tr, i].std(0) + 1e-8; mdl = lda_fit(Xi, y[tr])
            for j in range(len(steps)):
                tg[m, i, j] = (lda_score(mdl, (Xt[te, j] - mu_) / sd_).argmax(1) == y[te]).mean()
    R['tempgen'] = tg.mean(0).round(4).tolist()
    # abstaining sentences: threshold from leave-one-block-out posteriors inside the training modalities
    ab = []
    for m in range(3):
        tr = np.flatnonzero(D['p_ok'] & (D['modality'] != m)); te = np.flatnonzero(D['p_ok'] & (D['modality'] == m))
        cvp = np.zeros((len(tr), NC))
        for b in np.unique(D['block_id'][tr]):
            i = D['block_id'][tr] == b; sc = lda_score(lda_fit(D['p_amp'][tr][~i], y[tr][~i]), D['p_amp'][tr][i]); cvp[i] = softmax(sc)
        thr = choose_thr(cvp, y[tr], c['abstain'])
        out = {}
        for name, Xte, mdl_y in (('decoder', D['p_amp'][te], y[tr]), ('label_shuffle', D['p_amp'][te], perm_labels(y[tr], D['ses'][tr] * 10 + D['modality'][tr], np.random.default_rng(c['seeds']['shuffle_model']))),
                                 ('noise', e_feat(D, ('noise', 'p_amp'), te, D['p_amp'][tr], rng), y[tr])):
            p = softmax(lda_score(lda_fit(D['p_amp'][tr], mdl_y), Xte)); said = p.max(1) >= thr if thr is not None else np.zeros(len(te), bool)
            hit = p.argmax(1) == y[te]
            out[name] = dict(coverage=float(said.mean()), precision=float(hit[said].mean()) if said.any() else None,
                             sentence_error=float(1 - hit[said].mean()) if said.any() else None, n_said=int(said.sum()), n=len(te))
        ab.append(dict(threshold=thr, **out))
    R['abstain'] = ab
    return R


def softmax(s):
    s = s - s.max(1, keepdims=True); e = np.exp(s); return e / e.sum(1, keepdims=True)


def choose_thr(p, y, a):
    conf = p.max(1); hit = p.argmax(1) == y
    for q in np.arange(0.0, 0.91, 0.01):
        t = np.quantile(conf, q); said = conf >= t
        if said.mean() < a['min_coverage']:
            return None
        if hit[said].mean() >= a['min_precision']:
            return float(t)
    return None


# ------------------------------------------------------------------------------------------------------------- group stats
def group(res, subs, key, n_boot, seed):
    acc = np.stack([res[s][key][0] for s in subs]); perm = np.stack([res[s][key][1] for s in subs])   # S x 3, S x 3 x K
    pm = perm.mean(-1); ex_sm = acc - pm; ex_s = ex_sm.mean(1); G = float(ex_s.mean())
    Gk = (perm - pm[..., None]).mean(1).mean(0); p = float((1 + (Gk >= G - 1e-12).sum()) / (len(Gk) + 1))
    rng = np.random.default_rng(seed); bt = np.array([ex_s[rng.integers(0, len(subs), len(subs))].mean() for _ in range(n_boot)])
    return dict(G=G, p=p, lo95=float(np.percentile(bt, 2.5)), hi95=float(np.percentile(bt, 97.5)), lo90=float(np.percentile(bt, 5)), hi90=float(np.percentile(bt, 95)),
                per_modality=ex_sm.mean(0).round(4).tolist(), acc_per_modality=acc.mean(0).round(4).tolist(), per_subject=ex_s.round(4).tolist())


def holm(ps):
    o = np.argsort(ps); adj = np.zeros(len(ps)); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def run_all(Dsub, c, labels, seed, n_perm, what, workers):
    global _DS
    _DS = Dsub; subs = sorted(Dsub); jobs = [(s, c, None if labels is None else labels[s], seed + i, n_perm, what) for i, s in enumerate(subs)]
    if workers > 1:
        with Pool(workers) as pool:
            res = dict(pool.map(subj_job, jobs))
    else:
        res = dict(map(subj_job, jobs))
    return subs, res


def random_labels(Dsub, rng, D_signal=None):
    lab = {}
    for s, D in Dsub.items():
        y = np.zeros(len(D['concept']), int)
        for g in np.unique(D['ses'] * 10 + D['modality']):
            i = np.flatnonzero(D['ses'] * 10 + D['modality'] == g); y[i] = rng.permutation(np.arange(len(i)) % NC)
        lab[s] = y
    return lab


def plant(Dsub, lab, delta, rng):
    """Shared concept pattern u_y (all modalities) + modality-specific v_{m,y}; per-feature N(0,1) entries times delta."""
    out = {}
    for s, D in Dsub.items():
        D = dict(D); y = lab[s]
        for keys in (('p_amp', 'p_amp_ph'), ('p_de', 'i_de', 'i_de_ph', 'p_de_ph')):
            d = D[keys[0]].shape[1]; u = rng.standard_normal((NC, d)); v = rng.standard_normal((3, NC, d))
            for k in keys:
                if k.endswith('_ph'):
                    continue
                D[k] = D[k] + delta * (u[y] + v[D['modality'], y])
        out[s] = D
    return out


def adopt(r1, r2, c):
    adj = holm(np.array([r1['p'], r2['p']]))
    return [bool(adj[0] <= c['alpha'] and r1['G'] >= c['sesoi']), bool(adj[1] <= c['alpha'] and r2['G'] >= c['sesoi'])], adj.tolist()


# ------------------------------------------------------------------------------------------------------------------ stages
def synth(c, n_sub=3, n_per=50, delta=0.06, seed=0):
    rng = np.random.default_rng(seed); Ds = {}
    for si in range(n_sub):
        n = 3 * NC * n_per; mod = np.repeat(np.arange(3), NC * n_per); con = np.tile(np.repeat(np.arange(NC), n_per), 3)
        o = rng.permutation(n); mod, con = mod[o], con[o]
        D = dict(concept=con, modality=mod, block=np.arange(n) // 45, onset=np.arange(n) * 8.0, p_ok=np.ones(n, bool), i_ok=np.ones(n, bool), ses=np.zeros(n, int))
        for k, d in (('p_amp', 124), ('p_de', 620), ('p_pre', 124), ('p_gfp', 4), ('p_eog_amp', 4), ('p_eog_de', 20), ('p_amp_ph', 124), ('p_de_ph', 620),
                     ('i_amp', 124), ('i_de', 620), ('i_eog_amp', 4), ('i_eog_de', 20), ('i_amp_ph', 124), ('i_de_ph', 620)):
            D[k] = rng.standard_normal((n, d))
        D['p_ep'] = rng.standard_normal((n, 124, 192)).astype(np.float32); D['p_t'] = np.arange(-64, 128) / 128.0; D['block_id'] = D['block']
        Ds[f'sub-s{si}'] = D
    lab = {s: D['concept'] for s, D in Ds.items()}
    return Ds, plant(Ds, lab, delta, rng)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', required=True); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); c = yaml.safe_load(open(P(a.config))); od = P(c['out_dir']); os.makedirs(od, exist_ok=True); sd = c['seeds']
    if a.stage == 'smoke':
        t0 = time.time(); null, sig = synth(c)
        subs, r0 = run_all(null, c, None, 1, 20, ('H1', 'H2', 'DE', 'meters', 'report'), min(a.workers, 3))
        g0 = {k: group(r0, subs, k, 100, 0)['G'] for k in r0[subs[0]] if k != 'report'}
        t1 = time.time(); subs, r1 = run_all(sig, c, None, 1, 20, ('H1', 'H2'), min(a.workers, 3))
        g1 = {k: group(r1, subs, k, 100, 0)['G'] for k in r1[subs[0]]}
        print(json.dumps(dict(null=g0, planted=g1, seconds_full_20perm=round(t1 - t0, 1), seconds_h12_20perm=round(time.time() - t1, 1),
                              report_keys=list(r0[subs[0]]['report'])), indent=1), flush=True)
        return
    Dsub = load_subjects(c)
    if a.stage == 'sim':
        sc = c['sim']; rng = np.random.default_rng(sd['info0']); out = dict(stage='sim', subjects=sorted(Dsub), info0=[], power={})
        t0 = time.time()
        for r in range(sc['info0_reps']):
            lab = random_labels(Dsub, rng); subs, res = run_all(Dsub, c, lab, sd['info0'] * 1000 + r, sc['info0_perm'], ('H1', 'H2'), a.workers)
            g1, g2 = group(res, subs, 'H1', 200, r), group(res, subs, 'H2', 200, r); ad, adj = adopt(g1, g2, c)
            out['info0'].append(dict(G1=g1['G'], p1=g1['p'], G2=g2['G'], p2=g2['p'], adopt=ad))
            if r % 10 == 0:
                print('info0', r, round(time.time() - t0), np.mean([x['adopt'][0] for x in out['info0']]), np.mean([x['adopt'][1] for x in out['info0']]), flush=True)
        out['info0_rate'] = [float(np.mean([x['adopt'][i] for x in out['info0']])) for i in range(2)]
        rng = np.random.default_rng(sd['power'])
        for dl in sc['power_deltas']:
            rows = []
            for r in range(sc['power_reps']):
                lab = random_labels(Dsub, rng); Dp = plant(Dsub, lab, dl, rng)
                subs, res = run_all(Dp, c, lab, sd['power'] * 1000 + r, sc['power_perm'], ('H1', 'H2'), a.workers)
                g1, g2 = group(res, subs, 'H1', 200, r), group(res, subs, 'H2', 200, r); ad, _ = adopt(g1, g2, c)
                rows.append(dict(G1=g1['G'], G2=g2['G'], adopt=ad))
            out['power'][str(dl)] = dict(G1_mean=float(np.mean([x['G1'] for x in rows])), G2_mean=float(np.mean([x['G2'] for x in rows])),
                                         H1_rate=float(np.mean([x['adopt'][0] for x in rows])), H2_rate=float(np.mean([x['adopt'][1] for x in rows])))
            print('power', dl, out['power'][str(dl)], round(time.time() - t0), flush=True)
        best = min(out['power'], key=lambda k: abs(out['power'][k]['G1_mean'] - sc['power_target']))
        out['power_at_target'] = dict(delta=float(best), **out['power'][best]); out['seconds'] = round(time.time() - t0)
        out['info0_ok'] = all(v <= 0.05 for v in out['info0_rate'])
        json.dump(out, open(os.path.join(od, 'sim.json'), 'w'), indent=1); print(json.dumps({k: v for k, v in out.items() if k != 'info0'}, indent=1))
        return
    if a.stage == 'verdict':
        sim = json.load(open(os.path.join(od, 'sim.json'))); assert sim['info0_ok'], 'information-zero gate failed'
        assert not os.path.exists(os.path.join(od, 'verdict.json')), 'verdict already run'
        t0 = time.time(); subs, rb = run_all(Dsub, c, None, sd['perm'] + 7, c['n_perm'], ('DE',), a.workers)
        base = group(rb, subs, 'DE_H1', c['n_boot'], sd['boot'])
        os.makedirs(os.path.dirname(P(c['baseline_out'])), exist_ok=True)
        json.dump(dict(dataset='ds004306', analysis='H1 cross-modal perception transfer, DE + logistic regression', subjects=subs, **base,
                       sim_sha256=fsha(os.path.join(od, 'sim.json'))), open(P(c['baseline_out']), 'w'), indent=1)
        subs, res = run_all(Dsub, c, None, sd['perm'], c['n_perm'], ('H1', 'H2', 'meters', 'report'), a.workers)
        for s in subs:
            res[s]['DE_H1'] = rb[s]['DE_H1']
        G = {k: group(res, subs, k, c['n_boot'], sd['boot']) for k in res[subs[0]] if k != 'report'}
        ad, adj = adopt(G['H1'], G['H2'], c)
        meters = {k: dict(G=G[k]['G'], p=G[k]['p'], ok=bool(G[k]['p'] > 0.05 or G[k]['G'] < c['meter_max'])) for k in G if k.startswith('meter_')}
        ctl = {k: dict(G=G[k]['G'], p=G[k]['p'], ok=bool(abs(G[k]['G']) < 0.01 and G[k]['p'] > 0.05)) for k in G if k.startswith('ctl_')}
        # label-shuffle model (one fixed-seed permuted reader per subject x modality): first permutation column of H1 / H2
        shuf = {h: float(np.mean([np.mean(res[s][h][1][:, 0] - res[s][h][1].mean(1)) for s in subs])) for h in ('H1', 'H2')}
        rep = {s: res[s]['report'] for s in subs}
        bdiff = np.array([np.mean((res[s]['H1'][0] - res[s]['H1'][1].mean(1)) - (rb[s]['DE_H1'][0] - rb[s]['DE_H1'][1].mean(1))) for s in subs])
        rng = np.random.default_rng(sd['boot']); bt = np.array([bdiff[rng.integers(0, len(subs), len(subs))].mean() for _ in range(c['n_boot'])])
        within = np.array([[r['excess'] for r in rep[s]['within_modality']] for s in subs]).mean(0)
        part = {sid(r['participant_id']): r for r in csv.DictReader(open(os.path.join(P(c['raw_root']), 'participants.tsv'), encoding='utf-8-sig'), delimiter='\t')}
        vv = [float(part[sid(s)]['vviq']) if sid(s) in part and part[sid(s)]['vviq'] != 'n/a' else None for s in subs]
        from scipy.stats import spearmanr
        okv = [i for i, v in enumerate(vv) if v is not None]
        rho = spearmanr([vv[i] for i in okv], [G['H2']['per_subject'][i] for i in okv]) if len(okv) > 3 else None
        out = dict(stage='verdict', subjects=subs, sim_sha256=fsha(os.path.join(od, 'sim.json')), prep_sha256=fsha(os.path.join(P(c['cache_dir']), 'prep.json')),
                   H1=G['H1'], H2=G['H2'], holm_p=adj, adopted=dict(H1=ad[0], H2=ad[1]), meters=meters, controls=ctl, label_shuffle_model=shuf,
                   no_eeg_excess=0.0, baseline_DE=base, M_minus_DE=dict(diff=float(bdiff.mean()), lo95=float(np.percentile(bt, 2.5)), hi95=float(np.percentile(bt, 97.5))),
                   within_modality_excess=within.round(4).tolist(), undecidable_modalities=[int(m) for m in np.flatnonzero(within < c['sesoi'])],
                   shared_intensity_flag=bool(G['meter_G']['p'] <= c['alpha'] and G['meter_G']['G'] >= c['sesoi']),
                   power_at_target=sim['power_at_target'], low_power=bool(sim['power_at_target']['H1_rate'] < 0.8),
                   vviq_spearman_H2=None if rho is None else dict(rho=float(rho.statistic), p=float(rho.pvalue), n=len(okv)),
                   report=rep, seconds=round(time.time() - t0))
        json.dump(out, open(os.path.join(od, 'verdict.json'), 'w'), indent=1)
        print(json.dumps({k: v for k, v in out.items() if k != 'report'}, indent=1))


if __name__ == '__main__':
    main()
