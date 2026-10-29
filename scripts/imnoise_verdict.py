"""D-126 (family F-XV): one procedure for the imagery interval of seven public data sets.

usage: python scripts/imnoise_verdict.py configs/imnoise.yaml --mode smoke|info0|verdict [--data TOL,...] [--subjects N]
       post-result diagnoses: [--perm-mode foldbal] [--info0-labels foldbal] [--no-inject] [--holm-from results/imnoise/verdict.json] [--tag _x]

Decoders (D-126 section 2): D1 time-domain 100 ms bin means, D2 band log-variance (5 bands), D3 log-Euclidean covariance (1-40 Hz),
D4 200 ms windows every 100 ms (one test unit per window). Each unit: standardise on the training trials, PCA on the training
trials (Gram form), Ledoit-Wolf covariance of the training scores without labels, whitening; class k score = u.mu_k - |mu_k|^2/2.
Because nothing but the class means depends on the labels, every permutation is scored with the same prepared matrices.
Permutations (section 3): runs of equal labels are shuffled inside each pblock; IM36 / IM36A remap concepts to categories 18/18.
Subject score = mean over tasks of the normalised excess kappa = (balanced accuracy - permutation mean) / (1 - 1/K); group G_u =
mean over subjects; max-T over units with the permutation mean / sd of G_u.
Modes
  smoke    synthetic data (no information, then a planted signal) through every path
  info0    section 6: random labels (permutation generator) with their own permutations; no real-label score is computed
  verdict  real labels: hypothesis I, meters L, artefact A1/A2, cue path C, positive control PC, controls, report-only rows,
           and the signal-injection detection limits (section 5); writes <out_dir>/verdict.json
"""
import os
os.environ.setdefault('ESE_NUM_THREADS', '1')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ['ESE_NUM_THREADS'])
import sys, glob, json, time, zlib, argparse, hashlib
import numpy as np, yaml
from scipy.signal import butter, sosfiltfilt
from sklearn.covariance import ledoit_wolf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
C = None                                                   # config, set in main / worker init


# ----------------------------------------------------------------------------------------------------------------- views
VIEW_KEYS = ('X', 't', 'fs', 'y', 'task', 'K', 'cvset', 'fold', 'role', 'pblock', 'tpos')
OPT_KEYS = ('concept_id', 'concept_cat', 'incongruent', 'modality', 'session', 'word', 'task_name', 'delay')


def get_view(z, name, win):
    """main / pc / pre view; a view without its own arrays is a window of pc (pre inside the pc epoch) or of main."""
    if name != 'main' and f'{name}_X' in z:
        pre = name + '_'
    elif name == 'pre' and 'pc_X' in z and z['pc_t'][0] <= win[0] + 1e-9 and z['pc_t'][-1] >= win[1] - 1e-2:
        pre = 'pc_'
    else:
        pre = ''
    v = {k: z[pre + k] for k in VIEW_KEYS}
    v.update({k: z[pre + k] for k in OPT_KEYS if pre + k in z})
    v['fs'] = float(v['fs']); v['win'] = list(win)
    assert v['t'][0] <= win[0] + 1e-9 and v['t'][-1] >= win[1] - 1.5 / v['fs'], (name, win, v['t'][0], v['t'][-1])
    return v


def wsl(t, w):
    return np.flatnonzero((t >= w[0] - 1e-9) & (t < w[1] - 1e-9))


# -------------------------------------------------------------------------------------------------------------- features
def bands_of():
    d = C['decoders']; return [tuple(b) for b in d['de_bands']] + [tuple(d['rg_band'])]


def filtered(X, t, fs, win):
    """band-passed copies cropped to the window: list matching bands_of(); filters run on the whole epoch (linear)."""
    w = wsl(t, win); out = []
    for lo, hi in bands_of():
        sos = butter(C['decoders']['butter_order'], [lo, min(hi, 0.45 * fs)], btype='band', fs=fs, output='sos')
        out.append(sosfiltfilt(sos, X, axis=-1)[..., w].astype(np.float32))
    return out


def feat_td(Xw, fs):
    nb = int(round(C['decoders']['td_bin'] * fs)); m = Xw.shape[-1] // nb
    return Xw[..., :m * nb].reshape(Xw.shape[0], Xw.shape[1], m, nb).mean(-1).reshape(Xw.shape[0], -1)


def feat_de(F):
    return np.concatenate([np.log(f.var(-1) + 1e-12) for f in F[:-1]], 1)


def lw_batch(X):
    """Ledoit-Wolf shrunk covariances of a batch, X [n, T, C] (assume_centered, same formula as sklearn.covariance.ledoit_wolf)"""
    n, T, Cc = X.shape; X2 = X ** 2; emp = np.einsum('ntc,ntd->ncd', X, X, optimize=True) / T
    tr_ = X2.sum(1) / T; mu = tr_.sum(1) / Cc
    beta_ = np.einsum('ntc,ntd->n', X2, X2, optimize=True); delta_ = (emp ** 2).sum((1, 2))
    beta = (beta_ / T - delta_) / (Cc * T); delta = (delta_ - 2 * mu * tr_.sum(1) + Cc * mu ** 2) / Cc
    beta = np.minimum(beta, delta); s = np.where(beta == 0, 0.0, beta / np.where(delta == 0, 1, delta))
    return (1 - s)[:, None, None] * emp + (s * mu)[:, None, None] * np.eye(Cc)[None]


def feat_rg(f, chunk=256):
    n, ch, T = f.shape; iu = np.triu_indices(ch); wgt = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0)); out = np.empty((n, len(iu[0])), np.float32)
    for a in range(0, n, chunk):
        x = f[a:a + chunk].astype(np.float64); x = x - x.mean(2, keepdims=True)
        S = lw_batch(x.transpose(0, 2, 1)); S /= np.trace(S, axis1=1, axis2=2)[:, None, None]
        e, V = np.linalg.eigh(S); L = np.einsum('nij,nj,nkj->nik', V, np.log(np.maximum(e, 1e-12)), V, optimize=True)
        out[a:a + chunk] = L[:, iu[0], iu[1]] * wgt
    return out


def feat_tr(Xw, t_w, fs):
    d = C['decoders']; L = int(round(d['tr_len'] * fs)); S = int(round(d['tr_step'] * fs)); res = []
    for a in range(0, Xw.shape[-1] - L + 1, S):
        res.append((round(float(t_w[a]), 3), Xw[..., a:a + L].mean(-1)))
    return res


def units(X, t, fs, win, which=('D1', 'D2', 'D3', 'D4'), filt=None):
    """{unit name: (features [n, d], pca dims)}"""
    w = wsl(t, win); Xw = X[..., w]; out = {}; d = C['decoders']
    if 'D2' in which or 'D3' in which:
        F = filt if filt is not None else filtered(X, t, fs, win)
    if 'D1' in which: out['D1'] = (feat_td(Xw, fs), d['pca'])
    if 'D2' in which: out['D2'] = (feat_de(F), d['pca'])
    if 'D3' in which: out['D3'] = (feat_rg(F[-1]), d['pca'])
    if 'D4' in which:
        for tt, f in feat_tr(Xw, t[w], fs): out[f'D4@{tt:.2f}'] = (f, d['pca_tr'])
    if 'AMP' in which: out['AMP'] = (Xw.mean(-1), d['pca'])      # cue-path meter: channel mean amplitude over the window
    return out


def zall(F):
    mu = F.mean(0); sd = F.std(0); return (F - mu) / np.where(sd > 1e-12, sd, 1.0)


# ------------------------------------------------------------------------------------------------------------ decoder
def prepare(Ftr_all, Fte_all, q, splits):
    """per split: whitened training scores U_tr [n_tr, q'], test scores U_te; Fte_all supplies the test rows (noise / cue meters)."""
    res = []
    for tr, te in splits:
        A = Ftr_all[tr].astype(np.float64); B = Fte_all[te].astype(np.float64)
        mu = A.mean(0); sd = A.std(0); sd = np.where(sd > 1e-12, sd, 1.0); A = (A - mu) / sd; B = (B - mu) / sd
        k = min(q, len(tr) - 1, A.shape[1])
        if A.shape[1] < len(tr):                                       # PCA scores from the d x d scatter (same subspace as the Gram form)
            e, V = np.linalg.eigh(A.T @ A); e, V = e[::-1][:k], V[:, ::-1][:, :k]; keep = e > 1e-9 * max(e[0], 1e-12)
            Str = A @ V[:, keep]; Ste = B @ V[:, keep]
        else:
            G = A @ A.T; e, V = np.linalg.eigh(G); e, V = e[::-1][:k], V[:, ::-1][:, :k]
            keep = e > 1e-9 * max(e[0], 1e-12); e, V = e[keep], V[:, keep]
            Str = V * np.sqrt(e); Ste = (B @ A.T) @ V / np.sqrt(e)      # PCA scores (training mean is zero after standardisation)
        if Str.shape[1] == 0:                                          # constant features (flat channels): constant output, chance level
            res.append((tr, te, np.zeros((len(tr), 1), np.float32), np.zeros((len(te), 1), np.float32))); continue
        S, _ = ledoit_wolf(Str, assume_centered=True); ev, EV = np.linalg.eigh(S); Wh = EV / np.sqrt(np.maximum(ev, 1e-12))
        res.append((tr, te, (Str @ Wh).astype(np.float32), (Ste @ Wh).astype(np.float32)))
    return res


def predict(prep, Y, K, n):
    """Y [Pm, n] labels (all trials) -> predictions [Pm, n] (-1 where not tested)."""
    Pm = Y.shape[0]; out = np.full((Pm, n), -1, np.int16)
    for tr, te, Ut, Ue in prep:
        Ytr = Y[:, tr]; oh = (Ytr[:, None, :] == np.arange(K)[None, :, None]).astype(np.float32)      # [Pm, K, n_tr]
        nk = oh.sum(-1); mu = (oh @ Ut) / np.maximum(nk, 1)[..., None]                                  # [Pm, K, q]
        sc = np.einsum('mq,pkq->pmk', Ue, mu, optimize=True) - 0.5 * (mu ** 2).sum(-1)[:, None, :]
        sc[np.broadcast_to((nk == 0)[:, None, :], sc.shape)] = -np.inf
        out[:, te] = sc.argmax(-1)
    return out


def knn_predict(tpos, splits, Y, K, n, k):
    out = np.full((Y.shape[0], n), -1, np.int16)
    for tr, te in splits:
        nn = np.argsort(np.abs(tpos[te][:, None] - tpos[tr][None, :]), 1)[:, :k]; lab = Y[:, tr][:, nn]      # [Pm, n_te, k]
        cnt = (lab[..., None] == np.arange(K)).sum(2); out[:, te] = cnt.argmax(-1)
    return out


def bacc(pred, Y, mask, task, Ks):
    """normalised excess input: balanced accuracy per task -> list over tasks of [Pm] arrays, and K per task"""
    res = []
    for ti, K in enumerate(Ks):
        m = mask & (task == ti) & (pred[0] >= 0)
        if m.sum() == 0: continue
        p, y = pred[:, m], Y[:, m]; rec = []
        for k in range(K):
            yk = (y == k); nk = yk.sum(1)
            rec.append(np.where(nk > 0, ((p == k) & yk).sum(1) / np.maximum(nk, 1), np.nan))
        res.append((np.nanmean(np.stack(rec, 1), 1), K))
    return res


def kappa_of(res, pm=None):
    """subject kappa per permutation row: mean over tasks of (bacc - mean over permutation rows 1..) / (1 - 1/K)"""
    ks = []
    for b, K in res:
        base = b[1:].mean() if pm is None else pm
        ks.append((b - base) / (1 - 1.0 / K))
    return np.mean(ks, 0)


# ---------------------------------------------------------------------------------------------------------- permutations
def perm_labels(v, B, seed, mode=None):
    """[B+1, n]; row 0 = labels as given. mode (default C['perm_mode']): None = registered, 'plain' (amendment 3 diagnosis),
    'foldbal' (post-result diagnosis of D-126: IM36 / IM36A remaps that keep each test fold's category count)."""
    mode = C.get('perm_mode') if mode is None else mode
    y = v['y']; n = len(y); rng = np.random.default_rng(seed); Y = np.empty((B + 1, n), np.int16); Y[0] = y
    if 'concept_id' in v:
        cat = v['concept_cat']; cid = v['concept_id']; half = int(cat.sum())
        if mode == 'foldbal':
            grp = {}
            for c_, g in zip(cid.tolist(), zip(v['cvset'].tolist(), v['fold'].tolist())): grp.setdefault(g, set()).add(c_)
            groups = [np.array(sorted(cs)) for cs in grp.values()]
            assert sum(len(cs) for cs in groups) == len(set(cid.tolist())), 'a concept spans several folds'
            for b in range(1, B + 1):
                new = np.zeros(len(cat), np.int16)
                for cs in groups: new[rng.choice(cs, int(cat[cs].sum()), replace=False)] = 1
                Y[b] = new[cid]
            return Y
        for b in range(1, B + 1):
            new = np.zeros(len(cat), np.int16); new[rng.choice(len(cat), half, replace=False)] = 1; Y[b] = new[cid]
        return Y
    blocks = [np.flatnonzero(v['pblock'] == g) for g in np.unique(v['pblock'])]
    if mode == 'plain':                          # D-126 amendment 3 diagnosis: plain permutation inside each pblock
        for b in range(1, B + 1):
            for idx in blocks: Y[b, idx] = y[idx][rng.permutation(len(idx))]
        return Y
    runs = []
    for idx in blocks:
        yy = y[idx]; cut = np.flatnonzero(np.diff(yy) != 0) + 1; runs.append((idx, np.split(yy, cut)))
    for b in range(1, B + 1):
        for idx, rr in runs:
            Y[b, idx] = np.concatenate([rr[j] for j in rng.permutation(len(rr))])
    return Y


def relabel(v, ynew):
    w = dict(v); w['y'] = ynew
    if 'concept_id' in v:                                      # the remapped categories define a new concept -> category table
        cat = np.zeros(len(v['concept_cat']), int); cat[v['concept_id']] = ynew; w['concept_cat'] = cat
    return w


def splits_of(v, kind='cv'):
    cs, f, r, tk = v['cvset'], v['fold'], v['role'], v['task']; out = []
    for s in (np.unique(cs) if kind in ('cv', 'xses') else []):
        for ti in np.unique(tk[cs == s]):
            base = (cs == s) & (tk == ti)
            for ff in np.unique(f[base & (r > 0)]):
                te = np.flatnonzero(base & (f == ff) & (r > 0))
                if kind == 'cv':
                    tr = np.flatnonzero(base & (f != ff) & (r != 1))
                elif kind == 'xses':                           # TOL report: training runs of the other sessions only
                    tr = np.flatnonzero(base & (v['session'] != v['session'][te[0]]) & (r != 1))
                if len(tr) and len(te): out.append((tr, te))
    if kind == 'xmod':                                         # SEM cue path: train two modalities, test the third (block held out)
        out = []
        for s in np.unique(cs):
            for ff in np.unique(f[cs == s]):
                for m in np.unique(v['modality']):
                    te = np.flatnonzero((cs == s) & (f == ff) & (v['modality'] == m)); tr = np.flatnonzero((cs == s) & (f != ff) & (v['modality'] != m))
                    if len(tr) and len(te): out.append((tr, te))
    return out


# ------------------------------------------------------------------------------------------------------------ subject work
def tested_mask(v, splits):
    m = np.zeros(len(v['y']), bool)
    for _, te in splits: m[te] = True
    return m


def score_units(U, v, splits, Ys, mask=None, Fte=None):
    """{unit: subject kappa rows [Pm]} for label matrix Ys; also returns the predictions for subset scoring"""
    n = len(v['y']); Kmax = int(max(v['K'])); out, preds = {}, {}; tm = tested_mask(v, splits) if mask is None else mask
    for name, (F, q) in U.items():
        pr = prepare(F, F if Fte is None else Fte[name][0], q, splits); preds[name] = predict(pr, Ys, Kmax, n)
        out[name] = bacc(preds[name], Ys, tm, v['task'], v['K'])
    return out, preds


def resid_eeg(X, A):
    """least-squares removal of the artefact channels' signals from X (label-free, all samples of the subject)"""
    Xf = X.transpose(1, 0, 2).reshape(X.shape[1], -1).astype(np.float64); Af = A.transpose(1, 0, 2).reshape(A.shape[1], -1).astype(np.float64)
    Bm = np.linalg.lstsq(Af.T, Xf.T, rcond=None)[0]; R = Xf - Bm.T @ Af
    return R.reshape(X.shape[1], X.shape[0], X.shape[2]).transpose(1, 0, 2).astype(np.float32)


def artefact_split(z, v):
    if 'art' in z:
        return v['X'], z['art']
    ch = [str(c) for c in z['ch']]; idx = [ch.index(c) for c in z['art_names'] if c in ch]; rest = [i for i in range(len(ch)) if i not in idx]
    return v['X'][:, rest], v['X'][:, idx]


def inject(X, t, fs, win, Fx, y, K, kind, delta, rng):
    """section 5: class-specific evoked (5 Hz low-pass waveform) or induced (8-30 Hz random-phase source) signal. Returns the
    injected epochs and their band-passed window copies, using the linearity of the filters (Fx = filtered(X))."""
    n, ch, T = X.shape; ic = C['inject']; w = wsl(t, win); bands = bands_of()
    # D-126 amendment 4: spatial patterns drawn from the data's own channel covariance (a_y = S^1/2 g_y), so the injected signal
    # stays inside the subspace the recording occupies (average reference, ICA-removed components) with a data-like spatial spread
    Xc = X - X.mean(2, keepdims=True); Sg = np.einsum('nct,ndt->cd', Xc, Xc, optimize=True) / (n * T); e, V = np.linalg.eigh(Sg)
    a = rng.standard_normal((K, ch)) @ (V * np.sqrt(np.maximum(e, 0))) @ V.T; sd = np.ones(ch)
    def bp(x, lo, hi):
        sos = butter(C['decoders']['butter_order'], [lo, min(hi, 0.45 * fs)], btype='band', fs=fs, output='sos'); return sosfiltfilt(sos, x, axis=-1)[..., w]
    if kind == 'evoked':
        sos = butter(4, ic['evoked_lowpass'], btype='low', fs=fs, output='sos'); b = sosfiltfilt(sos, rng.standard_normal((K, T)), axis=-1)
        b /= np.sqrt((b ** 2).mean(1, keepdims=True)); S = delta * sd[None, :, None] * a[:, :, None] * b[:, None, :]          # [K, ch, T]
        Xi = X + S[y]; Fi = [f + bp(S, lo, hi)[y].astype(np.float32) for f, (lo, hi) in zip(Fx, bands)]
        return Xi.astype(np.float32), Fi
    sos = butter(4, ic['induced_band'], btype='band', fs=fs, output='sos'); s = sosfiltfilt(sos, rng.standard_normal((n, T)), axis=-1)
    s /= np.sqrt((s ** 2).mean(1, keepdims=True)); A = delta * sd[None, :] * a[y]                                           # [n, ch]
    Xi = X + A[:, :, None] * s[:, None, :]
    Fi = [f + (A[:, :, None] * bp(s, lo, hi)[:, None, :]).astype(np.float32) for f, (lo, hi) in zip(Fx, bands)]
    return Xi.astype(np.float32), Fi


def subject_job(args):
    ds, f, mode, cfg = args
    global C
    C = cfg; z = dict(np.load(f, allow_pickle=True)); sub = os.path.basename(f)[:-4]; W = C['windows'][ds]; B = C['n_perm']
    seed = lambda key: [C['seeds'][key], zlib.crc32(f'{ds}_{sub}'.encode())]
    t0 = time.time(); res = dict(sub=sub, tests={})
    v = get_view(z, 'main', W['main']); spl = splits_of(v); n = len(v['y'])
    U = units(v['X'], v['t'], v['fs'], v['win'])
    res['n'] = int(n); res['units'] = list(U)
    if mode == 'info0':
        ic = C['info0']; rows = []
        Yr = perm_labels(v, ic['reps'], seed('info0'), mode=C.get('info0_label_mode') or C.get('perm_mode'))   # random labels
        preps = {name: prepare(F, F, q, spl) for name, (F, q) in U.items()}; tm = tested_mask(v, spl)
        for r in range(1, ic['reps'] + 1):
            vr = relabel(v, Yr[r].astype(int)); Ys = perm_labels(vr, ic['n_perm'], [C['seeds']['info0'], r, zlib.crc32(f'{ds}_{sub}'.encode())])
            rows.append({name: kappa_of(bacc(predict(pr, Ys, int(max(v['K'])), n), Ys, tm, v['task'], v['K'])).tolist() for name, pr in preps.items()})
        res['info0'] = rows; res['seconds'] = round(time.time() - t0, 1); return res

    Y = perm_labels(v, B, seed('perm')); Kmax = int(max(v['K'])); tm = tested_mask(v, spl)
    def put(name, sc, extra=None):
        res['tests'][name] = {u: kappa_of(r).tolist() for u, r in sc.items()}
        if extra: res['tests'][name + '_raw'] = extra
    # hypothesis I
    sc, preds = score_units(U, v, spl, Y); put('I', sc)
    res['tests']['I_bacc'] = {u: [float(b[0]) for b, K in r] for u, r in sc.items()}
    res['perm_mean'] = {u: [float(b[1:].mean()) for b, K in r] for u, r in sc.items()}
    # report-only subsets of the same predictions
    if 'incongruent' in v:                                     # IM36 cue path C: incongruent concepts only
        msk = tm & v['incongruent'].astype(bool); put('C', {u: bacc(p, Y, msk, v['task'], v['K']) for u, p in preds.items()})
        for ti in np.unique(v['task_name']):
            msk = tm & (v['task_name'] == ti); put(f'R_task{int(ti)}', {u: bacc(p, Y, msk, v['task'], v['K']) for u, p in preds.items()})
    if 'modality' in v and ds == 'SEM':
        for m in np.unique(v['modality']):
            msk = tm & (v['modality'] == m); put(f'R_mod{int(m)}', {u: bacc(p, Y, msk, v['task'], v['K']) for u, p in preds.items()})
    if ds == 'GAO':
        for ti in range(len(v['K'])):
            msk = tm & (v['task'] == ti); put(f'R_cat{ti}', {u: bacc(p, Y, msk, v['task'], v['K']) for u, p in preds.items()})
    # label-shuffle and noise controls
    rng = np.random.default_rng(seed('noise'))
    mu = v['X'].mean((0, 2)); sd = v['X'].std((0, 2)); Xn = (mu[None, :, None] + sd[None, :, None] * rng.standard_normal(v['X'].shape)).astype(np.float32)
    Un = units(Xn, v['t'], v['fs'], v['win'])
    put('NOISE', score_units(U, v, spl, Y, Fte=Un)[0]); del Xn, Un
    # meters L: pre-stimulus window and time position
    if 'pre' in W:
        vp = get_view(z, 'pre', W['pre']); sp = splits_of(vp); Yp = perm_labels(vp, B, seed('perm'))
        put('L_pre', score_units(units(vp['X'], vp['t'], vp['fs'], vp['win'], which=('D1', 'D2')), vp, sp, Yp)[0])
    pk = knn_predict(v['tpos'], spl, Y, Kmax, n, C['knn_k']); put('L_T', {'T': bacc(pk, Y, tm, v['task'], v['K'])})
    # positive control
    if 'pc' in W:
        vc = get_view(z, 'pc', W['pc']); sc_ = splits_of(vc); Yc = perm_labels(vc, B, seed('perm'))
        put('PC', score_units(units(vc['X'], vc['t'], vc['fs'], vc['win'], which=('D1', 'D2')), vc, sc_, Yc)[0])
    # artefact paths
    E, A = artefact_split(z, v)
    put('A1', score_units(units(A, v['t'], v['fs'], v['win']), v, spl, Y)[0])
    R = resid_eeg(E, A); put('A2', score_units(units(R, v['t'], v['fs'], v['win']), v, spl, Y)[0]); del R
    # cue path C (data-set specific)
    if ds == 'TOL':
        Ucue = {k: (zall(F), q) for k, (F, q) in units(v['X'], v['t'], v['fs'], W['cue'], which=('AMP', 'D2')).items()}
        Umain = {k: (zall(F), q) for k, (F, q) in units(v['X'], v['t'], v['fs'], v['win'], which=('AMP', 'D2')).items()}
        put('C', score_units(Ucue, v, spl, Y, Fte=Umain)[0])
    elif ds == 'SEM':
        sx = splits_of(v, 'xmod'); put('C', score_units(U, v, sx, Y, mask=tested_mask(v, sx))[0])
    elif ds == 'GAO':
        vl = get_view(z, 'main', W['late']); put('C', score_units(units(vl['X'], vl['t'], vl['fs'], vl['win']), vl, spl, Y)[0])
    # report-only label / split variants
    if ds == 'UGR':
        vw = relabel(v, v['word'].astype(int)); vw['K'] = np.array([60]); Yw = perm_labels(vw, B, seed('perm'))
        put('R_word', score_units(U, vw, spl, Yw)[0])
    if ds == 'TOL':
        sx = splits_of(v, 'xses'); put('R_xses', score_units(U, v, sx, Y, mask=tested_mask(v, sx))[0])
    res['seconds'] = round(time.time() - t0, 1)
    return res


def inject_job(args):
    """section 5 for one subject and one (kind, delta): rows of subject kappa per unit (labels from the permutation generator)"""
    ds, f, kind, d_, perm_mean, cfg = args
    global C
    C = cfg; z = dict(np.load(f, allow_pickle=True)); sub = os.path.basename(f)[:-4]; ic = C['inject']
    v = get_view(z, 'main', C['windows'][ds]['main']); spl = splits_of(v); n = len(v['y']); Kmax = int(max(v['K'])); tm = tested_mask(v, spl)
    key = zlib.crc32(f'{ds}_{sub}_{kind}_{d_}'.encode()); Yi = perm_labels(v, ic['reps'], [C['seeds']['inject'], key])
    rng = np.random.default_rng([C['seeds']['inject'], key, 1]); rows = []; Fx = filtered(v['X'], v['t'], v['fs'], v['win'])
    for rep in range(1, ic['reps'] + 1):
        yi = Yi[rep].astype(int); Xi, Fi = inject(v['X'], v['t'], v['fs'], v['win'], Fx, yi, Kmax, kind, d_, rng)
        Ui = units(Xi, v['t'], v['fs'], v['win'], filt=Fi); Yrow = yi[None, :]
        sci = {u: bacc(predict(prepare(F, F, q, spl), Yrow, Kmax, n), Yrow, tm, v['task'], v['K']) for u, (F, q) in Ui.items()}
        rows.append({u: float(np.mean([(b[0] - pm) / (1 - 1.0 / K) for (b, K), pm in zip(r, perm_mean[u])])) for u, r in sci.items()})
    return rows


def run_inject(ds, files, subs, stI, cfg):
    """registered grid, then the extension rule of D-126 amendment 1: double above the grid while the top rate < power (up to 4
    steps), halve below it while the bottom rate >= power (up to 4 steps)"""
    from multiprocessing import Pool
    ic = cfg['inject']; units_ = sorted(subs[0]['tests']['I']); m = np.array(stI['null']['mean']); sd = np.array(stI['null']['sd'])
    Tnull = np.array(stI['null']['T']); out = {}
    def one(pool, kind, d_):
        R = pool.map(inject_job, [(ds, f, kind, d_, s['perm_mean'], cfg) for f, s in zip(files, subs)], chunksize=1)
        det, kap = [], []
        for r in range(ic['reps']):
            G = np.array([np.mean([R[i][r][u] for i in range(len(R))]) for u in units_]); Z = (G - m) / np.where(sd > 0, sd, 1)
            det.append((1 + (Tnull >= Z.max()).sum()) / (len(Tnull) + 1) <= cfg['alpha']); kap.append(float(G[int(np.argmax(Z))]))
        out[f'{kind}_{d_:g}'] = dict(kind=kind, delta=d_, rate=float(np.mean(det)), kappa=float(np.mean(kap)))
        print(ds, kind, d_, out[f'{kind}_{d_:g}'], flush=True); return out[f'{kind}_{d_:g}']['rate']
    with Pool(cfg.get('workers_by', {}).get(ds, cfg['workers'])) as pool:
        for kind in ('evoked', 'induced'):
            grid = list(ic[kind]); rates = [one(pool, kind, d_) for d_ in grid]
            for _ in range(ic.get('extend_steps', 4)):
                if rates[-1] >= ic['power']: break
                grid.append(grid[-1] * 2); rates.append(one(pool, kind, grid[-1]))
            for _ in range(ic.get('extend_steps', 4)):
                if rates[0] < ic['power']: break
                grid.insert(0, grid[0] / 2); rates.insert(0, one(pool, kind, grid[0]))
    mde = {}
    for kind in ('evoked', 'induced'):
        pts = sorted((v_['kappa'], v_['rate']) for v_ in out.values() if v_['kind'] == kind); x = [a for a, b in pts]; yv = [b for a, b in pts]; val = None
        for i in range(1, len(pts)):
            if yv[i - 1] < ic['power'] <= yv[i]:
                val = x[i - 1] + (ic['power'] - yv[i - 1]) * (x[i] - x[i - 1]) / max(yv[i] - yv[i - 1], 1e-9); break
        if val is None and yv and yv[0] >= ic['power']: val = x[0]
        mde[kind] = val
    return out, mde


# ------------------------------------------------------------------------------------------------------------ group stats
def maxT(test_by_sub, B_used=None):
    """test_by_sub: list over subjects of {unit: kappa rows [Pm]} -> group statistics with max-T over units"""
    units_ = sorted(test_by_sub[0]); Kap = np.array([[s[u] for u in units_] for s in test_by_sub])     # [S, U, Pm]
    G = Kap.mean(0); m = G[:, 1:].mean(1, keepdims=True); sdv = G[:, 1:].std(1, keepdims=True); Z = (G - m) / np.where(sdv > 0, sdv, 1)
    T = Z.max(0); p = (1 + (T[1:] >= T[0]).sum()) / len(T)
    best = int(np.argmax(Z[:, 0])); rng = np.random.default_rng(C['seeds']['boot']); S = Kap.shape[0]
    bs = rng.integers(0, S, (C['n_boot'], S)); bootG = Kap[:, :, 0][bs].mean(1)                     # [nb, U]
    lo, hi = np.percentile(bootG, [2.5, 97.5], axis=0)
    out = dict(p_maxT=float(p), T=float(T[0]), best_unit=units_[best], best_kappa=float(G[best, 0]), best_ci=[float(lo[best]), float(hi[best])],
               best_p_unit=float((1 + (G[best, 1:] >= G[best, 0]).sum()) / G.shape[1]), n_subjects=int(S),
               subject_kappa_best=[float(x) for x in Kap[:, best, 0]],
               units={u: dict(kappa=float(G[i, 0]), ci=[float(lo[i]), float(hi[i])], z=float(Z[i, 0]),
                              p_unit=float((1 + (G[i, 1:] >= G[i, 0]).sum()) / G.shape[1])) for i, u in enumerate(units_)},
               shuffle_first=dict(max_kappa=float(G[:, 1].max()), p=float((1 + (T[1:] >= T[1]).sum()) / len(T))) if G.shape[1] > 1 else None,
               null=dict(mean=m[:, 0].tolist(), sd=sdv[:, 0].tolist(), T=T[1:].tolist()))
    return out


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj.tolist()


def classify(ds, st, mde_max):
    I = st['I_holm'] <= C['alpha']; lim = C['meter_max']
    meter_ok = lambda t: t is None or t['p_maxT'] > C['alpha'] or t['best_kappa'] < lim
    pc = st['tests'].get('PC'); pc_ok = pc is None or pc['p_maxT'] <= C['alpha']
    L_ok = meter_ok(st['tests'].get('L_pre')) and meter_ok(st['tests'].get('L_T'))
    A_ok = st['tests']['A2']['p_maxT'] <= C['alpha']
    if ds == 'TOL': C_ok = meter_ok(st['tests']['C'])
    elif ds in ('SEM', 'IM36', 'IM36A', 'GAO'): C_ok = st['tests']['C']['p_maxT'] <= C['alpha']
    else: C_ok = True
    sesoi = C['sesoi_kappa']
    if pc is not None and not pc_ok and not I: return 1, '판정 불가(양성 대조 실패)'
    if not L_ok: return 2, '판정 불가(누설)'
    if I and A_ok and C_ok: return 3, '검출: 측정기·단서로 설명되지 않는 심상 정보' + (' (단서 정보 없음)' if ds == 'BCI' else '')
    if I: return 4, '검출되었으나 경로를 배제하지 못함: ' + ', '.join(n for n, ok in (('인공물(A2)', A_ok), ('단서(C)', C_ok)) if not ok)
    if st['tests']['I']['best_ci'][1] < sesoi and mde_max is not None and mde_max <= sesoi: return 5, '잡음과 구별되지 않음, SESOI 이상 배제'
    return 6, '잡음과 구별되지 않음, 검정력 부족'


# ------------------------------------------------------------------------------------------------------------ main
def synth(n_sub, info, rng):
    """smoke data: 3 blocks x 40 trials, 3 classes, 16 channels, 128 Hz, -0.5..3.5 s; info > 0 plants an evoked class pattern"""
    subs = []
    for s in range(n_sub):
        n, ch, fs = 120, 16, 128.0; t = -0.5 + np.arange(int(4 * fs)) / fs; y = np.tile(np.arange(3), 40); rng.shuffle(y)
        X = rng.standard_normal((n, ch, len(t))).astype(np.float32) * 10
        if info: X += (info * 10 * rng.standard_normal((3, ch, 1)) * np.sin(2 * np.pi * 3 * t))[y].astype(np.float32)
        blk = np.repeat(np.arange(3), 40)
        subs.append(dict(X=X, t=t, fs=fs, ch=np.array(['c%d' % i for i in range(ch)]), y=y, task=np.zeros(n, int), K=np.array([3]), cvset=np.zeros(n, int),
                         fold=blk, role=np.full(n, 2), pblock=blk, tpos=np.arange(n, dtype=float), art=rng.standard_normal((n, 2, len(t))).astype(np.float32),
                         art_names=np.array(['a1', 'a2'])))
    return subs


def run_dataset(ds, files, mode, cfg):
    from multiprocessing import Pool
    with Pool(cfg.get('workers_by', {}).get(ds, cfg['workers'])) as pool:
        subs = pool.map(subject_job, [(ds, f, mode, cfg) for f in files], chunksize=1)
    return subs


def main():
    global C
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--mode', required=True); ap.add_argument('--data', default='')
    ap.add_argument('--subjects', type=int, default=0); ap.add_argument('--info0-reps', type=int, default=0); ap.add_argument('--tag', default=''); ap.add_argument('--perm-mode', default='')
    ap.add_argument('--info0-labels', default=''); ap.add_argument('--no-inject', action='store_true'); ap.add_argument('--holm-from', default='')
    a = ap.parse_args(); C = yaml.safe_load(open(P(a.config)))
    if a.info0_reps: C['info0']['reps'] = a.info0_reps                  # diagnosis of the info-0 check (D-126 amendment 2)
    if a.perm_mode: C['perm_mode'] = a.perm_mode                         # D-126 amendment 3 diagnosis
    if a.info0_labels: C['info0_label_mode'] = a.info0_labels           # post-result diagnosis: generator of the random labels only
    out = P(C['out_dir']); os.makedirs(out, exist_ok=True); dsl = a.data.split(',') if a.data else C['datasets']; t0 = time.time()
    if a.mode == 'smoke':
        sm = os.path.join(out, 'smoke'); os.makedirs(sm, exist_ok=True); C.update(n_perm=100, n_boot=200, workers=2)
        C['inject'].update(reps=3, evoked=[0.05, 0.2], induced=[0.2, 0.8]); C['info0'].update(reps=3, n_perm=50)
        C['windows']['SYN'] = dict(main=[1.0, 3.0], pc=[1.0, 3.0], pre=[-0.5, 0.0]); rep = {}
        for info in (0.0, 0.3):
            files = []
            for i, d in enumerate(synth(4, info, np.random.default_rng(1 + int(info * 10)))):
                f = os.path.join(sm, f'syn{info}_{i}.npz'); np.savez(f, **d); files.append(f)
            subs = run_dataset('SYN', files, 'verdict', C); st = {k: maxT([s['tests'][k] for s in subs]) for k in subs[0]['tests'] if not k.endswith('_raw') and k != 'I_bacc'}
            inj, mde = run_inject('SYN', files, subs, st['I'], C)
            rep[str(info)] = dict({k: (v['p_maxT'], round(v['best_kappa'], 4)) for k, v in st.items()}, inject=inj, mde=mde, seconds=[s['seconds'] for s in subs])
            subs0 = run_dataset('SYN', files, 'info0', C); rep[str(info)]['info0_rows'] = len(subs0[0]['info0'])
            print(json.dumps(rep[str(info)]), flush=True)
        json.dump(rep, open(os.path.join(sm, 'smoke.json'), 'w'), indent=1); return
    res = {}
    for ds in dsl:
        files = sorted(glob.glob(os.path.join(P(C['cache_dir']), ds, 'sub-*.npz')))
        if a.subjects: files = files[:a.subjects]
        subs = run_dataset(ds, files, a.mode, C); print(ds, 'done', f'{time.time() - t0:.0f}s', [s['seconds'] for s in subs], flush=True)
        if a.mode == 'info0':
            R = len(subs[0]['info0']); rows = []
            for r in range(R):
                st = maxT([s['info0'][r] for s in subs]); rows.append(st['p_maxT'])
            res[ds] = dict(p=rows, rate=float(np.mean(np.array(rows) <= C['alpha'])), n_subjects=len(subs))
        else:
            st = {k: maxT([s['tests'][k] for s in subs]) for k in subs[0]['tests'] if not k.endswith('_raw') and k != 'I_bacc'}
            inj, mde = ({}, {'evoked': None, 'induced': None}) if a.no_inject else run_inject(ds, files, subs, st['I'], C)
            res[ds] = dict(tests=st, inject=inj, mde80=mde, subjects=[s['sub'] for s in subs], n_trials=[s['n'] for s in subs],
                           I_bacc={u: [s['tests']['I_bacc'][u] for s in subs] for u in subs[0]['tests']['I_bacc']}, seconds=[s['seconds'] for s in subs])
        json.dump(res, open(os.path.join(out, f'{a.mode}{a.tag}_partial.json'), 'w'))
    if a.mode == 'info0':
        allp = np.concatenate([np.array(v['p']) for v in res.values()])
        summ = dict(datasets=res, pooled_rate=float(np.mean(allp <= C['alpha'])),
                    passed=bool(all(v['rate'] <= C['info0']['max_rate_each'] for v in res.values()) and C['info0']['pooled_range'][0] <= float(np.mean(allp <= C['alpha'])) <= C['info0']['pooled_range'][1]))
        from scipy.stats import kstest, binomtest
        for v_ in res.values():
            v_['ks_uniform_p'] = float(kstest(v_['p'], 'uniform').pvalue); k_ = int(np.sum(np.array(v_['p']) <= C['alpha']))
            v_['binom_p_vs_alpha'] = float(binomtest(k_, len(v_['p']), C['alpha'], alternative='greater').pvalue)
        json.dump(summ, open(os.path.join(out, f'info0{a.tag}.json'), 'w'), indent=1); print(json.dumps({k: v['rate'] for k, v in res.items()}), summ['pooled_rate'], summ['passed']); return
    ps = [res[d]['tests']['I']['p_maxT'] for d in dsl]; adj = holm(ps)
    if a.holm_from:                                                      # post-result diagnosis: Holm over the main run's seven data sets
        M = json.load(open(P(a.holm_from))); allds = M['_meta']['datasets']
        pall = [res[d]['tests']['I']['p_maxT'] if d in dsl else M[d]['tests']['I']['p_maxT'] for d in allds]; ha = holm(pall)
        adj = [ha[allds.index(d)] for d in dsl]
    for d, pa in zip(dsl, adj):
        res[d]['I_holm'] = pa; mx = [m for m in res[d]['mde80'].values()]
        mde_max = None if any(m is None for m in mx) else max(mx); res[d]['mde80_max'] = mde_max
        res[d]['class'] = classify(d, res[d], mde_max)
        print(d, 'I p', round(res[d]['tests']['I']['p_maxT'], 4), 'holm', round(pa, 4), 'best', res[d]['tests']['I']['best_unit'], round(res[d]['tests']['I']['best_kappa'], 4), res[d]['class'], flush=True)
    res['_meta'] = dict(config=a.config, seconds=round(time.time() - t0, 1), datasets=dsl, args=vars(a))
    f = os.path.join(out, f'verdict{a.tag}.json'); json.dump(res, open(f, 'w'), indent=1)
    print('sha256', hashlib.sha256(open(f, 'rb').read()).hexdigest())


if __name__ == '__main__':
    main()
