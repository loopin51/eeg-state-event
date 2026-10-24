"""D-123 verdict (stage 1): (a) within-modality concept decoding on the test blocks (gate), (b) cross-modal transfer among the passed
modalities. Settings come from results/word_ds004306/explore.json (selected combo and windows, amendment 2). Server CPU (gpuq).

Features: band log-variance (5 bands, 620) of the EEG in the window of the trial's own modality.
  (a) z-score per session with the statistics of the development perception trials at that window (as in the exploration)
  (b) z-score per session x modality with the development statistics of that modality (centres every modality)
Reader: PCA 64 fitted on the (pseudo-averaged) training set -> shrinkage LDA. n_avg = 4 pseudo-trials (target training part and
the pooled other-subject part separately, test part separately). pool = all.
  (a) train: target development blocks + other subjects' development blocks of modality m; test: target test blocks of m
  (b) train: all blocks of the other passed modalities (target + other subjects); test: target test blocks of m
Score: balanced accuracy - mean over n_perm training-label permutations (inside subject x session x modality strata, before
averaging). Observed = mean of `draws` pseudo-trial draws; each permutation uses one draw (conservative null).
Family: Holm over a_image, a_t, a_a (alpha 0.05, excess >= sesoi); b opened when >= 2 a adopted, tested at alpha * r / 3.
usage: python scripts/word_verdict.py configs/sem_word.yaml --stage {smoke,sim,verdict} [--workers 8]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, csv, json, time, argparse, yaml, zlib, numpy as np
from multiprocessing import Pool
from scipy.signal import butter, sosfiltfilt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from sem_verdict import lw_cov, lda_fit, lda_score, pca_fit, perm_labels, group, holm, fsha, P, knn_time_pred, NC
from sem_prep import phase_rand

_D = {}


# ------------------------------------------------------------------------------------------------------------------ features
def de(ep, t, w, bands_sos):
    i = np.flatnonzero((t >= w[0] - 1e-9) & (t < w[1] - 1e-9))
    return np.concatenate([np.log(sosfiltfilt(sb, ep, axis=-1)[..., i].var(-1) + 1e-12) for sb in bands_sos], 1)


def gfp4(ep, t, w):
    i = np.flatnonzero((t >= w[0] - 1e-9) & (t < w[1] - 1e-9)); g = ep.std(1)
    return np.stack([g[:, k].mean(-1) for k in np.array_split(i, 4)], 1)


def zs_dev(X, dev, groups):
    out = np.empty_like(X)
    for gi in np.unique(groups):
        r = groups == gi; d = r & dev; mu, sd = X[d].mean(0), X[d].std(0) + 1e-8; out[r] = (X[r] - mu) / sd
    return out


def load(c, wins, smoke=False):
    pj = json.load(open(os.path.join(P(c['cache_dir']), 'prep.json'))); e = c['explore']
    sos = [butter(4, b, btype='band', fs=128.0, output='sos') for b in c['prep']['de_bands']]
    subs = {}
    for m in pj['sessions']:
        if m.get('missing'):
            continue
        f = P(m['file']); assert fsha(f) == m['sha256'], f
        z = np.load(f); ok = z['p_ok']; ep = z['p_ep'][ok].astype(np.float64); eo = z['p_eog_ep'][ok].astype(np.float64); t = z['p_t']
        if smoke:
            ep = ep[:, :16]
        mod = z['modality'][ok]; dev = (z['block'][ok] % 2) == e['dev_parity']
        rng = np.random.default_rng([c['prep']['phase_seed'], zlib.crc32(f"{m['sub']}_{m['ses']}_word".encode())])
        eph = phase_rand(ep, rng)
        F = {}
        for mi in range(3):
            w = wins[mi]
            F[('de', mi)] = de(ep, t, w, sos); F[('de_ph', mi)] = de(eph, t, w, sos); F[('eog', mi)] = de(eo, t, w, sos); F[('gfp', mi)] = gfp4(ep, t, w)
        F[('pre', 0)] = de(ep, t, (-0.5, 0.0), sos)
        rec = dict(concept=z['concept'][ok], modality=mod, dev=dev, onset=z['onset'][ok], block=z['block'][ok], F=F)
        S = subs.setdefault(m['sub'], []); rec['ses'] = np.full(ok.sum(), len(S)); S.append(rec)
    out = {}
    for s, parts in subs.items():
        R = {k: np.concatenate([p[k] for p in parts]) for k in ('concept', 'modality', 'dev', 'onset', 'block', 'ses')}
        allF = {k: np.concatenate([p['F'][k] for p in parts]) for k in parts[0]['F']}
        # (a) session-level z with development statistics at each window; (b) session x modality z
        R['A'] = {k: zs_dev(v, R['dev'], R['ses']) for k, v in allF.items()}
        R['B'] = {k: zs_dev(v, R['dev'], R['ses'] * 10 + R['modality']) for k, v in allF.items()}
        out[s] = R
    return out


# ------------------------------------------------------------------------------------------------------------------- reader
def pseudo(X, y, n, rng, strata=None):
    """n same-label trials averaged; with strata the groups never cross strata (target part)."""
    if n == 1:
        return X, y
    Xs, ys = [], []
    st = np.zeros(len(y), int) if strata is None else strata
    for s_ in np.unique(st):
        for k in range(NC):
            i = rng.permutation(np.flatnonzero((y == k) & (st == s_))); g = len(i) // n
            if g:
                Xs.append(X[i[:g * n]].reshape(g, n, -1).mean(1)); ys += [k] * g
    return (np.concatenate(Xs), np.array(ys)) if Xs else (X[:0], y[:0])


def bacc(pred, y):
    return float(np.mean([np.mean(pred[y == k] == k) for k in range(NC) if (y == k).any()]))


def fit(X, y, ncomp=64):
    k = min(ncomp, X.shape[1], X.shape[0] - 1); mu, V = pca_fit(X, k); return mu, V, lda_fit((X - mu) @ V, y)


def predict(mdl, X):
    mu, V, l = mdl; return lda_score(l, (X - mu) @ V).argmax(1)


def build(s, scheme, key_of, train_mods, test_mod, train_blocks, test_feat=None):
    """Training parts (target, others) and test part for subject s. key_of(m) -> feature key for modality m trials."""
    D = _D; T = D[s]
    def rows(R, mods, blocks):
        r = np.isin(R['modality'], mods)
        if blocks == 'dev':
            r &= R['dev']
        return np.flatnonzero(r)
    def feats(R, idx):
        X = np.zeros((len(idx), R[scheme][key_of(0)].shape[1]))
        for mi in range(3):
            j = R['modality'][idx] == mi
            if j.any():
                X[j] = R[scheme][key_of(mi)][idx[j]]
        return X
    it = rows(T, train_mods, train_blocks); Xt, yt, st = feats(T, it), T['concept'][it], T['ses'][it] * 10 + T['modality'][it]
    Xo, yo, so = [], [], []
    for j, o in enumerate(sorted(D)):
        if o == s:
            continue
        R = D[o]; io = rows(R, train_mods, train_blocks); Xo.append(feats(R, io)); yo.append(R['concept'][io]); so.append(j * 1000 + R['ses'][io] * 10 + R['modality'][io])
    ie = np.flatnonzero((T['modality'] == test_mod) & ~T['dev'])
    Xe = feats(T, ie) if test_feat is None else test_feat(T, ie)
    return (Xt, yt, st), (np.concatenate(Xo), np.concatenate(yo), np.concatenate(so)), (Xe, T['concept'][ie], ie)


def run_cell(tr_t, tr_o, te, n, draws, n_perm, rng, noise=False):
    """Observed (mean of draws) and permutation accuracies (one draw each) for one subject x test modality."""
    (Xt, yt, st), (Xo, yo, so), (Xe, ye, _) = tr_t, tr_o, te
    # amendment 3: the label-free PCA is fitted once per cell on the raw training trials (not per draw on pseudo-trials)
    Xall = np.concatenate([Xt, Xo]); mu, V = pca_fit(Xall, min(64, Xall.shape[1], Xall.shape[0] - 1))
    Xt, Xo, Xe = (Xt - mu) @ V, (Xo - mu) @ V, (Xe - mu) @ V
    def once(yt_, yo_):
        A, a = pseudo(Xt, yt_, n, rng); B, b = pseudo(Xo, yo_, n, rng)        # as in the exploration (sessions mixed)
        Xtr, ytr = np.concatenate([A, B]), np.concatenate([a, b]); E, e = pseudo(Xe, ye, n, rng)
        if noise:
            m_ = Xtr.mean(0); L = np.linalg.cholesky(lw_cov(Xtr - m_)); E = m_ + rng.standard_normal(E.shape) @ L.T
        return bacc(lda_score(lda_fit(Xtr, ytr), E).argmax(1), e)
    obs = float(np.mean([once(yt, yo) for _ in range(draws)]))
    perm = np.array([once(perm_labels(yt, st, rng), perm_labels(yo, so, rng)) for _ in range(n_perm)])
    return obs, perm


def time_cell(s, train_mods, test_mod, train_blocks, n_perm, rng):
    T = _D[s]; tr = np.flatnonzero(np.isin(T['modality'], train_mods) & (T['dev'] if train_blocks == 'dev' else True))
    te = np.flatnonzero((T['modality'] == test_mod) & ~T['dev']); st = T['ses'][tr] * 10 + T['modality'][tr]
    def acc(y):
        pr = np.zeros(len(te), int)
        for se in np.unique(T['ses'][te]):
            a, b = te[T['ses'][te] == se], tr[T['ses'][tr] == se]
            pr[T['ses'][te] == se] = knn_time_pred(T['onset'][b], y[T['ses'][tr] == se], T['onset'][a])
        return bacc(pr, T['concept'][te])
    y = T['concept'][tr]
    return acc(y), np.array([acc(perm_labels(y, st, rng)) for _ in range(n_perm)])


def subj_job(args):
    s, c, what, mods_b, n_perm, draws, seed = args
    rng = np.random.default_rng(seed); n = c['explore_sel']['n_avg']; out = {}
    for m in range(3):
        if 'a' in what and m in c['explore_sel']['open']:
            own = lambda k: (lambda mi: (k, mi))
            base = build(s, 'A', own('de'), [m], m, 'dev')
            out[('a', m)] = run_cell(*base, n, draws, n_perm, rng)
            if 'controls' in what:
                out[('a_noise', m)] = run_cell(*base, n, draws, n_perm, rng, noise=True)
                ph = build(s, 'A', own('de'), [m], m, 'dev', test_feat=lambda T, ie, m=m: T['A'][('de_ph', m)][ie])
                out[('a_phase', m)] = run_cell(*ph, n, draws, n_perm, rng)
            if 'meters' in what:
                out[('a_E', m)] = run_cell(*build(s, 'A', own('eog'), [m], m, 'dev'), n, draws, n_perm, rng)
                out[('a_P', m)] = run_cell(*build(s, 'A', lambda mi: ('pre', 0), [m], m, 'dev'), n, draws, n_perm, rng)
                out[('a_G', m)] = run_cell(*build(s, 'A', own('gfp'), [m], m, 'dev'), n, draws, n_perm, rng)
                out[('a_T', m)] = time_cell(s, [m], m, 'dev', n_perm, rng)
        if 'b' in what and m in mods_b:
            others = [x for x in mods_b if x != m]; own = lambda k: (lambda mi: (k, mi))
            base = build(s, 'B', own('de'), others, m, 'all')
            out[('b', m)] = run_cell(*base, n, draws, n_perm, rng)
            if 'controls' in what:
                out[('b_noise', m)] = run_cell(*base, n, draws, n_perm, rng, noise=True)
                ph = build(s, 'B', own('de'), others, m, 'all', test_feat=lambda T, ie, m=m: T['B'][('de_ph', m)][ie])
                out[('b_phase', m)] = run_cell(*ph, n, draws, n_perm, rng)
            if 'meters' in what:
                out[('b_E', m)] = run_cell(*build(s, 'B', own('eog'), others, m, 'all'), n, draws, n_perm, rng)
                out[('b_P', m)] = run_cell(*build(s, 'B', lambda mi: ('pre', 0), others, m, 'all'), n, draws, n_perm, rng)
                out[('b_G', m)] = run_cell(*build(s, 'B', own('gfp'), others, m, 'all'), n, draws, n_perm, rng)
                out[('b_T', m)] = time_cell(s, others, m, 'all', n_perm, rng)
    if 'baseline' in what:                       # DE + LR, D-121 style: own window, self only, single trials, development -> test
        from sklearn.linear_model import LogisticRegression
        for m in range(3):
            T = _D[s]; tr = np.flatnonzero((T['modality'] == m) & T['dev']); te = np.flatnonzero((T['modality'] == m) & ~T['dev'])
            X, Xe = T['A'][('de', m)][tr], T['A'][('de', m)][te]; y, ye = T['concept'][tr], T['concept'][te]; st = T['ses'][tr]
            fitp = lambda yy: bacc(LogisticRegression(C=c['classifier']['lr_C'], max_iter=2000).fit(X, yy).predict(Xe), ye)
            out[('base', m)] = (fitp(y), np.array([fitp(perm_labels(y, st, rng)) for _ in range(n_perm)]))
    return s, out


# ------------------------------------------------------------------------------------------------------------------- group
def G(res, subs, kind, mods, n_boot, seed):
    """Group excess over subjects (mean over the given test modalities), permutation p, subject bootstrap interval."""
    acc = np.array([[res[s][(kind, m)][0] for m in mods] for s in subs]); perm = np.array([[res[s][(kind, m)][1] for m in mods] for s in subs])
    shim = {s: {'x': (acc[i], perm[i])} for i, s in enumerate(subs)}
    out = group(shim, subs, 'x', n_boot, seed); out['modalities'] = [int(m) for m in mods]
    return out


def run_all(c, what, mods_b, n_perm, draws, seed, workers):
    subs = sorted(_D); jobs = [(s, c, what, mods_b, n_perm, draws, seed + i) for i, s in enumerate(subs)]
    with Pool(workers) as pool:
        res = dict(pool.map(subj_job, jobs))
    return subs, res


def gate(res, subs, c, n_boot, seed):
    opened = c['explore_sel']['open']; ga = {m: G(res, subs, 'a', [m], n_boot, seed) for m in opened}
    ps = np.array([ga[m]['p'] for m in opened]); adj = holm(ps) if len(ps) else ps
    passed = [m for m, p_ in zip(opened, adj) if p_ <= c['alpha'] and ga[m]['G'] >= c['sesoi']]
    return ga, dict(zip([int(m) for m in opened], adj.tolist())), passed


def randomize(rng):
    for s, R in _D.items():
        y = np.zeros(len(R['concept']), int)
        for g in np.unique(R['ses'] * 100 + R['modality'] * 10 + R['dev']):
            i = np.flatnonzero(R['ses'] * 100 + R['modality'] * 10 + R['dev'] == g); y[i] = rng.permutation(np.arange(len(i)) % NC)
        R['concept'] = y


def plant(delta, rng, base):
    """Shared concept pattern u_y (all subjects, modalities) + modality pattern v_{m,y}, per-feature N(0,1) x delta, on the de keys."""
    d = 620; u = rng.standard_normal((NC, d)); v = rng.standard_normal((3, NC, d))
    for s, R in _D.items():
        y = R['concept']
        for sch in ('A', 'B'):
            for mi in range(3):
                R[sch][('de', mi)] = base[s][sch][('de', mi)] + delta * (u[y] + v[R['modality'], y])


def main():
    global _D
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', required=True); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); c = yaml.safe_load(open(P(a.config))); od = P(c['out_dir']); sd = c['seeds']
    ex = json.load(open(os.path.join(od, 'explore.json')))
    names = c['modalities']; wins = [tuple(ex['selected'][nm]['window']) for nm in names]
    assert ex['chosen'] == dict(feature='de', reader='lda', n_avg=4, pool='all'), ex['chosen']     # the code implements this combo
    c['explore_sel'] = dict(n_avg=ex['chosen']['n_avg'], open=[i for i, nm in enumerate(names) if ex['selected'][nm]['open']])
    vc = c['verdict']; t0 = time.time()
    _D = load(c, wins, smoke=a.stage == 'smoke')
    if a.stage == 'smoke':
        _D = {s: _D[s] for s in sorted(_D)[:4]}; randomize(np.random.default_rng(0))     # no true-label score on test blocks
        subs, res = run_all(c, ('a', 'b', 'controls', 'meters', 'baseline'), [0, 1, 2], 5, 2, 1, min(a.workers, 4))
        print(json.dumps({f'{k[0]}_{k[1]}': [round(res[subs[0]][k][0], 3), round(float(res[subs[0]][k][1].mean()), 3)] for k in res[subs[0]]}), round(time.time() - t0), flush=True)
        return
    if a.stage == 'sim':
        sc = vc['sim']; rng = np.random.default_rng(sd['info0']); base = {s: {'A': dict(R['A']), 'B': dict(R['B'])} for s, R in _D.items()}
        out = dict(stage='sim', windows=wins, info0=[], power={})
        for r in range(sc['info0_reps']):
            randomize(rng); subs, res = run_all(c, ('a', 'b'), [0, 1, 2], sc['n_perm'], vc['draws'], sd['info0'] * 1000 + r, a.workers)
            ga, adj, passed = gate(res, subs, c, 200, r); gb = G(res, subs, 'b', [0, 1, 2], 200, r)
            out['info0'].append(dict(a={str(m): [ga[m]['G'], ga[m]['p']] for m in ga}, a_adopted=[int(m) for m in passed], b=[gb['G'], gb['p']],
                                     b_adopted=bool(gb['p'] <= c['alpha'] and gb['G'] >= c['sesoi'])))
            if r % 10 == 0:
                print('info0', r, round(time.time() - t0), flush=True)
        out['info0_rate'] = dict(a_any=float(np.mean([len(x['a_adopted']) > 0 for x in out['info0']])),
                                 a_each=[float(np.mean([m in x['a_adopted'] for x in out['info0']])) for m in range(3)],
                                 b_unconditional=float(np.mean([x['b_adopted'] for x in out['info0']])))
        rng = np.random.default_rng(sd['power'])
        for dl in sc['power_deltas']:
            rows = []
            for r in range(sc['power_reps']):
                randomize(rng); plant(dl, rng, base)
                subs, res = run_all(c, ('a', 'b'), [0, 1, 2], sc['n_perm'], vc['draws'], sd['power'] * 1000 + r, a.workers)
                ga, adj, passed = gate(res, subs, c, 200, r); gb = G(res, subs, 'b', [0, 1, 2], 200, r)
                rows.append(dict(aG=[ga[m]['G'] for m in sorted(ga)], a_n=len(passed), bG=gb['G'], b=bool(gb['p'] <= c['alpha'] * len(passed) / 3 and gb['G'] >= c['sesoi'] and len(passed) >= 2)))
            for s, R in _D.items():
                for sch in ('A', 'B'):
                    R[sch].update(base[s][sch])
            out['power'][str(dl)] = dict(aG_mean=np.mean([x['aG'] for x in rows], 0).tolist(), a_all3=float(np.mean([x['a_n'] == 3 for x in rows])),
                                         bG_mean=float(np.mean([x['bG'] for x in rows])), b_rate=float(np.mean([x['b'] for x in rows])))
            print('power', dl, out['power'][str(dl)], round(time.time() - t0), flush=True)
        out['info0_ok'] = bool(out['info0_rate']['a_any'] <= 0.05 and out['info0_rate']['b_unconditional'] <= 0.05)
        out['seconds'] = round(time.time() - t0)
        json.dump(out, open(os.path.join(od, 'sim.json'), 'w'), indent=1); print(json.dumps({k: v for k, v in out.items() if k != 'info0'}, indent=1))
        return
    if a.stage == 'verdict':
        sim = json.load(open(os.path.join(od, 'sim.json'))); assert sim['info0_ok']
        assert not os.path.exists(os.path.join(od, 'verdict.json')), 'verdict already run'
        subs, rb = run_all(c, ('baseline',), [], c['n_perm'], 1, sd['perm'] + 7, a.workers)
        base = {names[m]: G(rb, subs, 'base', [m], c['n_boot'], sd['boot']) for m in range(3)}
        json.dump(dict(dataset='ds004306', analysis='D-123 (a) within-modality, DE + LR, own window, single trials, development -> test blocks',
                       subjects=subs, per_modality=base, sim_sha256=fsha(os.path.join(od, 'sim.json'))), open(P(c['baseline_out']), 'w'), indent=1)
        subs, ra = run_all(c, ('a', 'controls', 'meters'), [], c['n_perm'], vc['draws'], sd['perm'], a.workers)
        ga, adj, passed = gate(ra, subs, c, c['n_boot'], sd['boot'])
        out = dict(stage='verdict', subjects=subs, windows=wins, explore_sha256=fsha(os.path.join(od, 'explore.json')), sim_sha256=fsha(os.path.join(od, 'sim.json')),
                   a={names[m]: ga[m] for m in ga}, a_holm={names[m]: v for m, v in adj.items()}, a_adopted=[names[m] for m in passed], baseline_DE_LR=base)
        def side(res, prefix, mods):
            r = {}
            for k in ('noise', 'phase'):
                g = G(res, subs, f'{prefix}_{k}', mods, c['n_boot'], sd['boot']); r[f'ctl_{k}'] = dict(G=g['G'], p=g['p'], ok=bool(abs(g['G']) < 0.01 and g['p'] > 0.05))
            g0 = G(res, subs, prefix, mods, c['n_boot'], sd['boot'])
            r['ctl_label_shuffle'] = float(np.mean([np.mean([res[s][(prefix, m)][1][0] - res[s][(prefix, m)][1].mean() for m in mods]) for s in subs]))
            r['ctl_no_eeg'] = 0.0
            for k in ('E', 'P', 'G', 'T'):
                g = G(res, subs, f'{prefix}_{k}', mods, c['n_boot'], sd['boot']); r[f'meter_{k}'] = dict(G=g['G'], p=g['p'], ok=bool(g['p'] > 0.05 or g['G'] < c['meter_max']))
            return r
        out['a_side'] = {names[m]: side(ra, 'a', [m]) for m in c['explore_sel']['open']}
        if len(passed) >= 2:
            subs, rbx = run_all(c, ('b', 'controls', 'meters'), passed, c['n_perm'], vc['draws'], sd['perm'] + 1, a.workers)
            gb = G(rbx, subs, 'b', passed, c['n_boot'], sd['boot']); alpha_b = c['alpha'] * len(passed) / 3
            out['b'] = dict(gb, alpha=alpha_b, adopted=bool(gb['p'] <= alpha_b and gb['G'] >= c['sesoi']),
                            per_test_modality={names[m]: G(rbx, subs, 'b', [m], c['n_boot'], sd['boot'])['G'] for m in passed})
            out['b_side'] = side(rbx, 'b', passed)
        else:
            out['b'] = dict(opened=False, reason=f'{len(passed)} modality passed the gate')
        out['power'] = sim['power']; out['seconds'] = round(time.time() - t0)
        json.dump(out, open(os.path.join(od, 'verdict.json'), 'w'), indent=1); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
