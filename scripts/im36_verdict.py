"""D-124 verdict: animal / tool category in the imagery windows (0.5-3.0 s after each task onset; the photo and the mask are gone)
for concepts held out of training, and on the concepts whose cue photo the EEG-free low-level predictor gets least right.
Server CPU (gpuq).
Reader: DE (64 ch x 5 bands) z-scored per participant (no labels) -> PCA 64 fitted on the training epochs -> shrinkage LDA. Training
pools the training-concept epochs of every participant of the same dataset (all four tasks); the test is each participant's
epochs of the held-out concepts (6 leave-concept-out folds, 3 animals + 3 tools each). A held-out concept never enters training
from any participant.
  K1  balanced accuracy over all held-out epochs - permutation mean
  K2  the same predictions restricted to the low-level incongruent concepts (results/im36/lowlevel.json)
Permutation: the concept -> category map is permuted (18 / 18) and the readers are retrained; the test labels stay true.
Group = mean over participants; p = (1 + #{G_k >= G}) / (K + 1); Holm over K1, K2; adopt at Holm p <= alpha and G >= sesoi.
Meters: C (reader trained on the cue-photo window of the training concepts, tested on the imagery windows), E (frontal channels),
P (pre-cue fixation), G (GFP), T (cue onset time, kNN). Controls: noise, phase randomisation, label-shuffle, no EEG.
Baseline DE + LR per participant (own epochs only). Report: per task, per position, ds004517 replication, low-level accuracy.
usage: python scripts/im36_verdict.py configs/im36.yaml --stage {smoke,sim,verdict} [--workers 8]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, json, time, argparse, yaml, numpy as np
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from sem_verdict import lw_cov, pca_fit, holm, fsha, P, knn_time_pred

KEYS = ('de', 'de_ph', 'front', 'gfp', 'cue', 'pre')
_S = {}


def lda2(X, y):
    M = np.stack([X[y == k].mean(0) for k in (0, 1)]); C = lw_cov(X - M[y]); W = np.linalg.solve(C, (M[1] - M[0])); b = -0.5 * (M[1] + M[0]) @ W
    return W, b


def bacc(pred, y):
    return float(np.mean([np.mean(pred[y == k] == k) for k in (0, 1) if (y == k).any()]))


def load(c, ds):
    pj = json.load(open(os.path.join(P(c['cache_dir']), 'prep.json'))); out = {}
    for m in pj['sessions']:
        if m['ds'] != ds:
            continue
        f = P(m['file']); assert fsha(f) == m['sha256'], f
        z = dict(np.load(f, allow_pickle=False)); ok = z['t_ok']
        R = {k: z[k][ok] for k in ('trial', 'position', 'task', 'delay', 'onset', 'category', 'concept')}
        for k in KEYS:
            X = z[k][ok].astype(np.float64); R[k] = (X - X.mean(0)) / (X.std(0) + 1e-8)
        out[m['sub']] = R
    return out


def fold_models(c, D, fo, cmap_list, key_tr, unique_trial=False):
    """For each fold: PCA (label free) on the pooled training epochs and one LDA per concept->category map in cmap_list."""
    subs = sorted(D); models = []
    for k in range(c['folds']):
        Xs, cs = [], []
        for s in subs:
            R = D[s]; tr = np.array([fo[x] != k for x in R['concept']])
            if unique_trial:                                   # cue / pre features are per trial: one row per trial
                _, first = np.unique(R['trial'][tr], return_index=True); idx = np.flatnonzero(tr)[first]
            else:
                idx = np.flatnonzero(tr)
            Xs.append(R[key_tr][idx]); cs.append(R['concept'][idx])
        X = np.concatenate(Xs); cn = np.concatenate(cs); mu, V = pca_fit(X, min(c['pca'], X.shape[1])); Z = (X - mu) @ V
        models.append([(mu, V, lda2(Z, np.array([cm[x] for x in cn]))) for cm in cmap_list])
    return models


def evaluate(c, D, fo, models, key_te, inc=None, transform=None, rng=None):
    """acc[s, k_model] for K1 (all held-out epochs) and K2 (incongruent concepts)."""
    subs = sorted(D); nm = len(models[0]); A1 = np.zeros((len(subs), nm)); A2 = np.zeros((len(subs), nm))
    for i, s in enumerate(subs):
        R = D[s]; pr = np.zeros((nm, len(R['category'])), int)
        for k in range(c['folds']):
            te = np.array([fo[x] == k for x in R['concept']]); Xte = R[key_te][te]
            if transform is not None:
                Xte = transform(Xte, s, k)
            for j, (mu, V, (W, b)) in enumerate(models[k]):
                pr[j, te] = (((Xte - mu) @ V) @ W + b > 0).astype(int)
        y = R['category']; m2 = np.isin(R['concept'], inc) if inc is not None else None
        for j in range(nm):
            A1[i, j] = bacc(pr[j], y)
            if m2 is not None:
                A2[i, j] = bacc(pr[j][m2], y[m2])
    return A1, A2


def stat(A, n_boot, seed):
    """A[s, 0] observed, A[s, 1:] permutations."""
    pm = A[:, 1:].mean(1); ex = A[:, 0] - pm; G = float(ex.mean()); Gk = (A[:, 1:] - pm[:, None]).mean(0)
    p = float((1 + (Gk >= G - 1e-12).sum()) / (len(Gk) + 1)); rng = np.random.default_rng(seed)
    bt = np.array([ex[rng.integers(0, len(ex), len(ex))].mean() for _ in range(n_boot)])
    return dict(G=G, p=p, lo95=float(np.percentile(bt, 2.5)), hi95=float(np.percentile(bt, 97.5)), per_subject=ex.round(4).tolist(),
                acc=float(A[:, 0].mean()))


def cmaps(concepts, cat_true, n_perm, rng):
    base = dict(zip(concepts, cat_true)); out = [base]
    for _ in range(n_perm):
        perm = rng.permutation(cat_true); out.append(dict(zip(concepts, perm)))
    return out


def time_meter(c, D, fo, maps):
    subs = sorted(D); A = np.zeros((len(subs), len(maps)))
    for i, s in enumerate(subs):
        R = D[s]; _, first = np.unique(R['trial'], return_index=True)
        for j, cm in enumerate(maps):
            pr = np.zeros(len(R['category']), int)
            for k in range(c['folds']):
                tr = np.array([fo[x] != k for x in R['concept'][first]]); te = np.array([fo[x] == k for x in R['concept']])
                yt = np.array([cm[x] for x in R['concept'][first][tr]])
                p_ = knn_time_pred(R['onset'][first][tr], yt, R['onset'][te]); pr[te] = np.minimum(p_, 1)
            A[i, j] = bacc(pr, R['category'])
    return A


def run(c, D, ll, n_perm, seed, what, cat_override=None):
    concepts = ll['concepts']; cat_true = np.array(ll['category']) if cat_override is None else cat_override
    fo = ll['fold']; inc = ll['incongruent']; rng = np.random.default_rng(seed); maps = cmaps(concepts, cat_true, n_perm, rng)
    if cat_override is not None:
        for s, R in D.items():
            R['category'] = np.array([dict(zip(concepts, cat_true))[x] for x in R['concept']])
    res = {}
    M = fold_models(c, D, fo, maps, 'de'); res['K1'], res['K2'] = evaluate(c, D, fo, M, 'de', inc)
    if 'controls' in what:
        res['ctl_phase_K1'], res['ctl_phase_K2'] = evaluate(c, D, fo, M, 'de_ph', inc)
        nrng = np.random.default_rng(c['seeds']['noise'])
        def noise(X, s, k):
            return nrng.standard_normal(X.shape)              # features are z-scored: unit Gaussian matches their marginal scale
        res['ctl_noise_K1'], res['ctl_noise_K2'] = evaluate(c, D, fo, M, 'de', inc, transform=noise)
    if 'meters' in what:
        Mc = fold_models(c, D, fo, maps, 'cue', unique_trial=True); res['meter_C_K1'], res['meter_C_K2'] = evaluate(c, D, fo, Mc, 'de', inc)
        Me = fold_models(c, D, fo, maps, 'front'); res['meter_E_K1'], res['meter_E_K2'] = evaluate(c, D, fo, Me, 'front', inc)
        Mp = fold_models(c, D, fo, maps, 'pre', unique_trial=True); res['meter_P_K1'], res['meter_P_K2'] = evaluate(c, D, fo, Mp, 'pre', inc)
        Mg = fold_models(c, D, fo, maps, 'gfp'); res['meter_G_K1'], res['meter_G_K2'] = evaluate(c, D, fo, Mg, 'gfp', inc)
        res['meter_T_K1'] = time_meter(c, D, fo, maps)
    return res


def plant(D, ll, delta, rng, base):
    u = rng.standard_normal((2, base[sorted(base)[0]].shape[1]))
    for s, R in D.items():
        R['de'] = base[s] + delta * u[R['category']]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', required=True); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); c = yaml.safe_load(open(P(a.config))); od = P(c['out_dir']); sd = c['seeds']; t0 = time.time()
    ll = json.load(open(os.path.join(od, 'lowlevel.json'))); D = load(c, 'ds004514')
    if a.stage == 'smoke':
        rng = np.random.default_rng(0); fake = rng.permutation(np.repeat([0, 1], 18))
        D = {s: D[s] for s in sorted(D)[:3]}; r = run(c, D, ll, 5, 1, ('controls', 'meters'), cat_override=fake)
        print({k: round(float(v[:, 0].mean() - v[:, 1:].mean()), 3) for k, v in r.items()}, round(time.time() - t0), flush=True)
        return
    if a.stage == 'sim':
        sc = c['sim']; rng = np.random.default_rng(sd['info0']); out = dict(stage='sim', info0=[], power={})
        for r in range(sc['info0_reps']):
            fake = rng.permutation(np.repeat([0, 1], 18)); res = run(c, D, ll, sc['n_perm'], sd['info0'] * 1000 + r, (), cat_override=fake)
            k1, k2 = stat(res['K1'], 200, r), stat(res['K2'], 200, r); adj = holm(np.array([k1['p'], k2['p']]))
            out['info0'].append(dict(K1=[k1['G'], k1['p']], K2=[k2['G'], k2['p']], adopt=[bool(adj[0] <= c['alpha'] and k1['G'] >= c['sesoi']), bool(adj[1] <= c['alpha'] and k2['G'] >= c['sesoi'])]))
            if r % 20 == 0:
                print('info0', r, round(time.time() - t0), flush=True)
        out['info0_rate'] = [float(np.mean([x['adopt'][i] for x in out['info0']])) for i in range(2)]
        rng = np.random.default_rng(sd['power']); base = {s: R['de'].copy() for s, R in D.items()}
        for dl in sc['power_deltas']:
            rows = []
            for r in range(sc['power_reps']):
                fake = rng.permutation(np.repeat([0, 1], 18))
                for s, R in D.items():
                    R['category'] = np.array([dict(zip(ll['concepts'], fake))[x] for x in R['concept']])
                plant(D, ll, dl, rng, base); res = run(c, D, ll, sc['n_perm'], sd['power'] * 1000 + r, (), cat_override=fake)
                k1, k2 = stat(res['K1'], 200, r), stat(res['K2'], 200, r); adj = holm(np.array([k1['p'], k2['p']]))
                rows.append(dict(G1=k1['G'], G2=k2['G'], a1=bool(adj[0] <= c['alpha'] and k1['G'] >= c['sesoi']), a2=bool(adj[1] <= c['alpha'] and k2['G'] >= c['sesoi'])))
            for s, R in D.items():
                R['de'] = base[s]
            out['power'][str(dl)] = dict(G1_mean=float(np.mean([x['G1'] for x in rows])), G2_mean=float(np.mean([x['G2'] for x in rows])),
                                         K1_rate=float(np.mean([x['a1'] for x in rows])), K2_rate=float(np.mean([x['a2'] for x in rows])))
            print('power', dl, out['power'][str(dl)], round(time.time() - t0), flush=True)
        out['info0_ok'] = all(v <= 0.05 for v in out['info0_rate']); out['seconds'] = round(time.time() - t0)
        json.dump(out, open(os.path.join(od, 'sim.json'), 'w'), indent=1); print(json.dumps({k: v for k, v in out.items() if k != 'info0'}, indent=1))
        return
    if a.stage == 'verdict':
        sim = json.load(open(os.path.join(od, 'sim.json'))); assert sim['info0_ok']; assert not os.path.exists(os.path.join(od, 'verdict.json'))
        from sklearn.linear_model import LogisticRegression
        fo = ll['fold']; base = {}
        for s, R in sorted(D.items()):
            pr = np.zeros(len(R['category']), int)
            for k in range(c['folds']):
                te = np.array([fo[x] == k for x in R['concept']]); pr[te] = LogisticRegression(C=c['classifier']['lr_C'], max_iter=3000).fit(R['de'][~te], R['category'][~te]).predict(R['de'][te])
            base[s] = bacc(pr, R['category'])
        json.dump(dict(dataset='ds004514', analysis='D-124 K1, DE + LR, own epochs, leave-concept-out', per_subject=base, mean_bacc=float(np.mean(list(base.values()))),
                       sim_sha256=fsha(os.path.join(od, 'sim.json'))), open(P(c['baseline_out']), 'w'), indent=1)
        res = run(c, D, ll, c['n_perm'], sd['perm'], ('controls', 'meters'))
        S = {k: stat(v, c['n_boot'], sd['boot']) for k, v in res.items()}; adj = holm(np.array([S['K1']['p'], S['K2']['p']]))
        out = dict(stage='verdict', subjects=sorted(D), lowlevel=dict(accuracy=ll['accuracy'], incongruent=ll['incongruent']),
                   K1=S['K1'], K2=S['K2'], holm_p=adj.tolist(), adopted=dict(K1=bool(adj[0] <= c['alpha'] and S['K1']['G'] >= c['sesoi']), K2=bool(adj[1] <= c['alpha'] and S['K2']['G'] >= c['sesoi'])),
                   meters={k: dict(G=v['G'], p=v['p'], ok=bool(v['p'] > 0.05 or v['G'] < c['meter_max'])) for k, v in S.items() if k.startswith('meter_')},
                   controls={k: dict(G=v['G'], p=v['p'], ok=bool(abs(v['G']) < 0.01 and v['p'] > 0.05)) for k, v in S.items() if k.startswith('ctl_')},
                   label_shuffle={k: float(np.mean(res[k][:, 1] - res[k][:, 2:].mean(1))) for k in ('K1', 'K2')}, no_eeg=0.0,
                   baseline_DE_LR=dict(mean_bacc=float(np.mean(list(base.values())))), sim_sha256=fsha(os.path.join(od, 'sim.json')),
                   cue_flag=bool(S['meter_C_K1']['p'] <= c['alpha'] and S['meter_C_K1']['G'] >= c['sesoi']))
        # report-only: per task and per position (observed reader, K1 predictions recomputed per subset)
        M = fold_models(c, D, fo, cmaps(ll['concepts'], np.array(ll['category']), 0, np.random.default_rng(0)), 'de'); rep = {}
        for name, sel in [(f'task{t}', lambda R, t=t: R['task'] == t) for t in range(4)] + [(f'pos{p}', lambda R, p=p: R['position'] == p) for p in range(4)]:
            v = []
            for s, R in sorted(D.items()):
                pr = np.zeros(len(R['category']), int)
                for k in range(c['folds']):
                    te = np.array([fo[x] == k for x in R['concept']]); mu, V, (W, b) = M[k][0]; pr[te] = (((R['de'][te] - mu) @ V) @ W + b > 0).astype(int)
                m_ = sel(R); v.append(bacc(pr[m_], R['category'][m_]))
            rep[name] = dict(mean_bacc=float(np.mean(v)), per_subject=np.round(v, 4).tolist())
        rep['delay_by_position'] = [float(np.mean(np.concatenate([R['delay'][R['position'] == p] for R in D.values()]))) for p in range(4)]
        D7 = load(c, 'ds004517')
        if D7:
            r7 = run(c, D7, ll, 200, sd['perm'] + 9, ()); rep['ds004517'] = dict(K1=stat(r7['K1'], 200, 1), K2=stat(r7['K2'], 200, 2), subjects=sorted(D7))
        out['report'] = rep; out['power'] = sim['power']; out['seconds'] = round(time.time() - t0)
        json.dump(out, open(os.path.join(od, 'verdict.json'), 'w'), indent=1); print(json.dumps({k: v for k, v in out.items()}, indent=1))


if __name__ == '__main__':
    main()
