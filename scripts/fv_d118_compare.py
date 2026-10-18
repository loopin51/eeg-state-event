"""D-118 comparison (amendments 1-2), development concepts only, server CPU (gpuq).

Per arm (E1 bm, E2 atm, E3 mlp, E4 hive) x target (10) x seed (2): the development run's set-A predictions at ses123
(dev_scores.npz A_ses4_E_ses123, concept means of session 4). Read-out (amendment 2): per fold, a ridge from the arm's target
space to the OpenCLIP ViT-H-14 image embedding, fitted on the 14,540 non-development training images (penalty by 5-fold concept
CV, folds sha256("{concept}|D118-ridge") order i % 5); input side = the fold's fused embeddings of those images (E4) or their
ViT-H-14 image embeddings (E1-E3). Read-out predictions -> cosine to the 200 development names (names_text_P1) -> rank
percentile cells (target x concept). Seed mean = mean of the L2-normalised seed outputs after read-out.
Adoption (amendment 1): challenger vs E1 paired cell difference, pigeonhole 5th percentile > 0, point >= 0.01, and the same
sign in each seed separately; the low-level dependence D (pair test) is reported by fv_pairs per arm and joined here when given.
Report: raw MRP without read-out (E1-E3), development image retrieval top-1 among the 200 development concepts (candidate = mean
ViT-H-14 embedding of a concept's 10 images, after read-out), per-seed values.
Output: results/fv_d118/compare.json.

usage: python scripts/fv_d118_compare.py configs/fv_train.yaml [--pairs-d JSON]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, time, hashlib, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
from ese.things.data import Meta
from ese.things import verdict as V
ARMS = {'E1': 'bm', 'E2': 'atm', 'E3': 'mlp', 'E4': 'hive'}; SEEDS = [20261801, 20261802]; ALPHAS = [1e-1, 1, 10, 100, 1e3, 1e4]


def ridge_fit(X, Y, groups, alphas):
    """closed-form ridge with intercept; penalty by grouped CV (mean cosine of held-out predictions) -> (W, b, alpha, cv)"""
    def fit(Xa, Ya, a):
        mx, my = Xa.mean(0), Ya.mean(0); Xc = Xa - mx; W = np.linalg.solve(Xc.T @ Xc + a * np.eye(Xc.shape[1]), Xc.T @ (Ya - my)); return W, my - mx @ W
    cv = {}
    for a in alphas:
        sc = []
        for g in range(5):
            tr, te = groups != g, groups == g; W, b = fit(X[tr], Y[tr], a); sc.append(float((V.unit(X[te] @ W + b) * V.unit(Y[te])).sum(1).mean()))
        cv[a] = float(np.mean(sc))
    best = max(cv, key=cv.get); W, b = fit(X, Y, best); return W, b, best, cv


def folds(tc, meta):
    dci = np.array([meta.train_concepts.index(x) for x in meta.dev_concepts(tc['dev']['salt'], tc['dev']['n'])])
    nondev = np.flatnonzero(~np.isin(meta.img_concept, dci))
    order = {c: i for i, c in enumerate(sorted(set(meta.img_concept[nondev]), key=lambda j: hashlib.sha256(f'{meta.train_concepts[j]}|D118-ridge'.encode()).hexdigest()))}
    return nondev, np.array([order[meta.img_concept[r]] % 5 for r in nondev])


def readout_for(tc, meta, fused):
    """E4 read-out of one run: ridge fused -> ViT-H-14 on the non-development training images (penalty by grouped CV)"""
    nondev, grp = folds(tc, meta); Timg = np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['train_image'])).astype(np.float64)
    W, b, _, _ = ridge_fit(np.asarray(fused, np.float64)[nondev], Timg[nondev], grp, ALPHAS); return W, b


def run_dir(tc, cand, seed, t):
    m = [x for x in glob.glob(os.path.join(P(tc['out_root']), f'*_fv_dev_{cand}_s{seed}_t{t:02d}')) if os.path.exists(os.path.join(x, 'dev_scores.npz'))]
    assert len(m) == 1, (cand, seed, t, m); return m[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--pairs-d'); ap.add_argument('--arms', nargs='+', default=list(ARMS))
    a = ap.parse_args(); tc = yaml.safe_load(open(P(a.config))); t0 = time.time(); meta = Meta(P(tc['meta']))
    from fv_train import name_rows
    dev_c = meta.dev_concepts(tc['dev']['salt'], tc['dev']['n']); dci = np.array([meta.train_concepts.index(x) for x in dev_c])
    td = P(tc['targets']['dir']); Timg = np.load(os.path.join(td, tc['targets']['train_image'])).astype(np.float64)
    Tdev = V.unit(np.load(os.path.join(td, tc['targets']['names_text']))[name_rows(tc, meta, dev_c)].astype(np.float64))
    nondev, grp = folds(tc, meta); assert len(nondev) == 14540
    concept_img = V.unit(np.stack([Timg[meta.img_concept == j].mean(0) for j in dci]))       # development image-retrieval candidates
    W0, b0, al0, cv0 = ridge_fit(Timg[nondev], Timg[nondev], grp, ALPHAS)                    # E1-E3 read-out (target space = ViT-H-14)
    targets = tc['subjects']; res = dict(readout_E1_E3=dict(alpha=al0, cv=cv0), arms={}); cells = {}
    for arm in a.arms:
        cand = ARMS[arm]; per_seed = []; raw_seed = []; img_seed = []; meta_r = {}
        for seed in SEEDS:
            M, Mraw, IMG = np.full((len(targets), len(dci)), np.nan), np.full((len(targets), len(dci)), np.nan), []
            for i, t in enumerate(targets):
                d = run_dir(tc, cand, seed, t); z = np.load(os.path.join(d, 'dev_scores.npz')); E = z['A_ses4_E_ses123'].astype(np.float64); tr = z['A_ses4_true']
                if cand == 'hive':
                    Fz = z['fused_train_images'].astype(np.float64); W, b, al, cv = ridge_fit(Fz[nondev], Timg[nondev], grp, ALPHAS); meta_r[f's{seed}_t{t:02d}'] = dict(alpha=al, cv=cv)
                else: W, b = W0, b0; Mraw[i, tr] = V.cells(V.unit(E) @ Tdev.T, tr)
                R = V.unit(E @ W + b); M[i, tr] = V.cells(R @ Tdev.T, tr); IMG.append(float(np.mean(np.argmax(R @ concept_img.T, 1) == tr)))
            per_seed.append(M); raw_seed.append(Mraw); img_seed.append(float(np.mean(IMG)))
        # seed mean after read-out
        Ms = np.full((len(targets), len(dci)), np.nan)
        for i, t in enumerate(targets):
            Es, tr = [], None
            for seed in SEEDS:
                d = run_dir(tc, cand, seed, t); z = np.load(os.path.join(d, 'dev_scores.npz')); tr = z['A_ses4_true']
                if cand == 'hive':
                    Fz = z['fused_train_images'].astype(np.float64); W, b, _, _ = ridge_fit(Fz[nondev], Timg[nondev], grp, [meta_r[f's{seed}_t{t:02d}']['alpha']])
                else: W, b = W0, b0
                Es.append(V.unit(z['A_ses4_E_ses123'].astype(np.float64) @ W + b))
            Ms[i, tr] = V.cells(V.unit(np.mean(Es, 0)) @ Tdev.T, tr)
        cells[arm] = (Ms, per_seed)
        res['arms'][arm] = dict(candidate=cand, mrp_seedmean=float(np.nanmean(Ms)), mrp_per_seed=[float(np.nanmean(x)) for x in per_seed],
                                raw_mrp_per_seed=[float(np.nanmean(x)) if not np.isnan(x).all() else None for x in raw_seed], image_top1_per_seed=img_seed,
                                n_cells=int(np.isfinite(Ms).sum()), readout=meta_r or 'E1-E3 shared')
        print(arm, json.dumps({k: v for k, v in res['arms'][arm].items() if k != 'readout'}), f'({time.time() - t0:.0f}s)', flush=True)
    if 'E1' in cells:
        rng = np.random.default_rng(20261891); base, base_s = cells['E1']; res['adoption'] = {}
        for arm in [x for x in a.arms if x != 'E1']:
            Dm = cells[arm][0] - base; ok = np.isfinite(Dm); Dm = np.where(ok, Dm, 0.0)
            draws = []
            for _ in range(1000):
                si = rng.integers(0, Dm.shape[0], Dm.shape[0]); ci = rng.integers(0, Dm.shape[1], Dm.shape[1]); w = ok[np.ix_(si, ci)]
                draws.append(float(Dm[np.ix_(si, ci)][w].mean()))
            pt = float(Dm[ok].mean()); lo = float(np.percentile(draws, 5)); seeds_dir = [float(np.nanmean(cells[arm][1][k] - base_s[k])) for k in range(len(SEEDS))]
            res['adoption'][arm] = dict(point=pt, lower5=lo, per_seed=seeds_dir, passed_mrp=bool(lo > 0 and pt >= 0.01 and all(x > 0 for x in seeds_dir)))
        if a.pairs_d: res['pairs_D'] = json.load(open(P(a.pairs_d)))
    od = P('results/fv_d118'); os.makedirs(od, exist_ok=True); res['seconds'] = time.time() - t0
    json.dump(res, open(os.path.join(od, 'compare.json'), 'w'), indent=1); print(json.dumps(res.get('adoption', {})))


if __name__ == '__main__':
    main()
