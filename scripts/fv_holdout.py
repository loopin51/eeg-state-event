"""D-115 amendment 1: held-out blur predictors H of the centred image CLIP embedding (no EEG, server gpuq CPU).

Target: C = original image CLIP minus the mean over the 14,540 non-development training images, L2-normalised (the D-115 space).
Fit rows: those 14,540 images; penalties by concept-level 5-fold CV (fold salt 20262131, mean cosine on the held-out folds).
- H_a: blurred (sigma 8) image CLIP -> C, kernel ridge with an RBF kernel exp(-gamma |x - y|^2); gamma = gamma_rel / median squared
  distance among every median_stride-th fit row; the target mean of the training rows is removed and added back.
- H_b: AlexNet fc7 of the blurred image -> C, ridge (scripts/fv_gate.py ridge_fit, fv_eegfree.yaml ridge.alphas).
- H_c: AlexNet conv5 channel mean of the blurred image -> C, ridge (simulation only).
H_b and H_c need the fv_embed_blur.py run folder (holdout.features_dir); a predictor whose input is missing is skipped. Each
predictor is cached in <out dir>/holdout_<H>.npz with its settings; a cached predictor with the same settings is reused.
Outputs: holdout_pred.npz (keys H_a / H_b / H_c [2000, 1024] float32 L2-normalised, development rows; rows = fv_clip index rows) and
the summary json (CV tables, chosen penalties, sha256, development name MRP: concept row = mean of its 10 image predictions,
normalised, against the 200 development names P1, ese.things.verdict.cells; the same for C itself as a reference in this space).
usage: python scripts/fv_holdout.py configs/fv_pairs_build.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import csv, hashlib, json, resource, sys, time
import numpy as np, yaml
from scipy.linalg import eigh, solve
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'src'))
from fv_gate import ridge_fit, sha256
from ese.things.data import Meta
from ese.things.verdict import cells, unit
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def folds_of(groups, k, salt):
    ug = sorted(set(groups), key=lambda g: hashlib.sha256(f'{g}|{salt}'.encode()).hexdigest())
    f = {g: i % k for i, g in enumerate(ug)}; return np.array([f[g] for g in groups])


def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0)


def cos_mean(Pr, Y): return float((unit(Pr) * unit(Y)).sum(1).mean())


def krr(X, Y, f, spec, t0):
    """RBF kernel ridge; returns predict(Z) and info"""
    Xs = X[::spec['median_stride']]; D = sqdist(Xs, Xs); med = float(np.median(D[np.triu_indices(len(Xs), 1)]))
    gammas = [g / med for g in spec['gamma_rel']]; al = spec['alphas']; cv = np.zeros((len(gammas), len(al)))
    for k in range(f.max() + 1):
        tr, va = f != k, f == k; ym = Y[tr].mean(0); Yc = Y[tr] - ym; Dtr = sqdist(X[tr], X[tr]); Dva = sqdist(X[va], X[tr])
        for gi, g in enumerate(gammas):
            w, V = eigh(np.exp(-g * Dtr)); VtY = V.T @ Yc; Kva = np.exp(-g * Dva)
            for ai, a in enumerate(al): cv[gi, ai] += cos_mean(Kva @ (V @ (VtY / (w + a)[:, None])) + ym, Y[va]) / (f.max() + 1)
            print(f'H_a fold {k} gamma_rel {spec["gamma_rel"][gi]} cv so far {np.round(cv[gi], 4).tolist()} {time.time() - t0:.0f}s', flush=True)
        del Dtr, Dva
    gi, ai = np.unravel_index(int(np.argmax(cv)), cv.shape); g, a = gammas[gi], al[ai]
    ym = Y.mean(0); coef = solve(np.exp(-g * sqdist(X, X)) + a * np.eye(len(X)), Y - ym, assume_a='pos')
    info = dict(kind='rbf kernel ridge', median_sqdist=med, gamma_rel=spec['gamma_rel'][gi], gamma=g, alpha=a,
                cv_cos=[[round(v, 5) for v in r] for r in cv.tolist()], cv_axes=dict(rows='gamma_rel', cols='alphas'))
    return (lambda Z: np.exp(-g * sqdist(Z, X)) @ coef + ym), info


def main():
    t0 = time.time(); c = yaml.safe_load(open(P(sys.argv[1]))); h = c['holdout']
    ef = yaml.safe_load(open(P(c['eegfree_config']))); tc = yaml.safe_load(open(P(c['train_config'])))
    E = P(ef['embed_out']); idx = json.load(open(os.path.join(E, 'index.json'))); meta = Meta(P(tc['meta']))
    split, concept = np.array(idx['split']), np.array(idx['concept'])
    assert list(concept[split == 'train']) == meta.train_img_concept
    dev = meta.dev_concepts(tc['dev']['salt'], tc['dev']['n'])
    fit = (split == 'train') & ~np.isin(concept, dev); assert fit.sum() == 14540
    rows = np.where(np.isin(concept, dev))[0]; assert len(rows) == 2000
    C0 = np.load(os.path.join(E, c['space'])).astype(np.float64); mu = C0[fit].mean(0); C = unit(C0 - mu)
    f = folds_of(list(concept[fit]), h['folds'], h['fold_salt'])
    names = list(csv.DictReader(open(P(tc['targets']['names_table'])), delimiter='\t')); row_of = {(r['split'], r['folder']): i for i, r in enumerate(names)}
    T = unit(np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['names_text'])).astype(np.float64))[[row_of[('train', d)] for d in dev]]
    dev_img = [np.where(concept[rows] == d)[0] for d in dev]

    def name_mrp(Pd):    # Pd: [2000, d] development-row predictions in the C space
        return float(cells(unit(np.stack([unit(Pd[ix]).mean(0) for ix in dev_img])) @ T.T, np.arange(len(dev))).mean())

    feats = {}
    if h.get('features_dir'):
        fd = P(h['features_dir']); rj = json.load(open(os.path.join(fd, 'rows.json')))
        assert rj['train_rows'] == list(range(16540)) and rj['dev_rows'] == rows.tolist(), 'fv_embed_blur rows differ from fv_clip index rows'
        for H in ('H_b', 'H_c'): feats[H] = (os.path.join(fd, h[H]['input']), np.load(os.path.join(fd, h[H]['input'])).astype(np.float64))
    feats['H_a'] = (os.path.join(E, h['H_a']['input']), np.load(os.path.join(E, h['H_a']['input'])).astype(np.float64))
    od = os.path.dirname(P(h['out'])); os.makedirs(od, exist_ok=True); preds, summ = {}, {}
    for H in ('H_a', 'H_b', 'H_c'):
        if H not in feats: summ[H] = 'not computed: holdout.features_dir is null'; continue
        path, X = feats[H]; key = dict(spec=h[H], fold_salt=h['fold_salt'], folds=h['folds'], input_sha256=sha256(path), alphas=ef['ridge']['alphas'] if H != 'H_a' else None)
        cache = os.path.join(od, f'holdout_{H}.npz')
        if os.path.exists(cache) and json.loads(str(np.load(cache)['key'])) == key:
            z = np.load(cache); pd_, info = z['pred'].astype(np.float64), json.loads(str(z['info'])); info['cached'] = True
        else:
            Xf = X[:16540] if len(X) >= 16540 else X; assert len(Xf) == 16540
            if H == 'H_a': pr, info = krr(Xf[fit[:16540]], C[fit], f, h['H_a'], t0)
            else: pr, info = ridge_fit(Xf[fit[:16540]], C[fit], ef['ridge']['alphas'], list(concept[fit]), h['folds'], h['fold_salt']); info = dict(kind='ridge', **info)
            pd_ = unit(pr(Xf[rows])); info.update(n_features=int(X.shape[1]), input=os.path.relpath(path, ROOT), input_sha256=key['input_sha256'], seconds=round(time.time() - t0, 1))
            np.savez(cache, pred=pd_.astype(np.float32), key=json.dumps(key), info=json.dumps(info))
        info['dev_name_mrp'] = name_mrp(pd_); info['dev_cos_to_C'] = cos_mean(pd_, C[rows]); preds[H] = pd_.astype(np.float32); summ[H] = info
        print(H, json.dumps({k: v for k, v in info.items() if k != 'cv_cos'}), f'{time.time() - t0:.0f}s', flush=True)
    out = P(h['out']); np.savez(out, rows=rows, **preds)
    res = dict(stage='D-115 amendment 1 held-out blur predictors (no EEG)', config=sys.argv[1], fit_images=int(fit.sum()), dev_images=len(rows),
               target='C = clip_img minus the fit-image mean, L2-normalised', fold_salt=h['fold_salt'], predictors=summ,
               reference_C_dev_name_mrp=name_mrp(C[rows]), out=dict(path=h['out'], sha256=sha256(out), keys=list(preds) + ['rows']),
               seconds=round(time.time() - t0, 1), peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
    json.dump(res, open(P(h['summary']), 'w'), indent=1); print(json.dumps({k: v for k, v in res.items() if k != 'predictors'}), flush=True)


if __name__ == '__main__':
    main()
