"""F-V (D-112 sections 5.4-5.6) linear references for one target subject (CPU): DE + ridge and time-domain ridge baselines,
the frontal eye-channel meter and the pre-stimulus-window meter. All four are within-subject ridges from EEG features to the
ViT-H-14 image embedding, scored by cosine against the concept name text embeddings (same candidates as the decoder).

Input unit = per-image repetition mean over the given sessions (as the decoder), computed here from the single-trial cache
(sub-XX/ses-YY_training.npy, whitened, -0.2..1.0 s) so that the pre-stimulus window is available:
  de    : log band power (1-4, 4-8, 8-13, 13-30, 30-45 Hz; Welch, one 250-sample segment) x 63 channels, 0..1 s
  time  : 63 channels x 250 samples, 0..1 s
  eye   : Fp1, Fp2, AF7, AF8 x 250 samples, 0..1 s
  pre   : 63 channels x 50 samples, -0.2..0 s (baseline-corrected window: per-channel mean 0; RSVP order check)
Ridge: features and targets standardised / centred on the fit set; penalty by grouped CV over concepts (5 folds, sha256
order of "{folder}|{salt}"), score = mean cosine on the held-out folds (as stageB_b1_meters.fit_predict); dual form, one
eigendecomposition of the training kernel per fold (same solution as the primal ridge).

--mode dev: fit on the target's sessions 1-3, non-development concepts (calibration data of the decoder); evaluate on the
  development concepts (A: concepts in session 4, the D-112 development set; B: each development concept in its own session
  3 or 4, report only), candidates = the 200 development concept names. Writes results/baselines/fv_ridge_dev/t{S}.json and
  .npz (score matrices). The sealed test runs are never opened here (verdict mode belongs to the verdict job).

usage: python scripts/fv_ridge.py configs/fv_train.yaml --target S --mode dev [--features de time eye pre] [--fake-targets]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, hashlib, argparse, yaml, numpy as np
from scipy import linalg
from scipy.signal import welch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.things.data import Meta
from fv_train import name_rows, P

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]; ALPHAS = np.logspace(-1, 6, 15); EYE = ['Fp1', 'Fp2', 'AF7', 'AF8']


def image_means(cache, s, sessions, codes, tsel):
    """per-image repetition mean over `sessions` restricted to `codes`, time samples tsel -> (codes, X float32 [n, C, len(tsel)], n)
    (preallocated sum array: one copy of the result in memory)"""
    evs = [(e, np.load(os.path.join(cache, f'sub-{s:02d}', f'ses-{e:02d}_training_events.npz'))) for e in sessions]
    cs = np.unique(np.concatenate([v['code'][np.isin(v['code'], codes)] for _, v in evs])); pos = np.full(int(cs.max()) + 1, -1); pos[cs] = np.arange(len(cs))
    Xs = None; n = np.zeros(len(cs), np.int64)
    for e, v in evs:
        X = np.load(os.path.join(cache, f'sub-{s:02d}', f'ses-{e:02d}_training.npy'), mmap_mode='r'); code, rep = v['code'], v['rep']
        if Xs is None: Xs = np.zeros((len(cs), X.shape[1], len(tsel)), np.float32)
        for r in np.unique(rep):                                                   # codes are unique inside one repetition index
            keep = np.where(np.isin(code, codes) & (rep == r))[0]
            for a in range(0, len(keep), 4096):
                ii = keep[a:a + 4096]; Xs[pos[code[ii]]] += np.asarray(X[ii][:, :, tsel], np.float32)
            n[pos[code[keep]]] += 1
    Xs /= n[:, None, None]; return cs, Xs, n


def features(kind, X, ch, fs):
    if kind == 'de':
        fr, p = welch(X, fs=fs, nperseg=X.shape[-1], axis=-1)
        return np.log(np.stack([p[..., (fr >= lo) & (fr < hi)].mean(-1) for lo, hi in BANDS], -1) + 1e-12).reshape(len(X), -1)
    if kind == 'eye': return X[:, [ch.index(c) for c in EYE]].reshape(len(X), -1)
    return X.reshape(len(X), -1)                                                   # time, pre


def unit(A): return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def _solver(Zt, Yt):
    """ridge fit on (Zt, Yt) reused for every alpha, by an eigendecomposition of the smaller Gram matrix:
    features <= samples: G = Zt' Zt (p x p), coefficients Q diag(1/(lam + a)) Q' Zt' Yt;
    otherwise the kernel K = Zt Zt' (n x n), coefficients Zt' Q diag(1/(lam + a)) Q' Yt. Returns (basis, projected targets, eigenvalues)
    with coefficients = basis @ (proj / (lam + a))."""
    if Zt.shape[1] <= Zt.shape[0]:
        lam, Q = linalg.eigh(Zt.T @ Zt, driver='evd', overwrite_a=True); return Q, Q.T @ (Zt.T @ Yt), np.maximum(lam, 0)
    lam, Q = linalg.eigh(Zt @ Zt.T, driver='evd', overwrite_a=True); return Zt.T @ Q, Q.T @ Yt, np.maximum(lam, 0)


def ridge_cv(X, Y, groups):
    """ridge with the penalty chosen by grouped CV (score = mean held-out cosine, as stageB_b1_meters.fit_predict). Every fit
    uses the eigendecomposition of the smaller Gram matrix (_solver), the same solution as the primal SVD form (checked on
    random data in both regimes: max prediction difference about 1e-5 in float32)."""
    xm, xs = X.mean(0, dtype=np.float64).astype(np.float32), (X.std(0, dtype=np.float64) + 1e-8).astype(np.float32); ym = Y.mean(0); Z = np.asarray(X, np.float32) - xm; Z /= xs
    Yc = (Y - ym).astype(np.float32); sc = np.zeros(len(ALPHAS)); G = np.unique(groups)
    for g in G:
        tr, va = np.where(groups != g)[0], np.where(groups == g)[0]; Bas, R, d = _solver(Z[tr], Yc[tr]); A = Z[va] @ Bas; del Bas
        for j, a in enumerate(ALPHAS): sc[j] += np.mean(np.sum(unit(A @ (R / (d + a)[:, None])) * unit(Yc[va]), 1))
    a = ALPHAS[int(np.argmax(sc))]; Bas, R, d = _solver(Z, Yc); B = (Bas @ (R / (d + a)[:, None])).astype(np.float32); del Bas, Z
    return (lambda Xn: ((Xn - xm) / xs).astype(np.float32) @ B + ym), float(a), (sc / len(G)).tolist()


def concept_means(codes, X, nrep, img_concept, dci):
    ci = img_concept[codes - 1]; keep, out = [], []
    for j in dci:
        m = ci == j
        if m.any(): keep.append(j); out.append((X[m] * nrep[m].reshape((-1,) + (1,) * (X.ndim - 1))).sum(0) / nrep[m].sum())
    return np.array(keep), np.stack(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--target', type=int, required=True); ap.add_argument('--mode', choices=['dev'], required=True)
    ap.add_argument('--features', nargs='+', default=['de', 'time', 'eye', 'pre']); ap.add_argument('--fake-targets', action='store_true'); ap.add_argument('--out-dir'); a = ap.parse_args()
    c = yaml.safe_load(open(P(a.config))); pc = yaml.safe_load(open(P(c['prep_config']))); cache = P(pc['cache_dir']); t0 = time.time()
    meta = Meta(P(c['meta'])); dev_c = meta.dev_concepts(c['dev']['salt'], c['dev']['n']); dci = np.array([meta.train_concepts.index(x) for x in dev_c])
    fit_c = [x for x in meta.train_concepts if x not in set(dev_c)]; fit_codes, dev_codes = meta.codes_of(fit_c), meta.codes_of(dev_c)
    fold = {x: i % 5 for i, x in enumerate(sorted(fit_c, key=lambda x: hashlib.sha256(f'{x}|{c["ridge"]["salt"]}'.encode()).hexdigest()))}
    if a.fake_targets: Timg = np.random.default_rng(0).standard_normal((16540, 1024)).astype(np.float32); cand = np.random.default_rng(1).standard_normal((200, 1024)).astype(np.float32)
    else:
        td = P(c['targets']['dir']); Timg = np.load(os.path.join(td, c['targets']['train_image'])); cand = np.load(os.path.join(td, c['targets']['names_text']))[name_rows(c, meta, dev_c)]
    ch = json.load(open(os.path.join(cache, 'channels.json')))['ch_names']; times = np.load(os.path.join(cache, f'sub-{a.target:02d}', 'ses-01_mvnn.npz'))['times']
    win = {'pre': np.where(times < 0)[0]}; post = np.where(times >= c['ridge']['input_tmin'] - 1e-9)[0]; fs = pc['epoch']['sfreq']
    res = dict(target=a.target, mode=a.mode, fake_targets=a.fake_targets, n_dev_concepts=len(dev_c), features={}); arrays = {}
    cache_X = {}
    for kind in a.features:
        tsel = win.get(kind, post); key = 'pre' if kind == 'pre' else 'post'
        if key not in cache_X:
            fc, FX, _ = image_means(cache, a.target, [1, 2, 3], fit_codes, tsel)
            e4c, E4, n4 = image_means(cache, a.target, [4], dev_codes, tsel); e3c, E3, n3 = image_means(cache, a.target, [3], dev_codes, tsel)
            cache_X[key] = (fc, FX, e4c, E4, n4, e3c, E3, n3)
        fc, FX, e4c, E4, n4, e3c, E3, n3 = cache_X[key]; t1 = time.time()
        pred, alpha, cv = ridge_cv(features(kind, FX, ch, fs), Timg[fc - 1], np.array([fold[meta.train_concepts[meta.img_concept[k - 1]]] for k in fc]))
        out = res['features'][kind] = dict(alpha=alpha, cv_score=cv, n_fit_images=int(len(fc)), n_features=int(features(kind, FX[:2], ch, fs).shape[1]))
        for setname, parts in (('A_ses4', [(e4c, E4, n4)]), ('B_ses34', [(e3c, E3, n3), (e4c, E4, n4)])):
            keep_all, Xc_all = [], []
            for cc, XX, nn in parts:
                k, Xc = concept_means(cc, XX, nn, meta.img_concept, dci); keep_all += list(k); Xc_all.append(Xc)
            keep = np.array(keep_all); assert len(set(keep)) == len(keep), 'a development concept in both sessions 3 and 4'
            Sm = unit(pred(features(kind, np.concatenate(Xc_all), ch, fs))) @ unit(cand).T; true = np.array([list(dci).index(j) for j in keep])
            rk = (Sm > Sm[np.arange(len(Sm)), true][:, None]).sum(1); arrays[f'{kind}_{setname}_S'] = Sm.astype(np.float32); arrays[f'{kind}_{setname}_true'] = true
            out[setname] = dict(n_concepts=int(len(keep)), mrp=float(np.mean(1 - rk / (Sm.shape[1] - 1))), top1=float(np.mean(rk < 1)), top5=float(np.mean(rk < 5)))
        print(f'{kind}: alpha {alpha:g} A {out["A_ses4"]["mrp"]:.4f} B {out["B_ses34"]["mrp"]:.4f} ({time.time() - t1:.0f}s)', flush=True)
    import resource; res['peak_rss_gb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6; res['alpha_grid'] = [float(ALPHAS[0]), float(ALPHAS[-1])]
    res['seconds'] = time.time() - t0; o = a.out_dir or os.path.join(ROOT, 'results/baselines', 'fv_ridge_dev' + ('_fake' if a.fake_targets else '')); os.makedirs(o, exist_ok=True)
    json.dump(res, open(os.path.join(o, f't{a.target:02d}.json'), 'w'), indent=1); np.savez_compressed(os.path.join(o, f't{a.target:02d}.npz'), **arrays)
    print('SUMMARY', json.dumps({k: {s: round(v[s]['mrp'], 4) for s in ('A_ses4', 'B_ses34')} for k, v in res['features'].items()}), flush=True)


if __name__ == '__main__':
    main()
