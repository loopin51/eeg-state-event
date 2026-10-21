"""F-IX (D-119 amendment 1 G): linear references as predicted embeddings (server CPU, gpuq), for the method S read-outs.
Same features and penalty rule as scripts/fv_ridge.py (de, time, eye, pre; grouped 5-fold CV over concepts, salt FV-ridge):
  dev  : fit on the subject's sessions 1-3, non-development concepts; predict the 2,000 development photos (repetition mean over
         sessions 1-4, the rows of the D-115 decoder outputs)
  test : fit on sessions 1-3, all training concepts; predict the session-4 test images (mean of every session-4 repetition, as V1
         avg20). The test epochs are read again from the raw files (sha256 against the seal table) for the pre-stimulus window.
Direct meters (D-119 amendment 1, review 2 item 1): for the eye and pre features, softmax linear classifiers from the standardised
features straight to the labels (colour: 11 pixel fractions, category: 1/|L| over the 18 classes), L2 penalty by 3-fold concept CV
(salt FV-S) in dev mode, the same penalty in test mode -> logits dev_direct_<kind>_<col|cat> [2000, K], test_direct_<kind>_<col|cat> [200, K].
The session-4 test raw file is opened only with --unseal (sha256 against the seal table; D-119 amendment 1: this baseline job computes
no decoder score).
Output per subject: data/cache/fv_s/ridge_tXX.npz (dev_<kind> [2000, 1024], test_<kind> [200, 1024], raw ridge outputs, not
L2-normalised; dev_rows = training codes - 1) and results/fv_s/ridge_manifest.json (sha256, alphas, seconds).
usage: python scripts/fv_s_ridge.py configs/fv_s.yaml --unseal [--subjects 1 2 ...]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, hashlib, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.things.data import Meta
from fv_ridge import image_means, features, ridge_cv
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
KINDS = {'post': ('de', 'time', 'eye'), 'pre': ('pre',)}


def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def peak_gb():
    import resource; return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--subjects', type=int, nargs='+'); ap.add_argument('--unseal', action='store_true'); a = ap.parse_args()
    assert a.unseal, 'this job re-opens the session-4 test raw files: pass --unseal (D-119 amendment 1)'
    from fv_s import Setup, fit_softmax, ce, salt_folds
    st = Setup(yaml.safe_load(open(P(a.config)))); dm = yaml.safe_load(open(P(a.config)))['direct']
    c = yaml.safe_load(open(P(a.config))); tc = yaml.safe_load(open(P(c['train_config']))); vc = yaml.safe_load(open(P('configs/fv_verdict.yaml')))
    pc = yaml.safe_load(open(P(tc['prep_config']))); cache = P(pc['cache_dir']); root = P(pc['data_root']); meta = Meta(P(tc['meta'])); fs = pc['epoch']['sfreq']
    from ese.things import preprocess as PP
    dev_c = meta.dev_concepts(tc['dev']['salt'], tc['dev']['n']); fit_c = [x for x in meta.train_concepts if x not in set(dev_c)]
    fit_codes, dev_codes, all_codes = meta.codes_of(fit_c), meta.codes_of(dev_c), np.arange(1, len(meta.img_concept) + 1)
    salt = tc['ridge']['salt']; foldf = lambda cs: {x: i % 5 for i, x in enumerate(sorted(cs, key=lambda x: hashlib.sha256(f'{x}|{salt}'.encode()).hexdigest()))}
    fold_dev, fold_all = foldf(fit_c), foldf(meta.train_concepts); grp = lambda fc, fd: np.array([fd[meta.train_concepts[meta.img_concept[k - 1]]] for k in fc])
    Timg = np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['train_image'])); ch = json.load(open(os.path.join(cache, 'channels.json')))['ch_names']
    od = P(c['cache_dir']); os.makedirs(od, exist_ok=True); mf = P(os.path.join(c['out_dir'], 'ridge_manifest.json')); os.makedirs(os.path.dirname(mf), exist_ok=True)
    man = json.load(open(mf)) if os.path.exists(mf) else dict(stage='ridge', subjects={}); t0 = time.time()
    for s in a.subjects or tc['subjects']:
        t1 = time.time(); times = np.load(os.path.join(cache, f'sub-{s:02d}', 'ses-01_mvnn.npz'))['times']
        win = {'pre': np.where(times < 0)[0], 'post': np.where(times >= tc['ridge']['input_tmin'] - 1e-9)[0]}
        E = PP.test_epochs(os.path.join(root, f'sub-{s:02d}', f"ses-{vc['eval_session']:02d}", vc['test_name']), pc,
                           os.path.join(cache, f'sub-{s:02d}', f"ses-{vc['eval_session']:02d}_mvnn.npz"), P(vc['seal_table']), unseal=True)
        assert np.allclose(E['times'], times), 'test and training epoch grids differ'
        full20 = np.stack([E['X'][E['code'] == k].mean(0) for k in range(1, 201)]).astype(np.float32); nrep = np.bincount(E['code'], minlength=201)[1:]; del E
        out, alph = {}, {}
        for w, kinds in KINDS.items():
            tsel = win[w]
            dc, DX, dn = image_means(cache, s, [1, 2, 3, 4], dev_codes, tsel); assert np.array_equal(dc, dev_codes), 'development photos missing'
            fc, FX, _ = image_means(cache, s, [1, 2, 3], fit_codes, tsel)
            for kind in kinds:
                pred, al, _ = ridge_cv(features(kind, FX, ch, fs), Timg[fc - 1], grp(fc, fold_dev)); out[f'dev_{kind}'] = pred(features(kind, DX, ch, fs)).astype(np.float32); alph[f'dev_{kind}'] = al; del pred
            l2d = {}
            for kind in [k for k in kinds if k in dm['kinds']]:
                Xf = features(kind, FX, ch, fs).astype(np.float64); xm, xs = Xf.mean(0), Xf.std(0) + 1e-8; Xf = (Xf - xm) / xs; Xd = (features(kind, DX, ch, fs) - xm) / xs
                cf = [meta.train_concepts[meta.img_concept[k - 1]] for k in fc]; ff = dict(zip(sorted(set(cf)), salt_folds(sorted(set(cf)), dm['salt'], dm['folds']))); g = np.array([ff[x] for x in cf])
                for hd in ('col', 'cat'):
                    keep = np.ones(len(fc), bool) if hd == 'col' else st.has_cat[st.img_name[fc - 1]]
                    Y = st.frac[fc - 1] if hd == 'col' else st.Ycat_name[st.img_name[fc - 1]]
                    cv = [float(np.mean([ce(Xf[keep & (g == f)] @ W + b, Y[keep & (g == f)]) for f in range(dm['folds'])
                                         for W, b, _ in [fit_softmax(Xf[keep & (g != f)], Y[keep & (g != f)], l2, dm['maxiter'])]])) for l2 in dm['l2_grid']]
                    l2 = dm['l2_grid'][int(np.argmin(cv))]; W, b, _ = fit_softmax(Xf[keep], Y[keep], l2, dm['maxiter']); out[f'dev_direct_{kind}_{hd}'] = (Xd @ W + b).astype(np.float32)
                    l2d[f'{kind}_{hd}'] = l2; alph[f'direct_{kind}_{hd}_cv'] = cv; alph[f'direct_{kind}_{hd}_l2'] = l2
                del Xf, Xd
            del FX
            fc, FX, _ = image_means(cache, s, [1, 2, 3], all_codes, tsel)
            for kind in kinds:
                pred, al, _ = ridge_cv(features(kind, FX, ch, fs), Timg[fc - 1], grp(fc, fold_all)); out[f'test_{kind}'] = pred(features(kind, full20[..., tsel], ch, fs)).astype(np.float32); alph[f'test_{kind}'] = al; del pred
            for kind in [k for k in kinds if k in dm['kinds']]:
                Xf = features(kind, FX, ch, fs).astype(np.float64); xm, xs = Xf.mean(0), Xf.std(0) + 1e-8; Xf = (Xf - xm) / xs; Xt = (features(kind, full20[..., tsel], ch, fs) - xm) / xs
                for hd in ('col', 'cat'):
                    keep = np.ones(len(fc), bool) if hd == 'col' else st.has_cat[st.img_name[fc - 1]]; Y = st.frac[fc - 1] if hd == 'col' else st.Ycat_name[st.img_name[fc - 1]]
                    W, b, _ = fit_softmax(Xf[keep], Y[keep], l2d[f'{kind}_{hd}'], dm['maxiter']); out[f'test_direct_{kind}_{hd}'] = (Xt @ W + b).astype(np.float32)
                del Xf, Xt
            del FX, DX
            print(f'sub-{s:02d} {w}: done ({time.time() - t1:.0f}s, peak {peak_gb():.1f} GB)', flush=True)
        f = os.path.join(od, f'ridge_t{s:02d}.npz'); np.savez(f, dev_rows=dev_codes - 1, **out)
        man['subjects'][str(s)] = dict(file=os.path.relpath(f, ROOT), sha256=fsha(f), alpha=alph, test_reps=dict(min=int(nrep.min()), max=int(nrep.max())), seconds=round(time.time() - t1, 1), peak_rss_gb=peak_gb())
        json.dump(man, open(mf, 'w'), indent=1)
    print('ridge done', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
