"""Stage 0 (D-065, faculty direction 1): oracle upper-bound curves for the ds004408 3-s chunk identification.

1. Re-fits the pooled ridge models of D-061 (EEG->TXT, EEG->ENV; same chunks, features, folds, penalty search, pool of
   P0-a passers) without permutations and saves the held-out predictions of every subject.
2. Oracle curves: the held-out target (centred on the training mean) plus noise, swept over noise scale. Two noise types:
   isotropic, and Gaussian noise with the covariance of the actual pooled ridge residuals (prediction minus its
   least-squares projection on the target). x axis = achieved mean cosine between prediction and target.
3. Identification under four candidate designs: (i) 19 nearest chunks in the same run, (ii) 19 chunks drawn from one
   quartile of target-space cosine distance (other runs of the test fold), (iii) 19 chunks with the nearest acoustic
   features (other runs of the test fold), (iv) rank among all chunks of the test fold (top-k, mean rank percentile).
4. The actual predictions are scored with the same designs and placed on the curves (observed vs expected at the
   observed cosine). Candidate-target residual cosine distances are summarised per design.

usage: python scripts/ds004408_oracle.py configs/ds004408_oracle.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, yaml, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ds004408_identify import CACHE, FS, WIN, DELAY, FOLDS, chunks, embed, svd, ridge_fit, cos_rows, local_candidates
from ds004408_identify_pool import SUBJECTS, unit, inner_alpha
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COPPER, TEAL, PURPLE, INK, PAPER = '#B5652D', '#2E6E6B', '#5B4A86', '#182129', '#EEF0EC'


def build_designs(run, tec, Yn_te, AUDF, cfg, rng):
    """Candidate index matrices (column 0 = target) over test-fold positions; the local design is added by the caller."""
    d = cfg['designs']; m = len(tec); r_te = run[tec]; out = {}
    S = Yn_te @ Yn_te.T; A = AUDF[tec]; Ad = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    k = d['quartile_n'] - 1
    for q in range(4): out[f'quartile{q + 1}'] = []
    out['acoustic'] = []
    for i in range(m):
        other = np.where(r_te != r_te[i])[0]
        dist = 1 - S[i, other]; edges = np.quantile(dist, [0, .25, .5, .75, 1])
        for q in range(4):
            sel = other[(dist >= edges[q]) & (dist <= edges[q + 1])]
            out[f'quartile{q + 1}'].append(np.r_[i, rng.choice(sel, k, replace=len(sel) < k)])
        out['acoustic'].append(np.r_[i, other[np.argsort(Ad[i, other])[:d['acoustic_n'] - 1]]])
    return {key: np.array(v) for key, v in out.items()}


def score(P, Yn_te, designs, topk):
    """P: predictions for test positions (m x d). Returns per-design mean top-1 and pool metrics."""
    Pn = unit(P); res = {}
    for name, C in designs.items():
        sim = np.einsum('md,mkd->mk', Pn[C[:, 0]], Yn_te[C]); res[name] = float((sim.argmax(1) == 0).mean())
    S = Pn @ Yn_te.T; rank = (S > S[np.arange(len(S)), np.arange(len(S))][:, None]).sum(1)
    for k in topk: res[f'pool_top{k}'] = float((rank < k).mean())
    res['pool_rank_pct'] = float((1 - rank / (len(S) - 1)).mean())
    return res


def main(cfg_path):
    c = yaml.safe_load(open(cfg_path)); out = os.path.join(ROOT, c['out_dir']); os.makedirs(out, exist_ok=True); t0 = time.time()
    rng = np.random.default_rng(c['seed']); pool = [s for s in SUBJECTS if s not in c['exclude_from_pool']]
    St = np.load(os.path.join(CACHE, 'stim.npz'), allow_pickle=True); CH = chunks(St)
    run = np.concatenate([[r] * len(ch) for r, ch in enumerate(CH)]); tstart = np.concatenate([[x[0] for x in ch] for ch in CH]); texts = [x[1] for ch in CH for x in ch]
    TXT = embed(texts); n0, nw = int(round(DELAY * FS)), int(WIN * FS); i0 = np.round(tstart * FS).astype(int)
    ENV = np.stack([np.asarray(St['env20'][r], np.float64)[i:i + nw] for r, i in zip(run, i0)]); ENVz = (ENV - ENV.mean(0)) / (ENV.std(0) + 1e-8)
    BAND = np.stack([np.asarray(St['bands20'][r], np.float64)[:, i:i + nw].mean(1) for r, i in zip(run, i0)])
    AUDF = np.c_[ENV, BAND]; AUDF = (AUDF - AUDF.mean(0)) / (AUDF.std(0) + 1e-8)
    TGT = {'TXT': TXT, 'ENV': ENVz}; tnames = c['targets']
    EEG = {}
    for s in SUBJECTS:
        E = np.load(os.path.join(CACHE, f'sub-{s:03d}.npz'), allow_pickle=True)
        eeg = [np.asarray(x, np.float64) for x in E['eeg20']]; eeg = [(x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-12) for x in eeg]
        EEG[s] = np.stack([eeg[r][:, i + n0:i + n0 + nw].ravel() for r, i in zip(run, i0)]).astype(np.float32)
    print(f'{len(run)} chunks, pool {len(pool)}, features ready ({time.time() - t0:.0f}s)', flush=True)

    # 1. pooled ridge predictions (held-out runs, every subject)
    PRED = {t: np.zeros((len(SUBJECTS), len(run), TGT[t].shape[1]), np.float32) for t in tnames}; alphas = {}
    for fi, f in enumerate(FOLDS):
        trc, tec = np.where(~np.isin(run, f))[0], np.where(np.isin(run, f))[0]; tr_runs = [r for r in range(20) if r not in f]
        grp_c = np.array([tr_runs.index(r) % 4 for r in run[trc]])
        Xtr = np.concatenate([EEG[s][trc] for s in pool]).astype(np.float64); grp = np.tile(grp_c, len(pool)); xm = Xtr.mean(0); Xtr -= xm
        Yfull = {t: np.tile(TGT[t][trc], (len(pool), 1)) for t in tnames}
        al_list = inner_alpha(Xtr, [Yfull[t] for t in tnames], grp); U, s_, Vt = svd(Xtr); del Xtr
        for t, al in zip(tnames, al_list):
            W = ridge_fit(U, s_, Vt, Yfull[t] - TGT[t][trc].mean(0), al); alphas[f'EEG->{t}|{fi}'] = al
            for si, s in enumerate(SUBJECTS): PRED[t][si, tec] = (EEG[s][tec].astype(np.float64) - xm) @ W
        del U, s_, Vt, Yfull
        print(f'fold {fi} ridge done, alphas {[alphas[f"EEG->{t}|{fi}"] for t in tnames]} ({time.time() - t0:.0f}s)', flush=True)
    np.savez_compressed(os.path.join(out, 'pooled_predictions.npz'), run=run, tstart=tstart, subjects=np.array(SUBJECTS),
                        **{f'pred_{t}': PRED[t] for t in tnames}, **{f'target_{t}': TGT[t] for t in tnames})

    # 2-4. oracle curves and observed points
    grid = 10 ** np.linspace(**{k: v for k, v in c['noise']['scale_grid'].items()}); res = {'alphas': alphas, 'targets': {}}
    pidx = [SUBJECTS.index(s) for s in pool]
    for t in tnames:
        Y = TGT[t]; curves = {nt: [] for nt in c['noise']['types']}; observed = {'pool17': [], 'all19': []}; dists = {}
        per_fold = []
        for fi, f in enumerate(FOLDS):
            trc, tec = np.where(~np.isin(run, f))[0], np.where(np.isin(run, f))[0]
            Yc = Y[tec] - Y[trc].mean(0); Yn = unit(Yc)
            D = build_designs(run, tec, Yn, AUDF, c, rng)
            pos = np.full(len(run), -1); pos[tec] = np.arange(len(tec)); D['local'] = pos[local_candidates(run, tstart, tec)]
            for name, C in D.items():
                dists.setdefault(name, []).append((1 - np.einsum('md,mkd->mk', Yn[C[:, 0]], Yn[C[:, 1:]])).ravel())
            # residual covariance from the pooled-subject predictions of this fold
            P_all = PRED[t][pidx][:, tec].astype(np.float64)                      # S x m x d
            b = (P_all * Yc).sum((1, 2)) / (Yc ** 2).sum()                       # per-subject LS slope on the target
            R = (P_all - b[:, None, None] * Yc).reshape(-1, Yc.shape[1]); R -= R.mean(0)
            Sig = R.T @ R / len(R); Sig *= (Yc ** 2).mean() * Yc.shape[1] / np.trace(Sig)
            L = np.linalg.cholesky(Sig + 1e-8 * np.trace(Sig) / len(Sig) * np.eye(len(Sig)))
            rms = np.sqrt((Yc ** 2).mean())
            fold_curves = {}
            for nt in c['noise']['types']:
                rows = []
                for g in grid:
                    acc = []
                    for _ in range(c['noise']['draws']):
                        Z = rng.standard_normal(Yc.shape)
                        N = g * rms * Z if nt == 'isotropic' else g * (Z @ L.T)
                        P = Yc + N; sc = score(P, Yn, D, c['designs']['pool_topk']); sc['cosine'] = float(cos_rows(P, Yc).mean()); acc.append(sc)
                    rows.append({k: float(np.mean([a[k] for a in acc])) for k in acc[0]})
                fold_curves[nt] = rows
            obs = []
            for si, s in enumerate(SUBJECTS):
                P = PRED[t][si, tec].astype(np.float64); sc = score(P, Yn, D, c['designs']['pool_topk']); sc['cosine'] = float(cos_rows(P, Yc).mean()); sc['subject'] = s
                sc['n'] = int(len(tec)); obs.append(sc)
            per_fold.append({'curves': fold_curves, 'observed': obs, 'n_test': int(len(tec))})
            print(f'{t} fold {fi} oracle done ({time.time() - t0:.0f}s)', flush=True)
        # average over folds weighted by test size
        w = np.array([pf['n_test'] for pf in per_fold], float); w /= w.sum()
        for nt in c['noise']['types']:
            for gi, g in enumerate(grid):
                curves[nt].append({'scale': float(g), **{k: float(sum(wf * pf['curves'][nt][gi][k] for wf, pf in zip(w, per_fold))) for k in per_fold[0]['curves'][nt][gi]}})
        subj = {}
        for si, s in enumerate(SUBJECTS):
            keys = [k for k in per_fold[0]['observed'][si] if k not in ('subject', 'n')]
            subj[s] = {k: float(sum(wf * pf['observed'][si][k] for wf, pf in zip(w, per_fold))) for k in keys}
        summ = {}
        for label, group in (('pool17', pool), ('all19', SUBJECTS)):
            m = {k: float(np.mean([subj[s][k] for s in group])) for k in subj[SUBJECTS[0]]}
            exp = {}
            for nt in c['noise']['types']:
                xs = np.array([r['cosine'] for r in curves[nt]]); order = np.argsort(xs)
                exp[nt] = {k: float(np.interp(m['cosine'], xs[order], np.array([r[k] for r in curves[nt]])[order])) for k in m if k != 'cosine'}
            summ[label] = {'observed_mean': m, 'expected_at_observed_cosine': exp}
        dsum = {name: {q: float(np.quantile(np.concatenate(v), q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)} for name, v in dists.items()}
        res['targets'][t] = {'curves': curves, 'subjects': subj, 'summary': summ, 'candidate_distance_quantiles': dsum}
        print(t, json.dumps(summ), flush=True)
    res['config'] = c; json.dump(res, open(os.path.join(out, 'metrics.json'), 'w'), indent=1)

    # figure: top-1 vs cosine per design, isotropic (solid) and residual (dashed), observed pool17 mean (dot)
    designs = ['local', 'quartile1', 'quartile4', 'acoustic', 'pool_top10']; cols = [COPPER, PURPLE, TEAL, INK, '#8A8F93']
    fig, axes = plt.subplots(1, len(tnames), figsize=(6 * len(tnames), 4.2), squeeze=False)
    for ax, t in zip(axes[0], tnames):
        R = res['targets'][t]
        for dname, col in zip(designs, cols):
            for nt, ls in zip(c['noise']['types'], ['-', '--']):
                ax.plot([r['cosine'] for r in R['curves'][nt]], [r[dname] for r in R['curves'][nt]], ls, color=col, lw=1.2, label=f'{dname} ({nt})' if t == tnames[0] else None)
            o = R['summary']['pool17']['observed_mean']; ax.plot(o['cosine'], o[dname], 'o', color=col, ms=5)
        ax.set_xscale('log'); ax.set_xlabel('mean cosine(prediction, target)'); ax.set_ylabel('identification'); ax.set_title(f'EEG->{t}: oracle curves, observed (dots)', fontsize=9)
    axes[0][0].legend(fontsize=6, ncol=2); fig.patch.set_facecolor(PAPER); fig.tight_layout(); fig.savefig(os.path.join(out, 'oracle_curves.png'), dpi=130)
    print(f'done ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
