"""ds004408: pooled cross-subject training for the 3-s chunk identification (D-061).

Same chunks, features, targets and candidate sets as scripts/ds004408_identify.py (D-059 P1). Folds = the 5 run folds;
in each fold one ridge model is trained on the training runs of the pool subjects (P0-a passers) and applied to the
held-out runs of every subject, so no audio segment is ever in training and test at once. Penalty by 4-fold grouped CV
over training runs (mean cosine). Chance: 100 shuffles of the training pairs (penalty fixed), per-subject 95th percentile.
Rows: EEG->ENV (sensitivity gate G1), EEG->TXT (primary), PRE->TXT, ORDER->TXT (meters), AUDF->TXT (stimulus only),
FUSED->TXT = L2-normalised AUDF prediction + L2-normalised EEG prediction (semantic increment, fixed 1:1).

usage: python scripts/ds004408_identify_pool.py OUT_JSON [--perms 100] [--exclude 1 15]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ds004408_identify import CACHE, FS, WIN, DELAY, PRE, FOLDS, ALPHAS, chunks, embed, svd, ridge_fit, cos_rows, local_candidates, boot

SUBJECTS = list(range(1, 20))


def unit(P): return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)


def hits_from(P, Yn, C, pos):
    sim = np.einsum('md,mkd->mk', unit(P)[pos[C[:, 0]]], Yn[C]); return (sim.argmax(1) == 0).astype(float)


def inner_alpha(Xtr, Ys, grp):
    """Penalty with the best mean held-out cosine over 4 run groups, for each target in Ys (list); one SVD in memory at a time."""
    scores = [np.zeros(len(ALPHAS)) for _ in Ys]
    for g in range(4):
        tr, va = grp != g, grp == g; xm = Xtr[tr].mean(0); U, s, Vt = svd(Xtr[tr] - xm); Xva = Xtr[va] - xm
        for k, Y in enumerate(Ys):
            ym = Y[tr].mean(0)
            for j, a in enumerate(ALPHAS): scores[k][j] += cos_rows(Xva @ ridge_fit(U, s, Vt, Y[tr] - ym, a), Y[va] - ym).mean()
        del U, s, Vt
    return [float(ALPHAS[int(np.argmax(sc))]) for sc in scores]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out_json'); ap.add_argument('--perms', type=int, default=100)
    ap.add_argument('--exclude', type=int, nargs='*', default=[1, 15]); a = ap.parse_args(); t0 = time.time()
    pool = [s for s in SUBJECTS if s not in a.exclude]
    St = np.load(os.path.join(CACHE, 'stim.npz'), allow_pickle=True); CH = chunks(St)
    run = np.concatenate([[r] * len(c) for r, c in enumerate(CH)]); tstart = np.concatenate([[x[0] for x in c] for c in CH]); texts = [x[1] for c in CH for x in c]
    TXT = embed(texts); n0, nw, npre = int(round(DELAY * FS)), int(WIN * FS), int(PRE * FS); i0 = np.round(tstart * FS).astype(int)
    ENV = np.stack([np.asarray(St['env20'][r], np.float64)[i:i + nw] for r, i in zip(run, i0)]); ENVz = (ENV - ENV.mean(0)) / (ENV.std(0) + 1e-8)
    BAND = np.stack([np.asarray(St['bands20'][r], np.float64)[:, i:i + nw].mean(1) for r, i in zip(run, i0)])
    AUDF = np.c_[ENV, BAND]; AUDF = (AUDF - AUDF.mean(0)) / (AUDF.std(0) + 1e-8); ORDER = np.c_[tstart, tstart ** 2]; ORDER = (ORDER - ORDER.mean(0)) / ORDER.std(0)
    EEG, PREF = {}, {}
    for s in SUBJECTS:
        E = np.load(os.path.join(CACHE, f'sub-{s:03d}.npz'), allow_pickle=True)
        eeg = [np.asarray(x, np.float64) for x in E['eeg20']]; eeg = [(x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-12) for x in eeg]
        EEG[s] = np.stack([eeg[r][:, i + n0:i + n0 + nw].ravel() for r, i in zip(run, i0)]).astype(np.float32)
        PREF[s] = np.stack([eeg[r][:, i - npre:i].ravel() for r, i in zip(run, i0)]).astype(np.float32)
    print(f'{len(run)} chunks, {len(pool)} pool subjects, features ready ({time.time() - t0:.0f}s)', flush=True)
    rows = ['EEG->ENV', 'EEG->TXT', 'PRE->TXT', 'ORDER->TXT', 'AUDF->TXT', 'FUSED->TXT']
    H = {r: {s: [] for s in SUBJECTS} for r in rows}; NULL = {r: {s: np.zeros(a.perms) for s in SUBJECTS} for r in ['EEG->ENV', 'EEG->TXT', 'PRE->TXT', 'ORDER->TXT']}
    NN = {s: 0 for s in SUBJECTS}; alphas = {}; rng = np.random.default_rng(0)
    for fi, f in enumerate(FOLDS):
        trc, tec = np.where(~np.isin(run, f))[0], np.where(np.isin(run, f))[0]; tr_runs = [r for r in range(20) if r not in f]
        C = local_candidates(run, tstart, tec); pos = np.full(len(run), -1); pos[tec] = np.arange(len(tec))
        grp_c = np.array([tr_runs.index(r) % 4 for r in run[trc]])
        Yn = {}; Ym = {}
        for tname, Y in (('TXT', TXT), ('ENV', ENVz)):
            Ym[tname] = Y[trc].mean(0); Yn[tname] = unit(Y - Ym[tname])
        preds = {}
        # stimulus-only models (same for every subject): AUDF->TXT, ORDER->TXT
        for name, X in (('AUDF->TXT', AUDF), ('ORDER->TXT', ORDER)):
            Xtr = X[trc]; xm = Xtr.mean(0)
            al, = inner_alpha(Xtr, [TXT[trc]], grp_c); U, s_, Vt = svd(Xtr - xm); Ytr = TXT[trc] - Ym['TXT']
            preds[name] = (X[tec] - xm) @ ridge_fit(U, s_, Vt, Ytr, al); alphas[f'{name}|{fi}'] = al
            if name == 'ORDER->TXT':
                for q in range(a.perms):
                    h = hits_from((X[tec] - xm) @ ridge_fit(U, s_, Vt, Ytr[rng.permutation(len(trc))], al), Yn['TXT'], C, pos)
                    for s in SUBJECTS: NULL[name][s][q] += h.sum()
        # EEG-based pooled models
        for fname, F in (('EEG', EEG), ('PRE', PREF)):
            Xtr = np.concatenate([F[s][trc] for s in pool]).astype(np.float64); grp = np.tile(grp_c, len(pool)); xm = Xtr.mean(0); Xtr -= xm
            tnames = ('ENV', 'TXT') if fname == 'EEG' else ('TXT',)
            Yfull = {t: np.tile((TXT if t == 'TXT' else ENVz)[trc], (len(pool), 1)) for t in tnames}
            al_list = inner_alpha(Xtr, [Yfull[t] for t in tnames], grp)   # Xtr is centred; inner folds re-centre on their own rows
            U, s_, Vt = svd(Xtr); del Xtr
            print(f'fold {fi} {fname}: SVDs done ({time.time() - t0:.0f}s)', flush=True)
            Xte = {s: F[s][tec].astype(np.float64) - xm for s in SUBJECTS}
            for tname, al in zip(tnames, al_list):
                name = f'{fname}->{tname}'; Ytr = Yfull[tname] - Ym[tname]; alphas[f'{name}|{fi}'] = al; W = ridge_fit(U, s_, Vt, Ytr, al)
                for s in SUBJECTS:
                    P = Xte[s] @ W; H[name][s].append(hits_from(P, Yn[tname], C, pos))
                    if name == 'EEG->TXT': H['FUSED->TXT'][s].append(hits_from(unit(P) + unit(preds['AUDF->TXT']), Yn['TXT'], C, pos))
                for q in range(a.perms):   # one permuted model at a time
                    Wp = ridge_fit(U, s_, Vt, Ytr[rng.permutation(len(Ytr))], al)
                    for s in SUBJECTS: NULL[name][s][q] += hits_from(Xte[s] @ Wp, Yn[tname], C, pos).sum()
            del U, s_, Vt, Yfull, Xte
        for s in SUBJECTS:
            NN[s] += len(C); H['AUDF->TXT'][s].append(hits_from(preds['AUDF->TXT'], Yn['TXT'], C, pos)); H['ORDER->TXT'][s].append(hits_from(preds['ORDER->TXT'], Yn['TXT'], C, pos))
        print(f'fold {fi} done ({time.time() - t0:.0f}s)', flush=True)
    res = dict(pool=pool, n_chunks=int(len(run)), alphas=alphas, subjects={}, summary={})
    for s in SUBJECTS:
        r = {}
        for name in rows:
            h = np.concatenate(H[name][s]); r[name] = dict(top1=float(h.mean()), n=int(len(h)),
                                                           null_p95=float(np.quantile(NULL[name][s] / NN[s], 0.95)) if name in NULL else None)
        r['increment'] = float((np.concatenate(H['FUSED->TXT'][s]) - np.concatenate(H['AUDF->TXT'][s])).mean()); res['subjects'][s] = r
    for label, group in (('pool17', pool), ('all19', SUBJECTS)):
        sm = {}
        for name in rows:
            v = np.array([res['subjects'][s][name]['top1'] for s in group]); thr = [res['subjects'][s][name]['null_p95'] for s in group if res['subjects'][s][name]['null_p95'] is not None]
            sm[name] = dict(mean=float(v.mean()), ci95=boot(v), chance_threshold=float(np.mean(thr)) if thr else None, per_subject=v.round(4).tolist())
        inc = np.array([res['subjects'][s]['increment'] for s in group]); sm['increment'] = dict(mean=float(inc.mean()), ci95=boot(inc))
        res['summary'][label] = sm
    sm = res['summary']['pool17']
    if sm['EEG->ENV']['ci95'][0] <= sm['EEG->ENV']['chance_threshold']: verdict = 'G1 failed: readout sensitivity insufficient, semantic verdict withheld'
    elif sm['EEG->TXT']['ci95'][0] <= sm['EEG->TXT']['chance_threshold']: verdict = 'G1 passed; heard chunk NOT identified from EEG by text embedding'
    elif any(sm[m]['mean'] > sm[m]['chance_threshold'] for m in ('PRE->TXT', 'ORDER->TXT')): verdict = 'withheld (leak meter above chance)'
    else: verdict = 'heard chunk identified; semantic increment ' + ('passed' if sm['increment']['ci95'][0] > 0 else 'not passed')
    res['verdict'] = verdict; json.dump(res, open(a.out_json, 'w'), indent=1)
    for label in res['summary']:
        for k, v in res['summary'][label].items(): print(label, k, v, flush=True)
    print('VERDICT:', verdict, flush=True)


if __name__ == '__main__':
    main()
