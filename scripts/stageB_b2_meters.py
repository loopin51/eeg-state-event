"""Stage B B2 (D-091) linear references (CPU): DE + ridge baseline and the ORDER, ACOUSTIC and RUN meters, ranked inside the
acoustic pair sets of scripts/stageB_b2_prepare.py.

Target: the fold's residual language vector (items.npz R_f{f}: language vector minus its linear acoustic prediction, D-091
amendment 3), re-standardised with the training grid windows (with words) of each run fold.
  DE:       5-band log power x 128 channels (decoder setting of configs/stageB_b1.yaml; CAR also leaves out the BIDS `bad`
            channels when car_exclude_bad, D-091 amendment 4), ridge per subject on its own 14 training runs (penalty by
            grouped CV, 7 groups of 2 runs). Every subject (19).
  ORDER:    polynomial and Fourier terms of the relative position inside the run (fitted on training runs).
  ACOUSTIC: the primary pairing features (mel 120, envelope 30, log2 F0 30, voiced fraction 30; F0 after the 1-kHz
            low-pass of D-091 amendment 2), fitted on training runs.
  ACOUSTIC_NL (report only, amendments 3 and 5): random Fourier features of the same standardised features, then ridge;
            one map per seed in meters.acoustic_nl.seeds, cells averaged over seeds (per-seed cells in ACOUSTIC_NL_seeds).
  ACOUSTIC_orig (report only, amendment 3): ACOUSTIC on the original (not residualised) language target.
  RUN:      no fit; for each query the mean target of all queries of its run (constant per run; D-091 amendment 5 replaced
            the amendment-1 leave-30-s-out mean, which made the observed and null candidate sets asymmetric). Run identity
            path, report only: within-run pools give exactly the null by construction, the cross pool shows the path.
Per pool (within, cross, mel_within, within_f0raw) and run: mean of 1 - (number of pairs scoring above the true segment) / 19
over the queries with a valid pair set (cosine scores), the query count, and the mean of the meter's own circular-shift null
(amendment 3: prediction rows fixed at query i, answer and candidates replaced by the shifted partner j and the pairs of j;
same shift maps as scripts/stageB_b2_verdict.py).

usage: python scripts/stageB_b2_meters.py configs/stageB_b2.yaml [--smoke]
       --smoke: run fold 0 only, validation-run pools, DE for subject fold 0 [:2]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np
from scipy.signal import welch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageB_b1_meters import BANDS, ALPHAS, fit_predict, unit
from stageB_b1_verdict import shift_maps

POOLS = ('within', 'cross', 'mel_within', 'within_f0raw')


def fit_predict_svd(Xtr, Ytr, Xte, groups):
    """same estimator and penalty selection as stageB_b1_meters.fit_predict, one SVD per fold (for wide feature maps)"""
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8; ym = Ytr.mean(0); Xtr = (Xtr - xm) / xs; Xte = (Xte - xm) / xs; sc = np.zeros(len(ALPHAS))
    def coefs(X, Y):
        U, s, Vt = np.linalg.svd(X, full_matrices=False); UY = U.T @ Y; return lambda a: Vt.T @ ((s / (s ** 2 + a))[:, None] * UY)
    for g in np.unique(groups):
        tr, va = groups != g, groups == g; B = coefs(Xtr[tr], Ytr[tr] - ym)
        for j, a in enumerate(ALPHAS): sc[j] += np.mean(np.sum(unit(Xtr[va] @ B(a)) * unit(Ytr[va] - ym), 1))
    a = ALPHAS[int(np.argmax(sc))]; return Xte @ coefs(Xtr, Ytr - ym)(a), float(a)


def run_maps(c2, runs, onsets):
    """the verdict's shift maps: one rng stream over the sorted evaluation runs"""
    rng = np.random.default_rng(c2['shift_null']['seed']); out = {}
    for r in sorted(runs): out.update(shift_maps({r: {'onset': onsets[r]}}, c2['shift_null']['n_shift'], c2['shift_null']['min_shift_s'], rng))
    return out


def pair_cells(S, pool, pairs, run, maps):
    """S [n_pool, n_pool] scores (rows: queries, cols: candidates, both in pool order); pairs: global indices (-1 rows)"""
    loc = {g: i for i, g in enumerate(pool)}; out = {}; n_p = pairs.shape[1]
    v = pairs[:, 0] >= 0; P = np.array([[loc[g] for g in row] if ok else [0] * n_p for row, ok in zip(pairs, v)])
    rk = (S[np.arange(len(pool))[:, None], P] > np.diag(S)[:, None]).sum(1); mrp = 1 - rk / n_p
    for r in np.unique(run[pool]):
        m = np.where(run[pool] == r)[0]; mv = m[v[m]]
        J = m[maps[int(r)]]; okJ = v[J]                                                              # [n_shift, n_r]
        rkn = (S[m[None, :, None], P[J]] > S[m[None, :], J][..., None]).sum(-1); xn = np.where(okJ, 1 - rkn / n_p, np.nan)
        out[str(int(r) + 1)] = dict(mrp=float(mrp[mv].mean()), n=int(len(mv)), top1=float((rk[mv] < 1).mean()), null_mean=float(np.nanmean(xn)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args(); t0 = time.time()
    c2 = yaml.safe_load(open(a.config)); c = yaml.safe_load(open(os.path.join(ROOT, c2['b1_config']))); sp = c['split']
    d = os.path.join(ROOT, c['cache_dir']); st = c['settings'][c['decoder_setting']]
    meta = json.load(open(os.path.join(d, 'eeg_meta.json'))); fs = meta['sample_rate']; offs = meta['run_offsets']; lens = meta['run_lengths']
    w, e = c['window'], c['eeg']; T = int(round((w['tmax'] - w['tmin']) * fs))
    Z = np.load(os.path.join(ROOT, c2['cache'], 'items.npz'), allow_pickle=True); PR = np.load(os.path.join(ROOT, c2['cache'], 'pairs.npz'))
    run, t, isq, ht, R0 = Z['run'], Z['t'], Z['is_query'], Z['has_text'], Z['R']; split = 'valid' if a.smoke else 'test'
    rel = t / (np.array(lens)[run] / fs)
    POS = np.c_[rel, rel ** 2, rel ** 3, *[g(2 * np.pi * k * rel) for k in (1, 2, 4, 8) for g in (np.sin, np.cos)]]
    AC = np.c_[Z['mel'], Z['env'], Z['f0'], Z['vf']]; nl = c2['meters']['acoustic_nl']
    folds = [0] if a.smoke else range(5); fd = {}
    ev_runs = sorted(set(int(r) for f in folds for r in run[PR[f'pool_{split}_f{f}']]))
    maps = run_maps(c2, ev_runs, {r: t[isq & (run == r)] for r in ev_runs})
    for f in folds:
        test = [r - 1 for r in sp['run_folds'][f]]; valid = [r - 1 for r in sp['valid_runs'][f]]; train = [r for r in range(20) if r not in test + valid]
        tr_i = np.where(~isq & ht & np.isin(run, train))[0]; pool = PR[f'pool_{split}_f{f}']; R = Z[f'R_f{f}']
        assert (np.diff(t[pool])[np.diff(run[pool]) == 0] > 0).all(), 'pool not in onset order within runs'
        Y = (R - R[tr_i].mean(0)) / (R[tr_i].std(0) + 1e-6); Y0 = (R0 - R0[tr_i].mean(0)) / (R0[tr_i].std(0) + 1e-6)
        grp = np.array([train.index(r) // 2 for r in run[tr_i]]); fd[f] = (train, tr_i, pool, Y, Y0, grp)
    names = ('ORDER', 'ACOUSTIC', 'ACOUSTIC_NL', 'ACOUSTIC_orig', 'RUN')
    res = {'eval_split': split, **{k: {p: {} for p in POOLS} for k in names}, 'DE': {}, 'alpha': {},
           'ACOUSTIC_NL_seeds': {str(sd): {p: {} for p in POOLS} for sd in nl['seeds']}}
    for f, (train, tr_i, pool, Y, Y0, grp) in fd.items():
        cells = lambda S: {p: pair_cells(S, pool, PR[f'pairs_{p}_{split}_f{f}'], run, maps) for p in POOLS}
        for name, X, YY, fit in (('ORDER', POS, Y, fit_predict), ('ACOUSTIC', AC, Y, fit_predict), ('ACOUSTIC_orig', AC, Y0, fit_predict)):
            P, al = fit(X[tr_i], YY[tr_i], X[pool], grp); res['alpha'][f'{name}_f{f}'] = al
            for p, v in cells(unit(P) @ unit(YY[pool]).T).items(): res[name][p].update(v)
            print(f'{name} fold {f} ({time.time() - t0:.0f}s)', flush=True)
        xm, xs = AC[tr_i].mean(0), AC[tr_i].std(0) + 1e-8
        for sd in nl['seeds']:
            g = np.random.default_rng(sd); Wf = g.normal(0, 1 / np.sqrt(AC.shape[1]), (AC.shape[1], nl['n_features'])); bf = g.uniform(0, 2 * np.pi, nl['n_features'])
            RFF = np.sqrt(2 / nl['n_features']) * np.cos(((AC - xm) / xs) @ Wf + bf)
            P, al = fit_predict_svd(RFF[tr_i], Y[tr_i], RFF[pool], grp); res['alpha'][f'ACOUSTIC_NL_{sd}_f{f}'] = al
            for p, v in cells(unit(P) @ unit(Y[pool]).T).items(): res['ACOUSTIC_NL_seeds'][str(sd)][p].update(v)
            print(f'ACOUSTIC_NL seed {sd} fold {f} ({time.time() - t0:.0f}s)', flush=True)
        same = (run[pool][:, None] == run[pool][None]).astype(float); P = (same @ Y[pool]) / same.sum(1, keepdims=True)
        for p, v in cells(unit(P) @ unit(Y[pool]).T).items(): res['RUN'][p].update(v)
    for p in POOLS:                                                                                  # seed-mean cells
        for r in res['ACOUSTIC_NL_seeds'][str(nl['seeds'][0])][p]:
            xs_ = [res['ACOUSTIC_NL_seeds'][str(sd)][p][r] for sd in nl['seeds']]
            res['ACOUSTIC_NL'][p][r] = dict(n=xs_[0]['n'], **{k: float(np.mean([x[k] for x in xs_])) for k in ('mrp', 'top1', 'null_mean')})
    def agg(x, key): return round(float(np.average([v[key] for v in x.values()], weights=[v['n'] for v in x.values()])), 4)
    print('meters (mrp / own null mean)', json.dumps({k: {p: (agg(res[k][p], 'mrp'), agg(res[k][p], 'null_mean')) for p in POOLS} for k in names}),
          f'({time.time() - t0:.0f}s)', flush=True)
    items = list(zip(run.tolist(), t.tolist())); subjects = sp['subject_folds'][0][:2] if a.smoke else sorted(int(s) for s in meta['subjects'])
    for s in subjects:
        eeg = np.load(os.path.join(d, f"{st['source']}_sub-{s:03d}.npy"), mmap_mode='r'); flat = np.array(meta['subjects'][str(s)]['flat'])
        if c2['car_exclude_bad']: flat = flat | np.array(meta['subjects'][str(s)]['bad'])
        X = np.empty((len(items), 128 * len(BANDS)), np.float32)
        for b0 in range(0, len(items), 1000):
            xs_ = []
            for (r, tt) in items[b0:b0 + 1000]:
                i = offs[r] + int(round((tt + w['tmin'] + e['offset_s']) * fs)); x = np.asarray(eeg[:, i:i + T], np.float32).copy()
                if st['reference'] == 'car': x -= x[~flat[r]].mean(0, keepdims=True)
                x -= x[:, : int(round(e['baseline_s'] * fs))].mean(1, keepdims=True); xs_.append(np.clip(x, -e['clamp'], e['clamp']))
            fr, pw = welch(np.stack(xs_), fs=fs, nperseg=min(fs, T), axis=-1)
            X[b0:b0 + len(xs_)] = np.log(np.stack([pw[..., (fr >= l) & (fr < h)].mean(-1) for l, h in BANDS], -1) + 1e-8).reshape(len(xs_), -1)
        out = res['DE'][str(s)] = {p: {} for p in POOLS}
        for f, (train, tr_i, pool, Y, Y0, grp) in fd.items():
            P, al = fit_predict(X[tr_i], Y[tr_i], X[pool], grp); S = unit(P) @ unit(Y[pool]).T; res['alpha'][f'DE_{s}_f{f}'] = al
            for p in POOLS: out[p].update(pair_cells(S, pool, PR[f'pairs_{p}_{split}_f{f}'], run, maps))
        print(f"DE sub-{s:03d}: within-pool MRP {agg(out['within'], 'mrp'):.4f} null mean {agg(out['within'], 'null_mean'):.4f} ({time.time() - t0:.0f}s)", flush=True)
    p = os.path.join(ROOT, 'results/baselines', 'stageB_b2_meters' + ('_smoke' if a.smoke else '') + '.json'); os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), indent=1); print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
