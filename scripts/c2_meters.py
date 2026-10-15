"""C2 stimulus-only meter UNATTENDED (D-097 amendment 2 condition 3, a C2 prerequisite; CPU, no EEG).

For every query window (query_step_s grid of the test trials) the window mean of the unattended-stream w2v features predicts the
attended-stream window mean by ridge; the attended-stream query windows of the same trial are ranked (within-trial MRP, cells
keyed by (subject, trial)), as the B1 ORDER meter (stageB_b1_meters.fit_predict and cells). The ridge of trial fold f is fitted
on the calibration-candidate trials of fold f of all 18 subjects (never a test or validation pair of fold f; each subject's
trial gives the direction it was heard in); penalty by grouped CV over pair groups (pair rank // 5 within the fold's training
pairs); target standardisation from the same training windows. The 90% interval (C2 cell resampling) and the condition
(upper < 0.52) are applied by the C2 verdict script.

usage: python scripts/c2_meters.py configs/c2_p1.yaml [--smoke]
       --smoke: trial fold 0, ranking on the validation trials (never the test trials)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageB_b1_meters import ALPHAS, unit, cells


def fit_predict(Xtr, Ytr, Xte, groups):
    """stageB_b1_meters.fit_predict with one SVD per CV split instead of one per (split, penalty); same estimator and choice"""
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8; ym = Ytr.mean(0); Xtr = (Xtr - xm) / xs; Xte = (Xte - xm) / xs; sc = np.zeros(len(ALPHAS)); Yc = Ytr - ym

    def coefs(X, Y):
        U, sv, Vt = np.linalg.svd(X, full_matrices=False); UY = U.T @ Y
        return lambda a: Vt.T @ ((sv / (sv ** 2 + a))[:, None] * UY)
    for g in np.unique(groups):
        tr, va = groups != g, groups == g; B = coefs(Xtr[tr], Yc[tr]); Yv = unit(Yc[va])
        for j, a in enumerate(ALPHAS): sc[j] += np.mean(np.sum(unit(Xtr[va] @ B(a)) * Yv, 1))
    a = ALPHAS[int(np.argmax(sc))]; return Xte @ coefs(Xtr, Yc)(a), float(a)
from c2_dtu_split import DTUSplit, trials_from_cache, path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    cc = yaml.safe_load(open(path(a.config))); c = yaml.safe_load(open(path(cc['b1_config']))); fc = yaml.safe_load(open(path(cc['folds_config'])))
    t0 = time.time(); cache = path(fc['cache_dir']); meta = json.load(open(os.path.join(cache, 'eeg_meta.json'))); fs = meta['sample_rate']
    st = json.load(open(os.path.join(cache, 'stimuli.json'))); st = st.get('stimuli', st); F = np.load(os.path.join(cache, 'w2v_stim.npy'), mmap_mode='r')
    split = DTUSplit(fc, trials_from_cache(fc['cache_dir'])); w, e = c['window'], c['eeg']; T = int(round((w['tmax'] - w['tmin']) * fs)); lo = -w['tmin']
    step = fc['query_step_s']; assert step == w['train_step_s'], 'one grid for training items and queries'

    def grid(r):
        hi = r['length'] / fs - w['tmax'] - max(e['offset_s'], 0); return np.arange(lo, hi + 1e-9, step)

    def pooled(stim, ts, s0=0.0):
        o = st[stim]['offset']; return np.stack([np.asarray(F[:, o + int(round((s0 + t + w['tmin']) * fs)): o + int(round((s0 + t + w['tmin']) * fs)) + T], np.float32).mean(1) for t in ts])

    feats = {}                                                             # (subject, trial) -> (unattended X, attended Y) per grid window
    for s in split.subjects:
        for r in split.trials[s]:
            ts = grid(r); s0 = r.get('stim_start_s', 0.0); feats[(s, r['trial'])] = (pooled(r['unattended'], ts, s0), pooled(r['stim'], ts, s0))
    print(f'features of {len(feats)} trials ready ({time.time() - t0:.0f}s)', flush=True)
    res = {'meter': 'UNATTENDED', 'eval_split': 'valid' if a.smoke else 'test', 'folds': {}, 'cells': {}}
    for f in ([0] if a.smoke else range(5)):
        tr = [(s, r) for s in split.subjects for r in split.sets(s, f)['calib']]; ev = [(s, r) for s in split.subjects for r in split.sets(s, f)['valid' if a.smoke else 'test']]
        pairs = sorted({r['pair'] for _, r in tr}); gid = {p: i // 5 for i, p in enumerate(pairs)}
        Xtr = np.concatenate([feats[(s, r['trial'])][0] for s, r in tr]); Ytr = np.concatenate([feats[(s, r['trial'])][1] for s, r in tr])
        grp = np.concatenate([np.full(len(feats[(s, r['trial'])][0]), gid[r['pair']]) for s, r in tr])
        fm, fsd = Ytr.mean(0), Ytr.std(0) + 1e-6; Ytr = (Ytr - fm) / fsd
        Xte = np.concatenate([feats[(s, r['trial'])][0] for s, r in ev]); Yte = (np.concatenate([feats[(s, r['trial'])][1] for s, r in ev]) - fm) / fsd
        cell = np.concatenate([np.full(len(feats[(s, r['trial'])][0]), n) for n, (s, r) in enumerate(ev)])
        P, al = fit_predict(Xtr, Ytr, Xte, grp)
        for n, (s, r) in enumerate(ev):                                     # one cell at a time (a fold-wide score matrix is ~20k x 20k)
            m = cell == n; cl, _ = cells(P[m], Yte[m], np.zeros(int(m.sum()), int), np.zeros(int(m.sum())))
            res['cells'].setdefault(str(s), {})[str(r['trial'])] = dict(cl['1'], fold=f, pair=r['pair'])
        res['folds'][str(f)] = dict(alpha=al, train_windows=int(len(Xtr)), train_pairs=len(pairs), eval_cells=len(ev))
        print(f'trial fold {f}: alpha {al:.3g}, {len(Xtr)} training windows over {len(pairs)} pairs, {len(ev)} cells ({time.time() - t0:.0f}s)', flush=True)
    allc = [v for d in res['cells'].values() for v in d.values()]
    res['pooled_mrp_report'] = float(np.average([v['mrp'] for v in allc], weights=[v['n'] for v in allc])); res['seconds'] = time.time() - t0
    print(f'UNATTENDED pooled within-trial MRP {res["pooled_mrp_report"]:.4f} over {len(allc)} cells', flush=True)
    p = path(cc['meters']['out'].replace('.json', '_smoke.json') if a.smoke else cc['meters']['out']); os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), indent=1); print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
