"""Stage B B1 verdict (D-085) from the 25 cross-fitted runs, the 5 phase-randomised runs and the DE / ORDER baselines.

Primary statistic: query-weighted mean of the within-run mean rank percentile (MRP) over the 17 pool subjects and 20 runs.
Null: 1,000 circular label shifts; per draw every run r gets one shift (same for all subjects) of k segments in onset order,
k in [k_min, n - k_min] with k_min = the number of segments in min_shift seconds (bijective; D-085 amendment).
Interval: subject x run pigeonhole bootstrap (Owen 2007), 1,000 draws. Verdict rules as registered in D-085.

usage: python scripts/stageB_b1_verdict.py configs/stageB_b1.yaml [--smoke]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_run(c, k, f, phaserand, smoke):
    pat = f"*_stageB_b1_{c['decoder_setting']}_k{k}_f{f}{'_phaserand' if phaserand else ''}{'_smoke' if smoke else ''}"
    hits = sorted(glob.glob(os.path.join(ROOT, c['out_root'], pat)))
    hits = [h for h in hits if os.path.exists(os.path.join(h, 'scores.npz'))]
    assert len(hits) == 1, f'{pat}: {len(hits)} finished run folders'; return hits[0]


def load_cells(c, folds, phaserand, smoke):
    """per target subject: run -> dict(R = rank matrix, onset, rank_full, rank_cross, n_fold, pre_R, pre_rows)"""
    out, hist = {}, {}
    for k, f in folds:
        d = find_run(c, k, f, phaserand, smoke); cf = yaml.safe_load(open(os.path.join(d, 'config.yaml')))['run']
        Z = np.load(os.path.join(d, 'scores.npz')); run = Z['run']; on = Z['onset']; hist[os.path.basename(d)] = json.load(open(os.path.join(d, 'history.json')))
        for s in cf['targets']:
            for r in np.unique(run):
                m = np.where(run == r)[0]; S = Z[f'main_s{s}_r{r}']; R = rank_matrix(S)
                P = Z[f'pre_s{s}_r{r}']; rows = np.where(~np.isnan(np.diag(P)))[0]
                out.setdefault(s, {})[int(r)] = dict(S=S, R=R, onset=on[m], rank_full=Z[f'main_s{s}_rank_full'][m],
                                                     rank_cross=Z[f'main_s{s}_rank_cross'][m], n_fold=len(run),
                                                     pre_R=rank_matrix(P[np.ix_(rows, rows)]), pre_rows=rows)   # PRE: queries and candidates = segments with a valid earlier window
    return out, hist


def rank_matrix(S):
    """R[i, j] = number of candidates scoring strictly above candidate j in row i"""
    Ssort = np.sort(S, axis=1); return np.stack([len(row) - np.searchsorted(srt, row, side='right') for row, srt in zip(S, Ssort)])


def mrp_true(R, rows=None):
    n = R.shape[1]; i = np.arange(len(R)); j = i if rows is None else rows
    return 1 - R[i, j] / (n - 1)


def shift_maps(cells_any, n_shift, min_shift, rng):
    """one shared circular shift per run per draw, in segment order (a bijection): query i's "true" segment becomes the
    segment k places later in onset order (cyclic), k uniform in [k_min, n - k_min], k_min = median over the run's segments of the number of onsets in [x, x + min_shift).
    (D-085 amendment: the registered nearest-onset mapping is not one-to-one and biased the null below 0.5 in the smoke test.)"""
    maps = {}
    for r, v in cells_any.items():
        on = v['onset']; order = np.argsort(on); n = len(on); span = on.max() - on.min()
        so = on[order]; kmin = int(np.ceil(np.median(np.searchsorted(so, so + min_shift) - np.arange(n))))   # median segments per 30 s
        pos = np.empty(n, int); pos[order] = np.arange(n)                                                # rank of each query in onset order
        ks = rng.integers(kmin, n - kmin + 1, n_shift); maps[r] = order[(pos[None] + ks[:, None]) % n]
    return maps


def stat(M, w):
    """query-weighted mean of subject x run cell means: w[r] queries per run"""
    return float((M.mean(0) * w).sum() / w.sum())


def pigeonhole(M, w, n_boot, rng):
    ns, nr = M.shape; out = np.empty(n_boot)
    for b in range(n_boot):
        si = rng.integers(0, ns, ns); ri = rng.integers(0, nr, nr); out[b] = stat(M[np.ix_(si, ri)], w[ri])
    return out


def q(x): return {p: float(np.percentile(x, p)) for p in (2.5, 5, 95, 97.5)}


def wolpaw(p, n):
    p = min(max(p, 1e-12), 1 - 1e-12); return float(np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1)))


def analyse(cells, subjects, runs, maps, n_shift):
    """observed cell matrix, null statistics (group and per subject), and descriptive layers"""
    M = np.array([[mrp_true(cells[s][r]['R']).mean() for r in runs] for s in subjects]); w = np.array([len(cells[subjects[0]][r]['onset']) for r in runs], float)
    null = np.empty((n_shift, len(subjects), len(runs))); null_top1 = np.empty(n_shift)
    for b in range(n_shift):
        t1 = 0.0
        for jr, r in enumerate(runs):
            j = maps[r][b]
            for js, s in enumerate(subjects):
                R = cells[s][r]['R']; rk = R[np.arange(len(R)), j]; null[b, js, jr] = np.mean(1 - rk / (R.shape[1] - 1)); t1 += (rk == 0).sum()
        null_top1[b] = t1 / (len(subjects) * w.sum())
    g_null = np.array([stat(null[b], w) for b in range(n_shift)]); subj_obs = (M * w).sum(1) / w.sum(); subj_null = (null * w).sum(2) / w.sum()
    return M, w, g_null, null_top1, subj_obs, subj_null


def layers(cells, subjects, runs, k_nb, excl):
    out = {'top1': [], 'top5': [], 'top10': [], 'nb_top1': [], 'excl30_mrp': [], 'cross_mrp': [], 'full_top10': [], 'n_within': []}
    for s in subjects:
        for r in runs:
            v = cells[s][r]; R, S, on = v['R'], v['S'], v['onset']; rk = R[np.arange(len(R)), np.arange(len(R))]; n = len(on)
            out['top1'] += list(rk < 1); out['top5'] += list(rk < 5); out['top10'] += list(rk < 10); out['n_within'].append(n)
            dt = np.abs(on[:, None] - on[None]); np.fill_diagonal(dt, np.inf); nb = np.argsort(dt, 1)[:, :k_nb]
            out['nb_top1'] += list(np.diag(S) > S[np.arange(n)[:, None], nb].max(1))
            far = dt >= excl; out['excl30_mrp'] += list(1 - ((S > np.diag(S)[:, None]) & far).sum(1) / np.maximum(far.sum(1), 1))
            nc = v['n_fold'] - n; out['cross_mrp'] += list(1 - v['rank_cross'] / nc); out['full_top10'] += list(v['rank_full'] < 10)
    return {k: float(np.mean(x)) for k, x in out.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); sp = c['split']; v = c['verdict']; nb, ns = c['bootstrap']['n_boot'], c['shift_null']['n_shift']
    folds = [(0, 0)] if a.smoke else [(k, f) for k in range(5) for f in range(5)]
    cells, hist = load_cells(c, folds, False, a.smoke)
    pool = sorted(s for fs in sp['subject_folds'] for s in fs if s in cells); allsub = sorted(cells); runs = sorted(cells[pool[0]])
    maps = shift_maps(cells[pool[0]], ns, c['shift_null']['min_shift_s'], np.random.default_rng(c['shift_null']['seed']))
    M, w, g_null, null_top1, subj_obs, subj_null = analyse(cells, pool, runs, maps, ns)
    T = stat(M, w); thr = float(np.percentile(g_null, 97.5)); p = float((1 + (g_null >= T).sum()) / (1 + ns))
    rng = np.random.default_rng(c['bootstrap']['seed']); boot = pigeonhole(M, w, nb, rng)
    # PRE meter (same queries minus those whose earlier window starts before the run)
    Mp = np.array([[mrp_true(cells[s][r]['pre_R']).mean() for r in runs] for s in pool]); wp = np.array([len(cells[pool[0]][r]['pre_rows']) for r in runs], float)
    pre = dict(mrp=stat(Mp, wp), ci=q(pigeonhole(Mp, wp, nb, rng)))
    # baselines
    bl = json.load(open(os.path.join(ROOT, 'results/baselines', 'stageB_b1_de_order' + ('_smoke' if a.smoke else '') + '.json')))
    rk = [str(r + 1) for r in runs]
    Mo = np.array([[bl['ORDER'][x]['mrp'] for x in rk]] * len(pool)); order = dict(mrp=stat(Mo, w), ci=q(pigeonhole(Mo, w, nb, rng)))
    de = bl['DE'][c['decoder_setting']]; Md = np.array([[de[str(s)]['cells'][x]['mrp'] for x in rk] for s in pool])
    assert all(de[str(pool[0])]['cells'][x]['n'] == int(wi) for x, wi in zip(rk, w)), 'DE and decoder query sets differ'
    de_res = dict(mrp=stat(Md, w), ci=q(pigeonhole(Md, w, nb, rng)), diff=stat(M - Md, w), diff_ci=q(pigeonhole(M - Md, w, nb, rng)))
    # report only (D-085 amendment 2): pools without the subjects with many BIDS 'bad' channels (CAR does not exclude them)
    clean = {}
    for name, ex in c['report']['bad_channel_pools'].items():
        keep = [i for i, s in enumerate(pool) if s not in ex]
        clean[name] = dict(subjects=[pool[i] for i in keep], mrp=stat(M[keep], w), ci=q(pigeonhole(M[keep], w, nb, rng)),
                           de_diff=stat(M[keep] - Md[keep], w), de_diff_ci=q(pigeonhole(M[keep] - Md[keep], w, nb, rng)))
    # pipeline control
    pr = None
    if not a.smoke:
        pc, _ = load_cells(c, [(c['phaserand_subject_fold'], f) for f in range(5)], True, False); ps = sorted(s for s in pc if s in pool)
        Mr, wr, gr, _, _, _ = analyse(pc, ps, runs, maps, ns); Tr = stat(Mr, wr); thr_r = float(np.percentile(gr, 97.5))
        pr = dict(subjects=ps, mrp=Tr, threshold=thr_r, leak=bool(Tr > thr_r))
    # verdict (D-085)
    lo, hi = 0.5 - v['meter_margin'], 0.5 + v['meter_margin']
    meters_ok = all(x['ci'][95] < hi for x in (pre, order)); bci = q(boot)              # D-085 amendment: one-sided (upper bound)
    meters_ok_two_sided = all(lo < x['ci'][5] and x['ci'][95] < hi for x in (pre, order))   # registered two-sided rule, reported
    core = p <= v['alpha'] and bci[2.5] > thr
    if pr is not None and pr['leak']: verdict = 'hold_pipeline'
    elif core and meters_ok: verdict = 'present'
    elif core: verdict = 'hold_meter'
    elif bci[95] < 0.5 + v['sesoi']: verdict = 'sesoi_excluded'
    else: verdict = 'undecidable'
    verdict_two_sided = verdict if verdict not in ('present', 'hold_meter') else ('present' if meters_ok_two_sided else 'hold_meter')
    note = ('order_below: the position path may have lowered the decoder score' if verdict in ('sesoi_excluded', 'undecidable') and order['ci'][5] < lo else None)
    # reports
    N = np.mean([len(cells[pool[0]][r]['onset']) for r in runs]); lay = layers(cells, pool, runs, c['eval']['neighbours'], c['eval']['exclusion_s'][1])
    M19, w19, *_ = analyse(cells, allsub, runs, {r: maps[r][:1] for r in runs}, 1)
    exceed = subj_obs > np.percentile(subj_null, 95, axis=0); m_obs = subj_obs.min(); m_null = subj_null.min(1)
    pg = float((1 + (m_null >= m_obs).sum()) / (1 + ns)); N_s = len(pool)
    gamma = max(0.0, (0.05 ** (1 / N_s) - pg ** (1 / N_s)) / (1 - pg ** (1 / N_s))) if pg <= 0.05 else 0.0
    best_ep = {name: int(np.argmin([h['valid'] for h in H if h['stage'] == 'shared'])) for name, H in hist.items()}
    n_ep = {name: sum(h['stage'] == 'shared' for h in H) for name, H in hist.items()}
    res = dict(decoder_setting=c['decoder_setting'], eval_split='valid' if a.smoke else 'test', subjects=pool, runs=[r + 1 for r in runs],
               verdict=verdict, verdict_two_sided_meters=verdict_two_sided, order_note=note, mrp=T, permutation_threshold=thr, permutation_p=p, bootstrap=bci, meters=dict(ORDER=order, PRE=pre, ok=meters_ok, ok_two_sided=meters_ok_two_sided),
               DE=de_res, without_bad_channel_subjects=clean, phaserand=pr, mrp_19=stat(M19, w19), layers=lay, per_subject=dict(mrp=subj_obs.tolist(), exceed_own_95=exceed.tolist(),
               prevalence_p=pg, prevalence_gamma0=gamma), bits=dict(per_selection=wolpaw(lay['top1'], N), per_min=wolpaw(lay['top1'], N) * 20,
               threshold_top1=float(np.percentile(null_top1, 97.5)), threshold_per_min=wolpaw(float(np.percentile(null_top1, 97.5)), N) * 20, n_candidates=N),
               shared_best_epoch=best_ep, shared_epochs=n_ep)
    out = os.path.join(ROOT, c['out'] + ('_smoke' if a.smoke else '')); os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, 'verdict.json'), 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('verdict', 'mrp', 'permutation_threshold', 'permutation_p', 'bootstrap')}), flush=True)


if __name__ == '__main__':
    main()
