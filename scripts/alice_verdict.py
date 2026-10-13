"""D-092 verdict: REVE-Base front end vs the Défossez front end on Alice EEG (component-selection rule, outside F1).

Statistic per arm: query-weighted mean of the within-segment mean rank percentile (MRP) over the present target x segment
cells (33 subjects x 12 segments; S26, S34, S35, S36 have no segment 1). Δ = MRP(REVE) − MRP(Défossez) on paired cells.
Intervals: subject x segment pigeonhole bootstrap (Owen 2007), 1,000 draws; "90% interval" = 5th..95th percentiles.
Existence per arm: 1,000 circular shifts in segment order (one shift per segment and draw, shared by all subjects and both
arms; k uniform in [k_min, n − k_min], k_min = ceil(median over queries of the number of onsets in [x, x + 15 s))), p =
(1 + #null >= observed) / 1,001 (D-085 amendment).
Verdict (D-092 + supplement 6), checked in the order of B1 (D-085): adopt = (1) Δ 90% lower > 0, (2) REVE p <= 0.05,
(3) per arm PRE 90% upper < 0.52 and no phase-randomisation leak, and ΔPRE 90% upper < +0.02; hold = (1) and (2) without (3);
futile = Δ 90% upper < +0.02; otherwise undecidable.
Reports (never in the verdict): ORDER reference line, DE + ridge differences, pool Q1, position-removed scores (supplement 6),
top-k layers, per-subject values, frozen-encoder runs, encoder learning-rate selection runs, epochs.

usage: python scripts/alice_verdict.py configs/alice_verdict.yaml [--smoke]
       --smoke: the two smoke runs (validation segment only) plus the null-uniformity check (uninformative score matrices)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))


def find_run(c, pat):
    hits = [h for h in sorted(glob.glob(os.path.join(ROOT, c['out_root'], '*_' + pat))) if os.path.exists(os.path.join(h, 'scores.npz'))]
    assert len(hits) == 1, f'{pat}: {len(hits)} finished run folders'; return hits[0]


def rank_matrix(S):
    Ssort = np.sort(S, axis=1); return np.stack([len(row) - np.searchsorted(srt, row, side='right') for row, srt in zip(S, Ssort)])


def load(c, pats):
    """cells[s][r] (r 1-based segment) = dict(S, R, onset, P (PRE subset matrix), pre_rows); plus run summaries"""
    cells, runs = {}, {}
    for pat in pats:
        d = find_run(c, pat); Z = np.load(os.path.join(d, 'scores.npz')); m = json.load(open(os.path.join(d, 'metrics.json'))); runs[os.path.basename(d)] = m
        h = json.load(open(os.path.join(d, 'history.json'))); runs[os.path.basename(d)]['history'] = h
        for s, v in m['targets'].items():
            for r in v['segments']:
                S = Z[f'main_s{s}_r{r}']; P = Z[f'pre_s{s}_r{r}']
                cells.setdefault(int(s), {})[int(r)] = dict(S=S, R=rank_matrix(S), onset=Z[f'onset_r{r}'], P=P, PR=rank_matrix(P), pre_rows=Z[f'pre_rows_r{r}'],
                                                            rank_full=Z[f'main_s{s}_rank_full'][Z[f'main_s{s}_segs'] == int(r)])
    return cells, runs


def mrp_diag(R): n = R.shape[1]; return 1 - np.diag(R) / (n - 1)


def matrix(cells, subjects, segs, key='R'):
    M = np.full((len(subjects), len(segs)), np.nan)
    for i, s in enumerate(subjects):
        for j, r in enumerate(segs):
            if r in cells.get(s, {}): M[i, j] = mrp_diag(cells[s][r][key]).mean()
    return M


def stat(M, w):
    ok = ~np.isnan(M); return float(np.nansum(M * w) / (ok * w).sum())


def pigeonhole(M, w, n_boot, rng):
    ns, nr = M.shape; out = np.empty(n_boot)
    for b in range(n_boot):
        si = rng.integers(0, ns, ns); ri = rng.integers(0, nr, nr); out[b] = stat(M[np.ix_(si, ri)], w[ri])
    return out


def q(x): return {p: float(np.percentile(x, p)) for p in (2.5, 5, 95, 97.5)}


def shift_maps(onsets, n_shift, min_shift, rng):
    maps = {}
    for r, on in onsets.items():
        order = np.argsort(on); n = len(on); so = on[order]
        kmin = int(np.ceil(np.median(np.searchsorted(so, so + min_shift) - np.arange(n)))); pos = np.empty(n, int); pos[order] = np.arange(n)
        ks = rng.integers(kmin, n - kmin + 1, n_shift); maps[r] = order[(pos[None] + ks[:, None]) % n]
    return maps


def null_stats(cells, subjects, segs, w, maps, n_shift, key='R'):
    null = np.full((n_shift, len(subjects), len(segs)), np.nan)
    for j, r in enumerate(segs):
        for i, s in enumerate(subjects):
            if r not in cells.get(s, {}): continue
            R = cells[s][r][key]; rk = R[np.arange(len(R))[None], maps[r]]; null[:, i, j] = (1 - rk / (R.shape[1] - 1)).mean(1)
    return np.array([stat(null[b], w) for b in range(n_shift)]), null


def residual_scores(S, Pz):
    """S − b·Pz with b from the off-diagonal cells after removing row means (row intercepts do not change row ranks)"""
    off = ~np.eye(len(S), dtype=bool)
    def dm(A): return A - np.where(off, A, 0).sum(1, keepdims=True) / off.sum(1, keepdims=True)
    Sd, Pd = dm(S), dm(Pz); b = (Sd[off] * Pd[off]).sum() / max((Pd[off] ** 2).sum(), 1e-12); return S - b * Pz, float(b)


def position_scores(c):
    """per segment (1-based): cosine of the ORDER position prediction of each test query with the pooled wav2vec target of each
    candidate in the same segment; ORDER fitted on the training windows of the fold in which the segment is a test segment"""
    from alice_meters import fit_predict, unit
    m = yaml.safe_load(open(os.path.join(ROOT, c['meters_config']))); d = os.path.join(ROOT, m['cache_dir'])
    meta = json.load(open(os.path.join(d, 'eeg_meta.json'))); sp = json.load(open(os.path.join(d, 'split.json')))['main']
    src = meta['sources'][m['source']]; fs = src['sample_rate']; offs, lens = src['run_offsets'], src['run_lengths']; durs = [L / fs for L in lens]
    w = m['window']; T = int(round((w['tmax'] - w['tmin']) * fs)); lo = -w['tmin']; Wd = np.load(os.path.join(d, 'words.npz'), allow_pickle=True)
    def hi(r): return durs[r] - w['tmax'] - max(w['offset_s'], 0)
    grid = [(r, float(t)) for r in range(12) for t in np.arange(lo, hi(r) + 1e-9, w['train_step_s'])]
    qs = [(r, float(t)) for r in range(12) for t in (lambda on: on[(on >= lo) & (on <= hi(r))])(np.unique(Wd['onset'][Wd['seg'] == r]))]
    items = grid + qs; ng = len(grid); seg = np.array([r for r, _ in items]); tt = np.array([t for _, t in items])
    F = np.load(os.path.join(d, m['features']['w2v'] + '.npy'), mmap_mode='r')
    pooled = np.stack([np.asarray(F[:, offs[r] + int(round((t + w['tmin']) * fs)):][:, :T], np.float32).mean(1) for r, t in items])
    rel = tt / np.array(durs)[seg]; POS = np.c_[rel, rel ** 2, rel ** 3, *[g(2 * np.pi * k * rel) for k in (1, 2, 4, 8) for g in (np.sin, np.cos)]]
    g0 = np.r_[0, np.cumsum(durs)][:-1]; s0 = g0[seg] + tt + w['tmin'] + w['offset_s']; s1 = s0 + (w['tmax'] - w['tmin']); out = {}
    for f, fold in enumerate(sp['segment_folds']):
        test = [x - 1 for x in fold]; valid = sp['valid_segment'][f] - 1; train = [r for r in range(12) if r not in test + [valid]]
        b0, b1 = g0[min(test)], g0[max(test)] + durs[max(test)]
        def dist(b): return np.where((s0 <= b) & (s1 >= b), 0.0, np.minimum(np.abs(s0 - b), np.abs(s1 - b)))
        ok = np.minimum(dist(b0), dist(b1)) >= m['guard_s']; tr = np.where(np.isin(seg[:ng], train) & ok[:ng])[0]
        samp = np.concatenate([np.arange(offs[r], offs[r] + lens[r]) for r in train]); samp = np.sort(samp[:: max(1, len(samp) // 200000)])
        Fs = np.asarray(F[:, samp], np.float32); Y = (pooled - Fs.mean(1)) / (Fs.std(1) + 1e-6)
        for r in test:
            te = ng + np.where(seg[ng:] == r)[0]; P, _ = fit_predict(POS[tr], Y[tr], POS[te], seg[tr]); out[r + 1] = unit(P) @ unit(Y[te]).T
    return out


def wolpaw(p, n):
    p = min(max(p, 1e-12), 1 - 1e-12); return float(np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1)))


def layers(cells, subjects, segs):
    out = {'top1': [], 'top5': [], 'top10': [], 'full_top10': []}
    for s in subjects:
        for r in segs:
            if r not in cells.get(s, {}): continue
            rk = np.diag(cells[s][r]['R']); out['top1'] += list(rk < 1); out['top5'] += list(rk < 5); out['top10'] += list(rk < 10)
            out['full_top10'] += list(cells[s][r]['rank_full'] < 10)
    return {k: float(np.mean(v)) for k, v in out.items()}


def null_uniformity(cells, subjects, segs, w, c, rng):
    """uninformative score matrices: (a) collapsed (every row = the column mean), (b) rows permuted at random inside every
    cell (no query-candidate correspondence); the p of the shift null should not be small for (a) and be near uniform for (b)"""
    ns = c['null_check']['n_shift']; onsets = {r: cells[subjects[0]][r]['onset'] for r in segs}
    res = {}
    col = {s: {r: dict(R=rank_matrix(np.repeat(v['S'].mean(0, keepdims=True), len(v['S']), 0) + 1e-9 * rng.standard_normal(v['S'].shape)))
               for r, v in cells[s].items()} for s in subjects}
    maps = shift_maps(onsets, ns, c['shift_null']['min_shift_s'], rng); M = matrix(col, subjects, segs); g, _ = null_stats(col, subjects, segs, w, maps, ns)
    res['collapsed'] = dict(mrp=stat(M, w), p=float((1 + (g >= stat(M, w)).sum()) / (1 + ns)))
    ps = []
    for _ in range(c['null_check']['n_replicates']):
        perm = {s: {r: dict(R=rank_matrix(v['S'][rng.permutation(len(v['S']))])) for r, v in cells[s].items()} for s in subjects}
        maps = shift_maps(onsets, ns, c['shift_null']['min_shift_s'], rng); M = matrix(perm, subjects, segs); g, _ = null_stats(perm, subjects, segs, w, maps, ns)
        ps.append(float((1 + (g >= stat(M, w)).sum()) / (1 + ns)))
    ps = np.array(ps); from scipy.stats import kstest
    res['row_permuted'] = dict(n=len(ps), frac_p_le_005=float((ps <= 0.05).mean()), frac_p_le_010=float((ps <= 0.10).mean()), mean_p=float(ps.mean()),
                               ks_uniform_p=float(kstest(ps, 'uniform').pvalue), deciles=np.percentile(ps, np.arange(10, 100, 10)).round(3).tolist())
    return res


def analyse_arm(cells, subjects, segs, w, maps, ns, nb, rng):
    M = matrix(cells, subjects, segs); T = stat(M, w); g, null = null_stats(cells, subjects, segs, w, maps, ns)
    Mp = matrix(cells, subjects, segs, 'PR'); wp = np.array([len(cells[next(s for s in subjects if r in cells[s])][r]['pre_rows']) for r in segs], float)
    subj = np.array([stat(M[i:i + 1], w) for i in range(len(subjects))])
    return dict(M=M, Mp=Mp, wp=wp, res=dict(mrp=T, p=float((1 + (g >= T).sum()) / (1 + ns)), threshold=float(np.percentile(g, 97.5)),
                                           ci=q(pigeonhole(M, w, nb, rng)), pre=dict(mrp=stat(Mp, wp), ci=q(pigeonhole(Mp, wp, nb, rng))),
                                           per_subject=dict(zip(map(str, subjects), subj.round(5).tolist())), layers=layers(cells, subjects, segs)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); nb, ns = c['bootstrap']['n_boot'], c['shift_null']['n_shift']; v = c['verdict']
    sp = json.load(open(os.path.join(ROOT, c['cache_dir'], 'split.json')))['main']
    if a.smoke: pats = {arm: [c['smoke_runs'][arm]] for arm in ('defossez', 'reve')}
    else: pats = {arm: [f'alice_{arm}_k{k}_f{f}' for k in range(5) for f in range(4)] for arm in ('defossez', 'reve')}
    C, RUNS = {}, {}
    for arm, pl in pats.items(): C[arm], RUNS[arm] = load(c, pl)
    subjects = sorted(set(C['defossez']) & set(C['reve'])); segs = sorted({r for s in subjects for r in C['defossez'][s]})
    onsets = {r: C['defossez'][next(s for s in subjects if r in C['defossez'][s])][r]['onset'] for r in segs}; w = np.array([len(onsets[r]) for r in segs], float)
    maps = shift_maps(onsets, ns, c['shift_null']['min_shift_s'], np.random.default_rng(c['shift_null']['seed'])); rng = np.random.default_rng(c['bootstrap']['seed'])
    A = {arm: analyse_arm(C[arm], subjects, segs, w, maps, ns, nb, rng) for arm in C}
    D = A['reve']['M'] - A['defossez']['M']; Dp = A['reve']['Mp'] - A['defossez']['Mp']
    delta = dict(value=stat(D, w), ci=q(pigeonhole(D, w, nb, rng))); dpre = dict(value=stat(Dp, A['reve']['wp']), ci=q(pigeonhole(Dp, A['reve']['wp'], nb, rng)))
    # pipeline control (phase-randomised, subject fold 0 x 4 segment folds)
    pr = {}
    if not a.smoke:
        for arm in C:
            pc, _ = load(c, [f'alice_{arm}_k{c["phaserand_subject_fold"]}_f{f}_phaserand' for f in range(4)]); ps = sorted(pc)
            g, null = null_stats(pc, ps, segs, w, maps, ns); Mr = matrix(pc, ps, segs); Tr = stat(Mr, w)
            subj_thr = np.nanpercentile((null * w).sum(2) / ((~np.isnan(null)) * w).sum(2), 97.5, axis=0); subj_obs = np.array([stat(Mr[i:i + 1], w) for i in range(len(ps))])
            pr[arm] = dict(subjects=ps, mrp=Tr, threshold=float(np.percentile(g, 97.5)), leak=bool(Tr > np.percentile(g, 97.5)),
                           subjects_above_own_975=[s for s, o, t in zip(ps, subj_obs, subj_thr) if o > t])
    up = lambda x: x['ci'][95]; lo = lambda x: x['ci'][5]
    c1 = lo(delta) > 0; c2 = A['reve']['res']['p'] <= v['alpha']
    c3 = dict(pre_ok={arm: up(A[arm]['res']['pre']) < 0.5 + v['meter_margin'] for arm in A}, phaserand_ok={arm: (not pr[arm]['leak']) if arm in pr else None for arm in A},
              dpre_ok=up(dpre) < v['sesoi'])
    c3_all = all(c3['pre_ok'].values()) and all(x is not False for x in c3['phaserand_ok'].values()) and c3['dpre_ok'] and (a.smoke or all(x is True for x in c3['phaserand_ok'].values()))
    verdict = 'adopt' if c1 and c2 and c3_all else 'hold' if c1 and c2 else 'futile' if up(delta) < v['sesoi'] else 'undecidable'
    # reports
    bl = json.load(open(os.path.join(ROOT, c['baselines'])))
    Md = np.full_like(A['defossez']['M'], np.nan)
    for i, s in enumerate(subjects):
        for j, r in enumerate(segs):
            x = bl['DE'][str(s)]['cells'].get(str(r))
            if x is not None and not np.isnan(A['defossez']['M'][i, j]): Md[i, j] = x['mrp']
    de = dict(mrp=stat(Md, w), **{f'{arm}_minus_DE': dict(value=stat(A[arm]['M'] - Md, w), ci=q(pigeonhole(A[arm]['M'] - Md, w, nb, rng))) for arm in A})
    keep = [i for i, s in enumerate(subjects) if s not in c['report_pool_exclude']]
    q1 = dict(subjects=[subjects[i] for i in keep], **{arm: stat(A[arm]['M'][keep], w) for arm in A}, delta=dict(value=stat(D[keep], w), ci=q(pigeonhole(D[keep], w, nb, rng))),
              de=stat(Md[keep], w))
    pos = position_scores(c); posrm = {}
    for arm in A:
        Mres = np.full_like(A[arm]['M'], np.nan); bs = []
        for i, s in enumerate(subjects):
            for j, r in enumerate(segs):
                if r in C[arm][s]:
                    Rr, b = residual_scores(C[arm][s][r]['S'], pos[r][np.ix_(np.arange(len(pos[r])), np.arange(len(pos[r])))]); Mres[i, j] = mrp_diag(rank_matrix(Rr)).mean(); bs.append(b)
        posrm[arm] = dict(M=Mres, mrp=stat(Mres, w), slope_median=float(np.median(bs)))
    Dr = posrm['reve']['M'] - posrm['defossez']['M']
    posrm_rep = dict(**{arm: dict(mrp=posrm[arm]['mrp'], slope_median=posrm[arm]['slope_median']) for arm in A}, delta=dict(value=stat(Dr, w), ci=q(pigeonhole(Dr, w, nb, rng))))
    N = float(np.mean(w)); res = dict(eval_split='valid' if a.smoke else 'test', subjects=subjects, segments=segs, verdict=verdict,
               conditions=dict(c1_delta_lower_gt0=c1, c2_reve_p=c2, c3=c3, c3_all=c3_all, futile_delta_upper_lt_sesoi=up(delta) < v['sesoi']),
               arms={arm: A[arm]['res'] for arm in A}, delta=delta, delta_pre=dpre, phaserand=pr,
               order_reference=dict(mrp=bl['ORDER_mrp'], upper90_segment_bootstrap=c['order_upper']), DE=de, Q1=q1, position_removed=posrm_rep,
               bits={arm: dict(per_selection=wolpaw(A[arm]['res']['layers']['top1'], N)) for arm in A}, n_candidates_mean=N,
               runs={arm: {k: dict(summary=v_['summary']) for k, v_ in RUNS[arm].items()} for arm in RUNS})
    if a.smoke: res['null_check'] = {arm: null_uniformity(C[arm], subjects, segs, w, c, np.random.default_rng(7)) for arm in C}
    out = os.path.join(ROOT, c['out'] + ('_smoke' if a.smoke else '')); os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, 'verdict.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ('verdict', 'conditions', 'delta', 'delta_pre')}, default=float), flush=True)
    if a.smoke: print(json.dumps(res['null_check']), flush=True)


if __name__ == '__main__':
    main()
