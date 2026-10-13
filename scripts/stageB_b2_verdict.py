"""Stage B B2 verdict (D-091 with amendment 1) from the 25 cross-fitted language-target runs, the 5 phase-randomised runs,
the pair sets (scripts/stageB_b2_prepare.py) and the meters / DE baseline (scripts/stageB_b2_meters.py).

Primary statistic: per query, 1 - (number of its 19 within-run acoustic pairs scoring above the true segment) / 19;
query-weighted mean of subject x run cell means over the 17 pool subjects and 20 runs. Null: the B1 onset-order circular
shift (k_min from 30 s, one shift per run and draw shared by all subjects); in a draw query i's answer becomes j = shift(i)
and its candidates the pairs of j. Interval: subject x run pigeonhole bootstrap. Meters ORDER / PRE / ACOUSTIC (D-091
amendment 3): cell-wise excess over the meter's own circular-shift null mean (prediction rows fixed, answer and candidates
shifted), one-sided 90% upper < meter_margin (two-sided and raw reported); RUN, ACOUSTIC_NL and ACOUSTIC_orig are report
only. The decoder's excess over its own null mean is reported as a description. Report (i): the B1 w2v decoder ranked inside the same within-run
pools, from the B1 within-run score matrices.

usage: python scripts/stageB_b2_verdict.py configs/stageB_b2.yaml [--smoke]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageB_b1_verdict import stat, pigeonhole, q, wolpaw, shift_maps

POOLS = ('within', 'cross', 'mel_within', 'within_f0raw')


def find_run(c, k, f, phaserand, smoke, kind='b2'):
    pat = f"*_stageB_{kind}_{c['decoder_setting']}_k{k}_f{f}{'_phaserand' if phaserand else ''}{'_smoke' if smoke else ''}"
    hits = [h for h in sorted(glob.glob(os.path.join(ROOT, c['out_root'], pat))) if os.path.exists(os.path.join(h, 'scores.npz'))]
    assert len(hits) == 1, f'{pat}: {len(hits)} finished run folders'; return hits[0]


def load(c, PR, folds, phaserand, smoke, split):
    """cells[pool][s][r] = dict(rk, valid, rows, onset, pairrows) with S kept per (s, f) for the null"""
    data, hist = {}, {}
    for k, f in folds:
        d = find_run(c, k, f, phaserand, smoke); cf = yaml.safe_load(open(os.path.join(d, 'config.yaml')))['run']
        Z = np.load(os.path.join(d, 'scores.npz')); hist[os.path.basename(d)] = json.load(open(os.path.join(d, 'history.json')))
        g = Z['item_idx']; loc = {int(x): i for i, x in enumerate(g)}; run = Z['run']; on = Z['onset']; pre_ok = Z['pre_ok']
        pool = PR[f'pool_{split}_f{f}']; posp = {int(x): i for i, x in enumerate(pool)}; pr = {}
        for p in POOLS:
            A = PR[f'pairs_{p}_{split}_f{f}'][[posp[int(x)] for x in g]]
            pr[p] = np.array([[loc[int(x)] for x in row] if row[0] >= 0 else [-1] * len(row) for row in A])
        for s in cf['targets']:
            data[s] = dict(S=Z[f'main_s{s}'].astype(np.float32), P=Z[f'pre_s{s}'].astype(np.float32), run=run, onset=on, pre_ok=pre_ok, pairs=pr, fold=f)
    return data, hist


def rank_rows(S, rows, cand, pairs):
    """rank of S[row, cand] among S[row, pairs[cand]] for aligned arrays rows, cand (pairs rows of -1 give nan)"""
    P = pairs[cand]; ok = P[..., 0] >= 0; rk = (S[rows[..., None], np.maximum(P, 0)] > S[rows, cand][..., None]).sum(-1)
    return np.where(ok, 1 - rk / P.shape[-1], np.nan), rk


def cells(data, subjects, runs, pool, kind='S', rows_ok=None):
    M = np.full((len(subjects), len(runs)), np.nan); W = np.zeros(len(runs)); T1 = []
    for js, s in enumerate(subjects):
        v = data[s]; S = v[kind]
        for jr, r in enumerate(runs):
            m = np.where((v['run'] == r) & (v['pre_ok'] if kind == 'P' else True))[0]; x, rk = rank_rows(S, m, m, v['pairs'][pool])
            M[js, jr] = np.nanmean(x); W[jr] = np.isfinite(x).sum(); T1 += list(rk[np.isfinite(x)] < 1)
    return M, W, float(np.mean(T1))


def null_dist(data, subjects, runs, pool, maps, n_shift, kind='S'):
    """rows fixed at query i, answer and candidates replaced by the shifted partner j and its pairs (PRE, D-091 amendment 5: rows
    and answers with pre_ok only, as in the observed PRE cells)"""
    null = np.empty((n_shift, len(subjects), len(runs)))
    for js, s in enumerate(subjects):
        v = data[s]
        for jr, r in enumerate(runs):
            m = np.where(v['run'] == r)[0]; J = m[maps[r]]                                   # [n_shift, n_r] answer rows
            keep = v['pre_ok'][m] if kind == 'P' else np.ones(len(m), bool)
            x, _ = rank_rows(v[kind], np.broadcast_to(m[keep], J[:, keep].shape), J[:, keep], v['pairs'][pool])
            if kind == 'P': x = np.where(v['pre_ok'][J[:, keep]], x, np.nan)
            null[:, js, jr] = np.nanmean(x, 1)
    return null


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    c2 = yaml.safe_load(open(a.config)); c = yaml.safe_load(open(os.path.join(ROOT, c2['b1_config']))); c['out_root'] = c2['out_root']
    sp = c['split']; v = c2['verdict']; nb, ns = c2['bootstrap']['n_boot'], c2['shift_null']['n_shift']; split = 'valid' if a.smoke else 'test'
    PR = np.load(os.path.join(ROOT, c2['cache'], 'pairs.npz'))
    folds = [(0, 0)] if a.smoke else [(k, f) for k in range(5) for f in range(5)]
    data, hist = load(c, PR, folds, False, a.smoke, split)
    pool_s = sorted(s for fs in sp['subject_folds'] for s in fs if s in data); allsub = sorted(data)
    runs = sorted(np.unique(np.concatenate([data[s]['run'] for s in pool_s])).tolist())
    def by_fold_maps(seed):
        rng = np.random.default_rng(seed); out = {}
        for r in runs:
            s0 = next(s for s in pool_s if r in data[s]['run']); on = data[s0]['onset'][data[s0]['run'] == r]
            out.update(shift_maps({r: {'onset': on}}, ns, c2['shift_null']['min_shift_s'], rng))
        return out
    maps = by_fold_maps(c2['shift_null']['seed']); P1 = c2['pairs']['primary']
    M, w, top1 = cells(data, pool_s, runs, P1); null = null_dist(data, pool_s, runs, P1, maps, ns)
    T = stat(M, w); g_null = np.array([stat(null[b], w) for b in range(ns)]); thr = float(np.percentile(g_null, 97.5)); p = float((1 + (g_null >= T).sum()) / (1 + ns))
    rng = np.random.default_rng(c2['bootstrap']['seed']); boot = pigeonhole(M, w, nb, rng); bci = q(boot)
    dec_excess = dict(null_mean=stat(null.mean(0), w), excess=stat(M - null.mean(0), w), ci=q(pigeonhole(M - null.mean(0), w, nb, rng)))
    Mp, wp, _ = cells(data, pool_s, runs, P1, 'P'); Np = null_dist(data, pool_s, runs, P1, maps, ns, 'P').mean(0)
    pre = dict(mrp=stat(Mp, wp), ci=q(pigeonhole(Mp, wp, nb, rng)), null_mean=stat(Np, wp), excess=stat(Mp - Np, wp), excess_ci=q(pigeonhole(Mp - Np, wp, nb, rng)))
    bl = json.load(open(os.path.join(ROOT, 'results/baselines', 'stageB_b2_meters' + ('_smoke' if a.smoke else '') + '.json'))); rk = [str(r + 1) for r in runs]
    meters = {'PRE': pre}
    for name in ('ORDER', 'ACOUSTIC', 'RUN', 'ACOUSTIC_NL', 'ACOUSTIC_orig'):
        for pl in POOLS:
            Mo = np.array([[bl[name][pl][x]['mrp'] for x in rk]] * len(pool_s)); wo = np.array([bl[name][pl][x]['n'] for x in rk], float)
            No = np.array([[bl[name][pl][x]['null_mean'] for x in rk]] * len(pool_s))
            meters[f'{name}_{pl}'] = dict(mrp=stat(Mo, wo), ci=q(pigeonhole(Mo, wo, nb, rng)), null_mean=stat(No, wo), excess=stat(Mo - No, wo),
                                          excess_ci=q(pigeonhole(Mo - No, wo, nb, rng)))
    Md = np.array([[bl['DE'][str(s)][P1][x]['mrp'] for x in rk] for s in pool_s])
    de = dict(mrp=stat(Md, w), ci=q(pigeonhole(Md, w, nb, rng)), diff=stat(M - Md, w), diff_ci=q(pigeonhole(M - Md, w, nb, rng)),
              note='B2 DE baseline: CAR leaves out BIDS bad and flat channels (D-091 amendment 4); B1 DE: A4 CAR (flat only)')
    pr = None
    if not a.smoke:
        pdata, _ = load(c, PR, [(c2['phaserand_subject_fold'], f) for f in range(5)], True, False, split); ps = sorted(s for s in pdata if s in pool_s)
        Mr, wr, _ = cells(pdata, ps, runs, P1); gr = null_dist(pdata, ps, runs, P1, maps, ns); Tr = stat(Mr, wr)
        thr_r = float(np.percentile([stat(gr[b], wr) for b in range(ns)], 97.5)); pr = dict(subjects=ps, mrp=Tr, threshold=thr_r, leak=bool(Tr > thr_r))
    nl_seeds = {}                                                                                   # amendment 5: per-seed ACOUSTIC_NL excess
    for sd, cl in bl['ACOUSTIC_NL_seeds'].items():
        Mo = np.array([[cl[P1][x]['mrp'] - cl[P1][x]['null_mean'] for x in rk]] * len(pool_s)); wo = np.array([cl[P1][x]['n'] for x in rk], float)
        nl_seeds[sd] = stat(Mo, wo)
    nl_rule = dict(acoustic_nl_mean_excess=meters[f'ACOUSTIC_NL_{P1}']['excess'], acoustic_nl_seed_range=[min(nl_seeds.values()), max(nl_seeds.values())],
                   acoustic_nl_seed_excess=nl_seeds, decoder_excess=dec_excess['excess'], decoder_excess_lower90=dec_excess['ci'][5])
    nl_rule['within_nonlinear_acoustic_range'] = bool(nl_rule['decoder_excess_lower90'] < nl_rule['acoustic_nl_mean_excess'])
    mg = v['meter_margin']; judged = [meters['PRE'], meters[f'ORDER_{P1}'], meters[f'ACOUSTIC_{P1}']]
    meters_ok = all(x['excess_ci'][95] < mg for x in judged); meters_ok2 = all(-mg < x['excess_ci'][5] and x['excess_ci'][95] < mg for x in judged)
    core = p <= v['alpha'] and bci[2.5] > thr
    if pr is not None and pr['leak']: verdict = 'hold_pipeline'
    elif core and meters_ok: verdict = 'present'
    elif core: verdict = 'hold_meter'
    elif bci[95] < 0.5 + v['sesoi']: verdict = 'sesoi_excluded'
    else: verdict = 'undecidable'
    verdict2 = verdict if verdict not in ('present', 'hold_meter') else ('present' if meters_ok2 else 'hold_meter')
    # reports
    sens = {}
    for pl in POOLS[1:]:
        Ms, ws, _ = cells(data, pool_s, runs, pl); D_ = Ms - M
        sens[pl] = dict(mrp=stat(Ms, ws), ci=q(pigeonhole(Ms, ws, nb, rng)), minus_primary=stat(D_, w), minus_primary_ci=q(pigeonhole(D_, w, nb, rng)))
    M19, w19, _ = cells(data, allsub, runs, P1)
    bad = {}
    for name, ex in c['report']['bad_channel_pools'].items():
        keep = [i for i, s in enumerate(pool_s) if s not in ex]; bad[name] = dict(subjects=[pool_s[i] for i in keep], mrp=stat(M[keep], w), ci=q(pigeonhole(M[keep], w, nb, rng)))
    subj_obs = (M * w).sum(1) / w.sum(); subj_null = (null * w).sum(2) / w.sum(); exceed = subj_obs > np.percentile(subj_null, 95, axis=0)
    w2v = None
    if not a.smoke:                                                                     # report (i): B1 w2v decoder in the within-run pools
        w2v = {}
        for pl in ('within', 'mel_within'):
            Mw = np.full((len(pool_s), len(runs)), np.nan)
            for k, f in folds:
                d = find_run(c, k, f, False, False, 'b1'); Zb = np.load(os.path.join(d, 'scores.npz')); cf = yaml.safe_load(open(os.path.join(d, 'config.yaml')))['run']
                for s in cf['targets']:
                    if s not in pool_s: continue
                    v2 = data[s]
                    for jr, r in enumerate(runs):
                        if r not in Zb['run']: continue
                        m = np.where(v2['run'] == r)[0]; on_b1 = Zb['onset'][Zb['run'] == r]; assert np.allclose(on_b1, v2['onset'][m])
                        Sb = np.full_like(v2['S'], np.nan); Sb[np.ix_(m, m)] = Zb[f'main_s{s}_r{r}']; x, _ = rank_rows(Sb, m, m, v2['pairs'][pl])
                        Mw[pool_s.index(s), jr] = np.nanmean(x)
            w2v[pl] = dict(mrp=stat(Mw, w), ci=q(pigeonhole(Mw, w, nb, rng)), b2_minus_w2v=stat(M - Mw, w) if pl == 'within' else None,
                           note='B1 scores: CAR leaves out flat channels only, shared network chosen per 1,200-update epoch; '
                                '(ii) leaves out BIDS bad channels too and is evaluated every 150 updates (D-091 amendment 4)')
    best_ep = {nm: min((h for h in H if h['stage'] == 'shared'), key=lambda h: h['valid'])['update'] for nm, H in hist.items()}   # updates (amendment 4)
    def stop_without_floor(H):
        """update at which the patience rule alone (no min_updates) would have stopped, or None"""
        best, bad = float('inf'), 0
        for h in (h for h in H if h['stage'] == 'shared'):
            if h['valid'] < best: best, bad = h['valid'], 0
            else:
                bad += 1
                if bad >= c2['shared_training']['patience_units']: return h['update']
        return None
    floor = {nm: (lambda st: st is not None and st < best_ep[nm])(stop_without_floor(H)) for nm, H in hist.items()}
    best_dist = {str(u): int(sum(x == u for x in best_ep.values())) for u in sorted(set(best_ep.values()))}
    res = dict(decoder_setting=c['decoder_setting'], eval_split=split, primary_pool=P1, subjects=pool_s, runs=[r + 1 for r in runs], verdict=verdict,
               verdict_two_sided_meters=verdict2, mrp=T, permutation_threshold=thr, permutation_p=p, bootstrap=bci, decoder_null_excess=dec_excess, nonlinear_acoustic_rule=nl_rule, top1=top1,
               bits_per_min=wolpaw(top1, 20) * 20, meters=meters, meters_ok=meters_ok, meters_ok_two_sided=meters_ok2, DE=de, phaserand=pr,
               sensitivity=sens, mrp_19=stat(M19, w19), bad_channel_pools=bad, w2v_report=w2v, n_queries_per_run=w.tolist(),
               per_subject=dict(mrp=subj_obs.tolist(), exceed_own_95=exceed.tolist()), shared_best_update=best_ep,
               shared_best_update_counts=best_dist, min_updates_floor_used=dict(n=int(sum(floor.values())), runs=[k for k, x in floor.items() if x]))
    out = os.path.join(ROOT, c2['out'] + ('_smoke' if a.smoke else '')); os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, 'verdict.json'), 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('verdict', 'mrp', 'permutation_threshold', 'permutation_p', 'bootstrap')}), flush=True)
    print('decoder excess over own null mean', json.dumps(dec_excess), flush=True)
    print('nonlinear acoustic rule', json.dumps(nl_rule), flush=True)
    print('meters (mrp, null mean, excess, excess 90%)', json.dumps({k: (round(x['mrp'], 4), round(x['null_mean'], 4), round(x['excess'], 4),
          [round(x['excess_ci'][5], 4), round(x['excess_ci'][95], 4)]) for k, x in meters.items()}), flush=True)


if __name__ == '__main__':
    main()
