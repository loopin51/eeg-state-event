"""C2 verdict and E1 (D-097 section 14, amendments 2 and 4, F-II alpha record): Delta_k2^DTU = MRP_P2_DTU(k2) - MRP_P1_DTU(k2)
on the DTU two-talker test trials (attended stream), cell = (target subject, test pair), 18 subjects x 62 pairs (cells a
subject did not hear weigh 0), seed-stratified pigeonhole interval and categories of C1 (scripts/c1_stats.py), alpha_A of C1,
E1 at kmax, prerequisites 1-5 on DTU plus the stimulus-only meter UNATTENDED, protection at kmax. This script is the only
reader of the C2 test_cells.json files; the two arms' test cells are opened once, here.

E1 (no alpha): P1_DTU at kmax, within-trial MRP of the seed-mean cells under the segment-order circular shift (k_min 10 s, one
shift per pair for every subject who heard it, 1,000 draws, seed 20261012): p <= 0.05 and the pigeonhole 2.5th percentile above
the null 97.5th percentile. Failing E1 -> 'hold_E1' and chain A's alpha is not used.
Shift-null statistics (E1, phase randomisation, collapse) use only the cells whose query count equals the pair's full query
count; a shorter trial admits no shift of at least 10 s on both sides (S8 trial 46, 33 queries), so it is left out of those
statistics and counted in the output. The interval, the categories and the reports use every cell.
C2 is confirmatory only when the C1 verdict opened it (results/c1/verdict.json c2_opened, i.e. C1 'adopt' with its
prerequisites); otherwise everything here is a descriptive result (D-097 section 14).

usage: python scripts/c2_verdict.py configs/c2_verdict.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, hashlib, argparse, datetime, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import c1_stats as CS
from stageB_b1_verdict import shift_maps
from c2_dtu_split import DTUSplit, trials_from_cache, path
from c1_verdict import pretrain_device


def is_cells(val): return isinstance(val, dict) and val and all(isinstance(x, dict) and 'mrp' in x for x in val.values())


def arm_cfg(p):
    cc = yaml.safe_load(open(path(p))); c = yaml.safe_load(open(path(cc['b1_config']))); fc = yaml.safe_load(open(path(cc['folds_config'])))
    return cc, c, fc


def find_run(v, arm, setting, seed, K, F, phaserand=False):
    pat = f"*_c2_{arm}_{setting}_s{seed}_k{K}_f{F}{'_phaserand' if phaserand else ''}"
    hits = [h for h in sorted(glob.glob(os.path.join(path(v['runs_root']), pat))) if os.path.exists(os.path.join(h, 'test_cells.json'))]
    assert len(hits) == 1, f'{pat}: {len(hits)} finished run folders'; return hits[0]


def load_arm(v, cfg, seeds, split, tp, phaserand=False):
    """runs[seed][(K, F)] = folder; cells[seed][subject][key][pair] = cell dict (test_cells.json, trial number -> pair)"""
    cc, c, _ = arm_cfg(cfg); runs, cells = {}, {}
    folds = [(cc['phaserand_subject_fold'], f) for f in range(5)] if phaserand else [(K, F) for K in range(5) for F in range(5)]
    for seed in seeds:
        runs[seed], cells[seed] = {}, {}
        for K, F in folds:
            d = find_run(v, cc['arm'], c['decoder_setting'], seed, K, F, phaserand); runs[seed][(K, F)] = d
            T = json.load(open(os.path.join(d, 'test_cells.json')))
            assert T['summary']['eval_split'] == 'test' and T['summary']['seed'] == seed and T['summary']['trial_fold'] == F, d
            assert sorted(int(s) for s in T['targets']) == split.subject_folds[K], (d, sorted(T['targets']))
            for s, r in T['targets'].items():
                want = {str(x['trial']) for x in split.sets(int(s), F)['test']}
                for key, val in r.items():
                    if not is_cells(val): continue
                    assert set(val) == want, (d, s, key, sorted(val))
                    dst = cells[seed].setdefault(int(s), {}).setdefault(key, {})
                    for tr, x in val.items(): dst[tp[int(s)][int(tr)]] = x
    return cc, c, runs, cells


def tensor(cells, seeds, subjects, units, key):
    """m [n_seeds, S, U] cell MRP and n [S, U] query counts (identical across seeds, asserted)"""
    m = np.full((len(seeds), len(subjects), len(units)), np.nan); n = np.zeros((len(subjects), len(units)))
    for i, sd in enumerate(seeds):
        for js, s in enumerate(subjects):
            for ju, u in enumerate(units):
                x = cells[sd].get(s, {}).get(key, {}).get(u)
                if x is None: continue
                m[i, js, ju] = x['mrp']; assert n[js, ju] in (0, x['n']), (key, s, u, n[js, ju], x['n']); n[js, ju] = x['n']
    for i in range(len(seeds)): assert np.isfinite(m[i][n > 0]).all(), f'{key}: a seed misses cells'
    return m, n


def topk(cells, seeds, key, which):
    num = den = 0.0
    for sd in seeds:
        for s, r in cells[sd].items():
            for x in r.get(key, {}).values(): num += x[which] * x['n']; den += x['n']
    return num / den


def scores(run_dirs, key, subjects, tp):
    """mats[s][pair] = within-trial score matrix of `key` (e.g. main_kmax); onsets[pair] = query times of the pair's full-length trials;
    short[(s, pair)] = cells whose trial is shorter than the pair's full length (left out of shift-null statistics)"""
    mats, ons = {}, {}
    for d in run_dirs:
        Z = np.load(os.path.join(d, 'scores.npz'))
        for f in Z.files:
            if not f.startswith(key + '_s') or f.endswith('_onset'): continue
            s, t = f[len(key) + 2:].split('_t'); s, t = int(s), int(t)
            if s not in subjects: continue
            p = tp[s][t]; mats.setdefault(s, {})[p] = Z[f].astype(np.float32); ons.setdefault(p, []).append(np.asarray(Z[f + '_onset'], np.float64))
    onsets, short = {}, []
    for p, L in ons.items():
        full = max(L, key=len); onsets[p] = full
        for x in L: assert len(x) == len(full) and np.allclose(x, full) or len(x) < len(full), f'pair {p}: different query grids'
    for s in mats:
        for p in list(mats[s]):
            if mats[s][p].shape[0] != len(onsets[p]): short.append((s, p)); del mats[s][p]
    return mats, onsets, sorted(short)


def maps_for(onsets, v):
    sn = v['shift_null']; rng = np.random.default_rng(sn['seed'])
    return shift_maps({p: {'onset': onsets[p]} for p in sorted(onsets)}, sn['n_shift'], sn['min_shift_s'], rng)


def counts(mats, maps):
    """query counts aligned with CS.shift_null_cells (sorted subjects x sorted pairs of the maps), 0 where no cell"""
    subs, units = sorted(mats), sorted(maps)
    return np.array([[mats[s][u].shape[0] if u in mats[s] else 0 for u in units] for s in subs], float)


def shift_stat(per_seed_mats, maps):
    """seed-mean observed statistic and null statistics (the same shift maps for every seed)"""
    obs, null = zip(*[CS.shift_null_cells(m, maps) for m in per_seed_mats]); n = counts(per_seed_mats[0], maps)
    o = np.stack(obs).mean(0); g = np.stack(null).mean(0)                 # same nan pattern in every seed
    return CS.level(o, n), np.array([CS.level(x, n) for x in g]), n


def judge(point, boot, alpha, sesoi):
    iv = CS.interval(boot, alphas=(0.04, 0.05)); lo = iv[f'lower_{alpha}']; cat, note = CS.category(point, lo, iv['p95'], sesoi)
    return dict(category=cat, note=note, lower=lo)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); v = yaml.safe_load(open(path(a.config)))
    v1 = yaml.safe_load(open(path(v['c1_verdict_config']))); fam = v1['family_state']; sesoi = v1['sesoi']
    alpha, alpha_reason, _ = CS.alpha_chain_A(fam); alpha_used = alpha if alpha is not None else v1['alpha']['full']
    c1v = json.load(open(path(v['c1_verdict_file']))) if os.path.exists(path(v['c1_verdict_file'])) else None
    confirmatory = bool(c1v and c1v.get('c2_opened'))
    iv_var = v1['interval_variant']; assert iv_var in ('inflate3', 'inflate5'), f'C2 interval variant {iv_var}'
    seeds = v['seeds']
    if iv_var == 'inflate5': assert len(seeds) == 5, 'amendment 5: the chosen variant needs 5 seeds'
    _, _, fc = arm_cfg(v['arms']['P1']); split = DTUSplit(fc, trials_from_cache(fc['cache_dir']))
    tp = {s: {r['trial']: r['pair'] for r in R} for s, R in split.trials.items()}
    subjects = split.subjects; units = sorted(fc['partner']); kd, ke = v['decision_k'], v['e1']['k']
    cc1, c, runs1, cells1 = load_arm(v, v['arms']['P1'], seeds, split, tp); cc2, _, runs2, cells2 = load_arm(v, v['arms']['P2'], seeds, split, tp)
    rng = np.random.default_rng(v['bootstrap']['seed']); nb = v['bootstrap']['n_boot']
    SP = lambda arms, nn: CS.seed_pigeonhole(arms, nn, nb, rng, seed_inflate=True)
    T = {}
    for arm, cells in (('P1', cells1), ('P2', cells2)):
        keys = sorted({k for sd in seeds for s in cells[sd] for k in cells[sd][s]}); T[arm] = {k: tensor(cells, seeds, subjects, units, k) for k in keys}
    m1, n = T['P1'][kd]; m2, n2 = T['P2'][kd]; assert np.array_equal(n, n2), 'arms differ in query counts'
    # --- primary (the first n_boot draws of the registered generator)
    point = CS.delta(m1, m2, n); boot = SP([m1, m2], n); iv = CS.interval(boot)
    by_alpha = {str(al): judge(point, boot, al, sesoi) for al in (v1['alpha']['chain_A'], v1['alpha']['full'])}
    category = by_alpha[str(alpha_used)]['category']
    # --- protection at kmax
    pk = v['protect']; a_, b_ = T['P1'][pk['k']][0], T['P2'][pk['k']][0]; nk = T['P1'][pk['k']][1]
    dk = CS.delta(a_, b_, nk); btk = SP([a_, b_], nk); lowk = float(np.percentile(btk, pk['percentile']))
    protect = dict(k=pk['k'], delta=dk, interval=CS.interval(btk), lower=lowk, ok=bool(lowk > pk['margin']),
                   label='' if lowk > pk['margin'] else f'exchange with a {pk["k"]} loss')
    # --- E1 (P1_DTU at kmax, no alpha)
    e = v['e1']; per_seed, onsets, short = [], None, None
    for sd in seeds:
        mats, onsets, short = scores(runs1[sd].values(), f'main_{ke}', subjects, tp); per_seed.append(mats)
    mp = maps_for(onsets, v); Te, ge, ne = shift_stat(per_seed, mp); thr = float(np.percentile(ge, e['null_percentile']))
    bte = SP([T['P1'][ke][0]], T['P1'][ke][1]); lo_e = float(np.percentile(bte, e['boot_percentile'])); pe = CS.perm_p(Te, ge)
    e1 = dict(k=ke, mrp_shift_cells=Te, null_mean=float(ge.mean()), null_threshold=thr, p=pe, mrp_all_cells=CS.level(CS.seed_mean(T['P1'][ke][0]), T['P1'][ke][1]),
              boot_percentile=lo_e, short_cells_left_out=[list(x) for x in short], ok=bool(pe <= e['p_max'] and lo_e > thr))
    # --- prerequisites (section 7 on DTU, amendment 2 condition 3)
    pr = v['prerequisites']; prereq = {}
    q = pr['phaserand']; ph = {}
    for arm, cfg in (('P1', v['arms']['P1']), ('P2', v['arms']['P2'])):
        _, _, prun, _ = load_arm(v, cfg, [q['seed']], split, tp, phaserand=True)
        tg = split.subject_folds[q['subject_fold']]; mats, ons, sh = scores(prun[q['seed']].values(), f'main_{q["k"]}', tg, tp)
        Tp, gp, _ = shift_stat([mats], maps_for(ons, v)); thp = float(np.percentile(gp, q['percentile']))
        ph[arm] = dict(subjects=tg, mrp=Tp, threshold=thp, p=CS.perm_p(Tp, gp), ok=bool(Tp <= thp), short_cells_left_out=[list(x) for x in sh],
                       runs=sorted(os.path.basename(x) for x in prun[q['seed']].values()))
    prereq['1_phaserand'] = dict(arms=ph, ok=all(x['ok'] for x in ph.values()))
    q = pr['pre']; pk_ = f"pre_{q['k']}"; p1, np1 = T['P1'][pk_]; p2, np2 = T['P2'][pk_]; assert np.array_equal(np1, np2)
    pre = {}
    for arm, mm in (('P1', p1), ('P2', p2)):
        bt = SP([mm], np1); pre[arm] = dict(mrp=CS.level(CS.seed_mean(mm), np1), interval=CS.interval(bt)); pre[arm]['ok'] = bool(pre[arm]['interval']['p95'] < q['level_upper'])
    btd = SP([p1, p2], np1); pre['diff'] = dict(delta=CS.delta(p1, p2, np1), interval=CS.interval(btd)); pre['diff']['ok'] = bool(pre['diff']['interval']['p95'] < q['diff_upper'])
    prereq['2_pre'] = dict(pre, ok=pre['P1']['ok'] and pre['P2']['ok'] and pre['diff']['ok'])
    q = pr['de_baseline']; de_path = path(q['file']); sha = hashlib.sha256(open(de_path, 'rb').read()).hexdigest()
    all_runs = [d for R in (runs1, runs2) for sd in R for d in R[sd].values()]
    first_tc = min(os.path.getmtime(os.path.join(d, 'test_cells.json')) for d in all_runs)
    t_de = datetime.datetime.fromisoformat(q['committed_at']).timestamp() if q.get('committed_at') else float('inf')
    prereq['3_de_first'] = dict(sha256_ok=sha == q['sha256'], de_committed_at=q['committed_at'], commit=q['commit'],
                                first_test_cells_mtime=datetime.datetime.fromtimestamp(first_tc).isoformat(), ok=bool(sha == q['sha256'] and t_de < first_tc))
    q = pr['collapse']; col = {}
    for arm, R in (('P1', runs1), ('P2', runs2)):
        mats, ons, sh = scores(R[q['seed']].values(), f'main_{q["k"]}', subjects, tp); mpc = maps_for(ons, v)
        col[arm] = dict(CS.collapse_check(mats, mpc, counts(mats, mpc), q['p_min'], q['ks_min'], q['identity_passes']), short_cells_left_out=len(sh))
    prereq['4_collapse'] = dict(arms=col, ok=all(x['passed'] for x in col.values()))
    devs = {}
    for arm, cfg in (('P1', v['arms']['P1']), ('P2', v['arms']['P2'])):
        R = runs1 if arm == 'P1' else runs2; _, _, prun, _ = load_arm(v, cfg, [pr['phaserand']['seed']], split, tp, phaserand=True)
        for d in [d for sd in R for d in R[sd].values()] + list(prun[pr['phaserand']['seed']].values()):
            devs[os.path.basename(d)] = json.load(open(os.path.join(d, 'metrics.json')))['summary']['device']
    inits = sorted({json.load(open(os.path.join(d, 'metrics.json')))['summary'].get('init') for d in all_runs} - {None})
    for ck in inits:
        folder, dv = pretrain_device(ck, v['runs_root']); devs[folder] = dv
    bad_dev = {k: x for k, x in devs.items() if pr['device'] not in str(x)}
    prereq['5_device'] = dict(n_jobs=len(devs), n_pretrain=len(inits), not_matching=bad_dev, ok=not bad_dev)
    q = pr['unattended']; M = json.load(open(path(q['file']))); assert M['eval_split'] == 'test', 'UNATTENDED must be measured on the test trials'
    mu = np.full((1, len(subjects), len(units)), np.nan); nu = np.zeros((len(subjects), len(units)))
    for s, r in M['cells'].items():
        for tr, x in r.items():
            js, ju = subjects.index(int(s)), units.index(tp[int(s)][int(tr)]); assert x['pair'] == units[ju]; mu[0, js, ju] = x['mrp']; nu[js, ju] = x['n']
    assert np.array_equal(nu, n), 'UNATTENDED and decoder query sets differ'
    btu = SP([mu], nu); ivu = CS.interval(btu)
    prereq['6_unattended'] = dict(mrp=CS.level(mu[0], nu), interval=ivu, upper=q['upper'], ok=bool(ivu['p95'] < q['upper']))
    prereq_ok = all(x['ok'] for x in prereq.values())
    if not e1['ok']: verdict = 'hold_E1'
    elif not prereq['6_unattended']['ok']: verdict = 'hold_meter'
    elif not prereq_ok: verdict = 'hold'
    else: verdict = category
    # --- reports (section 13 items that apply to C2)
    curves, deltas = {}, {}
    for key in v['ks']:
        a_, nk = T['P1'][key]; b_, _ = T['P2'][key]
        for arm, mm, cells in (('P1', a_, cells1), ('P2', b_, cells2)):
            curves.setdefault(arm, {})[key] = dict(mrp=CS.level(CS.seed_mean(mm), nk), interval=CS.interval(SP([mm], nk)),
                                                   per_seed={str(sd): CS.level(mm[i], nk) for i, sd in enumerate(seeds)},
                                                   seed_sd=float(np.std([CS.level(mm[i], nk) for i in range(len(seeds))], ddof=1)),
                                                   top1=topk(cells, seeds, key, 'top1'), top10=topk(cells, seeds, key, 'top10'))
        deltas[key] = dict(delta=CS.delta(a_, b_, nk), interval=CS.interval(SP([a_, b_], nk)))
    unatt = {}
    for arm in ('P1', 'P2'):
        uk = f'unattended_{kd}'; mm, nn = T[arm][uk]; unatt[arm] = dict(mrp=CS.level(CS.seed_mean(mm), nn), interval=CS.interval(SP([mm], nn)))
    DE = json.load(open(de_path)); de = {}
    for key in v['ks']:
        if key == 'k0': continue
        Md = np.full((1, len(subjects), len(units)), np.nan); nd = np.zeros_like(n)
        for s, r in DE['subjects'].items():
            for tr, x in r[key]['cells'].items():
                js, ju = subjects.index(int(s)), units.index(tp[int(s)][int(tr)]); Md[0, js, ju] = x['mrp']; nd[js, ju] = x['n']
        nk = T['P1'][key][1]; assert np.array_equal(nd, nk), f'DE and decoder query sets differ at {key}'
        de[key] = dict(DE=CS.level(Md[0], nk), **{f'{arm}_minus_DE': dict(delta=CS.delta(Md, T[arm][key][0], nk),
                       interval=CS.interval(SP([Md, T[arm][key][0]], nk))) for arm in ('P1', 'P2')})
    meta = {}
    for arm, R in (('P1', runs1), ('P2', runs2)):
        mins, ep, best_u = {}, {}, {}
        for sd in seeds:
            for d in R[sd].values():
                Mx = json.load(open(os.path.join(d, 'metrics.json'))); best_u[os.path.basename(d)] = Mx['summary']['shared_best_update']
                for s, r in Mx['targets'].items():
                    for key, x in r.items():
                        if key.endswith('_meta'): mins.setdefault(key[:-5], []).append(x['minutes']); ep.setdefault(key[:-5], []).append(x['best_epoch'])
        meta[arm] = dict(minutes={k: float(np.mean(x)) for k, x in mins.items()}, trials={k: v_ for k, v_ in fc['k_trials'].items()},
                         calibration_best_epoch_counts={k: {str(e_): int(sum(y == e_ for y in x)) for e_ in sorted(set(x))} for k, x in ep.items()},
                         shared_best_update_counts={str(u): int(sum(x == u for x in best_u.values())) for u in sorted(set(best_u.values()))})
    res = dict(statistic=f'Delta_{kd}^DTU = MRP_P2_DTU({kd}) - MRP_P1_DTU({kd}), query-weighted, 18 subjects x 62 pairs (cells not heard weigh 0)',
               confirmatory=confirmatory, c1_verdict=(c1v or {}).get('verdict'), seeds=seeds, alpha_in_force=alpha_used if confirmatory else None,
               alpha_reason=alpha_reason, verdict=verdict if confirmatory else f'descriptive_{verdict}', category=category,
               category_by_alpha=by_alpha, delta=point, interval=iv, interval_variant=iv_var, e1=e1, prerequisites=prereq,
               prerequisites_ok=prereq_ok, protection=protect, curves=curves, delta_by_k=deltas, unattended_report=unatt,
               de=dict(by_k=de, note='DE: CAR without bad and flat channels (none in DTU), same k trials, leave-one-trial-out penalty'), meta=meta,
               labels=dict(budget='within-subject calibration k trials (k1 4, k2 7, k4 14, kmax all; 7 validation trials separate), device RTX 5080',
                           condition='another listening condition (attended stream of two talkers)'),
               run_folders={arm: {str(sd): sorted(os.path.basename(d) for d in R[sd].values()) for sd in seeds} for arm, R in (('P1', runs1), ('P2', runs2))})
    out = path(v['out']); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(dict(verdict=res['verdict'], category=category, alpha=res['alpha_in_force'], delta=point, lower=by_alpha[str(alpha_used)]['lower'],
                          p95=iv['p95'], e1=e1['ok'], prerequisites={k: x['ok'] for k, x in prereq.items()}, protection=protect['ok'])), flush=True)


if __name__ == '__main__':
    main()
