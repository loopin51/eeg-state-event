"""C1 verdict (D-097 sections 4-9 and 13 with amendments 1-5): Delta_k2 = MRP_P2(k2) - MRP_P1(k2) on the 17 pool subjects x
20 test runs, seed-stratified pigeonhole interval (scripts/c1_stats.py), five categories, family alpha and transitions,
prerequisites 1-5, protection condition, sensitivity and the report items. This script and scripts/c1_mde.py are the only
readers of test_cells.json; the two arms' test cells are opened once, here.

Inputs: configs/c1_verdict.yaml (arms, seeds, rules), the arm configs (configs/c1_p1.yaml, configs/c1_p2.yaml), run folders
<runs_root>/<date>_c1_<arm>_<setting>_s<seed>_k<K>_f<F>[_phaserand] (exactly one finished folder per job), the MDE record
(<mde.file>, which fixes 3 or 5 seeds) and the DE baseline (results/baselines/c1_de.json).

usage: python scripts/c1_verdict.py configs/c1_verdict.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, hashlib, argparse, datetime, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import c1_stats as CS
from stageB_b1_verdict import shift_maps, wolpaw


def path(p): return p if os.path.isabs(p) else os.path.join(ROOT, p)


def arm_setup(arm_cfg):
    cc = yaml.safe_load(open(path(arm_cfg))); c = yaml.safe_load(open(path(cc['b1_config']))); sp = c['split']
    pool = sorted(s for fs in sp['subject_folds'] for s in fs); extra = sorted(s for v in sp.get('extra_targets', {}).values() for s in v)
    return cc, c, pool, extra


def find_run(v, arm, setting, seed, K, F, phaserand=False):
    pat = f"*_c1_{arm}_{setting}_s{seed}_k{K}_f{F}{'_phaserand' if phaserand else ''}"
    hits = [h for h in sorted(glob.glob(os.path.join(path(v['runs_root']), pat))) if os.path.exists(os.path.join(h, 'test_cells.json'))]
    assert len(hits) == 1, f'{pat}: {len(hits)} finished run folders'; return hits[0]


def load_arm(v, arm_cfg, seeds, phaserand=False):
    """runs[seed][(K, F)] = folder; cells[seed][subject][key][run] = cell dict (test_cells.json)"""
    cc, c, pool, extra = arm_setup(arm_cfg); sp = c['split']; runs, cells = {}, {}
    folds = [(cc['phaserand_subject_fold'], f) for f in range(5)] if phaserand else [(K, F) for K in range(5) for F in range(5)]
    for seed in seeds:
        runs[seed], cells[seed] = {}, {}
        for K, F in folds:
            d = find_run(v, cc['arm'], c['decoder_setting'], seed, K, F, phaserand); runs[seed][(K, F)] = d
            T = json.load(open(os.path.join(d, 'test_cells.json')))
            assert T['summary']['eval_split'] == 'test' and T['summary']['seed'] == seed, d
            want = {str(r) for r in sp['run_folds'][F]}
            for s, r in T['targets'].items():
                if s.startswith('_'): continue
                for key, val in r.items():
                    if isinstance(val, dict) and val and all(isinstance(x, dict) and 'mrp' in x for x in val.values()):
                        assert set(val) == want, (d, s, key, sorted(val))
                        cells[seed].setdefault(int(s), {}).setdefault(key, {}).update(val)
    return cc, c, runs, cells


def tensor(cells, seeds, subjects, units, key):
    """m [n_seeds, S, U] cell MRP and n [S, U] query counts (identical across seeds, asserted)"""
    m = np.full((len(seeds), len(subjects), len(units)), np.nan); n = np.zeros((len(subjects), len(units)))
    for i, sd in enumerate(seeds):
        for js, s in enumerate(subjects):
            for ju, u in enumerate(units):
                x = cells[sd].get(s, {}).get(key, {}).get(str(u))
                if x is None: continue
                m[i, js, ju] = x['mrp']
                assert n[js, ju] in (0, x['n']), (key, s, u, n[js, ju], x['n']); n[js, ju] = x['n']
    assert np.isfinite(m[:, n > 0]).all(), f'{key}: missing cells'
    return m, n


def topk(cells, seeds, subjects, units, key, which):
    num = den = 0.0
    for sd in seeds:
        for s in subjects:
            for u in units:
                x = cells[sd][s][key][str(u)]; num += x[which] * x['n']; den += x['n']
    return num / den


def scores(run_dirs, key, subjects):
    """mats[s][run] = within-run score matrix of `key` (e.g. main_k2) and onsets[run] from the saved scores.npz files"""
    mats, onsets = {}, {}
    for d in run_dirs:
        Z = np.load(os.path.join(d, 'scores.npz'))
        for f in Z.files:
            if not f.startswith(key + '_s') or f.endswith('_onset'): continue
            s, r = f[len(key) + 2:].split('_r'); s, r = int(s), int(r)
            if s not in subjects: continue
            mats.setdefault(s, {})[r] = Z[f].astype(np.float32); on = Z[f + '_onset']
            if r in onsets: assert np.allclose(onsets[r], on), (d, f)
            onsets[r] = on
    return mats, onsets


def maps_for(onsets, v):
    sn = v['shift_null']; rng = np.random.default_rng(sn['seed'])
    return shift_maps({r: {'onset': onsets[r]} for r in sorted(onsets)}, sn['n_shift'], sn['min_shift_s'], rng)


def judge(point, boot, alpha, sesoi, welch_p=None):
    """section 6 categories; with the amendment-5 Welch condition when welch_p is given (primary Delta_k2 only)"""
    iv = CS.interval(boot); lo = iv[f'lower_{alpha}']
    cat, note = CS.category(point, lo, iv['p95'], sesoi) if welch_p is None else CS.category_welch(point, lo, iv['p95'], welch_p, alpha, sesoi)
    return dict(category=cat, note=note, lower=lo, welch_p=welch_p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); v = yaml.safe_load(open(path(a.config)))
    fam = v['family_state']; alpha, alpha_reason, alpha_graph = CS.alpha_chain_A(fam); out = path(v['out']); os.makedirs(os.path.dirname(out), exist_ok=True)
    if alpha is None:
        json.dump(dict(verdict='not_opened', reason=alpha_reason), open(out, 'w'), indent=1); print('C1 not opened:', alpha_reason); return
    mde = json.load(open(path(v['mde']['file']))); seeds = mde['seeds_required']
    assert set(v['seeds']) <= set(seeds), 'MDE record does not cover the registered seeds'
    cc1, c, runs1, cells1 = load_arm(v, v['arms']['P1'], seeds); cc2, _, runs2, cells2 = load_arm(v, v['arms']['P2'], seeds)
    _, _, pool, extra = arm_setup(v['arms']['P1']); all19 = sorted(pool + extra); units = list(range(1, 21)); sesoi = v['sesoi']
    rng = np.random.default_rng(v['bootstrap']['seed']); nb = v['bootstrap']['n_boot']; kd = v['decision_k']
    iv_var = v['interval_variant']; assert iv_var in ('inflate3', 'inflate3_welch', 'inflate5', 'inflate5_welch'), f'amendment 5: interval variant {iv_var}'
    inflate, use_welch = True, iv_var.endswith('_welch')
    if iv_var.startswith('inflate5'): assert len(seeds) == 5, 'amendment 5: the chosen variant needs 5 seeds'
    SP = lambda arms, nn: CS.seed_pigeonhole(arms, nn, nb, rng, seed_inflate=inflate)
    T = {}                                                                       # T[arm][key] = (m, n)
    for arm, cells in (('P1', cells1), ('P2', cells2)):
        keys = sorted({k for sd in seeds for s in pool for k in cells[sd][s]})
        T[arm] = {k: tensor(cells, seeds, pool, units, k) for k in keys}
    m1, n = T['P1'][f'k{kd}']; m2, n2 = T['P2'][f'k{kd}']; assert np.array_equal(n, n2), 'arms differ in query counts'
    # --- primary (the first n_boot draws of the registered generator)
    point = CS.delta(m1, m2, n); boot = SP([m1, m2], n); iv = CS.interval(boot); W = CS.welch(m1, m2, n) if use_welch else None
    by_alpha = {str(al): judge(point, boot, al, sesoi, W['p'] if W else None) for al in (v['alpha']['chain_A'], v['alpha']['full'])}
    verdict = by_alpha[str(alpha)]['category']
    # --- protection (section 8)
    pk = v['protect']; a14, b14 = T['P1'][f'k{pk["k"]}'][0], T['P2'][f'k{pk["k"]}'][0]; n14 = T['P1'][f'k{pk["k"]}'][1]
    d14 = CS.delta(a14, b14, n14); bt14 = SP([a14, b14], n14); low14 = float(np.percentile(bt14, pk['percentile']))
    protect = dict(delta=d14, interval=CS.interval(bt14), lower=low14, ok=bool(low14 > pk['margin']),
                   label='' if low14 > pk['margin'] else 'exchange with a k14 loss')
    # --- prerequisites (section 7)
    pr = v['prerequisites']; prereq = {}
    q = pr['phaserand']; ph = {}
    for arm, cfg in (('P1', v['arms']['P1']), ('P2', v['arms']['P2'])):
        ccx, _, prun, pcells = load_arm(v, cfg, [q['seed']], phaserand=True)
        tg = sorted(c['split']['subject_folds'][q['subject_fold']]); mats, ons = scores(prun[q['seed']].values(), f'main_k{q["k"]}', tg)
        obs, null = CS.shift_null_cells(mats, maps_for(ons, v)); npr = tensor(pcells, [q['seed']], tg, units, f'k{q["k"]}')[1]
        Tobs = CS.level(obs, npr); g = np.array([CS.level(x, npr) for x in null]); thr = float(np.percentile(g, q['percentile']))
        ph[arm] = dict(subjects=tg, mrp=Tobs, threshold=thr, p=CS.perm_p(Tobs, g), ok=bool(Tobs <= thr), runs=sorted(os.path.basename(x) for x in prun[q['seed']].values()))
    prereq['1_phaserand'] = dict(arms=ph, ok=all(x['ok'] for x in ph.values()))
    q = pr['pre']; pk_ = f"pre_k{q['k']}"; p1, np1 = T['P1'][pk_]; p2, np2 = T['P2'][pk_]; assert np.array_equal(np1, np2)
    pre = {}
    for arm, mm in (('P1', p1), ('P2', p2)):
        bt = SP([mm], np1); pre[arm] = dict(mrp=CS.level(CS.seed_mean(mm), np1), interval=CS.interval(bt))
        pre[arm]['ok'] = bool(pre[arm]['interval']['p95'] < q['level_upper'])
    btd = SP([p1, p2], np1); pre['diff'] = dict(delta=CS.delta(p1, p2, np1), interval=CS.interval(btd))
    pre['diff']['ok'] = bool(pre['diff']['interval']['p95'] < q['diff_upper'])
    prereq['2_pre'] = dict(pre, ok=pre['P1']['ok'] and pre['P2']['ok'] and pre['diff']['ok'])
    q = pr['de_baseline']; de_path = path(q['file']); sha = hashlib.sha256(open(de_path, 'rb').read()).hexdigest()
    t_de = datetime.datetime.fromisoformat(q['committed_at']).timestamp()
    all_runs = [d for R in (runs1, runs2) for sd in R for d in R[sd].values()]
    first_tc = min(os.path.getmtime(os.path.join(d, 'test_cells.json')) for d in all_runs)
    prereq['3_de_first'] = dict(sha256_ok=sha == q['sha256'], de_committed_at=q['committed_at'],
                                first_test_cells_mtime=datetime.datetime.fromtimestamp(first_tc).isoformat(), ok=bool(sha == q['sha256'] and t_de < first_tc))
    q = pr['collapse']; col = {}
    for arm, R in (('P1', runs1), ('P2', runs2)):
        mats, ons = scores(R[q['seed']].values(), f'main_k{q["k"]}', pool); nn = T[arm][f'k{q["k"]}'][1]
        col[arm] = CS.collapse_check(mats, maps_for(ons, v), nn, q['p_min'], q['ks_min'], q['identity_passes'])
    prereq['4_collapse'] = dict(arms=col, ok=all(x['passed'] for x in col.values()))
    devs = {}
    for arm, R in (('P1', runs1), ('P2', runs2)):
        for sd in R:
            for d in R[sd].values(): devs[os.path.basename(d)] = json.load(open(os.path.join(d, 'metrics.json')))['summary']['device']
    for arm, cfg in (('P1', v['arms']['P1']), ('P2', v['arms']['P2'])):
        _, _, prun, _ = load_arm(v, cfg, [pr['phaserand']['seed']], phaserand=True)
        for d in prun[pr['phaserand']['seed']].values(): devs[os.path.basename(d)] = json.load(open(os.path.join(d, 'metrics.json')))['summary']['device']
    inits = sorted({json.load(open(os.path.join(d, 'metrics.json')))['summary'].get('init') for d in all_runs} - {None})
    for ck in inits:
        mp = os.path.join(os.path.dirname(path(ck)), 'metrics.json')
        devs[os.path.basename(os.path.dirname(path(ck)))] = json.load(open(mp))['summary'].get('device') if os.path.exists(mp) else 'missing'
    bad_dev = {k: x for k, x in devs.items() if pr['device'] not in str(x)}
    prereq['5_device'] = dict(n_jobs=len(devs), not_matching=bad_dev, ok=not bad_dev)
    prereq_ok = all(x['ok'] for x in prereq.values())
    final = verdict if prereq_ok else 'hold'
    # --- sensitivity (section 9)
    sens = {}
    for name, ex in v['sensitivity'].items():
        nm = n.copy()
        for s in ex['subjects']:
            for r in ex['runs']: nm[pool.index(s), units.index(r)] = 0
        bt = SP([m1, m2], nm); pt = CS.delta(m1, m2, nm)
        sens[name] = dict(excluded_cells=int((n > 0).sum() - (nm > 0).sum()), delta=pt, interval=CS.interval(bt), category=judge(pt, bt, alpha, sesoi))
    # --- reports (section 13)
    curves, deltas = {}, {}
    for k in v['ks']:
        key = f'k{k}'; a_, nk = T['P1'][key]; b_, _ = T['P2'][key]
        for arm, mm, cells in (('P1', a_, cells1), ('P2', b_, cells2)):
            bt = SP([mm], nk)
            curves.setdefault(arm, {})[key] = dict(mrp=CS.level(CS.seed_mean(mm), nk), interval=CS.interval(bt),
                                                   per_seed={str(sd): CS.level(mm[i], nk) for i, sd in enumerate(seeds)},
                                                   seed_sd=float(np.std([CS.level(mm[i], nk) for i in range(len(seeds))], ddof=1)),
                                                   top1=topk(cells, seeds, pool, units, key, 'top1'), top10=topk(cells, seeds, pool, units, key, 'top10'))
        deltas[key] = dict(delta=CS.delta(a_, b_, nk), interval=CS.interval(SP([a_, b_], nk)))
    gst = {}
    for k in v['gstar_ks']:
        key = f'k{k}_gstar'
        if key in T['P1'] and key in T['P2']:
            a_, nk = T['P1'][key]; b_, _ = T['P2'][key]
            gst[key] = dict(P1=CS.level(CS.seed_mean(a_), nk), P2=CS.level(CS.seed_mean(b_), nk), delta=CS.delta(a_, b_, nk),
                            interval=CS.interval(SP([a_, b_], nk)))
    t90 = {}
    for arm in ('P1', 'P2'):
        L = [curves[arm][f'k{k}']['mrp'] for k in v['ks']]; gain = L[-1] - L[0]; hit = [k for k, x in zip(v['ks'], L) if x - L[0] >= 0.9 * gain]
        interp = None
        for (ka, xa), (kb, xb) in zip(zip(v['ks'], L), zip(v['ks'][1:], L[1:])):
            if xa - L[0] < 0.9 * gain <= xb - L[0]: interp = ka + (L[0] + 0.9 * gain - xa) / (xb - xa) * (kb - ka); break
        t90[arm] = dict(first_k=hit[0] if hit else None, interpolated=interp if gain > 0 else None, gain=gain)
    # cells with all 19 subjects (sub-001, sub-015 evaluated only)
    r19 = {}
    for arm, cells in (('P1', cells1), ('P2', cells2)): r19[arm] = tensor(cells, seeds, all19, units, f'k{kd}')
    r19 = dict(delta=CS.delta(r19['P1'][0], r19['P2'][0], r19['P1'][1]), P1=CS.level(CS.seed_mean(r19['P1'][0]), r19['P1'][1]),
               P2=CS.level(CS.seed_mean(r19['P2'][0]), r19['P2'][1]), interval=CS.interval(SP([r19['P1'][0], r19['P2'][0]], r19['P1'][1])))
    badp = {}
    for name, ex in c['report']['bad_channel_pools'].items():
        nm = n.copy(); nm[[pool.index(s) for s in ex]] = 0
        badp[name] = dict(excluded_subjects=ex, delta=CS.delta(m1, m2, nm), interval=CS.interval(SP([m1, m2], nm)))
    DE = json.load(open(de_path)); de = {}
    for k in v['ks']:
        if k == 0: continue
        Md = np.array([[DE['subjects'][str(s)][f'k{k}']['cells'][str(u)]['mrp'] for u in units] for s in pool])[None]
        nd = np.array([[DE['subjects'][str(s)][f'k{k}']['cells'][str(u)]['n'] for u in units] for s in pool]); nk = T['P1'][f'k{k}'][1]
        assert np.array_equal(nd, nk), f'DE and decoder query sets differ at k{k}'
        de[f'k{k}'] = dict(DE=CS.level(Md[0], nk), **{f'{arm}_minus_DE': dict(delta=CS.delta(Md, T[arm][f'k{k}'][0], nk),
                           interval=CS.interval(SP([Md, T[arm][f'k{k}'][0]], nk))) for arm in ('P1', 'P2')})
    other = {}
    for arm, R in (('P1', runs1), ('P2', runs2)):                 # other adapter: own k14 layer minus the mean of the other targets' k14 layers
        num = den = 0.0
        for sd in seeds:
            for d in R[sd].values():
                Tc = json.load(open(os.path.join(d, 'test_cells.json')))['targets']
                for s, r in Tc.items():
                    if s.startswith('_') or int(s) not in pool or 'other_adapter_k14' not in r: continue
                    for u, x in r['k14'].items():
                        oth = np.mean([r['other_adapter_k14'][o][u]['mrp'] for o in r['other_adapter_k14']]); num += (x['mrp'] - oth) * x['n']; den += x['n']
        other[arm] = num / den if den else None
    dev = {}
    for arm, R in (('P1', runs1), ('P2', runs2)):
        cc_ = cc1 if arm == 'P1' else cc2; vals = []
        for sd in seeds:
            dv = json.load(open(os.path.join(R[sd][(cc_['dev_cell']['subject_fold'], cc_['dev_cell']['run_fold'])], 'metrics.json')))['targets'].get('_dev_trainers_valid', {})
            num = sum(x['mrp'] * x['n'] for r in dv.values() for x in r.values()); den = sum(x['n'] for r in dv.values() for x in r.values())
            vals.append(num / den if den else np.nan)
        dev[arm] = dict(per_seed=dict(zip(map(str, seeds), vals)), mean=float(np.mean(vals)))
    dev['D_dev'] = dev['P2']['mean'] - dev['P1']['mean']
    meta = {}
    cache = yaml.safe_load(open(path(cc1['b1_config'])))['cache_dir']; em = path(os.path.join(cache, 'eeg_meta.json'))
    lens = json.load(open(em)) if os.path.exists(em) else None
    for arm, R in (('P1', runs1), ('P2', runs2)):
        best_u, ep = {}, {}
        for sd in seeds:
            for (K, F), d in R[sd].items():
                M = json.load(open(os.path.join(d, 'metrics.json'))); best_u[os.path.basename(d)] = M['summary']['shared_best_update']
                cfg_run = yaml.safe_load(open(os.path.join(d, 'config.yaml')))['run']
                for s, r in M['targets'].items():
                    if s.startswith('_'): continue
                    for key, x in r.items():
                        if key.endswith('_meta') and 'best_epoch' in x:
                            kk = key[:-5]; ep.setdefault(kk, []).append(x['best_epoch'])
                            if lens is not None:
                                fs = lens['sample_rate']; mins = (sum(lens['run_lengths'][rr - 1] for rr in x['runs']) +
                                                                   sum(lens['run_lengths'][rr] for rr in cfg_run['valid_runs'])) / fs / 60
                                meta.setdefault(arm, {}).setdefault('minutes', {}).setdefault(kk, []).append(mins)
        meta.setdefault(arm, {})['shared_best_update_counts'] = {str(u): int(sum(x == u for x in best_u.values())) for u in sorted(set(best_u.values()))}
        meta[arm]['calibration_best_epoch_counts'] = {kk: {str(e): int(sum(x == e for x in xs)) for e in sorted(set(xs))} for kk, xs in ep.items()}
        if 'minutes' in meta[arm]: meta[arm]['minutes'] = {kk: float(np.mean(xs)) for kk, xs in meta[arm]['minutes'].items()}
    nq = float(n[n > 0].mean())
    info = {arm: dict(top1=curves[arm][f'k{kd}']['top1'], mean_candidates=nq, bits_per_selection=wolpaw(curves[arm][f'k{kd}']['top1'], int(round(nq))),
                      label='heard-speech identification information (not a communication ITR)') for arm in ('P1', 'P2')}
    res = dict(statistic=f'Delta_k{kd} = MRP_P2(k{kd}) - MRP_P1(k{kd}), query-weighted, 17 pool subjects x 20 test runs (test runs reused: opened in B1, D-090, B2)',
               seeds=seeds, alpha_in_force=alpha, alpha_reason=alpha_reason, alpha_graph_reverdict=alpha_graph, verdict=final, category=verdict,
               category_by_alpha=by_alpha, delta=point, interval=iv, interval_variant=iv_var, welch=W, prerequisites=prereq, prerequisites_ok=prereq_ok, protection=protect,
               chain_A_passes_to_L1=('not applicable: L1 closed (amendment 1), chain A keeps its alpha' if CS.l1_closed(fam)
                                     else CS.chain_A_all_rejected(verdict, fam['c2_exists']) if not fam['c2_exists'] else 'decided after C2'),
               c2_opened=bool(prereq_ok and verdict == 'adopt'), sensitivity=sens, curves=curves, delta_by_k=deltas, gstar=gst, t90=t90,
               subjects_19=r19, bad_channel_pools=badp, de=dict(by_k=de, note='DE: CAR without BIDS bad and flat channels, same k runs; near chance in D-090'),
               other_adapter_k14=other, development=dev, meta=meta, information=info, mde=mde,
               labels=dict(budget='within-subject calibration k runs (2 validation runs for early stopping separate), device RTX 5080',
                           p1_minus_p0='device difference included', reuse='test runs reused (opened in B1, D-090, B2)'),
               run_folders={arm: {str(sd): sorted(os.path.basename(d) for d in R[sd].values()) for sd in seeds} for arm, R in (('P1', runs1), ('P2', runs2))})
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(dict(verdict=final, category=verdict, alpha=alpha, delta=point, lower=by_alpha[str(alpha)]['lower'], p95=iv['p95'],
                          prerequisites={k: x['ok'] for k, x in prereq.items()}, protection=protect['ok'])), flush=True)


if __name__ == '__main__':
    main()
