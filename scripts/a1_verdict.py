"""F-III A1 verdict (D-105 sections 4-8, amendments 1-5): Delta_J = MRP_N+LS(J250, k2) - MRP_P1+LS(J250, k2) on the 17 pool
subjects x 20 test runs with the frozen LS rule (configs/a1.yaml lag_search.rule), the D-097 amendment 5 (3) interval
(scripts/c1_stats.py), the five categories at both chain-A levels (the level in force from configs/a1.yaml family_state),
prerequisites 1-7, the protection condition and the report items. This script and scripts/a1_mde.py are the only readers of
the A1 test_cells.json and scores.npz; the two arms' test cells are opened once, here.

Not opened (verdict.json says why, nothing is read): the development gate did not pass (D-105 amendment 2: no automatic
P1 arm), or configs/a1_n.yaml candidate differs from the gate's chosen candidate.
Shift null with the 30 s exclusion (prerequisites 1 and 5, D-100): the truth of query i becomes its shift partner j (B1
amendment 1 onset-order shift, one shared map per run and draw); candidates are the queries >= 30 s from both i and j;
pairs (i, draw) with j within 30 s of i are left out.

usage: python scripts/a1_verdict.py configs/a1_verdict.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, hashlib, argparse, datetime, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import c1_stats as CS
from stageB_b1_verdict import shift_maps


def path(p): return p if os.path.isabs(p) else os.path.join(ROOT, p)
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def arm_setup(arm_cfg):
    cc = yaml.safe_load(open(path(arm_cfg))); c = yaml.safe_load(open(path(cc['b1_config']))); sp = c['split']
    pool = sorted(s for fs in sp['subject_folds'] for s in fs); extra = sorted(s for v in sp.get('extra_targets', {}).values() for s in v)
    return cc, c, pool, extra


def find_run(v, cc, c, seed, K, F, phaserand=False):
    cand = cc.get('candidate') if cc['arm'] == 'N' else None
    pat = f"*_a1_{cc['arm']}{'-' + cand if cand else ''}_{c['decoder_setting']}_s{seed}_k{K}_f{F}{'_phaserand' if phaserand else ''}"
    hits = [h for h in sorted(glob.glob(os.path.join(path(v['runs_root']), pat))) if os.path.exists(os.path.join(h, 'test_cells.json'))]
    assert len(hits) == 1, f'{pat}: {len(hits)} finished run folders'; return hits[0]


def load_arm(v, arm_cfg, seeds, phaserand=False):
    """runs[seed][(K, F)] = folder; cells[seed][subject][block][run] = cell dict (test_cells.json; block 'a1_k2', 'k2', ...)"""
    cc, c, pool, extra = arm_setup(arm_cfg); sp = c['split']; runs, cells = {}, {}
    folds = [(cc['phaserand_subject_fold'], f) for f in range(5)] if phaserand else [(K, F) for K in range(5) for F in range(5)]
    for seed in seeds:
        runs[seed], cells[seed] = {}, {}
        for K, F in folds:
            d = find_run(v, cc, c, seed, K, F, phaserand); runs[seed][(K, F)] = d
            T = json.load(open(os.path.join(d, 'test_cells.json'))); S = T['summary']
            assert S['eval_split'] == 'test' and S['seed'] == seed and S['arm'] == cc['arm'], d
            assert S['ls_rule'] == v['_rule'], (d, S['ls_rule'], v['_rule'])
            if cc['arm'] == 'N': assert S['candidate'] == cc['candidate'], d
            want = {str(r) for r in sp['run_folds'][F]}
            for s, r in T['targets'].items():
                if s.startswith('_'): continue
                for key, val in r.items():
                    if isinstance(val, dict) and val and all(isinstance(x, dict) and 'n' in x for x in val.values()):
                        assert set(val) <= want, (d, s, key, sorted(val))
                        cells[seed].setdefault(int(s), {}).setdefault(key, {}).update(val)
    return cc, c, runs, cells


def tensor(cells, seeds, subjects, units, block, field):
    """m [n_seeds, S, U] cell values of block/field and n [S, U] query counts (identical across seeds, asserted)"""
    m = np.full((len(seeds), len(subjects), len(units)), np.nan); n = np.zeros((len(subjects), len(units)))
    for i, sd in enumerate(seeds):
        for js, s in enumerate(subjects):
            for ju, u in enumerate(units):
                x = cells[sd].get(s, {}).get(block, {}).get(str(u))
                if x is None: continue
                m[i, js, ju] = x[field]
                assert n[js, ju] in (0, x['n']), (block, s, u, n[js, ju], x['n']); n[js, ju] = x['n']
    assert np.isfinite(m[:, n > 0]).all(), f'{block}/{field}: missing cells'
    return m, n


def scores(run_dirs, prefix, subjects):
    """mats[s][run] = within-run score matrix (rows queries, columns candidates = the same queries) and onsets[run] from
    scores.npz keys <prefix>_s<s>_r<run>_j250 / _onset (prefix 'k2' or 'pre_k2')"""
    mats, onsets = {}, {}
    for d in run_dirs:
        Z = np.load(os.path.join(d, 'scores.npz'))
        for f in Z.files:
            if not (f.startswith(prefix + '_s') and f.endswith('_j250')): continue
            s, r = f[len(prefix) + 2:-len('_j250')].split('_r'); s, r = int(s), int(r)
            if s not in subjects: continue
            mats.setdefault(s, {})[r] = Z[f].astype(np.float64); on = Z[f[:-len('_j250')] + '_onset'].astype(np.float64)
            if r in onsets: assert np.allclose(onsets[r], on), (d, f)
            onsets[r] = on
    return mats, onsets


def maps_for(onsets, v):
    sn = v['shift_null']; return shift_maps({r: {'onset': onsets[r]} for r in sorted(onsets)}, sn['n_shift'], sn['min_shift_s'], np.random.default_rng(sn['seed']))


def excl_null(S, t, maps, excl, chunk=25):
    """observed 30-s-excluded MRP of the diagonal truth and the null mean per draw (truth -> partner j, candidates >= excl
    from both i and j, pairs with j within excl of i left out); numpy version of scripts/d100_tau50.excl_stats"""
    n = S.shape[0]; t = np.asarray(t, np.float64); far = np.abs(t[:, None] - t[None]) >= excl; ar = np.arange(n)
    above = (S > np.diag(S)[:, None]) & far; obs = float((1 - above.sum(1) / np.maximum(far.sum(1), 1)).mean())
    out = []
    for b0 in range(0, len(maps), chunk):
        J = maps[b0:b0 + chunk]; v = S[ar[None], J]; cand = far[None] & far[J]
        rk = ((S[None] > v[..., None]) & cand).sum(-1); cnt = cand.sum(-1); ok = far[ar[None], J] & (cnt > 0)
        mrp = np.where(ok, 1 - rk / np.maximum(cnt, 1), 0.0); out.append(mrp.sum(1) / np.maximum(ok.sum(1), 1))
    return obs, np.concatenate(out)


def null_cells(mats, onsets, maps, excl, collapse=False):
    """obs [S, U] and null [n_shift, S, U] over the subjects of mats and the runs of maps (nan where no cell)"""
    subs = sorted(mats); units = sorted(maps); ns = next(iter(maps.values())).shape[0]
    obs = np.full((len(subs), len(units)), np.nan); null = np.full((ns, len(subs), len(units)), np.nan)
    for js, s in enumerate(subs):
        for ju, u in enumerate(units):
            if u not in mats[s]: continue
            S = mats[s][u]
            if collapse: S = np.broadcast_to(S.mean(0, keepdims=True), S.shape).copy()      # column means only (information 0)
            obs[js, ju], null[:, js, ju] = excl_null(S, onsets[u], maps[u], excl)
    return subs, units, obs, null


def judge(point, boot, alpha, sesoi):
    iv = CS.interval(boot); lo = iv[f'lower_{alpha}']; cat, note = CS.category(point, lo, iv['p95'], sesoi)
    return dict(category=cat, note=note, lower=lo)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); v = yaml.safe_load(open(path(a.config)))
    A = yaml.safe_load(open(path(v['a1_config']))); fam = A['family_state']; rule = A['lag_search']['rule']; v['_rule'] = rule
    out = path(v['out']); os.makedirs(os.path.dirname(out), exist_ok=True)
    assert rule in A['lag_search']['rules'], 'LS rule not frozen'
    dev = json.load(open(path(v['development']['file']))); g = dev.get('gate')
    ccN = yaml.safe_load(open(path(v['arms']['N'])))
    if g is None or not g['passed']:
        reason = 'development gate not run' if g is None else 'A1 development-stage futility (D-105 section 9, amendment 2): A1 test not opened'
        json.dump(dict(verdict='not_opened', reason=reason), open(out, 'w'), indent=1); print('A1 not opened:', reason); return
    assert ccN['candidate'] == g['chosen'], f"configs/a1_n.yaml candidate {ccN['candidate']} != development gate {g['chosen']}"
    alpha = fam['alpha_chain_A']; alphas = sorted({v['alpha']['chain_A'], v['alpha']['after_T1_fail']}); assert alpha in alphas
    mde = json.load(open(path(v['mde']['file']))); seeds = v['seeds']; assert mde['seeds'] == seeds, 'MDE record seeds differ'
    cc1, c, runs1, cells1 = load_arm(v, v['arms']['P1'], seeds); ccn, _, runsN, cellsN = load_arm(v, v['arms']['N'], seeds)
    _, _, pool, extra = arm_setup(v['arms']['P1']); all19 = sorted(pool + extra); units = list(range(1, 21)); sesoi = v['sesoi']
    rng = np.random.default_rng(v['bootstrap']['seed']); nb = v['bootstrap']['n_boot']; kd = v['decision_k']; excl = A['queries']['exclusion_s']
    SP = lambda arms, nn: CS.seed_pigeonhole(arms, nn, nb, rng, seed_inflate=True)
    J = f'j250_{rule}'; T = {'P1': cells1, 'N': cellsN}
    ten = lambda arm, block, field, subs=pool: tensor(T[arm], seeds, subs, units, block, field)
    m1, n = ten('P1', f'a1_k{kd}', J); mN, nN = ten('N', f'a1_k{kd}', J); assert np.array_equal(n, nN), 'arms differ in query counts'
    # --- primary (the first n_boot draws of the registered generator)
    point = CS.delta(m1, mN, n); boot = SP([m1, mN], n); iv = CS.interval(boot)
    by_alpha = {str(al): judge(point, boot, al, sesoi) for al in alphas}; category = by_alpha[str(alpha)]['category']
    # --- protection (section 7)
    pk = v['protect']; a0, _ = ten('P1', f"a1_k{pk['k']}", pk['key']); b0, n0 = ten('N', f"a1_k{pk['k']}", pk['key'])
    bt0 = SP([a0, b0], n0); low0 = float(np.percentile(bt0, pk['percentile']))
    protect = dict(delta=CS.delta(a0, b0, n0), interval=CS.interval(bt0), lower=low0, ok=bool(low0 > pk['margin']),
                   label='' if low0 > pk['margin'] else 'exchange with fine-alignment performance (no new-model contribution sentence)')
    # --- prerequisites (section 6)
    pr = v['prerequisites']; prereq = {}
    q = pr['phaserand']; ph = {}; prdirs = {}
    for arm, cfg in (('P1', v['arms']['P1']), ('N', v['arms']['N'])):
        ccx, cx, prun, pcells = load_arm(v, cfg, [q['seed']], phaserand=True); prdirs[arm] = prun[q['seed']]
        tg = sorted(cx['split']['subject_folds'][q['subject_fold']]); mats, ons = scores(prun[q['seed']].values(), f"k{q['k']}", tg)
        subs, uu, obs, null = null_cells(mats, ons, maps_for(ons, v), excl)
        npr = tensor(pcells, [q['seed']], subs, uu, f"a1_k{q['k']}", J)[1]
        Tobs = CS.level(obs, npr); gn = np.array([CS.level(x, npr) for x in null]); thr = float(np.percentile(gn, q['percentile']))
        ph[arm] = dict(subjects=subs, mrp=Tobs, threshold=thr, p=CS.perm_p(Tobs, gn), ok=bool(Tobs <= thr), runs=sorted(os.path.basename(x) for x in prun[q['seed']].values()))
    prereq['1_phaserand'] = dict(arms=ph, ok=all(x['ok'] for x in ph.values()))
    q = pr['pre']; p1, np1 = ten('P1', f"a1_pre_k{q['k']}", J); p2, np2 = ten('N', f"a1_pre_k{q['k']}", J); assert np.array_equal(np1, np2)
    pre = {}
    for arm, mm in (('P1', p1), ('N', p2)):
        pre[arm] = dict(mrp=CS.level(CS.seed_mean(mm), np1), interval=CS.interval(SP([mm], np1))); pre[arm]['ok'] = bool(pre[arm]['interval']['p95'] < q['level_upper'])
    pre['diff'] = dict(delta=CS.delta(p1, p2, np1), interval=CS.interval(SP([p1, p2], np1))); pre['diff']['ok'] = bool(pre['diff']['interval']['p95'] < q['diff_upper'])
    prereq['2_pre'] = dict(pre, ok=pre['P1']['ok'] and pre['N']['ok'] and pre['diff']['ok'])
    q = pr['de_baseline']; de_path = path(q['file']); de_sha = sha(de_path); t_de = datetime.datetime.fromisoformat(q['committed_at']).timestamp()
    all_runs = [d for R in (runs1, runsN) for sd in R for d in R[sd].values()] + [d for R in prdirs.values() for d in R.values()]
    first_tc = min(os.path.getmtime(os.path.join(d, 'test_cells.json')) for d in all_runs)
    prereq['3_de_first'] = dict(sha256_ok=de_sha == q['sha256'], de_commit=q['commit'], de_committed_at=q['committed_at'],
                                first_test_cells_mtime=datetime.datetime.fromtimestamp(first_tc).isoformat(), ok=bool(de_sha == q['sha256'] and t_de < first_tc))
    DE = json.load(open(de_path)); assert DE['rules'] and rule in DE['rules'] and DE['eval_split'] == 'test'

    def de_tensor(k, field, subs=pool):
        Md = np.array([[DE['subjects'][str(s)][f'k{k}']['cells'][str(u)][field] for u in units] for s in subs])[None]
        nd = np.array([[DE['subjects'][str(s)][f'k{k}']['cells'][str(u)]['n'] for u in units] for s in subs], float); return Md, nd
    q = pr['n_gt_de']; Md, nd = de_tensor(q['k'], J); assert np.array_equal(nd, n), 'DE and decoder query sets differ'
    btd = SP([Md, mN], n); prereq['4_n_gt_de'] = dict(DE=CS.level(Md[0], n), delta=CS.delta(Md, mN, n), interval=CS.interval(btd),
                                                     lower=float(np.percentile(btd, q['percentile'])), ok=bool(np.percentile(btd, q['percentile']) > 0))
    q = pr['collapse']; col = {}
    for arm, R in (('P1', runs1), ('N', runsN)):
        mats, ons = scores(R[q['seed']].values(), f"k{q['k']}", pool); subs, uu, obs, null = null_cells(mats, ons, maps_for(ons, v), excl, collapse=True)
        nn = tensor(T[arm], seeds, subs, uu, f"a1_k{q['k']}", J)[1]
        Tc = CS.level(obs, nn); gc = np.array([CS.level(x, nn) for x in null]); p = CS.perm_p(Tc, gc)
        ps = [CS.perm_p(CS.level(obs[js:js + 1], nn[js:js + 1]), [CS.level(x[js:js + 1], nn[js:js + 1]) for x in null]) for js in range(len(subs))]
        from scipy.stats import kstest
        ksp = float(kstest(ps, 'uniform').pvalue); dmax = float(np.nanmax(np.abs(null - obs[None]))); ident = dmax < q['identity_tol']
        col[arm] = dict(stat=Tc, null_mean=float(gc.mean()), p=p, subject_p=ps, ks_p=ksp, identity=bool(ident), max_abs_null_minus_obs=dmax,
                        passed=bool(ident or (p >= q['p_min'] and ksp >= q['ks_min'])))
    prereq['5_collapse'] = dict(arms=col, ok=all(x['passed'] for x in col.values()))
    devs = {os.path.basename(d): json.load(open(os.path.join(d, 'metrics.json')))['summary']['device'] for d in all_runs}
    bad_dev = {k: x for k, x in devs.items() if pr['device'] not in str(x)}
    prereq['6_device'] = dict(n_jobs=len(devs), not_matching=bad_dev, ok=not bad_dev)
    prereq['7_mde'] = dict(MDE=mde['base']['MDE'], threshold=mde['threshold'], committed_before_N=mde.get('note', ''), ok=bool(mde['base']['MDE'] <= mde['threshold']))
    prereq_ok = all(x['ok'] for x in prereq.values())
    final = category if prereq_ok else ('hold (MDE)' if not prereq['7_mde']['ok'] else 'hold')
    # --- reports (section 8; no alpha)
    curves, deltas = {}, {}
    for k in v['ks']:
        for fld in (J, 'j250_fixed', 'd0_fixed'):
            a_, nk = ten('P1', f'a1_k{k}', fld); b_, _ = ten('N', f'a1_k{k}', fld); key = f'k{k}_{fld}'
            for arm, mm in (('P1', a_), ('N', b_)):
                per = [CS.level(mm[i], nk) for i in range(len(seeds))]
                curves.setdefault(arm, {})[key] = dict(mrp=CS.level(CS.seed_mean(mm), nk), interval=CS.interval(SP([mm], nk)),
                                                      per_seed=dict(zip(map(str, seeds), per)), seed_sd=float(np.std(per, ddof=1)))
            deltas[key] = dict(delta=CS.delta(a_, b_, nk), interval=CS.interval(SP([a_, b_], nk)))
    rules_all = {}
    for rl in A['lag_search']['rules']:
        a_, nk = ten('P1', f'a1_k{kd}', f'j250_{rl}'); b_, _ = ten('N', f'a1_k{kd}', f'j250_{rl}')
        rules_all[rl] = dict(delta=CS.delta(a_, b_, nk), interval=CS.interval(SP([a_, b_], nk)), frozen=rl == rule)
    c1def = {}
    for k in v['ks']:
        a_, nk = ten('P1', f'k{k}', 'mrp'); b_, _ = ten('N', f'k{k}', 'mrp')
        c1def[f'k{k}'] = dict(P1=CS.level(CS.seed_mean(a_), nk), N=CS.level(CS.seed_mean(b_), nk), delta=CS.delta(a_, b_, nk),
                              **{f'{arm}_{w}': CS.level(CS.seed_mean(ten(arm, f'k{k}', w)[0]), nk) for arm in T for w in ('top1', 'top10')})
    r19 = {arm: ten(arm, f'a1_k{kd}', J, all19) for arm in T}
    r19 = dict(P1=CS.level(CS.seed_mean(r19['P1'][0]), r19['P1'][1]), N=CS.level(CS.seed_mean(r19['N'][0]), r19['N'][1]),
               delta=CS.delta(r19['P1'][0], r19['N'][0], r19['P1'][1]), interval=CS.interval(SP([r19['P1'][0], r19['N'][0]], r19['P1'][1])))
    de = {}
    for k in v['ks']:
        if k == 0: continue
        for fld in (J, 'j250_fixed', 'd0_fixed'):
            Mk, nk_ = de_tensor(k, fld); nk = ten('P1', f'a1_k{k}', fld)[1]; assert np.array_equal(nk_, nk), f'DE query set differs at k{k}'
            de[f'k{k}_{fld}'] = dict(DE=CS.level(Mk[0], nk), **{f'{arm}_minus_DE': dict(delta=CS.delta(Mk, ten(arm, f'a1_k{k}', fld)[0], nk),
                                     interval=CS.interval(SP([Mk, ten(arm, f'a1_k{k}', fld)[0]], nk))) for arm in T})
    meta = {}
    for arm, R in (('P1', runs1), ('N', runsN)):
        best_u, ep, mask = [], {}, []
        for sd in seeds:
            for d in R[sd].values():
                M = json.load(open(os.path.join(d, 'metrics.json'))); best_u.append(M['summary']['shared_best_update'])
                if M['summary'].get('lse_masked_fraction_shared') is not None: mask.append(M['summary']['lse_masked_fraction_shared'])
                for s, r in M['targets'].items():
                    if s.startswith('_'): continue
                    for key, x in r.items():
                        if key.endswith('_meta') and 'best_epoch' in x: ep.setdefault(key[:-5], []).append(x['best_epoch'])
        meta[arm] = dict(shared_best_update_counts={str(u): best_u.count(u) for u in sorted(set(best_u))},
                         calibration_best_epoch_counts={kk: {str(e): xs.count(e) for e in sorted(set(xs))} for kk, xs in ep.items()},
                         lse_masked_fraction_shared_max=max(mask) if mask else None)
    res = dict(statistic=f'Delta_J = MRP_N+LS(J250, k{kd}) - MRP_P1+LS(J250, k{kd}), LS rule {rule}, query-weighted, 17 pool subjects x 20 test runs '
                         '(test runs reused: opened in B1, D-090, B2, D-100, D-101)',
               candidate=ccN['candidate'], seeds=seeds, alpha_in_force=alpha, family_state=fam, verdict=final, category=category, category_by_alpha=by_alpha,
               delta=point, interval=iv, interval_variant='inflate5 (D-097 amendment 5 (3))', prerequisites=prereq, prerequisites_ok=prereq_ok, protection=protect,
               reports=dict(curves=curves, delta_by_k=deltas, ls_rules=rules_all, c1_definition=c1def, subjects_19=r19, de=de, meta=meta,
                            note='section 8: tau50, J500, stretch and band retention, G*, T90, P1 - P0 and other-adapter are inferred after the verdict'),
               development=dict(file=v['development']['file'], sha256=sha(path(v['development']['file'])), chosen=g['chosen'], gate=g['gate']),
               mde=mde, de_baseline=dict(file=v['prerequisites']['de_baseline']['file'], sha256=de_sha),
               labels=dict(budget='within-subject calibration k2 (2 validation runs for early stopping separate), timing error J250, device RTX 5080',
                           reuse='test runs reused (opened in B1, D-090, B2, D-100, D-101)', motivation='tau50 motivated this question'),
               run_folders={arm: {str(sd): sorted(os.path.basename(d) for d in R[sd].values()) for sd in seeds} for arm, R in (('P1', runs1), ('N', runsN))})
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(dict(verdict=final, category=category, alpha=alpha, delta=point, lower=by_alpha[str(alpha)]['lower'], p95=iv['p95'],
                          prerequisites={k: x['ok'] for k, x in prereq.items()}, protection=protect['ok'])), flush=True)


if __name__ == '__main__':
    main()
