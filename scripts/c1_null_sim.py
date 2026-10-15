"""Null simulation of the C1 verdict code (code check only; no EEG, no decoder output and no real test cell is read).

1. End to end: fake run folders for both arms (3 seeds x 25 jobs + 5 phase-randomised jobs, fake pre-training folders, a fake DE
   file with the same query counts) built from random score matrices, then scripts/c1_mde.py and scripts/c1_verdict.py on them.
   Scenario `info0` (information 0 in both arms) must not give 'adopt'; scenario `positive` (P2 diagonal shifted in k2 and k14)
   checks that the pipeline can reach 'adopt'.
2. Monte Carlo: cell-level replicates with the real query counts of the 17 x 20 test cells (DE baseline file), both arms with
   information 0, optional seed effect common to every cell; rates of rejection (one-sided lower bound > 0 at 0.04 / 0.05) and of
   each category.

usage: python scripts/c1_null_sim.py configs/c1_null_sim.yaml <out_dir> [--part e2e|mc|all|amendment5]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, hashlib, argparse, subprocess, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import c1_stats as CS
from c1_verdict import path, arm_setup


def cell_from(S):
    rk = (S > np.diag(S)[:, None]).sum(1); n = len(S)
    return dict(mrp=float(np.mean(1 - rk / (n - 1))), n=n, top1=float((rk < 1).mean()), top10=float((rk < 10).mean()))


def uniform_cell(rng, n):
    rk = rng.integers(0, n, n); return dict(mrp=float(np.mean(1 - rk / (n - 1))), n=int(n), top1=float((rk < 1).mean()), top10=float((rk < 10).mean()))


def build_tree(sc, effect, out, v, de, rng):
    _, c, pool, extra = arm_setup(v['arms']['P1']); sp = c['split']; runs_root = os.path.join(out, 'runs'); os.makedirs(runs_root, exist_ok=True)
    nq = {r: max(20, int(round(sc['query_scale'] * de['subjects'][str(pool[0])]['k2']['cells'][str(r)]['n']))) for r in range(1, 21)}
    onsets = {r: np.sort(rng.uniform(0.5, 180.0, nq[r])) for r in nq}
    seeds = v['seeds']; ks = [1, 2, 4, 14]
    for sd in seeds:                                                                # fake pre-training folders (P2 init)
        d = os.path.join(runs_root, f'2026-10-15_c1_P2_pretrain_s{sd}_eq'); os.makedirs(d, exist_ok=True); open(os.path.join(d, 'pretrain.pt'), 'w').close()
        json.dump(dict(summary=dict(device='NVIDIA GeForce RTX 5080')), open(os.path.join(d, 'metrics.json'), 'w'))
    jobs = [(arm, sd, K, F, False) for arm in ('P1', 'P2') for sd in seeds for K in range(5) for F in range(5)]
    jobs += [(arm, seeds[0], 0, F, True) for arm in ('P1', 'P2') for F in range(5)]
    for arm, sd, K, F, phr in jobs:
        d = os.path.join(runs_root, f"2026-10-15_c1_{arm}_A4_s{sd}_k{K}_f{F}{'_phaserand' if phr else ''}"); os.makedirs(d, exist_ok=True)
        targets = sp['subject_folds'][K] + sp.get('extra_targets', {}).get(K, []); trainers = sorted(s for j, fs in enumerate(sp['subject_folds']) if j != K for s in fs)
        test = sp['run_folds'][F]; valid = sp['valid_runs'][F]; res, arrays = {}, {}
        eff = effect if arm == 'P2' and not phr else 0.0
        for s in targets:
            o = res[str(s)] = {}
            for key in ['k0'] + [f'k{k}' for k in ks] + ['pre_k2']:
                o[key] = {}
                for r in test:
                    on = onsets[r]
                    if key in ('k0', 'k2', 'k14', 'pre_k2'):
                        rows = np.where(on >= 10.0)[0] if key == 'pre_k2' else np.arange(len(on)); S = rng.standard_normal((len(rows), len(rows)))
                        if key in ('k2', 'k14') and eff: S[np.arange(len(rows)), np.arange(len(rows))] += eff
                        o[key][str(r)] = cell_from(S); arrays[f'{"pre" if key == "pre_k2" else "main"}_{"k2" if key == "pre_k2" else key}_s{s}_r{r}'] = S.astype(np.float16)
                        arrays[f'{"pre" if key == "pre_k2" else "main"}_{"k2" if key == "pre_k2" else key}_s{s}_r{r}_onset'] = on[rows]
                    else: o[key][str(r)] = uniform_cell(rng, len(on))
            for k in ks:
                o[f'k{k}_meta'] = dict(runs=sorted(rng.choice([x for x in range(1, 21) if x not in test + valid], k, replace=False).tolist()),
                                       best_valid=5.0, init_valid=5.1, kept_init=False, best_epoch=0, epochs_run=3)
                o[f'k{k}_gstar'] = {str(r): uniform_cell(rng, nq[r]) for r in test}
            o['other_adapter_k14'] = {str(x): {str(r): uniform_cell(rng, nq[r]) for r in test} for x in targets if x != s}
        if (K, F) == (0, 0) and not phr:
            res['_dev_trainers_valid'] = {str(s): {str(r): uniform_cell(rng, nq[r]) for r in valid} for s in trainers}
        init = os.path.join(runs_root, f'2026-10-15_c1_P2_pretrain_s{sd}_eq', 'pretrain.pt') if arm == 'P2' else None
        summ = dict(arm=arm, seed=sd, eval_runs=test, eval_split='test', n_queries=0, shared_best_valid=5.0, shared_best_update=450, shared_updates=1200,
                    car_exclude_bad=True, device='NVIDIA GeForce RTX 5080', init=init, seconds=1.0)
        meta_only = {s: {kk: x for kk, x in r.items() if kk.endswith('_meta')} for s, r in res.items() if not s.startswith('_')}
        meta_only.update({s: r for s, r in res.items() if s.startswith('_')})
        json.dump(dict(summary=summ, targets=res), open(os.path.join(d, 'test_cells.json'), 'w'))
        json.dump(dict(summary=summ, targets=meta_only), open(os.path.join(d, 'metrics.json'), 'w'))
        json.dump([dict(stage='shared', update=u, train=5.0, valid=5.0 - 0.01 * (u == 450)) for u in range(150, 1350, 150)], open(os.path.join(d, 'history.json'), 'w'))
        yaml.safe_dump(dict(run=dict(arm=arm, seed=sd, targets=targets, trainers=trainers, valid_runs=[r - 1 for r in valid])), open(os.path.join(d, 'config.yaml'), 'w'))
        np.savez_compressed(os.path.join(d, 'scores.npz'), **arrays)
    fde = {'setting': 'A4', 'car_exclude_bad': True, 'ks': ks, 'eval_split': 'test',
           'subjects': {str(s): {f'k{k}': dict(cells={str(r): uniform_cell(rng, nq[r]) for r in range(1, 21)}) for k in ks} for s in sorted(pool + extra)}}
    fp = os.path.join(out, 'c1_de_fake.json'); json.dump(fde, open(fp, 'w'))
    vv = json.loads(json.dumps(v)); vv['runs_root'] = runs_root; vv['out'] = os.path.join(out, 'verdict.json'); vv['mde']['file'] = os.path.join(out, 'mde.json')
    vv['prerequisites']['de_baseline'].update(file=fp, sha256=hashlib.sha256(open(fp, 'rb').read()).hexdigest(), committed_at='2000-01-01T00:00:00+00:00')
    vp = os.path.join(out, 'c1_verdict_sim.yaml'); yaml.safe_dump(vv, open(vp, 'w')); return vp


def e2e(cfg, out, v, de):
    sc = cfg['end_to_end']; res = {}
    for name, x in sc['scenarios'].items():
        rng = np.random.default_rng([sc['seed'], list(sc['scenarios']).index(name)]); o = os.path.join(out, name); t0 = time.time()
        vp = build_tree(sc, x['effect_P2'], o, v, de, rng)
        for scr in ('c1_mde.py', 'c1_verdict.py'):
            r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', scr), vp], capture_output=True, text=True)
            print(f'[{name}] {scr}: {r.stdout.strip()[-1500:]}', flush=True)
            if r.returncode: print(r.stderr[-3000:], flush=True); raise SystemExit(f'{scr} failed in scenario {name}')
        V = json.load(open(os.path.join(o, 'verdict.json')))
        res[name] = dict(verdict=V['verdict'], category=V['category'], delta=V['delta'], lower_0_04=V['category_by_alpha']['0.04']['lower'],
                         alpha_in_force=V['alpha_in_force'], alpha_reason=V['alpha_reason'], chain_A_passes_to_L1=V['chain_A_passes_to_L1'],
                         category_by_alpha={k: x['category'] for k, x in V['category_by_alpha'].items()},
                         p95=V['interval']['p95'], prerequisites={k: x['ok'] for k, x in V['prerequisites'].items()},
                         collapse_identity={a: V['prerequisites']['4_collapse']['arms'][a]['identity'] for a in ('P1', 'P2')},
                         phaserand={a: (V['prerequisites']['1_phaserand']['arms'][a]['mrp'], V['prerequisites']['1_phaserand']['arms'][a]['threshold']) for a in ('P1', 'P2')},
                         protection=V['protection']['ok'], seconds=time.time() - t0)
    return res


def mc(cfg, v, de, block='monte_carlo'):
    q = cfg[block]; _, _, pool, _ = arm_setup(v['arms']['P1']); units = range(1, 21)
    n = np.array([[de['subjects'][str(s)]['k2']['cells'][str(u)]['n'] for u in units] for s in pool], float)
    rank_sd = np.sqrt(1 / 12 / n); out = {}; alpha = v['alpha']['chain_A']
    conds = [dict(seed_sd=t, effect=0.0) for t in q['seed_sd']] + q.get('power', [])
    for var in q.get('variants', [dict(name='registered', n_seeds=len(v['seeds']), seed_inflate=False)]):
      k = var['n_seeds']; use_w = var.get('welch', False)
      for cd in conds:
        tau, eff = cd['seed_sd'], cd['effect']
        rng = np.random.default_rng([q['seed'], int(round(tau * 1e4)), int(round(eff * 1e4))]); cats = []; rej = []; lo5 = []
        for _ in range(q['n_rep']):
            m = [0.5 + rng.normal(0, tau, (k, 1, 1)) + rng.standard_normal((k,) + n.shape) * np.sqrt(rank_sd ** 2 + q['cell_sd_extra'] ** 2) for _ in range(2)]
            m[1] = m[1] + eff
            pt = CS.delta(m[0], m[1], n); bt = CS.seed_pigeonhole(m, n, q['n_boot'], rng, seed_inflate=var['seed_inflate']); iv = CS.interval(bt)
            wp = CS.welch(m[0], m[1], n)['p'] if use_w else 0.0
            cat = CS.category_welch(pt, iv[f'lower_{alpha}'], iv['p95'], wp, alpha, v['sesoi'])[0]
            cats.append(cat); rej.append(cat in CS.REJECTED); lo5.append(iv['lower_0.05'])
        key = f"{var['name']}_seedsd{tau}" + (f'_effect{eff}' if eff else '')
        out[key] = r_ = dict(n_rep=q['n_rep'], n_seeds=k, seed_inflate=var['seed_inflate'], welch=use_w, seed_sd=tau, effect=eff,
                             reject=float(np.mean(rej)), adopt=float(np.mean([x == 'adopt' for x in cats])),
                             categories={c_: float(np.mean([x == c_ for x in cats])) for c_ in CS.CATEGORIES})
        print(f"{key}: {json.dumps(r_)}", flush=True)
    return out


def select(res, q):
    """amendment 5 selection rule (order of q['variants'] = cost order: 3 seeds before 5, without ④ before with ④)"""
    rows = []
    for var in q['variants']:
        nul = [res[f"{var['name']}_seedsd{t}"]['reject'] for t in q['seed_sd']]
        pw = [res[f"{var['name']}_seedsd{p['seed_sd']}_effect{p['effect']}"]['adopt'] for p in q['power']]
        rows.append(dict(name=var['name'], max_null_reject=max(nul), null_reject=nul, power=min(pw), ok_null=max(nul) <= q['max_reject'], ok_power=min(pw) >= q['min_power']))
    both = [r for r in rows if r['ok_null'] and r['ok_power']]
    if both: pick, why = both[0], 'cheapest candidate meeting both conditions'
    elif not any(r['ok_null'] for r in rows): pick, why = min(rows, key=lambda r: r['max_null_reject']), 'no candidate meets the rejection condition: lowest rejection'
    else: pick, why = max((r for r in rows if r['ok_null']), key=lambda r: r['power']), 'no candidate meets the power condition: highest power among those meeting the rejection condition'
    return dict(table=rows, chosen=pick['name'], reason=why)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('out_dir'); ap.add_argument('--part', default='all')
    a = ap.parse_args(); cfg = yaml.safe_load(open(path(a.config))); v = yaml.safe_load(open(path(cfg['verdict_config']))); de = json.load(open(path(cfg['de_file'])))
    os.makedirs(a.out_dir, exist_ok=True); res = {}
    if a.part in ('e2e', 'all'): res['end_to_end'] = e2e(cfg, a.out_dir, v, de); print(json.dumps(res['end_to_end'], indent=1), flush=True)
    if a.part in ('mc', 'all'): res['monte_carlo'] = mc(cfg, v, de)
    if a.part == 'amendment5':
        res['amendment5'] = mc(cfg, v, de, 'amendment5'); res['selection'] = select(res['amendment5'], cfg['amendment5']); print(json.dumps(res['selection'], indent=1), flush=True)
    json.dump(res, open(os.path.join(a.out_dir, f'null_sim_{a.part}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
