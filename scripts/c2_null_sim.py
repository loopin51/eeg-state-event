"""Null simulation of the C2 verdict code (code check only; no EEG, no decoder output and no real test cell is read).

1. End to end: fake run folders of both arms on the real DTU split (5 seeds x 25 jobs + 5 phase-randomised jobs per arm, fake
   pre-training folders, fake DE, UNATTENDED and C1 verdict files with the same cells and query counts), built from iid normal
   score matrices with a diagonal shift per arm and k (PRE and the unattended report stay information 0), then scripts/c2_verdict.py on them. `info0` must give 'hold_E1' and never
   'adopt'; `equal` (same information in both arms) passes E1 and must not give 'adopt'; `positive` checks that the pipeline can
   reach 'adopt'. Only the score matrices that c2_verdict.py reads are written (P1 kmax for E1, seed 2036 k2 for the collapse
   check, phase-randomised k2).
2. Monte Carlo: cell-level replicates with the real C2 query counts (18 subjects x 62 pairs), both arms information 0, optional
   seed effect common to every cell; rates of rejection (one-sided lower bound > 0) and of each category at the alpha in force.

usage: python scripts/c2_null_sim.py configs/c2_null_sim.yaml <out_dir> [--part e2e|mc|all]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, hashlib, argparse, subprocess, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import c1_stats as CS
from c2_dtu_split import DTUSplit, trials_from_cache, path
from c2_verdict import arm_cfg

KS = ['k0', 'k1', 'k2', 'k4', 'kmax']


def cell_from(S):
    rk = (S > np.diag(S)[:, None]).sum(1); n = len(S)
    return dict(mrp=float(np.mean(1 - rk / (n - 1))), n=n, top1=float((rk < 1).mean()), top10=float((rk < 10).mean()))


def grids(v):
    """query times of every (subject, trial) as DTUData.grid (query_step_s) and the PRE subset"""
    cc, c, fc = arm_cfg(v['arms']['P1']); meta = json.load(open(os.path.join(path(fc['cache_dir']), 'eeg_meta.json'))); fs = meta['sample_rate']
    w, e = c['window'], c['eeg']; split = DTUSplit(fc, trials_from_cache(fc['cache_dir'])); q, pre = {}, {}
    for s, R in split.trials.items():
        for r in R:
            hi = r['length'] / fs - w['tmax'] - max(e['offset_s'], 0); t = np.arange(-w['tmin'], hi + 1e-9, fc['query_step_s'])
            q[(s, r['trial'])] = t; pre[(s, r['trial'])] = t[t + w['tmin'] + e['offset_s'] - c['eval']['pre_shift_s'] >= 0]
    return cc, c, fc, split, q, pre


def build_tree(sc, out, v, rng):
    cc, c, fc, split, q, pre = grids(v); runs_root = os.path.join(out, 'runs'); os.makedirs(runs_root, exist_ok=True); seeds = v['seeds']
    for sd in seeds:                                                                # fake C1 pre-training folders (P2_DTU init)
        d = os.path.join(runs_root, f'2026-10-16_c1_P2_pretrain_s{sd}_eq'); os.makedirs(d, exist_ok=True); open(os.path.join(d, 'pretrain.pt'), 'w').close()
        json.dump(dict(seed=sd, candidate='eq', device='NVIDIA GeForce RTX 5080', seconds=1.0), open(os.path.join(d, 'metrics.json'), 'w'))   # c1_pretrain.py format
    jobs = [(arm, sd, K, F, False) for arm in ('P1', 'P2') for sd in seeds for K in range(5) for F in range(5)]
    jobs += [(arm, seeds[0], cc['phaserand_subject_fold'], F, True) for arm in ('P1', 'P2') for F in range(5)]
    for arm, sd, K, F, phr in jobs:
        name = 'P1_DTU' if arm == 'P1' else 'P2_DTU'; eff = {} if phr else sc[arm]
        d = os.path.join(runs_root, f"2026-10-16_c2_{name}_{c['decoder_setting']}_s{sd}_k{K}_f{F}{'_phaserand' if phr else ''}"); os.makedirs(d, exist_ok=True)
        res, arrays = {}, {}
        for s in split.subject_folds[K]:
            o = res[str(s)] = {}; test = split.sets(s, F)['test']; o['n_trials'] = len(test)
            for key in KS + ['pre_k2', 'unattended_k2']:
                kk = 'k2' if key in ('pre_k2', 'unattended_k2') else key; sh = eff.get(kk, eff.get('all', 0.0)) if key not in ('pre_k2', 'unattended_k2') else 0.0   # earlier EEG and unattended candidates: information 0
                o[key] = {}
                for r in test:
                    t = (pre if key == 'pre_k2' else q)[(s, r['trial'])]; S = rng.standard_normal((len(t), len(t)))
                    S[np.arange(len(t)), np.arange(len(t))] += sh; o[key][str(r['trial'])] = cell_from(S)
                    keep = (key == 'kmax' and arm == 'P1' and not phr) or (key == 'k2' and sd == seeds[0])
                    if keep: arrays[f'main_{key}_s{s}_t{r["trial"]}'] = S.astype(np.float16); arrays[f'main_{key}_s{s}_t{r["trial"]}_onset'] = t
            for k in KS[1:]:
                o[f'{k}_meta'] = dict(trials=[], n_trials=fc['k_trials'].get(int(k[1:]) if k != 'kmax' else 'max', 0), minutes=5.8, best_valid=5.0, init_valid=5.1,
                                      kept_init=False, best_epoch=0, epochs_run=3)
        init = f'/shared/runs/loopin51/ese-c1-p2pre-s{sd}-eq/2026-10-16_c1_P2_pretrain_s{sd}_eq/pretrain.pt' if arm == 'P2' else None   # k3s path, fetched copy in runs_root
        summ = dict(arm=name, seed=sd, subject_fold=K, trial_fold=F, eval_split='test', shared_best_valid=5.0, shared_best_update=450, shared_updates=1200,
                    car_exclude_bad=True, device='NVIDIA GeForce RTX 5080', init=init, seconds=1.0)
        meta_only = {s: {kk: x for kk, x in r.items() if kk.endswith('_meta') or kk == 'n_trials'} for s, r in res.items()}
        json.dump(dict(summary=summ, targets=res), open(os.path.join(d, 'test_cells.json'), 'w'))
        json.dump(dict(summary=summ, targets=meta_only), open(os.path.join(d, 'metrics.json'), 'w'))
        np.savez_compressed(os.path.join(d, 'scores.npz'), **arrays)
    uni = lambda n: cell_from(rng.standard_normal((n, n)))
    fde = {'setting': c['decoder_setting'], 'car_exclude_bad': True, 'ks': ['1', '2', '4', 'max'], 'eval_split': 'test', 'subjects': {}}
    fm = {'meter': 'UNATTENDED', 'eval_split': 'test', 'cells': {}}
    for s in split.subjects:
        tests = [(f, r) for f in range(5) for r in split.sets(s, f)['test']]
        fde['subjects'][str(s)] = {k: dict(cells={str(r['trial']): uni(len(q[(s, r['trial'])])) for _, r in tests}) for k in KS[1:]}
        fm['cells'][str(s)] = {str(r['trial']): dict(uni(len(q[(s, r['trial'])])), fold=f, pair=r['pair']) for f, r in tests}
    fp = os.path.join(out, 'c2_de_fake.json'); json.dump(fde, open(fp, 'w')); mp = os.path.join(out, 'meters_fake.json'); json.dump(fm, open(mp, 'w'))
    cp = os.path.join(out, 'c1_verdict_fake.json'); json.dump(dict(verdict='adopt', c2_opened=True), open(cp, 'w'))
    vv = json.loads(json.dumps(v)); vv.update(runs_root=runs_root, out=os.path.join(out, 'verdict.json'), c1_verdict_file=cp)
    vv['prerequisites']['de_baseline'].update(file=fp, sha256=hashlib.sha256(open(fp, 'rb').read()).hexdigest(), committed_at='2000-01-01T00:00:00+00:00', commit='fake')
    vv['prerequisites']['unattended']['file'] = mp
    vp = os.path.join(out, 'c2_verdict_sim.yaml'); yaml.safe_dump(vv, open(vp, 'w')); return vp


def e2e(cfg, out, v):
    sc = cfg['end_to_end']; res = {}
    for i, (name, x) in enumerate(sc['scenarios'].items()):
        rng = np.random.default_rng([sc['seed'], i]); o = os.path.join(out, name); t0 = time.time(); vp = build_tree(x, o, v, rng); t1 = time.time()
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'c2_verdict.py'), vp], capture_output=True, text=True)
        print(f'[{name}] c2_verdict.py: {r.stdout.strip()[-1500:]}', flush=True)
        if r.returncode: print(r.stderr[-3000:], flush=True); raise SystemExit(f'c2_verdict.py failed in scenario {name}')
        V = json.load(open(os.path.join(o, 'verdict.json'))); P = V['prerequisites']
        res[name] = dict(verdict=V['verdict'], category=V['category'], confirmatory=V['confirmatory'], alpha_in_force=V['alpha_in_force'], delta=V['delta'],
                         p95=V['interval']['p95'], category_by_alpha={k: y['category'] for k, y in V['category_by_alpha'].items()},
                         e1={k: V['e1'][k] for k in ('ok', 'mrp_shift_cells', 'null_threshold', 'p', 'boot_percentile', 'short_cells_left_out')},
                         prerequisites={k: y['ok'] for k, y in P.items()}, unattended_p95=P['6_unattended']['interval']['p95'],
                         collapse_identity={a: P['4_collapse']['arms'][a]['identity'] for a in ('P1', 'P2')},
                         phaserand={a: (P['1_phaserand']['arms'][a]['mrp'], P['1_phaserand']['arms'][a]['threshold']) for a in ('P1', 'P2')},
                         protection=V['protection']['ok'], curves_k2={a: V['curves'][a]['k2']['mrp'] for a in ('P1', 'P2')},
                         build_seconds=t1 - t0, verdict_seconds=time.time() - t1)
    return res


def mc(cfg, v):
    q = cfg['monte_carlo']; _, _, fc, split, grid, _ = grids(v); units = sorted(fc['partner']); subjects = split.subjects
    n = np.zeros((len(subjects), len(units)))
    for js, s in enumerate(subjects):
        for f in range(5):
            for r in split.sets(s, f)['test']: n[js, units.index(r['pair'])] = len(grid[(s, r['trial'])])
    v1 = yaml.safe_load(open(path(v['c1_verdict_config']))); alpha = CS.alpha_chain_A(v1['family_state'])[0]; k = q['n_seeds']
    sd_cell = np.sqrt(1 / 12 / np.maximum(n, 1) + q['cell_sd_extra'] ** 2); out = {}
    for tau in q['seed_sd']:
        rng = np.random.default_rng([q['seed'], int(round(tau * 1e4))]); cats = []
        for _ in range(q['n_rep']):
            m = [0.5 + rng.normal(0, tau, (k, 1, 1)) + rng.standard_normal((k,) + n.shape) * sd_cell for _ in range(2)]
            pt = CS.delta(m[0], m[1], n); iv = CS.interval(CS.seed_pigeonhole(m, n, q['n_boot'], rng, seed_inflate=True))
            cats.append(CS.category(pt, iv[f'lower_{alpha}'], iv['p95'], v1['sesoi'])[0])
        out[f'seedsd{tau}'] = r_ = dict(n_rep=q['n_rep'], n_seeds=k, alpha=alpha, cells=int((n > 0).sum()), seed_sd=tau,
                                       reject=float(np.mean([x in CS.REJECTED for x in cats])), adopt=float(np.mean([x == 'adopt' for x in cats])),
                                       categories={c_: float(np.mean([x == c_ for x in cats])) for c_ in CS.CATEGORIES})
        print(f'seedsd{tau}: {json.dumps(r_)}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('out_dir'); ap.add_argument('--part', default='all')
    a = ap.parse_args(); cfg = yaml.safe_load(open(path(a.config))); v = yaml.safe_load(open(path(cfg['verdict_config']))); os.makedirs(a.out_dir, exist_ok=True); res = {}
    if a.part in ('e2e', 'all'): res['end_to_end'] = e2e(cfg, a.out_dir, v); print(json.dumps(res['end_to_end'], indent=1), flush=True)
    if a.part in ('mc', 'all'): res['monte_carlo'] = mc(cfg, v)
    json.dump(res, open(os.path.join(a.out_dir, f'c2_null_sim_{a.part}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
