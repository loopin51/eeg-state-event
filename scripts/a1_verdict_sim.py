"""Full-path check of the F-III A1 verdict (D-105 amendment 3: the null simulation through the verdict script).

Every repetition writes information-0 fake run folders in the registered formats (test_cells.json, scores.npz, metrics.json;
both arms x 5 seeds x 25 cells + 5 phase-randomised folds each), a fake DE baseline and a fake development record (gate
passed), then runs scripts/a1_mde.py and scripts/a1_verdict.py unchanged on them and records the category, the verdict, the
prerequisites and the protection result. Score matrices are i.i.d. normal (rows queries, columns the same queries); with
`effect` > 0 the N arm's diagonal score is raised (the decision field, delta 0 and J250 fixed of every k; PRE and the
phase-randomised folds stay at information 0). Query counts are small (configs/a1_verdict_sim.yaml queries) so that a
repetition takes about a minute; this is a code-path check with an empirical error rate from few repetitions, not the
2,000-repetition cell-level simulation (scripts/a1_null_sim.py). With so few queries per run the 30-s exclusion leaves the
collapsed matrices almost invariant to the shift (null sd ~6e-4, subject p's bunched around 0.3-0.6), so prerequisite 5's KS
check fails here; at the real A1 query counts it does not (scripts/a1_collapse_sim.py, results/a1/collapse_sim.json).

usage: python scripts/a1_verdict_sim.py configs/a1_verdict_sim.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, shutil, hashlib, argparse, datetime, subprocess, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(p): return p if os.path.isabs(p) else os.path.join(ROOT, p)


def mrp_excl(S, t, excl):
    far = np.abs(t[:, None] - t[None]) >= excl if excl > 0 else ~np.eye(len(t), dtype=bool)
    return float((1 - ((S > np.diag(S)[:, None]) & far).sum(1) / np.maximum(far.sum(1), 1)).mean())


def rank_cell(S):
    rk = (S > np.diag(S)[:, None]).sum(1); n = len(S)
    return dict(mrp=float(np.mean(1 - rk / (n - 1))), n=n, top1=float((rk < 1).mean()), top10=float((rk < 10).mean()))


def onsets(q, r):
    n = q['base'] + q['step'] * r; return np.sort(np.random.default_rng([20261107, r]).uniform(0, q['span_s'], n))


def write_job(d, arm, cand, seed, targets, runs, q, A, rng, eff, rule, ks):
    os.makedirs(d); fields = ['d0_fixed', 'j250_fixed'] + [f'j250_{x}' for x in A['lag_search']['rules']]; J = f'j250_{rule}'
    excl = A['queries']['exclusion_s']; tg, meta, arrays = {}, {}, {}
    for s in targets:
        o = tg[str(s)] = {}; mt = meta[str(s)] = {}
        for k in ks:
            blk = o[f'a1_k{k}'] = {}; c1 = o[f'k{k}'] = {}
            for r in runs:
                t = onsets(q, r); n = len(t); cell = dict(n=n)
                for f in fields:
                    S = rng.standard_normal((n, n))
                    if f in (J, 'd0_fixed', 'j250_fixed') and eff: S[np.arange(n), np.arange(n)] += eff
                    cell[f] = mrp_excl(S, t, excl)
                    if k == 2 and f == J: arrays[f'k2_s{s}_r{r}_j250'] = S.astype(np.float16); arrays[f'k2_s{s}_r{r}_onset'] = t
                blk[str(r)] = cell; c1[str(r)] = rank_cell(rng.standard_normal((n, n)))
            if k: o[f'k{k}_meta'] = mt[f'k{k}_meta'] = dict(runs=[1], best_valid=0.0, init_valid=0.0, kept_init=False, best_epoch=0, epochs_run=4)
        pre = o['a1_pre_k2'] = {}
        for r in runs:
            t = onsets(q, r); n = len(t); S = rng.standard_normal((n, n)); pre[str(r)] = {'n': n, J: mrp_excl(S, t, excl)}
            arrays[f'pre_k2_s{s}_r{r}_j250'] = S.astype(np.float16); arrays[f'pre_k2_s{s}_r{r}_onset'] = t
    summ = dict(arm=arm, candidate=cand, seed=seed, eval_split='test', ls_rule=rule, device='NVIDIA GeForce RTX 5080', shared_best_update=1200,
                lse_masked_fraction_shared=0.001 if arm == 'N' else None)
    np.savez_compressed(os.path.join(d, 'scores.npz'), **arrays)
    json.dump(dict(summary=summ, targets=tg), open(os.path.join(d, 'test_cells.json'), 'w'))
    json.dump(dict(summary=summ, targets=meta), open(os.path.join(d, 'metrics.json'), 'w'))


def one_rep(v, W, rep_seed, eff):
    V = yaml.safe_load(open(path(v['verdict_config']))); A = yaml.safe_load(open(path(V['a1_config']))); rule = A['lag_search']['rule']
    c = yaml.safe_load(open(path(yaml.safe_load(open(path(V['arms']['P1'])))['b1_config']))); sp = c['split']; q = v['queries']
    if os.path.exists(W): shutil.rmtree(W)
    os.makedirs(os.path.join(W, 'runs')); t_de = datetime.datetime.now() - datetime.timedelta(seconds=60)
    cand = v['candidate']; arms = {}
    for arm in ('P1', 'N'):
        cc = yaml.safe_load(open(path(V['arms'][arm])))
        if arm == 'N': cc['candidate'] = cand
        p = os.path.join(W, f'arm_{arm}.yaml'); yaml.safe_dump(cc, open(p, 'w')); arms[arm] = p
    ks = V['ks']; extra = sp.get('extra_targets', {})
    for ai, arm in enumerate(('P1', 'N')):
        cn = cand if arm == 'N' else None
        for seed in V['seeds']:
            for K in range(5):
                tgt = sp['subject_folds'][K] + extra.get(K, extra.get(str(K), []))
                for F in range(5):
                    rng = np.random.default_rng([*rep_seed, ai, seed, K, F])
                    d = os.path.join(W, 'runs', f"2026-01-01_a1_{arm}{'-' + cn if cn else ''}_{c['decoder_setting']}_s{seed}_k{K}_f{F}")
                    write_job(d, arm, cn, seed, tgt, sp['run_folds'][F], q, A, rng, eff if arm == 'N' else 0.0, rule, ks)
        for F in range(5):
            rng = np.random.default_rng([*rep_seed, ai, 9999, F]); K = 0
            d = os.path.join(W, 'runs', f"2026-01-01_a1_{arm}{'-' + cn if cn else ''}_{c['decoder_setting']}_s{V['seeds'][0]}_k{K}_f{F}_phaserand")
            write_job(d, arm, cn, V['seeds'][0], sp['subject_folds'][K], sp['run_folds'][F], q, A, rng, 0.0, rule, ks)
    rng = np.random.default_rng([*rep_seed, 7]); fields = ['d0_fixed', 'j250_fixed'] + [f'j250_{x}' for x in A['lag_search']['rules']]
    pool19 = sorted(s for fs in sp['subject_folds'] for s in fs) + sorted(s for x in extra.values() for s in x)
    DE = dict(rules=A['lag_search']['rules'], eval_split='test', subjects={})
    for s in pool19:
        DE['subjects'][str(s)] = {}
        for k in ks:
            if k == 0: continue
            cells = {}
            for r in range(1, 21):
                t = onsets(q, r); n = len(t); cells[str(r)] = dict(n=n, **{f: mrp_excl(rng.standard_normal((n, n)), t, A['queries']['exclusion_s']) for f in fields})
            DE['subjects'][str(s)][f'k{k}'] = dict(cells=cells)
    dep = os.path.join(W, 'de.json'); json.dump(DE, open(dep, 'w')); de_sha = hashlib.sha256(open(dep, 'rb').read()).hexdigest()
    dsp = os.path.join(W, 'dev_stage.json'); json.dump(dict(ls=dict(rule=rule), gate=dict(passed=True, chosen=cand, gate=dict(i=True, ii=True, iii=True))), open(dsp, 'w'))
    V['runs_root'] = os.path.join(W, 'runs'); V['arms'] = arms; V['development'] = dict(file=dsp); V['out'] = os.path.join(W, 'verdict.json')
    V['prerequisites']['de_baseline'].update(file=dep, sha256=de_sha, committed_at=t_de.isoformat(timespec='seconds'), commit='sim')
    V['mde']['file'] = os.path.join(W, 'mde.json'); vp = os.path.join(W, 'verdict.yaml'); yaml.safe_dump(V, open(vp, 'w'))
    py = sys.executable; t0 = time.time()
    for scr in ('scripts/a1_mde.py', 'scripts/a1_verdict.py'):
        r = subprocess.run([py, os.path.join(ROOT, scr), vp], capture_output=True, text=True)
        assert r.returncode == 0, f'{scr} failed:\n{r.stderr[-3000:]}'
    R = json.load(open(V['out'])); M = json.load(open(V['mde']['file']))
    return dict(category=R['category'], verdict=R['verdict'], delta=R['delta'], lower=R['category_by_alpha'][str(R['alpha_in_force'])]['lower'],
                category_by_alpha={k: x['category'] for k, x in R['category_by_alpha'].items()}, MDE=M['base']['MDE'],
                prerequisites={k: x['ok'] for k, x in R['prerequisites'].items()}, protection=R['protection']['ok'],
                phaserand_p={a_: x['p'] for a_, x in R['prerequisites']['1_phaserand']['arms'].items()},
                collapse={a_: dict(p=x['p'], ks_p=x['ks_p'], identity=x['identity']) for a_, x in R['prerequisites']['5_collapse']['arms'].items()},
                seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); v = yaml.safe_load(open(path(a.config))); t0 = time.time(); res = dict(conditions={})
    for cd in v['conditions']:
        rows = []
        for i in range(cd['n_rep']):
            rows.append(one_rep(v, v['work_dir'], [v['seed'], int(round(cd['effect'] * 1000)), i], cd['effect']))
            print(cd['name'], i, json.dumps(rows[-1]), flush=True)
        cats = [x['category'] for x in rows]
        res['conditions'][cd['name']] = dict(effect=cd['effect'], n_rep=cd['n_rep'], rows=rows,
                                             category_rate={k: cats.count(k) / len(cats) for k in sorted(set(cats))},
                                             prerequisite_pass_rate={k: float(np.mean([x['prerequisites'][k] for x in rows])) for k in rows[0]['prerequisites']})
        print(cd['name'], json.dumps({k: x for k, x in res['conditions'][cd['name']].items() if k != 'rows'}), flush=True)
    res['seconds'] = time.time() - t0; out = path(v['out']); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    shutil.rmtree(v['work_dir'], ignore_errors=True); print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
