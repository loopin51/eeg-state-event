"""F-V (D-112 section 4.4-4.5): development candidate choice and gate D from the development runs (CPU, no test data).

Reads the dev run folders of scripts/fv_train.py (results/runs/*_fv_dev_<cand>_s<seed>_t<target>[_shuffle]/dev_scores.npz),
set A (development concepts in the target's session 4) unless --set B_ses34 (report only). Per candidate the predicted
embeddings are averaged over seeds, L2-normalised and scored again against the 200 development concept names; the cell is
(target, concept) and the statistic is the mean of the cell values (each target weighted by its concept count).
  choice : higher seed-mean MRP at condition ses123; difference < tie (0.005) -> bm.
  D (i)  : MRP minus the mean of the concept-permutation null (per target, the true columns of the evaluated concepts are
           re-assigned among themselves, a bijection; n_perm draws) >= +0.10; the null 95th percentile is reported too;
  D (ii) : label-shuffle development model (same candidate, same procedure): 90% pigeonhole interval of MRP contains 0.5;
  D (iii): Gaussian input of the chosen candidate: 90% pigeonhole upper bound of MRP < 0.51.
Pigeonhole: targets x concepts with replacement (each target's concept draws inside its own concepts), n_boot draws.

usage: python scripts/fv_dev_gate.py configs/fv_train.yaml [--set A_ses4] [--choose-only] [--out results/fv/dev_gate.json]
       --choose-only: the candidate choice only (written to results/fv/dev_choice_<set>.json, printed: the chosen name)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def unit(A): return A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)


def runs(c, cand, target, shuffle):
    pat = f"*_fv_dev_{cand}_s*_t{target:02d}{'_shuffle' if shuffle else ''}"
    return sorted(d for d in glob.glob(os.path.join(P(c['out_root']), pat)) if os.path.exists(os.path.join(d, 'dev_scores.npz')) and not d.endswith('_smoke'))


def cells(c, cand, target, shuffle, setname, cond, cand_emb):
    """seed-mean embedding scores -> (rank-percentile cells [m], score matrix [m, K], true [m], seeds)"""
    ds = runs(c, cand, target, shuffle); assert ds, f'no dev runs for {cand} t{target} shuffle={shuffle}'
    Z = [np.load(os.path.join(d, 'dev_scores.npz')) for d in ds]; true = Z[0][f'{setname}_true']
    assert all(np.array_equal(z[f'{setname}_true'], true) for z in Z), 'seeds evaluated different concepts'
    E = unit(np.mean([unit(z[f'{setname}_E_{cond}']) for z in Z], 0)); S = E @ unit(cand_emb).T
    rk = (S > S[np.arange(len(S)), true][:, None]).sum(1); return 1 - rk / (S.shape[1] - 1), S, true, [os.path.basename(d) for d in ds]


def mrp_of(S, true):
    rk = (S > S[np.arange(len(S)), true][:, None]).sum(1); return 1 - rk / (S.shape[1] - 1)


def pigeonhole(vals, n_boot, rng):
    """vals: list of per-target cell arrays -> bootstrap means (targets, then concepts inside each drawn target)"""
    out = np.empty(n_boot); T = len(vals)
    for b in range(n_boot):
        ts = rng.integers(0, T, T); out[b] = np.mean(np.concatenate([vals[t][rng.integers(0, len(vals[t]), len(vals[t]))] for t in ts]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--set', default='A_ses4'); ap.add_argument('--out'); ap.add_argument('--choose-only', action='store_true'); a = ap.parse_args()
    c = yaml.safe_load(open(P(a.config))); g = c['gate_d']; targets = c['dev']['dev_targets']
    sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    from ese.things.data import Meta
    from fv_train import name_rows
    meta = Meta(P(c['meta'])); dev_c = meta.dev_concepts(c['dev']['salt'], c['dev']['n'])
    cand_emb = np.load(os.path.join(P(c['targets']['dir']), c['targets']['names_text']))[name_rows(c, meta, dev_c)]
    res = dict(set=a.set, targets=targets, candidates={})
    for cand in ('bm', 'nice'):
        try: v = [cells(c, cand, t, False, a.set, 'ses123', cand_emb) for t in targets]
        except AssertionError as e: res['candidates'][cand] = dict(missing=str(e)); continue
        allv = np.concatenate([x[0] for x in v]); res['candidates'][cand] = dict(mrp=float(allv.mean()), per_target={str(t): float(x[0].mean()) for t, x in zip(targets, v)},
                                                                                 n_concepts={str(t): int(len(x[0])) for t, x in zip(targets, v)}, runs=[x[3] for x in v])
    ok = {k: v['mrp'] for k, v in res['candidates'].items() if 'mrp' in v}; assert ok, 'no candidate has dev runs'
    assert len(ok) == 2 or not a.choose_only, f'candidate choice needs both candidates ({sorted(ok)} present)'
    chosen = 'bm' if ('bm' in ok and (len(ok) == 1 or ok['bm'] >= max(ok.values()) - g['tie'])) else max(ok, key=ok.get); res['chosen'] = chosen
    if a.choose_only:                                                           # before the label-shuffle runs exist: the choice only, values kept in the file
        out = P(os.path.join(g['out_dir'], f'dev_choice_{a.set}.json')); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
        print(json.dumps({'chosen': chosen, 'out': os.path.relpath(out, ROOT)}), flush=True); return
    rng = np.random.default_rng(g['perm_seed']); v = [cells(c, chosen, t, False, a.set, 'ses123', cand_emb) for t in targets]
    obs = float(np.concatenate([x[0] for x in v]).mean()); null = np.empty(g['n_perm'])
    for b in range(g['n_perm']):
        null[b] = np.mean(np.concatenate([mrp_of(S, true[rng.permutation(len(true))]) for _, S, true, _ in v]))
    exc = obs - null.mean(); res['D_i'] = dict(mrp=obs, null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)), excess=float(exc),
                                                excess_over_p95=float(obs - np.percentile(null, 95)), threshold=g['excess_min'], passed=bool(exc >= g['excess_min']))
    rb = np.random.default_rng(g['boot_seed'])
    try:
        sv = [cells(c, chosen, t, True, a.set, 'ses123', cand_emb)[0] for t in targets]; bs = pigeonhole(sv, g['n_boot'], rb); lo, hi = np.percentile(bs, [5, 95])
        res['D_ii'] = dict(mrp=float(np.concatenate(sv).mean()), ci90=[float(lo), float(hi)], passed=bool(lo <= 0.5 <= hi))
    except AssertionError as e: res['D_ii'] = dict(missing=str(e), passed=None)
    nv = [cells(c, chosen, t, False, a.set, 'ses123_noise', cand_emb)[0] for t in targets]; bn = pigeonhole(nv, g['n_boot'], rb); lo, hi = np.percentile(bn, [5, 95])
    res['D_iii'] = dict(mrp=float(np.concatenate(nv).mean()), ci90=[float(lo), float(hi)], threshold_upper=g['noise_upper'], passed=bool(hi < g['noise_upper']))
    res['gate_D'] = None if res['D_ii']['passed'] is None else bool(res['D_i']['passed'] and res['D_ii']['passed'] and res['D_iii']['passed'])
    out = P(a.out or os.path.join(c['gate_d']['out_dir'], f'dev_gate_{a.set}.json')); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('chosen', 'gate_D')}), flush=True)


if __name__ == '__main__':
    main()
