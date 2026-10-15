"""F-III S-diagnosis descriptive summary (D-105 section 10; D-104 decision 7 (a) and (b)). No alpha, no verdict.

Reads the a1_diag metrics.json of the 5 existing P1 k0_f0 models and the 5 P2 hours development runs, and writes per arm:
  - the 26-cell query-weighted MRP of every layer (own, pseudo-target k2) x (d0_fixed, j250_fixed, j250_<rule>) per seed and
    the 5-seed mean and sd;
  - the own-layer delta grid (fixed and every LS rule; weights n_grid, query margin 0.75 s);
  - the negative-pair loss decomposition (5-seed mean of every field);
  - the reproduction checks and run cost (seconds, memory).
P2h - P1 per seed (same seed) is recorded only (next family's pretraining question; not part of F-III).
Decision 7 (b): from each C1 P2 pretraining history.json, the best update, the per-dataset best update, the per-dataset valid
loss at the overall best and at the last evaluation, the number of evaluations after the best, and the chance loss.

usage: python scripts/a1_diag_summary.py configs/a1_diag_summary.yaml
"""
import os, glob, json, hashlib, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(p): return p if os.path.isabs(p) else os.path.join(ROOT, p)
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def one(pat):
    hits = [h for h in sorted(glob.glob(pat)) if os.path.exists(os.path.join(h, 'metrics.json'))]
    assert len(hits) == 1, f'{pat}: {len(hits)} finished folders'; return hits[0]


def pooled(cells, key, w):
    return float(sum(c[key] * c[w] for c in cells) / sum(c[w] for c in cells))


def ms(xs): return dict(mean=float(np.mean(xs)), sd=float(np.std(xs, ddof=1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    v = yaml.safe_load(open(path(a.config))); root = path(v['runs_root'])
    ds = json.load(open(path(v['dev_stage'])))['ls']; frozen = ds['rule']
    res = dict(frozen_rule=frozen, n_seeds=ds['n_seeds'], arms={}, note='descriptive (D-105 section 10); no alpha')
    per_arm = {}
    for arm, pat in v['diag_pattern'].items():
        seeds = {}
        for s in v['seeds']:
            d = one(os.path.join(root, pat.format(seed=s))); M = json.load(open(os.path.join(d, 'metrics.json'))); rules = M['summary']['rules']
            keys = ['d0_fixed', 'j250_fixed'] + [f'j250_{r}' for r in rules]
            lay = {}
            for L in v['layers']:
                cl = [c for k, c in M['cells'].items() if k.startswith(L + '_')]; assert len(cl) == 26, (d, L, len(cl))
                lay[L] = {k: pooled(cl, k, 'n_main') for k in keys}; lay[L]['n_queries'] = int(sum(c['n_main'] for c in cl))
            gl = [c for k, c in M['cells'].items() if k.startswith(v['grid_layer'] + '_')]
            gk = sorted({k for k in gl[0] if k.startswith('grid')}, key=lambda k: (float(k[4:].split('_')[0]), k))
            grid = {k: pooled(gl, k, 'n_grid') for k in gk}
            seeds[str(s)] = dict(folder=os.path.basename(d), sha256=sha(os.path.join(d, 'metrics.json')), model=M['summary']['model'],
                                 layers=lay, grid=grid, negatives=M['negatives'], checks=M['checks'],
                                 seconds=M['summary']['seconds'], max_memory_allocated_gb=M['summary']['max_memory_allocated_gb'], device=M['summary']['device'])
        S = list(seeds.values())
        summ = dict(layers={L: {k: ms([x['layers'][L][k] for x in S]) for k in S[0]['layers'][L] if k != 'n_queries'} for L in v['layers']},
                    grid={k: ms([x['grid'][k] for x in S]) for k in S[0]['grid']},
                    negatives={k: float(np.mean([x['negatives'][k] for x in S])) for k in S[0]['negatives']},
                    max_check={k: float(max(abs(x['checks'][k]) for x in S)) for k in S[0]['checks']},
                    seconds=ms([x['seconds'] for x in S]), max_memory_allocated_gb=float(max(x['max_memory_allocated_gb'] for x in S)))
        res['arms'][arm] = dict(seeds=seeds, summary=summ); per_arm[arm] = seeds
    diff = {}
    for L in v['layers']:
        for k in ['d0_fixed', 'j250_fixed', f'j250_{frozen}']:
            xs = [per_arm['P2h'][str(s)]['layers'][L][k] - per_arm['P1'][str(s)]['layers'][L][k] for s in v['seeds']]
            diff[f'{L}_{k}'] = dict(per_seed=dict(zip(map(str, v['seeds']), map(float, xs))), **ms(xs))
    res['P2h_minus_P1'] = dict(values=diff, note='recorded only: next family pretraining question, not part of F-III (D-105 section 10)')
    pre = {}
    for s in v['seeds']:
        for c in v['pretrain_candidates']:
            hits = [h for h in sorted(glob.glob(os.path.join(root, v['pretrain_pattern'].format(seed=s, cand=c)))) if os.path.exists(os.path.join(h, 'history.json'))]
            if not hits: continue
            assert len(hits) == 1, hits; d = hits[0]; H = [h for h in json.load(open(os.path.join(d, 'history.json'))) if h.get('stage') == 'pretrain']
            M = json.load(open(os.path.join(d, 'metrics.json'))); dsn = list(H[0]['valid_per_dataset'])
            ib = int(np.argmin([h['valid'] for h in H])); assert H[ib]['update'] == M['best_update'], d
            pre[f's{s}_{c}'] = dict(folder=os.path.basename(d), sha256=sha(os.path.join(d, 'history.json')), updates=M['updates'], best_update=M['best_update'],
                                    evals_after_best=len(H) - 1 - ib, chance_valid=M['chance_valid'],
                                    valid_at_best=H[ib]['valid'], valid_last=H[-1]['valid'], train_at_best=H[ib]['train'], train_last=H[-1]['train'],
                                    per_dataset={n: dict(best_update=int(H[int(np.argmin([h['valid_per_dataset'][n] for h in H]))]['update']),
                                                         best=float(min(h['valid_per_dataset'][n] for h in H)),
                                                         at_overall_best=H[ib]['valid_per_dataset'][n], last=H[-1]['valid_per_dataset'][n],
                                                         first=H[0]['valid_per_dataset'][n]) for n in dsn})
    res['pretrain_history'] = pre
    out = path(v['out']); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(dict(arms={a_: x['summary'] for a_, x in res['arms'].items()}, P2h_minus_P1={k: {kk: vv for kk, vv in x.items() if kk != 'per_seed'} for k, x in diff.items()},
                          pretrain={k: {kk: x[kk] for kk in ('best_update', 'updates', 'evals_after_best')} | {n: (p['best_update'], round(p['at_overall_best'], 3), round(p['last'], 3)) for n, p in x['per_dataset'].items()} for k, x in pre.items()})), flush=True)


if __name__ == '__main__':
    main()
