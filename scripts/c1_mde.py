"""C1 MDE (D-097 section 10), recorded after the P1 seeds are trained and before any P2 test cell is computed.

Reads only the P1 test_cells.json files and writes / prints seed-variation quantities only; the P1 level (pooled MRP) is never
written or printed.
  SE_0 : sd (ddof 1) of the null-pair statistic Delta0* over 1,000 draws (seed 20261098). In every draw two seed sets of size n
         are drawn independently with replacement from the n P1 seeds (set A, then set B), then subjects and runs as in the C1
         interval (scripts/c1_stats.seed_pigeonhole); Delta0* = query-weighted mean of (mean over set B - mean over set A).
         SE_0 = sd x sqrt(n / (n - 1)) (undoes the (n - 1) / n variance shrinkage of drawing n seeds with replacement).
  MDE  : (z_{1-0.04} + z_{0.80}) x SE_0 = 2.592 x SE_0 (alpha fixed at 0.04 even when chain A later holds 0.05).
  s_M  : sample sd (ddof 1) of the seed-wise pooled k2 MRP; reference MDE ~ 2.592 x sqrt(2 / n) x s_M (report).
  cell seed variance: query-weighted mean over cells of the across-seed variance (ddof 1).
  common share: s_M^2 / cell seed variance (1 when the seed difference is common to every cell, about 1 / (number of cells)
         when the cells vary independently).
Decision: MDE > 0.03 -> 5 seeds ({2036, ..., 2040}) in both arms. When the extended seeds are also present the 5-seed MDE is
recomputed and recorded (report); there is no further rule (D-096 decision 4).

usage: python scripts/c1_mde.py configs/c1_verdict.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, argparse, yaml, numpy as np
from scipy.stats import norm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import c1_stats as CS
from c1_verdict import path, arm_setup, find_run, load_arm, tensor


def mde_block(m, n, q):
    k = len(m); rng = np.random.default_rng(q['seed'])
    boot = CS.seed_pigeonhole([m, m], n, q['n_boot'], rng)                        # two independent seed sets from the same arm
    se0 = float(boot.std(ddof=1) * np.sqrt(k / (k - 1))); z = float(norm.ppf(1 - q['alpha']) + norm.ppf(q['power']))
    per_seed = np.array([CS.level(m[i], n) for i in range(k)]); s_m = float(per_seed.std(ddof=1))
    cv = CS.level(np.where(n > 0, m.var(0, ddof=1), 0.0), n)
    return dict(n_seeds=k, SE_0=se0, z_sum=z, MDE=z * se0, s_M=s_m, MDE_reference=z * np.sqrt(2 / k) * s_m,
                cell_seed_variance=cv, common_share=float(s_m ** 2 / cv) if cv > 0 else None, n_cells=int((n > 0).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); v = yaml.safe_load(open(path(a.config))); q = v['mde']
    _, _, pool, _ = arm_setup(v['arms']['P1']); units = list(range(1, 21)); key = f"k{q['k']}"
    _, _, _, cells = load_arm(v, v['arms']['P1'], v['seeds']); m, n = tensor(cells, v['seeds'], pool, units, key)
    res = dict(statistic=f'P1 null pair, {key}, 17 pool subjects x 20 test runs', base=mde_block(m, n, q), threshold=q['threshold'])
    res['extend'] = bool(res['base']['MDE'] > q['threshold'])
    res['seeds_required'] = v['seeds'] + (v['seeds_extended'] if res['extend'] else [])
    ext = v['seeds'] + v['seeds_extended']
    have = bool(v['seeds_extended']) and all(glob.glob(os.path.join(path(v['runs_root']), f"*_c1_P1_*_s{sd}_k{K}_f{F}")) for sd in v['seeds_extended'] for K in range(5) for F in range(5))
    if have:
        _, _, _, cells5 = load_arm(v, v['arms']['P1'], ext); m5, n5 = tensor(cells5, ext, pool, units, key); res['five_seeds'] = mde_block(m5, n5, q)
    out = path(q['file']); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    show = {k: res['base'][k] for k in ('n_seeds', 'SE_0', 'MDE', 's_M', 'MDE_reference', 'cell_seed_variance', 'common_share')}
    print(json.dumps(dict(base=show, extend=res['extend'], seeds_required=res['seeds_required'],
                          five_seeds={k: res['five_seeds'][k] for k in show} if 'five_seeds' in res else None)), flush=True)


if __name__ == '__main__':
    main()
