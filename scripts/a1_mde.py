"""F-III A1 MDE (D-105 section 6, prerequisite 7; the D-097 section 10 method on the A1 decision quantity), recorded after the
P1 arm's 5 seeds are trained and before any N test cell is computed.

Reads only the P1 test_cells.json files (block a1_k2, field j250_<frozen rule>) and writes / prints seed-variation quantities
only; the P1 level is never written or printed. SE_0, MDE = (z_{1-0.04} + z_{0.80}) x SE_0 = 2.592 x SE_0, s_M and the cell
seed variance are defined as in scripts/c1_mde.py (alpha fixed at 0.04 even though chain A holds 0.05). MDE > 0.03 ->
the A1 verdict is "hold (MDE)"; there is no seed-extension rule.

usage: python scripts/a1_mde.py configs/a1_verdict.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, argparse, datetime, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from c1_mde import mde_block
from a1_verdict import path, arm_setup, load_arm, tensor


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); v = yaml.safe_load(open(path(a.config))); q = v['mde']
    A = yaml.safe_load(open(path(v['a1_config']))); rule = A['lag_search']['rule']; assert rule in A['lag_search']['rules'], 'LS rule not frozen'; v['_rule'] = rule
    _, _, pool, _ = arm_setup(v['arms']['P1']); units = list(range(1, 21))
    _, _, runs, cells = load_arm(v, v['arms']['P1'], v['seeds']); m, n = tensor(cells, v['seeds'], pool, units, f"a1_k{q['k']}", f'j250_{rule}')
    res = dict(statistic=f"P1 null pair, J250 + LS ({rule}) k{q['k']}, 17 pool subjects x 20 test runs", seeds=v['seeds'], base=mde_block(m, n, q),
               threshold=q['threshold'], written_at=datetime.datetime.now().isoformat(timespec='seconds'),
               note='recorded and committed before any N test cell is computed (D-105 section 6, 7)')
    res['hold'] = bool(res['base']['MDE'] > q['threshold'])
    out = path(q['file']); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    show = {k: res['base'][k] for k in ('n_seeds', 'SE_0', 'MDE', 's_M', 'MDE_reference', 'cell_seed_variance', 'common_share')}
    print(json.dumps(dict(base=show, hold=res['hold'])), flush=True)


if __name__ == '__main__':
    main()
