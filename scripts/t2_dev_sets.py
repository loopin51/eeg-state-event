"""F-IV T2 development-cell matched sets (D-109 amendment 1), from the 0-T2 attribute file only (no EEG, no score).

Development cells: B1 k0_f0, validation runs 5 and 13. Level L0 (as chosen by 0-T2): the candidates come from the same run
only (the 0-T2 rule of section 3 restricted to the run). Level L1 or above: the pool is the same run plus the other validation
run (5 <-> 13). Tolerances of the chosen level (L0/L1: n_L and D_T +-10 %, n_S +-1; L2: +-15 %, +-1; L3: +-20 %, +-2), same word
count, a different word sequence, the 30-s rule for same-run candidates; the 19 candidates with the smallest
sha256(query key + "|" + candidate key + "|20261301") hex strings; queries with fewer than 19 are left out.

usage: python scripts/t2_dev_sets.py configs/t2.yaml [--sets PATH --attrs PATH --out PATH]
"""
import os, sys, json, hashlib, argparse, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from t2_io import read_sets, read_attrs
TOL = {'L0': (0.10, 1, 0.10), 'L1': (0.10, 1, 0.10), 'L2': (0.15, 1, 0.15), 'L3': (0.20, 2, 0.20)}
DEV_RUNS = (5, 13)


def path(p): return p if os.path.isabs(p) else os.path.join(ROOT, p)
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--sets'); ap.add_argument('--attrs'); ap.add_argument('--out'); a = ap.parse_args()
    v = yaml.safe_load(open(path(a.config))); setp, attp = path(a.sets or v['matched_sets']), path(a.attrs or v['match_attrs'])
    level, S = read_sets(setp); A = read_attrs(attp)
    assert level in TOL, f'chosen level unknown: {level}'; tl, ts, td = TOL[level]
    keys = {r: [k for k, x in A.items() if int(x[0]) == r] for r in DEV_RUNS}; out, n_short = {}, 0
    for r in DEV_RUNS:
        pool = keys[r] + ([] if level == 'L0' else keys[DEV_RUNS[1] if r == DEV_RUNS[0] else DEV_RUNS[0]])
        for q in keys[r]:
            _, tq, _, nw, nL, nS, DT, seq = A[q]; el = []
            for c in pool:
                if c == q: continue
                rc, tc, _, cw, cL, cS, cD, cseq = A[c]
                if int(rc) == r and abs(tc - tq) < 30.0: continue
                if cw != nw or cseq == seq or abs(cL - nL) > tl * nL or abs(cS - nS) > ts or abs(cD - DT) > td * DT: continue
                el.append(c)
            el.sort(key=lambda c: hashlib.sha256(f'{q}|{c}|20261301'.encode()).hexdigest())
            if len(el) >= 19: out[q] = dict(cands=el[:19], level=level, pool='same run' if level == 'L0' else 'same run + other validation run')
            else: n_short += 1
    if level == 'L0':                                                       # must reproduce the 0-T2 sets of the two runs exactly
        ref = {k: x['cands'] for k, x in S.items() if int(A[k][0]) in DEV_RUNS and len(x['cands']) == 19}
        assert ref == {k: x['cands'] for k, x in out.items()}, 'L0 development sets differ from the 0-T2 matched sets'
    res = dict(level=level, dev_runs=list(DEV_RUNS), sets=[dict(q=k, cands=x['cands'], cands_same_run=[], pool=x['pool']) for k, x in out.items()], n_queries=len(out), n_short=n_short,
               inputs=dict(matched_sets=sha(setp), match_attrs=sha(attp)), note='D-109 amendment 1; stimulus attributes only')
    o = path(a.out or v['matched_sets_dev']); os.makedirs(os.path.dirname(o), exist_ok=True); json.dump(res, open(o, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('level', 'n_queries', 'n_short')}), flush=True)


if __name__ == '__main__':
    main()
