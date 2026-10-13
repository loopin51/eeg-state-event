"""Stage B calibration curve report (D-090, CPU, descriptive): pools the 25 stageB_calib runs and the DE curve.

For every condition (k0_population, other_adapter, k1, k2, k4, k8, k14_b1) the statistic is the B1 primary statistic
(query-weighted mean of subject x run cell within-run MRP over the 17 pool subjects and 20 runs) with a subject x run
pigeonhole bootstrap interval (1,000 draws); also within-run top-1 / top-10, Wolpaw bits/min, the 19-subject value, the
paired differences k - k0 and k14 - k, own vs other adapter, decoder - DE at the same k, and the mean best epoch.
other_adapter: per target and run, the mean over the other targets' adapters of the same fold.

usage: python scripts/stageB_calib_report.py configs/stageB_calib.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageB_b1_verdict import stat, pigeonhole, q, wolpaw


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); b1 = yaml.safe_load(open(os.path.join(ROOT, c['b1_config']))); sp = b1['split']; nb = b1['bootstrap']['n_boot']
    cells, hist = {}, {}                                                       # cells[cond][subject][run] = dict
    for k in range(5):
        for f in range(5):
            d = sorted(glob.glob(os.path.join(ROOT, c['out_root'], f"*_stageB_calib_{b1['decoder_setting']}_k{k}_f{f}")))
            assert len(d) == 1, (k, f, d); m = json.load(open(os.path.join(d[0], 'metrics.json'))); assert m['summary']['eval_split'] == 'test'
            hist[(k, f)] = json.load(open(os.path.join(d[0], 'history.json')))
            for s, v in m['targets'].items():
                for cond, cl in v.items():
                    if cond.endswith('_meta'): continue
                    if cond == 'other_adapter':                                # mean over the other targets' adapters
                        cl = {r: dict(mrp=float(np.mean([o[r]['mrp'] for o in cl.values()])), n=next(iter(cl.values()))[r]['n'],
                                      top1=float(np.mean([o[r]['top1'] for o in cl.values()])), top10=float(np.mean([o[r]['top10'] for o in cl.values()])))
                             for r in next(iter(cl.values()))}
                    cells.setdefault(cond, {}).setdefault(int(s), {}).update(cl)
    de = json.load(open(os.path.join(ROOT, c['de']['out']))); de14 = json.load(open(os.path.join(ROOT, 'results/baselines/stageB_b1_de_order.json')))['DE'][b1['decoder_setting']]
    for kk in c['ks']: cells[f'de_k{kk}'] = {int(s): v[f'k{kk}']['cells'] for s, v in de['subjects'].items()}
    cells['de_k14'] = {int(s): v['cells'] for s, v in de14.items()}
    pool = sorted(s for fs in sp['subject_folds'] for s in fs); allsub = sorted(cells['k14_b1']); runs = [str(r) for r in range(1, 21)]
    w = np.array([cells['k14_b1'][pool[0]][r]['n'] for r in runs], float); N = float(np.mean(w))
    def mat(cond, key='mrp', subs=pool): return np.array([[cells[cond][s][r][key] for r in runs] for s in subs])
    rng = np.random.default_rng(b1['bootstrap']['seed']); out = {'pool': pool, 'n_candidates': N, 'conditions': {}, 'differences': {}}
    conds = ['k0_population', 'other_adapter'] + [f'k{k}' for k in c['ks']] + ['k14_b1'] + [f'de_k{k}' for k in c['ks']] + ['de_k14']
    for cond in conds:
        M = mat(cond); t1 = stat(mat(cond, 'top1'), w)
        out['conditions'][cond] = dict(mrp=stat(M, w), ci=q(pigeonhole(M, w, nb, rng)), top1=t1, top10=stat(mat(cond, 'top10'), w),
                                       bits_per_min=wolpaw(t1, N) * 20, mrp_19=stat(mat(cond, subs=allsub), w))
    pairs = [(f'k{k}', 'k0_population') for k in c['ks']] + [('k14_b1', f'k{k}') for k in c['ks']] + [('k14_b1', 'k0_population'), ('k14_b1', 'other_adapter')] \
        + [(f'k{k}', f'de_k{k}') for k in c['ks']] + [('k14_b1', 'de_k14')]
    for x, y in pairs:
        D = mat(x) - mat(y); out['differences'][f'{x} - {y}'] = dict(diff=stat(D, w), ci=q(pigeonhole(D, w, nb, rng)))
    ep = {}
    for H in hist.values():
        for h in H:
            if h['stage'].startswith('cal_') and h['epoch'] >= 0: ep.setdefault(h['stage'], []).append(h['epoch'])
    for k in c['ks']:
        out['conditions'][f'k{k}']['mean_epochs_run'] = float(np.mean([len(v) for st, v in ep.items() if st.endswith(f'_k{k}')]))
    o = os.path.join(ROOT, c['out']); os.makedirs(o, exist_ok=True); json.dump(out, open(os.path.join(o, 'curve.json'), 'w'), indent=1)
    print(json.dumps({k: round(v['mrp'], 4) for k, v in out['conditions'].items()}), flush=True)


if __name__ == '__main__':
    main()
