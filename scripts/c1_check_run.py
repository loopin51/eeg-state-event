"""First-job check of a C1 run folder (D-097 section 15, same structure as scripts/stageB_b2_check_run.py, D-091 amendments 6-8).

Never compares scores with the true candidates and never reads the cell values of test_cells.json (only that the file exists),
so no performance number is computed.
Pass conditions:
  (1) result files present (config.yaml, metrics.json, history.json, scores.npz, shared.pt, target_layers_k14.pt, test_cells.json),
      summary.eval_split == test, summary.car_exclude_bad is true, summary.device contains the registered device, no non-finite
      value in history.json;
  (2) per target, the k2 within-run score matrices: (within-candidate sd over queries) / (within-query sd over candidates) >= 0.5
      (D-091 amendment 6 (a); < 0.5 means a nearly constant output), computed per test run and averaged with query weights per
      target (the smallest run value is reported too).
Report only: (b) top share of the column-centred argmax (D-091 amendment 6 (b), report only since amendment 7), seconds, shared
best update and whether the 1,200-update floor changed the stop, calibration best epochs, k3s attempt files, and the CUDA
caching-allocator OOM-retry warnings in k3s_stdout.log (job 326 on the RTX 4070 logged one while scoring and still finished;
report only, the run is unchanged when the allocator's retry succeeds, an uncaught OOM fails the job).

usage: python scripts/c1_check_run.py configs/c1_p1.yaml <run_dir> [<run_dir> ...]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, yaml, numpy as np

FILES = ('config.yaml', 'metrics.json', 'history.json', 'scores.npz', 'shared.pt', 'target_layers_k14.pt', 'test_cells.json')
K3S = ('k3s_commit.txt', 'k3s_attempts.txt', 'k3s_pip_freeze.txt')
OOM_WARN = 'memory allocation failed with OOM'


def check(cc, d, device='RTX 5080', key='main_k2', ratio_min=0.5):
    out = dict(run=os.path.basename(d.rstrip('/'))); miss = [f for f in FILES if not os.path.exists(os.path.join(d, f))]
    out['missing_files'] = miss; out['k3s_files'] = {f: os.path.exists(os.path.join(d, f)) for f in K3S}
    if os.path.exists(os.path.join(d, 'k3s_attempts.txt')): out['k3s_attempts'] = open(os.path.join(d, 'k3s_attempts.txt')).read().strip()
    log = os.path.join(d, 'k3s_stdout.log')
    out['allocator_oom_warnings'] = sum(OOM_WARN in l for l in open(log, errors='replace')) if os.path.exists(log) else None
    if 'metrics.json' in miss or 'history.json' in miss: out['pass'] = False; return out
    s = json.load(open(os.path.join(d, 'metrics.json')))['summary']; H = json.load(open(os.path.join(d, 'history.json')))
    nonfinite = sum(1 for h in H for f in ('train', 'valid') if h.get(f) is not None and not np.isfinite(h[f]))
    hs = [h for h in H if h['stage'] == 'shared']; sh = cc['shared_training']
    best, bad, stop0 = float('inf'), 0, None
    for h in hs:
        if h['valid'] < best: best, bad = h['valid'], 0
        else:
            bad += 1
            if bad >= sh['patience_units']: stop0 = h['update']; break
    cal = {}
    for h in H:
        if h['stage'].startswith('cal_') and h['epoch'] >= 0: cal.setdefault(h['stage'], []).append(h)
    out.update(eval_split=s.get('eval_split'), car_exclude_bad=s.get('car_exclude_bad'), device=s.get('device'), seconds=s.get('seconds'),
               max_memory_allocated_gb=s.get('max_memory_allocated_gb'), shared_best_update=s.get('shared_best_update'),
               shared_updates=s.get('shared_updates'), shared_best_valid=s.get('shared_best_valid'), nonfinite_history=nonfinite,
               patience_only_stop=stop0, floor_extended_training=bool(stop0 is not None and stop0 < sh['min_updates']),
               floor_changed_selection=bool(stop0 is not None and s.get('shared_best_update') is not None and stop0 < s['shared_best_update']),
               calibration_best_epoch={k: min(v, key=lambda h: h['valid'])['epoch'] for k, v in cal.items()})
    c1 = (not miss and s.get('eval_split') == 'test' and s.get('car_exclude_bad') is True and device in str(s.get('device')) and nonfinite == 0)
    ratios = {}
    if 'scores.npz' not in miss:
        Z = np.load(os.path.join(d, 'scores.npz'))
        for f in sorted(Z.files):
            if not f.startswith(key + '_s') or f.endswith('_onset'): continue
            S = Z[f].astype(np.float32); sub = f[len(key) + 2:].split('_r')[0]
            Sc = S - S.mean(0, keepdims=True); arg = Sc.argmax(1)
            ratios.setdefault(sub, []).append((float(S.std(0).mean() / S.std(1).mean()), float(np.bincount(arg).max() / len(arg)), len(S)))
    per = {sub: dict(ratio=float(np.average([x[0] for x in v], weights=[x[2] for x in v])), ratio_min_run=min(x[0] for x in v),
                     top_share=float(np.average([x[1] for x in v], weights=[x[2] for x in v])), runs=len(v)) for sub, v in ratios.items()}
    out['targets'] = per; c2 = bool(per) and all(x['ratio'] >= ratio_min for x in per.values())
    out['pass_files'] = bool(c1); out['pass_ratio'] = bool(c2); out['pass'] = bool(c1 and c2)
    return out


def main():
    cc = yaml.safe_load(open(sys.argv[1])); res = [check(cc, d) for d in sys.argv[2:]]
    for r in res: print(json.dumps(r), flush=True)
    print('ALL PASS' if all(r['pass'] for r in res) else 'FAIL: ' + ', '.join(r['run'] for r in res if not r['pass']), flush=True)


if __name__ == '__main__':
    main()
