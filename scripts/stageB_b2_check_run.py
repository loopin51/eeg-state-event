"""B2 run folder check before the remaining jobs are submitted (D-091 amendment 6, D-095). Reads the test score matrices but never
compares them with the true candidates, so no performance number is computed. Pass: all files present, eval_split test,
car_exclude_bad true, finite history, and per target (main) within-candidate sd over queries / within-query sd over candidates
>= 0.5 and top share of the column-centred argmax <= 0.10. Seconds, shared best update and the 1,200-update floor are reported only.
usage: python scripts/stageB_b2_check_run.py <run_dir> [patience_units=4] [min_updates=1200]"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, numpy as np
d = sys.argv[1]; pat = int(sys.argv[2]) if len(sys.argv) > 2 else 4; mn = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
m = json.load(open(os.path.join(d, 'metrics.json'))); H = json.load(open(os.path.join(d, 'history.json')))
s = m['summary']; hs = [h for h in H if h['stage'] == 'shared']
print('files', sorted(os.listdir(d)))
print('summary', json.dumps({k: s.get(k) for k in ('eval_split', 'n_queries', 'shared_best_valid', 'shared_best_update', 'shared_updates',
                                               'car_exclude_bad', 'seconds')}))
print('shared valid curve', [(h.get('update', h.get('epoch')), round(h['valid'], 4)) for h in hs])
best, bad, stop0 = float('inf'), 0, None
for h in hs:
    if h['valid'] < best: best, bad = h['valid'], 0
    else:
        bad += 1
        if bad >= pat: stop0 = h.get('update'); break
print(f'patience-only stop: {stop0}; floor {mn} used (stop0 < best update): {stop0 is not None and s.get("shared_best_update") is not None and stop0 < s["shared_best_update"]}')
nonfinite = [h for h in H if (h.get('valid') is not None and not np.isfinite(h['valid'])) or (h.get('train') is not None and not np.isfinite(h['train']))]
print('non-finite history entries', len(nonfinite))
cal = {}
for h in H:
    if h['stage'].startswith('cal_'): cal.setdefault(h['stage'], []).append(h)
print('calibration epochs per target', {k: len(v) - 1 for k, v in cal.items()},
      'best epoch', {k: min(v, key=lambda h: h['valid'])['epoch'] for k, v in cal.items()})
Z = np.load(os.path.join(d, 'scores.npz'))
for k in sorted(Z.files):
    if not (k.startswith('main_') or k.startswith('pre_')) or Z[k].ndim != 2: continue
    S = Z[k].astype(np.float32); rows = np.where(np.isfinite(S).all(1))[0]; S = S[rows]
    col_sd = S.std(0).mean(); row_sd = S.std(1).mean(); Sc = S - S.mean(0, keepdims=True); arg = Sc.argmax(1)
    print(f'{k}: rows {len(rows)}/{Z[k].shape[0]}, within-candidate sd over queries / within-query sd over candidates '
          f'{col_sd / row_sd:.3f}, column-centred argmax distinct {len(np.unique(arg)) / len(rows):.3f}, '
          f'top share {np.bincount(arg).max() / len(rows):.3f}')
