"""F-IV T2 re-inference check (D-109 section 6), before any TTS score is computed.

For each stored B1 job (25 main jobs; every target x test run cell, 19 subjects x 20 runs) the network is rebuilt from
shared.pt and target_layers.pt (scripts/t2_lib.Job, the B1 input path, no training) and the recorded-w2v within-run score
matrices of the B1 test queries are recomputed on this GPU and compared with the stored scores.npz main_s{s}_r{r}.
Per cell: max |score difference|, its ratio to the stored cell's score sd, and the B1-definition MRP difference. Only the
differences are written and printed; no MRP level. Pass: every cell |MRP difference| <= mrp_tol and max |difference| <=
score_tol_rel x score sd.

usage: python scripts/t2_reinfer_check.py configs/t2.yaml [--jobs K_F ...] [--targets N]
       --jobs / --targets: subset for the code-path smoke (the registered check is all 25 jobs, all targets)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, hashlib, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from t2_lib import Job, scores, b1_mrp, path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--jobs', nargs='*'); ap.add_argument('--targets', type=int)
    ap.add_argument('--out'); a = ap.parse_args(); v = yaml.safe_load(open(path(a.config))); q = v['reinfer_check']; t0 = time.time()
    jobs = [tuple(map(int, j.split('_'))) for j in a.jobs] if a.jobs else [(K, F) for K in range(5) for F in range(5)]
    cells, info = [], {}
    for K, F in jobs:
        J = Job(v['b1_config'], K, F); Z = np.load(os.path.join(J.folder, 'scores.npz'))
        items = J.D.items['eval']; run = np.array([r for r, _ in items]); onset = np.array([t for _, t in items])
        assert np.array_equal(run, Z['run']) and np.allclose(onset, Z['onset']), 'query set differs from the stored job'
        tg = J.targets[:a.targets] if a.targets else J.targets
        info[f'k{K}_f{F}'] = dict(folder=os.path.basename(J.folder), scores_sha256=hashlib.sha256(open(os.path.join(J.folder, 'scores.npz'), 'rb').read()).hexdigest(),
                                  n_queries=len(items), targets=tg)
        ch = J.c['eval']['score_chunk']; Yc = {}
        for s in tg:                                                               # B1 chunking: 128 queries in B1 eval order, across runs
            Snew = {r: np.full((int((run == r).sum()), int((run == r).sum())), np.nan, np.float32) for r in sorted(set(run))}
            pos = {i: int(np.where(np.where(run == run[i])[0] == i)[0][0]) for i in range(len(items))}
            for b0 in range(0, len(items), ch):
                ii = list(range(b0, min(b0 + ch, len(items)))); E = J.est([items[i] for i in ii], s, chunk=ch)
                for r in sorted({run[i] for i in ii}):
                    if r not in Yc:
                        Yc.clear(); torch.cuda.empty_cache(); Yc[r] = J.recorded_windows([items[i] for i in np.where(run == r)[0]])
                    sel = [j for j, i in enumerate(ii) if run[i] == r]; Snew[r][[pos[ii[j]] for j in sel]] = scores(E[sel], Yc[r])
                del E
            for r in sorted(set(run)):
                ir = np.where(run == r)[0]; S_new = Snew[r]; S_old = Z[f'main_s{s}_r{r}']
                d = np.abs(S_new - S_old); sd = float(np.nanstd(S_old)); m_new, n_new = b1_mrp(S_new); m_old, n_old = b1_mrp(S_old)
                assert n_new == n_old
                cells.append(dict(job=f'k{K}_f{F}', subject=int(s), run=int(r) + 1, n=int(len(ir)), max_abs_diff=float(np.nanmax(d)),
                                  max_abs_diff_rel_sd=float(np.nanmax(d) / sd), mrp_diff=m_new - m_old,
                                  ok=bool(abs(m_new - m_old) <= q['mrp_tol'] and np.nanmax(d) / sd <= q['score_tol_rel'])))
            torch.cuda.empty_cache()
        Yc.clear(); torch.cuda.empty_cache()
        print(f'k{K}_f{F}: {sum(c["job"] == f"k{K}_f{F}" for c in cells)} cells, max |MRP diff| '
              f'{max(abs(c["mrp_diff"]) for c in cells if c["job"] == f"k{K}_f{F}"):.2e}, max rel score diff '
              f'{max(c["max_abs_diff_rel_sd"] for c in cells if c["job"] == f"k{K}_f{F}"):.2e} ({time.time() - t0:.0f}s)', flush=True)
        del J, Z; torch.cuda.empty_cache()
    res = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, tolerances=q, jobs=info, cells=cells, n_cells=len(cells),
               max_abs_mrp_diff=max(abs(c['mrp_diff']) for c in cells), max_rel_score_diff=max(c['max_abs_diff_rel_sd'] for c in cells),
               max_abs_score_diff=max(c['max_abs_diff'] for c in cells), passed=all(c['ok'] for c in cells),
               complete=not (a.jobs or a.targets), max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2 ** 30, seconds=time.time() - t0,
               note='differences only (D-109 section 6); no MRP level is written')
    out = path(a.out or q['out']); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    print('SUMMARY', json.dumps({k: res[k] for k in ('device', 'n_cells', 'max_abs_mrp_diff', 'max_rel_score_diff', 'max_abs_score_diff', 'passed', 'complete',
                                                     'max_memory_allocated_gb', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
