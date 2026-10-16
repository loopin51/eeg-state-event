"""F-IV T2 development stage (D-109 section 7, amendment 1; alpha-free, before the test): alignment rule selection and the gate.

Reads the dev score matrices of scripts/t2_infer.py (--mode dev and --mode dev --noise; B1 k0_f0, the 13 training subjects'
own layers, validation runs 5 and 13, results/t2/matched_sets_dev.json). Per cell (subject, run): matched TTS MRP for fixed
and lse0.5, the 30-s-excluded shift-null mean (t2_lib.cell_null, seed of configs/t2.yaml shift_null), excess, noise-input
excess, recorded MRP and the unmatched TTS MRP (report). Rule: lse0.5 if pooled MRP(lse0.5) - MRP(fixed) >= 0.01, else fixed.
Gate: (i) excess of the selected rule >= +0.04, (ii) noise-input excess 90% upper < 0.01 (subject 13 x run 2 pigeonhole,
1,000, seed dev_gate.seed). Also writes the D variance components (recorded - TTS, selected rule) for the H_T2b MDE (section 12).
The rule is written into configs/t2.yaml by an amendment, not by this script. Development values are proxies, not effects.

usage: python scripts/t2_dev.py configs/t2.yaml [--scores DIR] [--sets PATH --attrs PATH --out PATH]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, argparse, hashlib, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import t2_lib as T
from t2_io import read_sets, read_attrs, full
class _NS(dict): __getattr__ = dict.__getitem__
L = _NS(run_maps=T.run_maps, cand_index=T.cand_index, cell_obs=T.cell_obs, cell_null=T.cell_null, pooled=T.pooled, pigeonhole=T.pigeonhole)


def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
def parse_key(k): r, w = k.split('_'); return int(r[1:]) - 1, int(w[1:])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--scores'); ap.add_argument('--sets'); ap.add_argument('--attrs')
    ap.add_argument('--out'); a = ap.parse_args(); v = yaml.safe_load(open(os.path.join(ROOT, a.config))); g = v['dev_gate']
    sd_dir = os.path.join(ROOT, a.scores or v['infer_out']); setp = os.path.join(ROOT, a.sets or v['matched_sets_dev']); attp = os.path.join(ROOT, a.attrs or v['match_attrs'])
    sets = full(read_sets(setp)[1]); attrs = read_attrs(attp)
    Z, Zn = np.load(os.path.join(sd_dir, 'dev_k0_f0.npz')), np.load(os.path.join(sd_dir, 'dev_noise_k0_f0.npz'))
    sm = json.load(open(os.path.join(sd_dir, 'dev_summary.json'))); smn = json.load(open(os.path.join(sd_dir, 'dev_noise_summary.json')))
    assert sm['matched_sets']['sha256'] == sha(setp) == smn['matched_sets']['sha256'], 'dev scores were made from another matched-set file'
    subs = sm['jobs']['k0_f0']['subjects']; runs = sm['jobs']['k0_f0']['runs']; assert subs == smn['jobs']['k0_f0']['subjects']
    rows_t = {r: np.array([attrs[k][1] for k in Z[f'r{r}_rows']]) for r in runs}
    maps = L['run_maps'](rows_t, v['shift_null']['n_shift'], v['shift_null']['min_shift_s'], v['shift_null']['seed'])
    cells = {key: np.full((len(subs), len(runs)), np.nan) for key in ('tts_fixed', 'tts_lse0.5', 'null_fixed', 'null_lse0.5', 'rec_fixed', 'rec_lse0.5',
                                                                     'noise_fixed', 'noise_lse0.5', 'noise_null_fixed', 'noise_null_lse0.5', 'unmatched_fixed', 'unmatched_lse0.5')}
    w = np.zeros(len(runs)); kept = []; n_drop_u = 0
    for jr, r in enumerate(runs):
        rows, cols = list(Z[f'r{r}_rows']), list(Z[f'r{r}_cols']); assert rows == list(Zn[f'r{r}_rows']) and cols == list(Zn[f'r{r}_cols'])
        C = L['cand_index'](rows, cols, sets); col_t = np.array([attrs[k][1] for k in cols]); col_same = np.array([parse_key(k)[0] == r - 1 for k in cols])
        um = T.unmatched(rows, attrs, rows_t[r], v['density']['unmatched_seed'], r); keep_u = np.array([um[k] is not None for k in rows])
        Cu = L['cand_index']([k for k in rows if um[k] is not None], cols, sets, extra=um); w[jr] = len(rows); n_drop_u += int((~keep_u).sum())
        for js, s in enumerate(subs):
            for rule in ('fixed', 'lse0.5'):
                S = Z[f'tts_{rule}_s{s}_r{r}'].astype(np.float64); Sn = Zn[f'tts_{rule}_s{s}_r{r}'].astype(np.float64)
                cells[f'tts_{rule}'][js, jr] = L['cell_obs'](S, C).mean()
                nl, kp, kc = L['cell_null'](S, C, rows_t[r], col_t, col_same, maps[r], v['exclusion_s']); cells[f'null_{rule}'][js, jr] = nl.mean(); kept.append((kp, kc))
                cells[f'noise_{rule}'][js, jr] = L['cell_obs'](Sn, C).mean()
                cells[f'noise_null_{rule}'][js, jr] = L['cell_null'](Sn, C, rows_t[r], col_t, col_same, maps[r], v['exclusion_s'])[0].mean()
                cells[f'rec_{rule}'][js, jr] = L['cell_obs'](Z[f'rec_{rule}_s{s}_r{r}'].astype(np.float64), C).mean()
                cells[f'unmatched_{rule}'][js, jr] = L['cell_obs'](S[keep_u], Cu).mean()
    P = {k: L['pooled'](x, w) for k, x in cells.items()}; dsel = P['tts_lse0.5'] - P['tts_fixed']
    rule = 'lse0.5' if dsel >= v['align']['select_margin'] else 'fixed'
    ex = cells[f'tts_{rule}'] - cells[f'null_{rule}']; nex = cells[f'noise_{rule}'] - cells[f'noise_null_{rule}']
    nb = L['pigeonhole'](nex, w, 1000, g['seed']); ex_pt = L['pooled'](ex, w)
    gate = dict(i=bool(ex_pt >= g['excess_min']), ii=bool(np.percentile(nb, 95) < g['noise_upper']))
    D = cells[f'rec_{rule}'] - cells[f'tts_{rule}']; dm = D.mean(1)
    res = dict(rule_selection=dict(mrp_fixed=P['tts_fixed'], mrp_lse=P['tts_lse0.5'], diff=dsel, margin=v['align']['select_margin'], rule=rule),
               gate=dict(excess=ex_pt, excess_min=g['excess_min'], noise_excess=L['pooled'](nex, w), noise_excess_ci90=[float(np.percentile(nb, 5)), float(np.percentile(nb, 95))],
                         noise_upper=g['noise_upper'], conditions=gate, passed=all(gate.values()),
                         decision='open the test (rule frozen by an amendment)' if all(gate.values()) else 'development-stage futility on density-matched text candidates'),
               report={k: P[k] for k in P} | dict(excess_fixed=L['pooled'](cells['tts_fixed'] - cells['null_fixed'], w), excess_lse=L['pooled'](cells['tts_lse0.5'] - cells['null_lse0.5'], w),
                                                  null_kept_pair_fraction=float(np.mean([x[0] for x in kept])), null_kept_candidate_fraction=float(np.mean([x[1] for x in kept])),
                                                  unmatched_rows_dropped=n_drop_u),
               mde_components=dict(rule=rule, D_pooled=L['pooled'](D, w), sigma_subj=float(np.std(dm, ddof=1)), sigma_cell=float(np.std(D - dm[:, None], ddof=1)),
                                   n_subjects=len(subs), n_runs=len(runs)),
               cells={k: x.tolist() for k, x in cells.items()}, subjects=subs, runs=runs, run_weights=w.tolist(),
               inputs=dict(matched_sets_dev=dict(file=os.path.relpath(setp, ROOT), sha256=sha(setp)), match_attrs=dict(file=os.path.relpath(attp, ROOT), sha256=sha(attp)),
                           dev=sm['jobs']['k0_f0']['sha256'], dev_noise=smn['jobs']['k0_f0']['sha256'], device=sm['device']),
               note='development values are proxies and are not reported as effect estimates (D-109 section 7)')
    out = os.path.join(ROOT, a.out or 'results/t2/dev_stage.json'); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('rule_selection', 'gate')}), flush=True)


if __name__ == '__main__':
    main()
