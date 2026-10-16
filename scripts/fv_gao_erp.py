"""D-113 4.1 and 5.5 (label-free): observation-onset check and the pipeline control R-b1 on the cached Gao runs.

1. ERP check: per trial, the mean of O1, Oz, O2 on [-7.0, 1.0) s around the event m, baseline [-6.2, -6.0). Subject ERP =
   mean over all the subject's trials; grand average = mean over subjects. Peak = first local maximum in the search range
   (-6.0, -5.0] whose value is positive and >= rel_height x the largest positive value in that range. Pass when the peak lies
   in [-5.95, -5.75]; otherwise shift = peak - expected (-5.90), applied once to the observation-anchored windows.
2. Pipeline control R-b1 (after the shift): per subject, d = sd over time of the trial-mean ERP in obs1 [-6, -5) minus the
   same in fixation [-7, -6) (each window's ERP has its own mean removed). Null: each trial's two window segments swapped at
   random (n_perm, seed); a subject passes when d > the q-quantile of the null. The control passes with >= min_subjects.
No label is read. Output: results/gao_imagery/erp_check.json
usage (gpuq): env/bin/python -I scripts/fv_gao_erp.py configs/fv_gao.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import glob, json, sys
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P = lambda p: os.path.join(ROOT, p)
c = yaml.safe_load(open(P(sys.argv[1]))); e = c['erp']; pc = c['pipeline_control']; cache = P(c['cache']); fs = c['preprocess']['sfreq']
t0, t1 = -7.0, 1.0; n = int(round((t1 - t0) * fs)); times = t0 + np.arange(n) / fs
bl = (times >= e['baseline'][0] - 1e-9) & (times < e['baseline'][1] - 1e-9)
per_sub = {}
for js in sorted(glob.glob(os.path.join(cache, 'sub-*.json'))):
    meta = json.load(open(js)); X = np.load(js[:-5] + '.npy', mmap_mode='r'); ci = [meta['ch_names'].index(ch) for ch in e['channels']]
    occ = np.asarray(X[ci]).mean(0); sub = os.path.basename(js).split('_')[0]
    ep = np.stack([occ[m + int(round(t0 * fs)): m + int(round(t0 * fs)) + n] for m in meta['m']])
    per_sub.setdefault(sub, []).append(ep - ep[:, bl].mean(1, keepdims=True))
subs = sorted(per_sub); E = {s: np.concatenate(per_sub[s]) for s in subs}
erp_sub = np.stack([E[s].mean(0) for s in subs]); ga = erp_sub.mean(0)
sr = (times > e['search'][0] + 1e-9) & (times <= e['search'][1] + 1e-9); idx = np.flatnonzero(sr)
top = ga[idx].max(); peak = None
for i in idx:
    if 0 < i < n - 1 and ga[i] > ga[i - 1] and ga[i] >= ga[i + 1] and ga[i] > 0 and ga[i] >= e['rel_height'] * top:
        peak = float(times[i]); break
ok = peak is not None and e['pass'][0] - 1e-9 <= peak <= e['pass'][1] + 1e-9
shift = 0.0 if ok or peak is None else round(peak - e['expected'], 3)
res = dict(n_subjects=len(subs), n_trials=int(sum(len(E[s]) for s in subs)), peak_s=peak, peak_uV=None if peak is None else float(ga[times == peak][0] * 1e6),
           search_max_uV=float(top * 1e6), passed=bool(ok), shift_s=shift, note='shift applies once to the observation-anchored windows' if shift else '',
           grand_average_uV=dict(t0=t0, sfreq=fs, values=[round(float(v * 1e6), 4) for v in ga]))
# per-subject peaks (report only)
res['subject_peaks_s'] = {}
for s, w in zip(subs, erp_sub):
    pk = None; tp = w[idx].max()
    for i in idx:
        if w[i] > w[i - 1] and w[i] >= w[i + 1] and w[i] > 0 and w[i] >= e['rel_height'] * tp: pk = float(times[i]); break
    res['subject_peaks_s'][s] = pk
# pipeline control R-b1
win = c['windows']['obs_anchored']; rng = np.random.default_rng(pc['seed'])
sel = lambda w: (times >= w[0] + shift - 1e-9) & (times < w[1] + shift - 1e-9)
A, B = sel(win['obs1']), sel(win['fixation']); assert A.sum() == B.sum()
size = lambda Z: (Z - Z.mean()).std()
rb1 = {}
for s in subs:
    a, b = E[s][:, A], E[s][:, B]; d = size(a.mean(0)) - size(b.mean(0)); null = np.empty(pc['n_perm'])
    for k in range(pc['n_perm']):
        sw = rng.random(len(a)) < 0.5; aa = np.where(sw[:, None], b, a); bb = np.where(sw[:, None], a, b); null[k] = size(aa.mean(0)) - size(bb.mean(0))
    rb1[s] = dict(d_uV=float(d * 1e6), null_q_uV=float(np.quantile(null, pc['q']) * 1e6), passed=bool(d > np.quantile(null, pc['q'])))
npass = sum(v['passed'] for v in rb1.values())
res['pipeline_control'] = dict(subjects=rb1, n_pass=int(npass), min_subjects=pc['min_subjects'], passed=bool(npass >= pc['min_subjects']))
out = P(os.path.join(c['out'], 'erp_check.json')); json.dump(res, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('grand_average_uV', 'subject_peaks_s')}, indent=1)[:3000])
