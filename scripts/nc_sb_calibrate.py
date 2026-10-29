"""D-128 SB strength calibration (docs/08 §7 H1, §13 C7). No real label is read and no compared model is involved.

usage: env/bin/python scripts/nc_sb_calibrate.py configs/nc_train.yaml [--smoke]

Background = PhysioNet-MI task windows (as scripts/nc_train.py loads them) with random balanced labels drawn from
sb_params.cal_seed (different from the benchmark label seed). Each component alone is added at amplitudes amp x
background RMS on a grid; the linear readout (D-126 D1 100 ms bin means + D2 log-variance in the 6 N1 bands, standardised,
shrinkage LDA) is scored by subject 5-fold (the benchmark fold assignment); kappa pooled over test folds. The amplitude
giving the target kappa is interpolated on log amp and checked once. Writes results/nc/sb_calibration.json.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, copy, argparse
import numpy as np, yaml, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.reve_nc import band_split
from ese.sb.generate import SB
from nc_train import TaskData, subject_folds


def features(X, bands, chunk=400):
    out = []
    for i in range(0, len(X), chunk):
        x = torch.from_numpy(X[i:i + chunk]).double(); n, C, L = x.shape
        d1 = x.reshape(n, C, L // 20, 20).mean(-1).reshape(n, -1)                   # 100 ms bins at 200 Hz
        d2 = torch.log((band_split(x, 200, bands) ** 2).mean(-1) + 1e-12).reshape(n, -1)
        out.append(torch.cat([d1, d2], 1).float().numpy())
    return np.concatenate(out)


def kappa_cv(Fe, y, subj, folds):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import cohen_kappa_score
    pred = np.zeros_like(y)
    for te_s in folds:
        te = np.isin(subj, te_s); tr = ~te; mu, sd = Fe[tr].mean(0), Fe[tr].std(0) + 1e-8
        m = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit((Fe[tr] - mu) / sd, y[tr]); pred[te] = m.predict((Fe[te] - mu) / sd)
    return float(cohen_kappa_score(y, pred))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    c = yaml.safe_load(open(os.path.join(ROOT, a.config))); p = c['sb_params']; cal = p['calibration']
    c2 = copy.deepcopy(c); c2['tasks']['physionet_mi']['null'] = False          # task windows only (memory)
    t0 = time.time(); d = TaskData(c2, 'physionet_mi', torch.device('cpu'))
    if a.smoke:
        keep = np.isin(d.subj, np.arange(20)); d.x, d.subj, d.t_order = d.x[keep], d.subj[keep], d.t_order[keep]
    g = SB(d.x, d.names, d.subj, d.t_order, d.K, 200, p['cal_seed']); y = g.labels(p['cal_seed'])
    folds = subject_folds(sorted(set(d.subj.tolist())), 5, c['seed']); bands = d.bands
    out = dict(cal_seed=p['cal_seed'], n_windows=int(len(y)), n_subjects=int(len(set(d.subj.tolist()))), bg_rms=g.bg_rms,
               readout='D1 100 ms bin means + D2 log-variance (6 bands), standardised, shrinkage LDA (lsqr, Ledoit-Wolf)',
               load_seconds=round(time.time() - t0, 1), components={})
    out['kappa_background'] = kappa_cv(features(d.x, bands), y, d.subj, folds); print(json.dumps(dict(background=out['kappa_background'])), flush=True)
    for kind, target in cal['targets'].items():
        tk = time.time(); comp, _ = g.render(kind, y); s = g.norm(kind, y); grid = []
        for amp in cal['grid']:
            k = kappa_cv(features(d.x + (amp * s) * comp, bands), y, d.subj, folds); grid.append((amp, k))
            print(json.dumps(dict(kind=kind, amp=amp, kappa=round(k, 4))), flush=True)
            if k > 2 * target: break
        A = np.log([g_[0] for g_ in grid]); K_ = np.array([g_[1] for g_ in grid]); i = int(np.searchsorted(K_, target))
        if 0 < i < len(K_): amp = float(np.exp(A[i - 1] + (target - K_[i - 1]) * (A[i] - A[i - 1]) / (K_[i] - K_[i - 1])))
        else: amp = None
        chk = kappa_cv(features(d.x + (amp * s) * comp, bands), y, d.subj, folds) if amp else None
        out['components'][kind] = dict(target=target, grid=grid, amp=amp, kappa_at_amp=chk, seconds=round(time.time() - tk, 1))
        print(json.dumps({kind: out['components'][kind]}), flush=True); del comp
    out['seconds'] = round(time.time() - t0, 1)
    dst = os.path.join(ROOT, 'results', 'nc', 'sb_calibration' + ('_smoke' if a.smoke else '') + '.json'); os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, 'w'), indent=1); print(json.dumps(dict(written=dst, seconds=out['seconds'])), flush=True)


if __name__ == '__main__':
    main()
