"""Stage 0-b' (D-040): leak-controlled within-subject calibration curves on BCI2020 Track 3.

Each cue in Track 3 was followed by four consecutive imagery trials of the same word, and the released
Training/Validation sets mix these siblings. For every Validation trial we drop, from the calibration set, the r
Training trials closest to it in raw DC-offset space (per-channel means, standardized within subject, label-free),
then fit the readout. Leak meters: L1 = DC-offset 1-NN label copy, L2 = pre-cue (-0.5..0 s) DE + LDA.
r* = smallest r whose two meters both have mean <= 0.23 and a subject-bootstrap CI containing 0.2.
Methods: M1 post-cue (0..2.5 s) DE, 6 bands, P1; M1f full-epoch DE, P1; M2 frozen front end (190-subject model).
All readouts are shrinkage LDA with uniform priors.

usage: python scripts/stage0_leakctl.py OUT_JSON
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ese.eval.features import de_features, BANDS
from ese.eval.readout import bootstrap_ci
from prepare_bci2020 import load_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

K = 5; KS = [25, 50, 100, 250]; RS = [0, 3, 6, 12, 24]; N_DRAWS = 5
BANDS6 = dict(BANDS, gamma_high=(45, 95))
RAW = os.path.join(ROOT, 'data/raw/bci2020/BCI2020 EEG Signal for Words')


def fit_predict(Xtr, ytr, Xte):
    cls = np.unique(ytr)
    if len(cls) == 1: return np.full(len(Xte), cls[0])
    return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=np.ones(len(cls)) / len(cls)).fit(Xtr, ytr).predict(Xte)


def main():
    out_path = sys.argv[1]; t0 = time.time()
    d = os.path.join(ROOT, 'data/cache/bci2020_p1'); h = json.load(open(os.path.join(d, 'header.json')))
    X = np.load(os.path.join(d, 'x.npy')); meta = pd.read_parquet(os.path.join(d, 'meta.parquet'))
    y = meta['label'].to_numpy(); subj = meta['subject'].to_numpy(); sess = meta['session'].to_numpy(); fs, on = h['sfreq'], h['onset_sample']
    feats = {'M1': de_features(np.ascontiguousarray(X[:, :, on:on + int(2.5 * fs)]), sfreq=fs, bands=BANDS6, nperseg=int(fs)),
             'M1f': de_features(X, sfreq=fs, bands=BANDS6, nperseg=int(fs)),
             'M2': np.load(os.path.join(ROOT, 'results/features/bci2020_frozen.npy')),
             'L2': de_features(np.ascontiguousarray(X[:, :, :on]), sfreq=fs, bands=BANDS6, nperseg=on)}
    assert all(len(v) == len(y) for v in feats.values())
    print(f'features ready ({time.time() - t0:.0f}s)', flush=True)
    acc = {m: {r: {k: {} for k in KS} for r in RS} for m in ['M1', 'M1f', 'M2', 'L1', 'L2']}
    for s in np.unique(subj):
        s = int(s)
        xt, yt, _, _, _ = load_split(os.path.join(RAW, 'Training set', f'Data_Sample{s + 1:02d}.mat'), 'epo_train')
        xv, yv, _, _, _ = load_split(os.path.join(RAW, 'Validation set', f'Data_Sample{s + 1:02d}.mat'), 'epo_validation')
        tr = np.where((subj == s) & (sess == 0))[0]; te = np.where((subj == s) & (sess == 1))[0]
        assert (y[tr] == yt).all() and (y[te] == yv).all()
        dc = np.concatenate([xt, xv]).mean(2); dc = (dc - dc.mean(0)) / (dc.std(0) + 1e-9); dtr, dte = dc[:300], dc[300:]
        D = ((dte[:, None] - dtr[None]) ** 2).sum(-1); near = np.argsort(D, 1)                  # [50, 300] local train indices
        for k in KS:
            vals = {m: {r: [] for r in RS} for m in acc}
            for draw in range(N_DRAWS):
                rng = np.random.default_rng(10000 * s + 100 * k + draw)
                cal = np.sort(np.concatenate([rng.choice(np.where(yt == c)[0], k // K, replace=False) for c in range(K)]))
                for r in RS:
                    pred = {m: np.empty(len(te), int) for m in acc}; cache = {}
                    for j in range(len(te)):
                        keep = np.setdiff1d(cal, near[j, :r]) if r else cal
                        key = keep.tobytes()
                        if key not in cache:
                            cache[key] = {m: fit_predict(feats[m][tr[keep]], yt[keep], feats[m][te]) for m in ['M1', 'M1f', 'M2', 'L2']}
                        for m in ['M1', 'M1f', 'M2', 'L2']: pred[m][j] = cache[key][m][j]
                        pred['L1'][j] = yt[keep[np.argmin(D[j, keep])]]
                    for m in acc: vals[m][r].append(float((pred[m] == yv).mean()))
            for m in acc:
                for r in RS: acc[m][r][k][s] = float(np.mean(vals[m][r]))
        print(f'subject {s}: k=250 ' + ' | '.join(f'r={r}: L1 {acc["L1"][r][250][s]:.2f} L2 {acc["L2"][r][250][s]:.2f} M1 {acc["M1"][r][250][s]:.2f}'
                                               for r in [0, 12, 24]) + f' ({time.time() - t0:.0f}s)', flush=True)
    res = dict(ks=KS, rs=RS, n_draws=N_DRAWS, table={})
    for m in acc:
        res['table'][m] = {str(r): {str(k): dict(mean=float(np.mean(list(v.values()))), ci95=bootstrap_ci(np.array(list(v.values()))), per_subject=v)
                                    for k, v in acc[m][r].items()} for r in RS}
    # r*: leak meters at k = 250
    rstar = None
    for r in RS:
        ok = all(res['table'][m][str(r)]['250']['mean'] <= 0.23 and res['table'][m][str(r)]['250']['ci95'][0] <= 0.2 <= res['table'][m][str(r)]['250']['ci95'][1] for m in ['L1', 'L2'])
        if ok: rstar = r; break
    res['r_star'] = rstar
    if rstar is not None:
        res['primary_M1_k250'] = res['table']['M1'][str(rstar)]['250']
        a, b = acc['M2'][rstar][50], acc['M1'][rstar][50]; dd = np.array([a[k] - b[k] for k in sorted(a)])
        res['M2_minus_M1_k50'] = dict(diff=float(dd.mean()), ci95=bootstrap_ci(dd), wins=int((dd > 0).sum()), n=len(dd))
    json.dump(res, open(out_path, 'w'), indent=1)
    for m in acc:
        for r in RS:
            print(f'{m:3s} r={r:2d} ' + ' '.join(f'k={k}: {res["table"][m][str(r)][str(k)]["mean"]:.3f} {np.round(res["table"][m][str(r)][str(k)]["ci95"], 3)}' for k in KS))
    print('r* =', rstar, '| primary', res.get('primary_M1_k250', {}).get('mean'), res.get('primary_M1_k250', {}).get('ci95'), '| M2-M1 k50', res.get('M2_minus_M1_k50'))


if __name__ == '__main__':
    main()
