"""Chisco trial-mixed split (official `index % 5 == 1`) vs day-held-out split: leakage size (D-052, diagnostic).

MIX: per subject, raw_ok trials sorted by (run, onset time); every trial whose position % 5 == 1 is a test trial, the rest
calibrate. Same feature sets and shrinkage LDA as scripts/chisco_calib.py. Primary = uncentred (the official pipeline does
not centre); day-centred is secondary. ADJ copies the label of the calibration trial nearest in time within the same run
(no EEG); its chance threshold comes from circularly shifting the test labels within each run.
DAY values are read from results/chisco_calib.json (D-048).

usage: python scripts/chisco_mixed_split.py OUT_JSON [--perms 100]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.eval.readout import bootstrap_ci
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, accuracy_score

FEATS = os.path.join(ROOT, 'data/cache/chisco_feats')
SETS = ['DE_IMAG', 'DE_READ', 'EMG', 'EOG', 'PRE', 'DC', 'ORDER', 'LEN']; SUBJECTS = [1, 2, 3, 4, 5]


def center(X, groups):
    X = X.astype(np.float64).copy()
    for g in np.unique(groups): X[groups == g] -= X[groups == g].mean(0)
    return X


def lda_pred(Xtr, ytr, Xte):
    cls = np.unique(ytr)
    return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=np.ones(len(cls)) / len(cls)).fit(Xtr, ytr).predict(Xte)


def adj_pred(run, t, ycal, cal, te):
    """label of the calibration trial nearest in onset time within the same run."""
    out = np.empty(len(te), int)
    for i, j in enumerate(te):
        c = cal[run[cal] == run[j]]
        out[i] = ycal[np.searchsorted(cal, c[np.argmin(np.abs(t[c] - t[j]))])]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out_json'); ap.add_argument('--perms', type=int, default=100); a = ap.parse_args()
    t0 = time.time(); res = dict(perms=a.perms, subjects={}); rows = ['ADJ'] + [f'{f}|raw' for f in SETS] + [f'{f}|centred' for f in SETS]
    null = {r: np.zeros((len(SUBJECTS), a.perms)) for r in rows}
    for si, sub in enumerate(SUBJECTS):
        Z = np.load(os.path.join(FEATS, f'sub-{sub:02d}.npz')); m = Z['raw_ok']
        run, t, y, day = Z['run'][m], Z['ORDER'][m][:, 0].astype(np.float64), Z['label'][m], Z['day'][m]
        order = np.lexsort((t, run)); pos = np.empty(len(order), int); pos[order] = np.arange(len(order))
        te, cal = np.where(pos % 5 == 1)[0], np.where(pos % 5 != 1)[0]; r = dict(n_test=int(len(te)), n_cal=int(len(cal)))
        rng = np.random.default_rng(7 + sub)
        p = adj_pred(run, t, y[cal], cal, te)
        r['ADJ'] = dict(bacc=float(balanced_accuracy_score(y[te], p)), acc=float(accuracy_score(y[te], p)))
        for q in range(a.perms):   # circular shift of the test labels within each run
            ys = y[te].copy()
            for rr in np.unique(run[te]):
                k = np.where(run[te] == rr)[0]
                if len(k) > 1: ys[k] = np.roll(ys[k], rng.integers(1, len(k)))
            null['ADJ'][si, q] = balanced_accuracy_score(ys, p)
        for f in SETS:
            for mode in ('raw', 'centred'):
                X = Z[f][m].astype(np.float64) if mode == 'raw' else center(Z[f][m], day)
                p = lda_pred(X[cal], y[cal], X[te])
                r[f'{f}|{mode}'] = dict(bacc=float(balanced_accuracy_score(y[te], p)), acc=float(accuracy_score(y[te], p)))
                for q in range(a.perms):
                    null[f'{f}|{mode}'][si, q] = balanced_accuracy_score(y[te], lda_pred(X[cal], rng.permutation(y[cal]), X[te]))
            print(f'sub-{sub:02d} {f:8s} raw {r[f"{f}|raw"]["bacc"]:.4f} centred {r[f"{f}|centred"]["bacc"]:.4f} '
                  f'| ADJ {r["ADJ"]["bacc"]:.4f} ({time.time() - t0:.0f}s)', flush=True)
        res['subjects'][sub] = r
        json.dump(res, open(a.out_json, 'w'), indent=1)
    day = json.load(open(os.path.join(ROOT, 'results/chisco_calib.json')))
    summary = {}
    for rname in rows:
        v = np.array([res['subjects'][s][rname]['bacc'] for s in SUBJECTS]); acc = np.array([res['subjects'][s][rname]['acc'] for s in SUBJECTS])
        summary[rname] = dict(mean=float(v.mean()), ci95=bootstrap_ci(v), per_subject=v.round(4).tolist(), acc_mean=float(acc.mean()),
                              chance_p95=float(np.quantile(null[rname].mean(0), 0.95)), null_mean=float(null[rname].mean()))
    for f in ['DE_IMAG', 'DE_READ']:
        dd = np.array([day['subjects'][str(s)][f]['curve']['all'] for s in SUBJECTS])
        mx = np.array(summary[f'{f}|centred']['per_subject']); mr = np.array(summary[f'{f}|raw']['per_subject'])
        summary[f'leak_{f}'] = dict(day=dd.round(4).tolist(), mix_raw_minus_day=dict(mean=float((mr - dd).mean()), ci95=bootstrap_ci(mr - dd)),
                                    mix_centred_minus_day=dict(mean=float((mx - dd).mean()), ci95=bootstrap_ci(mx - dd)))
    res['summary'] = summary
    json.dump(res, open(a.out_json, 'w'), indent=1)
    for k, v in summary.items():
        if k.startswith('leak_'): print(k, v, flush=True)
        else: print(f'{k:16s} bacc {v["mean"]:.4f} {np.round(v["ci95"], 4)} acc {v["acc_mean"]:.4f} | per-subject {v["per_subject"]} | chance p95 {v["chance_p95"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
