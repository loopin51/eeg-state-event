"""Phase 0.2: FBCSP (9 bands 4-40 Hz, 4 CSP pairs/band, log-variance) + shrinkage-LDA, LOSO,
binary valence. CSP filters are fit on the pooled training subjects with per-window
trace-normalized covariances (a label-free per-subject scale normalization).
Writes results/baselines/<dataset>_fbcsp.json.
"""
import os
# Cap BLAS/OpenMP thread pools BEFORE numpy/torch import. The read-out battery runs thousands of small
# LDA / logistic fits; unconstrained OpenBLAS (MAX_THREADS=64) grabs every core and spends most of it
# spinning in thread-pool synchronisation. 8 threads is as fast and leaves the shared server usable.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import os, sys, json, argparse, time, numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.linalg import eigh
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.eval.readout import bootstrap_ci
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]
N_PAIRS = 4

def band_covs(windows, sfreq, idx, batch=256):
    """-> covs [len(BANDS), n, C, C] trace-normalized."""
    C = windows.shape[1]; covs = np.zeros((len(BANDS), len(idx), C, C), np.float32)
    sos = [butter(4, b, btype='band', fs=sfreq, output='sos') for b in BANDS]
    for i in range(0, len(idx), batch):
        x = np.asarray(windows[idx[i:i + batch]], dtype=np.float32)
        for bi, s in enumerate(sos):
            y = sosfiltfilt(s, x, axis=-1)
            cv = np.einsum('nct,ndt->ncd', y, y) / y.shape[-1]
            cv /= np.trace(cv, axis1=1, axis2=2)[:, None, None]
            covs[bi, i:i + batch] = cv
    return covs

def csp_filters(cov_a, cov_b, n_pairs=N_PAIRS):
    Ca, Cb = cov_a.mean(0), cov_b.mean(0)
    w, V = eigh(Ca, Ca + Cb + 1e-6 * np.eye(len(Ca)))
    order = np.argsort(w)
    return V[:, np.r_[order[:n_pairs], order[-n_pairs:]]]           # [C, 2*n_pairs]

def feats(covs, W):   # covs [n,C,C], W [C,k] -> log-var [n,k]
    v = np.einsum('ck,ncd,dk->nk', W, covs, W)
    return np.log(v / v.sum(1, keepdims=True))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ap.add_argument('--cache', default=os.path.join(ROOT, 'data/cache'))
    a = ap.parse_args(); t0 = time.time()
    c = Cache(a.cache, a.dataset); meta = c.meta
    m = np.where(meta.valence.to_numpy() != 0)[0]
    y = (meta.valence.to_numpy()[m] > 0).astype(int); subj = meta.subject.to_numpy()[m]
    covs = band_covs(c.windows, c.header['sfreq'], m)
    print(f'[{a.dataset}] covariances {covs.shape} in {time.time()-t0:.0f}s', flush=True)
    per = {}
    for s in np.unique(subj):
        te, tr = subj == s, subj != s
        if len(np.unique(y[te])) < 2: continue
        Ftr, Fte = [], []
        for bi in range(len(BANDS)):
            W = csp_filters(covs[bi][tr][y[tr] == 1], covs[bi][tr][y[tr] == 0])
            Ftr.append(feats(covs[bi][tr], W)); Fte.append(feats(covs[bi][te], W))
        Ftr, Fte = np.concatenate(Ftr, 1), np.concatenate(Fte, 1)
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ftr, y[tr])
        per[int(s)] = float(roc_auc_score(y[te], clf.decision_function(Fte)))
    arr = np.array(list(per.values()))
    out = dict(dataset=a.dataset, method='FBCSP 9 bands x 4 pairs + shrinkage LDA, LOSO', per_subject=per,
               mean=float(arr.mean()), std=float(arr.std()), ci95=bootstrap_ci(arr), n_subjects=len(arr), seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.join(ROOT, 'results/baselines'), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, 'results/baselines', f'{a.dataset}_fbcsp.json'), 'w'), indent=1)
    print(f"[{a.dataset}] FBCSP LOSO valence AUC {out['mean']:.3f} {out['ci95']}", flush=True)

if __name__ == '__main__':
    main()
