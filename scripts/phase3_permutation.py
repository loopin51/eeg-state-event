"""Permutation control for Phase 3.0 cross-subject accuracies (added after D-035 results; sanity check).
Labels are permuted within each subject (class balance per subject preserved); the full LOSO LDA pipeline is re-run
per permutation. p = fraction of null mean accuracies >= observed.
Usage: python scripts/phase3_permutation.py OUT_JSON [--n 200] [--ckpt PATH]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, argparse, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.data.cache import Cache
from ese.eval.features import de_features
from ese.eval.readout import subject_center
from phase3_feasibility import frozen_features, lda_pred

ap = argparse.ArgumentParser(); ap.add_argument('out_json'); ap.add_argument('--n', type=int, default=200)
ap.add_argument('--ckpt', default=os.path.join(ROOT, 'results/runs/20260922_2135_clisa_align_only/best.pt')); a = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
c = Cache(os.path.join(ROOT, 'data/cache'), 'bci2020'); m = c.meta; y = m['label'].to_numpy(); subj = m['subject'].to_numpy()
feats = {'DE': de_features(c.windows, sfreq=100, nperseg=100), 'frozen': frozen_features(a.ckpt, c.windows, c.positions, dev)}
np.save(os.path.join(ROOT, 'results/features/bci2020_de.npy'), feats['DE']); np.save(os.path.join(ROOT, 'results/features/bci2020_frozen.npy'), feats['frozen'])
def loso(X, yy): return float(np.mean([(lda_pred(X[subj != s], yy[subj != s], X[subj == s]) == yy[subj == s]).mean() for s in np.unique(subj)]))
rng = np.random.default_rng(0); out = {}
for name, X in feats.items():
    Xc = subject_center(X, subj); obs = loso(Xc, y); null = []
    for _ in range(a.n):
        yp = y.copy()
        for s in np.unique(subj): k = subj == s; yp[k] = rng.permutation(y[k])
        null.append(loso(Xc, yp))
    null = np.array(null); out[name] = dict(observed=obs, null_mean=float(null.mean()), null_p95=float(np.quantile(null, .95)), null_p99=float(np.quantile(null, .99)), p=float((null >= obs).mean()), n=a.n)
    print(f"[{name}] observed {obs:.4f} | null mean {null.mean():.4f} p95 {np.quantile(null, .95):.4f} p99 {np.quantile(null, .99):.4f} | p = {out[name]['p']:.3f}", flush=True)
json.dump(out, open(a.out_json, 'w'), indent=1)
