"""Fairness control for reports/phase12_full_kr.md §3: run the DE baseline through the SAME read-outs as
the model (shrinkage LDA and a small MLP), LOSO with per-subject centring. Without this, comparing the
model's MLP read-out to the baseline's LDA read-out would credit the read-out to the model.
Usage: python scripts/de_readout_variants.py [--datasets faced,seediv,seedvii,deap]
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
import sys, json, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.eval.readout import subject_center, bootstrap_ci
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

FITS = {'lda': lambda a, b, c: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(a, b).decision_function(c),
        'mlp': lambda a, b, c: MLPClassifier((64,), alpha=1.0, max_iter=300, random_state=0).fit(a, b).predict_proba(c)[:, 1]}

ap = argparse.ArgumentParser(); ap.add_argument('--datasets', default='faced,seediv,seedvii,deap'); a = ap.parse_args()
out = {}
for nm in a.datasets.split(','):
    Z = np.load(os.path.join(ROOT, f'results/features/{nm}_de.npy')); meta = Cache(os.path.join(ROOT, 'data/cache'), nm).meta
    subj = meta['subject'].to_numpy(); m = meta['valence'].to_numpy() != 0
    y = (meta['valence'].to_numpy()[m] > 0).astype(int); X = subject_center(Z, subj)[m]; s = subj[m]
    out[nm] = {}
    for tag, fit in FITS.items():
        per = []
        for sb in np.unique(s):
            te, tr = s == sb, s != sb
            if len(np.unique(y[te])) < 2: continue
            per.append(float(roc_auc_score(y[te], fit(X[tr], y[tr], X[te]))))
        arr = np.array(per); out[nm][tag] = dict(mean=float(arr.mean()), ci95=bootstrap_ci(arr), n=len(arr))
        print(f"[{nm}] DE+{tag} AUC {arr.mean():.3f} [{out[nm][tag]['ci95'][0]:.3f}, {out[nm][tag]['ci95'][1]:.3f}]", flush=True)
json.dump(out, open(os.path.join(ROOT, 'results/baselines/de_readout_variants.json'), 'w'), indent=1)
