"""Is the valence information present but not linearly readable? Compares read-outs on the SAME saved
representations, all LOSO with per-subject centring:
  lda_mean   : shrinkage LDA on the window-mean of z_state          (what the gates use)
  lr_traj    : logistic regression on the flattened z_state(tau) trajectory (6x32 = 192 dims)
  mlp_mean   : one-hidden-layer MLP on the window mean
  lda_event  : shrinkage LDA on z_event mean (reference)
Usage: python scripts/readout_variants.py RUN_DIR [--datasets seediv,faced,seedvii]
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
import sys, json, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.eval.readout import subject_center, bootstrap_ci
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

def loso(X, y, subj, fit):
    per = {}
    Xc = subject_center(X, subj)
    for s in np.unique(subj):
        te, tr = subj == s, subj != s
        if len(np.unique(y[te])) < 2: continue
        per[int(s)] = float(roc_auc_score(y[te], fit(Xc[tr], y[tr], Xc[te])))
    a = np.array(list(per.values()))
    return dict(mean=float(a.mean()), ci95=bootstrap_ci(a), n=len(a))

FITS = {
    'lda':  lambda Xtr, ytr, Xte: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xtr, ytr).decision_function(Xte),
    'lr':   lambda Xtr, ytr, Xte: LogisticRegression(C=0.1, max_iter=1000).fit(Xtr, ytr).decision_function(Xte),
    'mlp':  lambda Xtr, ytr, Xte: MLPClassifier((64,), alpha=1.0, max_iter=300, random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1],
}

ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--datasets', default='seediv,faced,seedvii,deap')
a = ap.parse_args(); out = {}
for nm in a.datasets.split(','):
    p = os.path.join(a.run, f'z_{nm}.npz')
    if not os.path.exists(p): continue
    d = np.load(p); Zs, Ze = d['z_state'], d['z_event_mean']
    meta = Cache(os.path.join(ROOT, 'data/cache'), nm).meta
    subj = meta['subject'].to_numpy(); m = meta['valence'].to_numpy() != 0
    y = (meta['valence'].to_numpy()[m] > 0).astype(int)
    variants = dict(lda_mean=(Zs.mean(1)[m], 'lda'), lr_traj=(Zs.reshape(len(Zs), -1)[m], 'lr'),
                    mlp_mean=(Zs.mean(1)[m], 'mlp'), lda_event=(Ze[m], 'lda'))
    out[nm] = {}
    for k, (X, fit) in variants.items():
        r = loso(X, y, subj[m], FITS[fit]); out[nm][k] = r
        print(f"[{nm}] {k:10s} AUC {r['mean']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]", flush=True)
json.dump(out, open(os.path.join(a.run, 'readout_variants.json'), 'w'), indent=1)
