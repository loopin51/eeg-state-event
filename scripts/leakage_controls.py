"""Leakage controls for stimulus-aligned representations (docs/05 G2-2).
When valence is a function of clip identity and every subject sees the same clips, a representation that
merely identifies clips looks like an emotion decoder under subject-LOSO. Two controls, same read-out
(subject-centred shrinkage LDA) for every feature set:
  unseen_subjects : LOSO folds restricted to subjects Stage A never saw (split.json 'val')
  clip_holdout    : GroupKFold over clips (valence-stratified); train on some clips, test on unseen clips
Feature sets: DE (baseline), and z_state window means from each RUN given.
Usage: python scripts/leakage_controls.py RUN_DIR [RUN_DIR ...]
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
import sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.eval.readout import subject_center, bootstrap_ci
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

def lda(a, b, c): return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(a, b).decision_function(c)

def subject_folds(X, y, subj, only=None):
    per = []
    for s in np.unique(subj):
        if only is not None and s not in only: continue
        te, tr = subj == s, subj != s
        if len(np.unique(y[te])) < 2: continue
        per.append(float(roc_auc_score(y[te], lda(X[tr], y[tr], X[te]))))
    a = np.array(per); return dict(mean=float(a.mean()) if len(a) else None, ci95=bootstrap_ci(a) if len(a) > 1 else None, n=len(a))

def clip_folds(X, y, clip, k=6, seed=0):
    rng = np.random.default_rng(seed); pos = np.unique(clip[y == 1]); neg = np.unique(clip[y == 0])
    rng.shuffle(pos); rng.shuffle(neg); k = min(k, len(pos), len(neg))
    per = []
    for f in range(k):
        te_clips = set(pos[f::k]) | set(neg[f::k]); te = np.isin(clip, list(te_clips)); tr = ~te
        per.append(float(roc_auc_score(y[te], lda(X[tr], y[tr], X[te]))))
    a = np.array(per); return dict(mean=float(a.mean()), ci95=bootstrap_ci(a), n_folds=k)

runs = sys.argv[1:]; out = {}
for nm in ['seediv', 'deap', 'faced', 'seedvii']:
    meta = Cache(os.path.join(ROOT, 'data/cache'), nm).meta
    subj = meta['subject'].to_numpy(); m = meta['valence'].to_numpy() != 0
    y = (meta['valence'].to_numpy()[m] > 0).astype(int); clip = meta['clip_id'].to_numpy()[m]; s = subj[m]
    feats = {'DE': np.load(os.path.join(ROOT, f'results/features/{nm}_de.npy'))}
    val_subs = None
    for r in runs:
        feats[os.path.basename(r)] = np.load(os.path.join(r, f'z_{nm}.npz'))['z_state'].mean(1)
        val_subs = set(json.load(open(os.path.join(r, 'split.json')))['val'][nm])
    out[nm] = {}
    for tag, Z in feats.items():
        X = subject_center(Z, subj)[m]
        res = dict(loso_all=subject_folds(X, y, s), loso_unseen=subject_folds(X, y, s, only=val_subs), clip_holdout=clip_folds(X, y, clip))
        out[nm][tag] = res
        f = lambda d: f"{d['mean']:.3f}" if d['mean'] is not None else 'n/a'
        print(f"[{nm}] {tag:32s} LOSO-all {f(res['loso_all'])} | unseen-subj {f(res['loso_unseen'])} (n={res['loso_unseen']['n']}) | clip-holdout {f(res['clip_holdout'])}", flush=True)
json.dump(out, open(os.path.join(ROOT, 'results/leakage_controls.json'), 'w'), indent=1)
