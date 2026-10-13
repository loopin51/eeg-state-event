"""Diagnostic for L_stim-align learnability: can a LINEAR model on DE features identify WHICH clip a
window comes from, for a held-out subject? (LOSO, subject-centred, nearest-class-mean / LDA.)
If this is near chance, cross-subject stimulus alignment has little signal to latch onto in that dataset;
if well above chance, the align objective is learnable and the pilot's flat align loss is an optimisation
/ batch-composition problem. Usage: python scripts/diagnose_clip_id.py seediv deap faced
"""
import os
# Cap BLAS/OpenMP thread pools BEFORE numpy/torch import. The read-out battery runs thousands of small
# LDA / logistic fits; unconstrained OpenBLAS (MAX_THREADS=64) grabs every core and spends most of it
# spinning in thread-pool synchronisation. 8 threads is as fast and leaves the shared server usable.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import os, sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.eval.readout import subject_center, bootstrap_ci
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

out = {}
for nm in sys.argv[1:]:
    c = Cache(os.path.join(ROOT, 'data/cache'), nm); m = c.meta
    Z = subject_center(np.load(os.path.join(ROOT, f'results/features/{nm}_de.npy')), m.subject.to_numpy())
    res = {}
    for sess in sorted(m.session.unique()):
        k = (m.session == sess).to_numpy(); Zs, clip, subj = Z[k], m.clip_id.to_numpy()[k], m.subject.to_numpy()[k]
        accs = []
        for s in np.unique(subj):
            te, tr = subj == s, subj != s
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Zs[tr], clip[tr])
            accs.append(float((clf.predict(Zs[te]) == clip[te]).mean()))
        accs = np.array(accs); n_clip = len(np.unique(clip))
        res[f'session{sess}'] = dict(acc=float(accs.mean()), ci95=bootstrap_ci(accs), chance=1 / n_clip, n_clips=int(n_clip), n_subjects=int(len(accs)))
        print(f"[{nm}] session {sess}: clip-ID LOSO acc {accs.mean():.3f} {np.round(bootstrap_ci(accs),3)} (chance {1/n_clip:.3f}, {n_clip} clips, {len(accs)} subj)", flush=True)
    out[nm] = res
p = os.path.join(ROOT, 'results/diagnosis_clip_id.json'); prev = json.load(open(p)) if os.path.exists(p) else {}; prev.update(out); json.dump(prev, open(p, 'w'), indent=1)
