"""Phase 0.2: DE (5 bands x channels) + shrinkage-LDA / logistic-regression baselines under LOSO,
with label-shuffle floor, subject-bootstrap CI, ISC test and circular-shift read-out on the
DE-derived valence score. Writes results/baselines/<dataset>_de.json (+ DE feature cache).
Usage: python scripts/baseline_de_lr.py --dataset seediv|deap
"""
import os
# Cap BLAS/OpenMP thread pools BEFORE numpy/torch import. The read-out battery runs thousands of small
# LDA / logistic fits; unconstrained OpenBLAS (MAX_THREADS=64) grabs every core and spends most of it
# spinning in thread-pool synchronisation. 8 threads is as fast and leaves the shared server usable.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import os, sys, json, argparse, time, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.eval.features import de_features
from ese.eval.readout import loso_auc, subject_center, isc_test, circular_shift_labels, bootstrap_ci
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

def loso_logreg(Z, y, subject, C=0.1):
    Zc = subject_center(Z, subject); per = {}
    for s in np.unique(subject):
        te, tr = subject == s, subject != s
        if len(np.unique(y[te])) < 2: continue
        clf = LogisticRegression(C=C, max_iter=2000).fit(Zc[tr], y[tr])
        per[int(s)] = float(roc_auc_score(y[te], clf.decision_function(Zc[te])))
    a = np.array(list(per.values())); return dict(per_subject=per, mean=float(a.mean()), ci95=bootstrap_ci(a), n_subjects=len(a))

def loso_multiclass(Z, y, subject, C=0.1):
    Zc = subject_center(Z, subject); per = {}
    for s in np.unique(subject):
        te, tr = subject == s, subject != s
        clf = LogisticRegression(C=C, max_iter=2000).fit(Zc[tr], y[tr])
        per[int(s)] = float(accuracy_score(y[te], clf.predict(Zc[te])))
    a = np.array(list(per.values())); return dict(per_subject=per, mean=float(a.mean()), ci95=bootstrap_ci(a), chance=float(1 / len(np.unique(y))))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ap.add_argument('--cache', default=os.path.join(ROOT, 'data/cache'))
    a = ap.parse_args(); t0 = time.time()
    c = Cache(a.cache, a.dataset); meta = c.meta
    fdir = os.path.join(ROOT, 'results/features'); os.makedirs(fdir, exist_ok=True)
    fpath = os.path.join(fdir, f'{a.dataset}_de.npy')
    if os.path.exists(fpath): Z = np.load(fpath)
    else: Z = de_features(c.windows, sfreq=c.header['sfreq']); np.save(fpath, Z)
    print(f'[{a.dataset}] DE features {Z.shape} in {time.time()-t0:.0f}s', flush=True)
    out = dict(dataset=a.dataset, n_windows=int(len(meta)), n_subjects=int(meta.subject.nunique()), features='DE 5 bands x channels, subject-centred')
    # --- binary valence (neutral excluded) ---
    m = meta.valence.to_numpy() != 0
    y = (meta.valence.to_numpy()[m] > 0).astype(int); subj = meta.subject.to_numpy()[m]; trial = meta.trial.to_numpy()[m]
    # trial id must be unique per (session, trial) for the shuffle
    trial_key = (meta.session.to_numpy()[m] * 1000 + trial)
    lda = loso_auc(Z[m], y, subj, trial=trial_key, n_shuffle=5); out['valence_lda'] = {k: v for k, v in lda.items() if k != 'direction_mean'}
    out['valence_logreg'] = loso_logreg(Z[m], y, subj)
    print(f"[{a.dataset}] valence LOSO AUC  LDA {lda['mean']:.3f} {lda['ci95']}  shuffle {lda['label_shuffle_mean']:.3f} | LR {out['valence_logreg']['mean']:.3f}", flush=True)
    # --- multiclass label (SEED-IV 4 emotions; DEAP valence>5) ---
    lab = meta.label.to_numpy()
    if len(np.unique(lab)) > 2:
        out['label_multiclass_logreg'] = loso_multiclass(Z, lab, meta.subject.to_numpy())
        print(f"[{a.dataset}] {len(np.unique(lab))}-class LOSO acc {out['label_multiclass_logreg']['mean']:.3f} (chance {out['label_multiclass_logreg']['chance']:.3f})", flush=True)
    # --- ISC of the DE valence score on stimulus-locked clips ---
    score = subject_center(Z, meta.subject.to_numpy()) @ lda['direction_mean']
    traj = {}
    for (cl, s), g in meta.assign(score=score).groupby(['clip_id', 'subject']):
        if cl < 0: continue
        traj.setdefault(int(cl), {})[int(s)] = g.sort_values('t_onset')['score'].to_numpy()
    isc = isc_test(traj, n_null=1000); out['isc_de_valence_score'] = {k: v for k, v in isc.items() if k != 'null'}
    print(f"[{a.dataset}] ISC {isc['isc']:.4f} null p99 {isc['null_p99']:.4f} p={isc['p_value']:.3f} pairs={isc['n_pairs']}", flush=True)
    # --- circular-shift read-out ---
    shifts = {}
    for dlt in [0, 30, 60, 90, 180]:
        ys_all = circular_shift_labels(meta, (meta.valence.to_numpy() > 0).astype(int), dlt)
        r = loso_auc(Z[m], ys_all[m], subj, trial=trial_key, n_shuffle=0); shifts[str(dlt)] = dict(mean=r['mean'], ci95=r['ci95'])
    out['circular_shift_auc'] = shifts; print(f"[{a.dataset}] shift AUC: " + ', '.join(f"{k}s={v['mean']:.3f}" for k, v in shifts.items()), flush=True)
    out['seconds'] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(ROOT, 'results/baselines'), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, 'results/baselines', f'{a.dataset}_de.json'), 'w'), indent=1)
    print('saved', flush=True)

if __name__ == '__main__':
    main()
