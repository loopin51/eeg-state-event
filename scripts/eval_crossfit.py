"""Cross-fitted evaluation (D-027). Given K fold runs (each Stage A trained WITHOUT its outer-test subjects),
evaluate every subject only through the encoder that never saw it:
  - read-out (subject-centred shrinkage LDA) fitted on that fold's Stage-A TRAIN subjects (their labels only),
    applied to its outer-TEST subjects -> per-subject AUC
  - subject x clip disjoint: read-out fitted on train subjects x train clips, tested on test subjects x held-out
    clips (valence-stratified clip folds)
  - ISC of the test subjects' scores per clip (all encoded by the same fold encoder)
DE features go through the identical splits for a matched baseline.
Extra frozen feature sets (e.g. public foundation-model features from extract_fm_features.py) go through the
identical splits and read-out when listed in EXTRA_FEATURES (comma-separated names; loads
results/features/<dataset>_<name>.npy). Each gets its own paired model-minus-baseline comparison.
Usage: [EXTRA_FEATURES=cbramod,labram] python scripts/eval_crossfit.py OUT_JSON FOLD_RUN_DIR [FOLD_RUN_DIR ...]
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
import sys, json, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.eval.readout import subject_center, bootstrap_ci, isc_test
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

def lda(a, b, c): return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(a, b).decision_function(c)

def clip_split(clip, y, k=6, seed=0):
    rng = np.random.default_rng(seed); pos = np.unique(clip[y == 1]); neg = np.unique(clip[y == 0])
    rng.shuffle(pos); rng.shuffle(neg); k = min(k, len(pos), len(neg))
    return [set(pos[f::k]) | set(neg[f::k]) for f in range(k)]

out_path, runs = sys.argv[1], sys.argv[2:]; out = {}
for nm in ['seediv', 'deap', 'faced', 'seedvii']:
    if not all(os.path.exists(os.path.join(r, f'z_{nm}.npz')) for r in runs): continue
    meta = Cache(os.path.join(ROOT, 'data/cache'), nm).meta
    # z_*.npz rows follow MultiCache order, which drops subjects outside cfg['subjects'] (smoke configs).
    cfg0 = yaml.safe_load(open(os.path.join(runs[0], 'config.yaml'))); keep = np.ones(len(meta), bool)
    if cfg0.get('subjects') and nm in cfg0['subjects']: keep = meta['subject'].isin(cfg0['subjects'][nm]).to_numpy()
    de_full = np.load(os.path.join(ROOT, f'results/features/{nm}_de.npy'))[keep]
    meta = meta[keep].reset_index(drop=True)
    subj = meta['subject'].to_numpy(); val = meta['valence'].to_numpy(); m = val != 0
    y_all = (val > 0).astype(int); clip = meta['clip_id'].to_numpy()
    fixed = {'DE': subject_center(de_full, subj)}
    for ex in [e for e in os.environ.get('EXTRA_FEATURES', '').split(',') if e]:
        fp = os.path.join(ROOT, f'results/features/{nm}_{ex}.npy')
        if os.path.exists(fp): fixed[ex] = subject_center(np.load(fp)[keep], subj)
        else: print(f'[{nm}] extra feature set {ex} not found: {fp}', flush=True)
    res = {t: dict(subj_auc={}, subjclip_auc=[], scores=np.full(len(meta), np.nan)) for t in ['model', *fixed]}
    for r in runs:
        sp = json.load(open(os.path.join(r, 'split.json'))); test = set(sp['test'][nm]); train = set(sp['train'][nm]) | set(sp['val'][nm])
        Z = subject_center(np.load(os.path.join(r, f'z_{nm}.npz'))['z_state'].mean(1), subj)
        tr = np.isin(subj, list(train)) & m; te_all = np.isin(subj, list(test))
        folds = clip_split(clip[m], y_all[m])
        for tag, X in (('model', Z), *fixed.items()):
            sc = lda(X[tr], y_all[tr], X[te_all]); res[tag]['scores'][te_all] = sc
            for s_ in test:
                k = (subj[te_all] == s_) & m[te_all]
                if len(np.unique(y_all[te_all][k])) == 2: res[tag]['subj_auc'][int(s_)] = float(roc_auc_score(y_all[te_all][k], sc[k]))
            for tc in folds:
                trk = tr & ~np.isin(clip, list(tc)); tek = te_all & m & np.isin(clip, list(tc))
                if len(np.unique(y_all[tek])) == 2:
                    res[tag]['subjclip_auc'].append(float(roc_auc_score(y_all[tek], lda(X[trk], y_all[trk], X[tek]))))
    out[nm] = {}
    # Paired model-minus-baseline differences: same subject (unseen-subject AUC) / same (fold, clip-fold) cell.
    for base in fixed:
        ks = sorted(set(res['model']['subj_auc']) & set(res[base]['subj_auc']))
        ds = np.array([res['model']['subj_auc'][k] - res[base]['subj_auc'][k] for k in ks])
        dc = np.array(res['model']['subjclip_auc']) - np.array(res[base]['subjclip_auc'])
        pr = dict(subj_diff=float(ds.mean()), subj_diff_ci95=bootstrap_ci(ds), subj_wins=int((ds > 0).sum()), n_subj=len(ds),
                  cell_diff=float(dc.mean()), cell_diff_ci95=bootstrap_ci(dc), cell_wins=int((dc > 0).sum()), n_cells=len(dc),
                  per_subject={int(k): [res['model']['subj_auc'][k], res[base]['subj_auc'][k]] for k in ks})
        out[nm]['paired' if base == 'DE' else f'paired_{base}'] = pr
        print(f"[{nm}] PAIRED model-{base:7s} subject {pr['subj_diff']:+.3f} [{pr['subj_diff_ci95'][0]:+.3f}, {pr['subj_diff_ci95'][1]:+.3f}] "
              f"(model better in {pr['subj_wins']}/{pr['n_subj']}) | subj x clip cell {pr['cell_diff']:+.3f} "
              f"[{pr['cell_diff_ci95'][0]:+.3f}, {pr['cell_diff_ci95'][1]:+.3f}] ({pr['cell_wins']}/{pr['n_cells']})", flush=True)
    for tag, d in res.items():
        a = np.array(list(d['subj_auc'].values())); b = np.array(d['subjclip_auc'])
        traj = {}
        for (cl, s_), g in meta.assign(p=d['scores']).dropna(subset=['p']).groupby(['clip_id', 'subject']):
            if cl >= 0: traj.setdefault(int(cl), {})[int(s_)] = g.sort_values('t_onset')['p'].to_numpy()
        isc = isc_test(traj, n_null=500) if traj else None
        out[nm][tag] = dict(unseen_subject_auc=float(a.mean()), ci95=bootstrap_ci(a), n_subjects=len(a),
                            subject_clip_disjoint_auc=float(b.mean()), subject_clip_ci95=bootstrap_ci(b), n_cells=len(b),
                            isc=isc['isc'] if isc else None, isc_p=isc['p_value'] if isc else None)
        o = out[nm][tag]
        print(f"[{nm}] {tag:5s} unseen-subject AUC {o['unseen_subject_auc']:.3f} [{o['ci95'][0]:.3f}, {o['ci95'][1]:.3f}] (n={o['n_subjects']}) | "
              f"subj x clip disjoint {o['subject_clip_disjoint_auc']:.3f} [{o['subject_clip_ci95'][0]:.3f}, {o['subject_clip_ci95'][1]:.3f}] | ISC {o['isc']:.4f} (p={o['isc_p']:.3f})", flush=True)
json.dump(out, open(out_path, 'w'), indent=1)
