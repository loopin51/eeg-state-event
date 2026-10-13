"""Cross-fitted secondary analyses (roadmap 2.6, 2.7). Same fold runs and splits as eval_crossfit.py:
every test subject is scored only by the encoder that never saw it, and every read-out is fitted on that
fold's Stage-A training subjects.

  (A) G2-3  SEED-VII continuous-label tracking. Per test subject, per emotional trial: Pearson r between the
            sign-corrected valence score and the self-reported intensity curve; circular-shift null per trial.
            Model vs DE, paired by subject.
  (B) G2-1  Cross-dataset valence-direction agreement within each fold encoder: cosine between the per-dataset
            shrinkage-LDA directions, subject-bootstrap interval. Only defined for the model (DE feature spaces
            differ across montages).
  (C)       Cross-dataset transfer: read-out fitted on dataset A's training subjects, applied to dataset B's
            test subjects (AUC). Diagonal = within-dataset. Only for the model, same reason.

Usage: python scripts/eval_crossfit_extra.py OUT_JSON FOLD_RUN_DIR [FOLD_RUN_DIR ...]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.eval.readout import subject_center, bootstrap_ci
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

NAMES = ['seediv', 'deap', 'faced', 'seedvii']


def fit_lda(X, y):
    return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)


def load(runs, nm):
    meta = Cache(os.path.join(ROOT, 'data/cache'), nm).meta.reset_index(drop=True)
    subj = meta['subject'].to_numpy()
    folds = []
    for r in runs:
        p = os.path.join(r, f'z_{nm}.npz')
        if not os.path.exists(p): return None
        sp = json.load(open(os.path.join(r, 'split.json')))
        folds.append(dict(Z=subject_center(np.load(p)['z_state'].mean(1), subj),
                          train=set(sp['train'][nm]) | set(sp['val'][nm]), test=set(sp['test'][nm])))
    de = subject_center(np.load(os.path.join(ROOT, f'results/features/{nm}_de.npy')), subj)
    return dict(meta=meta, subj=subj, folds=folds, de=de)


def _resid(v, g):
    """Residual of v after linear regression on g (with intercept)."""
    A = np.c_[np.ones_like(g), g]; coef, *_ = np.linalg.lstsq(A, v, rcond=None); return v - A @ coef


def trial_corrs(meta, score, rng, min_len=6, n_null=1000, partial_group=False):
    """Returns per-subject mean r and pooled (a, b) centred series for the circular-shift null.
    partial_group=True: remove, within each trial, the part of both the score and the self-report that is
    linearly explained by the OTHER subjects' mean self-report for the same (clip, t_onset). What remains is
    each subject's idiosyncratic deviation from the stimulus-driven group curve."""
    per_subj, pairs = {}, []
    m = meta.assign(score=score)
    if partial_group:
        tot = m.groupby(['clip_id', 't_onset'])['cont_label'].agg(['sum', 'count']).rename(columns={'sum': 'gs', 'count': 'gn'})
        m = m.join(tot, on=['clip_id', 't_onset'])
        m['group_loo'] = (m['gs'] - m['cont_label']) / (m['gn'] - 1).replace(0, np.nan)
    for s_, gs in m[m['valence'] != 0].groupby('subject'):
        rs = []
        for _, g in gs.groupby(['session', 'trial']):
            g = g.sort_values('t_onset'); a = g['score'].to_numpy() * np.sign(g['valence'].iloc[0]); b = g['cont_label'].to_numpy()
            if partial_group:
                gl = g['group_loo'].to_numpy()
                if np.isnan(gl).any() or np.std(gl) == 0: continue
                a, b = _resid(a, gl), _resid(b, gl)
            if len(a) >= min_len and np.std(a) > 0 and np.std(b) > 0 and not np.isnan(b).any():
                pairs.append((a - a.mean(), b - b.mean())); rs.append(np.corrcoef(a, b)[0, 1])
        if rs: per_subj[int(s_)] = float(np.mean(rs))
    def mean_r(shift):
        v = []
        for a, b in pairs:
            aa = np.roll(a, int(rng.integers(1, len(a)))) if shift else a
            v.append(aa @ b / (np.linalg.norm(aa) * np.linalg.norm(b) + 1e-12))
        return float(np.mean(v))
    real = mean_r(False); null = np.array([mean_r(True) for _ in range(n_null)])
    return per_subj, dict(mean_r=real, n_trials=len(pairs), null_p99=float(np.quantile(null, .99)), p_value=float((null >= real).mean()))


def continuous_label(D, seed=0, partial_group=False):
    rng = np.random.default_rng(seed); meta, subj = D['meta'], D['subj']
    m = meta['valence'].to_numpy() != 0; y = (meta['valence'].to_numpy() > 0).astype(int)
    out = {}
    for tag in ('model', 'DE'):
        score = np.full(len(meta), np.nan)
        for f in D['folds']:
            X = f['Z'] if tag == 'model' else D['de']
            tr = np.isin(subj, list(f['train'])) & m; te = np.isin(subj, list(f['test']))
            score[te] = fit_lda(X[tr], y[tr]).decision_function(X[te])
        keep = ~np.isnan(score)
        # group curve uses ALL subjects' self-reports (labels are comparable across folds); scores are per fold
        mk = meta.copy(); mk['score_ok'] = keep
        per, agg = trial_corrs(mk.reset_index(drop=True), np.where(keep, score, np.nan), rng, partial_group=partial_group) if partial_group else trial_corrs(meta[keep].reset_index(drop=True), score[keep], rng)
        out[tag] = dict(per_subject=per, **agg)
    ks = sorted(set(out['model']['per_subject']) & set(out['DE']['per_subject']))
    d = np.array([out['model']['per_subject'][k] - out['DE']['per_subject'][k] for k in ks])
    out['paired'] = dict(diff=float(d.mean()), ci95=bootstrap_ci(d), wins=int((d > 0).sum()), n=len(d))
    return out


def directions_and_transfer(data, n_boot=200, seed=0):
    rng = np.random.default_rng(seed); names = [n for n in NAMES if data.get(n)]
    K = len(data[names[0]]['folds'])
    cos = {f'{a}-{b}': [] for i, a in enumerate(names) for b in names[i + 1:]}
    cos_boot = {k: [] for k in cos}
    transfer = {a: {b: [] for b in names} for a in names}
    for k in range(K):
        dirs, models = {}, {}
        for nm in names:
            D = data[nm]; f = D['folds'][k]; meta = D['meta']
            m = meta['valence'].to_numpy() != 0; y = (meta['valence'].to_numpy() > 0).astype(int)
            tr = np.isin(D['subj'], list(f['train'])) & m
            clf = fit_lda(f['Z'][tr], y[tr]); models[nm] = clf
            w = clf.coef_[0]; dirs[nm] = w / np.linalg.norm(w)
            # subject bootstrap of the direction
            subs = np.array(sorted(f['train'])); boots = []
            for _ in range(n_boot):
                bs = rng.choice(subs, len(subs)); idx = np.concatenate([np.where((D['subj'] == s_) & m)[0] for s_ in bs])
                if len(np.unique(y[idx])) < 2: continue
                wb = fit_lda(f['Z'][idx], y[idx]).coef_[0]; boots.append(wb / np.linalg.norm(wb))
            dirs[nm + '_boot'] = boots
        for key in cos:
            a, b = key.split('-')
            cos[key].append(float(dirs[a] @ dirs[b]))
            n = min(len(dirs[a + '_boot']), len(dirs[b + '_boot']))
            cos_boot[key].extend(float(dirs[a + '_boot'][i] @ dirs[b + '_boot'][i]) for i in range(n))
        for a in names:
            for b in names:
                D = data[b]; f = D['folds'][k]; meta = D['meta']
                m = meta['valence'].to_numpy() != 0; y = (meta['valence'].to_numpy() > 0).astype(int)
                te = np.isin(D['subj'], list(f['test'])) & m
                transfer[a][b].append(float(roc_auc_score(y[te], models[a].decision_function(f['Z'][te]))))
    cos_out = {k: dict(mean=float(np.mean(v)), per_fold=v, boot_ci95=[float(np.quantile(cos_boot[k], .025)), float(np.quantile(cos_boot[k], .975))])
               for k, v in cos.items()}
    tr_out = {a: {b: dict(mean=float(np.mean(v)), per_fold=v) for b, v in row.items()} for a, row in transfer.items()}
    return cos_out, tr_out


def main():
    out_path, runs = sys.argv[1], sys.argv[2:]
    data = {nm: load(runs, nm) for nm in NAMES}
    out = {}
    if data.get('seedvii') is not None:
        cl = continuous_label(data['seedvii']); out['continuous_label_seedvii'] = cl
        for tag in ('model', 'DE'):
            print(f"[G2-3] {tag:5s} r = {cl[tag]['mean_r']:.4f} (null p99 {cl[tag]['null_p99']:.4f}, p={cl[tag]['p_value']:.3f}, {cl[tag]['n_trials']} trials)", flush=True)
        p = cl['paired']; print(f"[G2-3] paired model-DE per subject {p['diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] ({p['wins']}/{p['n']})", flush=True)
        cp = continuous_label(data['seedvii'], partial_group=True); out['continuous_label_seedvii_partial_group'] = cp
        for tag in ('model', 'DE'):
            print(f"[G2-3 idiosyncratic] {tag:5s} partial r | group curve = {cp[tag]['mean_r']:.4f} (null p99 {cp[tag]['null_p99']:.4f}, p={cp[tag]['p_value']:.3f}, {cp[tag]['n_trials']} trials)", flush=True)
        p = cp['paired']; print(f"[G2-3 idiosyncratic] paired model-DE {p['diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] ({p['wins']}/{p['n']})", flush=True)
    names = [n for n in NAMES if data.get(n)]
    if len(names) >= 2 and not os.environ.get('SKIP_DIRECTIONS'):
        cos, tr = directions_and_transfer(data)
        out['direction_cosine'] = cos; out['transfer_auc'] = tr
        for k, v in cos.items():
            print(f"[G2-1] cos({k}) = {v['mean']:+.3f}  fold range [{min(v['per_fold']):+.3f}, {max(v['per_fold']):+.3f}]  subject-bootstrap 95% [{v['boot_ci95'][0]:+.3f}, {v['boot_ci95'][1]:+.3f}]", flush=True)
        print('[transfer] rows = read-out fitted on, cols = applied to (unseen subjects)', flush=True)
        print('           ' + ' '.join(f'{b:>8s}' for b in names), flush=True)
        for a in names:
            print(f'  {a:8s} ' + ' '.join(f"{tr[a][b]['mean']:8.3f}" for b in names), flush=True)
    json.dump(out, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
