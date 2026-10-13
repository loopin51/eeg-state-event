"""Stage-B style read-out and the falsification battery (docs/05 §1, §3).

All functions take per-window features Z [N, D] plus per-window metadata
(subject, session, trial, clip_id, t_onset, labels). They are representation-
agnostic: used for DE features (baselines), z_state (model), or anything else.

  subject_center     per-subject z-score (label-free, transductive on test subject)
  loso_auc           leave-one-subject-out AUC with shrinkage LDA + label-shuffle floor
  bootstrap_ci       subject-level bootstrap CI of the mean AUC
  isc_test           inter-subject correlation of a 1-D trajectory vs circular-shift null
  circular_shift_labels  labels shifted along each session's timeline by Δ seconds
"""
import numpy as np, pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score


def subject_center(Z, subject):
    Z = np.asarray(Z, dtype=np.float64).copy()
    for s in np.unique(subject):
        m = subject == s
        mu, sd = Z[m].mean(0), Z[m].std(0); sd[sd == 0] = 1.0
        Z[m] = (Z[m] - mu) / sd
    return Z


def _fit_score(Ztr, ytr, Zte):
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ztr, ytr)
    return clf.decision_function(Zte), clf.coef_[0]


def loso_auc(Z, y, subject, trial=None, n_shuffle=5, seed=0, center=True):
    """y in {0,1}. Returns dict with per-subject AUCs, mean, pooled, label-shuffle floor, direction.
    Label shuffle permutes labels at TRIAL level within each training subject (windows of a
    trial keep a common label), so the floor reflects the trial-clustered structure."""
    rng = np.random.default_rng(seed)
    Z = subject_center(Z, subject) if center else np.asarray(Z, dtype=np.float64)
    y = np.asarray(y).astype(int); subject = np.asarray(subject)
    trial = np.asarray(trial) if trial is not None else np.arange(len(y))
    subs = np.unique(subject)
    per, per_shuf, pooled_s, pooled_y, dirs = {}, {s: [] for s in subs}, [], [], []
    for s in subs:
        te, tr = subject == s, subject != s
        if len(np.unique(y[te])) < 2: continue
        sc, w = _fit_score(Z[tr], y[tr], Z[te])
        per[int(s)] = float(roc_auc_score(y[te], sc)); dirs.append(w)
        pooled_s.append(sc); pooled_y.append(y[te])
        for k in range(n_shuffle):
            ysh = y.copy()
            for s2 in subs[subs != s]:
                m = subject == s2
                t_ids = np.unique(trial[m]); t_lab = {t: y[m][trial[m] == t][0] for t in t_ids}
                perm = rng.permutation([t_lab[t] for t in t_ids])
                remap = dict(zip(t_ids, perm))
                ysh[m] = [remap[t] for t in trial[m]]
            sc_sh, _ = _fit_score(Z[tr], ysh[tr], Z[te])
            per_shuf[int(s)].append(float(roc_auc_score(y[te], sc_sh)))
    aucs = np.array(list(per.values()))
    shuf = np.array([np.mean(v) for v in per_shuf.values() if v])
    return dict(per_subject=per, mean=float(aucs.mean()), std=float(aucs.std()),
                pooled=float(roc_auc_score(np.concatenate(pooled_y), np.concatenate(pooled_s))),
                label_shuffle_mean=float(shuf.mean()) if len(shuf) else None,
                ci95=bootstrap_ci(aucs, seed=seed), n_subjects=int(len(aucs)),
                direction_mean=np.mean(dirs, axis=0))


def bootstrap_ci(values, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(values)
    b = np.array([rng.choice(v, len(v)).mean() for _ in range(n_boot)])
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def isc_test(traj, n_null=1000, seed=0, min_len=5):
    """traj: dict clip_id -> dict subject -> 1-D array a(τ) sampled on a common grid (t_onset order).
    Real ISC = mean Pearson r over all subject pairs and clips (truncated to the clip's common length).
    Null: every subject's trajectory is circularly shifted by an independent random offset (>=1 step)
    before the pairwise correlations are recomputed; repeated n_null times. Vectorized per clip."""
    rng = np.random.default_rng(seed)
    mats = []
    for c, d in traj.items():
        arrs = [np.asarray(v, dtype=np.float64) for v in d.values()]
        if len(arrs) < 2: continue
        L = min(len(v) for v in arrs)
        if L < min_len: continue
        A = np.stack([v[:L] for v in arrs])                       # [S, L]
        A = A - A.mean(1, keepdims=True); sd = A.std(1, keepdims=True); sd[sd == 0] = 1.0
        mats.append(A / sd)
    n_pairs = sum(A.shape[0] * (A.shape[0] - 1) // 2 for A in mats)
    def mean_r(shift):
        num, den = 0.0, 0
        for A in mats:
            S, L = A.shape
            B = A
            if shift:
                offs = rng.integers(1, L, size=S)
                idx = (np.arange(L)[None, :] - offs[:, None]) % L
                B = np.take_along_axis(A, idx, axis=1)
            R = B @ B.T / L
            iu = np.triu_indices(S, 1); num += R[iu].sum(); den += len(iu[0])
        return num / den if den else float('nan')
    real = mean_r(False)
    null = np.array([mean_r(True) for _ in range(n_null)])
    return dict(isc=float(real), n_pairs=int(n_pairs), n_clips=len(mats), null_mean=float(null.mean()),
                null_p99=float(np.quantile(null, .99)), null_p95=float(np.quantile(null, .95)),
                p_value=float((null >= real).mean()), null=null)


def circular_shift_labels(meta, y, delta_s):
    """Shift each window's label by delta_s seconds along its (subject, session) timeline.
    Timeline = trials in presentation order, each trial's windows by t_onset; the shift wraps
    around within the session. Returns shifted y aligned to meta's row order."""
    meta = meta.reset_index(drop=True).copy(); meta['y'] = np.asarray(y)
    meta['row'] = np.arange(len(meta))
    out = np.asarray(y).copy()
    for (s, se), g in meta.groupby(['subject', 'session']):
        g = g.sort_values(['trial', 't_onset'])
        # absolute session time: cumulative trial durations (windows are on a 5 s hop)
        t_abs, acc, prev = [], 0.0, None
        for tr, gg in g.groupby('trial', sort=True):
            if prev is not None: acc += prev
            t_abs.extend((acc + gg['t_onset']).tolist()); prev = float(gg['t_onset'].max()) + 5.0
        t_abs = np.array(t_abs); total = t_abs.max() + 5.0
        target = (t_abs + delta_s) % total
        src = np.argmin(np.abs(t_abs[None, :] - target[:, None]), axis=1)
        out[g['row'].to_numpy()] = g['y'].to_numpy()[src]
    return out


def continuous_label_test(Z, meta, n_null=1000, seed=0, min_len=6):
    """G2-3 (SEED-VII): does a LOSO valence score track the subject's self-reported intensity curve WITHIN a trial?
    For each held-out subject: fit the shrinkage-LDA valence direction on the other subjects (valence != 0 windows),
    score every window of the held-out subject, and for each emotional trial correlate
        sign(trial valence) * score(τ)   with   cont_label(τ)
    over the trial's windows. Report the mean r over trials, and a null where each trial's score series is circularly
    shifted by an independent random offset (destroys timing, keeps autocorrelation)."""
    rng = np.random.default_rng(seed); meta = meta.reset_index(drop=True)
    subj = meta['subject'].to_numpy(); val = meta['valence'].to_numpy(); Zc = subject_center(Z, subj)
    pairs, per_subject = [], {}
    for s in np.unique(subj):
        tr = (subj != s) & (val != 0); te = subj == s
        sc, _ = _fit_score(Zc[tr], (val[tr] > 0).astype(int), Zc[te])
        mt = meta[te].assign(score=sc); rs = []
        for (_, _), g in mt[mt['valence'] != 0].groupby(['session', 'trial']):
            g = g.sort_values('t_onset'); a = g['score'].to_numpy() * np.sign(g['valence'].iloc[0]); b = g['cont_label'].to_numpy()
            if len(a) >= min_len and np.nanstd(b) > 0 and np.std(a) > 0 and not np.isnan(b).any():
                pairs.append((a - a.mean(), b - b.mean())); rs.append(np.corrcoef(a, b)[0, 1])
        per_subject[int(s)] = float(np.mean(rs)) if rs else float('nan')
    def mean_r(shift):
        out = []
        for a, b in pairs:
            aa = np.roll(a, int(rng.integers(1, len(a)))) if shift else a
            out.append((aa @ b) / (np.linalg.norm(aa) * np.linalg.norm(b) + 1e-12))
        return float(np.mean(out))
    real = mean_r(False); null = np.array([mean_r(True) for _ in range(n_null)])
    ps = np.array([v for v in per_subject.values() if not np.isnan(v)])
    return dict(mean_r=real, n_trials=len(pairs), null_mean=float(null.mean()), null_p99=float(np.quantile(null, .99)),
                p_value=float((null >= real).mean()), per_subject_mean=float(ps.mean()), ci95=bootstrap_ci(ps, seed=seed), per_subject=per_subject)
