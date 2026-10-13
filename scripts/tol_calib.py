"""Thinking Out Loud (Nieto et al., 2022, OpenNeuro ds003626): session-held-out per-user calibration curves (D-045).

Per subject, one session is held out for testing (3-fold rotation) and calibration trials come from the other two
sessions of the same condition. Readout = shrinkage LDA with uniform priors; metric = balanced accuracy (4 words).
Features (per trial):
  DE      6-band DE (1-4, 4-8, 8-13, 13-30, 30-45, 45-100 Hz), 1.5-3.5 s after cue, 128 EEG ch (primary)
  EXG     same DE on the 8 EXG channels (muscle control)
  PRE     same DE on -0.5..0 s (pre-cue fixation; leak meter c)
  DC      raw-BDF per-channel mean and slope over -0.5..4 s, 128 EEG ch (leak meter b)
  ORDER   trial position within the session, linear + quadratic (leak meter a)
Features are centred per session and condition (label-free) unless --no-center.
k values with fewer than k/4 calibration trials per class in the pool are skipped (None).
Chance threshold: labels permuted within the calibration set, 200 times, 95th percentile of the subject mean.

usage: python scripts/tol_calib.py OUT_JSON [--perms 200]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.eval.readout import bootstrap_ci
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

RAW = os.path.join(ROOT, 'data/raw/thinking_out_loud')
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45), (45, 100)]
KS = [20, 40, 80, 'all']; N_DRAWS = 5; K = 4
CONDS = {0: 'pronounced', 1: 'inner', 2: 'visualized'}


def de(x, fs):
    """x [n, C, T] -> [n, C*6] log band power."""
    f, P = welch(x, fs=fs, nperseg=min(x.shape[-1], int(fs)), axis=-1)
    return np.log(np.stack([P[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in BANDS], -1) + 1e-30).reshape(len(x), -1)


def load_subject(sub):
    import mne
    feats = {k: [] for k in ['DE', 'EXG', 'PRE', 'DC', 'ORDER']}; y, cond, sess = [], [], []
    for ses in (1, 2, 3):
        d = os.path.join(RAW, 'derivatives', f'sub-{sub:02d}', f'ses-{ses:02d}'); p = f'sub-{sub:02d}_ses-{ses:02d}'
        ev = np.load(os.path.join(d, f'{p}_events.dat'), allow_pickle=True)
        e = mne.read_epochs(os.path.join(d, f'{p}_eeg-epo.fif'), verbose=False); fs = e.info['sfreq']
        x = e.get_data(); t = e.times; w = (t >= 1.5) & (t < 3.5); pre = (t >= -0.5) & (t < 0)
        xe = mne.read_epochs(os.path.join(d, f'{p}_exg-epo.fif'), verbose=False).get_data()
        assert len(x) == len(ev) == len(xe)
        feats['DE'].append(de(x[:, :, w], fs)); feats['EXG'].append(de(xe[:, :, w], fs)); feats['PRE'].append(de(x[:, :, pre], fs))
        raw = mne.io.read_raw_bdf(os.path.join(RAW, f'sub-{sub:02d}', f'ses-{ses:02d}', 'eeg', f'{p}_task-innerspeech_eeg.bdf'), verbose=False)
        eeg_idx = list(range(128)); rfs = raw.info['sfreq']; tt = np.arange(int(-0.5 * rfs), int(4.0 * rfs)) / rfs; dc = []
        for s0 in ev[:, 0]:
            seg = raw.get_data(picks=eeg_idx, start=int(s0 + tt[0] * rfs), stop=int(s0 + tt[0] * rfs) + len(tt))
            dc.append(np.r_[seg.mean(1), np.polyfit(tt, seg.T, 1)[0]])
        feats['DC'].append(np.array(dc))
        o = np.arange(len(ev), dtype=float); feats['ORDER'].append(np.c_[o, o ** 2])
        y.append(ev[:, 1]); cond.append(ev[:, 2]); sess.append(np.full(len(ev), ses))
    return {k: np.concatenate(v) for k, v in feats.items()}, np.concatenate(y).astype(int), np.concatenate(cond).astype(int), np.concatenate(sess)


def center(X, groups):
    X = X.astype(np.float64).copy()
    for g in np.unique(groups): X[groups == g] -= X[groups == g].mean(0)
    return X


def fit_score(Xtr, ytr, Xte, yte):
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=np.ones(K) / K).fit(Xtr, ytr)
    return balanced_accuracy_score(yte, clf.predict(Xte))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out_json'); ap.add_argument('--perms', type=int, default=200)
    ap.add_argument('--no-center', action='store_true'); a = ap.parse_args(); t0 = time.time()
    res = dict(ks=[str(k) for k in KS], perms=a.perms, centered=not a.no_center, subjects={})
    null = {c: {f: np.zeros((10, a.perms)) for f in ['DE', 'EXG', 'PRE', 'DC', 'ORDER']} for c in CONDS}
    for si, sub in enumerate(range(1, 11)):
        F, y, cond, sess = load_subject(sub); r = {}
        for c, cname in CONDS.items():
            mc = cond == c; r[cname] = {}
            for fname, X in F.items():
                Xc = X[mc] if a.no_center else center(X[mc], sess[mc]); yc, sc = y[mc], sess[mc]; curve = {}
                for k in KS:
                    accs = []
                    for ts in (1, 2, 3):
                        pool, te = np.where(sc != ts)[0], np.where(sc == ts)[0]
                        if k != 'all' and min((yc[pool] == q).sum() for q in range(K)) < k // K: continue   # too few calibration trials
                        for d in range(1 if k == 'all' else N_DRAWS):
                            rng = np.random.default_rng(1000 * sub + 10 * ts + d)
                            cal = pool if k == 'all' else np.concatenate([rng.choice(pool[yc[pool] == q], k // K, replace=False) for q in range(K)])
                            accs.append(fit_score(Xc[cal], yc[cal], Xc[te], yc[te]))
                    curve[str(k)] = float(np.mean(accs)) if accs else None
                r[cname][fname] = curve
                # permutation null at k = all
                rng = np.random.default_rng(99 + sub)
                for p in range(a.perms):
                    accs = []
                    for ts in (1, 2, 3):
                        pool, te = np.where(sc != ts)[0], np.where(sc == ts)[0]
                        accs.append(fit_score(Xc[pool], rng.permutation(yc[pool]), Xc[te], yc[te]))
                    null[c][fname][si, p] = np.mean(accs)
        res['subjects'][sub] = r
        print(f"sub-{sub:02d} inner k=all: " + ' '.join(f"{f} {r['inner'][f]['all']:.3f}" for f in F) +
              f" | pronounced DE {r['pronounced']['DE']['all']:.3f} EXG {r['pronounced']['EXG']['all']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(a.out_json, 'w'), indent=1)
    summary = {}
    for c, cname in CONDS.items():
        summary[cname] = {}
        for f in null[c]:
            per = {k: np.array([res['subjects'][s][cname][f][k] for s in res['subjects'] if res['subjects'][s][cname][f][k] is not None]) for k in res['ks']}
            per = {k: v for k, v in per.items() if len(v)}
            thr = float(np.quantile(null[c][f].mean(0), 0.95))
            summary[cname][f] = dict(curve={k: dict(mean=float(v.mean()), ci95=bootstrap_ci(v), per_subject=v.round(4).tolist()) for k, v in per.items()},
                                     chance_p95=thr, null_mean=float(null[c][f].mean()))
    res['summary'] = summary
    inner = summary['inner']; prim = inner['DE']['curve']['all']
    meters_ok = all(inner[m]['curve']['all']['mean'] <= inner[m]['chance_p95'] for m in ['ORDER', 'DC', 'PRE'])
    detected = prim['ci95'][0] > inner['DE']['chance_p95']
    res['verdict'] = dict(primary=prim, chance_p95=inner['DE']['chance_p95'], leak_meters_at_chance=meters_ok,
                          exg_above_chance=inner['EXG']['curve']['all']['mean'] > inner['EXG']['chance_p95'],
                          result='detected' if (detected and meters_ok) else ('withheld (leak meter above chance)' if not meters_ok else 'not detected'))
    json.dump(res, open(a.out_json, 'w'), indent=1)
    for cname in CONDS.values():
        for f, v in summary[cname].items():
            print(f"{cname:10s} {f:5s} " + ' '.join(f"k={k}: {v['curve'][k]['mean']:.3f}(n={len(v['curve'][k]['per_subject'])})" for k in v['curve']) +
                  f" | ci(all) {np.round(v['curve']['all']['ci95'], 3)} | chance p95 {v['chance_p95']:.3f}", flush=True)
    print('VERDICT', res['verdict'], flush=True)


if __name__ == '__main__':
    main()
