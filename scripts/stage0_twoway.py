"""Stage 0-d (D-039): two-way (subject x clip) bootstrap of the cross-fit headline, model minus DE.

Cell A(s, p, n) = fraction of (window of positive clip p, window of negative clip n) pairs of test subject s
ordered correctly by the valence score. Two conditions per cross-fit fold:
  same_clip   : readout fit on the fold's train subjects, all clips (one LDA per fold).
  clip_disjoint: readout fit on the fold's train subjects without clips p and n (one LDA per pair).
The model-minus-DE cell differences are averaged, and the CI resamples subjects, positive clips and negative clips
independently (1000 draws). DEAP is excluded: its labels are per-subject median splits, so a clip has no fixed sign.

usage: python scripts/stage0_twoway.py <out.json> <run_cf0> ... <run_cf4>
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('ESE_NUM_THREADS', '8'))
for _v in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']: os.environ.setdefault(_v, os.environ['OMP_NUM_THREADS'])
import sys, json, time, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.eval.readout import subject_center
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def lda(a, b, c): return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(a, b).decision_function(c)


def pair_acc(sp, sn):
    """Fraction of (pos, neg) window pairs with sp > sn (ties count half)."""
    d = sp[:, None] - sn[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def twoway_ci(D, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); nS, nP, nN = D.shape; out = np.empty(n_boot)
    for b in range(n_boot):
        s, p, n = rng.integers(0, nS, nS), rng.integers(0, nP, nP), rng.integers(0, nN, nN)
        out[b] = np.nanmean(D[np.ix_(s, p, n)])
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def subject_ci(D, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); per = np.nanmean(D.reshape(D.shape[0], -1), 1)
    return [float(np.quantile([rng.choice(per, len(per)).mean() for _ in range(n_boot)], q)) for q in (.025, .975)]


def main():
    out_path, runs = sys.argv[1], sys.argv[2:]; out = {}
    for nm in ['faced', 'seediv', 'seedvii']:
        t0 = time.time()
        meta = Cache(os.path.join(ROOT, 'data/cache'), nm).meta.reset_index(drop=True)
        subj = meta['subject'].to_numpy(); val = meta['valence'].to_numpy(); clip = meta['clip_id'].to_numpy()
        m = val != 0; y = (val > 0).astype(int)
        pos_clips = np.unique(clip[m & (y == 1)]); neg_clips = np.unique(clip[m & (y == 0)])
        assert not set(pos_clips) & set(neg_clips), 'clip with both signs'
        subjects = np.unique(subj); si = {s: i for i, s in enumerate(subjects)}
        pi = {c: i for i, c in enumerate(pos_clips)}; ni = {c: i for i, c in enumerate(neg_clips)}
        feats = {'DE': subject_center(np.load(os.path.join(ROOT, f'results/features/{nm}_de.npy')), subj)}
        A = {(t, c): np.full((len(subjects), len(pos_clips), len(neg_clips)), np.nan)
             for t in ['model', 'DE'] for c in ['same_clip', 'clip_disjoint']}
        for r in runs:
            sp = json.load(open(os.path.join(r, 'split.json')))
            test = np.array(sorted(sp['test'][nm])); train = set(sp['train'][nm]) | set(sp['val'][nm])
            X = dict(feats, model=subject_center(np.load(os.path.join(r, f'z_{nm}.npz'))['z_state'].mean(1), subj))
            tr = np.isin(subj, list(train)) & m
            for tag in ['model', 'DE']:
                F = X[tag]
                # same_clip: one readout per fold
                te = np.isin(subj, test) & m; sc = np.full(len(meta), np.nan); sc[te] = lda(F[tr], y[tr], F[te])
                for s in test:
                    ks = subj == s
                    for p in pos_clips:
                        a = sc[ks & (clip == p)]
                        if not len(a): continue
                        for n in neg_clips:
                            b = sc[ks & (clip == n)]
                            if len(b): A[(tag, 'same_clip')][si[s], pi[p], ni[n]] = pair_acc(a, b)
                # clip_disjoint: leave the pair (p, n) out of the readout
                for p in pos_clips:
                    for n in neg_clips:
                        trk = tr & (clip != p) & (clip != n); tek = np.isin(subj, test) & ((clip == p) | (clip == n))
                        s_ = lda(F[trk], y[trk], F[tek]); st, ct = subj[tek], clip[tek]
                        for s in test:
                            a, b = s_[(st == s) & (ct == p)], s_[(st == s) & (ct == n)]
                            if len(a) and len(b): A[(tag, 'clip_disjoint')][si[s], pi[p], ni[n]] = pair_acc(a, b)
            print(f'[{nm}] fold {os.path.basename(r)} done ({time.time() - t0:.0f}s)', flush=True)
        out[nm] = dict(n_subjects=len(subjects), n_pos_clips=len(pos_clips), n_neg_clips=len(neg_clips))
        for cond in ['same_clip', 'clip_disjoint']:
            D = A[('model', cond)] - A[('DE', cond)]
            r = dict(model=float(np.nanmean(A[('model', cond)])), DE=float(np.nanmean(A[('DE', cond)])),
                     diff=float(np.nanmean(D)), diff_ci95_twoway=twoway_ci(D), diff_ci95_subject_only=subject_ci(D),
                     model_ci95_twoway=twoway_ci(A[('model', cond)] - .5), DE_ci95_twoway=twoway_ci(A[('DE', cond)] - .5))
            r['model_ci95_twoway'] = [v + .5 for v in r['model_ci95_twoway']]; r['DE_ci95_twoway'] = [v + .5 for v in r['DE_ci95_twoway']]
            out[nm][cond] = r
            print(f"[{nm}] {cond:13s} model {r['model']:.3f} {np.round(r['model_ci95_twoway'], 3)} | DE {r['DE']:.3f} "
                  f"{np.round(r['DE_ci95_twoway'], 3)} | diff {r['diff']:+.3f} two-way {np.round(r['diff_ci95_twoway'], 3)} "
                  f"subject-only {np.round(r['diff_ci95_subject_only'], 3)}", flush=True)
        json.dump(out, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
