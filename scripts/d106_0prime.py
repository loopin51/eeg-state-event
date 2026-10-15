"""D-106 section 2: gate 0' - within-form contextual predictability contrast on the ds004408 words (no EEG, alpha-free).

Tokens: the 11,419 words without contraction fragments; forms (lower-case TextGrid word) with >= 4 tokens; target z = GPT-2
surprisal standardised within the form. Valid pairs: two tokens of a form in different runs or >= 30 s apart, z different.
Statistic: concordance A = (pairs whose prediction difference has the sign of the z difference + 0.5 x prediction ties) /
valid pairs; chance 0.5. Step -1: close if valid pairs < 1,000 or MDE = 2.486 x SD0 > 0.02 (SD0: A of a standard-normal
prediction (seed 20261217) under 1,000 within-form permutations of z). Meters (scalar ridge with grouped-CV correlation, RFF
ridge averaged over 5 seeds, 50-NN mean), forms in 5 folds by sha256(form + salt): ACOUSTIC (WORD_AC + log duration + local
speech rate (syllables in t +- 1 s / 2 s) + preceding pause), ACOUSTIC_NL, ACOUSTIC_KNN, ORDER (position terms, story position,
log within-form occurrence). Null: within-form permutation of z, 1,000 draws; excess = A - null mean; 90% interval from a
1,000-draw form bootstrap. Pass: step -1 passed and every meter's excess 90% upper < 0.02.

usage: python scripts/d106_0prime.py configs/d106_gates.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, hashlib, argparse, yaml, numpy as np
from sklearn.neighbors import NearestNeighbors
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from d098_gate0 import rff
from stageB_b1_meters import ALPHAS
from t1_tts_render import FRAG
import d106_feats as DF


def sha(s): return hashlib.sha256(s.encode()).hexdigest()


def ridge_scalar(Xtr, ytr, Xte, groups):
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8; Xtr = (Xtr - xm) / xs; Xte = (Xte - xm) / xs; ym = ytr.mean(); sc = np.zeros(len(ALPHAS))
    for g in np.unique(groups):
        tr, va = groups != g, groups == g; U, s, Vt = np.linalg.svd(Xtr[tr], full_matrices=False); Uty = U.T @ (ytr[tr] - ym); Xv = Xtr[va] @ Vt.T
        for j, a in enumerate(ALPHAS):
            p = Xv @ ((s / (s ** 2 + a)) * Uty); sc[j] += np.corrcoef(p, ytr[va])[0, 1] if p.std() > 0 else 0
    a = ALPHAS[int(np.argmax(sc))]; U, s, Vt = np.linalg.svd(Xtr, full_matrices=False); return Xte @ (Vt.T @ ((s / (s ** 2 + a)) * (U.T @ (ytr - ym))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); c = yaml.safe_load(open(a.config)); g = c['zero_prime']; t0 = time.time()
    W = DF.load_words(); word = np.asarray(W['word']).astype(str); n = len(word)
    lem = np.load(os.path.join(ROOT, c['cache'], 'lemma.npz')); frag = lem['pos'] == 'FRAG'
    XW, okw = DF.word_ac(W, c['g0t']['word_ac']['span_s'], c['g0t']['word_ac']['bins'])
    run = np.asarray(W['run']); on = np.asarray(W['onset']); off = np.asarray(W['offset']); nsyl = np.asarray(W['n_syl'], float)
    rate = np.array([nsyl[(run == run[i]) & (on >= on[i] - 1) & (on < on[i] + 1)].sum() / 2.0 for i in range(n)])
    prev_off = np.r_[np.nan, off[:-1]]; prev_off[np.r_[True, run[1:] != run[:-1]]] = 0.0; pause = np.clip(on - prev_off, 0, 2.0)
    use = ~frag & okw & np.isfinite(np.asarray(W['surprisal'], float))
    forms, cnt = np.unique(word[use], return_counts=True); keep_forms = set(forms[cnt >= g['min_tokens_per_form']].tolist())
    use &= np.isin(word, list(keep_forms)); idx = np.where(use)[0]; fw = word[idx]; flist = sorted(set(fw.tolist())); fi = np.array([flist.index(x) for x in fw])
    s = np.asarray(W['surprisal'], float)[idx]; z = np.zeros(len(idx))
    for k in range(len(flist)):
        m = fi == k; z[m] = (s[m] - s[m].mean()) / (s[m].std() + 1e-12)
    # valid pairs per form
    pairs = []
    for k in range(len(flist)):
        m = np.where(fi == k)[0]; x, y = np.triu_indices(len(m), 1); i, j = m[x], m[y]
        ok = ((run[idx[i]] != run[idx[j]]) | (np.abs(on[idx[i]] - on[idx[j]]) >= g['min_sep_s'])) & (z[i] != z[j]); pairs.append(np.c_[i[ok], j[ok]])
    P = np.concatenate(pairs).astype(int); pf = fi[P[:, 0]]
    res = dict(forms=len(flist), tokens=int(len(idx)), valid_pairs=int(len(P)))
    rng = np.random.default_rng(g['null']['seed']); perm_z = []
    for _ in range(g['null']['n']):
        zz = z.copy()
        for k in range(len(flist)):
            m = np.where(fi == k)[0]; zz[m] = z[rng.permutation(m)]
        perm_z.append(zz)
    perm_z = np.array(perm_z)
    def conc_counts(pred, zz):
        dp = pred[P[:, 0]] - pred[P[:, 1]]; dz = zz[P[:, 0]] - zz[P[:, 1]]
        return (np.sign(dp) == np.sign(dz)) * 1.0 + 0.5 * (dp == 0) * (dz != 0)
    def null_form_counts(pred):
        acc = np.zeros(len(flist)); tot = []
        for zz in perm_z:
            cc = conc_counts(pred, zz); acc += np.bincount(pf, cc, len(flist)); tot.append(cc.mean())
        return acc / len(perm_z), np.array(tot)
    # step -1
    rng_m = np.random.default_rng(g['mde']['seed']); rp = rng_m.standard_normal(len(idx))
    A0 = null_form_counts(rp)[1]; sd0 = float(A0.std()); mde = g['mde']['factor'] * sd0
    res.update(sd0=sd0, mde=mde, tokens_per_form={str(q): int(v) for q, v in zip(*np.unique(np.bincount(fi), return_counts=True))})
    print(json.dumps(res), flush=True)
    if len(P) < g['min_pairs'] or mde > g['mde']['max']:
        res['verdict'] = 'closed at step -1 (insufficient pairs or MDE > 0.02)'
    else:
        rel, glob, POS = DF.positions(W); occ = np.log(DF.occurrence_index(list(word)))
        XA = np.c_[XW, np.log(rate + 1e-3), pause][idx]; XO = np.c_[POS, glob, occ][idx]
        folds = {q: k % g['n_folds'] for k, q in enumerate(sorted(flist, key=lambda q: sha(q + g['fold_salt'])))}; tf = np.array([folds[q] for q in fw])
        def cv_pred(X, kind, seed=None):
            pred = np.zeros(len(idx))
            for f in range(g['n_folds']):
                te = tf == f; tr = ~te; trf = sorted(set(fw[tr].tolist()), key=lambda q: sha(q + g['fold_salt'])); gm = {q: k % 5 for k, q in enumerate(trf)}
                grp = np.array([gm[q] for q in fw[tr]]); Xtr, Xte = X[tr], X[te]
                if kind == 'knn':
                    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8; nn = NearestNeighbors(n_neighbors=g['knn']).fit((Xtr - xm) / xs)
                    _, nb = nn.kneighbors((Xte - xm) / xs); pred[te] = z[tr][nb].mean(1); continue
                if kind == 'rff':
                    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8; Xtr, Xte = rff((Xtr - xm) / xs, (Xte - xm) / xs, g['rff']['n'], seed)
                pred[te] = ridge_scalar(Xtr, z[tr], Xte, grp)
            return pred
        rng_b = np.random.default_rng(g['bootstrap']['seed']); FB = rng_b.integers(0, len(flist), (g['bootstrap']['n'], len(flist)))
        npair_f = np.bincount(pf, minlength=len(flist)).astype(float); meters = {}
        for name, X, kind, seeds in (('ACOUSTIC', XA, 'ridge', [None]), ('ACOUSTIC_NL', XA, 'rff', g['rff']['seeds']), ('ACOUSTIC_KNN', XA, 'knn', [None]), ('ORDER', XO, 'ridge', [None])):
            obs_f = np.zeros(len(flist)); null_f = np.zeros(len(flist))
            for sd in seeds:
                pred = cv_pred(X, kind, sd); co = conc_counts(pred, z); cnf, _ = null_form_counts(pred)
                obs_f += np.bincount(pf, co, len(flist)) / len(seeds); null_f += cnf / len(seeds)
            A = obs_f.sum() / npair_f.sum(); An = null_f.sum() / npair_f.sum(); ex = (obs_f - null_f)
            bs = np.array([ex[b].sum() / npair_f[b].sum() for b in FB])
            meters[name] = dict(A=float(A), null_mean=float(An), excess=float(A - An), excess_ci90=[float(np.percentile(bs, 5)), float(np.percentile(bs, 95))])
            print(name, round(A - An, 4), [round(x, 4) for x in meters[name]['excess_ci90']], f'({time.time() - t0:.0f}s)', flush=True)
        res['meters'] = meters; res['verdict'] = 'pass' if all(m['excess_ci90'][1] < g['pass_excess_upper90'] for m in meters.values()) else 'fail'
    res['seconds'] = time.time() - t0; out = os.path.join(ROOT, c['out']); os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, 'zero_prime.json'), 'w'), indent=1); print('verdict', res['verdict'], flush=True)


if __name__ == '__main__':
    main()
