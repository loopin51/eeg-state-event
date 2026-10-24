"""D-123 exploration (development blocks only, no alpha): which window / feature / reader / averaging / pooling decodes the concept
inside each modality of ds004306. Test blocks (odd block index) are never loaded into any score here.
For every target subject and modality: leave-one-development-block-out (block id = session * 100 + block) within the modality;
training = the other development blocks of the subject (+ every development block of the other subjects when pool = all);
pseudo-trials average n same-concept trials inside the training set and inside the held-out block separately.
Score = balanced accuracy - 1/3, averaged over subjects. Selection rule (fixed before running, D-123):
  1. combo (feature, reader, n_avg, pool) = argmax over combos of mean over the two word modalities (t, a) of the best window score
  2. window per modality (image, t, a) = argmax of that combo's score; ties -> earlier start, then shorter
  3. a modality whose selected score < min_dev_excess is closed (not tested)
Output: results/word_ds004306/explore.json
usage: python scripts/word_explore.py configs/sem_word.yaml [--workers 8] [--smoke]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, json, time, itertools, argparse, yaml, numpy as np
from multiprocessing import Pool
from scipy.signal import butter, sosfiltfilt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from sem_verdict import lda_fit, lda_score, pca_fit, fsha, P, NC

_F = {}
_DRAWS = 10                                            # = explore.pseudo_draws (checked in main)


def windows(e):
    w = [(s, round(s + l, 3)) for s in e['window_starts'] for l in e['window_lengths'] if s + l <= e['window_max_end'] + 1e-9]
    return sorted(w)


def load(c, smoke=False):
    """Development perception trials only: per session, epochs -> features for every window x feature type."""
    pj = json.load(open(os.path.join(P(c['cache_dir']), 'prep.json'))); e = c['explore']; W = windows(e)
    bands = [butter(4, b, btype='band', fs=128.0, output='sos') for b in c['prep']['de_bands']]
    subs = {}
    for m in pj['sessions']:
        if m.get('missing'):
            continue
        f = P(m['file']); assert fsha(f) == m['sha256'], f
        z = np.load(f); dev = (z['block'] % 2 == e['dev_parity']) & z['p_ok']
        ep = z['p_ep'][dev].astype(np.float64); t = z['p_t']
        if smoke:
            ep = ep[:, :16]
        filt = [sosfiltfilt(sb, ep, axis=-1) for sb in bands]
        feats = {}
        for (a, b) in W:
            i = np.flatnonzero((t >= a - 1e-9) & (t < b - 1e-9)); th = np.array_split(i, 3)
            feats[(a, b, 'mean')] = ep[..., i].mean(-1)
            feats[(a, b, 'bins3')] = np.concatenate([ep[..., k].mean(-1) for k in th], 1)
            feats[(a, b, 'de')] = np.concatenate([np.log(fb[..., i].var(-1) + 1e-12) for fb in filt], 1)
        for k in feats:                                    # session z-score over all development perception trials (no labels)
            X = feats[k]; feats[k] = (X - X.mean(0)) / (X.std(0) + 1e-8)
        rec = dict(concept=z['concept'][dev], modality=z['modality'][dev], bid=np.full(dev.sum(), 0) + z['block'][dev], feats=feats)
        S = subs.setdefault(m['sub'], []); rec['bid'] = rec['bid'] + 100 * len(S); S.append(rec)
    out = {}
    for s, parts in subs.items():
        out[s] = dict(concept=np.concatenate([p['concept'] for p in parts]), modality=np.concatenate([p['modality'] for p in parts]),
                      bid=np.concatenate([p['bid'] for p in parts]),
                      feats={k: np.concatenate([p['feats'][k] for p in parts]) for k in parts[0]['feats']})
    return out, W


def pseudo(X, y, n, rng):
    if n == 1:
        return X, y
    Xs, ys = [], []
    for k in range(NC):
        i = rng.permutation(np.flatnonzero(y == k)); g = len(i) // n
        if g:
            Xs.append(X[i[:g * n]].reshape(g, n, -1).mean(1)); ys += [k] * g
    return (np.concatenate(Xs), np.array(ys)) if Xs else (X[:0], y[:0])


def bacc(pred, y):
    r = [np.mean(pred[y == k] == k) for k in range(NC) if (y == k).any()]
    return float(np.mean(r)) if r else np.nan


def fit_predict(Xtr, ytr, Xte, reader, C, pca):
    if pca:
        mu, V = pca_fit(Xtr, min(64, Xtr.shape[0] - 1)); Xtr = (Xtr - mu) @ V; Xte = (Xte - mu) @ V
    if reader == 'lda':
        return lda_score(lda_fit(Xtr, ytr), Xte).argmax(1)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=500).fit(Xtr, ytr).predict(Xte)


def job(args):
    s, m, key, reader, n, pool, C, seed = args
    D = _F; T = D[s]; rng = np.random.default_rng(seed); accs = []
    others = [o for o in D if o != s]
    Xo = np.concatenate([D[o]['feats'][key][D[o]['modality'] == m] for o in others]) if pool == 'all' else None
    yo = np.concatenate([D[o]['concept'][D[o]['modality'] == m] for o in others]) if pool == 'all' else None
    rows = np.flatnonzero(T['modality'] == m); X = T['feats'][key][rows]; y = T['concept'][rows]; b = T['bid'][rows]
    for blk in np.unique(b):
        tr, te = b != blk, b == blk
        for _ in range(1 if n == 1 else _DRAWS):       # amendment 1: n_avg > 1 averages random pseudo-trial draws
            Xtr, ytr = pseudo(X[tr], y[tr], n, rng); Xte, yte = pseudo(X[te], y[te], n, rng)
            if pool == 'all':
                Xa, ya = pseudo(Xo, yo, n, rng); Xtr = np.concatenate([Xtr, Xa]); ytr = np.concatenate([ytr, ya])
            if len(yte) == 0 or len(np.unique(ytr)) < NC:
                continue
            pr = fit_predict(Xtr, ytr, Xte, reader, C, key[2] == 'de'); accs.append((bacc(pr, yte), len(yte)))
    a = np.array(accs); return (s, m, key, reader, n, pool), float(np.average(a[:, 0], weights=a[:, 1]) - 1 / 3) if len(a) else np.nan


def main():
    global _F
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); c = yaml.safe_load(open(P(a.config))); e = c['explore']; od = P(c['out_dir']); os.makedirs(od, exist_ok=True)
    assert e['pseudo_draws'] == _DRAWS
    t0 = time.time(); _F, W = load(c, a.smoke); subs = sorted(_F)
    if a.smoke:
        subs = subs[:3]; _F = {s: _F[s] for s in subs}; W = W[:2]
    combos = list(itertools.product(e['features'], e['readers'], e['n_avg'], e['pools']))
    jobs = [(s, m, (w[0], w[1], f), r, n, p, e['lr_C_explore'], e['seed']) for s in subs for m in range(3) for w in W for (f, r, n, p) in combos]
    print('jobs', len(jobs), 'load s', round(time.time() - t0), flush=True)
    with Pool(a.workers) as pool:
        res = pool.map(job, jobs, chunksize=8)
    tab = {}
    for (s, m, key, r, n, p), v in res:
        tab.setdefault((m, key[0], key[1], key[2], r, n, p), []).append(v)
    score = {k: float(np.nanmean(v)) for k, v in tab.items()}
    best_w = {}
    for (m, ws, we, f, r, n, p), v in score.items():
        k = (m, f, r, n, p); cand = (v, -ws, -(we - ws))
        if k not in best_w or cand > best_w[k][0]:
            best_w[k] = (cand, (ws, we))
    combo_score = {(f, r, n, p): float(np.mean([best_w[(m, f, r, n, p)][0][0] for m in (1, 2)])) for (f, r, n, p) in combos}
    chosen = max(combo_score, key=lambda k: combo_score[k])
    sel = {}
    for m, name in enumerate(c['modalities']):
        (v, _, _), w = best_w[(m,) + chosen]
        sel[name] = dict(window=list(w), dev_excess=v, open=bool(v >= e['min_dev_excess']))
    out = dict(stage='explore', subjects=subs, n_jobs=len(jobs), seconds=round(time.time() - t0), smoke=a.smoke,
               chosen=dict(feature=chosen[0], reader=chosen[1], n_avg=chosen[2], pool=chosen[3]), combo_score={'|'.join(map(str, k)): v for k, v in combo_score.items()},
               selected=sel, table=[dict(modality=c['modalities'][k[0]], window=[k[1], k[2]], feature=k[3], reader=k[4], n_avg=k[5], pool=k[6], dev_excess=v) for k, v in sorted(score.items())])
    json.dump(out, open(os.path.join(od, 'explore_smoke.json' if a.smoke else 'explore.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'table'}, indent=1))


if __name__ == '__main__':
    main()
