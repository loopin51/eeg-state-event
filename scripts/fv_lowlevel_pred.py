"""D-114 section 1: low-level predictor L of the image CLIP embedding (no EEG).

Four candidates (configs/fv_lowlevel_pred.yaml): L_blur and L_phase use the blurred / phase-randomised image CLIP embedding as is;
L_all and L_classic are ridge maps (scripts/fv_gate.py ridge_fit: columns standardised on the training rows, penalty from
fv_eegfree.yaml ridge.alphas by concept-level 5-fold CV with fold salt 20262001) from the low-level feature blocks to the
original image CLIP embedding. The ridge is trained on the 1,454 training concepts left after removing the 200 development
concepts (14,540 images; test concepts are a separate split). Every prediction is L2-normalised.

L_mix (D-114 amendment 1): the weighted sum of the four candidates' row-standardised name scores, weights summing to 1 on a 0.1
grid (286 combinations), chosen by development MRP (ties: first in lexicographic weight order). It exists only as score matrices.

L* selection: for each of the 200 development concepts, the mean of its 10 image predictions (normalised again) is scored against
the 200 development names (names_text P1); MRP = mean rank-percentile cell (ese.things.verdict.cells). The best of the five
candidates is L* unless L_mix, then L_all, is within the margin of it (then that one). The MRP of each candidate on the 200 test images against the 200 test
names is reported as well (no EEG, not verdict information).
Outputs: <embed_out>/lowlevel_pred.npz (keys L_blur, L_phase, L_all, L_classic; [16740, 1024] float32, index.json row order, not
committed), <embed_out>/lowlevel_scores.npz ([200, 200] name scores per candidate: dev, dev_first, test; not committed) and the
summary json (chosen = L*, pred_file and score_file paths and sha256, L_mix weights).
usage: python scripts/fv_lowlevel_pred.py configs/fv_lowlevel_pred.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import csv, itertools, json, resource, sys, time
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'src'))
from fv_gate import ridge_fit, sha256
from ese.things.data import Meta
from ese.things.verdict import cells, unit
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def rowz(S):
    """standardise each row over its candidates (mean 0, sd 1)"""
    return (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)


def main():
    t0 = time.time(); c = yaml.safe_load(open(P(sys.argv[1])))
    ef = yaml.safe_load(open(P(c['eegfree_config']))); tc = yaml.safe_load(open(P(c['train_config'])))
    E = P(ef['embed_out']); idx = json.load(open(os.path.join(E, 'index.json'))); meta = Meta(P(tc['meta']))
    split, concept = np.array(idx['split']), np.array(idx['concept']); tr_all, te = split == 'train', split == 'test'
    assert tr_all.sum() == 16540 and te.sum() == 200 and list(concept[tr_all]) == meta.train_img_concept and list(concept[te]) == meta.test_img_concept
    dev = meta.dev_concepts(tc['dev']['salt'], tc['dev']['n']); dset = set(dev)
    fit = tr_all & ~np.isin(concept, dev); assert fit.sum() == 14540 and len(set(concept[fit])) == 1454

    low = np.load(os.path.join(E, 'lowlevel.npz'))
    F = {k: np.asarray(low[k] if f == 'lowlevel.npz' else np.load(os.path.join(E, f)), np.float64) for k, f in c['features'].items()}
    assert all(len(v) == len(split) for v in F.values())
    Y = np.load(os.path.join(E, c['target'])).astype(np.float64)

    rows = list(csv.DictReader(open(P(tc['targets']['names_table'])), delimiter='\t'))
    row_of = {(r['split'], r['folder']): i for i, r in enumerate(rows)}
    T = unit(np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['names_text'])).astype(np.float64))
    dev_rows = np.array([row_of[('train', f)] for f in dev]); test_folders = list(concept[te]); test_rows = np.array([row_of[('test', f)] for f in test_folders])
    assert len(set(test_folders)) == 200
    dev_img = [np.where(concept == f)[0] for f in dev]; assert all(len(x) == 10 for x in dev_img)

    res = dict(stage='D-114 section 1 low-level predictor L (no EEG)', config=sys.argv[1], fold_salt=c['fold_salt'], alphas=ef['ridge']['alphas'],
               n_fit_images=int(fit.sum()), n_fit_concepts=int(len(set(concept[fit]))), n_dev_concepts=len(dev), n_test=int(te.sum()),
               names_text=tc['targets']['names_text'], feature_dims={k: int(v.shape[1]) for k, v in F.items()},
               inputs={f: sha256(os.path.join(E, f)) for f in sorted(set(c['features'].values()) | {c['target'], 'index.json'})},
               names_text_sha256=sha256(os.path.join(P(tc['targets']['dir']), tc['targets']['names_text'])), names_table_sha256=sha256(P(tc['targets']['names_table'])),
               candidates={})
    groups = list(concept[fit]); arrays, S = {}, {}
    first = np.array([ix[0] for ix in dev_img])                                 # sensitivity rows: the concept's first image only
    for cand, spec in c['candidates'].items():
        if 'mix' in spec: continue
        info = {}
        if 'as_is' in spec: pred = unit(F[spec['as_is']]); info['features'] = [spec['as_is']]
        else:
            X = np.concatenate([F[k] for k in spec['ridge']], 1)
            f, ri = ridge_fit(X[fit], Y[fit], ef['ridge']['alphas'], groups, ef['ridge']['folds'], c['fold_salt'])
            pred = unit(f(X)); info.update(features=spec['ridge'], n_features=int(X.shape[1]), ridge=ri)
        Pd = unit(np.stack([pred[ix].mean(0) for ix in dev_img]))
        S[cand] = dict(dev=Pd @ T[dev_rows].T, dev_first=pred[first] @ T[dev_rows].T, test=pred[te] @ T[test_rows].T)
        arrays[cand] = pred.astype(np.float32); res['candidates'][cand] = info
    for cand, spec in c['candidates'].items():                                  # L_mix: weighted sum of row-standardised scores
        if 'mix' not in spec: continue
        n = int(round(1 / spec['step'])); Z = {k: {r: rowz(S[k][r]) for r in S[k]} for k in spec['mix']}
        grid = [w for w in itertools.product(range(n + 1), repeat=len(spec['mix'])) if sum(w) == n]   # lexicographic order
        g_mrp = [float(cells(sum(wi / n * Z[k]['dev'] for wi, k in zip(w, spec['mix'])), np.arange(len(dev))).mean()) for w in grid]
        bi = int(np.argmax(g_mrp)); wbest = grid[bi]                                # ties: first in lexicographic grid order
        S[cand] = {r: sum(wi / n * Z[k][r] for wi, k in zip(wbest, spec['mix'])) for r in ('dev', 'dev_first', 'test')}
        res['candidates'][cand] = dict(mix=spec['mix'], step=spec['step'], weights={k: wi / n for wi, k in zip(wbest, spec['mix'])},
                                       n_grid=len(grid), n_ties_at_best=int(sum(abs(m - g_mrp[bi]) < 1e-12 for m in g_mrp)),
                                       tie_rule='first in lexicographic grid order of the weights (mix order)',
                                       grid_dev_mrp=[[*(wi / n for wi in w), m] for w, m in zip(grid, g_mrp)])
    for cand in res['candidates']:
        info = res['candidates'][cand]; dc, fc, tc_ = (cells(S[cand][r], np.arange(len(dev) if r != 'test' else 200)) for r in ('dev', 'dev_first', 'test'))
        info.update(dev_mrp=float(dc.mean()), dev_first_mrp=float(fc.mean()), test_mrp=float(tc_.mean()), test_top1=float((tc_ == 1).mean()))
        print(cand, json.dumps({k: v for k, v in info.items() if k not in ('ridge', 'grid_dev_mrp')}), f'{time.time() - t0:.0f}s', flush=True)

    s = c['select']; mrp = {k: v['dev_mrp'] for k, v in res['candidates'].items()}; best = max(mrp, key=mrp.get)
    star = next((k for k in s['tie_order'] if mrp[best] - mrp[k] < s['margin']), best)
    fn = os.path.join(E, c['out_npz']); np.savez(fn, **arrays)
    sc = os.path.join(E, c['out_scores']); np.savez(sc, dev_folders=np.array(dev), test_folders=np.array(test_folders), dev_name_rows=dev_rows, test_name_rows=test_rows,
                                                   **{f'{k}|{r}': v[r].astype(np.float32) for k, v in S.items() for r in v})
    res['mix_weights'] = next(v['weights'] for v in res['candidates'].values() if 'weights' in v)
    res['chosen'] = star; res['pred_file'] = dict(path=os.path.join(ef['embed_out'], c['out_npz']), sha256=sha256(fn), keys=list(arrays),
                                                 shape=[len(split), 1024], dtype='float32', rows='fv_clip/index.json order (0-16539 training codes, 16540-16739 test codes)')
    res['score_file'] = dict(path=os.path.join(ef['embed_out'], c['out_scores']), sha256=sha256(sc),
                             keys='<cand>|dev (concept mean of 10 image predictions), <cand>|dev_first (first image), <cand>|test (test image); [200, 200] rows = '
                                  'dev_folders / test_folders order, columns = the same concepts\' names (names_text rows dev_name_rows / test_name_rows); '
                                  'embedding candidates: cosine; L_mix: weighted sum of row-standardised scores')
    res['select'] = dict(L_star=star, best_by_dev_mrp=best, tie_order=s['tie_order'], margin=s['margin'], dev_mrp=mrp,
                         rule='best development MRP; the first of tie_order whose development MRP is within margin of the best is chosen instead')
    res['seconds'] = round(time.time() - t0, 1); res['peak_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    out = P(c['out_json']); json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(dict(L_star=star, dev_mrp=mrp, test_mrp={k: v['test_mrp'] for k, v in res['candidates'].items()})), flush=True)


if __name__ == '__main__':
    main()
