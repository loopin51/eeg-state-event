"""D-115 section 1: blur-level pairs for the sibling-vs-foil test (no EEG).

For each of the 2,000 development images i (200 development concepts x 10 images, the concepts the dev-mode decoders never saw),
pick a sibling s (another image of the same concept) and a foil f (an image of another development concept) such that every
low-level predictor in P (blurred sigma 8 image CLIP, D-114 L_all and L_classic predictions) puts f closer to i than s:
cos(P(i), C(f)) - cos(P(i), C(s)) > 0 for all p, in the space C = original image CLIP minus the mean over the 14,540
non-development training images, L2-normalised (P(i) minus the same mean, normalised). Among the valid (s, f) the pair with the
largest minimum margin is kept (ties: smaller sibling row, then smaller foil row); images without a valid pair are dropped.
Reported: images and concepts with a valid pair, the share of pairs where C(i) itself picks s (reference ceiling), the stop rule
(fewer than 100 concepts or 400 images), and, when the sigma-4 blurred CLIP of the development images is given (scripts/fv_embed_blur.py
run folder), how many pairs it also sends to f (report-only subset).
Image ids: row = fv_clip/index.json row = training event code - 1.
usage: python scripts/fv_pairs.py configs/fv_pairs.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import json, resource, sys, time
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'src'))
from fv_gate import sha256
from ese.things.data import Meta
from ese.things.verdict import unit
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def load(E, spec):
    f, _, k = spec.partition(':')
    return (np.load(os.path.join(E, f))[k] if k else np.load(os.path.join(E, f))).astype(np.float64)


def main():
    t0 = time.time(); c = yaml.safe_load(open(P(sys.argv[1])))
    ef = yaml.safe_load(open(P(c['eegfree_config']))); tc = yaml.safe_load(open(P(c['train_config'])))
    E = P(ef['embed_out']); idx = json.load(open(os.path.join(E, 'index.json'))); meta = Meta(P(tc['meta']))
    split, concept = np.array(idx['split']), np.array(idx['concept'])
    assert list(concept[split == 'train']) == meta.train_img_concept
    dev = meta.dev_concepts(tc['dev']['salt'], tc['dev']['n'])
    fit = (split == 'train') & ~np.isin(concept, dev); assert fit.sum() == 14540
    lows = json.load(open(P(c['lowlevel_summary'])))['pred_file']
    assert sha256(os.path.join(E, os.path.basename(lows['path']))) == lows['sha256'], 'lowlevel_pred.npz differs from the D-114 summary'

    C0 = load(E, c['space']); mu = C0[fit].mean(0); C = unit(C0 - mu)
    Pr = {k: unit(load(E, v) - mu) for k, v in c['predictors'].items()}
    rows = np.where(np.isin(concept, dev))[0]; assert len(rows) == 2000              # development images, index order
    by_c = {f: rows[concept[rows] == f] for f in dev}; assert all(len(v) == 10 for v in by_c.values())
    B4 = None
    if c.get('blur4'):
        bd = P(c['blur4']); assert json.load(open(os.path.join(bd, 'rows.json')))['rows'] == rows.tolist(), 'sigma-4 rows differ from the development images'
        B4 = np.load(os.path.join(bd, 'clip_img_blur4.npy')).astype(np.float64); assert B4.shape == (2000, C0.shape[1])
        B8 = np.load(os.path.join(bd, 'clip_img_blur8.npy')).astype(np.float64); b8_check = float((unit(B8) * unit(load(E, c['predictors']['blur8'])[rows])).sum(1).min())
        B4 = unit(B4 - mu); b4_row = {r: j for j, r in enumerate(rows)}

    pairs, n_valid = [], []
    for i in rows:
        sib = by_c[concept[i]][by_c[concept[i]] != i]; foil = rows[concept[rows] != concept[i]]
        M = np.stack([(C[foil] @ Pr[k][i])[None, :] - (C[sib] @ Pr[k][i])[:, None] for k in Pr])          # [p, 9, 1990]
        m = M.min(0); ok = m > 0; n_valid.append(int(ok.sum()))
        if not ok.any(): continue
        best = np.where(ok, m, -np.inf); a = int(np.argmax(best)); si, fi = divmod(a, len(foil))             # row-major: smaller s, then f
        n_tie = int((best == best.flat[a]).sum()); s, f = int(sib[si]), int(foil[fi])
        rec = dict(i=int(i), s=s, f=f, concept=str(concept[i]), foil_concept=str(concept[f]), min_margin=float(m[si, fi]),
                   margins={k: float(M[j, si, fi]) for j, k in enumerate(Pr)}, n_valid_combos=n_valid[-1], n_ties=n_tie,
                   orig_picks_s=bool(C[i] @ C[s] > C[i] @ C[f]))
        if B4 is not None: rec['blur4_picks_f'] = bool(B4[b4_row[i]] @ C[f] > B4[b4_row[i]] @ C[s])
        pairs.append(rec)

    n_img, n_con = len(pairs), len({p['concept'] for p in pairs})
    res = dict(stage='D-115 section 1 blur-level pairs (no EEG)', config=sys.argv[1], predictors=list(c['predictors']),
               space=c['space'], mean_over='14,540 non-development training images', n_dev_images=len(rows), n_dev_concepts=len(dev),
               n_images_with_pair=n_img, n_concepts_with_pair=n_con,
               per_concept_images=dict(sorted({f: sum(p['concept'] == f for p in pairs) for f in dev}.items())),
               orig_picks_s=dict(n=int(sum(p['orig_picks_s'] for p in pairs)), share=float(np.mean([p['orig_picks_s'] for p in pairs])) if pairs else None),
               min_margin=dict(median=float(np.median([p['min_margin'] for p in pairs])) if pairs else None, min=float(min(p['min_margin'] for p in pairs)) if pairs else None),
               n_valid_combos=dict(zero=int(sum(v == 0 for v in n_valid)), median=float(np.median(n_valid))),
               ties_at_best=int(sum(p['n_ties'] > 1 for p in pairs)),
               stop_rule=dict(min_concepts=c['min_concepts'], min_images=c['min_images'], eeg_stage_open=bool(n_con >= c['min_concepts'] and n_img >= c['min_images'])),
               blur4=(dict(folder=c['blur4'], sha256=sha256(os.path.join(P(c['blur4']), 'clip_img_blur4.npy')), n_picks_f=int(sum(p['blur4_picks_f'] for p in pairs)),
                           n_pairs=n_img, sigma8_rerun_min_cosine_to_clip_img_blur=b8_check)
                      if B4 is not None else 'not computed'),
               inputs={**{v.partition(':')[0]: sha256(os.path.join(E, v.partition(':')[0])) for v in list(c['predictors'].values()) + [c['space']]},
                       'index.json': sha256(os.path.join(E, 'index.json'))},
               ids='row = fv_clip/index.json row = training event code - 1', pairs=pairs)
    res['seconds'] = round(time.time() - t0, 1); res['peak_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    out = P(c['out']); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('pairs', 'per_concept_images', 'inputs')}), flush=True)


if __name__ == '__main__':
    main()
