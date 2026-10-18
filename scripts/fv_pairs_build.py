"""D-115 section 1 and amendment 1: blur-level pairs for the sibling-vs-foil test (no EEG).

For each of the 2,000 development images i (200 development concepts x 10 images, the concepts the dev-mode decoders never saw),
a combination of a sibling s (another image of the same concept) and a foil f (an image of another development concept) is valid
when every low-level predictor in P (blurred sigma 8 image CLIP, D-114 L_all and L_classic predictions) puts f closer to i than s:
cos(P(i), C(f)) - cos(P(i), C(s)) > 0 for all p, in the space C = original image CLIP minus the mean over the 14,540
non-development training images, L2-normalised (P(i) minus the same mean, normalised). Images without a valid combination are
dropped. Two pair lists over the same images:
- main (amendment 1): one valid combination drawn at random, rng = default_rng(pair_seed), images in index order, one
  rng.integers(n_valid) per image over its valid combinations in (sibling row, foil row) order;
- maxmargin (report only, the 08d0365 file): the valid combination with the largest minimum margin over P (ties: smaller sibling
  row, then smaller foil row).
Reported per list: images and concepts, r = share of pairs where C(i) picks s (feasibility (1), r >= r_min on the main list), the
size rule (feasibility (2)), pair accuracy A_H = share where the held-out predictor H(i) (scripts/fv_holdout.py) picks s, hub
tendency of f and s (mean cosine in C to the other 1,999 development images), foil reuse counts, and the sigma-4 blurred CLIP
(report only) share that also picks f. Image ids: row = fv_clip/index.json row = training event code - 1.
usage: python scripts/fv_pairs_build.py configs/fv_pairs_build.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import collections, json, resource, sys, time
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
    co = P(c['center_out']); os.makedirs(os.path.dirname(co), exist_ok=True); np.save(co, mu)
    Pr = {k: unit(load(E, v) - mu) for k, v in c['predictors'].items()}
    rows = np.where(np.isin(concept, dev))[0]; assert len(rows) == 2000              # development images, index order
    pos = {int(r): j for j, r in enumerate(rows)}
    by_c = {f: rows[concept[rows] == f] for f in dev}; assert all(len(v) == 10 for v in by_c.values())

    hp = P(c['holdout']['out']); H = {}
    if os.path.exists(hp):
        z = np.load(hp); assert z['rows'].tolist() == rows.tolist(), 'holdout rows differ'
        H = {k: unit(z[k].astype(np.float64)) for k in ('H_a', 'H_b', 'H_c') if k in z.files}
    B4, b8_check = None, None
    if c.get('blur4'):
        bd = P(c['blur4']); assert json.load(open(os.path.join(bd, 'rows.json')))['dev_rows'] == rows.tolist(), 'sigma-4 rows differ'
        B4 = unit(np.load(os.path.join(bd, 'clip_img_blur4_dev.npy')).astype(np.float64) - mu)
        b8_check = float((unit(np.load(os.path.join(bd, 'clip_img_blur8_dev.npy')).astype(np.float64)) * unit(load(E, c['predictors']['blur8'])[rows])).sum(1).min())
    hub = (C[rows] @ C[rows].T).sum(1); hub = (hub - 1) / (len(rows) - 1)           # mean cosine to the other development images

    rng = np.random.default_rng(c['pair_seed']); lists = {'main': [], 'maxmargin': []}; n_valid = []
    for i in rows:
        sib = by_c[concept[i]][by_c[concept[i]] != i]; foil = rows[concept[rows] != concept[i]]
        M = np.stack([(C[foil] @ Pr[k][i])[None, :] - (C[sib] @ Pr[k][i])[:, None] for k in Pr])          # [p, 9, 1990]
        m = M.min(0); ok = m > 0; n_valid.append(int(ok.sum()))
        if not ok.any(): continue
        best = np.where(ok, m, -np.inf); a_mm = int(np.argmax(best)); n_tie = int((best == best.flat[a_mm]).sum())
        flat_ok = np.flatnonzero(ok); a_rd = int(flat_ok[rng.integers(len(flat_ok))])                     # row-major = (sibling, foil) order
        for name, a in (('maxmargin', a_mm), ('main', a_rd)):
            si, fi = divmod(a, len(foil)); s, f = int(sib[si]), int(foil[fi])
            rec = dict(i=int(i), s=s, f=f, concept=str(concept[i]), foil_concept=str(concept[f]), min_margin=float(m[si, fi]),
                       margins={k: float(M[j, si, fi]) for j, k in enumerate(Pr)}, n_valid_combos=n_valid[-1],
                       orig_picks_s=bool(C[i] @ C[s] > C[i] @ C[f]), hub_s=float(hub[pos[s]]), hub_f=float(hub[pos[f]]))
            if name == 'maxmargin': rec['n_ties'] = n_tie
            for k, Hk in H.items(): rec[f'{k}_picks_s'] = bool(Hk[pos[i]] @ C[s] > Hk[pos[i]] @ C[f])
            if B4 is not None: rec['blur4_picks_f'] = bool(B4[pos[i]] @ C[f] > B4[pos[i]] @ C[s])
            lists[name].append(rec)

    base = dict(config=sys.argv[1], predictors=list(c['predictors']), space=c['space'],
                center=dict(source=c['space'], n_images=int(fit.sum()), file=c['center_out'], dtype='float64', sha256_of_mean=sha256(co)),
                inputs={**{v.partition(':')[0]: sha256(os.path.join(E, v.partition(':')[0])) for v in list(c['predictors'].values()) + [c['space']]},
                        'index.json': sha256(os.path.join(E, 'index.json'))},
                holdout=dict(path=c['holdout']['out'], sha256=sha256(hp), keys=sorted(H)) if H else 'not available',
                n_dev_images=len(rows), n_dev_concepts=len(dev), n_valid_combos=dict(zero=int(sum(v == 0 for v in n_valid)), median=float(np.median(n_valid))),
                ids='row = fv_clip/index.json row = training event code - 1')
    for name, pairs in lists.items():
        n_img, n_con = len(pairs), len({p['concept'] for p in pairs}); r = float(np.mean([p['orig_picks_s'] for p in pairs]))
        reuse = collections.Counter(p['f'] for p in pairs)
        res = dict(stage=f'D-115 section 1 / amendment 1 pairs: {name}' + (' (report only)' if name == 'maxmargin' else ' (main)'),
                   selection='random valid combination, seed %d' % c['pair_seed'] if name == 'main' else 'largest minimum margin over P (ties: smaller sibling row, then foil row)',
                   **base, n_images_with_pair=n_img, n_concepts_with_pair=n_con,
                   orig_picks_s=dict(n=int(sum(p['orig_picks_s'] for p in pairs)), share=r),
                   A_H={k: float(np.mean([p[f'{k}_picks_s'] for p in pairs])) for k in H},
                   hub=dict(f_mean=float(np.mean([p['hub_f'] for p in pairs])), s_mean=float(np.mean([p['hub_s'] for p in pairs])),
                            f_median=float(np.median([p['hub_f'] for p in pairs])), s_median=float(np.median([p['hub_s'] for p in pairs])),
                            all_dev_mean=float(hub.mean()), definition='mean cosine in C to the other 1,999 development images'),
                   foil_reuse=dict(n_distinct_f=len(reuse), max=max(reuse.values()), histogram={str(k): v for k, v in sorted(collections.Counter(reuse.values()).items())}),
                   min_margin=dict(median=float(np.median([p['min_margin'] for p in pairs])), min=float(min(p['min_margin'] for p in pairs))),
                   blur4=(dict(folder=c['blur4'], sha256=sha256(os.path.join(P(c['blur4']), 'clip_img_blur4_dev.npy')), n_picks_f=int(sum(p['blur4_picks_f'] for p in pairs)),
                               n_pairs=n_img, sigma8_rerun_min_cosine_to_clip_img_blur=b8_check) if B4 is not None else 'not computed'),
                   per_concept_images=dict(sorted({f: sum(p['concept'] == f for p in pairs) for f in dev}.items())))
        if name == 'main':
            res['feasibility'] = dict(r=r, r_min=c['r_min'], pass_r=bool(r >= c['r_min']), n_concepts=n_con, n_images=n_img,
                                      min_concepts=c['min_concepts'], min_images=c['min_images'], pass_size=bool(n_con >= c['min_concepts'] and n_img >= c['min_images']),
                                      note='gates (3) decoder-level signal and (4) leave-one-out blur simulations: eeg-b1 gate.json')
        else:
            res['ties_at_best'] = int(sum(p['n_ties'] > 1 for p in pairs))
        res['seconds'] = round(time.time() - t0, 1); res['peak_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2); res['pairs'] = pairs
        out = P(c['out_main'] if name == 'main' else c['out_maxmargin']); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
        print(name, json.dumps({k: v for k, v in res.items() if k not in ('pairs', 'per_concept_images', 'inputs', 'center')}), flush=True)


if __name__ == '__main__':
    main()
