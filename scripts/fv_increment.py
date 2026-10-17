"""F-VI (D-114 sections 2-3, db3c843; amendments 1 0736f9c, 2 5af193e, 3 8373346, 4 ef1778d): does the F-V decoder output
add name information over the name ranking of a low-level predictor? Two hypotheses, Holm: I1 (base L*, chosen by eeg-a4)
and I2 (base L_classic). Server CPU only (gpuq). Order (coordination session, option A): dev -> commit -> sim -> commit -> test.

  Combination, per row: s_B = row-standardised score of the base over the 200 names (L_mix: weighted sum of the row-standardised
  scores of the four embedding candidates, standardised again), s_E = row-standardised cos(z, names); s = (1 - w) s_B + w s_E.
  delta = MRP(combined) - MRP(base). Cells: ese.things.verdict.cells.

  dev  : per hypothesis, w = argmax over w_grid of the pooled combined MRP on development rows (bm dev runs, seeds 20261801/02
         normalised mean, ses123, set A, targets 1 2 3 4 5 9; L row = normalised mean of the predictions of the concept's 10 images;
         ties -> smaller w; w = 0 closes that hypothesis). Report: curves per target, sensitivity curve (L from the first image),
         second low-level predictor (I1: the non-L* candidate whose combination with L* raises the EEG-free development MRP most,
         amendment 1; I2: L_phase, amendment 4) and its own w_LL on the 200 development concepts. Cross-check of the L score rows
         against eeg-a4's lowlevel_scores.npz. Writes dev_w.json.
  sim  : no test EEG, 10 subjects x 200 concepts as the test, each hypothesis at its w:
         (i)   z = isotropic Gaussian rows, J1 stand-in = independent Gaussian rows (n_sim draws);
         (ii)  z = one fixed vector in every row (mean development decoder output per target / pooled, mean label-shuffle
               development output per target / pooled): per vector delta, 5th percentile, p, I alone, J1b (amendment 2);
         (iii) z = the second low-level predictor's score rows (report only);
         planted-signal MDE: z = a * original-image CLIP of the true concept + g / sqrt(d); per hypothesis a bisected until the
         Holm adoption rate is 80%; MDE = mean delta there, m80 = mean z-alone test MRP there ("relative to a signal correlated
         with L*", amendment 2). stop (per hypothesis, amendment 4): case (i) unadjusted joint rate (p <= 0.05, point, J1, J1b)
         > 5%, or a case (ii) joint pass. Writes sim.json.
  test : once. z = V2 seed-mean embeddings (ses123, avg20). One crossed subject x concept resample index set (1,000, seed
         20262011) for every statistic of both hypotheses: p = (#{delta*_b <= 0} + 1) / 1001, Holm (smallest p vs 0.025, then
         0.05; a closed hypothesis keeps p = 1). Adopted: Holm-significant, point >= min_point (I1 0.005, I2 0.01), J1 (z -> V2
         label-shuffle output) and J1b (z -> the subject's mean z row) under rule N (D-112 amendment 5) at the same w.
         Report only: delta_top1, delta_MRR (same w and resamples), delta_centred, k0, single trials, the test w curve, delta per
         L candidate (w fixed), delta per subject, delta_LL. Refuses to run twice.
Config smoke: true replaces the development EEG rows by EEG-free rows; it is a code check and never reported.

usage: python scripts/fv_increment.py configs/fv_increment.yaml --stage {dev,sim,test} [--config-override YAML]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, time, hashlib, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.things.data import Meta
from ese.things import verdict as V
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def load_cfg(path, override):
    c = yaml.safe_load(open(P(path)))
    if override:
        def merge(a, b):
            for k, v in b.items(): a[k] = merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
            return a
        c = merge(c, yaml.safe_load(open(P(override))))
    return c


# ---------------------------------------------------------------- scores, cells, resampling
def zrow(S):
    S = np.asarray(S, np.float64); return (S - S.mean(-1, keepdims=True)) / (S.std(-1, keepdims=True) + 1e-12)


def combined(ZB, ZE, w): return (1 - w) * ZB + w * ZE                          # both row-standardised


def per_subject(f, SE):
    SE = np.asarray(SE, np.float64); return f(SE) if SE.ndim == 2 else np.stack([f(x) for x in SE])


def dcells(ZB, SE, w, true, kind='mrp'):
    """ZB [n, K] standardised base rows; SE [n, K] or [ns, n, K] raw z-side score rows -> delta cells (combined - base)"""
    m = {'mrp': V.cells, 'top1': lambda S, t: (V.ranks(S, t) == 0).astype(np.float64), 'mrr': lambda S, t: 1.0 / (V.ranks(S, t) + 1)}[kind]
    b = m(ZB, true); return per_subject(lambda x: m(combined(ZB, zrow(x), w), true) - b, SE)


def const(SE): SE = np.asarray(SE, np.float64); return np.broadcast_to(SE.mean(-2, keepdims=True), SE.shape)


def resample(ns, nc, nb, rng):
    """crossed subject x concept resampling as multiplicity counts (same draw order as ese.things.verdict.pigeonhole)"""
    Cs, Cc = np.zeros((nb, ns)), np.zeros((nb, nc))
    for b in range(nb): Cs[b] = np.bincount(rng.integers(0, ns, ns), minlength=ns); Cc[b] = np.bincount(rng.integers(0, nc, nc), minlength=nc)
    return Cs, Cc


def bmeans(M, R): Cs, Cc = R; return np.einsum('bs,sc,bc->b', Cs, np.asarray(M, np.float64), Cc) / (Cs.shape[1] * Cc.shape[1])


def pval(bm): return float((np.sum(bm <= 0) + 1) / (len(bm) + 1))


def holm(ps, alphas):
    """ps {h: p} -> {h: significant}; smallest p first against alphas[0], then alphas[1], ...; stops at the first failure"""
    sig, go = {}, True
    for j, h in enumerate(sorted(ps, key=lambda k: (ps[k], k))): sig[h] = bool(go and ps[h] <= alphas[j]); go = sig[h]
    return sig


def judge(T, R, c):
    """T {h: dict(D, Dsh (None: skipped), Dc) or None (closed)}, R resample counts -> per-hypothesis results with Holm adoption"""
    rl = c['rules']; out, ps = {}, {}
    for h, t in T.items():
        if t is None: out[h] = dict(status='closed (w = 0)'); ps[h] = 1.0; continue
        D = t['D']; bm = bmeans(D, R); pt = float(D.mean()); mn = c['hypotheses'][h]['min_point']; r = dict(point=pt, lower=float(np.percentile(bm, rl['lower_pct'])), p=pval(bm), min_point=mn)
        r['J1'] = None if t['Dsh'] is None else dict(V.control_rule(t['Dsh'], bmeans(t['Dsh'], R), pt, rl['control_ratio'], rl['control_lower_pct']), delta_shuf=float(t['Dsh'].mean()))
        r['J1b'] = dict(V.control_rule(t['Dc'], bmeans(t['Dc'], R), pt, rl['control_ratio'], rl['control_lower_pct']), delta_const=float(t['Dc'].mean()))
        r['controls'] = bool((r['J1'] is None or r['J1']['passed']) and r['J1b']['passed'])
        r['I_alone'] = bool(r['p'] <= rl['alpha'] and pt >= mn); r['joint_unadjusted'] = bool(r['I_alone'] and r['controls']); out[h] = r; ps[h] = r['p']
    sig = holm(ps, c['holm_alphas'])
    for h, r in out.items():
        if 'p' in r: r['holm_significant'] = sig[h]; r['adopted'] = bool(sig[h] and r['point'] >= r['min_point'] and r['controls'])
    return out


# ---------------------------------------------------------------- low-level predictors
def L_raw(emb, Tn, cand, mixw):
    if cand == 'L_mix': return sum(float(wk) * zrow(emb[k] @ Tn.T) for k, wk in mixw.items() if wk)
    return emb[cand] @ Tn.T


def lowlevel(c):
    pj = P(c['lowlevel']['predictor_json']); lp = json.load(open(pj)); pf = lp['pred_file']; f = P(pf['path']); h = fsha(f)
    assert h == pf['sha256'], f'L prediction file sha256 {h} differs from {c["lowlevel"]["predictor_json"]}'
    Z = np.load(f); cands = c['lowlevel']['candidates']; pred = {k: V.unit(np.asarray(Z[k], np.float64)) for k in cands}
    assert all(v.shape == (c['lowlevel']['n_train_rows'] + 200, c['z_dim']) for v in pred.values()), {k: v.shape for k, v in pred.items()}
    mixw = {k: float(lp['mix_weights'][k]) for k in cands}; assert abs(sum(mixw.values()) - 1) < 1e-9, mixw; assert lp['chosen'] in cands + ['L_mix'], lp['chosen']
    return pred, lp['chosen'], mixw, dict(predictor_json=c['lowlevel']['predictor_json'], predictor_json_sha256=fsha(pj), pred_file=pf['path'], pred_sha256=h, chosen=lp['chosen'], mix_weights=mixw)


def all_cands(c): return list(c['lowlevel']['candidates']) + ['L_mix']


def bases(c, lstar): return {h: (lstar if v['base'] == 'Lstar' else v['base']) for h, v in c['hypotheses'].items()}


def mrp_curve(ZB, ZE, true, grid): return np.array([V.cells(combined(ZB, ZE, w), true).mean() for w in grid])


def choose(curve, grid):
    i = int(np.flatnonzero(curve >= curve.max() - 1e-12)[0]); return float(grid[i]), i                 # ties -> smaller w


def dev_names(tc, meta):
    from fv_train import name_rows
    dev_c = meta.dev_concepts(tc['dev']['salt'], tc['dev']['n'])
    return dev_c, V.unit(np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['names_text']))[name_rows(tc, meta, dev_c)])


def test_names(tc, meta):
    from fv_verdict import names
    trow, _, _ = names(tc, meta); return V.unit(np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['names_text']))[trow])


def dev_runs(c, tc, t, shuffle):
    d = c['dev']; out = []
    for s in ([d['shuffle_seed']] if shuffle else d['seeds']):
        m = [x for x in glob.glob(os.path.join(P(tc['out_root']), f"*_fv_dev_{d['candidate']}_s{s}_t{t:02d}{'_shuffle' if shuffle else ''}")) if os.path.exists(os.path.join(x, 'dev_scores.npz'))]
        assert len(m) == 1, (t, s, shuffle, m); out.append(m[0])
    return out


def dev_eeg(c, tc, t, shuffle, rng):
    """normalised (seed-mean) development output rows of target t and their true columns; smoke: EEG-free rows"""
    d = c['dev']; ds = dev_runs(c, tc, t, shuffle); Z = [np.load(os.path.join(x, 'dev_scores.npz')) for x in ds]; true = Z[0][f"{d['set']}_true"]
    assert all(np.array_equal(z[f"{d['set']}_true"], true) for z in Z), 'seeds evaluated different concepts'
    E = V.seed_mean([z[f"{d['set']}_E_{d['cond']}"] for z in Z])
    if c['smoke']:
        E = rng.standard_normal(E.shape) / np.sqrt(E.shape[1])
        if not shuffle: E = E + c.get('smoke_signal', 0.0) * smoke_concept_clip(c, tc)[true]
        E = V.unit(E)
    return E, true, [os.path.basename(x) for x in ds]


def smoke_concept_clip(c, tc):
    meta = Meta(P(tc['meta'])); dev_c = meta.dev_concepts(tc['dev']['salt'], tc['dev']['n']); ic = np.array(meta.train_img_concept)
    U = np.load(P(c['lowlevel']['clip_img']), mmap_mode='r'); return V.unit(np.stack([np.asarray(U[np.flatnonzero(ic == x)], np.float64).mean(0) for x in dev_c]))


# ---------------------------------------------------------------- dev
def stage_dev(c, vc, tc, meta, a):
    pred, lstar, mixw, prov = lowlevel(c); d = c['dev']; grid = np.array(c['w_grid'], np.float64); out = P(c['out_dir']); os.makedirs(out, exist_ok=True); B = bases(c, lstar)
    dev_c, Tdev = dev_names(tc, meta); ic = np.array(meta.train_img_concept); nt = c['lowlevel']['n_train_rows']
    rows = [np.flatnonzero(ic == x) for x in dev_c]; assert all(len(r) == d['images_per_concept'] for r in rows)
    emb10 = {k: V.unit(np.stack([Y[r].mean(0) for r in rows])) for k, Y in pred.items()}; emb1 = {k: Y[[r[0] for r in rows]] for k, Y in pred.items()}
    ZB10 = {k: zrow(L_raw(emb10, Tdev, k, mixw)) for k in all_cands(c)}; ZB1 = {k: zrow(L_raw(emb1, Tdev, k, mixw)) for k in all_cands(c)}
    xc = None
    if c['lowlevel'].get('scores_file'):                                         # cross-check against eeg-a4's score matrices (raw, before the combination)
        sf = c['lowlevel']['scores_file']; assert fsha(P(sf['path'])) == sf['sha256'], 'L score file sha256 differs'; SF = np.load(P(sf['path'])); Tt = test_names(tc, meta)
        mine = {f'{k}|{part}': L_raw(e, T, k, mixw) for k in all_cands(c) for part, e, T in (('dev', emb10, Tdev), ('dev_first', emb1, Tdev), ('test', {q: Y[nt:] for q, Y in pred.items()}, Tt))}
        xc = {k: float(np.abs(v - SF[k]).max()) for k, v in mine.items()}; assert max(xc.values()) < 1e-4, xc
    t200 = np.arange(len(dev_c)); ll = {}                                         # second low-level predictors, EEG-free (before any EEG row is read)
    for h, v in c['hypotheses'].items():
        cs = [k for k in all_cands(c) if k != B[h]] if v['second'] == 'auto' else [v['second']]; ll[h] = {}
        for k in cs:
            cu = mrp_curve(ZB10[B[h]], ZB10[k], t200, grid); wk, j = choose(cu, grid); ll[h][k] = dict(curve=cu.tolist(), w_LL=wk, gain=float(cu[j] - cu[0]))
    rng = np.random.default_rng([c['seeds']['sim'], 0]); ZE, TR, runs = [], [], {}
    for t in d['targets']:
        E, true, runs[str(t)] = dev_eeg(c, tc, t, False, rng); ZE.append(zrow(E @ Tdev.T)); TR.append(true)
    ZEall, true = np.concatenate(ZE), np.concatenate(TR); hyp = {}
    for h in c['hypotheses']:
        b = B[h]; curve = mrp_curve(ZB10[b][true], ZEall, true, grid); w, i = choose(curve, grid); sens = mrp_curve(ZB1[b][true], ZEall, true, grid)
        L2 = max(ll[h], key=lambda k: (ll[h][k]['gain'], -all_cands(c).index(k)))
        hyp[h] = dict(base=b, w=w, closed=bool(w == 0.0), curve=curve.tolist(), mrp_base=float(curve[0]), mrp_eeg=float(curve[-1]), mrp_at_w=float(curve[i]), delta_at_w=float(curve[i] - curve[0]),
                      per_target={str(t): mrp_curve(ZB10[b][tr], ze, tr, grid).tolist() for t, ze, tr in zip(d['targets'], ZE, TR)},
                      sensitivity_first_image=dict(curve=sens.tolist(), w_argmax=choose(sens, grid)[0], mrp_base=float(sens[0]),
                                                   note='report only: L row = prediction of the first image of the concept; its ceiling is lower than the test rows, so its w can be larger (amendment 3)'),
                      second_predictor=dict(L2=L2, w_LL=ll[h][L2]['w_LL'], candidates=ll[h], rows='200 development concepts, EEG-free'))
    res = dict(stage='dev', smoke=bool(c['smoke']), lowlevel=prov, Lstar=lstar, targets=d['targets'], seeds=d['seeds'], set=d['set'], cond=d['cond'], n_rows=int(len(true)), w_grid=grid.tolist(),
               hypotheses=hyp, dev_mrp_L_200={k: float(V.cells(ZB10[k], t200).mean()) for k in all_cands(c)}, scores_crosscheck_max_abs=xc, runs=runs,
               other_L_curves={k: mrp_curve(ZB10[k][true], ZEall, true, grid).tolist() for k in all_cands(c)})
    fn = os.path.join(out, 'dev_w.json'); json.dump(res, open(fn, 'w'), indent=1)
    print(json.dumps(dict(smoke=res['smoke'], Lstar=lstar, **{h: dict(base=v['base'], w=v['w'], closed=v['closed'], L2=v['second_predictor']['L2']) for h, v in hyp.items()})), flush=True)


# ---------------------------------------------------------------- sim (no test EEG)
def setup(c, tc, meta, dw):
    pred, lstar, mixw, prov = lowlevel(c); assert lstar == dw['Lstar'] and prov['pred_sha256'] == dw['lowlevel']['pred_sha256'], 'L* or prediction file changed since dev'
    Tn = test_names(tc, meta); nt = c['lowlevel']['n_train_rows']; emb_t = {k: Y[nt:] for k, Y in pred.items()}
    ZB = {k: zrow(L_raw(emb_t, Tn, k, mixw)) for k in all_cands(c)}; return pred, lstar, mixw, prov, Tn, emb_t, ZB


def stage_sim(c, vc, tc, meta, a):
    out = P(c['out_dir']); dw = json.load(open(os.path.join(out, 'dev_w.json'))); assert dw['smoke'] == bool(c['smoke'])
    pred, lstar, mixw, prov, Tn, emb_t, ZB = setup(c, tc, meta, dw); H = dw['hypotheses']; live = [h for h in H if not H[h]['closed']]; assert live, 'every hypothesis closed at the development stage'
    ns, n, dim = len(vc['subjects']), 200, c['z_dim']; true = np.arange(n); rl = c['rules']; nb = c['n_boot']; t0 = time.time()
    res = dict(stage='sim', smoke=bool(c['smoke']), lowlevel=prov, Lstar=lstar, w={h: H[h]['w'] for h in H}, base={h: H[h]['base'] for h in H}, closed=[h for h in H if H[h]['closed']],
               subjects=ns, n_boot=nb, seed=c['seeds']['sim'], z_dim=dim, mrp_base_test={h: float(V.cells(ZB[H[h]['base']], true).mean()) for h in H})
    gauss = lambda g: V.unit(g.standard_normal((ns, n, dim))) @ Tn.T

    def tri(SE, SS):
        return {h: (None if H[h]['closed'] else dict(D=dcells(ZB[H[h]['base']], SE, H[h]['w'], true), Dsh=None if SS is None else dcells(ZB[H[h]['base']], SS, H[h]['w'], true),
                                                       Dc=dcells(ZB[H[h]['base']], const(SE), H[h]['w'], true))) for h in H}
    # (i) isotropic Gaussian
    g = np.random.default_rng([c['seeds']['sim'], 1]); rec = {h: {k: [] for k in ('delta', 'delta_shuf', 'delta_const', 'I_alone', 'J1', 'J1b', 'joint_unadjusted', 'adopted')} for h in live}
    for _ in range(c['n_sim']):
        SE = gauss(g); SS = gauss(g); J = judge(tri(SE, SS), resample(ns, n, nb, np.random.default_rng(g.integers(1 << 62))), c)
        for h in live:
            r = J[h]
            for k, v in (('delta', r['point']), ('delta_shuf', r['J1']['delta_shuf']), ('delta_const', r['J1b']['delta_const']), ('I_alone', r['I_alone']), ('J1', r['J1']['passed']),
                         ('J1b', r['J1b']['passed']), ('joint_unadjusted', r['joint_unadjusted']), ('adopted', r['adopted'])): rec[h][k].append(float(v))
    res['case_i_gaussian'] = {h: dict(n_sim=c['n_sim'], rates={k: float(np.mean(rec[h][k])) for k in ('I_alone', 'J1', 'J1b', 'joint_unadjusted', 'adopted')},
                                      delta=dict(mean=float(np.mean(rec[h]['delta'])), p5=float(np.percentile(rec[h]['delta'], 5)), p95=float(np.percentile(rec[h]['delta'], 95))),
                                      delta_shuf_mean=float(np.mean(rec[h]['delta_shuf'])), delta_const_mean=float(np.mean(rec[h]['delta_const']))) for h in live}
    res['case_i_gaussian']['seconds'] = time.time() - t0
    # (ii) one fixed vector in every row (development outputs; no rate, amendment 2)
    gs = np.random.default_rng([c['seeds']['sim'], 0]); vecs = {}
    for shuf, tag in ((False, 'dev_decoder_mean'), (True, 'dev_shuffle_mean')):
        allE = []
        for t in c['dev']['targets']:
            E, _, _ = dev_eeg(c, tc, t, shuf, gs); vecs[f'{tag}_t{t:02d}'] = E.mean(0); allE.append(E)
        vecs[f'{tag}_pooled'] = np.concatenate(allE).mean(0)
    case2 = {}
    for j, (k, v) in enumerate(vecs.items()):
        SE = np.broadcast_to((V.unit(v) @ Tn.T)[None, None], (ns, n, n)); J = judge(tri(SE, None), resample(ns, n, nb, np.random.default_rng([c['seeds']['sim'], 2, j])), c)
        case2[k] = dict(norm_of_mean_row=float(np.linalg.norm(v)), **{h: dict(delta=J[h]['point'], p5=J[h]['lower'], p=J[h]['p'], I_alone=J[h]['I_alone'], J1b=J[h]['J1b']['passed'],
                                                                              joint=bool(J[h]['I_alone'] and J[h]['J1b']['passed'])) for h in live})
    res['case_ii_fixed_vector'] = dict(vectors=case2, any_joint={h: bool(any(x[h]['joint'] for x in case2.values())) for h in live},
                                       note='amendment 2: no rate; the J1b input z_bar is the vector itself, so delta_const = delta and the joint pass is structurally impossible; the hub-cancelling path is read from I alone')
    # (iii) second low-level predictor as z (report only)
    res['case_iii_second_predictor'] = {}
    for h in live:
        L2 = H[h]['second_predictor']['L2']; SE = np.broadcast_to(L_raw(emb_t, Tn, L2, mixw)[None], (ns, n, n)); g3 = np.random.default_rng([c['seeds']['sim'], 3, live.index(h)])
        SS = gauss(g3); T = {q: None for q in H}; b = ZB[H[h]['base']]; T[h] = dict(D=dcells(b, SE, H[h]['w'], true), Dsh=dcells(b, SS, H[h]['w'], true), Dc=dcells(b, const(SE), H[h]['w'], true))
        r = judge(T, resample(ns, n, nb, np.random.default_rng(g3.integers(1 << 62))), c)[h]
        res['case_iii_second_predictor'][h] = dict(L2=L2, w=H[h]['w'], point=r['point'], lower=r['lower'], p=r['p'], I_alone=r['I_alone'], J1=r['J1']['passed'], J1b=r['J1b']['passed'],
                                                    joint_unadjusted=r['joint_unadjusted'], note='report only, not a stop condition; J1 input = Gaussian stand-in; judged alone (no Holm partner)')
    # planted-signal MDE and m80 per hypothesis (signal = original-image CLIP of the true concept)
    U = V.unit(np.load(P(c['lowlevel']['clip_img']), mmap_mode='r')[c['lowlevel']['n_train_rows']:].astype(np.float64)); assert U.shape == (n, dim); memo = {}

    def rate(amp):
        if amp in memo: return memo[amp]
        acc = {h: dict(adopted=[], joint=[], delta=[]) for h in live}; mz = []
        for b in range(c['n_mde']):
            gb = np.random.default_rng([c['seeds']['sim'], 4, b]); G = gb.standard_normal((ns, n, dim)) / np.sqrt(dim); SS = gauss(gb)
            SE = V.unit(amp * U[None] + G) @ Tn.T; J = judge(tri(SE, SS), resample(ns, n, nb, np.random.default_rng(gb.integers(1 << 62))), c)
            mz.append(float(np.mean([V.cells(x, true).mean() for x in SE])))
            for h in live: acc[h]['adopted'].append(J[h]['adopted']); acc[h]['joint'].append(J[h]['joint_unadjusted']); acc[h]['delta'].append(J[h]['point'])
        memo[amp] = dict(m_z=float(np.mean(mz)), **{h: {k: float(np.mean(v)) for k, v in acc[h].items()} for h in live}); return memo[amp]
    t1 = time.time(); res['mde'] = {}
    for h in live:
        lo, hi = 0.0, 0.05; rh = rate(hi)
        while rh[h]['adopted'] < rl['mde_power'] and hi < 64: lo, hi = hi, hi * 2; rh = rate(hi)
        reached = rh[h]['adopted'] >= rl['mde_power']
        for _ in range(c['mde_bisect'] if reached else 0):
            mid = (lo + hi) / 2; rm = rate(mid)
            if rm[h]['adopted'] >= rl['mde_power']: hi, rh = mid, rm
            else: lo = mid
        res['mde'][h] = dict(reached=bool(reached), mde=rh[h]['delta'] if reached else None, m80=rh['m_z'] if reached else None, amplitude=hi if reached else None,
                             adoption_rate_at_amplitude=rh[h]['adopted'] if reached else None, bracket=[lo, hi] if reached else None,
                             max_adoption_rate=float(max(m[h]['adopted'] for m in memo.values())), delta_ceiling=float(1 - res['mrp_base_test'][h]),
                             m80_vs_decoder_alone=None if not reached else ('m80 > 0.7707' if rh['m_z'] > 0.7707 else 'm80 <= 0.7707'))
    res['mde_trace'] = [dict(a=k, **v) for k, v in sorted(memo.items())]
    res['mde_common'] = dict(n_per_amplitude=c['n_mde'], power=rl['mde_power'], model='z = a * unit original-image CLIP (true concept) + g / sqrt(d)', rate='Holm adoption of the hypothesis',
                             label='relative to a signal correlated with L* (amendment 2)', seconds=time.time() - t1)
    res['stop'] = {h: bool(res['case_i_gaussian'][h]['rates']['joint_unadjusted'] > rl['sim_stop_rate'] or res['case_ii_fixed_vector']['any_joint'][h]) for h in live}
    res['stop_any'] = bool(any(res['stop'].values()))
    json.dump(res, open(os.path.join(out, 'sim.json'), 'w'), indent=1)
    print(json.dumps(dict(smoke=res['smoke'], case_i={h: res['case_i_gaussian'][h]['rates'] for h in live}, case_ii_any_joint=res['case_ii_fixed_vector']['any_joint'],
                          mde={h: {k: res['mde'][h][k] for k in ('reached', 'mde', 'm80')} for h in live}, stop=res['stop'])), flush=True)


# ---------------------------------------------------------------- test (once)
def stage_test(c, vc, tc, meta, a):
    out = P(c['out_dir']); fn = os.path.join(out, 'test.json'); assert not os.path.exists(fn), f'{fn} exists: the test is computed once'
    dw = json.load(open(os.path.join(out, 'dev_w.json'))); sm = json.load(open(os.path.join(out, 'sim.json'))); assert dw['smoke'] == sm['smoke'] == bool(c['smoke'])
    assert not sm['stop_any'], f'information-zero stop condition met: {sm["stop"]}'
    pred, lstar, mixw, prov, Tn, emb_t, ZB = setup(c, tc, meta, dw); assert sm['lowlevel']['pred_sha256'] == prov['pred_sha256'] and sm['Lstar'] == lstar
    H = dw['hypotheses']; assert sm['w'] == {h: H[h]['w'] for h in H}; live = [h for h in H if not H[h]['closed']]; assert live, 'every hypothesis closed at the development stage'
    grid = np.array(c['w_grid'], np.float64); vo = P(vc['out_dir']); v1 = json.load(open(os.path.join(vo, 'v1_manifest.json'))); d2 = P(c['test']['v2_dir'])
    v2 = json.load(open(os.path.join(d2, 'v2_manifest.json'))); assert v2['inputs'] == v1['inputs'], 'V2 inputs differ from V1'
    for k, hh in v2['files'].items(): assert fsha(os.path.join(d2, k)) == hh, f'{k}: embedding sha256 differs from V2'
    true = np.arange(200); nseed = len(vc['final']['seeds']); subs = vc['subjects']; cond = c['test']['cond']; conds = [cond] + list(c['test']['report_conds']); A = {}
    for s in subs:
        Z = np.load(os.path.join(d2, f'z_emb_t{s:02d}.npz')); sc = Z['single_code']
        for cd in conds:
            SE = V.seed_mean([Z[f'seed{i}|{cd}|avg20'] for i in range(nseed)]) @ Tn.T
            if cd == cond:
                SS = V.unit(np.asarray(Z[f'shuffle|{cd}|avg20'], np.float64)) @ Tn.T; A.setdefault('z_alone', []).append(V.cells(SE, true))
            for h in live:
                b, w = ZB[H[h]['base']], H[h]['w']; A.setdefault((h, 'D', cd), []).append(dcells(b, SE, w, true)); A.setdefault((h, 'curve', cd), []).append([V.cells(combined(b, zrow(SE), x), true) for x in grid])
                for k in all_cands(c): A.setdefault((h, 'L', cd, k), []).append(dcells(ZB[k], SE, w, true))
                if cd == cond:
                    for nm, X, kind in (('Dsh', SS, 'mrp'), ('Dc', const(SE), 'mrp'), ('Dcen', SE - SE.mean(0, keepdims=True), 'mrp'), ('top1', SE, 'top1'), ('mrr', SE, 'mrr')):
                        A.setdefault((h, nm, cd), []).append(dcells(b, X, w, true, kind))
        for cd in vc['single_conditions']:
            SE = V.seed_mean([Z[f'seed{i}|{cd}|single'] for i in range(nseed)]) @ Tn.T
            for h in live:
                d = dcells(ZB[H[h]['base']][sc - 1], SE, H[h]['w'], sc - 1); A.setdefault((h, 'single', cd), []).append(np.array([d[sc == k].mean() for k in range(1, 201)]))
    A = {k: np.array(v) for k, v in A.items()}; R = resample(len(subs), 200, c['n_boot'], np.random.default_rng(c['seeds']['boot']))
    J = judge({h: (dict(D=A[(h, 'D', cond)], Dsh=A[(h, 'Dsh', cond)], Dc=A[(h, 'Dc', cond)]) if h in live else None) for h in H}, R, c)
    hyp = {}
    for h in H:
        if h not in live: hyp[h] = dict(J[h], base=H[h]['base']); continue
        b = ZB[H[h]['base']]; cl = V.cells(b, true); L2, wLL = H[h]['second_predictor']['L2'], H[h]['second_predictor']['w_LL']; dLL = V.cells(combined(b, ZB[L2], wLL), true) - cl
        bt = lambda X: dict(point=float(X.mean()), p5=float(np.percentile(bmeans(X, R), 5)), p95=float(np.percentile(bmeans(X, R), 95)))
        hyp[h] = dict(J[h], base=H[h]['base'], w=H[h]['w'], mrp_base=float(cl.mean()), mrp_combined=float((A[(h, 'D', cond)] + cl).mean()),
                      report=dict(delta_top1=bt(A[(h, 'top1', cond)]), delta_MRR=bt(A[(h, 'mrr', cond)]), delta_centred=float(A[(h, 'Dcen', cond)].mean()),
                                  k0={cd: float(A[(h, 'D', cd)].mean()) for cd in c['test']['report_conds']}, single={cd: float(A[(h, 'single', cd)].mean()) for cd in vc['single_conditions']},
                                  w_curve={cd: dict(mrp=A[(h, 'curve', cd)].mean((0, 2)).tolist(), delta=(A[(h, 'curve', cd)].mean((0, 2)) - cl.mean()).tolist()) for cd in conds},
                                  per_L={k: dict(mrp_L=float(V.cells(ZB[k], true).mean()), delta={cd: float(A[(h, 'L', cd, k)].mean()) for cd in conds}) for k in all_cands(c)},
                                  per_subject={str(s): float(A[(h, 'D', cond)][i].mean()) for i, s in enumerate(subs)},
                                  delta_LL=dict(L2=L2, w_LL=wLL, delta=float(dLL.mean()), note='EEG-free' + ('; with base L_mix a small delta_LL is not evidence that the other-function path is small (amendment 2)' if H[h]['base'] == 'L_mix' else ''))))
    res = dict(stage='test', smoke=bool(c['smoke']), lowlevel=prov, Lstar=lstar, subjects=subs, cond=cond, v2_dir=c['test']['v2_dir'], n_boot=c['n_boot'], boot_seed=c['seeds']['boot'],
               holm_alphas=c['holm_alphas'], dev_w_sha256=fsha(os.path.join(out, 'dev_w.json')), sim_sha256=fsha(os.path.join(out, 'sim.json')),
               z_alone_mrp=float(A['z_alone'].mean()), hypotheses=hyp, adopted={h: bool(hyp[h].get('adopted', False)) for h in H})
    json.dump(res, open(fn, 'w'), indent=1)
    print(json.dumps(dict(smoke=res['smoke'], **{h: {k: hyp[h].get(k) for k in ('point', 'p', 'holm_significant', 'adopted')} for h in H})), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', choices=['dev', 'sim', 'test'], required=True); ap.add_argument('--config-override'); a = ap.parse_args()
    c = load_cfg(a.config, a.config_override); vc = load_cfg(c['verdict_config'], None); tc = yaml.safe_load(open(P(c['train_config']))); meta = Meta(P(tc['meta']))
    {'dev': stage_dev, 'sim': stage_sim, 'test': stage_test}[a.stage](c, vc, tc, meta, a)


if __name__ == '__main__':
    main()
