"""F-V design check of gate D (computed after the gate D verdict 1e948dd; not a change of the verdict): information-zero pass rates.

Same design as scripts/fv_dev_gate.py: development targets (dev.dev_targets), the development concepts each target is
evaluated on (set A, the true columns stored by the chosen candidate's development runs: concept membership only, no score),
the 200 development concept names of names_text (P1) as candidates, cell = rank percentile, statistic = mean over all cells,
pigeonhole = targets with replacement, then concepts with replacement inside each drawn target (fv_dev_gate.pigeonhole),
n_boot draws, 5th / 95th percentiles. Predicted embeddings are drawn as isotropic 1024-d Gaussians, independent per target and
concept (information zero). Per simulation: point estimate, 90% interval, (iii) pass = upper < noise_upper, (ii) pass = the
interval contains 0.5. n_sim simulations with default_rng(seed); each simulation's bootstrap uses a child seed.

Verdict design (same file, --verdict-seed): 10 subjects x the 200 test concepts, the 200 test names of names_text (P1; EEG-free,
as the MDE stage), crossed subject x concept pigeonhole (ese.things.verdict.pigeonhole, n_boot), information-zero predictions
(isotropic Gaussians per subject and concept). Pass rates of premise 2 (cells minus the permutation-null mean, 95th percentile
< phase_excess_max; the null mean of a bijective concept permutation is exactly 0.5 without ties, used here instead of the
1,000-permutation estimate of V3) and of premises 3 and 4 (cells' 95th percentile < meter_upper_max; one independent draw each).

Rule N (--rule-n, D-112 amendment 5 draft; computed after the gate D verdict; EEG-free): control excess E_c = MRP_c minus the
concept-permutation null mean; pass = (a) the pigeonhole 5th percentile of E_c <= 0 and (b) the point E_c <= ratio x E_dec (the
decoder excess on the same set). The permutation null mean is the exact expectation over all bijections (mean of the cells of
every evaluated row at every evaluated true column, per target / subject) instead of the n_perm estimate. Predictions with a
given true excess: x = beta * name_text(true) + g / sqrt(d), g ~ N(0, I_d), beta calibrated by bisection on n_cal common draws
(beta = 0 is information zero). Per simulation: controls at E_c in {0} + ctrl_excess, decoder point estimates at dec_excess, both
designs (gate D: targets -> concepts pigeonhole; verdict: crossed pigeonhole); in the verdict design three independent
information-zero controls (premises 2-4) give the joint pass rate. Writes results/fv/dev_gate_rule_n_sim.json.

usage: python scripts/fv_dev_gate_null_sim.py configs/fv_train.yaml [--n-sim 1000] [--seed 20261871] [--verdict-seed 20261872]
       python scripts/fv_dev_gate_null_sim.py configs/fv_train.yaml --rule-n [--rule-n-seed 20261873] [--ratio 0.1]
           [--dec-excess 0.10 0.20 0.31] [--ctrl-excess 0.01 0.03] [--n-cal 200]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.things.data import Meta
from fv_train import name_rows
from fv_dev_gate import runs, pigeonhole, unit, P


def all_cells(S):
    """S [n, K] (no ties) -> rank-percentile cell of every column [n, K]"""
    n, K = S.shape; pos = np.empty((n, K)); pos[np.arange(n)[:, None], np.argsort(-S, axis=1)] = np.arange(K); return 1 - pos / (K - 1)


def excess_cells(X, cand, true):
    """predictions X [n, d], candidates [K, d], true [n] -> cells minus the exact permutation-null mean of this unit [n]"""
    C = all_cells(unit(X) @ cand.T); return C[np.arange(len(true)), true] - C[:, true].mean()


def draw(rng, cand, true, beta, G=None):
    G = rng.standard_normal((len(true), cand.shape[1])) if G is None else G; return beta * cand[true] + G / np.sqrt(cand.shape[1])


def calibrate(cand, units, target, n_cal, rng):
    """beta with mean pooled excess = target over n_cal common draws of the whole design (units: list of true arrays)"""
    Gs = [[rng.standard_normal((len(t), cand.shape[1])).astype(np.float32) for t in units] for _ in range(n_cal)]
    f = lambda b: float(np.mean([np.concatenate([excess_cells(draw(None, cand, t, b, g), cand, t) for t, g in zip(units, G)]).mean() for G in Gs]))
    lo, hi = 0.0, 1.0
    while f(hi) < target: hi *= 2
    for _ in range(30):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


def rule_n(a, c, meta, cand, trues, Tn, ns, n_boot_d, n_boot_v):
    from ese.things import verdict as V
    t0 = time.time(); rng = np.random.default_rng(a.rule_n_seed); levels = sorted(set(a.ctrl_excess) | set(a.dec_excess)); out = dict(
        note='design check of rule N computed after the gate D verdict (1e948dd); EEG-free, test not opened; not a change of the verdict',
        seed=a.rule_n_seed, n_sim=a.n_sim, ratio=a.ratio, lower_pct=5, dec_excess=a.dec_excess, ctrl_excess=a.ctrl_excess, n_cal=a.n_cal,
        null_mean='exact expectation over all concept bijections (per target / subject)',
        model='x = beta * name_text(true) + g / sqrt(d), g ~ N(0, I); beta = 0 for information zero')
    designs = dict(gate_D=dict(units=trues, cand=cand, n_boot=n_boot_d, boot=lambda v, r: pigeonhole(v, n_boot_d, r)),
                   verdict=dict(units=[np.arange(200)] * ns, cand=Tn, n_boot=n_boot_v, boot=lambda v, r: V.pigeonhole(np.stack(v), n_boot_v, r)))
    for name, D in designs.items():
        cd, units = D['cand'], D['units']; beta = {0.0: 0.0}
        for e in levels: beta[e] = calibrate(cd, units, e, a.n_cal, np.random.default_rng(rng.integers(1 << 62)))
        sim = lambda e: [excess_cells(draw(rng, cd, t, beta[e]), cd, t) for t in units]
        ctrl_lv = [0.0] + list(a.ctrl_excess); n_extra = 2 if name == 'verdict' else 0
        pt = {e: np.empty(a.n_sim) for e in ctrl_lv}; lo5 = {e: np.empty(a.n_sim) for e in ctrl_lv}; dec = {e: np.empty(a.n_sim) for e in a.dec_excess}
        extra_pt, extra_lo = np.empty((n_extra, a.n_sim)), np.empty((n_extra, a.n_sim))
        for i in range(a.n_sim):
            for e in ctrl_lv:
                v = sim(e); pt[e][i] = np.concatenate(v).mean(); lo5[e][i] = np.percentile(D['boot'](v, np.random.default_rng(rng.integers(1 << 62))), 5)
            for k in range(n_extra):
                v = sim(0.0); extra_pt[k, i] = np.concatenate(v).mean(); extra_lo[k, i] = np.percentile(D['boot'](v, np.random.default_rng(rng.integers(1 << 62))), 5)
            for e in a.dec_excess: dec[e][i] = np.concatenate(sim(e)).mean()
        res = dict(n_units=len(units), n_concepts=[int(len(t)) for t in units], n_candidates=int(cd.shape[0]), n_boot=D['n_boot'],
                   beta={f'{e:.2f}': float(beta[e]) for e in levels}, decoder={f'{e:.2f}': dict(point_mean=float(dec[e].mean()), point_sd=float(dec[e].std(ddof=1))) for e in a.dec_excess},
                   control={})
        for e in ctrl_lv:
            pa = lo5[e] <= 0; r = dict(point_mean=float(pt[e].mean()), point_sd=float(pt[e].std(ddof=1)), lower5_mean=float(lo5[e].mean()), pass_a=float(pa.mean()), by_dec_excess={})
            for d in a.dec_excess:
                pb = pt[e] <= a.ratio * dec[d]; pbf = pt[e] <= a.ratio * d
                r['by_dec_excess'][f'{d:.2f}'] = dict(pass_b=float(pb.mean()), pass_ab=float((pa & pb).mean()), pass_b_fixed=float(pbf.mean()), pass_ab_fixed=float((pa & pbf).mean()))
            res['control'][f'{e:.2f}'] = r
        if n_extra:
            P0 = np.vstack([pt[0.0][None], extra_pt]); L0 = np.vstack([lo5[0.0][None], extra_lo]); res['premises_2_4_joint_info_zero'] = {
                f'{d:.2f}': float(((L0 <= 0) & (P0 <= a.ratio * dec[d][None])).all(0).mean()) for d in a.dec_excess}
        out[name] = res; print(name, json.dumps(res['control']), round(time.time() - t0), 's', flush=True)
    out['seconds'] = time.time() - t0; o = P(os.path.join(c['gate_d']['out_dir'], 'dev_gate_rule_n_sim.json')); json.dump(out, open(o, 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--n-sim', type=int, default=1000); ap.add_argument('--seed', type=int, default=20261871); ap.add_argument('--verdict-seed', type=int, default=20261872)
    ap.add_argument('--rule-n', action='store_true'); ap.add_argument('--rule-n-seed', type=int, default=20261873); ap.add_argument('--ratio', type=float, default=0.1)
    ap.add_argument('--dec-excess', type=float, nargs='+', default=[0.10, 0.20, 0.31]); ap.add_argument('--ctrl-excess', type=float, nargs='+', default=[0.01, 0.03]); ap.add_argument('--n-cal', type=int, default=200); a = ap.parse_args()
    c = yaml.safe_load(open(P(a.config))); g = c['gate_d']; targets = c['dev']['dev_targets']; t0 = time.time()
    meta = Meta(P(c['meta'])); dev_c = meta.dev_concepts(c['dev']['salt'], c['dev']['n'])
    cand = unit(np.load(os.path.join(P(c['targets']['dir']), c['targets']['names_text']))[name_rows(c, meta, dev_c)])
    chosen = json.load(open(P(os.path.join(g['out_dir'], 'dev_choice_A_ses4.json'))))['chosen']
    trues = [np.load(os.path.join(runs(c, chosen, t, False)[0], 'dev_scores.npz'))['A_ses4_true'] for t in targets]
    if a.rule_n:
        vc = yaml.safe_load(open(P('configs/fv_verdict.yaml'))); import csv
        rows = list(csv.DictReader(open(P(c['targets']['names_table'])), delimiter='\t')); tidx = {r['folder']: i for i, r in enumerate(rows) if r['split'] == 'test'}
        Tn = unit(np.load(os.path.join(P(c['targets']['dir']), c['targets']['names_text']))[[tidx[f] for f in meta.test_img_concept]])
        return rule_n(a, c, meta, cand, trues, Tn, len(vc['subjects']), g['n_boot'], vc['n_boot'])
    rng = np.random.default_rng(a.seed); pt, lo, hi = np.empty(a.n_sim), np.empty(a.n_sim), np.empty(a.n_sim)
    for i in range(a.n_sim):
        vals = []
        for tr in trues:
            S = unit(rng.standard_normal((len(tr), cand.shape[1]))) @ cand.T; t = S[np.arange(len(tr)), tr]; vals.append(1 - (S > t[:, None]).sum(1) / (S.shape[1] - 1))
        b = pigeonhole(vals, g['n_boot'], np.random.default_rng(rng.integers(1 << 62))); pt[i] = np.concatenate(vals).mean(); lo[i], hi[i] = np.percentile(b, [5, 95])
    res = dict(note='design check computed after the gate D verdict (1e948dd); not a change of the verdict', chosen=chosen, targets=targets,
               n_concepts=[int(len(x)) for x in trues], n_candidates=int(cand.shape[0]), n_sim=a.n_sim, seed=a.seed, n_boot=g['n_boot'],
               point=dict(mean=float(pt.mean()), sd=float(pt.std(ddof=1)), p5=float(np.percentile(pt, 5)), p95=float(np.percentile(pt, 95))),
               upper90=dict(mean=float(hi.mean()), p5=float(np.percentile(hi, 5)), p50=float(np.median(hi))), half_width_mean=float(((hi - lo) / 2).mean()),
               pass_rate_iii=float((hi < g['noise_upper']).mean()), noise_upper=g['noise_upper'], pass_rate_ii=float(((lo <= 0.5) & (0.5 <= hi)).mean()),
               seconds=time.time() - t0)
    # verdict premises 2-4 under information zero
    from ese.things import verdict as V
    vc = yaml.safe_load(open(P('configs/fv_verdict.yaml'))); rl = vc['rules']
    import csv; rows = list(csv.DictReader(open(P(c['targets']['names_table'])), delimiter='\t')); tidx = {r['folder']: i for i, r in enumerate(rows) if r['split'] == 'test'}
    Tn = unit(np.load(os.path.join(P(c['targets']['dir']), c['targets']['names_text']))[[tidx[f] for f in meta.test_img_concept]]); ns = len(vc['subjects']); true = np.arange(200)
    rv = np.random.default_rng(a.verdict_seed); u2, u3, u4 = np.empty(a.n_sim), np.empty(a.n_sim), np.empty(a.n_sim); p3 = np.empty(a.n_sim)
    for i in range(a.n_sim):
        draw = lambda: np.stack([V.cells(unit(rv.standard_normal((200, Tn.shape[1]))) @ Tn.T, true) for _ in range(ns)])
        M2, M3, M4 = draw() - 0.5, draw(), draw(); br = np.random.default_rng(rv.integers(1 << 62))
        u2[i] = np.percentile(V.pigeonhole(M2, vc['n_boot'], br), rl['premise_upper_pct']); u3[i] = np.percentile(V.pigeonhole(M3, vc['n_boot'], br), rl['premise_upper_pct'])
        u4[i] = np.percentile(V.pigeonhole(M4, vc['n_boot'], br), rl['premise_upper_pct']); p3[i] = M3.mean()
    res['verdict_design'] = dict(seed=a.verdict_seed, n_sim=a.n_sim, n_subjects=ns, n_concepts=200, n_boot=vc['n_boot'], null_mean_used=0.5,
                                 point_mean=float(p3.mean()), point_sd=float(p3.std(ddof=1)),
                                 premise2=dict(upper95_mean=float(u2.mean()), threshold=rl['phase_excess_max'], pass_rate=float((u2 < rl['phase_excess_max']).mean())),
                                 premise3=dict(upper95_mean=float(u3.mean()), threshold=rl['meter_upper_max'], pass_rate=float((u3 < rl['meter_upper_max']).mean())),
                                 premise4=dict(upper95_mean=float(u4.mean()), threshold=rl['meter_upper_max'], pass_rate=float((u4 < rl['meter_upper_max']).mean())))
    res['seconds'] = time.time() - t0
    o = P(os.path.join(g['out_dir'], 'dev_gate_null_sim.json')); json.dump(res, open(o, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('point', 'upper90', 'half_width_mean', 'pass_rate_iii', 'pass_rate_ii', 'verdict_design')}), flush=True)


if __name__ == '__main__':
    main()
