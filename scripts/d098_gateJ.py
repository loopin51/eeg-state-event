"""D-098 gate J: forward acoustic-pair ceiling test on the 26 development cells already opened by D-094 (no other EEG is scored).

Pooled forward TRF (20 Hz, lags -100..700 ms, 128 channels) fitted to the mean EEG of the 13 training subjects of subject fold 0
on the 14 training runs of B1 run fold 0; evaluated per subject on the validation runs 5 and 13 (D-069 1-8 Hz cache).
  A  = env + 8 log bands + word onset (D-069 A)          Aplus = A + env edge + 8 band edges + phone onset
  L1 = Aplus + Zipf, GPT-2 surprisal, entropy at word onsets
  L2 = L1 + phoneme surprisal, cohort entropy at phone onsets (values standardised with training-run words / phones)
Identification: queries = word-onset windows [t - 0.5, t + 2.5) of the B2 validation pool (pool_valid_f0), candidates = the
query + its 19 B2 primary within-run acoustic pairs (pairs_within_valid_f0); score = correlation of the observed (128 x 60)
and predicted EEG windows; MRP = (below + 0.5 ties) / 19. Delta = MRP(L2) - MRP(Aplus).
Interval: 1,000 draws of subject refit (multinomial weights over the 13 subjects, penalties fixed) x pigeonhole over 30-s query
blocks of the two runs. The refit is computed exactly from per-subject fits (ridge is linear in the subject weights; draw 0 is
also refitted directly as a check). Null: 100 circular shifts (>= 10 s, every run) of the 5 added regressors, refit. Value-shuffle control:
word values permuted among words, phone values among phones, within run (one permutation).
Premise: Aplus within-run identification over all validation-pool queries of the run (B1 type) > its segment-order shift null
p95 (1,000) and bootstrap 5th percentile > 0.5. Verdict rules: docs/DECISIONS.md D-098.

usage: python scripts/d098_gateJ.py configs/d098_gates.yaml [--smoke]
       --smoke: every subject's EEG is circularly shifted by half of each run (stimulus alignment broken, no information)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ds004408_p0 import lagged, ridge, corr, z
from d098_gate0 import shift_maps


def chunks(M, st, L):
    X = np.stack([M[s:s + L].ravel() for s in st]).astype(np.float32); X -= X.mean(1, keepdims=True)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args(); t0 = time.time()
    c = yaml.safe_load(open(a.config)); g = c['gate_J']; fs = c['fs']; b1 = yaml.safe_load(open(os.path.join(ROOT, c['b1_config']))); sp = b1['split']
    f0 = g['train_runs_from_b1_fold']; test = [r - 1 for r in sp['run_folds'][f0]]; valid = [r - 1 for r in sp['valid_runs'][f0]]
    train = [r for r in range(20) if r not in test + valid]; ev = [r - 1 for r in g['eval_runs']]; assert sorted(ev) == sorted(valid)
    subs = g['subjects']; ns = len(subs); assert sorted(subs) == sorted(s for j, fs_ in enumerate(sp['subject_folds']) if j != 0 for s in fs_)
    Fz = np.load(os.path.join(ROOT, c['features_cache'], 'features.npz'), allow_pickle=True); reg = list(Fz['reg']); names = list(Fz['reg_names'])
    W = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True)
    I = np.load(os.path.join(ROOT, 'data/cache/stageB_b2/items.npz'), allow_pickle=True); Pz = np.load(os.path.join(ROOT, 'data/cache/stageB_b2/pairs.npz'))
    runs = sorted(set(train + ev))
    E = {s: np.load(os.path.join(ROOT, g['eeg'], f'sub-{s:03d}.npz'), allow_pickle=True)['eeg20'] for s in subs}
    T = {r: min(len(reg[r]), *(E[s][r].shape[1] for s in subs)) for r in runs}
    Y = {s: {r: z(np.asarray(E[s][r][:, :T[r]], np.float64)).T.astype(np.float32) for r in runs} for s in subs}; del E
    if a.smoke: Y = {s: {r: np.roll(Y[s][r], T[r] // 2, 0) for r in runs} for s in subs}
    Ybar = {r: np.mean([Y[s][r] for s in subs], 0) for r in runs}
    # regressors
    base_A = [names.index(x) for x in ['env'] + [f'band{b}' for b in range(8)] + ['word_onset']]
    plus = [names.index(x) for x in ['env_edge'] + [f'band_edge{b}' for b in range(8)] + ['phone_onset']]
    wtr = np.isin(W['run'], train); ph_run, ph_on = Fz['ph_run'], Fz['ph_onset']; ptr = np.isin(ph_run, train)
    def stdz(v, m): v = np.asarray(v, float); return (v - v[m].mean()) / v[m].std()
    wv = np.stack([stdz(W[k], wtr) for k in ('zipf', 'surprisal', 'entropy')], 1); pv = np.stack([stdz(Fz[k], ptr) for k in ('ph_surprisal', 'cohort_entropy')], 1)
    rng_vs = np.random.default_rng(g['value_shuffle_seed']); wv_s, pv_s = wv.copy(), pv.copy()
    for r in range(20):
        m = np.flatnonzero(W['run'] == r); wv_s[m] = wv[rng_vs.permutation(m)]; m = np.flatnonzero(ph_run == r); pv_s[m] = pv[rng_vs.permutation(m)]
    def imp(r, onsets, vals):
        M = np.zeros((T[r], vals.shape[1])); p = np.round(onsets * fs).astype(int); k = p < T[r]; np.add.at(M, p[k], vals[k]); return M
    def lex(r, wvals, pvals): mw = W['run'] == r; mp = ph_run == r; return imp(r, W['onset'][mw], wvals[mw]), imp(r, ph_on[mp], pvals[mp])
    LEX = {r: lex(r, wv, pv) for r in runs}; LEXS = {r: lex(r, wv_s, pv_s) for r in runs}
    lags = list(range(int(round(g['lags_ms'][0] / 1000 * fs)), int(round(g['lags_ms'][1] / 1000 * fs)) + 1))
    def design(m, r, shift=0, shuffled=False):
        R = reg[r][:T[r]].astype(np.float64); cols = [R[:, base_A]]
        if m != 'A': cols.append(R[:, plus])
        lw, lp = (LEXS if shuffled else LEX)[r]
        if m in ('L1', 'L2'): cols.append(np.roll(lw, shift, 0) if m == 'L2' else lw)
        if m == 'L2': cols.append(np.roll(lp, shift, 0))
        return lagged(np.hstack(cols), lags)
    # queries and pairs
    pool = Pz[g['pool_key']]; P = Pz[g['pairs_key']]; loc = {int(x): k for k, x in enumerate(pool)}; PL = np.vectorize(loc.get)(P); assert (PL >= 0).all()
    qrun, qt = I['run'][pool], I['t'][pool]; assert set(qrun.tolist()) == set(ev) and (I['run'][P] == qrun[:, None]).all()
    L = int(round(3.0 * fs)); st = np.round((qt - 0.5) * fs).astype(int); assert all(st[qrun == r].max() + L <= T[r] for r in ev)
    O = {s: np.zeros((len(pool), Y[s][ev[0]].shape[1] * L), np.float32) for s in subs}
    for r in ev:
        ir = np.flatnonzero(qrun == r)
        for s in subs: O[s][ir] = chunks(Y[s][r], st[ir], L)
    blocks = np.array([int(r) * 1000 + int(t // g['block_s']) for r, t in zip(qrun, qt)]); ub = np.unique(blocks); bidx = np.searchsorted(ub, blocks)
    # fits
    lams = 10.0 ** np.arange(g['lambdas_log10'][0], g['lambdas_log10'][1] + 1); groups = [train[i::4] for i in range(4)]
    def stats(m, shift_r=None, shuffled=False):
        Xr = {r: design(m, r, 0 if shift_r is None else shift_r[r], shuffled) for r in runs}; Cr = {r: Xr[r].T @ Xr[r] for r in train}
        G = np.stack([sum(Xr[r].T @ Y[s][r] for r in train) for s in subs]).astype(np.float64)        # [S, p, 128]
        return Xr, Cr, G
    def pick_lambda(Xr, Cr, G):
        sc = np.zeros(len(lams)); Gr = {r: np.stack([Xr[r].T @ Y[s][r] for s in subs]).mean(0) for r in train}
        for gr in groups:
            tr = [r for r in train if r not in gr]; Ct = sum(Cr[r] for r in tr); Bt = sum(Gr[r] for r in tr)
            for j, lam in enumerate(lams):
                Wm = ridge(Ct, Bt, lam); sc[j] += np.mean([np.mean([corr(Xr[r] @ Wm, Y[s][r]).mean() for s in subs]) for r in gr])
        return float(lams[int(np.argmax(sc))])
    def predict_chunks(Xr, Wm):
        Pc = np.zeros((len(pool), O[subs[0]].shape[1]), np.float32)
        for r in ev:
            ir = np.flatnonzero(qrun == r); Pc[ir] = chunks((Xr[r] @ Wm).astype(np.float32), st[ir], L)
        return Pc
    def pair_rp(Pc, full=True):
        """[S, Q] pair MRP and (full) [S, Q] B1-type within-run MRP over all pool queries of the run"""
        rp = np.zeros((ns, len(pool))); rw = np.zeros((ns, len(pool))) if full else None
        for si, s in enumerate(subs):
            tru = np.einsum('qd,qd->q', O[s], Pc); cand = np.einsum('qd,qkd->qk', O[s], Pc[PL])
            rp[si] = ((cand < tru[:, None]).sum(1) + 0.5 * (cand == tru[:, None]).sum(1)) / PL.shape[1]
            if full:
                for r in ev:
                    ir = np.flatnonzero(qrun == r); Sm = O[s][ir] @ Pc[ir].T; rw[si, ir] = 1 - (Sm > np.diag(Sm)[:, None]).sum(1) / (len(ir) - 1)
        return rp, rw
    models = list(g['models']) + ['L2_vs']; fit, rp, rw, rpred = {}, {}, {}, {}
    for m in models:
        Xr, Cr, G = stats('L2' if m == 'L2_vs' else m, shuffled=(m == 'L2_vs'))
        lam = fit['L2']['lam'] if m == 'L2_vs' else pick_lambda(Xr, Cr, G); Ct = sum(Cr.values()); Wm = ridge(Ct, G.mean(0), lam)
        fit[m] = dict(lam=lam, Ct=Ct, G=G, Xev={r: Xr[r] for r in ev}); rp[m], rw[m] = pair_rp(predict_chunks(Xr, Wm))
        rpred[m] = float(np.mean([corr(Xr[r] @ Wm, Y[s][r]).mean() for s in subs for r in ev]))
        print(f'{m}: lambda {lam:g}, p {Xr[ev[0]].shape[1]}, pair MRP {rp[m].mean():.4f}, within-run MRP {rw[m].mean():.4f}, r {rpred[m]:.4f} ({time.time() - t0:.0f}s)', flush=True)
        del Xr, Cr
    pairs_of = dict(L2_Aplus=('L2', 'Aplus'), L1_Aplus=('L1', 'Aplus'), L2_L1=('L2', 'L1'), Aplus_A=('Aplus', 'A'), vs_Aplus=('L2_vs', 'Aplus'))
    obs = {k: float((rp[x] - rp[y]).mean()) for k, (x, y) in pairs_of.items()}
    # null: circular shift (>= 10 s, every run) of the 5 added L2 - Aplus regressors, refit with the fixed L2 penalty
    rng = np.random.default_rng(g['shift_seed']); null = []
    for b in range(g['n_shift']):
        sh = {r: int(rng.integers(int(g['min_shift_s'] * fs), T[r] - int(g['min_shift_s'] * fs))) for r in runs}
        Xr, Cr, G = stats('L2', shift_r=sh); Wm = ridge(sum(Cr.values()), G.mean(0), fit['L2']['lam'])
        null.append(float((pair_rp(predict_chunks(Xr, Wm), full=False)[0] - rp['Aplus']).mean()))
        if b % 20 == 0: print(f'shift {b}: {null[-1]:+.4f} ({time.time() - t0:.0f}s)', flush=True)
    # premise: Aplus B1-type within-run identification vs the segment-order shift null of the truth
    maps = shift_maps({r: qt[qrun == r] for r in ev}, g['premise_n_shift'], 30.0, np.random.default_rng(g['premise_seed']))
    Pc_A = predict_chunks(fit['Aplus']['Xev'], ridge(fit['Aplus']['Ct'], fit['Aplus']['G'].mean(0), fit['Aplus']['lam']))
    prem_null = np.zeros(g['premise_n_shift'])
    for s in subs:
        for r in ev:
            ir = np.flatnonzero(qrun == r); n = len(ir); Sm = O[s][ir] @ Pc_A[ir].T; srt = np.sort(Sm, 1); V = Sm[np.arange(n)[None, :], maps[r]]   # [B, n]
            for i in range(n): prem_null += 1 - (n - np.searchsorted(srt[i], V[:, i], side='right')) / (n - 1)
    prem_null /= ns * len(pool)
    # bootstrap: subject refit x query-block pigeonhole. With the design and the penalty fixed, the refitted weights are linear in the
    # subject weights (W_b = sum_s w_s W_s / ns), so the centred predicted window of draw b is sum_s w_s v_s / ns. Scores (observed
    # window . normalised predicted window) are therefore computed exactly from the numerators O_t . v_s and the Gram of the v_s per
    # window; draw 0 is also refitted directly and the two must agree.
    pos = np.zeros(len(pool), int)
    for r in ev: ir = np.flatnonzero(qrun == r); pos[ir] = np.arange(len(ir))
    def raw_chunks(Xr, Wm):
        Pc = np.zeros((len(pool), O[subs[0]].shape[1]), np.float32)
        for r in ev:
            ir = np.flatnonzero(qrun == r); M = (Xr[r] @ Wm).astype(np.float32); X = np.stack([M[k:k + L].ravel() for k in st[ir]]); Pc[ir] = X - X.mean(1, keepdims=True)
        return Pc
    terms = {}
    for m in models:
        V = np.stack([raw_chunks(fit[m]['Xev'], ridge(fit[m]['Ct'], fit[m]['G'][k], fit[m]['lam'])) for k in range(ns)])   # [S, Q, D]
        Gq = np.einsum('sqd,uqd->qsu', V, V, dtype=np.float64, optimize=True)                                                 # [Q, S, S]
        Np = np.zeros((ns, len(pool), PL.shape[1] + 1, ns)); Nf = {} if m == 'Aplus' else None
        for ti, s in enumerate(subs):
            for r in ev:
                ir = np.flatnonzero(qrun == r); n = len(ir)
                Sm = (O[s][ir] @ V[:, ir].reshape(ns * n, -1).T).reshape(n, ns, n).transpose(0, 2, 1).astype(np.float64)       # [n, n, S]
                Np[ti, ir, 0] = Sm[np.arange(n), np.arange(n)]; Np[ti, ir, 1:] = Sm[np.arange(n)[:, None], pos[PL[ir]]]
                if Nf is not None: Nf[(ti, r)] = Sm.astype(np.float32)
        terms[m] = (Gq, Np, Nf); del V
        print(f'boot terms {m} ({time.time() - t0:.0f}s)', flush=True)
    def draw_scores(m, wsub):
        Gq, Np, Nf = terms[m]; den = np.sqrt(np.einsum('s,qsu,u->q', wsub, Gq, wsub)) + 1e-30
        num = Np @ wsub; sc = num / np.concatenate([den[:, None], den[PL]], 1)[None]                                         # [S, Q, 1+K]
        tru, cand = sc[..., :1], sc[..., 1:]; rpm = ((cand < tru).sum(-1) + 0.5 * (cand == tru).sum(-1)) / PL.shape[1]
        rwm = None
        if Nf is not None:
            rwm = np.zeros((ns, len(pool)))
            for (ti, r), Sm in Nf.items():
                ir = np.flatnonzero(qrun == r); S2 = (Sm @ wsub.astype(np.float32)) / den[ir][None].astype(np.float32); rwm[ti, ir] = 1 - (S2 > np.diag(S2)[:, None]).sum(1) / (len(ir) - 1)
        return rpm, rwm
    rng = np.random.default_rng(g['boot_seed']); boot = {k: [] for k in pairs_of}; boot_prem = []
    for b in range(g['n_boot']):
        wsub = np.bincount(rng.integers(0, ns, ns), minlength=ns).astype(float); cq = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)).astype(float)[bidx]
        rpb = {}
        for m in models:
            rpb[m], rwb = draw_scores(m, wsub)
            if m == 'Aplus': rw_A = rwb
        if b == 0:   # direct refit of draw 0 (the original definition) must give the same numbers
            for m in models:
                Wm = ridge(fit[m]['Ct'], np.tensordot(wsub, fit[m]['G'], 1) / ns, fit[m]['lam']); rp0, rw0 = pair_rp(predict_chunks(fit[m]['Xev'], Wm), full=(m == 'Aplus'))
                d1 = float(np.abs(rp0 - rpb[m]).mean()); n1 = int((rp0 != rpb[m]).sum()); d2 = float(np.abs(rw0 - rw_A).mean()) if m == 'Aplus' else 0.0
                print(f'draw 0 check {m}: mean |pair MRP diff| {d1:.1e} ({n1} of {rp0.size} subject-queries differ), within-run {d2:.1e}, '
                      f'weighted pair MRP {float(wsub @ rp0 @ cq) / (wsub.sum() * cq.sum()):.6f} vs {float(wsub @ rpb[m] @ cq) / (wsub.sum() * cq.sum()):.6f}', flush=True)
                assert d1 < 1e-4 and d2 < 1e-4, m
        tot = wsub.sum() * cq.sum()
        for k, (x, y) in pairs_of.items(): boot[k].append(float(wsub @ (rpb[x] - rpb[y]) @ cq / tot))
        boot_prem.append(float(wsub @ rw_A @ cq / tot))
        if b % 100 == 0: print(f'boot {b}: L2-Aplus {boot["L2_Aplus"][-1]:+.4f} ({time.time() - t0:.0f}s)', flush=True)
    # verdict (D-098, in order)
    q = lambda x, p: float(np.percentile(x, p)); bt = np.array(boot['L2_Aplus']); bv = np.array(boot['vs_Aplus']); se = float(bt.std(ddof=1))
    a_pair = g['a_pair_factor'] * se; thr = max(a_pair, g['floor']); null_p95 = q(null, 95)
    prem = dict(mrp=float(rw['Aplus'].mean()), null_p95=q(prem_null, 95), null_mean=float(prem_null.mean()), boot_p5=q(boot_prem, 5))
    prem['passed'] = bool(prem['mrp'] > prem['null_p95'] and prem['boot_p5'] > 0.5)
    ci = [q(bt, 5), q(bt, 95)]; futile = ci[1] < thr
    exists = obs['L2_Aplus'] > null_p95 and ci[0] > 0 and not (obs['vs_Aplus'] > null_p95 and q(bv, 5) > 0)
    verdict = '보류(순방향 기준 미작동)' if not prem['passed'] else '무익' if futile else '있음' if exists else '결정 불가'
    sens = {str(k): dict(threshold=max(a_pair, 0.02 / k), futile=bool(ci[1] < max(a_pair, 0.02 / k))) for k in g['sensitivity_ratios']}
    res = dict(smoke=a.smoke, subjects=subs, eval_runs=g['eval_runs'], train_runs=[r + 1 for r in train], n_queries=int(len(pool)), n_blocks=int(len(ub)),
               verdict=verdict, delta=dict(obs=obs['L2_Aplus'], ci90=ci, se=se, null_p95=null_p95, null=null, lower_gt0=bool(ci[0] > 0)),
               a_pair=a_pair, threshold=thr, sensitivity=sens, premise=prem,
               value_shuffle=dict(obs=obs['vs_Aplus'], ci90=[q(bv, 5), q(bv, 95)], meets_existence=bool(obs['vs_Aplus'] > null_p95 and q(bv, 5) > 0)),
               report={k: dict(obs=obs[k], ci90=[q(boot[k], 5), q(boot[k], 95)]) for k in pairs_of},
               pair_mrp={m: float(rp[m].mean()) for m in models}, within_run_mrp={m: float(rw[m].mean()) for m in models}, pred_r=rpred,
               lambdas={m: fit[m]['lam'] for m in models},
               per_subject_delta=dict(zip([str(s) for s in subs], (rp['L2'] - rp['Aplus']).mean(1).tolist())),
               decoder_forward_ratio_report=(0.840 - 0.5) / (float(rp['Aplus'].mean()) - 0.5) if rp['Aplus'].mean() > 0.5 else None,
               seconds=time.time() - t0)
    o = os.path.join(ROOT, c['out']); os.makedirs(o, exist_ok=True)
    json.dump(res, open(os.path.join(o, 'gateJ' + ('_smoke' if a.smoke else '') + '.json'), 'w'), indent=1, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ('verdict', 'delta', 'a_pair', 'threshold', 'premise', 'pair_mrp')}, ensure_ascii=False, default=str)[:3000], flush=True)


if __name__ == '__main__':
    main()
