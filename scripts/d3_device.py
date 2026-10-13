"""Direction 3 faithfulness device: synthetic-evidence simulator (D-087), false-positive gate (D-088), p* and b (D-089).

Stage generate (GPU): for every trial of the hashed candidate file (D-086), greedy outputs of a fixed LM for the context
alone and with each of the 20 candidates as the evidence text (21 outputs per trial). Each output is scored by the e5
residual cosine (D-076; bge residual as sensitivity) to the 20 candidates; RR = 1 / rank of the true intent (ties
averaged). Column 0 of the RR table is "alone", column 1 + j is "evidence = candidate j". Candidate and target embeddings
for the R structure use the D-068 TXT embedding (ds004408_identify.embed), centred on the D-075 resid_fit mean.

Stage simulate (CPU): evidence index distributions q_i(j) at retrieval accuracy p.
  U: q_i(target) = p, other candidates (1 - p) / 19.
  R: evidence = argmax_j <t_i + g n, c_j / |c_j|> with n ~ N(0, Sigma * mean |t|^2) (D-068 residual covariance), g set by
     bisection so that mean_i q_i(target) = p (common random numbers); p = 0.05 uses g = inf (noise only).
  Swap evidence = the evidence index of another trial, applied to trial i's candidate list (zero information by design).
  Exact trial-level moments: delta_c(p) = mean_i E[d_i], sigma_c(p)^2 = mean_i E[d_i^2] - delta^2 for
  d = RR(real) - RR(swap) and d = RR(real) - RR(alone).
  tau_c(p) = SD over u ~ N(0, s^2) of delta_c(expit(logit p + u)) on the isotonic delta curve (D-083).
  p*_c(N, S) = smallest grid p from which delta >= z * sqrt(tau^2/S + sigma^2/(S N)) holds at every larger grid p, linear
  interpolation of the standardised effect; S = 1 drops tau. p* = max(p*_swap, p*_alone). Trial bootstrap for intervals.

usage: python scripts/d3_device.py configs/d3_device.yaml --stage generate [--n 30 --candidates PATH --tag pilot]
       python scripts/d3_device.py configs/d3_device.yaml --stage simulate   (functions used by D-088/D-089 drivers)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, hashlib, argparse, yaml, numpy as np, pandas as pd
from scipy.stats import norm
from scipy.special import expit, logit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d3_metric_gate import Embedder, len_feats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_trials(path, manifest=None, n=0):
    if manifest:
        h = json.load(open(os.path.join(ROOT, manifest)))['sha256']
        assert hashlib.sha256(open(os.path.join(ROOT, path), 'rb').read()).hexdigest() == h, 'candidate file changed'
    T = [json.loads(l) for l in open(os.path.join(ROOT, path))]
    return T[:n] if n else T


def prompt(tok, tr, evidence=None):
    lines = '\n'.join(f'{s}: {u}' for s, u in zip(tr['speakers'], tr['context']))
    hint = ('' if evidence is None else
            f'A brain-computer interface decoded a hint about what speaker A wants to say. The hint may be wrong.\nHint: "{evidence}"\n\n')
    msg = (f'Here is a conversation between two speakers, A and B.\n\n{lines}\n\n{hint}'
           'Write the next thing speaker A says. Reply with that one utterance only, without the speaker label or quotation marks.')
    return tok.apply_chat_template([{'role': 'user', 'content': msg}], tokenize=False, add_generation_prompt=True)


def rr_table(E, outputs, cands, target_index):
    """outputs [n][21], cands [n][20] -> RR [n, 21] by residual cosine; ties averaged."""
    n = len(cands); RR = np.zeros((n, len(outputs[0])))
    def resid(texts):
        R = E.raw(texts) - len_feats(texts) @ E.W; return R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    for i in range(n):
        S = resid(outputs[i]) @ resid(cands[i]).T; t = S[:, target_index[i]][:, None]
        rank = 1 + (S > t).sum(1) + 0.5 * ((S == t).sum(1) - 1); RR[i] = 1 / rank
    return RR


def generate(c, a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from d3_candidates import clean
    from ds004408_identify import embed
    t0 = time.time(); L = c['lm']; cand_path = a.candidates or c['candidates']
    T = load_trials(cand_path, None if a.candidates else c['candidates_manifest'], a.n)
    out = os.path.join(ROOT, c['out_dir'] + (f'_{a.tag}' if a.tag else '')); os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(L['model'], padding_side='left')
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(L['model'], dtype=getattr(torch, L['dtype'])).cuda().eval()
    jobs = [(i, None) for i in range(len(T))] + [(i, j) for i in range(len(T)) for j in range(len(T[i]['candidates']))]
    res = {}
    for b0 in range(0, len(jobs), L['batch_prompts']):
        bj = jobs[b0:b0 + L['batch_prompts']]
        enc = tok([prompt(tok, T[i], None if j is None else T[i]['candidates'][j]) for i, j in bj], return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            g = m.generate(**enc, do_sample=False, max_new_tokens=L['max_new_tokens'], pad_token_id=tok.pad_token_id)
        for (i, j), s in zip(bj, tok.batch_decode(g[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)): res[(i, j)] = clean(s)
        if b0 // L['batch_prompts'] % 50 == 0:
            print(f'{b0 + len(bj)}/{len(jobs)} ({time.time() - t0:.0f}s), cuda max {torch.cuda.max_memory_allocated() / 2**30:.2f} GB', flush=True)
    del m; torch.cuda.empty_cache()
    outputs = [[res[(i, None)]] + [res[(i, j)] for j in range(len(T[i]['candidates']))] for i in range(len(T))]
    cands = [tr['candidates'] for tr in T]; ti = [tr['target_index'] for tr in T]
    fit = list(pd.read_csv(os.path.join(ROOT, c['resid_fit']), sep='\t', keep_default_na=False).text); RR = {}
    for name, spec in c['metrics'].items():
        E = Embedder(spec, 128, fit); RR[name] = rr_table(E, outputs, cands, ti); del E
    flat = [s for cc in cands for s in cc]; emb = embed(flat).reshape(len(T), len(cands[0]), -1); mu = embed(fit).mean(0)
    np.savez_compressed(os.path.join(out, 'device.npz'), rr_e5=RR['primary'], rr_bge=RR['sensitivity'], emb=(emb - mu).astype(np.float32),
                        target_index=np.array(ti), trial=np.array([tr['trial'] for tr in T]))
    with open(os.path.join(out, 'outputs.jsonl'), 'w') as fh:
        for i, tr in enumerate(T): fh.write(json.dumps(dict(trial=tr['trial'], outputs=outputs[i]), ensure_ascii=False) + '\n')
    man = dict(n_trials=len(T), candidates=cand_path, lm=L, seconds=time.time() - t0, cuda_max_gb=torch.cuda.max_memory_allocated() / 2**30,
               sha256={f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in ('device.npz', 'outputs.jsonl')},
               copy_rate_evidence=float(np.mean([outputs[i][1 + j].strip().lower() == cands[i][j].strip().lower() for i in range(len(T)) for j in range(len(cands[i]))])),
               mean_rr_alone=float(RR['primary'][:, 0].mean()), mean_rr_target_evidence=float(np.mean([RR['primary'][i, 1 + ti[i]] for i in range(len(T))])),
               mean_rr_wrong_evidence=float(np.mean([RR['primary'][i, 1 + j] for i in range(len(T)) for j in range(len(cands[i])) if j != ti[i]])))
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1); print(json.dumps(man, indent=1), flush=True)


# ---------------------------------------------------------------- simulate
def q_uniform(n, K, ti, p):
    q = np.full((n, K), (1 - p) / (K - 1)); q[np.arange(n), ti] = p; return q


def q_swap_uniform(q):
    """U swap: the evidence index of another trial applied to this trial's list (mean over k != i of q_k)."""
    n = len(q); return (q.sum(0, keepdims=True) - q) / (n - 1)


class RStructure:
    """Residual-structure oracle (D-083): prediction = t_i + g n, n ~ N(0, Sigma * mean|t|^2); evidence = the candidate of
    trial i with the largest cosine to the prediction. Swap (D-050 "EEG of another trial"): the prediction of a random other
    trial k, t_k + g n, searched among trial i's candidates. Common random numbers over draws."""
    def __init__(self, emb, ti, sigma, draws, seed):
        n, K, d = emb.shape; rng = np.random.default_rng(seed); t = emb[np.arange(n), ti].astype(np.float64)
        Cn = emb / (np.linalg.norm(emb, axis=2, keepdims=True) + 1e-12); en = (t ** 2).sum(1).mean()
        Lc = np.linalg.cholesky(sigma + 1e-8 * np.eye(d)); self.A = np.einsum('nd,nkd->nk', t, Cn)
        self.B = np.zeros((draws, n, K), np.float32); self.Asw = np.zeros((draws, n, K), np.float32)
        self.Nt = np.zeros((draws, n)); self.NN = np.zeros((draws, n)); self.tt = (t ** 2).sum(1)
        for r in range(draws):
            N = rng.standard_normal((n, d)) @ Lc.T * np.sqrt(en); self.B[r] = np.einsum('nd,nkd->nk', N, Cn)
            self.Nt[r] = (N * t).sum(1); self.NN[r] = (N ** 2).sum(1)
            k = (np.arange(n) + rng.integers(1, n, n)) % n; self.Asw[r] = np.einsum('nd,nkd->nk', t[k], Cn)
        self.ti, self.n, self.K = ti, n, K

    def _hist(self, S):
        j = S.argmax(2); return (j[..., None] == np.arange(self.K)).mean(0)

    def q(self, g): return self._hist(self.B if np.isinf(g) else self.A[None] + g * self.B)

    def q_swap(self, g): return self._hist(self.B if np.isinf(g) else self.Asw + g * self.B)

    def acc(self, g): return float(self.q(g)[np.arange(self.n), self.ti].mean())

    def cosine(self, g):
        if np.isinf(g): return 0.0
        return float(((self.tt + g * self.Nt) / np.sqrt(self.tt * (self.tt + 2 * g * self.Nt + g ** 2 * self.NN))).mean())

    def scale_for(self, p, lo=-4.0, hi=4.0, iters=40):
        if p >= 1: return 0.0
        if p <= 1 / self.K: return np.inf
        for _ in range(iters):
            mid = (lo + hi) / 2
            if self.acc(10 ** mid) > p: lo = mid
            else: hi = mid
        return 10 ** ((lo + hi) / 2)

    def p_to_cos(self, log10_grid):
        """Monotone accuracy -> cosine table on a log10 g grid (used to carry bootstrap p* to c*)."""
        a = np.minimum.accumulate([self.acc(10 ** x) for x in log10_grid]); cs = np.array([self.cosine(10 ** x) for x in log10_grid])
        o = np.argsort(a); return lambda p: float(np.interp(p, a[o], cs[o]))


def structure_q(e, p, ti, K, R=None):
    """(q, q_swap) at retrieval accuracy p for error structure e in {U, R, I}; R/I given as RStructure."""
    if e == 'U': q = q_uniform(len(ti), K, ti, p); return q, q_swap_uniform(q)
    g = R.scale_for(p); return R.q(g), R.q_swap(g)


def moments(RR, q, qs, rows=None):
    """Exact trial-level moments of d_swap = RR(real) - RR(swap) and d_alone = RR(real) - RR(alone); real and swap evidence
    are independent draws (different trials' EEG). RR [n, 21]; q, qs [n, K]."""
    if rows is not None: RR, q, qs = RR[rows], q[rows], qs[rows]
    ev = RR[:, 1:]; al = RR[:, 0]
    m_real, m2_real = (q * ev).sum(1), (q * ev ** 2).sum(1); m_sw, m2_sw = (qs * ev).sum(1), (qs * ev ** 2).sum(1)
    d_sw = m_real - m_sw; e2_sw = m2_real + m2_sw - 2 * m_real * m_sw; d_al = m_real - al; e2_al = m2_real - 2 * m_real * al + al ** 2
    out = {}
    for k, (d, e2) in dict(swap=(d_sw, e2_sw), alone=(d_al, e2_al)).items():
        delta = float(d.mean()); out[k] = dict(delta=delta, sigma=float(np.sqrt(max(e2.mean() - delta ** 2, 0))), omega=float(d.std()), per_trial=d)
    return out


def iso(y):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(increasing=True).fit_transform(np.arange(len(y)), y)


def p_star(grid, delta, sigma, N, S, s_logit, z, n_gh=40, omega=None):
    """omega = SD over trials of the trial-level mean difference E[d | trial]. None: independent design (D-083 formula,
    tau^2/S + sigma^2/(S N)). Given: shared design (all subjects see the same N trials), tau^2/S + omega^2/N +
    (sigma^2 - omega^2)/(S N); sensitivity only (D-089)."""
    grid = np.asarray(grid, float); d = iso(np.asarray(delta)); x, w = np.polynomial.hermite_e.hermegauss(n_gh); w = w / w.sum()
    tau = np.array([0.0 if p >= 1 else float(np.sqrt(max(np.sum(w * np.interp(expit(logit(p) + s_logit * x), grid, d) ** 2)
                                                         - np.sum(w * np.interp(expit(logit(p) + s_logit * x), grid, d)) ** 2, 0))) for p in grid])
    sg2 = np.asarray(sigma) ** 2; om2 = 0 if omega is None else np.minimum(np.asarray(omega) ** 2, sg2)
    se = np.sqrt((0 if S == 1 else tau ** 2 / S) + om2 / N + (sg2 - om2) / (S * N)); zs = d / np.maximum(se, 1e-12); ok = zs >= z
    if not ok[-1]: return dict(p_star=None, reachable=False, z=zs.tolist(), tau=tau.tolist())
    k = len(ok) - 1
    while k > 0 and ok[k - 1]: k -= 1
    if k == 0: ps = float(grid[0])
    else: ps = float(grid[k - 1] + (z - zs[k - 1]) / (zs[k] - zs[k - 1]) * (grid[k] - grid[k - 1]))
    return dict(p_star=ps, reachable=True, z=zs.tolist(), tau=tau.tolist())


def sim_tests(RR, q, qs, N, S, reps, alpha, rng, shared=False, budget=1e6):
    """Simulated experiments at evidence distributions q_i (real) and qs_i (swap), drawn independently per trial.
    independent design (p* variance model, D-083): each subject's N trials are drawn with replacement independently;
      one-sided t test of mean d > 0 on subject means (S - 1 df; S = 1: trials, N - 1 df).
    shared design: all subjects see the same N trials; rejection requires the subject-level t AND the item-level t (item
      means over subjects, N - 1 df, Johnson 1978 skewness correction because item effects of d_alone are left-skewed) to
      exceed their critical values (S = 1: trial t only).
    Returns rejections [reps, 2] for (swap, alone)."""
    from scipy.stats import t as student_t
    n, K = q.shape; cq, cs = np.cumsum(q, 1), np.cumsum(qs, 1); ev, al = RR[:, 1:], RR[:, 0]
    crit_s, crit_i = student_t.ppf(1 - alpha, max(S - 1, 1)), student_t.ppf(1 - alpha, N - 1); rej = np.zeros((reps, 2), bool)
    def tstat(v, johnson=False):   # Johnson (1978) skewness-corrected t for the item-level test only (see shared design)
        k = v.shape[1]; m = v.mean(1); sd = v.std(1, ddof=1)
        mu3 = ((v - m[:, None]) ** 3).sum(1) * k / ((k - 1) * (k - 2)) if johnson else 0
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nan_to_num((m + mu3 / (6 * sd ** 2 * k) + mu3 / (3 * sd ** 4) * m ** 2) / (sd / np.sqrt(k)), nan=-np.inf)
    step = max(1, int(budget // (S * N)))
    for r0 in range(0, reps, step):
        m = min(step, reps - r0)
        idx = np.repeat(rng.integers(0, n, (m, 1, N)), S, 1) if shared else rng.integers(0, n, (m, S, N))
        jr = np.minimum((cq[idx] < rng.random((m, S, N, 1))).sum(-1), K - 1)
        js = np.minimum((cs[idx] < rng.random((m, S, N, 1))).sum(-1), K - 1)
        er = ev[idx, jr]
        for ci, x in enumerate((er - ev[idx, js], er - al[idx])):
            if S == 1: rej[r0:r0 + m, ci] = tstat(x[:, 0, :]) > crit_i; continue
            ok = tstat(x.mean(2)) > crit_s
            if shared: ok &= tstat(x.mean(1), johnson=True) > crit_i
            rej[r0:r0 + m, ci] = ok
    return rej


def cp_upper(x, n, conf=0.95):
    from scipy.stats import beta
    return 1.0 if x >= n else float(beta.ppf(conf, x + 1, n - x))


def wolpaw(p, K):
    if p is None: return None
    p = min(max(p, 1 / K), 1.0); b = np.log2(K) + (p * np.log2(p) if p > 0 else 0) + ((1 - p) * np.log2((1 - p) / (K - 1)) if p < 1 else 0)
    return float(b)


def load_device(c, tag=''):
    d = np.load(os.path.join(ROOT, c['out_dir'] + (f'_{tag}' if tag else ''), 'device.npz'))
    return {k: d[k] for k in d.files}


def gate(c, a):
    """D-088: zero-information false-positive gate and high-information positive control."""
    t0 = time.time(); sm = c['simulate']; gc = sm['gate']; D = load_device(c, a.tag); ti = D['target_index']; K = D['rr_e5'].shape[1] - 1
    sigma = np.load(os.path.join(ROOT, c['sigma']))['sigma']; rng = np.random.default_rng([c['seed'], 88])
    R = RStructure(D['emb'], ti, sigma, sm['r_calib_draws'], c['seed']); res = dict(n_trials=int(len(ti)), cells={}, positive_control={})
    zero = dict(U=(q_uniform(len(ti), K, ti, 1 / K),), R=(R.q(np.inf),))
    zero['U'] += (q_swap_uniform(zero['U'][0]),); zero['R'] += (R.q_swap(np.inf),)
    one = dict(U=structure_q('U', 1.0, ti, K), R=structure_q('R', 1.0, ti, K, R))
    res['zero_info_accuracy'] = {e: float(zero[e][0][np.arange(len(ti)), ti].mean()) for e in zero}
    # D-087 diagnostic: the draft index swap (another trial's evidence index applied to this trial's list) under R at zero
    # information, against the registered vector swap (identical to the real-evidence distribution at g = inf)
    qi = q_swap_uniform(zero['R'][0]); res['index_swap_check'] = dict(
        mean_tv_real_vs_index_swap=float(0.5 * np.abs(zero['R'][0] - qi).sum(1).mean()), mean_tv_real_vs_vector_swap=float(0.5 * np.abs(zero['R'][0] - zero['R'][1]).sum(1).mean()),
        mean_max_q_real=float(zero['R'][0].max(1).mean()),
        **{f'delta_swap_index_{m}': moments(D[f'rr_{m}'], zero['R'][0], qi)['swap']['delta'] for m in ('e5', 'bge')},
        **{f'delta_swap_vector_{m}': moments(D[f'rr_{m}'], *zero['R'])['swap']['delta'] for m in ('e5', 'bge')})
    # D-088: zero-information mean differences (Delta_alone here = effect of having any hint at all, not information use)
    res['zero_info_delta'] = {f'{m}|{e}': {k: moments(D[f'rr_{m}'], *zero[e])[k]['delta'] for k in ('swap', 'alone')} for m in ('e5', 'bge') for e in zero}
    for metric in ('e5', 'bge'):
        RR = D[f'rr_{metric}']
        for e in ('U', 'R'):
            for design in gc['designs']:
                for N in sm['N']:
                    for S in sm['S']:
                        reps = gc['primary_reps'] if [N, S] == list(gc['primary_cell']) and metric == 'e5' else gc['reps']   # D-088: gate cells only
                        rej = sim_tests(RR, *zero[e], N, S, reps, gc['alpha'], rng, shared=design == 'shared')
                        x = dict(swap=int(rej[:, 0].sum()), alone=int(rej[:, 1].sum()), both=int(rej.all(1).sum()))
                        res['cells'][f'{metric}|{e}|{design}|N{N}|S{S}'] = {k: dict(count=v, reps=reps, rate=v / reps, cp_upper=cp_upper(v, reps)) for k, v in x.items()}
            m = moments(RR, *one[e]); bs = rng.integers(0, len(ti), (sm['boot'], len(ti)))
            res['positive_control'][f'{metric}|{e}'] = {k: dict(delta=m[k]['delta'], lower95=float(np.quantile(m[k]['per_trial'][bs].mean(1), 0.05))) for k in ('swap', 'alone')}
            print(f'{metric} {e} done ({time.time() - t0:.0f}s)', flush=True)
    N0, S0 = gc['primary_cell']; verdict = {}
    for e in ('U', 'R'):
        cells = {dz: res['cells'][f'e5|{e}|{dz}|N{N0}|S{S0}'][gc['primary_stat']] for dz in gc['designs']}; pc = res['positive_control'][f'e5|{e}']['swap']
        verdict[e] = dict(fp_cp_upper={dz: v['cp_upper'] for dz, v in cells.items()}, fp_pass=all(v['cp_upper'] <= gc['cp_max'] for v in cells.values()),
                          pc_lower95=pc['lower95'], pc_pass=pc['lower95'] > 0,
                          # individual comparisons at the gate cell: reported; above cp_max -> warning on that comparison's p* (not a gate failure)
                          single={dz: {k: res['cells'][f'e5|{e}|{dz}|N{N0}|S{S0}'][k]['cp_upper'] for k in ('swap', 'alone')} for dz in gc['designs']},
                          single_warning=sorted({k for dz in gc['designs'] for k in ('swap', 'alone') if res['cells'][f'e5|{e}|{dz}|N{N0}|S{S0}'][k]['cp_upper'] > gc['cp_max']}))
    res['verdict'] = dict(per_structure=verdict, device_pass=all(v['fp_pass'] and v['pc_pass'] for v in verdict.values()))
    res['seconds'] = time.time() - t0; out = os.path.join(ROOT, sm['results_dir'] + (f'_{a.tag}' if a.tag else '')); os.makedirs(out, exist_ok=True)
    json.dump(dict(config=c, **res), open(os.path.join(out, 'gate.json'), 'w'), indent=1); print(json.dumps(res['verdict'], indent=1), flush=True)


def b_safe(cs, curve, mrp_B):
    """b_from_cstar with c* = 0 -> MRP_s 0.5, b 0; c* outside the curve -> (None, nan), flagged for the D-083 grid extension."""
    from d3_bconv import b_from_cstar
    if np.isinf(cs): return None, float('inf')            # p* unreachable -> b = +inf (D-083)
    if cs <= 0: return 0.5, 0.0
    try: return b_from_cstar(cs, curve, mrp_B)
    except ValueError: return None, float('nan')


def pstar_stage(c, a):
    """D-089: p* per error structure, metric and (N, S), trial-bootstrap intervals, c* and b (D-083). p*, the specification
    value and b use the independent design (D-083 variance formula); the shared design (item variance) is a sensitivity."""
    t0 = time.time(); sm = c['simulate']; D = load_device(c, a.tag); ti = D['target_index']; n = len(ti); K = D['rr_e5'].shape[1] - 1
    out = os.path.join(ROOT, sm['results_dir'] + (f'_{a.tag}' if a.tag else ''))
    g = json.load(open(os.path.join(out, 'gate.json')))
    if not g['verdict']['device_pass']:
        json.dump(dict(p_star=None, b=None, note='device failed the D-088 gate: p* and b undefined (D-083)'), open(os.path.join(out, 'pstar.json'), 'w')); return
    sigma = np.load(os.path.join(ROOT, c['sigma']))['sigma']; d = sigma.shape[0]
    St = dict(R=RStructure(D['emb'], ti, sigma, sm['r_calib_draws'], c['seed']), I=RStructure(D['emb'], ti, np.eye(d) / d, sm['r_calib_draws'], c['seed']))
    cj = json.load(open(os.path.join(ROOT, sm['bconv_curve']))); curve = cj['curves']; mrp_B = sm['mrp_B']; grid = sm['p_grid']
    rng = np.random.default_rng([c['seed'], 89]); bs = rng.integers(0, n, (sm['boot'], n)); res = dict(n_trials=n, structures={})
    lg = np.linspace(-4, 4, 161); to_cos = {e: St[e].p_to_cos(lg) for e in St}
    # D-089 reporting: target word-count bands (D-086 dropout), least-central baseline and alone RR (D-086 item 3)
    from d3_metric_gate import nwords
    T = load_trials(c['candidates'], c['candidates_manifest']); assert [t['trial'] for t in T] == D['trial'].tolist()
    tw = np.array([nwords(t['target']) for t in T]); bands = {f'{lo}-{hi}': np.where((tw >= lo) & (tw <= hi))[0] for lo, hi in sm['report_bands']}
    cm = json.load(open(os.path.join(ROOT, c['candidates_manifest'])))
    res['reporting'] = dict(baseline_least_central_rr=cm['baseline_least_central_rr'], chance_rr=cm['chance_rr'],
                            mean_rr_alone={m: float(D[f'rr_{m}'][:, 0].mean()) for m in ('e5', 'bge')}, band_n={b: int(len(r)) for b, r in bands.items()},
                            zero_info_accuracy=g['zero_info_accuracy'], zero_info_delta=g['zero_info_delta'], envelope_calibration=cj['calibration'])
    for e in ('U', 'R', 'I'):
        Q = [structure_q(e, p, ti, K, St.get(e)) for p in grid]
        # R/I: the p coordinate is the accuracy actually reached (grid values below the zero-information accuracy collapse
        # onto g = inf); duplicates are dropped keeping the g = inf point
        pg = np.array(grid, float) if e == 'U' else np.array([float(q[np.arange(n), ti].mean()) for q, _ in Q])
        keep = np.ones(len(grid), bool) if e == 'U' else np.r_[True, np.array(grid[1:]) > pg[0]]
        if e != 'U': pg = np.maximum.accumulate(pg)
        Q = [x for x, k in zip(Q, keep) if k]; pg = pg[keep]
        for metric in ('e5', 'bge'):
            RR = D[f'rr_{metric}']; M = [moments(RR, q, qs) for q, qs in Q]
            mom = {k: tuple(np.array([m[k][x] for m in M]) for x in ('delta', 'sigma', 'omega')) for k in ('swap', 'alone')}
            cell = dict(p_coord=pg.tolist(), moments={k: dict(delta=mom[k][0].tolist(), sigma=mom[k][1].tolist(), omega=mom[k][2].tolist()) for k in mom},
                        band_delta={b: {k: [float(m[k]['per_trial'][r].mean()) for m in M] for k in ('swap', 'alone')} for b, r in bands.items()})
            bmom = {k: tuple(np.zeros((sm['boot'], len(pg))) for _ in range(3)) for k in mom}
            for r in range(sm['boot']):
                for gi, (q, qs) in enumerate(Q):
                    mb = moments(RR, q, qs, bs[r])
                    for k in mom: bmom[k][0][r, gi], bmom[k][1][r, gi], bmom[k][2][r, gi] = mb[k]['delta'], mb[k]['sigma'], mb[k]['omega']
            for N in sm['N']:
                for S in sm['S']:
                    pf = lambda dl, sg, om=None: p_star(pg, dl, sg, N, S, sm['logit_sd_subject'], sm['z_power'], omega=om)
                    ps = {k: pf(*mom[k][:2]) for k in mom}
                    val = lambda r: r['p_star'] if r['reachable'] else np.inf
                    boot = np.array([max(val(pf(bmom[k][0][r], bmom[k][1][r])) for k in mom) for r in range(sm['boot'])])
                    ps_sh = {k: pf(*mom[k]) for k in mom}
                    boot_sh = np.array([max(val(pf(bmom[k][0][r], bmom[k][1][r], bmom[k][2][r])) for k in mom) for r in range(sm['boot'])])
                    pst_sh = max(val(ps_sh[k]) for k in mom)
                    shared = dict(p_star=None if np.isinf(pst_sh) else pst_sh, p_star_swap=ps_sh['swap']['p_star'], p_star_alone=ps_sh['alone']['p_star'],
                                  ci90=np.quantile(boot_sh, [0.05, 0.95]).tolist(), boot_unreachable=float(np.isinf(boot_sh).mean()))
                    vals = [ps[k]['p_star'] if ps[k]['reachable'] else np.inf for k in ('swap', 'alone')]; pst = max(vals)
                    lo, hi = np.quantile(boot, [0.05, 0.95])
                    cell[f'N{N}|S{S}'] = dict(p_star=None if np.isinf(pst) else pst, p_star_swap=ps['swap'], p_star_alone=ps['alone'],
                                              ci90=[float(lo), float(hi)], boot_unreachable=float(np.isinf(boot).mean()),
                                              bits_per_selection=wolpaw(None if np.isinf(pst) else pst, K),
                                              bits_per_min=None if np.isinf(pst) else wolpaw(pst, K) * 60 / sm['seconds_per_selection'],
                                              shared_design_sensitivity=shared)
                    if (N, S) == tuple(sm['b_cell']) and metric == 'e5':
                        bvals = {}
                        for src, curve_type in (('R', 'residual'), ('I', 'isotropic')):
                            if e not in ('U', src): continue
                            cs = np.inf if np.isinf(pst) else St[src].cosine(St[src].scale_for(pst)); mrp_s, b = b_safe(cs, curve[curve_type], mrp_B)
                            bb = np.array([b_safe(np.inf if np.isinf(x) else to_cos[src](x), curve[curve_type], mrp_B)[1] for x in boot], float)
                            bvals[f'{src}_{curve_type}'] = dict(c_star=cs, mrp_s=mrp_s, b=b, b_ci90=np.nanquantile(bb, [0.05, 0.95]).tolist(),
                                                                boot_outside_curve=float(np.isnan(bb).mean()))
                        cell[f'N{N}|S{S}']['b'] = bvals
            res['structures'][f'{e}|{metric}'] = cell; print(f'{e} {metric} done ({time.time() - t0:.0f}s)', flush=True)
    N0, S0 = sm['b_cell']; prim = res['structures']['R|e5'][f'N{N0}|S{S0}']
    # D-088: alone CP upper above cp_max at the gate cell -> p*_alone reached at zero information, uninterpretable; the claim rests on Delta_swap
    res['alone_uninterpretable'] = {e: 'alone' in g['verdict']['per_structure'][e]['single_warning'] for e in g['verdict']['per_structure']}
    res['registered'] = dict(R_zero_info_accuracy=g['zero_info_accuracy']['R'], p_star_R=prim['p_star'], b=prim['b']['R_residual']['b'], b_ci90=prim['b']['R_residual']['b_ci90'],
                             spec_p_star={f'N{N}|S{S}': max([res['structures'][f'{e}|e5'][f'N{N}|S{S}']['p_star'] or np.inf for e in ('U', 'R')]) for N in sm['N'] for S in sm['S']},
                             alone_uninterpretable=res['alone_uninterpretable'])
    res['seconds'] = time.time() - t0
    json.dump(dict(config=c, **res), open(os.path.join(out, 'pstar.json'), 'w'), indent=1, default=float); print(json.dumps(res['registered'], indent=1, default=float), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', required=True, choices=['generate', 'gate', 'pstar'])
    ap.add_argument('--n', type=int, default=0); ap.add_argument('--candidates'); ap.add_argument('--tag', default=''); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); dict(generate=generate, gate=gate, pstar=pstar_stage)[a.stage](c, a)
