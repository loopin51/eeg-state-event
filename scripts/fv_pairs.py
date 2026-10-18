"""F-VII (D-115 sections 2-3, a9d7685; amendments 1 6d6c5cb, 2 7cb51f6, 3 fdc27f2): blur-level information vs concept, pair
test on the development photos (concepts the development models never saw). Replaces the 08d0365 pair-construction script of
this path (now scripts/fv_pairs_build.py, eeg-a4).

Space: C(j) = unit(original-image CLIP(j) - mean over the 14,540 non-development training images). Pairs (eeg-a4): per photo i,
a sibling s (same concept) and a foil f (another development concept) that the three low-level predictors all put closer to i.
Pair accuracy of an output z (per subject: centred over the subject's 2,000 development photos, L2-normalised):
A = mean over pairs of 1[cos(z_i, C(s)) > cos(z_i, C(f))]. Held-out blur predictors H_a, H_b (comparison), H_c (simulation):
A_H on the same pairs, common to the subjects. D = A_dec - max(0.5, A_Ha, A_Hb).
Pigeonhole: subjects x concepts (the concept of i) with replacement; A is a ratio (correct / pairs) over the drawn cells, the
H accuracies are recomputed on the drawn concepts. One index set (seed 20262111) for every verdict statistic.

  infer  : EEG forward pass (server gpuq CPU). Per subject, development photos = the mean over every repetition of the four
           sessions' training runs; decoder seeds 20261801/02 (ses123, k0), label-shuffle model (ses123, k0), Gaussian input with
           the channel sd (seed [20262112, subject]) through the decoder seeds (ses123). Raw outputs to data/cache/fv_pairs/infer
           (not committed); manifest (sha256 of models, inputs, outputs; commits of the runs; development-concept MRP of the
           seed-mean decoder, set A ses123, as the pre-verdict quality check) to results/fv_pairs/infer_manifest.json.
  gate   : no EEG. Feasibility (1) r = C(i) picks s >= 0.6, (2) >= 100 concepts and 400 photos, (3) decoder-level C(i) signal:
           mean D >= +0.02, (4) leave-one-out blur simulations (z = H_x(i) + noise, baseline max(0.5, the other two H)) K1 pass
           rate <= 5% each; plus information zero (isotropic Gaussian, K1 pass rate <= 5%). Noise matching: per photo
           z = signal + sigma g, concept row = mean of its 10 photos; sigma such that the 200-way development name MRP of the
           concept rows is 0.808 (bisection on common random numbers; sigma = 0 when the noiseless signal is below 0.808 ->
           deterministic simulation, judged once); pair inputs use sigma x sqrt(2/4). Report: concept-prototype signal
           (mean C of the concept's other photos except s), A_H, H development name MRP. Writes results/fv_pairs/gate.json.
  verdict: once, after gate.json says open. K1 adopted when the 5th percentile of D > 0, D >= +0.02 and J passes (rule N on
           A - 0.5 for the label-shuffle output and the Gaussian input). Report: k0, per subject (old 6 / new 4), sigma-4
           subset, name form, largest-margin pairs.
usage: python scripts/fv_pairs.py configs/fv_pairs.yaml --stage {infer,gate,verdict} [--config-override YAML]
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


def peak_gb():
    import resource; return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ---------------------------------------------------------------- common
class Setup:
    """development photos, the centred space C, the pairs and the held-out predictors (no EEG)"""
    def __init__(self, c, tc, meta):
        self.dev_c = meta.dev_concepts(tc['dev']['salt'], tc['dev']['n']); ic = np.array(meta.train_img_concept); ci = {x: k for k, x in enumerate(self.dev_c)}
        self.rows = np.flatnonzero(np.isin(ic, self.dev_c)); assert len(self.rows) == 2000, len(self.rows)
        self.concept_of_row = {int(r): ci[ic[r]] for r in self.rows}
        nt = c['pairs']['n_train_rows']; X = np.load(P(c['pairs']['clip_img']), mmap_mode='r'); other = np.setdiff1d(np.arange(nt), self.rows)
        assert len(other) == 14540; self.centre = np.asarray(X[other], np.float64).mean(0); self.C = V.unit(np.asarray(X[:nt], np.float64) - self.centre)
        from fv_train import name_rows
        self.Tdev = V.unit(np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['names_text']))[name_rows(tc, meta, self.dev_c)])
        self.main = self.pairs(c['pairs']['main']); self.maxm = self.pairs(c['pairs']['maxmargin']) if c['pairs'].get('maxmargin') else None
        H = np.load(P(c['pairs']['holdout'])); hr = np.asarray(H['rows']); pos = {int(r): k for k, r in enumerate(hr)}; assert set(hr.tolist()) == set(self.rows.tolist())
        self.H = {k: V.unit(np.asarray(H[k], np.float64)[[pos[int(r)] for r in self.rows]]) for k in ('H_a', 'H_b', 'H_c')}      # rows in self.rows order
        self.pos = {int(r): k for k, r in enumerate(self.rows)}; self.files = {k: fsha(P(v)) for k, v in c['pairs'].items() if k in ('main', 'maxmargin', 'holdout') and v}

    def pairs(self, path):
        d = json.load(open(P(path))); p = d['pairs']; I = np.array([q['i'] for q in p]); S = np.array([q['s'] for q in p]); F = np.array([q['f'] for q in p])
        assert np.all(np.diff(I) > 0) and set(I) <= set(self.rows.tolist()), 'pairs: one pair per photo, i ascending, development photos only'
        cc = np.array([self.concept_of_row[int(i)] for i in I]); assert all(self.concept_of_row[int(s)] == k for s, k in zip(S, cc)) and all(self.concept_of_row[int(f)] != k for f, k in zip(F, cc))
        return dict(I=I, S=S, F=F, concept=cc, raw=d, b4=np.array([q.get('blur4_picks_f', None) for q in p], dtype=object))

    def name_mrp(self, Z):
        """Z [2000, d] photo rows (self.rows order) -> 200-way development name MRP of the concept rows (mean of the 10 photos)"""
        cr = np.array([self.concept_of_row[int(r)] for r in self.rows]); R = V.unit(np.stack([Z[cr == k].mean(0) for k in range(len(self.dev_c))]))
        return float(V.cells(R @ self.Tdev.T, np.arange(len(self.dev_c))).mean())


def centre_unit(Z): Z = np.asarray(Z, np.float64); return V.unit(Z - Z.mean(-2, keepdims=True))


def correct(Zc, pr, st):
    """Zc [2000, d] centred unit rows (setup row order) -> 1[cos(z_i, C(s)) > cos(z_i, C(f))] per pair"""
    zi = Zc[[st.pos[int(i)] for i in pr['I']]]; return (np.einsum('nd,nd->n', zi, st.C[pr['S']]) > np.einsum('nd,nd->n', zi, st.C[pr['F']])).astype(np.float64)


def cells(cor, pr, nconc):
    """per-pair correctness [n] or [ns, n] -> (correct sums per concept [.., nconc], pairs per concept [nconc])"""
    N = np.bincount(pr['concept'], minlength=nconc).astype(np.float64)
    if cor.ndim == 1: return np.bincount(pr['concept'], weights=cor, minlength=nconc), N
    return np.stack([np.bincount(pr['concept'], weights=x, minlength=nconc) for x in cor]), N


def resample(ns, concepts, nb, rng):
    """crossed subjects x concepts (only concepts with pairs) as multiplicity counts"""
    Cs, Cc = np.zeros((nb, ns)), np.zeros((nb, len(concepts)))
    for b in range(nb): Cs[b] = np.bincount(rng.integers(0, ns, ns), minlength=ns); Cc[b] = np.bincount(rng.integers(0, len(concepts), len(concepts)), minlength=len(concepts))
    return Cs, Cc, concepts


def A_boot(Sc, N, R):
    Cs, Cc, k = R
    if Sc.ndim == 1: return (Cc @ Sc[k]) / (Cc @ N[k])
    return np.einsum('bs,sc,bc->b', Cs, Sc[:, k], Cc) / (Cs.sum(1) * (Cc @ N[k]))


def k1(Sdec, Href, N, R, c):
    """Sdec [ns, nconc] correct sums of the output; Href {name: [nconc]} baseline predictors -> K1 rule pieces"""
    rl = c['rules']; a = float(Sdec.sum() / (Sdec.shape[0] * N.sum())); ah = {h: float(v.sum() / N.sum()) for h, v in Href.items()}
    base = max([0.5] + list(ah.values())); bb = np.max(np.stack([np.full(len(R[0]), 0.5)] + [A_boot(v, N, R) for v in Href.values()]), 0)
    db = A_boot(Sdec, N, R) - bb; d = a - base; lo = float(np.percentile(db, rl['lower_pct']))
    return dict(A=a, A_ref=ah, baseline=base, D=d, lower=lo, passed=bool(lo > 0 and d >= rl['k1_min']))


def ctl(Sc, N, R, a_dec, c):
    rl = c['rules']; a = float(Sc.sum() / (Sc.shape[0] * N.sum()))
    return dict(A=a, **V.control_rule(np.array([a - 0.5]), A_boot(Sc, N, R) - 0.5, a_dec - 0.5, rl['control_ratio'], rl['control_lower_pct']))


# ---------------------------------------------------------------- infer (EEG)
def run_dir(c, tc, t, seed, shuffle):
    m = [x for x in glob.glob(os.path.join(P(tc['out_root']), f"*_fv_dev_{c['dev_runs']['candidate']}_s{seed}_t{t:02d}{'_shuffle' if shuffle else ''}"))
         if os.path.exists(os.path.join(x, 'layers.pt')) and not x.endswith('_smoke')]
    assert len(m) == 1, (t, seed, shuffle, m); return m[0]


def stage_infer(c, tc, meta, a):
    import torch
    from ese.things.data import subject_mean
    from ese.things.models import build
    st_rows = np.flatnonzero(np.isin(np.array(meta.train_img_concept), meta.dev_concepts(tc['dev']['salt'], tc['dev']['n']))); dev_codes = st_rows + 1
    pc = yaml.safe_load(open(P(tc['prep_config']))); cache = P(pc['cache_dir']); ic = c['infer']; dv = ic['device']; od = P(ic['out_dir']); os.makedirs(od, exist_ok=True)
    pos = torch.as_tensor(np.load(P(tc['positions'])), dtype=torch.float32, device=dv); dr = c['dev_runs']; man = dict(stage='infer', runs={}, files={}, inputs={}, quality={})
    from fv_train import name_rows
    dev_c = meta.dev_concepts(tc['dev']['salt'], tc['dev']['n']); Tdev = V.unit(np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['names_text']))[name_rows(tc, meta, dev_c)])
    t0 = time.time()
    for t in dr['old_targets'] + dr['new_targets']:
        dirs = [(f'seed{j}', run_dir(c, tc, t, s, False)) for j, s in enumerate(dr['seeds'])] + [('shuffle', run_dir(c, tc, t, dr['shuffle_seed'], True))]
        if c['smoke']: codes, nrep = dev_codes, np.full(len(dev_codes), 4); X = np.random.default_rng([1, t]).standard_normal((len(dev_codes), pos.shape[0], c['smoke_T'])).astype(np.float32)   # code check: no EEG
        else: codes, X, nrep = subject_mean(cache, t, ic['sessions'], codes=dev_codes)
        assert np.array_equal(codes, dev_codes), f't{t}: development photos missing'
        X = np.asarray(X, np.float32); sd = X.std(axis=(0, 2)); Xn = (np.random.default_rng([ic['noise_seed'], t]).standard_normal(X.shape) * sd[None, :, None]).astype(np.float32)
        man['inputs'][f't{t:02d}'] = dict(n_photos=int(len(codes)), nrep_min=int(nrep.min()), nrep_mean=float(nrep.mean()), nrep_max=int(nrep.max()))
        out, Es = {}, []
        for tag, d in dirs:
            rc = yaml.safe_load(open(os.path.join(d, 'config.yaml'))); r = rc['run']; assert r['target'] == t and r['mode'] == 'dev' and r['candidate'] == dr['candidate'] and r['shuffle'] == (tag == 'shuffle')
            cm = open(os.path.join(d, 'k3s_commit.txt')).read().strip() if os.path.exists(os.path.join(d, 'k3s_commit.txt')) else None; assert c['smoke'] or cm == dr['commit'], (d, cm)
            ti = len(r['trainers']); net = build(r['candidate'], ti + 1, rc['model']).to(dv); net.load_state_dict(torch.load(os.path.join(d, 'shared.pt'), map_location=dv)['net']); net.eval()
            layers = torch.load(os.path.join(d, 'layers.pt'), map_location=dv)
            man['runs'][f't{t:02d}_{tag}'] = dict(dir=os.path.basename(d), commit=cm, shared_sha256=fsha(os.path.join(d, 'shared.pt')), layers_sha256=fsha(os.path.join(d, 'layers.pt')),
                                                  attempts=open(os.path.join(d, 'k3s_attempts.txt')).read().strip() if os.path.exists(os.path.join(d, 'k3s_attempts.txt')) else None)
            for lay in ic['layers']:
                with torch.no_grad():
                    net.subject.weights[ti] = layers[lay].to(dv)
                    for kind, XX in (('photo', X),) + ((('noise', Xn),) if (lay == 'ses123' and tag != 'shuffle') else ()):
                        o = [net(torch.from_numpy(XX[b:b + ic['batch']]).to(dv), pos[None].expand(len(XX[b:b + ic['batch']]), -1, -1), torch.full((len(XX[b:b + ic['batch']]),), ti, device=dv)).float().cpu().numpy()
                             for b in range(0, len(XX), ic['batch'])]
                        out[f'{tag}|{lay}|{kind}'] = np.concatenate(o).astype(np.float32)
            if tag != 'shuffle': Es.append(np.load(os.path.join(d, 'dev_scores.npz')))
        f = os.path.join(od, f'z_raw_t{t:02d}.npz'); np.savez(f, rows=st_rows, **out); man['files'][os.path.basename(f)] = fsha(f)
        if not c['smoke']:
            tr = Es[0]['A_ses4_true']; assert all(np.array_equal(z['A_ses4_true'], tr) for z in Es)
            E = V.seed_mean([z['A_ses4_E_ses123'] for z in Es]); man['quality'][f't{t:02d}'] = dict(dev_mrp_seedmean_A_ses123=float(V.cells(E @ Tdev.T, tr).mean()), n_concepts=int(len(tr)),
                                                                                              group='new' if t in dr['new_targets'] else 'old')
        print(f't{t:02d}: outputs saved ({time.time() - t0:.0f}s)', flush=True)
    man.update(device=dv, seconds=time.time() - t0, peak_rss_gb=peak_gb(), smoke=bool(c['smoke']))
    o = P(os.path.join(c['out_dir'], 'infer_manifest.json')); os.makedirs(os.path.dirname(o), exist_ok=True); json.dump(man, open(o, 'w'), indent=1)
    print(json.dumps(dict(quality={k: round(v['dev_mrp_seedmean_A_ses123'], 4) for k, v in man['quality'].items()})), flush=True)


# ---------------------------------------------------------------- gate (no EEG)
def stage_gate(c, tc, meta, a):
    st = Setup(c, tc, meta); pr = st.main; rl = c['rules']; nconc = len(st.dev_c); ns = len(c['dev_runs']['old_targets'] + c['dev_runs']['new_targets']); dim = c['z_dim']; t0 = time.time()
    concepts = np.unique(pr['concept']); res = dict(stage='gate', smoke=bool(c['smoke']), inputs=st.files, n_subjects_simulated=ns)
    ch = json.load(open(P(c['pairs']['main']))).get('center', {})
    if ch.get('file'): res['centre_check_max_abs'] = float(np.abs(np.load(P(ch['file'])) - st.centre).max()); assert res['centre_check_max_abs'] < 1e-6, res['centre_check_max_abs']
    Cdev = st.C[st.rows]; r = float(correct(V.unit(Cdev), pr, st).mean())                   # C(i) is already centred (training mean); used as is
    SH = {k: cells(correct(V.unit(v), pr, st), pr, nconc)[0] for k, v in st.H.items()}; N = cells(np.zeros(len(pr['I'])), pr, nconc)[1]
    res['feasibility'] = {'1_r': dict(value=r, min=rl['gate_r_min'], passed=bool(r >= rl['gate_r_min'])),
                          '2_size': dict(n_photos=int(len(pr['I'])), n_concepts=int(len(concepts)), passed=bool(len(concepts) >= rl['gate_min_concepts'] and len(pr['I']) >= rl['gate_min_photos']))}
    res['A_H'] = {k: float(v.sum() / N.sum()) for k, v in SH.items()}; res['H_dev_name_mrp'] = {k: st.name_mrp(v) for k, v in st.H.items()}
    res['baseline_main'] = max(0.5, res['A_H']['H_a'], res['A_H']['H_b'])

    def calibrate(sig, name):
        """sigma with concept-row name MRP = target (common random numbers); 0 when the noiseless signal is below the target"""
        m0 = st.name_mrp(sig)
        if m0 <= rl['target_mrp']: return 0.0, m0, m0
        G = [np.random.default_rng([c['seeds']['sim'], 9, k]).standard_normal(sig.shape) for k in range(c['n_calib'])]
        f = lambda s: float(np.mean([st.name_mrp(sig + s * g) for g in G])); lo, hi = 0.0, 0.05
        while f(hi) > rl['target_mrp']: lo, hi = hi, hi * 2
        for _ in range(c['calib_bisect']):
            mid = (lo + hi) / 2
            if f(mid) > rl['target_mrp']: lo = mid
            else: hi = mid
        return (lo + hi) / 2, m0, f((lo + hi) / 2)

    def simulate(sig, sigma, refs, case, n):
        """photo-level z = sig + sigma x factor x g per subject; K1 against refs; deterministic (one judgement) when sigma = 0"""
        sp = sigma * rl['photo_noise_factor']; det = sigma == 0.0; n = 1 if det else n; rec = []
        for b in range(n):
            g = np.random.default_rng([c['seeds']['sim'], case, b])
            Z = [centre_unit(sig + (0 if det else sp * g.standard_normal(sig.shape))) for _ in range(ns)]
            Sd = np.stack([cells(correct(z, pr, st), pr, nconc)[0] for z in Z]); rec.append(k1(Sd, refs, N, resample(ns, concepts, c['n_boot'], np.random.default_rng(g.integers(1 << 62))), c))
        D = np.array([x['D'] for x in rec]); A = np.array([x['A'] for x in rec]); ps = np.array([x['passed'] for x in rec], float)
        return dict(sigma=sigma, photo_sigma=sp, deterministic=bool(det), n=n, pass_rate=float(ps.mean()), D_mean=float(D.mean()), D_p5=float(np.percentile(D, 5)), D_p95=float(np.percentile(D, 95)),
                    A_mean=float(A.mean()), baseline=rec[0]['baseline'], A_ref=rec[0]['A_ref'])
    ref_main = {k: SH[k] for k in ('H_a', 'H_b')}
    # (3) decoder-level semantic signal C(i)
    s3, m0, m1 = calibrate(Cdev, 'C'); res['sim_semantic_C'] = dict(simulate(Cdev, s3, ref_main, 3, c['n_sim']), name_mrp_noiseless=m0, name_mrp_at_sigma=m1)
    res['feasibility']['3_semantic_signal'] = dict(D_mean=res['sim_semantic_C']['D_mean'], min=rl['gate_sim_d_min'], passed=bool(res['sim_semantic_C']['D_mean'] >= rl['gate_sim_d_min']))
    # report: concept prototype (mean C of the concept's other photos, except the sibling of i when i has a pair)
    cr = np.array([st.concept_of_row[int(x)] for x in st.rows]); sib = {int(i): int(s) for i, s in zip(pr['I'], pr['S'])}
    proto = V.unit(np.stack([Cdev[(cr == cr[k]) & (st.rows != st.rows[k]) & (st.rows != sib.get(int(st.rows[k]), -1))].mean(0) for k in range(len(st.rows))]))
    sp_, m0p, m1p = calibrate(proto, 'proto'); res['sim_prototype_report'] = dict(simulate(proto, sp_, ref_main, 5, c['n_sim']), name_mrp_noiseless=m0p, name_mrp_at_sigma=m1p)
    # (4) leave-one-out blur simulations
    res['sim_blur_leave_one_out'] = {}
    for j, (k, others) in enumerate((('H_a', ('H_b', 'H_c')), ('H_b', ('H_a', 'H_c')), ('H_c', ('H_a', 'H_b')))):
        sx, m0b, m1b = calibrate(st.H[k], k); res['sim_blur_leave_one_out'][k] = dict(simulate(st.H[k], sx, {o: SH[o] for o in others}, 10 + j, c['n_sim']), name_mrp_noiseless=m0b, name_mrp_at_sigma=m1b,
                                                                                   note='deterministic simulation, judged once (sigma = 0: the noiseless signal is below the target MRP)' if sx == 0.0 else '')
    res['feasibility']['4_blur_leave_one_out'] = dict(pass_rates={k: v['pass_rate'] for k, v in res['sim_blur_leave_one_out'].items()}, max_rate=rl['stop_rate'],
                                                      deterministic=[k for k, v in res['sim_blur_leave_one_out'].items() if v['deterministic']],
                                                      passed=bool(all(v['pass_rate'] <= rl['stop_rate'] for v in res['sim_blur_leave_one_out'].values())))
    # information zero
    res['sim_information_zero'] = simulate(np.zeros_like(Cdev), 1.0, ref_main, 20, c['n_sim']); res['sim_information_zero']['note'] = 'isotropic Gaussian z (no signal); sigma 1 per photo before the factor'
    res['information_zero_ok'] = bool(res['sim_information_zero']['pass_rate'] <= rl['stop_rate'])
    res['open_eeg'] = bool(all(v['passed'] for v in res['feasibility'].values()) and res['information_zero_ok']); res['seconds'] = time.time() - t0; res['peak_rss_gb'] = peak_gb()
    json.dump(res, open(P(os.path.join(c['out_dir'], 'gate.json')), 'w'), indent=1)
    print(json.dumps(dict(open_eeg=res['open_eeg'], feasibility={k: v['passed'] for k, v in res['feasibility'].items()}, A_H=res['A_H'], semantic_D=res['sim_semantic_C']['D_mean'],
                          blur=res['feasibility']['4_blur_leave_one_out']['pass_rates'], info0=res['sim_information_zero']['pass_rate'])), flush=True)


# ---------------------------------------------------------------- verdict (once)
def stage_verdict(c, tc, meta, a):
    out = P(os.path.join(c['out_dir'], 'verdict.json')); assert not os.path.exists(out), f'{out} exists: the verdict is computed once'
    gate = json.load(open(P(os.path.join(c['out_dir'], 'gate.json')))); assert gate['smoke'] == bool(c['smoke'])
    assert gate['open_eeg'] or (c['smoke'] and c.get('smoke_force_open')), 'the gate keeps the EEG closed'
    st = Setup(c, tc, meta); assert st.files == gate['inputs'], 'pair or held-out files changed since the gate'
    man = json.load(open(P(os.path.join(c['out_dir'], 'infer_manifest.json')))); od = P(c['infer']['out_dir'])
    for k, h in man['files'].items(): assert fsha(os.path.join(od, k)) == h, f'{k}: output sha256 differs from the infer manifest'
    dr = c['dev_runs']; subs = dr['old_targets'] + dr['new_targets']; nconc = len(st.dev_c); pr = st.main; concepts = np.unique(pr['concept']); N = cells(np.zeros(len(pr['I'])), pr, nconc)[1]
    Z = {t: np.load(os.path.join(od, f'z_raw_t{t:02d}.npz')) for t in subs}; assert all(np.array_equal(z['rows'], st.rows) for z in Z.values())
    zc = lambda t, key: centre_unit(Z[t][key]) if '|' in key else None
    dec = lambda t, lay, kind='photo': centre_unit(V.seed_mean([Z[t][f'seed{j}|{lay}|{kind}'] for j in range(len(dr['seeds']))]))
    Sd = lambda f, p=pr: np.stack([cells(correct(f(t), p, st), p, nconc)[0] for t in subs])
    SH = {k: cells(correct(v, pr, st), pr, nconc)[0] for k, v in st.H.items()}; R = resample(len(subs), concepts, c['n_boot'], np.random.default_rng(c['seeds']['boot']))
    S_dec = Sd(lambda t: dec(t, 'ses123')); K = k1(S_dec, {k: SH[k] for k in ('H_a', 'H_b')}, N, R, c)
    Jsh = ctl(Sd(lambda t: zc(t, 'shuffle|ses123|photo')), N, R, K['A'], c); Jno = ctl(Sd(lambda t: dec(t, 'ses123', 'noise')), N, R, K['A'], c)
    adopted = bool(K['passed'] and Jsh['passed'] and Jno['passed'])
    per = {str(t): float(S_dec[i].sum() / N.sum()) for i, t in enumerate(subs)}
    rep = dict(per_subject=per, **{f'{g}_mean': float(np.mean([per[str(t)] for t in dr[f'{g}_targets']])) for g in ('old', 'new') if dr[f'{g}_targets']})
    rep['k0'] = k1(Sd(lambda t: dec(t, 'k0')), {k: SH[k] for k in ('H_a', 'H_b')}, N, R, c)
    m4 = np.array([x is True for x in pr['b4']])
    if m4.any():
        p4 = {k: v[m4] for k, v in pr.items() if k in ('I', 'S', 'F', 'concept')}; N4 = cells(np.zeros(m4.sum()), p4, nconc)[1]
        rep['sigma4_subset'] = dict(n_pairs=int(m4.sum()), A=float(Sd(lambda t: dec(t, 'ses123'), p4).sum() / (len(subs) * N4.sum())))
    else: rep['sigma4_subset'] = dict(n_pairs=0, note='blur4_picks_f not available or never true')
    nf = np.array([st.dev_c.index(pr['raw']['pairs'][q]['foil_concept']) for q in range(len(pr['I']))])
    nm = np.stack([np.mean([(dec(t, 'ses123')[[st.pos[int(i)] for i in pr['I']]] * (st.Tdev[pr['concept']] - st.Tdev[nf])).sum(1) > 0]) for t in subs])
    rep['name_form'] = dict(A=float(nm.mean()), note='centred z_i: cos with the name of the concept of i vs the name of the foil concept (P1)')
    if st.maxm is not None:
        Nm = cells(np.zeros(len(st.maxm['I'])), st.maxm, nconc)[1]; rep['maxmargin_pairs'] = dict(n_pairs=int(len(st.maxm['I'])), A=float(Sd(lambda t: dec(t, 'ses123'), st.maxm).sum() / (len(subs) * Nm.sum())))
    res = dict(stage='verdict', smoke=bool(c['smoke']), subjects=subs, model='development bm, snapshot 11bf018, calibration cap 50, seeds 20261801/02 normalised mean, ses123',
               gate_sha256=fsha(P(os.path.join(c['out_dir'], 'gate.json'))), infer_manifest_sha256=fsha(P(os.path.join(c['out_dir'], 'infer_manifest.json'))), n_boot=c['n_boot'], boot_seed=c['seeds']['boot'],
               K1=K, J=dict(shuffle=Jsh, noise=Jno), adopted=adopted, report=rep, n_pairs=int(len(pr['I'])), n_concepts=int(len(concepts)))
    json.dump(res, open(out, 'w'), indent=1); print(json.dumps(dict(adopted=adopted, D=K['D'], lower=K['lower'], A=K['A'], baseline=K['baseline'], J_shuffle=Jsh['passed'], J_noise=Jno['passed'])), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', choices=['infer', 'gate', 'verdict'], required=True); ap.add_argument('--config-override'); a = ap.parse_args()
    c = load_cfg(a.config, a.config_override); tc = yaml.safe_load(open(P(c['train_config']))); meta = Meta(P(tc['meta']))
    {'infer': stage_infer, 'gate': stage_gate, 'verdict': stage_verdict}[a.stage](c, tc, meta, a)


if __name__ == '__main__':
    main()
