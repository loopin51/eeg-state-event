"""D-120 sections 2-4: ridge reference, gate, inference and the single verdict of the THINGS-EEG1 pair test.

Pairs, space and held-out predictors are D-115's (scripts/fv_pairs.py Setup with configs/fv_pairs.yaml): development photo i,
sibling s, foil f; C = centred original-image CLIP; A = mean 1[cos(z_i, C(s)) > cos(z_i, C(f))] with z centred over the 2,000
development photos and L2-normalised; D = A - max(0.5, A_Ha, A_Hb).
Concept row of photo outputs = unit(mean over the concept's photos of unit(z)); name MRP = rank percentile of the true P1 name.

  ridge  : (server gpuq CPU) time-domain ridge, flattened whitened group means (64 x 250) of the training concepts -> ViT-H-14
           image embedding; penalty from configs/mi_eeg1.yaml ridge.alphas by the 146-way validation concept-row name MRP (ties:
           larger). Weights to cache_dir/ridge.npz, record to out_dir/ridge.json.
  gate   : (server gpuq CPU; EEG of the development concepts not used) m_val = 146-way validation name MRP of the decoder seed
           mean (run folders' valid_out.npz); sigma with z = C(j) + sigma g on the validation photos at m_val; the blur signals
           H_a, H_b, H_c get their own sigma at the 200-way development name MRP of the C signal at that sigma (sigma = 0 when
           the noiseless blur signal is below it: deterministic, judged once). Feasibility (1) r >= 0.6, (2) >= 100 concepts and
           400 photos, (3) C-signal simulation mean D >= +0.02, (4) leave-one-out blur simulations and information zero pass
           rate <= 5%. Simulations draw concepts only. Report: name-form accuracy of H_a, H_b, C. Writes out_dir/gate.json.
  infer  : (k3s rtx5080-16g) one index set (seed boot): per draw b = 1..n_boot the kept subjects then the concepts with pairs,
           both with replacement; draw 0 = all subjects once. Per draw, the development inputs are re-averaged over the drawn
           subjects (presence mask), whitened with W, and passed through the decoder seeds (normalised seed mean), the
           label-shuffle model, the phase-randomised input (fixed phases per photo, seed phase) and the ridge; the Gaussian
           input (channel sd of the draw-0 inputs, seed noise) does not depend on the subject draw and is passed once.
           Stored: per-pair correctness [n_boot + 1, n_pairs] of every output, the draws, draw-0 outputs and the odd / even
           subject halves. No statistic is printed. Writes results/runs/<date>_mi_eeg1_infer/{infer.npz, infer_manifest.json}; after the
           fetch they are copied to cache_dir/infer/ and out_dir/ (sha256 checked by the verdict).
  verdict: (server gpuq CPU, once) M1 with the stored draws; J rule N for label-shuffle, Gaussian and phase-randomised
           outputs; reports r1-r6 (D-120 section 4). Writes out_dir/verdict.json.
usage: python scripts/mi_eeg1_pairs.py configs/mi_eeg1.yaml --stage {ridge,gate,infer,verdict} [--smoke] [--runs-root DIR] [--infer-dir DIR]
       --synthetic (infer): the real models and gate with Gaussian development inputs and 3 draws (code check; the verdict
       of such an output goes to out_dir + '_synthetic' and is never reported)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, time, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.things.data import Meta
from ese.things import verdict as V
from fv_pairs import Setup, centre_unit, correct, cells, fsha, peak_gb
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
KINDS = ('dec', 'shuffle', 'noise', 'phase', 'ridge')


class Ctx:
    def __init__(self, a):
        self.c = c = yaml.safe_load(open(P(a.config))); self.tc = yaml.safe_load(open(P(c['train_config']))); self.pc = yaml.safe_load(open(P(c['pairs_config'])))
        self.smoke = a.smoke; sfx = '_smoke' if a.smoke else ''; self.cache = P(c['cache_dir']) + sfx; self.out = P(c['out_dir']) + sfx; os.makedirs(self.out, exist_ok=True)
        self.meta = Meta(P(self.tc['meta'])); tc = self.tc; m = self.meta
        self.dev_c = m.dev_concepts(tc['dev']['salt'], tc['dev']['n']); self.valid_c = m.valid_concepts(self.dev_c, tc['valid']['salt'], tc['valid']['fraction'])
        self.ic = np.array(m.train_img_concept); self.runs_root = a.runs_root or P(tc['out_root'])
        from fv_train import name_rows
        self.Tall = np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['names_text'])); self.name_rows = lambda cs: name_rows(tc, m, cs)

    def run_dir(self, seed, shuffle):
        pat = f"*_mi_eeg1_bm_s{seed}{'_shuffle' if shuffle else ''}{'_smoke' if self.smoke else ''}"
        d = [x for x in glob.glob(os.path.join(self.runs_root, pat)) + glob.glob(os.path.join(self.runs_root, '*', pat)) if os.path.exists(os.path.join(x, 'shared.pt'))]
        assert len(d) == 1, (pat, d); return d[0]


def rows_mrp(Z, concept_of, concepts, T):
    """Z [n, d] photo outputs, concept_of [n] concept names -> mean name rank percentile of unit(mean unit(z)) concept rows"""
    U = V.unit(np.asarray(Z, np.float64)); R = V.unit(np.stack([U[concept_of == k].mean(0) for k in concepts]))
    return float(V.cells(R @ V.unit(T).T, np.arange(len(concepts))).mean())


# ---------------------------------------------------------------- ridge
def stage_ridge(x, a):
    c = x.c; t0 = time.time(); rows = np.load(os.path.join(x.cache, 'group_trainvalid_rows.npy')); X = np.load(os.path.join(x.cache, 'group_trainvalid.npy'), mmap_mode='r')
    mv = np.isin(x.ic[rows], x.valid_c); Y = V.unit(np.load(os.path.join(P(x.tc['targets']['dir']), x.tc['targets']['train_image'])).astype(np.float64))
    Xt = np.asarray(X[~mv], np.float32).reshape((~mv).sum(), -1); Xv = np.asarray(X[mv], np.float32).reshape(mv.sum(), -1)
    mx = Xt.mean(0); Xt -= mx; Xv -= mx; Yt = Y[rows[~mv]]; my = Yt.mean(0); Yt = Yt - my
    K = (Xt.astype(np.float64) @ Xt.T.astype(np.float64)); lam, U = np.linalg.eigh(K); UY = U.T @ Yt; Kv = Xv.astype(np.float64) @ Xt.T.astype(np.float64); KvU = Kv @ U
    T = x.Tall[x.name_rows(x.valid_c)]; cv = x.ic[rows[mv]]; rec = {}
    for al in c['ridge']['alphas']:
        Zv = KvU @ (UY / (lam + al)[:, None]) + my; rec[str(al)] = rows_mrp(Zv, cv, x.valid_c, T)
    best = max(c['ridge']['alphas'], key=lambda al: (rec[str(al)], al))
    B = Xt.T.astype(np.float64) @ (U @ (UY / (lam + best)[:, None])); np.savez(os.path.join(x.cache, 'ridge.npz'), B=B.astype(np.float32), mx=mx.astype(np.float32), my=my.astype(np.float32), alpha=best)
    res = dict(stage='ridge', smoke=x.smoke, valid_name_mrp=rec, alpha=best, n_train=int((~mv).sum()), n_valid=int(mv.sum()), ridge_sha256=fsha(os.path.join(x.cache, 'ridge.npz')), seconds=time.time() - t0, peak_rss_gb=peak_gb())
    json.dump(res, open(os.path.join(x.out, 'ridge.json'), 'w'), indent=1); print(json.dumps(dict(alpha=best, valid_name_mrp=rec)), flush=True)


# ---------------------------------------------------------------- gate (no development-concept EEG)
def k1_group(Sd, Href, N, Cc, k, rl):
    """one output per photo (group): Sd [nconc] correct sums; Href {name: [nconc]}; Cc [nb, len(k)] concept multiplicities"""
    a = float(Sd.sum() / N.sum()); ah = {h: float(v.sum() / N.sum()) for h, v in Href.items()}; base = max([0.5] + list(ah.values()))
    den = Cc @ N[k]; bb = np.max(np.stack([np.full(len(Cc), 0.5)] + [(Cc @ v[k]) / den for v in Href.values()]), 0); db = (Cc @ Sd[k]) / den - bb
    d = a - base; lo = float(np.percentile(db, rl['lower_pct'])); return dict(A=a, A_ref=ah, baseline=base, D=d, lower=lo, passed=bool(lo > 0 and d >= rl['k1_min']))


def stage_gate(x, a):
    c, pc = x.c, x.pc; rl = pc['rules']; t0 = time.time(); st = Setup(pc, x.tc, x.meta); pr = st.main; nconc = len(st.dev_c); concepts = np.unique(pr['concept'])
    N = cells(np.zeros(len(pr['I'])), pr, nconc)[1]; res = dict(stage='gate', smoke=x.smoke, inputs=st.files)
    # m_val: decoder seed mean on the validation photos
    runs = {s: x.run_dir(s, False) for s in c['seeds']['decoder']}; vo = [np.load(os.path.join(d, 'valid_out.npz')) for d in runs.values()]
    vr = vo[0]['rows']; assert all(np.array_equal(v['rows'], vr) for v in vo); Zv = V.seed_mean([v['z'] for v in vo]); cv = x.ic[vr]; Tv = x.Tall[x.name_rows(x.valid_c)]
    m_val = rows_mrp(Zv, cv, x.valid_c, Tv); res['m_val'] = m_val; res['runs'] = {str(s): dict(dir=os.path.basename(d), shared_sha256=fsha(os.path.join(d, 'shared.pt')), valid_out_sha256=fsha(os.path.join(d, 'valid_out.npz'))) for s, d in runs.items()}
    Cv = st.C[vr]; G = [np.random.default_rng([c['seeds']['sim'], 9, k]) for k in range(c['n_calib'])]; Gv = [g.standard_normal(Cv.shape) for g in G]

    def bisect(f, target):
        lo, hi = 0.0, 0.05
        while f(hi) > target: lo, hi = hi, hi * 2
        for _ in range(c['calib_bisect']):
            mid = (lo + hi) / 2
            if f(mid) > target: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    fv = lambda s: float(np.mean([rows_mrp(Cv + s * g, cv, x.valid_c, Tv) for g in Gv]))
    sigC = 0.0 if fv(0.0) <= m_val else bisect(fv, m_val); res['sigma_C'] = sigC; res['valid_mrp_at_sigma_C'] = fv(sigC)
    Cdev = st.C[st.rows]; cr = np.array([st.dev_c[st.concept_of_row[int(r)]] for r in st.rows]); Td = st.Tdev
    Gd = [np.random.default_rng([c['seeds']['sim'], 8, k]).standard_normal(Cdev.shape) for k in range(c['n_calib'])]
    fd = lambda sig, s: float(np.mean([rows_mrp(sig + s * g, cr, st.dev_c, Td) for g in Gd]))
    target_dev = fd(Cdev, sigC); res['dev_name_mrp_of_C_at_sigma_C'] = target_dev
    r = float(correct(V.unit(Cdev), pr, st).mean()); SH = {k: cells(correct(v, pr, st), pr, nconc)[0] for k, v in st.H.items()}
    res['feasibility'] = {'1_r': dict(value=r, min=rl['gate_r_min'], passed=bool(r >= rl['gate_r_min'])),
                          '2_size': dict(n_photos=int(len(pr['I'])), n_concepts=int(len(concepts)), passed=bool(len(concepts) >= rl['gate_min_concepts'] and len(pr['I']) >= rl['gate_min_photos']))}
    res['A_H'] = {k: float(v.sum() / N.sum()) for k, v in SH.items()}

    def simulate(sig, sigma, refs, case):
        det = sigma == 0.0; n = 1 if det else c['n_sim']; rec = []
        for b in range(n):
            g = np.random.default_rng([c['seeds']['sim'], case, b]); z = centre_unit(sig + (0 if det else sigma * g.standard_normal(sig.shape)))
            Sd = cells(correct(z, pr, st), pr, nconc)[0]; rr = np.random.default_rng(g.integers(1 << 62)); nb, nk = c['n_boot'], len(concepts)
            Cc = np.zeros((nb, nk)); np.add.at(Cc, (np.repeat(np.arange(nb), nk), rr.integers(0, nk, (nb, nk)).ravel()), 1.0)
            rec.append(k1_group(Sd, refs, N, Cc, concepts, rl))
        D = np.array([q['D'] for q in rec]); ps = np.array([q['passed'] for q in rec], float)
        return dict(sigma=sigma, deterministic=bool(det), n=n, pass_rate=float(ps.mean()), D_mean=float(D.mean()), D_p5=float(np.percentile(D, 5)), D_p95=float(np.percentile(D, 95)),
                    A_mean=float(np.mean([q['A'] for q in rec])), baseline=rec[0]['baseline'], A_ref=rec[0]['A_ref'])
    ref_main = {k: SH[k] for k in ('H_a', 'H_b')}
    res['sim_semantic_C'] = simulate(Cdev, sigC, ref_main, 3)
    res['feasibility']['3_semantic_signal'] = dict(D_mean=res['sim_semantic_C']['D_mean'], min=rl['gate_sim_d_min'], passed=bool(res['sim_semantic_C']['D_mean'] >= rl['gate_sim_d_min']))
    res['sim_blur_leave_one_out'] = {}
    for j, (k, others) in enumerate((('H_a', ('H_b', 'H_c')), ('H_b', ('H_a', 'H_c')), ('H_c', ('H_a', 'H_b')))):
        sig = st.H[k]; m0 = fd(sig, 0.0); sx = 0.0 if m0 <= target_dev else bisect(lambda s: fd(sig, s), target_dev)
        res['sim_blur_leave_one_out'][k] = dict(simulate(sig, sx, {o: SH[o] for o in others}, 10 + j), dev_name_mrp_noiseless=m0, dev_name_mrp_at_sigma=fd(sig, sx),
                                                note='deterministic simulation, judged once (sigma = 0: the noiseless signal is below the target MRP)' if sx == 0.0 else '')
    res['feasibility']['4_blur_leave_one_out'] = dict(pass_rates={k: v['pass_rate'] for k, v in res['sim_blur_leave_one_out'].items()}, max_rate=rl['stop_rate'],
                                                      deterministic=[k for k, v in res['sim_blur_leave_one_out'].items() if v['deterministic']],
                                                      passed=bool(all(v['pass_rate'] <= rl['stop_rate'] for v in res['sim_blur_leave_one_out'].values())))
    res['sim_information_zero'] = simulate(np.zeros_like(Cdev), 1.0, ref_main, 20); res['information_zero_ok'] = bool(res['sim_information_zero']['pass_rate'] <= rl['stop_rate'])
    nf = np.array([st.dev_c.index(pr['raw']['pairs'][q]['foil_concept']) for q in range(len(pr['I']))]); ii = [st.pos[int(i)] for i in pr['I']]
    nform = lambda Z: float(np.mean((centre_unit(Z)[ii] * (Td[pr['concept']] - Td[nf])).sum(1) > 0))
    res['name_form_reference'] = dict(H_a=nform(st.H['H_a']), H_b=nform(st.H['H_b']), H_c=nform(st.H['H_c']), C=nform(Cdev), note='EEG-free: cos with the name of the concept of i vs the name of the foil concept (P1), rows centred over the 2,000 development photos as the decoder output')
    res['open_eeg'] = bool(all(v['passed'] for v in res['feasibility'].values()) and res['information_zero_ok']); res['seconds'] = time.time() - t0; res['peak_rss_gb'] = peak_gb()
    json.dump(res, open(os.path.join(x.out, 'gate.json'), 'w'), indent=1)
    print(json.dumps(dict(open_eeg=res['open_eeg'], m_val=m_val, sigma_C=sigC, feasibility={k: v['passed'] for k, v in res['feasibility'].items()}, semantic_D=res['sim_semantic_C']['D_mean'],
                          blur=res['feasibility']['4_blur_leave_one_out']['pass_rates'], info0=res['sim_information_zero']['pass_rate'], name_form_reference=res['name_form_reference'])), flush=True)


# ---------------------------------------------------------------- infer (k3s GPU)
def stage_infer(x, a):
    import torch
    from ese.things.models import build
    from mi_eeg1_train import model_cfg
    c, pc = x.c, x.pc; t0 = time.time(); dv = 'cuda'; gate = json.load(open(os.path.join(x.out, 'gate.json'))); assert gate['open_eeg'] or x.smoke, 'the gate keeps the EEG closed'
    st = Setup(pc, x.tc, x.meta); assert st.files == gate['inputs']; pr = st.main; nconc = len(st.dev_c); concepts = np.unique(pr['concept']); dd = os.path.join(x.cache, 'dev')
    rows = np.load(os.path.join(dd, 'rows.npy')); assert np.array_equal(rows, st.rows); kept = np.load(os.path.join(dd, 'kept.npy')); subs = np.load(os.path.join(dd, 'subjects.npy'))[kept]
    if x.smoke or a.synthetic: c['n_boot'] = 3
    if x.smoke or a.synthetic:                                                 # code check: synthetic inputs, the development EEG is never read
        S_ = int(kept.sum()); SD = torch.from_numpy(np.random.default_rng(1).standard_normal((S_, len(rows), 64, c['epoch']['n_times'])).astype(np.float16)).to(dv); M = torch.ones((S_, len(rows)), device=dv)
    else: SD = torch.from_numpy(np.load(os.path.join(dd, 'subjects_dev.npy'), mmap_mode='r')[kept]).to(dv); M = torch.from_numpy(np.load(os.path.join(dd, 'mask.npy'))[kept]).to(dv).float()
    W = torch.from_numpy(np.load(os.path.join(x.cache, 'W.npy'))).to(dv); pos = torch.from_numpy(np.load(os.path.join(x.cache, 'positions.npy'))).float().to(dv)
    rz = np.load(os.path.join(x.cache, 'ridge.npz')); RB = torch.from_numpy(rz['B']).to(dv); Rmx = torch.from_numpy(rz['mx']).to(dv); Rmy = torch.from_numpy(rz['my']).to(dv)
    mcfg = model_cfg(c, x.tc); nets, man = {}, dict(stage='infer', smoke=x.smoke, runs={}, inputs={})
    for tag, (s, sh) in {**{f'seed{j}': (s, False) for j, s in enumerate(c['seeds']['decoder'])}, 'shuffle': (c['seeds']['shuffle_train'], True)}.items():
        d = x.run_dir(s, sh); n = build('bm', 1, mcfg).to(dv); n.load_state_dict(torch.load(os.path.join(d, 'shared.pt'), map_location=dv)['net']); n.eval(); nets[tag] = n
        man['runs'][tag] = dict(dir=os.path.basename(d), shared_sha256=fsha(os.path.join(d, 'shared.pt')))
    assert man['runs']['seed0']['shared_sha256'] == gate['runs'][str(c['seeds']['decoder'][0])]['shared_sha256'], 'decoder changed since the gate'
    S, n_ph = SD.shape[0], SD.shape[1]; B = c['n_boot']; rng = np.random.default_rng(c['seeds']['boot'])
    draws_s = np.zeros((B + 1, S), np.int64); draws_c = np.zeros((B + 1, len(concepts)), np.int64); draws_s[0] = 1; draws_c[0] = 1
    for b in range(1, B + 1): draws_s[b] = np.bincount(rng.integers(0, S, S), minlength=S); draws_c[b] = np.bincount(rng.integers(0, len(concepts), len(concepts)), minlength=len(concepts))
    T_ = SD.shape[-1]; ph = torch.from_numpy(np.exp(1j * np.random.default_rng(c['seeds']['phase']).uniform(0, 2 * np.pi, (n_ph, T_ // 2 + 1)))).to(dv).to(torch.complex64)
    ph[:, 0] = 1
    if T_ % 2 == 0: ph[:, -1] = 1

    def average(w):
        wm = M * torch.as_tensor(w, dtype=torch.float32, device=dv)[:, None]; den = wm.sum(0); assert bool((den > 0).all()), 'a photo without any drawn subject'
        X = torch.zeros(SD.shape[1:], dtype=torch.float32, device=dv)
        for s_ in torch.nonzero(wm.sum(1) > 0).flatten().tolist(): X += wm[s_][:, None, None] * SD[s_].float()
        return torch.einsum('ij,njt->nit', W, X / den[:, None, None])

    def fwd(net, X):
        with torch.no_grad():
            return torch.cat([net(X[b:b + c['infer']['batch']], pos[None].expand(len(X[b:b + c['infer']['batch']]), -1, -1), torch.zeros(len(X[b:b + c['infer']['batch']]), dtype=torch.long, device=dv)).float()
                              for b in range(0, len(X), c['infer']['batch'])]).cpu().numpy()
    dec = lambda X: V.seed_mean([fwd(nets[f'seed{j}'], X) for j in range(len(c['seeds']['decoder']))])
    phase = lambda X: torch.fft.irfft(torch.fft.rfft(X, dim=-1) * ph[:, None, :], n=T_, dim=-1)
    ridge = lambda X: ((X.reshape(len(X), -1) - Rmx) @ RB + Rmy).cpu().numpy()
    corr = lambda Z: correct(centre_unit(Z), pr, st).astype(bool)
    X0 = average(draws_s[0]); sd = X0.std(dim=(0, 2)).cpu().numpy()
    Xn = torch.from_numpy((np.random.default_rng(c['seeds']['noise']).standard_normal(tuple(X0.shape)) * sd[None, :, None]).astype(np.float32)).to(dv)
    Zn = dec(Xn); cn = corr(Zn); out = {k: np.zeros((B + 1, len(pr['I'])), bool) for k in KINDS}; z0 = {}
    for b in range(B + 1):
        X = X0 if b == 0 else average(draws_s[b]); Zd, Zs, Zp, Zr = dec(X), fwd(nets['shuffle'], X), dec(phase(X)), ridge(X)
        out['dec'][b], out['shuffle'][b], out['phase'][b], out['ridge'][b], out['noise'][b] = corr(Zd), corr(Zs), corr(Zp), corr(Zr), cn
        if b == 0: z0 = dict(z0_dec=Zd.astype(np.float32), z0_shuffle=Zs.astype(np.float32), z0_phase=Zp.astype(np.float32), z0_ridge=Zr.astype(np.float32), z0_noise=Zn.astype(np.float32))
        if b % 100 == 0: print(f'draw {b}/{B} ({time.time() - t0:.0f}s)', flush=True)
    half = {}
    for h, sel in (('odd', subs % 2 == 1), ('even', subs % 2 == 0)):
        Xh = average(sel.astype(np.int64)); half[f'zh_{h}_dec'] = dec(Xh).astype(np.float32); half[f'zh_{h}_ridge'] = ridge(Xh).astype(np.float32); man['inputs'][f'n_{h}'] = int(sel.sum())
    od = os.path.join(P(x.tc['out_root']), time.strftime('%Y-%m-%d') + '_mi_eeg1_infer' + ('_smoke' if x.smoke else '') + ('_synthetic' if a.synthetic else '')); os.makedirs(od, exist_ok=True); f = os.path.join(od, 'infer.npz')   # results/runs = RUN_DIR on k3s
    np.savez_compressed(f, rows=rows, draws_s=draws_s, draws_c=draws_c, concepts=concepts, subjects=subs, **{f'corr_{k}': v for k, v in out.items()}, **z0, **half)
    man.update(file=os.path.basename(f), sha256=fsha(f), n_subjects=int(S), n_boot=B, device=torch.cuda.get_device_name(0), seconds=time.time() - t0, gate_sha256=fsha(os.path.join(x.out, 'gate.json')),
               ridge_sha256=fsha(os.path.join(x.cache, 'ridge.npz')), synthetic=bool(a.synthetic), inputs_sha256=None if (x.smoke or a.synthetic) else {k: fsha(os.path.join(dd, k)) for k in ('subjects_dev.npy', 'mask.npy', 'kept.npy')})
    json.dump(man, open(os.path.join(od, 'infer_manifest.json'), 'w'), indent=1); print(f'infer done ({time.time() - t0:.0f}s)', flush=True)   # copied to out_dir / cache_dir/infer after the fetch


# ---------------------------------------------------------------- verdict (once)
def stage_verdict(x, a):
    c, pc = x.c, x.pc; rl = pc['rules']; idir = a.infer_dir or os.path.join(x.cache, 'infer'); man = json.load(open(os.path.join(idir, 'infer_manifest.json')))
    od = x.out + ('_synthetic' if man.get('synthetic') else ''); os.makedirs(od, exist_ok=True); out = os.path.join(od, 'verdict.json'); assert not os.path.exists(out), 'the verdict is computed once'
    if x.smoke or man.get('synthetic'): c['n_boot'] = 3
    gate = json.load(open(os.path.join(x.out, 'gate.json'))); assert gate['open_eeg'] or x.smoke
    f = os.path.join(idir, man['file']); assert fsha(f) == man['sha256'], 'infer output changed'; assert man['gate_sha256'] == fsha(os.path.join(x.out, 'gate.json'))
    json.dump(man, open(os.path.join(od, 'infer_manifest.json'), 'w'), indent=1)
    st = Setup(pc, x.tc, x.meta); assert st.files == gate['inputs']; pr = st.main; nconc = len(st.dev_c); Z = np.load(f); k = Z['concepts']; assert np.array_equal(k, np.unique(pr['concept']))
    rng = np.random.default_rng(c['seeds']['boot']); S = len(Z['subjects'])
    for b in range(1, c['n_boot'] + 1):                                        # the draws are the registered stream
        ds = np.bincount(rng.integers(0, S, S), minlength=S); dc = np.bincount(rng.integers(0, len(k), len(k)), minlength=len(k)); assert np.array_equal(ds, Z['draws_s'][b]) and np.array_equal(dc, Z['draws_c'][b])
    N = cells(np.zeros(len(pr['I'])), pr, nconc)[1]; Cc = Z['draws_c'][1:].astype(float); den = Cc @ N[k]
    SH = {h: cells(correct(v, pr, st), pr, nconc)[0] for h, v in st.H.items()}
    Sk = lambda kind: np.stack([cells(Z[f'corr_{kind}'][b].astype(float), pr, nconc)[0] for b in range(len(Z[f'corr_{kind}']))])   # [B + 1, nconc]
    Aboot = lambda Sm: np.einsum('bc,bc->b', Cc, Sm[1:][:, k]) / den
    A = {kind: float(Z[f'corr_{kind}'][0].mean()) for kind in KINDS}; SS = {kind: Sk(kind) for kind in KINDS}
    ah = {h: float(SH[h].sum() / N.sum()) for h in ('H_a', 'H_b')}; base = max([0.5] + list(ah.values()))
    bb = np.max(np.stack([np.full(len(Cc), 0.5)] + [(Cc @ SH[h][k]) / den for h in ('H_a', 'H_b')]), 0)
    db = Aboot(SS['dec']) - bb; D = A['dec'] - base; lo = float(np.percentile(db, rl['lower_pct']))
    M1 = dict(A=A['dec'], A_ref=ah, baseline=base, D=D, lower=lo, passed=bool(lo > 0 and D >= rl['k1_min']))
    J = {kind: dict(A=A[kind], **V.control_rule(np.array([A[kind] - 0.5]), Aboot(SS[kind]) - 0.5, A['dec'] - 0.5, rl['control_ratio'], rl['control_lower_pct'])) for kind in ('shuffle', 'noise', 'phase')}
    adopted = bool(M1['passed'] and all(v['passed'] for v in J.values()))
    rep = dict(intervals95={kind: [float(np.percentile(Aboot(SS[kind]), 2.5)), float(np.percentile(Aboot(SS[kind]), 97.5))] for kind in KINDS},
               intervals95_H={h: [float(np.percentile((Cc @ SH[h][k]) / den, 2.5)), float(np.percentile((Cc @ SH[h][k]) / den, 97.5))] for h in ('H_a', 'H_b')},
               D_interval90=[float(np.percentile(db, 5)), float(np.percentile(db, 95))])
    dr = Aboot(SS['dec']) - Aboot(SS['ridge'])
    rep['r1_ridge'] = dict(A=A['ridge'], D=A['ridge'] - base, dec_minus_ridge_A=A['dec'] - A['ridge'], lo95=float(np.percentile(dr, 2.5)), hi95=float(np.percentile(dr, 97.5)))
    nf = np.array([st.dev_c.index(pr['raw']['pairs'][q]['foil_concept']) for q in range(len(pr['I']))]); ii = [st.pos[int(i)] for i in pr['I']]
    nform = lambda Zr: float(np.mean((centre_unit(Zr)[ii] * (st.Tdev[pr['concept']] - st.Tdev[nf])).sum(1) > 0))
    rep['r2_name_form'] = dict(decoder=nform(Z['z0_dec']), ridge=nform(Z['z0_ridge']), shuffle=nform(Z['z0_shuffle']), reference_EEG_free=gate['name_form_reference'])
    cr = np.array([st.dev_c[st.concept_of_row[int(r)]] for r in st.rows])
    rep['r3_dev_name_mrp'] = {kind: rows_mrp(Z[f'z0_{kind}'], cr, st.dev_c, st.Tdev) for kind in ('dec', 'ridge', 'shuffle', 'noise', 'phase')}; rep['r3_dev_name_mrp']['EEG2_dev_decoder_reference'] = 0.808
    rep['r4_halves'] = {h: dict(A_dec=float(correct(centre_unit(Z[f'zh_{h}_dec']), pr, st).mean()), A_ridge=float(correct(centre_unit(Z[f'zh_{h}_ridge']), pr, st).mean()), n=man['inputs'][f'n_{h}']) for h in ('odd', 'even')}
    m4 = np.array([q is True for q in pr['b4']])
    if m4.any(): rep['r5_sigma4_subset'] = dict(n_pairs=int(m4.sum()), A=float(Z['corr_dec'][0][m4].mean()))
    if st.maxm is not None: rep['r5_maxmargin'] = dict(n_pairs=int(len(st.maxm['I'])), A=float(correct(centre_unit(Z['z0_dec']), st.maxm, st).mean()))
    rep['r6_EEG2_D115'] = dict(A=0.46977452772699574, D=-0.03022547227300426, name_form=0.6461913467397927, note='D-115 verdict (EEG2 development decoders, 10 subjects); descriptive comparison only')
    res = dict(stage='verdict', smoke=x.smoke, n_subjects=S, n_pairs=int(len(pr['I'])), n_concepts=int(len(k)), n_boot=c['n_boot'], boot_seed=c['seeds']['boot'], gate_sha256=fsha(os.path.join(x.out, 'gate.json')),
               infer_manifest_sha256=fsha(os.path.join(od, 'infer_manifest.json')), synthetic=bool(man.get('synthetic')), M1=M1, J=J, adopted=adopted, report=rep)
    json.dump(res, open(out, 'w'), indent=1); print(json.dumps(dict(adopted=adopted, A=M1['A'], D=D, lower=lo, J={kk: v['passed'] for kk, v in J.items()}, ridge=rep['r1_ridge'], dev_mrp=rep['r3_dev_name_mrp'])), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', choices=['ridge', 'gate', 'infer', 'verdict'], required=True); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--runs-root'); ap.add_argument('--synthetic', action='store_true'); ap.add_argument('--infer-dir'); a = ap.parse_args(); x = Ctx(a)
    {'ridge': stage_ridge, 'gate': stage_gate, 'infer': stage_infer, 'verdict': stage_verdict}[a.stage](x, a)


if __name__ == '__main__':
    main()
