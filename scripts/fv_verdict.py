"""F-V verdict job (D-112 section 6; amendment 3: three stages counted as the one verdict job; amendment 4: statistics).

  --stage mde : information-zero simulation (no EEG): decoder and label-shuffle predictions drawn as isotropic Gaussians in the
                1024-d target space, scored against the 200 test names; MDE = (z_alpha + z_power) x mean pigeonhole SD of the
                contrast (H1/H3: one 20-repetition input per cell; H4: mean over the trials of a cell). Commit before V1.
  --stage v1  : server CPU (gpuq). Needs --unseal. Every sealed test file is opened through rawio (sha256 = seal table),
                preprocessed with ese.things.preprocess.test_epochs (that session's TRAINING-run W), and turned into the input
                arrays of each subject: avg20 (mean of every session-4 repetition), r-curve means (first r repetitions of
                session 4), avg80 (every repetition of the four sessions), phase (phase-randomised avg20: one random phase per
                frequency, the same for all channels), noise (Gaussian, avg20 channel variances), single (session-4 trials).
                Written to inputs_dir (never committed) with sha256 in out_dir/v1_manifest.json. No decoder score is computed.
                The same job fits the within-subject ridges (DE, time, eye, pre; sessions 1-3, all training concepts, as
                scripts/fv_ridge.py) and writes their test score matrices (out_dir/ridge_test.npz) before any decoder score.
  --stage v2  : k3s RTX 5080. Inference only: checks the input sha256 against the V1 manifest, loads the frozen final runs
                (seeds x targets and the label-shuffle runs) and saves predicted embeddings (results/runs/<date>_fv_verdict_v2/
                z_emb_tXX.npz + v2_manifest.json). No cosine, MRP or top-k is computed or printed.
  --stage v3  : server CPU. Checks both manifests, then computes H1 -> H3 -> H4 (fixed sequence, H2 not tested: amendment 1),
                the premises 1-5 and the report-only items in one run: out_dir/verdict.json.
Cells: subject x test concept, rank percentile of the true name among the 200 test names (cosine of L2-normalised vectors).
Decoder embeddings are seed means (mean of L2-normalised seed outputs).

usage: python scripts/fv_verdict.py configs/fv_verdict.yaml --stage {mde,v1,v2,v3} [--unseal] [--subjects S ...]
                                    [--v2-dir DIR] [--data-root DIR] [--config-override YAML]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, csv, glob, json, time, hashlib, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.things.data import Meta
from ese.things import verdict as V

P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
AVG_KINDS = ['avg20', 'avg80', 'phase', 'noise']


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


def names(tc, meta):
    """test-name rows in test-code order (code k -> test_img_concepts[k-1]), all 1,854 rows, training flags, categories"""
    rows = list(csv.DictReader(open(P(tc['targets']['names_table'])), delimiter='\t'))
    test = {r['folder']: i for i, r in enumerate(rows) if r['split'] == 'test'}
    trow = np.array([test[f] for f in meta.test_img_concept]); assert len(set(trow)) == 200
    return trow, np.array([r['split'].startswith('train') for r in rows]), np.array([r['category'] for r in rows])


def kinds(c): return AVG_KINDS + [f'r{r}' for r in c['r_curve']]


def peak_gb():
    import resource; return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # Linux: kB


# ---------------------------------------------------------------- MDE (no EEG)
def stage_mde(c, tc, meta):
    trow, _, _ = names(tc, meta); Tn = V.unit(np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['names_text']))[trow])
    rng = np.random.default_rng(c['seeds']['mde']); ns, nc = len(c['subjects']), 200; true = np.arange(nc); z = sum(c['mde_power_z']); nt = max(c['r_curve'])
    sd = {'H1': [], 'H4': []}; t0 = time.time()
    for _ in range(c['n_mde']):
        M = {}
        for arm in ('dec', 'shuf'):
            M[arm] = np.stack([V.cells(V.unit(rng.standard_normal((nc, Tn.shape[1]))) @ Tn.T, true) for _ in range(ns)])
            M[arm + '1'] = np.stack([np.mean([V.cells(V.unit(rng.standard_normal((nc, Tn.shape[1]))) @ Tn.T, true) for _ in range(nt)], 0) for _ in range(ns)])
        br = np.random.default_rng(rng.integers(1 << 62))
        sd['H1'].append(V.pigeonhole(M['dec'] - M['shuf'], c['n_boot'], br).std()); sd['H4'].append(V.pigeonhole(M['dec1'] - M['shuf1'], c['n_boot'], br).std())
    res = dict(n_sim=c['n_mde'], z=c['mde_power_z'], trials_per_cell_assumed=nt, seconds=time.time() - t0,
               **{h: dict(sd_mean=float(np.mean(v)), sd_p95=float(np.percentile(v, 95)), mde=float(z * np.mean(v))) for h, v in sd.items()})
    res['H3'] = dict(res['H1'], note='same cell structure as H1'); return res


# ---------------------------------------------------------------- V1
def stage_v1(c, tc, meta, a):
    assert a.unseal, 'V1 opens the sealed test files: pass --unseal (D-112 amendment 3)'
    from ese.things import preprocess as PP
    from fv_ridge import image_means, features, ridge_cv, BANDS
    pc = yaml.safe_load(open(P(tc['prep_config']))); cache = P(pc['cache_dir']); root = P(a.data_root or pc['data_root'])
    inp = P(c['inputs_dir']); os.makedirs(inp, exist_ok=True); out = P(c['out_dir']); os.makedirs(out, exist_ok=True)
    trow, _, _ = names(tc, meta); Tn = np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['names_text']))[trow]
    Timg = np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['train_image'])); ch = json.load(open(os.path.join(cache, 'channels.json')))['ch_names']
    seal = P(a.seal_table or c['seal_table']); sealed = {k: v for k, v in __import__('ese.things.rawio', fromlist=['read_seal']).read_seal(seal).items()}
    fold = {x: i % 5 for i, x in enumerate(sorted(meta.train_concepts, key=lambda x: hashlib.sha256(f'{x}|{tc["ridge"]["salt"]}'.encode()).hexdigest()))}
    man = dict(stage='v1', seal_table=dict(file=os.path.relpath(seal, ROOT), sha256=fsha(seal)), subjects={}, inputs={}); ridge, ralpha = {}, {}; t0 = time.time()
    for s in a.subjects or c['subjects']:
        t1 = time.time(); per = {}; counts = {}
        for e in tc['sessions']:
            path = os.path.join(root, f'sub-{s:02d}', f'ses-{e:02d}', c['test_name'])
            E = PP.test_epochs(path, pc, os.path.join(cache, f'sub-{s:02d}', f'ses-{e:02d}_mvnn.npz'), seal, unseal=True)
            order = np.argsort(E['sample'], kind='stable'); per[e] = (E['X'][order], E['code'][order], E['times']); counts[e] = np.bincount(E['code'], minlength=201)[1:].tolist()
            assert set(np.unique(E['code'])) == set(range(1, 201)), f'sub-{s} ses-{e}: test codes {np.unique(E["code"])[:5]}...'
        X4, c4, times = per[c['eval_session']]; post = times >= tc['ridge']['input_tmin'] - 1e-9
        idx = {k: np.where(c4 == k)[0] for k in range(1, 201)}
        full20 = np.stack([X4[idx[k]].mean(0) for k in range(1, 201)]); arr = dict(avg20=full20[..., post])
        for r in c['r_curve']: arr[f'r{r}'] = np.stack([X4[idx[k][:r]][..., post].mean(0) for k in range(1, 201)])
        S80 = np.zeros_like(arr['avg20'], np.float64); n80 = np.zeros(200)
        for e in per:
            for k in range(1, 201): m = per[e][1] == k; S80[k - 1] += per[e][0][m][..., post].sum(0); n80[k - 1] += m.sum()
        arr['avg80'] = S80 / n80[:, None, None]; del S80
        F = np.fft.rfft(arr['avg20'], axis=-1); ph = np.stack([np.random.default_rng([c['seeds']['phase'], s, k]).uniform(0, 2 * np.pi, F.shape[-1]) for k in range(1, 201)])
        ph[:, 0] = 0; ph[:, -1] = 0 if arr['avg20'].shape[-1] % 2 == 0 else ph[:, -1]
        arr['phase'] = np.fft.irfft(F * np.exp(1j * ph)[:, None, :], n=arr['avg20'].shape[-1], axis=-1)
        sd = arr['avg20'].std(axis=(0, 2)); arr['noise'] = np.random.default_rng([c['seeds']['noise'], s]).standard_normal(arr['avg20'].shape) * sd[None, :, None]
        arr = {k: v.astype(np.float32) for k, v in arr.items()}; arr['single'] = X4[..., post].astype(np.float16); arr['single_code'] = c4
        del per, X4, F
        f = os.path.join(inp, f't{s:02d}.npz'); np.savez(f, **arr); man['inputs'][os.path.basename(f)] = fsha(f); del arr
        man['subjects'][str(s)] = dict(reps_per_code={str(e): dict(min=int(min(v)), max=int(max(v)), total=int(sum(v))) for e, v in counts.items()},
                                       sealed_sha256={str(e): sealed[os.path.join(f'sub-{s:02d}', f'ses-{e:02d}', c['test_name'])] for e in counts}, seconds_inputs=time.time() - t1)
        # within-subject ridges (sessions 1-3, all training concepts) -> test score matrices, before any decoder score;
        # one image-mean array per window (post: de, time, eye; pre), freed before the next window
        all_codes = np.arange(1, len(meta.img_concept) + 1)
        for win, kinds_w in (('post', ('de', 'time', 'eye')), ('pre', ('pre',))):
            tsel = np.where(times < 0)[0] if win == 'pre' else np.where(post)[0]
            fc, FX, _ = image_means(cache, s, [1, 2, 3], all_codes, tsel); grp = np.array([fold[meta.train_concepts[meta.img_concept[k - 1]]] for k in fc])
            for kind in kinds_w:
                pred, alpha, _ = ridge_cv(features(kind, FX, ch, pc['epoch']['sfreq']), Timg[fc - 1], grp)
                ridge[f'{kind}_t{s:02d}'] = (V.unit(pred(features(kind, full20[..., tsel], ch, pc['epoch']['sfreq']))) @ V.unit(Tn).T).astype(np.float32); ralpha[f'{kind}_t{s:02d}'] = alpha; del pred
            del FX
        man['subjects'][str(s)]['peak_rss_gb'] = peak_gb()
        man['subjects'][str(s)]['seconds'] = time.time() - t1; print(f'sub-{s:02d}: inputs and ridge done ({time.time() - t1:.0f}s)', flush=True)
    rf = os.path.join(out, 'ridge_test.npz'); np.savez_compressed(rf, **ridge)
    man.update(ridge_test=dict(file=os.path.relpath(rf, ROOT), sha256=fsha(rf), alpha=ralpha), seconds=time.time() - t0, peak_rss_gb=peak_gb())
    json.dump(man, open(os.path.join(out, 'v1_manifest.json'), 'w'), indent=1); print('V1 done', len(man['inputs']), 'input files', flush=True)


# ---------------------------------------------------------------- V2
def run_dirs(c, tt):
    f = c['final']; g = lambda seed, suf: sorted(glob.glob(P(f['runs_glob'].format(cand=f['candidate'], seed=seed, tt=f'{tt:02d}', suffix=suf))))
    dec = [g(s, '') for s in f['seeds']]; sh = g(f['shuffle_seed'], f['shuffle_suffix'])
    assert all(len(d) == 1 for d in dec) and len(sh) == 1, f't{tt}: final runs {dec} shuffle {sh}'; return [d[0] for d in dec], sh[0]


def stage_v2(c, tc, a):
    import torch
    from ese.things.models import build
    assert c['final']['candidate'] in ('bm', 'nice'), 'final.candidate is set by the freeze commit'
    v1 = json.load(open(os.path.join(P(c['out_dir']), 'v1_manifest.json'))); inp = P(c['inputs_dir'])
    for k, h in v1['inputs'].items(): assert fsha(os.path.join(inp, k)) == h, f'{k}: input sha256 differs from V1'
    out = P(a.v2_dir or os.path.join(tc['out_root'], time.strftime('%Y-%m-%d') + '_fv_verdict_v2')); os.makedirs(out, exist_ok=True)
    pos = torch.as_tensor(np.load(P(tc['positions'])), dtype=torch.float32, device='cuda'); man = dict(stage='v2', inputs=v1['inputs'], runs={}, files={})
    for s in a.subjects or c['subjects']:
        Z = np.load(os.path.join(inp, f't{s:02d}.npz')); dec, sh = run_dirs(c, s); res = {}
        for tag, d in [(f'seed{i}', x) for i, x in enumerate(dec)] + [('shuffle', sh)]:
            rc = yaml.safe_load(open(os.path.join(d, 'config.yaml'))); r = rc['run']; assert r['target'] == s and r['mode'] == c['final']['mode'] and r['candidate'] == c['final']['candidate']
            ti = len(r['trainers']); net = build(r['candidate'], ti + 1, rc['model']).cuda(); net.load_state_dict(torch.load(os.path.join(d, 'shared.pt'))['net']); net.eval()
            layers = torch.load(os.path.join(d, 'layers.pt')); man['runs'][f't{s:02d}_{tag}'] = dict(dir=os.path.basename(d), shared_sha256=fsha(os.path.join(d, 'shared.pt')), layers_sha256=fsha(os.path.join(d, 'layers.pt')))
            for cond in c['conditions']:
                with torch.no_grad():
                    net.subject.weights[ti] = layers[cond].cuda()
                    for kind in kinds(c) + (['single'] if cond in c['single_conditions'] else []):
                        X = torch.from_numpy(np.asarray(Z[kind], np.float32)); outs = []
                        for b in range(0, len(X), 256):
                            xb = X[b:b + 256].cuda(); outs.append(net(xb, pos[None].expand(len(xb), -1, -1), torch.full((len(xb),), ti, device='cuda')).float().cpu().numpy())
                        res[f'{tag}|{cond}|{kind}'] = np.concatenate(outs).astype(np.float16 if kind == 'single' else np.float32)
        f = os.path.join(out, f'z_emb_t{s:02d}.npz'); np.savez(f, single_code=Z['single_code'], **res); man['files'][os.path.basename(f)] = fsha(f); print(f'sub-{s:02d}: embeddings saved', flush=True)
    man['device'] = torch.cuda.get_device_name(0); man['peak_rss_gb'] = peak_gb(); man['max_gpu_gb'] = torch.cuda.max_memory_allocated() / 1e9; json.dump(man, open(os.path.join(out, 'v2_manifest.json'), 'w'), indent=1); print('V2 done', out, flush=True)


# ---------------------------------------------------------------- V3
def stage_v3(c, tc, meta, a):
    out = P(c['out_dir']); v1 = json.load(open(os.path.join(out, 'v1_manifest.json'))); d2 = P(a.v2_dir); v2 = json.load(open(os.path.join(d2, 'v2_manifest.json')))
    assert v2['inputs'] == v1['inputs'], 'V2 inputs differ from V1'; assert fsha(P(v1['ridge_test']['file'])) == v1['ridge_test']['sha256'], 'ridge test file changed'
    for k, h in v2['files'].items(): assert fsha(os.path.join(d2, k)) == h, f'{k}: embedding sha256 differs from V2'
    trow, is_train, cat = names(tc, meta); Tall = V.unit(np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['names_text']))); Tn = Tall[trow]
    subs = a.subjects or c['subjects']; true = np.arange(200); R = np.load(P(v1['ridge_test']['file'])); rl = c['rules']; nseed = len(c['final']['seeds'])
    M, S_dec = {}, {}                                                              # M[(arm, cond, kind)] -> [subjects, 200] cells
    for s in subs:
        Z = np.load(os.path.join(d2, f'z_emb_t{s:02d}.npz')); sc = Z['single_code']
        for cond in c['conditions']:
            for kind in kinds(c) + (['single'] if cond in c['single_conditions'] else []):
                for arm, E in (('dec', V.seed_mean([Z[f'seed{i}|{cond}|{kind}'] for i in range(nseed)])), ('shuf', V.unit(np.asarray(Z[f'shuffle|{cond}|{kind}'], np.float64)))):
                    S = E @ Tn.T
                    if kind == 'single': cl = V.cells(S, sc - 1); cl = np.array([cl[sc == k].mean() for k in range(1, 201)])
                    else: cl = V.cells(S, true)
                    M.setdefault((arm, cond, kind), []).append(cl)
                    if arm == 'dec' and kind == 'avg20': S_dec[(s, cond)] = S
        for kind in ('de', 'time', 'eye', 'pre'): M.setdefault(('ridge', kind, 'avg20'), []).append(V.cells(R[f'{kind}_t{s:02d}'], true))
    M = {k: np.stack(v) for k, v in M.items()}; rng = np.random.default_rng(c['seeds']['boot']); nb = c['n_boot']; boot = lambda X: V.pigeonhole(X, nb, rng)
    res = dict(subjects=subs, candidate=c['final']['candidate'], n_candidates=200, hypotheses={}, premises={}, report={})
    # premises
    nullm = {s: V.perm_null(S_dec[(s, 'ses123')], true, c['n_perm'], np.random.default_rng([c['seeds']['perm'], s])) for s in subs}
    best = max(('de', 'time'), key=lambda k: M[('ridge', k, 'avg20')].mean()); D1 = M[('dec', 'ses123', 'avg20')] - M[('ridge', best, 'avg20')]
    p1 = dict(baseline=best, baseline_mrp=float(M[('ridge', best, 'avg20')].mean()), other_mrp=float(M[('ridge', 'time' if best == 'de' else 'de', 'avg20')].mean()), diff=float(D1.mean()),
              lower90=float(np.percentile(boot(D1), rl['premise_lower_pct']))); p1['passed'] = p1['lower90'] > 0
    phase_null = {}
    for i, s in enumerate(subs):
        Zs = np.load(os.path.join(d2, f'z_emb_t{s:02d}.npz')); Sp = V.seed_mean([Zs[f'seed{j}|ses123|phase'] for j in range(nseed)]) @ Tn.T
        phase_null[s] = V.perm_null(Sp, true, c['n_perm'], np.random.default_rng([c['seeds']['perm'], s, 1])).mean()   # stream [perm, s, 1] for the phase input
    Xp = M[('dec', 'ses123', 'phase')] - np.array([phase_null[s] for s in subs])[:, None]
    p2 = dict(excess=float(Xp.mean()), upper90=float(np.percentile(boot(Xp), rl['premise_upper_pct']))); p2['passed'] = p2['upper90'] < rl['phase_excess_max']
    p3 = dict(mrp=float(M[('ridge', 'eye', 'avg20')].mean()), upper90=float(np.percentile(boot(M[('ridge', 'eye', 'avg20')]), rl['premise_upper_pct']))); p3['passed'] = p3['upper90'] < rl['meter_upper_max']
    p4 = dict(mrp=float(M[('ridge', 'pre', 'avg20')].mean()), upper90=float(np.percentile(boot(M[('ridge', 'pre', 'avg20')]), rl['premise_upper_pct']))); p4['passed'] = p4['upper90'] < rl['meter_upper_max']
    ex = {str(s): bool(M[('dec', 'ses123', 'avg20')][i].mean() > np.percentile(nullm[s], 95)) for i, s in enumerate(subs)}
    p5 = dict(exceed=ex, n_exceed=int(sum(ex.values())), passed=bool(sum(ex.values()) >= rl['prevalence_min']))
    res['premises'] = {'1_vs_ridge': p1, '2_phase': p2, '3_eye': p3, '4_pre': p4, '5_prevalence': p5}; gate = p2['passed'] and p3['passed'] and p4['passed'] and p5['passed']
    # fixed sequence H1 -> H3 -> H4
    def test(cond, kind, min_point):
        Dd = M[('dec', cond, kind)] - M[('shuf', cond, kind)]; r = V.rule(boot(Dd), Dd.mean(), rl['lower_pct'], min_point)
        r.update(decoder_mrp=float(M[('dec', cond, kind)].mean()), shuffle_mrp=float(M[('shuf', cond, kind)].mean())); return r
    h1 = test('ses123', 'avg20', rl['h1_min']); h1['premises_2_to_5'] = bool(gate); h1['adopted'] = bool(h1['adopted'] and gate)
    h1['wording'] = None if not h1['adopted'] else ('concept word retrieval not distinguishable from DE' if not p1['passed'] else 'model contribution over DE+ridge')
    res['hypotheses']['H1'] = h1; res['hypotheses']['H2'] = 'not tested (0-V failed, amendment 1)'
    res['hypotheses']['H3'] = test('k0', 'avg20', rl['h3_min']) if h1['adopted'] else 'not tested (H1 not adopted)'
    h3ok = isinstance(res['hypotheses']['H3'], dict) and res['hypotheses']['H3']['adopted']
    res['hypotheses']['H4'] = test('ses123', 'single', rl['h4_min']) if h3ok else 'not tested (H3 not adopted)'
    # report only
    rep = res['report']; Dm = lambda cond, kind: float(M[('dec', cond, kind)].mean())
    rep['r_curve_ses123'] = {str(r): Dm('ses123', f'r{r}') for r in c['r_curve']}; rep['avg80_ses123'] = Dm('ses123', 'avg80'); rep['noise_ses123'] = Dm('ses123', 'noise')
    rep['calibration_curve_avg20'] = {cond: Dm(cond, 'avg20') for cond in c['conditions']}; rep['single_k0'] = Dm('k0', 'single')
    rep['ridge_mrp'] = {k: float(M[('ridge', k, 'avg20')].mean()) for k in ('de', 'time', 'eye', 'pre')}; rep['per_subject_ses123'] = {str(s): float(M[('dec', 'ses123', 'avg20')][i].mean()) for i, s in enumerate(subs)}
    top = {}
    for cond in c['conditions']:
        rk = np.concatenate([V.ranks(S_dec[(s, cond)], true) for s in subs]); top[cond] = dict(top1=float((rk < 1).mean()), top5=float((rk < 5).mean()))
    rep['topk_avg20'] = top; rep['wolpaw_bits_per_selection_ses123'] = V.wolpaw(top['ses123']['top1'], 200)
    ms = json.load(open(P(os.path.join(c['eegfree_dir'], 'matched_sets.json')))); assert ms['test_folders'] == list(meta.test_img_concept)
    mcells = []
    for s in subs:
        S = S_dec[(s, 'ses123')]; mcells.append([1 - (S[i, ms['matched19'][i]] > S[i, i]).sum() / 19 for i in range(200)])
    rep['matched19_mrp_ses123'] = float(np.mean(mcells))
    t5, trainfrac = [], []
    for s in subs:
        Zs = np.load(os.path.join(d2, f'z_emb_t{s:02d}.npz')); E = V.seed_mean([Zs[f'seed{j}|ses123|avg20'] for j in range(nseed)]); S = E @ Tall.T
        o = np.argsort(-S, 1); t5.append([trow[i] in o[i, :5] for i in range(200)]); trainfrac.append(is_train[o[:, 0]])
    rep['open_vocab_1854'] = dict(top5=float(np.mean(t5)), top1_is_training_concept=float(np.mean(trainfrac)))
    tc_ = cat[trow]; win, btw = [], []
    for s in subs:
        S = S_dec[(s, 'ses123')]
        for i in range(200):
            same = np.where((tc_ == tc_[i]) & (true != i))[0]; other = np.where(tc_ != tc_[i])[0]
            if len(same): win.append(1 - (S[i, same] > S[i, i]).mean())
            btw.append(1 - (S[i, other] > S[i, i]).mean())
    rep['category'] = dict(within_mrp=float(np.mean(win)), between_mrp=float(np.mean(btw)), n_within_cells=len(win))
    res['peak_rss_gb'] = peak_gb(); json.dump(res, open(os.path.join(out, 'verdict.json'), 'w'), indent=1)
    print(json.dumps({k: (v['adopted'] if isinstance(v, dict) else v) for k, v in res['hypotheses'].items()}), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', choices=['mde', 'v1', 'v2', 'v3'], required=True)
    ap.add_argument('--unseal', action='store_true'); ap.add_argument('--subjects', type=int, nargs='+'); ap.add_argument('--v2-dir')
    ap.add_argument('--data-root'); ap.add_argument('--seal-table'); ap.add_argument('--config-override'); a = ap.parse_args()
    c = load_cfg(a.config, a.config_override); tc = yaml.safe_load(open(P(c['train_config'])))
    if a.stage == 'v2': return stage_v2(c, tc, a)
    meta = Meta(P(tc['meta']))
    if a.stage == 'mde':
        r = stage_mde(c, tc, meta); o = P(c['out_dir']); os.makedirs(o, exist_ok=True); json.dump(r, open(os.path.join(o, 'mde.json'), 'w'), indent=1)
        print(json.dumps({h: round(r[h]['mde'], 4) for h in ('H1', 'H3', 'H4')}), flush=True)
    elif a.stage == 'v1': stage_v1(c, tc, meta, a)
    else: stage_v3(c, tc, meta, a)


if __name__ == '__main__':
    main()
