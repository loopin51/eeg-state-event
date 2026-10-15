"""F-III A1 (D-105): one seed x subject fold x run fold of an arm, P1 or N (one of three candidates), with J250 and LS scoring.

The training path is scripts/c1_train.py (D-097) unchanged for P1: the same data, batches, random-number use, D-094 shared
procedure (evaluation every 150 updates on the B1 fixed validation batches, early stop after 4 evaluations without
improvement, never before 1,200 updates) and D-090 calibration (k in {0, 1, 2, 4, 14}, early stop on the target's two
validation runs, G*, other adapter). N changes only the training objective (shared training, its validation loss, the
calibration and its early-stopping loss; configs/a1_n.yaml):
  jitter  : CLIP loss with the EEG window of every item moved by U(-0.25, +0.25) s (targets unchanged)
  lse     : lag-marginalised CLIP loss over the 11 lags l of the 0.05 s grid in [-0.25, +0.25] s. The EEG window is forwarded
            once and the candidates' target windows are moved by -l (the same EEG-speech lag as moving the EEG window by +l;
            exact, one forward); logit(i, j) = log mean_m exp(s(i, j, m)) over the lags whose moved target window of j stays
            inside its run (lag 0 always is; the masked (candidate, lag) fraction is logged); cross-entropy over j. (A one-pass extended-window crop was rejected: on the trained P1 network its
            scores correlate 0.970 < 0.99 with per-window forwards, D-105.)
  lse_jit : lse with the EEG window moved by U(-0.1, +0.1) s per item (a real per-window shift)
Jitter draws come from separate generators, so the batch indices and subjects are the same as P1 for the same seed. Fixed
validation batches draw their jitter once (default_rng([0, 20261105])).

Additions for both arms: every target layer (k0, k1, k2, k4, k14 and G*) is saved (target_layers.pt); the test runs are also
scored under J250 (configs/a1.yaml: per-subject offsets, 30-s exclusion, margin 0.5 s) without search and with every LS rule,
and at delta 0 without search, for every k; score matrices (float16) of the k2 decision quantities (frozen LS rule, delta 0,
PRE) are kept for the verdict. In the development cell the A1 development metric (pseudo-target k2, J250, LS; delta 0) and
the C1 development metric (own layers, delta 0, exclusion 0) are stored. Blinding as in C1: test-run values only in
test_cells.json; scores.npz holds test matrices and is read only by the verdict scripts.

usage: python scripts/a1_train.py configs/a1_p1.yaml|configs/a1_n.yaml --seed S --subject-fold K --run-fold F
                                  [--candidate NAME] [--phaserand] [--smoke | --timing | --dev-only]
       --timing  : full shared training, first target only, evaluation on the validation runs; no MRP value is written
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys, json, time, copy, argparse, yaml, numpy as np, torch
import torch.nn.functional as Fn
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.brain_clip import BrainModule, clip_loss, clip_scores
from stageB_b1_train import RunData, run_epochs, check_setting
from stageB_b2_train import run_units
from c1_train import draw_runs, valid_queries
import a1_lib as L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--subject-fold', type=int, required=True); ap.add_argument('--run-fold', type=int, required=True)
    ap.add_argument('--phaserand', action='store_true'); ap.add_argument('--candidate')
    g = ap.add_mutually_exclusive_group(); g.add_argument('--smoke', action='store_true'); g.add_argument('--timing', action='store_true'); g.add_argument('--dev-only', action='store_true')
    a = ap.parse_args(); cc = yaml.safe_load(open(a.config)); c = yaml.safe_load(open(os.path.join(ROOT, cc['b1_config']))); A = yaml.safe_load(open(os.path.join(ROOT, cc['a1_config'])))
    sp = c['split']; K, F, seed = a.subject_fold, a.run_fold, a.seed
    if cc['arm'] == 'N':
        cand = a.candidate or cc['candidate']; assert cand in cc['candidates'], f'N: candidate {cand} not in {list(cc["candidates"])}'; obj = dict(cc['candidates'][cand], name=cand)
    else:
        assert a.candidate is None, 'P1 has no candidate'; cand = None; obj = dict(name='clip', jitter_s=0.0, lse=False)
    rule = A['lag_search']['rule']
    if not (a.smoke or a.timing or a.dev_only): assert rule in A['lag_search']['rules'], 'the LS rule must be frozen before any test-run scoring'
    if a.dev_only: assert (K, F) == (A['dev_cell']['subject_fold'], A['dev_cell']['run_fold']) and not a.phaserand, '--dev-only: development cell only'
    assert seed in cc['seeds'] or a.smoke or a.dev_only, f'seed {seed} not in {cc["seeds"]}'
    if not a.smoke: check_setting(c)
    st = c['settings'][c['decoder_setting']]; c['eeg']['reference'] = st['reference']; source = st['phaserand_source' if a.phaserand else 'source']
    test_runs = [r - 1 for r in sp['run_folds'][F]]; valid_runs = [r - 1 for r in sp['valid_runs'][F]]
    train_runs = [r for r in range(20) if r not in test_runs + valid_runs]
    targets = sp['subject_folds'][K] + sp.get('extra_targets', {}).get(K, [])
    trainers = sorted(s for j, fs in enumerate(sp['subject_folds']) if j != K for s in fs)
    o, cal, sh, cv = dict(c['optim']), dict(c['calibration']), dict(cc['shared_training']), dict(cc['calibration_curve'])
    o['updates_per_epoch'] = sh['eval_every']; ks = list(cv['ks']); nvb = 20
    if a.smoke:
        trainers = trainers[:3]; targets = targets[:2]; o.update(updates_per_epoch=10, batch_size=32); sh.update(eval_every=10, max_updates=20, min_updates=0)
        cal.update(updates_per_epoch=10, max_epochs=2, batch_size=32); ks = [1, 2]; nvb = 2
    if a.timing: targets = targets[:1]
    if a.dev_only: targets = []
    eval_runs = valid_runs if a.smoke or a.timing or a.dev_only else test_runs
    tag = '_phaserand' if a.phaserand else ''; tag += '_smoke' if a.smoke else '_timing' if a.timing else '_dev' if a.dev_only else ''
    name = time.strftime('%Y-%m-%d') + f"_a1_{cc['arm']}{'-' + cand if cand else ''}_{c['decoder_setting']}_s{seed}_k{K}_f{F}{tag}"
    out = os.path.join(ROOT, cc['out_root'], name); os.makedirs(out, exist_ok=True)
    is_dev = K == A['dev_cell']['subject_fold'] and F == A['dev_cell']['run_fold'] and not a.phaserand
    yaml.safe_dump(dict(c, a1=A, arm_config=cc, run=dict(arm=cc['arm'], candidate=cand, objective=obj, seed=seed, subject_fold=K, run_fold=F, phaserand=a.phaserand,
                                                     smoke=a.smoke, timing=a.timing, dev_only=a.dev_only, source=source, trainers=trainers, targets=targets, ks=ks,
                                                     train_runs=train_runs, valid_runs=valid_runs, eval_runs=eval_runs, dev_cell=is_dev, ls_rule=rule)),
                   open(os.path.join(out, 'config.yaml'), 'w'))
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = 'cuda'; t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    jr = np.random.default_rng([seed, 20261105, 1])                         # N jitter draws (separate: P1 batches unchanged)
    subjects = trainers + targets; nT = len(trainers)
    D = RunData(c, subjects, train_runs, valid_runs, eval_runs, source)
    if cc['car_exclude_bad']:
        meta = json.load(open(os.path.join(ROOT, c['cache_dir'], 'eeg_meta.json')))
        D.invalid = {s: np.array(meta['subjects'][str(s)]['flat']) | np.array(meta['subjects'][str(s)]['bad']) for s in subjects}
    D.to_device(dev)
    print(f'A1 {cc["arm"]} {obj["name"]} seed {seed} subject fold {K} run fold {F}: trainers {trainers} targets {targets} source {source}; windows train '
          f'{len(D.items["train"])} valid {len(D.items["valid"])} eval queries {len(D.items["eval"])} ({time.time() - t0:.0f}s)', flush=True)
    m = c['model']; net = BrainModule(out_dim=D.F.shape[0], n_subjects=len(subjects), merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                                      merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'],
                                      depth=m['depth'], subject_layers=True).to(dev)
    amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=o['precision'] == 'bf16'); hist = []
    tr_items, va_items = D.items['train'], D.items['valid']; vr = np.random.default_rng(0)
    lag_list = L.lags(A)

    def fwd(items, si, shift=0.0):                                         # c1_train.fwd (P1 path, scoring)
        X, P, Y = D.batch(items, si, shift)
        with torch.autocast(**amp): return net(X, P, torch.as_tensor(np.asarray(si), device=dev)).float(), Y

    def objective(items, si, jit):
        """training / validation loss of the arm; jit: per-item jitter in seconds (None for P1)"""
        if obj['name'] == 'clip':
            est, Y = fwd(items, si); return clip_loss(est, Y)
        Y = L.targets(D, items); sit = torch.as_tensor(np.asarray(si), device=dev)
        if not obj['lse']:
            X, P = L.batch_shift(D, items, si, jit)
            with torch.autocast(**amp): est = net(X, P, sit).float()
            return clip_loss(est, Y)
        X, P = L.batch_shift(D, items, si, jit)                           # lse: one forward; the lag moves the targets (-l)
        with torch.autocast(**amp): est = net(X, P, sit).float()
        return lse_loss(est, items, lag_list)

    mask_stat = dict(masked=0, total=0)
    def lse_loss(est, items, lags_):
        Sm, cnt = [], 0
        for l in lags_:
            Yl, ok = L.targets_shift(D, items, -l); Sm.append(clip_scores(est, Yl).masked_fill(~ok[None, :], float('-inf'))); cnt = cnt + ok.float()
        mask_stat['masked'] += int(len(lags_) * len(items) - cnt.sum().item()); mask_stat['total'] += len(lags_) * len(items)
        assert bool((cnt >= 1).all()), 'a candidate without any lag inside its run'   # lag 0 is always inside (items are in-run windows)
        return Fn.cross_entropy(torch.logsumexp(torch.stack(Sm), 0) - torch.log(cnt)[None, :], torch.arange(len(items), device=dev))

    def jitter(gen, n):
        return gen.uniform(-obj['jitter_s'], obj['jitter_s'], n) if obj['jitter_s'] > 0 else np.zeros(n)

    if obj['lse'] and a.smoke:                                             # check: the lag-0-only lse loss equals the CLIP loss
        it = tr_items[:32]; si0 = np.zeros(len(it), int); X, P = L.batch_shift(D, it, si0, np.zeros(len(it)))
        with torch.no_grad():
            est0 = net(X, P, torch.as_tensor(si0, device=dev)).float(); d0 = abs(lse_loss(est0, it, [0.0]).item() - clip_loss(est0, L.targets(D, it)).item())
        print(f'check lse lag-0 term vs CLIP loss: |diff| {d0:.2e}', flush=True); mask_stat.update(masked=0, total=0)
    # 1. shared network on the trainers (same fixed validation batches as B1/C1 for every seed and arm)
    opt = torch.optim.Adam(net.parameters(), lr=o['lr'])
    vb = [(vr.choice(len(va_items), o['batch_size'], replace=False), vr.integers(0, nT, o['batch_size'])) for _ in range(nvb)]
    vj = np.random.default_rng([0, 20261105]); vbj = [jitter(vj, o['batch_size']) for _ in range(nvb)]

    def step():
        net.train(); tl = 0.0
        for _ in range(o['updates_per_epoch']):
            idx = rng.choice(len(tr_items), o['batch_size'], replace=False); si = rng.integers(0, nT, o['batch_size'])
            loss = objective([tr_items[i] for i in idx], si, None if obj['name'] == 'clip' else jitter(jr, o['batch_size']))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item()
        return tl / o['updates_per_epoch']

    def vloss():
        net.eval(); vl = 0.0
        with torch.no_grad():
            for (idx, si), jj in zip(vb, vbj): vl += objective([va_items[i] for i in idx], si, None if obj['name'] == 'clip' else jj).item()
        return vl / len(vb), copy.deepcopy({kk: v.detach().cpu() for kk, v in net.state_dict().items()})

    best, state = run_units(step, vloss, sh['eval_every'], sh['max_updates'], sh['min_updates'], sh['patience_units'], hist, 'shared')
    net.load_state_dict(state); torch.save(state, os.path.join(out, 'shared.pt')); t_shared = time.time() - t0
    hs = [h for h in hist if h['stage'] == 'shared']
    shared_mask = mask_stat['masked'] / max(1, mask_stat['total']) if obj['lse'] else None; mask_stat.update(masked=0, total=0)
    print(f'shared done: best valid {best:.4f} @ {min(hs, key=lambda h: h["valid"])["update"]} ({t_shared:.0f}s)' + (f'; masked (candidate, lag) fraction {shared_mask:.4f}' if obj['lse'] else ''), flush=True)
    net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    W = net.subject.weights; mean_layer = W[:nT].mean(0).clone(); excl = A['queries']['exclusion_s']; rules = A['lag_search']['rules']

    def rank_cells(si, items, shift=0.0):                                  # C1 definition (delta 0, exclusion 0), kept for continuity
        run = np.array([x[0] for x in items]); cells = {}
        with torch.no_grad():
            for rr in sorted(set(run.tolist())):
                qi = [items[i] for i in np.where(run == rr)[0]]
                if shift: qi = [x for x in qi if x[1] + c['window']['tmin'] + c['eeg']['offset_s'] + shift >= 0]
                Y = torch.stack([D.Fg[:, D.feat_idx(x, t) + D.ar] for x, t in qi]).float(); ch = c['eval']['score_chunk']
                est = torch.cat([fwd(qi[b0:b0 + ch], np.full(len(qi[b0:b0 + ch]), si), shift)[0] for b0 in range(0, len(qi), ch)])
                S = clip_scores(est, Y); rk = (S > torch.diag(S)[:, None]).sum(1).cpu().numpy(); n = len(qi)
                cells[str(rr + 1)] = dict(mrp=float(np.mean(1 - rk / (n - 1))), n=n, top1=float((rk < 1).mean()), top10=float((rk < 10).mean()))
        return cells

    def a1_cells(si, subject, items, keep=None, pre=False):
        """J250 (+ every LS rule) and delta 0 per run on the queries inside the margin (pre: EEG 10 s earlier, its own subset)"""
        run = np.array([x[0] for x in items]); cells = {}; base0 = -c['eval']['pre_shift_s'] if pre else 0.0
        for rr in sorted(set(run.tolist())):
            allq = [items[i] for i in np.where(run == rr)[0]]
            qi = [allq[k] for k in L.inside(D, allq, A['queries']['margin_s'])]
            if pre: qi = [x for x in qi if x[1] + c['window']['tmin'] + c['eeg']['offset_s'] + base0 - A['queries']['margin_s'] >= 0]
            if len(qi) < 2: continue
            sis = np.full(len(qi), si); tt = [t for _, t in qi]; cand = L.targets(D, qi); cell = dict(n=len(qi))
            S0 = clip_scores(L.forward(net, D, qi, sis, np.full(len(qi), base0), amp), cand); cell['d0_fixed'] = L.mrp_excl(S0, tt, excl)
            SL = L.lag_stack(net, D, qi, sis, base0 + L.j_offsets(A, subject, qi), lag_list, cand, amp)
            cell['j250_fixed'] = L.mrp_excl(SL[lag_list.index(0.0)], tt, excl)
            for rl in rules: cell[f'j250_{rl}'] = L.mrp_excl(L.aggregate(SL, rl), tt, excl)
            if keep is not None:
                keep[f'r{rr + 1}_onset'] = np.array(tt)
                if rule: keep[f'r{rr + 1}_j250'] = L.aggregate(SL, rule).cpu().numpy().astype(np.float16)
                if not pre: keep[f'r{rr + 1}_d0'] = S0.cpu().numpy().astype(np.float16)
            cells[str(rr + 1)] = cell; del SL
        return cells

    def calibrate(ti, s, k, items_k, crng, cjr, cvb, cvj, g_ep, gstar):
        """D-090 calibration of the layer of subject index ti with the arm's objective; returns (best valid, layer, epochs)"""
        with torch.no_grad(): W[ti] = mean_layer
        W.requires_grad_(True); copt = torch.optim.Adam([W], lr=cal['lr']); ep_count = [0]

        def cvloss():
            vl = 0.0
            with torch.no_grad():
                for (idx, si), jj in zip(cvb, cvj): vl += objective([va_items[i] for i in idx], si, None if obj['name'] == 'clip' else jj).item()
            return vl / len(cvb), W[ti].detach().clone()

        def cstep():
            tl = 0.0
            for _ in range(cal['updates_per_epoch']):
                idx = crng.choice(len(items_k), cal['batch_size'], replace=False)
                loss = objective([items_k[i] for i in idx], np.full(cal['batch_size'], ti), None if obj['name'] == 'clip' else jitter(cjr, cal['batch_size']))
                copt.zero_grad(set_to_none=True); loss.backward(); copt.step(); tl += loss.item()
            if ep_count[0] == g_ep: gstar['layer'] = W[ti].detach().clone()
            ep_count[0] += 1
            return tl / cal['updates_per_epoch']
        init_vl, _ = cvloss(); tg = f'cal_{s}_k{k}'; hist.append(dict(stage=tg, epoch=-1, train=None, valid=init_vl))
        b, row = run_epochs(cstep, cvloss, cal['max_epochs'], cal['patience'], hist, tg)
        if init_vl <= b: b, row = init_vl, mean_layer.clone()
        W.requires_grad_(False)
        with torch.no_grad(): W[ti] = row
        ce = [h for h in hist if h['stage'] == tg and h['epoch'] >= 0]
        return b, row, init_vl, ce

    res, arrays, timing, layers_out = {}, {}, {}, {}
    # development cell: C1 development metric (own layers) and the A1 development metric (pseudo-target k2, J250, LS; delta 0)
    if is_dev or a.smoke:
        vq = valid_queries(D, c, valid_runs); own = W[:nT].detach().clone()
        res['_dev_trainers_valid'] = {str(s): rank_cells(j, vq) for j, s in enumerate(trainers)}
        pt = A['pseudo_target']; dev_a1 = {}; dev_layers = {}
        for j, s in enumerate(trainers[:2] if a.smoke else trainers):
            others = [kk for kk in range(nT) if kk != j]; runs_k = draw_runs(pt['draw_seed'], s, F, train_runs, pt['k'])
            items_k = [x for x in tr_items if x[0] in runs_k]; prng = np.random.default_rng([pt['calib_seed'], s, F, pt['k'], seed]); pjr = np.random.default_rng([pt['calib_seed'], s, F, pt['k'], seed, 1])
            with torch.no_grad(): W[j] = own[others].mean(0)
            W.requires_grad_(True); popt = torch.optim.Adam([W], lr=cal['lr'])
            for _ in range(pt['updates']):
                idx = prng.choice(len(items_k), cal['batch_size'], replace=False)
                loss = objective([items_k[i] for i in idx], np.full(cal['batch_size'], j), None if obj['name'] == 'clip' else jitter(pjr, cal['batch_size']))
                popt.zero_grad(set_to_none=True); loss.backward(); popt.step()
            W.requires_grad_(False); dev_layers[str(s)] = W[j].detach().cpu().clone()
            dev_a1[str(s)] = a1_cells(j, s, vq)
            with torch.no_grad(): W[j] = own[j]
            torch.cuda.empty_cache()
        res['_dev_a1_pseudo_k2'] = dev_a1; torch.save(dev_layers, os.path.join(out, 'dev_layers_pseudo_k2.pt'))
        print(f'dev cell done ({time.time() - t0:.0f}s)', flush=True)

    # 2.-3. calibration curve and evaluation, target by target
    q = D.items['eval']; own_k14 = {}
    for j, s in enumerate(targets):
        ti = nT + j; o_s = res[str(s)] = {}; t1 = time.time(); lay = layers_out[str(s)] = {}
        with torch.no_grad(): W[ti] = mean_layer
        lay['k0'] = mean_layer.cpu().clone(); o_s['k0'] = rank_cells(ti, q); o_s['a1_k0'] = a1_cells(ti, s, q)
        vrs = np.random.default_rng([cv['valid_seed'], s, F])
        cvb = [(vrs.choice(len(va_items), cal['batch_size'], replace=False), np.full(cal['batch_size'], ti)) for _ in range(nvb)]
        cvjg = np.random.default_rng([cv['valid_seed'], s, F, 1]); cvj = [jitter(cvjg, cal['batch_size']) for _ in range(nvb)]
        for k in ks:
            t2 = time.time(); runs_k = draw_runs(cv['draw_seed'], s, F, train_runs, k)
            items_k = [x for x in tr_items if x[0] in runs_k]; crng = np.random.default_rng([cv['calib_seed'], s, F, k, seed]); cjr = np.random.default_rng([cv['calib_seed'], s, F, k, seed, 1])
            gstar = {}; g_ep = cv['gstar_epoch'].get(k, cv['gstar_epoch'].get(str(k)))
            b, row, init_vl, ce = calibrate(ti, s, k, items_k, crng, cjr, cvb, cvj, g_ep, gstar); lay[f'k{k}'] = row.cpu().clone()
            o_s[f'k{k}'] = rank_cells(ti, q)
            keep = {} if k == 2 else None; o_s[f'a1_k{k}'] = a1_cells(ti, s, q, keep=keep)
            if keep is not None: arrays.update({f'k2_s{s}_{kk}': v for kk, v in keep.items()})
            if k == 2:
                keep = {}; o_s['a1_pre_k2'] = a1_cells(ti, s, q, keep=keep, pre=True); arrays.update({f'pre_k2_s{s}_{kk}': v for kk, v in keep.items()})
            o_s[f'k{k}_meta'] = dict(runs=[x + 1 for x in runs_k], best_valid=b, init_valid=init_vl, kept_init=bool(init_vl <= min(h['valid'] for h in ce)),
                                     best_epoch=int(min(ce, key=lambda h: h['valid'])['epoch']), epochs_run=len(ce))
            if 'layer' in gstar:
                lay[f'gstar_k{k}'] = gstar['layer'].cpu().clone()
                with torch.no_grad(): W[ti] = gstar['layer']
                o_s[f'k{k}_gstar'] = rank_cells(ti, q); o_s[f'k{k}_gstar_meta'] = dict(epoch=g_ep)
                with torch.no_grad(): W[ti] = row
            if k == 14: own_k14[s] = row.clone()
            timing[f'{s}_k{k}'] = time.time() - t2; torch.cuda.empty_cache()
            print(f'sub-{s:03d} k{k}: runs {[x + 1 for x in runs_k]} best valid {b:.4f} epoch {o_s[f"k{k}_meta"]["best_epoch"]} ({timing[f"{s}_k{k}"]:.0f}s)', flush=True)
        timing[f'{s}_total'] = time.time() - t1
    if cv.get('other_adapter') and 14 in ks and len(targets) > 1:
        for j, s in enumerate(targets):
            ti = nT + j; res[str(s)]['other_adapter_k14'] = {}
            for o_ in targets:
                if o_ != s:
                    with torch.no_grad(): W[ti] = own_k14[o_]
                    res[str(s)]['other_adapter_k14'][str(o_)] = rank_cells(ti, q)
            with torch.no_grad(): W[ti] = own_k14[s]
    torch.save(layers_out, os.path.join(out, 'target_layers.pt'))
    json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(out, 'scores.npz'), **arrays)
    summ = dict(arm=cc['arm'], candidate=cand, seed=seed, eval_runs=[r + 1 for r in eval_runs], eval_split='dev_only' if a.dev_only else 'valid' if a.smoke or a.timing else 'test',
                n_queries=len(q), shared_best_valid=best, shared_best_update=min(hs, key=lambda h: h['valid'])['update'], shared_updates=hs[-1]['update'],
                shared_seconds=t_shared, lse_masked_fraction_shared=shared_mask, lse_masked_fraction_rest=(mask_stat['masked'] / max(1, mask_stat['total']) if obj['lse'] else None),
                car_exclude_bad=cc['car_exclude_bad'], device=torch.cuda.get_device_name(), ls_rule=rule,
                max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2 ** 30, seconds=time.time() - t0, calibration_seconds=timing)
    if a.timing:                              # time and memory only: no MRP value is written (an N timing run precedes the development stage)
        def blind(x): return {k: blind(v) for k, v in x.items()} if isinstance(x, dict) else ('-' if isinstance(x, float) else x)
        res = {s_: ({k: (v if k.endswith('_meta') else blind(v)) for k, v in r.items()}) for s_, r in res.items()}
    if summ['eval_split'] == 'test':          # blinding: test-run cells only in test_cells.json (verdict and MDE scripts)
        meta_only = {s: {kk: v for kk, v in r.items() if kk.endswith('_meta')} for s, r in res.items() if not s.startswith('_')}
        meta_only.update({s: r for s, r in res.items() if s.startswith('_')})
        json.dump(dict(summary=summ, targets=res), open(os.path.join(out, 'test_cells.json'), 'w'), indent=1)
        json.dump(dict(summary=summ, targets=meta_only), open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    else:
        json.dump(dict(summary=summ, targets=res), open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    print('SUMMARY', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
