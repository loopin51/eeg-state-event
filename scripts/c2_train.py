"""C2 (D-097 section 14, amendments 2 and 4): one seed x subject fold x trial fold of an arm (P1_DTU, or P2_DTU with --init) on
the DTU two-talker trials, attended stream as target. The C1 job (scripts/c1_train.py) with the DTU trial structure:

1. shared network on the calibration-candidate trials of the training subjects (subject folds other than K), validation on
   their validation pairs of trial fold f; same procedure as C1 (D-091 amendment 4: evaluation every `eval_every` updates on
   fixed validation batches, early stopping after `patience_units` evaluations without improvement but never before
   `min_updates`; bad- and flat-channel-excluded CAR). Every item is (subject, trial, t) and uses its own subject layer;
   P2_DTU starts from the same seed's external pre-training checkpoint (--init, everything but the subject layers);
2. calibration curve in the same job: for every target, k0 (mean of the training subjects' layers) and the target's layer
   refitted from that mean on the nested draw of n_k calibration trials (c2_dtu_split.draw: k1 4, k2 7, k4 14, kmax all),
   B1 calibration settings, early stopping on the target's own 7 validation trials;
3. within-trial evaluation on the target's test trials (query_step_s grid, candidates = all query windows of the same trial):
   per (target, trial) MRP, top-1, top-10, query count; within-trial score matrices (float16) for `save_scores`, PRE (10 s
   earlier EEG) for `save_pre`, and the UNATTENDED report (same decoder output ranked among the unattended-stream windows of the
   trial) for `save_unattended`.
No G* or other-adapter reports (C1 section 13 items, not part of C2). No development cell (the development stage is C1 only).

Blinding (as C1): on the test trials the cell values are written to test_cells.json only; metrics.json keeps the summary and
calibration meta. Nothing about the test trials is printed.

usage: python scripts/c2_train.py configs/c2_p1.yaml --seed S --subject-fold K --trial-fold F [--phaserand] [--init CKPT]
                                  [--smoke | --timing] [--candidate NAME]
       --smoke : 3 trainers, 2 targets, 20 shared updates, k in {2, max} (PRE and UNATTENDED paths); evaluation on the validation trials (never the test trials)
       --timing: full shared training, first target only, all k; evaluation on the validation trials (never the test trials)
"""
import os, sys, glob, json, time, copy, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.brain_clip import BrainModule, clip_loss, clip_scores
from stageB_b1_train import run_epochs, check_setting
from stageB_b2_train import run_units
from c2_dtu_data import DTUData
from c2_dtu_split import path


def kname(k): return 'kmax' if str(k) == 'max' else f'k{k}'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--subject-fold', type=int, required=True); ap.add_argument('--trial-fold', type=int, required=True)
    ap.add_argument('--phaserand', action='store_true'); ap.add_argument('--init')
    g = ap.add_mutually_exclusive_group(); g.add_argument('--smoke', action='store_true'); g.add_argument('--timing', action='store_true')
    ap.add_argument('--candidate')
    a = ap.parse_args(); cc = yaml.safe_load(open(path(a.config))); c = yaml.safe_load(open(path(cc['b1_config'])))
    fc = yaml.safe_load(open(path(cc['folds_config']))); K, F, seed = a.subject_fold, a.trial_fold, a.seed
    if cc['init'] == 'pretrain' and a.init is None:                       # P2_DTU: the same seed's C1 pre-training checkpoint (DTU never in it)
        pc = yaml.safe_load(open(path(cc['pretrain_from'])))['pretrain']
        cand = a.candidate or pc['candidate']; assert cand, 'P2_DTU: no pre-training candidate given or frozen (C1 development stage)'
        hits = sorted(glob.glob(path(pc['ckpt_glob'].format(seed=seed, candidate=cand))))
        assert len(hits) == 1, f'{pc["ckpt_glob"]} seed {seed} candidate {cand}: {len(hits)} checkpoints'; a.init = hits[0]
    assert seed in cc['seeds'] or a.smoke, f'seed {seed} not in {cc["seeds"]}'
    assert (cc['init'] == 'none') == (a.init is None), f"arm {cc['arm']}: init {cc['init']} vs --init {a.init}"
    if not a.smoke: check_setting(c)
    st = c['settings'][c['decoder_setting']]; c['eeg']['reference'] = st['reference']; source = st['phaserand_source' if a.phaserand else 'source']
    o, cal, sh, cv = dict(c['optim']), dict(c['calibration']), dict(cc['shared_training']), dict(cc['calibration_curve'])
    o['updates_per_epoch'] = sh['eval_every']; ks = list(cv['ks']); nvb = 20
    D = DTUData(c, fc, K, F, source=source); trainers, targets = list(D.trainers), list(D.targets)
    if a.smoke:
        trainers = trainers[:3]; targets = targets[:2]; o.update(updates_per_epoch=10, batch_size=32); sh.update(eval_every=10, max_updates=20, min_updates=0)
        cal.update(updates_per_epoch=10, max_epochs=2, batch_size=32); ks = [2, 'max']; nvb = 2
        D.items = {kk: [x for x in v if x[0] in trainers] for kk, v in D.items.items()}
    if a.timing: targets = targets[:1]
    eval_split = 'valid' if a.smoke or a.timing else 'test'               # smoke and timing never touch the test trials
    tag = ('_phaserand' if a.phaserand else '') + ('_smoke' if a.smoke else '_timing' if a.timing else '')
    name = time.strftime('%Y-%m-%d') + f"_c2_{cc['arm']}_{c['decoder_setting']}_s{seed}_k{K}_f{F}{tag}"
    out = path(os.path.join(cc['out_root'], name)); os.makedirs(out, exist_ok=True)
    yaml.safe_dump(dict(c, c2=cc, folds=fc, run=dict(arm=cc['arm'], seed=seed, subject_fold=K, trial_fold=F, phaserand=a.phaserand, init=a.init,
                                                    smoke=a.smoke, timing=a.timing, candidate=a.candidate, source=source, trainers=trainers,
                                                    targets=targets, ks=[str(k) for k in ks], eval_split=eval_split)), open(os.path.join(out, 'config.yaml'), 'w'))
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = 'cuda'; t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    D.to_device(dev); subjects = trainers + targets; nT = len(trainers); sidx = {s: i for i, s in enumerate(subjects)}
    tr_items, va_items = D.items['train'], D.items['valid']
    print(f'{cc["arm"]} seed {seed} subject fold {K} trial fold {F}: trainers {trainers} targets {targets} source {source}; windows train '
          f'{len(tr_items)} valid {len(va_items)} ({time.time() - t0:.0f}s)', flush=True)
    m = c['model']; net = BrainModule(out_dim=D.F.shape[0], n_subjects=len(subjects), merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                                      merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'],
                                      depth=m['depth'], subject_layers=True).to(dev)
    if a.init:                                                            # P2_DTU: pre-trained weights, fresh DTU subject layers
        sd = torch.load(a.init, map_location='cpu'); sd = {kk: v for kk, v in sd.items() if not kk.startswith('subject.')}
        missing, unexpected = net.load_state_dict(sd, strict=False); assert not unexpected and all(x.startswith('subject.') for x in missing), (missing, unexpected)
    amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=o['precision'] == 'bf16'); hist = []; vr = np.random.default_rng(0)

    def fwd(items, si, shift=0.0, stream='stim'):
        (X, P, *_), Y = D.batch(items, si, shift, stream)
        with torch.autocast(**amp): return net(X, P, torch.as_tensor(np.asarray(si), device=dev)).float(), Y

    def own(items): return [sidx[x[0]] for x in items]                    # every DTU item uses its own subject's layer

    # 1. shared network on the trainers (fixed validation batches, the same for every seed and arm)
    opt = torch.optim.Adam(net.parameters(), lr=o['lr'])
    vb = [vr.choice(len(va_items), o['batch_size'], replace=False) for _ in range(nvb)]

    def step():
        net.train(); tl = 0.0
        for _ in range(o['updates_per_epoch']):
            it = [tr_items[i] for i in rng.choice(len(tr_items), o['batch_size'], replace=False)]
            est, Y = fwd(it, own(it)); loss = clip_loss(est, Y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item()
        return tl / o['updates_per_epoch']

    def vloss():
        net.eval(); vl = 0.0
        with torch.no_grad():
            for idx in vb:
                it = [va_items[i] for i in idx]; est, Y = fwd(it, own(it)); vl += clip_loss(est, Y).item()
        return vl / len(vb), copy.deepcopy({kk: v.detach().cpu() for kk, v in net.state_dict().items()})

    best, state = run_units(step, vloss, sh['eval_every'], sh['max_updates'], sh['min_updates'], sh['patience_units'], hist, 'shared')
    net.load_state_dict(state); torch.save(state, os.path.join(out, 'shared.pt')); t_shared = time.time() - t0
    hs = [h for h in hist if h['stage'] == 'shared']
    print(f'shared done: best valid {best:.4f} @ {min(hs, key=lambda h: h["valid"])["update"]} ({t_shared:.0f}s)', flush=True)
    net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    W = net.subject.weights; mean_layer = W[:nT].mean(0).clone()

    def rank_cells(si, items, shift=0.0, keep=None, stream='stim'):
        """within-trial ranking of `items` (query windows of one subject) for subject index si; candidates = the query windows of the
        same trial in `stream` ('stim' = attended; 'unattended' = UNATTENDED report); keep receives the score matrices"""
        cells = {}; ch = c['eval']['score_chunk']
        with torch.no_grad():
            for ri in sorted({x[1] for x in items}):
                qi = [x for x in items if x[1] == ri]; n = len(qi)
                Y = torch.cat([D.batch(qi[b0:b0 + ch], [si] * len(qi[b0:b0 + ch]), 0.0, stream)[1] for b0 in range(0, n, ch)]).float()
                est = torch.cat([fwd(qi[b0:b0 + ch], [si] * len(qi[b0:b0 + ch]), shift)[0] for b0 in range(0, n, ch)])
                S = clip_scores(est, Y); rk = (S > torch.diag(S)[:, None]).sum(1).cpu().numpy(); tr = str(D.recs[qi[0][0]][ri]['trial'])
                cells[tr] = dict(mrp=float(np.mean(1 - rk / (n - 1))), n=n, top1=float((rk < 1).mean()), top10=float((rk < 10).mean()))
                if keep is not None: keep[f't{tr}'] = S.cpu().numpy().astype(np.float16); keep[f't{tr}_onset'] = np.array([t for _, _, t in qi])
        return cells

    def pooled(cells): return float(np.average([v['mrp'] for v in cells.values()], weights=[v['n'] for v in cells.values()]))

    # 2.-3. calibration curve and evaluation, target by target
    res, arrays, timing = {}, {}, {}; which = 'valid' if eval_split == 'valid' else 'test'
    for j, s in enumerate(targets):
        ti = sidx[s]; o_s = res[str(s)] = {}; t1 = time.time(); q = D.queries(s, which); pre_q = D.queries(s, which, pre_shift=-c['eval']['pre_shift_s'])
        o_s['n_trials'] = len({x[1] for x in q})

        def evaluate(kn):
            keep = {} if kn in cc['save_scores'] else None; o_s[kn] = rank_cells(ti, q, keep=keep)
            if keep is not None: arrays.update({f'main_{kn}_s{s}_{kk}': v for kk, v in keep.items()})
            if kn in cc['save_pre']:
                keep = {}; o_s[f'pre_{kn}'] = rank_cells(ti, pre_q, shift=-c['eval']['pre_shift_s'], keep=keep)
                arrays.update({f'pre_{kn}_s{s}_{kk}': v for kk, v in keep.items()})
            if kn in cc['save_unattended']:
                keep = {}; o_s[f'unattended_{kn}'] = rank_cells(ti, q, keep=keep, stream='unattended')
                arrays.update({f'unatt_{kn}_s{s}_{kk}': v for kk, v in keep.items()})

        with torch.no_grad(): W[ti] = mean_layer
        evaluate('k0')
        vit = D.valid_items(s); vrs = np.random.default_rng([cv['valid_seed'], s, F])
        cvb = [vrs.choice(len(vit), min(cal['batch_size'], len(vit)), replace=False) for _ in range(nvb)]

        def cvloss():
            vl = 0.0
            with torch.no_grad():
                for idx in cvb:
                    it = [vit[i] for i in idx]; est, Y = fwd(it, [ti] * len(it)); vl += clip_loss(est, Y).item()
            return vl / len(cvb), W[ti].detach().clone()
        for k in ks:
            kn = kname(k); t2 = time.time(); recs_k = D.split.draw(s, F, k, cv['draw_seed']); items_k = D.grid(s, recs_k, c['window']['train_step_s'])
            crng = np.random.default_rng([cv['calib_seed'], s, F, 999 if kn == 'kmax' else int(k), seed])
            with torch.no_grad(): W[ti] = mean_layer
            W.requires_grad_(True); copt = torch.optim.Adam([W], lr=cal['lr'])

            def cstep():
                tl = 0.0
                for _ in range(cal['updates_per_epoch']):
                    it = [items_k[i] for i in crng.choice(len(items_k), min(cal['batch_size'], len(items_k)), replace=False)]
                    est, Y = fwd(it, [ti] * len(it)); loss = clip_loss(est, Y)
                    copt.zero_grad(set_to_none=True); loss.backward(); copt.step(); tl += loss.item()
                return tl / cal['updates_per_epoch']
            init_vl, _ = cvloss(); tg = f'cal_{s}_{kn}'; trials_k = [r['trial'] for r in recs_k]
            hist.append(dict(stage=tg, epoch=-1, train=None, valid=init_vl, trials=trials_k))
            b, row = run_epochs(cstep, cvloss, cal['max_epochs'], cal['patience'], hist, tg)
            if init_vl <= b: b, row = init_vl, mean_layer.clone()           # keep the initial layer if fitting never helped (B1, D-090)
            W.requires_grad_(False)
            with torch.no_grad(): W[ti] = row
            ce = [h for h in hist if h['stage'] == tg and h['epoch'] >= 0]; evaluate(kn)
            o_s[f'{kn}_meta'] = dict(trials=trials_k, n_trials=len(trials_k), minutes=sum(D.recs[s][r['index']]['length'] for r in recs_k) / D.fs / 60,
                                     best_valid=b, init_valid=init_vl, kept_init=bool(init_vl <= min(h['valid'] for h in ce)),
                                     best_epoch=int(min(ce, key=lambda h: h['valid'])['epoch']), epochs_run=len(ce))
            timing[f'{s}_{kn}'] = time.time() - t2
            print(f'sub-{s:03d} {kn}: {len(trials_k)} trials best valid {b:.4f} epoch {o_s[f"{kn}_meta"]["best_epoch"]} ({timing[f"{s}_{kn}"]:.0f}s)'
                  + (f' valid-trial MRP {pooled(o_s[kn]):.4f}' if a.smoke or a.timing else ''), flush=True)
        timing[f'{s}_total'] = time.time() - t1
    json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(out, 'scores.npz'), **arrays)
    summ = dict(arm=cc['arm'], seed=seed, subject_fold=K, trial_fold=F, eval_split=eval_split, n_queries={str(s): sum(v['n'] for v in res[str(s)]['k0'].values()) for s in targets},
                shared_best_valid=best, shared_best_update=min(hs, key=lambda h: h['valid'])['update'], shared_updates=hs[-1]['update'],
                shared_seconds=t_shared, car_exclude_bad=True, device=torch.cuda.get_device_name(), init=a.init,
                max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2 ** 30, seconds=time.time() - t0, calibration_seconds=timing)
    if eval_split == 'test':                  # blinding: test-trial cells only in test_cells.json (C2 MDE and verdict scripts)
        meta_only = {s: {kk: v for kk, v in r.items() if kk.endswith('_meta') or kk == 'n_trials'} for s, r in res.items()}
        json.dump(dict(summary=summ, targets=res), open(os.path.join(out, 'test_cells.json'), 'w'), indent=1)
        json.dump(dict(summary=summ, targets=meta_only), open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    else:
        json.dump(dict(summary=summ, targets=res), open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    print('SUMMARY', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
