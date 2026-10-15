"""C1 (D-097): one seed x subject fold x run fold of an arm (P1, or P2 with --init) of the calibration-efficiency comparison.

1. shared network on the training runs (14) of the pool subjects outside subject fold K, D-091 amendment 4 procedure
   (evaluation every `shared_training.eval_every` updates on the B1 fixed validation batches, early stopping after
   `patience_units` evaluations without improvement but never before `min_updates`; bad-channel-excluded CAR), w2v target;
   P2 starts from the external pre-training checkpoint of the same seed (--init, everything but the subject layers);
2. D-090 calibration curve in the same job: for every target, k0 (mean of the training subjects' layers) and the target's layer
   refitted from that mean on the first k of a fixed permutation of its 14 training runs, B1 calibration settings, early
   stopping on its 2 validation runs; plus the descriptive G* layer (fixed number of 200-update epochs, no validation) and
   the other targets' k14 layers (other adapter);
3. within-run evaluation on the test runs (word-onset queries, B1 definition): per (target, condition, run) MRP, top-1,
   top-10, query count; within-run score matrices (float16) for `save_scores` conditions and PRE (10 s earlier) for `save_pre`.
In the development cell (`dev_cell`) the training subjects' validation-run MRP of the selected shared network is also stored.

Blinding (D-097): on the test runs the cell values (MRP, top-k) are written to test_cells.json only, which is read by the MDE
script (seed deviations only) and the C1 verdict script; metrics.json keeps the summary, calibration meta and the
development-cell values. Nothing about the test runs is printed.

usage: python scripts/c1_train.py configs/c1_p1.yaml --seed S --subject-fold K --run-fold F [--phaserand] [--init CKPT]
                                  [--smoke | --timing | --dev-only] [--candidate NAME]
       --smoke : 3 trainers, 2 targets, 20 shared updates, k in {1, 14}; evaluation on the validation runs (never the test runs)
       --timing: full shared training, first target only, all k; evaluation on the validation runs (never the test runs)
       --dev-only: development cell only (D-097 section 11): shared training and the training-subject validation-run MRP, no
                   target, no calibration, the test runs are never loaded; folder suffix _dev-<candidate>
       --candidate: P2 (init: pretrain) without --init: the checkpoint is pretrain.ckpt_glob of the config for this seed and
                    candidate (default pretrain.candidate)
"""
import os, sys, glob, json, time, copy, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.brain_clip import BrainModule, clip_loss, clip_scores
from stageB_b1_train import RunData, run_epochs, check_setting
from stageB_b2_train import run_units


def draw_runs(seed, subject, run_fold, train_runs, k):       # identical to stageB_calib_curve.draw_runs (D-090 nested draw)
    return sorted(int(r) for r in np.random.default_rng([seed, subject, run_fold]).permutation(sorted(train_runs))[:k])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--subject-fold', type=int, required=True); ap.add_argument('--run-fold', type=int, required=True)
    ap.add_argument('--phaserand', action='store_true'); ap.add_argument('--init')
    g = ap.add_mutually_exclusive_group(); g.add_argument('--smoke', action='store_true'); g.add_argument('--timing', action='store_true')
    g.add_argument('--dev-only', action='store_true'); ap.add_argument('--candidate')
    a = ap.parse_args(); cc = yaml.safe_load(open(a.config)); c = yaml.safe_load(open(os.path.join(ROOT, cc['b1_config'])))
    sp = c['split']; K, F, seed = a.subject_fold, a.run_fold, a.seed
    if cc['init'] == 'pretrain' and a.init is None:                       # P2: the same seed's pre-training checkpoint of the candidate
        cand = a.candidate or cc['pretrain']['candidate']; assert cand, 'P2: no pre-training candidate given or frozen'
        hits = sorted(glob.glob(os.path.join(ROOT, cc['pretrain']['ckpt_glob'].format(seed=seed, candidate=cand))))
        assert len(hits) == 1, f'{cc["pretrain"]["ckpt_glob"]} seed {seed} candidate {cand}: {len(hits)} checkpoints'; a.init = hits[0]
    if a.dev_only: assert (K, F) == (cc['dev_cell']['subject_fold'], cc['dev_cell']['run_fold']) and not a.phaserand, '--dev-only: development cell only'
    assert seed in cc['seeds'] or a.smoke, f'seed {seed} not in {cc["seeds"]}'
    assert (cc['init'] == 'none') == (a.init is None), f"arm {cc['arm']}: init {cc['init']} vs --init {a.init}"
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
        cal.update(updates_per_epoch=10, max_epochs=2, batch_size=32); ks = [1, 14]; nvb = 2
    if a.timing: targets = targets[:1]
    if a.dev_only: targets = []
    eval_runs = valid_runs if a.smoke or a.timing or a.dev_only else test_runs    # smoke, timing and dev-only never touch the test runs
    tag = '_phaserand' if a.phaserand else ''; tag += '_smoke' if a.smoke else '_timing' if a.timing else ''
    tag += f'_dev-{a.candidate or cc.get("pretrain", {}).get("candidate") or "base"}' if a.dev_only else ''
    name = time.strftime('%Y-%m-%d') + f"_c1_{cc['arm']}_{c['decoder_setting']}_s{seed}_k{K}_f{F}{tag}"
    out = os.path.join(ROOT, cc['out_root'], name); os.makedirs(out, exist_ok=True)
    is_dev = K == cc['dev_cell']['subject_fold'] and F == cc['dev_cell']['run_fold'] and not a.phaserand
    yaml.safe_dump(dict(c, c1=cc, run=dict(arm=cc['arm'], seed=seed, subject_fold=K, run_fold=F, phaserand=a.phaserand, init=a.init, smoke=a.smoke,
                                           timing=a.timing, dev_only=a.dev_only, candidate=a.candidate, source=source, trainers=trainers, targets=targets, ks=ks, train_runs=train_runs,
                                           valid_runs=valid_runs, eval_runs=eval_runs, dev_cell=is_dev)), open(os.path.join(out, 'config.yaml'), 'w'))
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = 'cuda'; t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    subjects = trainers + targets; nT = len(trainers)
    D = RunData(c, subjects, train_runs, valid_runs, eval_runs, source)
    if cc['car_exclude_bad']:                                             # D-091 amendment 4: BIDS bad channels out of CAR and masked
        meta = json.load(open(os.path.join(ROOT, c['cache_dir'], 'eeg_meta.json')))
        D.invalid = {s: np.array(meta['subjects'][str(s)]['flat']) | np.array(meta['subjects'][str(s)]['bad']) for s in subjects}
    D.to_device(dev)
    print(f'{cc["arm"]} seed {seed} subject fold {K} run fold {F}: trainers {trainers} targets {targets} source {source}; windows train '
          f'{len(D.items["train"])} valid {len(D.items["valid"])} eval queries {len(D.items["eval"])} ({time.time() - t0:.0f}s)', flush=True)
    m = c['model']; net = BrainModule(out_dim=D.F.shape[0], n_subjects=len(subjects), merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                                      merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'],
                                      depth=m['depth'], subject_layers=True).to(dev)
    if a.init:                                                            # P2: pre-trained weights, fresh ds004408 subject layers
        sd = torch.load(a.init, map_location='cpu'); sd = {kk: v for kk, v in sd.items() if not kk.startswith('subject.')}
        missing, unexpected = net.load_state_dict(sd, strict=False); assert not unexpected and all(x.startswith('subject.') for x in missing), (missing, unexpected)
    amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=o['precision'] == 'bf16'); hist = []
    tr_items, va_items = D.items['train'], D.items['valid']; vr = np.random.default_rng(0)

    def fwd(items, si, shift=0.0):
        X, P, Y = D.batch(items, si, shift)
        with torch.autocast(**amp): return net(X, P, torch.as_tensor(np.asarray(si), device=dev)).float(), Y

    # 1. shared network on the trainers (same fixed validation batches as B1 for every seed and arm)
    opt = torch.optim.Adam(net.parameters(), lr=o['lr'])
    vb = [(vr.choice(len(va_items), o['batch_size'], replace=False), vr.integers(0, nT, o['batch_size'])) for _ in range(nvb)]

    def step():
        net.train(); tl = 0.0
        for _ in range(o['updates_per_epoch']):
            idx = rng.choice(len(tr_items), o['batch_size'], replace=False); si = rng.integers(0, nT, o['batch_size'])
            est, Y = fwd([tr_items[i] for i in idx], si); loss = clip_loss(est, Y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item()
        return tl / o['updates_per_epoch']

    def vloss():
        net.eval(); vl = 0.0
        with torch.no_grad():
            for idx, si in vb:
                est, Y = fwd([va_items[i] for i in idx], si); vl += clip_loss(est, Y).item()
        return vl / len(vb), copy.deepcopy({kk: v.detach().cpu() for kk, v in net.state_dict().items()})

    best, state = run_units(step, vloss, sh['eval_every'], sh['max_updates'], sh['min_updates'], sh['patience_units'], hist, 'shared')
    net.load_state_dict(state); torch.save(state, os.path.join(out, 'shared.pt')); t_shared = time.time() - t0
    hs = [h for h in hist if h['stage'] == 'shared']
    print(f'shared done: best valid {best:.4f} @ {min(hs, key=lambda h: h["valid"])["update"]} ({t_shared:.0f}s)', flush=True)
    net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    W = net.subject.weights; mean_layer = W[:nT].mean(0).clone()

    def rank_cells(si, items, shift=0.0, keep=None):
        """within-run ranking of `items` (word-onset queries) for subject index si; keep: dict to receive the score matrices"""
        run = np.array([x[0] for x in items]); cells = {}
        with torch.no_grad():
            for rr in sorted(set(run.tolist())):
                qi = [items[i] for i in np.where(run == rr)[0]]
                if shift:                                                   # PRE: queries (and candidates) whose earlier window is inside the run
                    qi = [x for x in qi if x[1] + c['window']['tmin'] + c['eeg']['offset_s'] + shift >= 0]
                Y = torch.stack([D.Fg[:, D.feat_idx(x, t) + D.ar] for x, t in qi]).float(); ch = c['eval']['score_chunk']
                est = torch.cat([fwd(qi[b0:b0 + ch], np.full(len(qi[b0:b0 + ch]), si), shift)[0] for b0 in range(0, len(qi), ch)])
                S = clip_scores(est, Y); rk = (S > torch.diag(S)[:, None]).sum(1).cpu().numpy(); n = len(qi)
                cells[str(rr + 1)] = dict(mrp=float(np.mean(1 - rk / (n - 1))), n=n, top1=float((rk < 1).mean()), top10=float((rk < 10).mean()))
                if keep is not None: keep[f'r{rr + 1}'] = S.cpu().numpy().astype(np.float16); keep[f'r{rr + 1}_onset'] = np.array([t for _, t in qi])
        return cells

    def pooled(cells): return float(np.average([v['mrp'] for v in cells.values()], weights=[v['n'] for v in cells.values()]))

    res, arrays, timing = {}, {}, {}
    # development cell: training subjects' validation-run MRP of the selected shared network (own training layers)
    if is_dev or a.smoke:
        vq = valid_queries(D, c, valid_runs); dev_res = {str(s): rank_cells(j, vq) for j, s in enumerate(trainers)}
        res['_dev_trainers_valid'] = dev_res
        print(f'dev cell: training-subject validation-run MRP {np.average([pooled(v) for v in dev_res.values()], weights=[sum(x["n"] for x in v.values()) for v in dev_res.values()]):.4f}', flush=True)

    # 2.-3. calibration curve and evaluation, target by target
    q = D.items['eval']; own_k14 = {}
    for j, s in enumerate(targets):
        ti = nT + j; o_s = res[str(s)] = {}; t1 = time.time()
        with torch.no_grad(): W[ti] = mean_layer
        keep = {} if 'k0' in cc['save_scores'] else None; o_s['k0'] = rank_cells(ti, q, keep=keep)
        if keep is not None: arrays.update({f'main_k0_s{s}_{kk}': v for kk, v in keep.items()})
        vrs = np.random.default_rng([cv['valid_seed'], s, F])
        cvb = [(vrs.choice(len(va_items), cal['batch_size'], replace=False), np.full(cal['batch_size'], ti)) for _ in range(nvb)]

        def cvloss():
            vl = 0.0
            with torch.no_grad():
                for idx, si in cvb:
                    est, Y = fwd([va_items[i] for i in idx], si); vl += clip_loss(est, Y).item()
            return vl / len(cvb), W[ti].detach().clone()
        for k in ks:
            t2 = time.time(); runs_k = draw_runs(cv['draw_seed'], s, F, train_runs, k)
            items_k = [x for x in tr_items if x[0] in runs_k]; crng = np.random.default_rng([cv['calib_seed'], s, F, k, seed])
            with torch.no_grad(): W[ti] = mean_layer
            W.requires_grad_(True); copt = torch.optim.Adam([W], lr=cal['lr']); gstar = {}; ep_count = [0]
            g_ep = cv['gstar_epoch'].get(k, cv['gstar_epoch'].get(str(k)))

            def cstep():
                tl = 0.0
                for _ in range(cal['updates_per_epoch']):
                    idx = crng.choice(len(items_k), cal['batch_size'], replace=False)
                    est, Y = fwd([items_k[i] for i in idx], np.full(cal['batch_size'], ti)); loss = clip_loss(est, Y)
                    copt.zero_grad(set_to_none=True); loss.backward(); copt.step(); tl += loss.item()
                if ep_count[0] == g_ep: gstar['layer'] = W[ti].detach().clone()   # G*: layer after epoch g_ep, no validation used
                ep_count[0] += 1
                return tl / cal['updates_per_epoch']
            init_vl, _ = cvloss(); tg = f'cal_{s}_k{k}'; hist.append(dict(stage=tg, epoch=-1, train=None, valid=init_vl, runs=[x + 1 for x in runs_k]))
            b, row = run_epochs(cstep, cvloss, cal['max_epochs'], cal['patience'], hist, tg)
            if init_vl <= b: b, row = init_vl, mean_layer.clone()           # keep the initial layer if fitting never helped (B1, D-090)
            W.requires_grad_(False)
            with torch.no_grad(): W[ti] = row
            ce = [h for h in hist if h['stage'] == tg and h['epoch'] >= 0]
            keep = {} if f'k{k}' in cc['save_scores'] else None; o_s[f'k{k}'] = rank_cells(ti, q, keep=keep)
            if keep is not None: arrays.update({f'main_k{k}_s{s}_{kk}': v for kk, v in keep.items()})
            if f'k{k}' in cc['save_pre']:
                keep = {}; o_s[f'pre_k{k}'] = rank_cells(ti, q, shift=-c['eval']['pre_shift_s'], keep=keep)
                arrays.update({f'pre_k{k}_s{s}_{kk}': v for kk, v in keep.items()})
            o_s[f'k{k}_meta'] = dict(runs=[x + 1 for x in runs_k], best_valid=b, init_valid=init_vl, kept_init=bool(init_vl <= min(h['valid'] for h in ce)),
                                     best_epoch=int(min(ce, key=lambda h: h['valid'])['epoch']), epochs_run=len(ce))
            if 'layer' in gstar:
                with torch.no_grad(): W[ti] = gstar['layer']
                o_s[f'k{k}_gstar'] = rank_cells(ti, q); o_s[f'k{k}_gstar_meta'] = dict(epoch=g_ep)
                with torch.no_grad(): W[ti] = row
            if k == 14: own_k14[s] = row.clone()
            timing[f'{s}_k{k}'] = time.time() - t2
            print(f'sub-{s:03d} k{k}: runs {[x + 1 for x in runs_k]} best valid {b:.4f} epoch {o_s[f"k{k}_meta"]["best_epoch"]} '
                  f'({timing[f"{s}_k{k}"]:.0f}s)' + (f' valid-run MRP {pooled(o_s[f"k{k}"]):.4f}' if a.smoke or a.timing else ''), flush=True)
        timing[f'{s}_total'] = time.time() - t1
    if cv.get('other_adapter') and 14 in ks and len(targets) > 1:          # descriptive: other targets' own k14 layers
        for j, s in enumerate(targets):
            ti = nT + j; res[str(s)]['other_adapter_k14'] = {}
            for o_ in targets:
                if o_ != s:
                    with torch.no_grad(): W[ti] = own_k14[o_]
                    res[str(s)]['other_adapter_k14'][str(o_)] = rank_cells(ti, q)
            with torch.no_grad(): W[ti] = own_k14[s]
    torch.save({str(s): own_k14[s].cpu() for s in own_k14}, os.path.join(out, 'target_layers_k14.pt'))
    json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(out, 'scores.npz'), **arrays)
    summ = dict(arm=cc['arm'], seed=seed, eval_runs=[r + 1 for r in eval_runs], eval_split='dev_only' if a.dev_only else 'valid' if a.smoke or a.timing else 'test',
                n_queries=len(q), shared_best_valid=best, shared_best_update=min(hs, key=lambda h: h['valid'])['update'], shared_updates=hs[-1]['update'],
                shared_seconds=t_shared, car_exclude_bad=cc['car_exclude_bad'], device=torch.cuda.get_device_name(), init=a.init,
                max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2 ** 30, seconds=time.time() - t0, calibration_seconds=timing)
    if summ['eval_split'] == 'test':          # D-097 blinding: test-run cells only in test_cells.json (MDE and verdict scripts)
        meta_only = {s: {kk: v for kk, v in r.items() if kk.endswith('_meta')} for s, r in res.items() if not s.startswith('_')}
        meta_only.update({s: r for s, r in res.items() if s.startswith('_')})
        json.dump(dict(summary=summ, targets=res), open(os.path.join(out, 'test_cells.json'), 'w'), indent=1)
        json.dump(dict(summary=summ, targets=meta_only), open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    else:
        json.dump(dict(summary=summ, targets=res), open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    print('SUMMARY', json.dumps(summ), flush=True)


def valid_queries(D, c, valid_runs):
    """word-onset queries of the validation runs (B1 query definition), for the development-cell report"""
    w, e = c['window'], c['eeg']; Wd = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True); out = []
    for r in valid_runs:
        hi = D.lens[r] / D.fs - w['tmax'] - max(e['offset_s'], 0); on = np.unique(Wd['onset'][Wd['run'] == r])
        out += [(r, float(t)) for t in on[(on >= D.lo) & (on <= hi)]]
    return out


if __name__ == '__main__':
    main()
