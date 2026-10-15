"""F-III A1 development on Alice (D-105 section 9 (b); development data only, D-092 values already open).

One arm (P1, or N with one candidate) and one seed on the D-092 split, subject fold 0 x segment fold 0: the shared network is
trained on the training segments (8) of the 26 subjects outside subject fold 0 with the D-094 procedure (150-update units,
patience 4, not before 1,200 updates; configs/alice_train.yaml main) and the arm's objective (configs/a1_p1.yaml /
configs/a1_n.yaml candidates, as scripts/a1_train.py); each of the 7 targets refits its layer from the trainers' mean on 6 of
its 8 training segments (first 6 of default_rng([20261014, s, 0]).permutation, about 6 min = the ds004408 k2 time) with the
D-090 calibration (200 updates per epoch, at most 20 epochs, patience 3, early stop on its validation segment, initial layer
kept if never better). Evaluation: test segments 1-3 joined as one "run" (concatenated time): word-onset queries whose EEG
windows moved by up to 0.5 s stay inside their segment, candidates >= 30 s away, J250 offsets default_rng([20261105, s, 1000])
over the sorted test onsets, delta 0 / J250 fixed / J250 + every LS rule. Output: metrics.json per target (query count, MRP).

usage: python scripts/a1_alice.py configs/a1_p1.yaml|configs/a1_n.yaml --seed S [--candidate NAME] [--smoke | --blind]
       --blind: the full run with the MRP values left out of metrics.json (time and memory measurement before registration)
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys, json, time, copy, argparse, yaml, numpy as np, torch
import torch.nn.functional as Fn
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.brain_clip import clip_loss, clip_scores, INVALID
from alice_train import Alice, build_net, pairs_for, fixed_batches
from stageB_b2_train import run_units
from stageB_b1_train import run_epochs
import a1_lib as L


def batch_shift(D, items, si, shifts):
    """Alice.batch (Défossez arm) with a per-item EEG shift; targets at the true times"""
    w, p = D.c['window'], D.pre; X = []
    for (r, t), s_i, sh in zip(items, si, shifts):
        j = min(max(int(round((t + w['tmin'] + w['offset_s'] + sh) * D.fs)), 0), int(D.durs[r] * D.fs) - D.T)   # clamped to the segment
        X.append(D.eeg[D.subjects[s_i]][:, D.offs[r] + j:D.offs[r] + j + D.T])
    X = torch.from_numpy(np.stack(X).astype(np.float32)).to(D.dev, non_blocking=True); S = torch.as_tensor(np.asarray(si), device=D.dev); bad = D.badg[S]
    ok = (~bad).float()[..., None]; X = X - (X * ok).sum(1, keepdim=True) / ok.sum(1, keepdim=True)
    X = X - X[..., : int(round(p['baseline_s'] * D.fs))].mean(-1, keepdim=True); X = X.clamp(-p['clamp'], p['clamp'])
    return X, D.posg[None].expand(len(S), -1, -1).masked_fill(bad[..., None], INVALID)


def targets_shift(D, items, shift):
    w = D.c['window']; ok, fi = [], []
    for r, t in items:
        j = D.foffs[r] + int(round((t + shift + w['tmin']) * D.ffs)); good = t + shift + w['tmin'] >= 0 and j + D.Tf <= D.foffs[r] + D.flens[r]
        ok.append(good); fi.append(j if good else D.feat_idx(r, t))
    fi = torch.as_tensor(fi, device=D.dev); return D.Fg[:, fi[:, None] + D.ar].permute(1, 0, 2).float(), torch.as_tensor(ok, device=D.dev)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--candidate')
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--blind', action='store_true'); a = ap.parse_args()   # --blind: full run, MRP values not written (timing)
    cc = yaml.safe_load(open(a.config)); A = yaml.safe_load(open(os.path.join(ROOT, cc['a1_config']))); c = yaml.safe_load(open(os.path.join(ROOT, 'configs/alice_train.yaml')))
    if cc['arm'] == 'N':
        cand = a.candidate or cc['candidate']; assert cand in cc['candidates']; obj = dict(cc['candidates'][cand], name=cand)
    else:
        assert a.candidate is None; cand = None; obj = dict(name='clip', jitter_s=0.0, lse=False)
    seed = a.seed; ac = dict(c['arms']['defossez']); o = dict(c['main']); cal = dict(batch_size=256, lr=3e-4, updates_per_epoch=200, max_epochs=20, patience=3)
    sp = json.load(open(os.path.join(ROOT, c['cache_dir'], 'split.json')))['main']; K, F = 0, 0
    test = [x - 1 for x in sp['segment_folds'][F]]; valid = sp['valid_segment'][F] - 1; train = [r for r in range(12) if r not in test + [valid]]
    targets = list(sp['subject_folds'][K]); trainers = sorted(s for j, fs in enumerate(sp['subject_folds']) if j != K for s in fs)
    if a.smoke:
        trainers, targets = trainers[:3], targets[:2]; o.update(updates_per_epoch=10, max_epochs=2, batch_size=32, valid_batches=2, min_epochs=0); cal.update(updates_per_epoch=5, max_epochs=2, batch_size=32)
    subjects = trainers + targets; nT = len(trainers); dev = 'cuda'; t0 = time.time()
    D = Alice(c, ac['source'], c['features']['main'], subjects, 'defossez', ac)
    def gitems(segs):
        it = [x for r in segs for x in D.grid(r)]; okg = D.guard_ok(it, test); return [x for x, g in zip(it, okg) if g]
    tr_items, va_items = gitems(train), gitems([valid]); D.feat_norm([(r, 0.0, D.durs[r]) for r in train]); D.to_device(dev)
    name = time.strftime('%Y-%m-%d') + f"_a1_alice_{cc['arm']}{'-' + cand if cand else ''}_s{seed}" + ('_smoke' if a.smoke else '_blind' if a.blind else '')
    out = os.path.join(ROOT, 'results/runs', name); os.makedirs(out, exist_ok=True)
    yaml.safe_dump(dict(alice=c, a1=A, arm_config=cc, run=dict(arm=cc['arm'], candidate=cand, objective=obj, seed=seed, trainers=trainers, targets=targets,
                                                             train_segments=[r + 1 for r in train], valid_segment=valid + 1, test_segments=[r + 1 for r in test], smoke=a.smoke)),
                   open(os.path.join(out, 'config.yaml'), 'w'))
    torch.manual_seed(seed); rng = np.random.default_rng(seed); jr = np.random.default_rng([seed, 20261105, 1]); vr = np.random.default_rng(0)
    net = build_net(c, 'defossez', len(subjects), D, dev); amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=o['precision'] == 'bf16')
    lag_list = L.lags(A); mask_stat = dict(masked=0, total=0)

    def objective(items, si, jit):
        sit = torch.as_tensor(np.asarray(si), device=dev)
        if obj['name'] == 'clip':
            X, P, Y = D.batch(items, si)
            with torch.autocast(**amp): est = net(X, P, sit).float()
            return clip_loss(est, Y)
        X, P = batch_shift(D, items, si, jit)
        with torch.autocast(**amp): est = net(X, P, sit).float()
        if not obj['lse']: return clip_loss(est, D.targets(items))
        Sm, cnt = [], 0
        for l in lag_list:
            Yl, okl = targets_shift(D, items, -l); Sm.append(clip_scores(est, Yl).masked_fill(~okl[None, :], float('-inf'))); cnt = cnt + okl.float()
        mask_stat['masked'] += int(len(lag_list) * len(items) - cnt.sum().item()); mask_stat['total'] += len(lag_list) * len(items)
        assert bool((cnt >= 1).all()), 'a candidate without any lag inside its segment'
        return Fn.cross_entropy(torch.logsumexp(torch.stack(Sm), 0) - torch.log(cnt)[None, :], torch.arange(len(items), device=dev))

    def jitter(gen, n): return gen.uniform(-obj['jitter_s'], obj['jitter_s'], n) if obj['jitter_s'] > 0 else np.zeros(n)
    trp, vap = pairs_for(D, tr_items, range(nT)), pairs_for(D, va_items, range(nT))
    vb = fixed_batches(vap, o['batch_size'], o['valid_batches'], vr); vj = np.random.default_rng([0, 20261105]); vbj = [jitter(vj, len(pr)) for pr in vb]
    opt = torch.optim.Adam(net.parameters(), lr=ac['lr']); hist = []

    def step():
        net.train(); tl = 0.0
        for _ in range(o['updates_per_epoch']):
            pr = trp[rng.choice(len(trp), o['batch_size'], replace=len(trp) < o['batch_size'])]
            loss = objective([tr_items[i] for i in pr[:, 0]], pr[:, 1], None if obj['name'] == 'clip' else jitter(jr, len(pr)))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item()
        return tl / o['updates_per_epoch']

    def vloss():
        net.eval(); vl = 0.0
        with torch.no_grad():
            for pr, jj in zip(vb, vbj): vl += objective([va_items[i] for i in pr[:, 0]], pr[:, 1], None if obj['name'] == 'clip' else jj).item()
        return vl / len(vb), copy.deepcopy({kk: v.detach().cpu() for kk, v in net.state_dict().items()})
    min_up = o['min_epochs'] * o['updates_per_epoch']; max_up = o['max_epochs'] * o['updates_per_epoch']
    best, state = run_units(step, vloss, o['updates_per_epoch'], max_up, min_up, o['patience'], hist, 'shared')
    net.load_state_dict(state); del state; torch.save(net.state_dict(), os.path.join(out, 'shared.pt'))
    hs = [h for h in hist if h['stage'] == 'shared']; print(f'shared done: best valid {best:.4f} @ {min(hs, key=lambda h: h["valid"])["update"]} ({time.time() - t0:.0f}s)', flush=True)
    net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    W = net.subject.weights; mean_layer = W[:nT].mean(0).clone(); excl = A['queries']['exclusion_s']; rules = A['lag_search']['rules']; res = {}; layers = {}
    q_all = [x for r in test for x in D.onsets(r)]
    for j, s in enumerate(targets):
        ti = nT + j; t1 = time.time(); v_seg = valid
        if valid in D.missing[s]: v_seg = next(r for r in [(valid + i) % 12 for i in range(1, 12)] if r not in test and r not in D.missing[s])
        segs_k = [int(r) for r in np.random.default_rng([20261014, s, 0]).permutation([r for r in train if r != v_seg and r not in D.missing[s]])[:6]]
        items_k = [x for x in tr_items if x[0] in segs_k]; own_va = gitems([v_seg]); crng = np.random.default_rng([20261016, s, 0, 2, seed]); cjr = np.random.default_rng([20261016, s, 0, 2, seed, 1])
        cvr = np.random.default_rng([20261015, s, 0]); cvb = [cvr.choice(len(own_va), min(cal['batch_size'], len(own_va)), replace=False) for _ in range(o['valid_batches'])]
        cvjg = np.random.default_rng([20261015, s, 0, 1]); cvj = [jitter(cvjg, len(b)) for b in cvb]
        with torch.no_grad(): W[ti] = mean_layer
        W.requires_grad_(True); copt = torch.optim.Adam([W], lr=cal['lr'])

        def cvloss():
            with torch.no_grad(): vl = sum(objective([own_va[i] for i in b], np.full(len(b), ti), None if obj['name'] == 'clip' else jj).item() for b, jj in zip(cvb, cvj)) / len(cvb)
            return vl, W[ti].detach().clone()

        def cstep():
            tl = 0.0
            for _ in range(cal['updates_per_epoch']):
                idx = crng.choice(len(items_k), cal['batch_size'], replace=len(items_k) < cal['batch_size'])
                loss = objective([items_k[i] for i in idx], np.full(len(idx), ti), None if obj['name'] == 'clip' else jitter(cjr, len(idx)))
                copt.zero_grad(set_to_none=True); loss.backward(); copt.step(); tl += loss.item()
            return tl / cal['updates_per_epoch']
        init_vl, _ = cvloss(); b, row = run_epochs(cstep, cvloss, cal['max_epochs'], cal['patience'], hist, f'cal_{s}')
        if init_vl <= b: b, row = init_vl, mean_layer.clone()
        W.requires_grad_(False)
        with torch.no_grad(): W[ti] = row
        layers[str(s)] = row.cpu().clone()
        # evaluation: test segments joined (concatenated time) as one run
        qs = [x for x in q_all if x[0] not in D.missing[s]]; w = c['window']
        qs = [(r, t) for r, t in qs if t + w['tmin'] + w['offset_s'] - A['queries']['margin_s'] >= 0 and t + w['tmax'] + w['offset_s'] + A['queries']['margin_s'] <= D.durs[r] - 1.0 / D.fs]
        gt = np.array([D.gstart[r] + t for r, t in qs]); order = np.argsort(gt); qs = [qs[i] for i in order]; gt = gt[order]
        allon = np.sort(np.array([D.gstart[r] + t for r, t in q_all])); dd = dict(zip(np.round(allon, 6).tolist(), np.random.default_rng([A['j250']['seed'], s, 1000]).uniform(-A['j250']['width_s'], A['j250']['width_s'], len(allon)).tolist()))
        delta = np.array([dd[round(float(g), 6)] for g in gt]); sis = np.full(len(qs), ti); Yc = D.targets(qs)

        def fwd_shift(shifts):
            outs = []
            with torch.no_grad():
                for b0 in range(0, len(qs), 128):
                    X, P = batch_shift(D, qs[b0:b0 + 128], sis[b0:b0 + 128], shifts[b0:b0 + 128])
                    with torch.autocast(**amp): outs.append(net(X, P, torch.as_tensor(sis[b0:b0 + 128], device=dev)).float())
            return torch.cat(outs)
        S0 = clip_scores(fwd_shift(np.zeros(len(qs))), Yc); cell = dict(n=len(qs), calibration_segments=[r + 1 for r in segs_k], valid_segment=v_seg + 1, best_valid=b, init_valid=init_vl)
        cell['d0_fixed'] = L.mrp_excl(S0, gt, excl); SL = torch.stack([clip_scores(fwd_shift(delta + l), Yc) for l in lag_list])
        cell['j250_fixed'] = L.mrp_excl(SL[lag_list.index(0.0)], gt, excl)
        for rl in rules: cell[f'j250_{rl}'] = L.mrp_excl(L.aggregate(SL, rl), gt, excl)
        res[str(s)] = cell if not (a.smoke or a.blind) else {kk: (v if kk in ('n', 'calibration_segments', 'valid_segment') else '-') for kk, v in cell.items()}
        del SL, S0, Yc; torch.cuda.empty_cache(); print(f'target S{s:02d}: n {len(qs)} ({time.time() - t1:.0f}s)', flush=True)
    torch.save(layers, os.path.join(out, 'target_layers.pt')); json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    summ = dict(arm=cc['arm'], candidate=cand, seed=seed, shared_best_valid=best, shared_best_update=min(hs, key=lambda h: h['valid'])['update'], shared_updates=hs[-1]['update'],
                device=torch.cuda.get_device_name(), lse_masked_fraction=(mask_stat['masked'] / max(1, mask_stat['total']) if obj['lse'] else None), seconds=time.time() - t0, max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2 ** 30, smoke=a.smoke)
    def clean(x, where='metrics'):                                           # torch / numpy scalars -> Python; anything larger is an error
        if isinstance(x, dict): return {str(k): clean(v, f'{where}.{k}') for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v, f'{where}[{i}]') for i, v in enumerate(x)]
        if hasattr(x, 'numel') or hasattr(x, 'size') and not isinstance(x, (int, float, str)):
            assert getattr(x, 'numel', lambda: x.size)() == 1, f'{where}: array of {getattr(x, "shape", None)} in the output'; return x.item()
        return x
    out_json = clean(dict(summary=summ, targets=res)); json.dump(out_json, open(os.path.join(out, 'metrics.json'), 'w'), indent=1); print('SUMMARY', json.dumps(out_json['summary']), flush=True)


if __name__ == '__main__':
    main()
