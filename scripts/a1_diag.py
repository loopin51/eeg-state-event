"""F-III S-diagnosis (D-104 decision 7 (a), D-105; descriptive, no alpha): re-score an existing model on the development cell.

The model folder is a C1 job of subject fold 0 x run fold 0 (P1 s{seed}_k0_f0, or a P2 --dev-only run); only its config.yaml,
shared.pt and metrics.json are read (never test_cells.json or scores.npz). Evaluation is on the development cell only: the
13 training subjects x validation runs 5 and 13 (word-onset queries), with two subject layers:
  own    : the subject's own training layer (the C1 development metric layer);
  pseudo : pseudo-target k2 (configs/a1.yaml pseudo_target): the layer restarts from the mean of the other training subjects'
           layers and is refitted on 2 of the subject's own training runs for 200 updates without validation.
For each (layer, subject, run): MRP with the 30-s exclusion (configs/a1.yaml queries) at delta 0 without search, under J250
(per-subject draw) without search and with every LS rule, and (own layer only) on the delta grid (fixed and every LS rule); plus the C1
definition (delta 0, exclusion 0, all validation queries) as a reproduction check against metrics.json. The validation CLIP
loss on the B1 fixed validation batches is decomposed by negative-pair composition (own layers).

usage: python scripts/a1_diag.py configs/a1.yaml --model DIR [--smoke] [--equiv]
       --smoke : 2 subjects, 1 run; prints the reproduction checks, shapes and timing only (no development value)
       --equiv : equivariance check of the one-pass extended-window forward (trained and randomly initialised network):
                 correlation and max |difference| of the 11 cropped-lag scores vs per-window forwards (no MRP)
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys, json, time, hashlib, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.brain_clip import BrainModule, clip_scores
from stageB_b1_train import RunData
from c1_train import draw_runs, valid_queries
import a1_lib as L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--model', required=True)
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--equiv', action='store_true'); a = ap.parse_args()
    A = yaml.safe_load(open(os.path.join(ROOT, a.config) if not os.path.isabs(a.config) else a.config)); t0 = time.time(); dev = 'cuda'
    md = a.model.rstrip('/'); rc = yaml.safe_load(open(os.path.join(md, 'config.yaml'))); r = rc['run']; cc = rc['c1']
    assert (r['subject_fold'], r['run_fold']) == (A['dev_cell']['subject_fold'], A['dev_cell']['run_fold']) and not r['phaserand'], 'development cell only'
    trainers, F, seed, train_runs, valid_runs = r['trainers'], r['run_fold'], r['seed'], r['train_runs'], r['valid_runs']; nT = len(trainers)
    rc['eeg']['reference'] = rc['settings'][rc['decoder_setting']]['reference']
    D = RunData(rc, trainers, train_runs, valid_runs, valid_runs, r['source'])            # eval = validation runs; test runs never listed
    if cc['car_exclude_bad']:
        meta = json.load(open(os.path.join(ROOT, rc['cache_dir'], 'eeg_meta.json')))
        D.invalid = {s: np.array(meta['subjects'][str(s)]['flat']) | np.array(meta['subjects'][str(s)]['bad']) for s in trainers}
    D.to_device(dev)
    m = rc['model']; mk = lambda: BrainModule(out_dim=D.F.shape[0], n_subjects=nT + len(r['targets']), merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                                              merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'], depth=m['depth'], subject_layers=True).to(dev)
    net = mk(); sp = os.path.join(md, 'shared.pt'); net.load_state_dict(torch.load(sp, map_location=dev)); net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=rc['optim']['precision'] == 'bf16')
    M = json.load(open(os.path.join(md, 'metrics.json'))); stored = M['targets']['_dev_trainers_valid']
    lag_list = L.lags(A); rules = A['lag_search']['rules']; excl = A['queries']['exclusion_s']; vq = valid_queries(D, rc, valid_runs)
    out = dict(summary=dict(model=os.path.basename(md), arm=cc['arm'], seed=seed, init=M['summary'].get('init'), shared_sha256=hashlib.sha256(open(sp, 'rb').read()).hexdigest(),
                            device=torch.cuda.get_device_name(), lags=lag_list, rules=rules, exclusion_s=excl), checks={}, cells={})
    subjects = trainers[:2] if a.smoke else trainers; runs = valid_runs[:1] if a.smoke else valid_runs

    if a.equiv:                                                            # one-pass lag crops vs per-window forwards (eval mode)
        res = {}
        for tag, model in (('trained', net), ('random', None)):
            if model is None:
                torch.manual_seed(0); model = mk(); model.eval()
            j = 0; s = trainers[j]; items = [x for x in vq if x[0] == runs[0]]; items = [items[k] for k in L.inside(D, items, A['queries']['margin_s'])][:256]
            si = np.full(len(items), j); base = L.j_offsets(A, s, items); cand = L.targets(D, items)
            Sw = L.lag_stack(model, D, items, si, base, lag_list, cand, amp)
            Se = []
            for b in range(0, len(items), 32):                             # crops of one chunk -> scores, so the crops are never all held
                E = L.ext_forward(model, D, items[b:b + 32], si[b:b + 32], base[b:b + 32], lag_list, amp)
                Se.append(torch.stack([clip_scores(E[li], cand) for li in range(len(lag_list))])); del E
            Se = torch.cat(Se, 1)
            x, y = Sw.flatten().double(), Se.flatten().double(); corr = float(torch.corrcoef(torch.stack([x, y]))[0, 1])
            per_lag = [float(torch.corrcoef(torch.stack([Sw[li].flatten().double(), Se[li].flatten().double()]))[0, 1]) for li in range(len(lag_list))]
            res[tag] = dict(n_queries=len(items), corr=corr, corr_per_lag=per_lag, max_abs_diff=float((x - y).abs().max()), score_sd=float(x.std()),
                            rel_max_abs_diff=float((x - y).abs().max() / x.std()), pass_min_corr=bool(corr >= A['equivariance']['min_corr']))
        out['equivariance'] = res; print('EQUIV', json.dumps(res), flush=True); return

    W = net.subject.weights; own = W[:nT].detach().clone()
    # reproduction check: C1 development-metric definition (own layer, delta 0, exclusion 0, all validation queries)
    rep = []
    for j, s in enumerate(subjects):
        for rr in runs:
            items = [x for x in vq if x[0] == rr]; si = np.full(len(items), j); cand = L.targets(D, items)
            S = clip_scores(L.forward(net, D, items, si, np.zeros(len(items)), amp), cand); v = L.mrp_excl(S, [t for _, t in items], 0)
            rep.append(abs(v - stored[str(s)][str(rr + 1)]['mrp'])); assert len(items) == stored[str(s)][str(rr + 1)]['n']
    out['checks']['c1_dev_metric_max_abs_diff'] = max(rep); print(f'check C1 dev metric reproduction: max |diff| {max(rep):.2e} over {len(rep)} cells', flush=True)

    for j, s in enumerate(subjects):
        layers = {'own': own[j]}
        others = [k for k in range(nT) if k != j]; mean_others = own[others].mean(0); torch.cuda.empty_cache()
        pt = A['pseudo_target']; runs_k = draw_runs(pt['draw_seed'], s, F, train_runs, pt['k'])
        layers['pseudo'] = L.refit_layer(net, D, j, mean_others, runs_k, D.items['train'], pt, np.random.default_rng([pt['calib_seed'], s, F, pt['k'], seed]), amp)
        for ln, layer in layers.items():
            with torch.no_grad(): W[j] = layer
            for rr in runs:
                allq = [x for x in vq if x[0] == rr]; key = f'{ln}_s{s}_r{rr + 1}'; cell = {}
                for sub, margin in (('main', A['queries']['margin_s']),) + ((('grid', A['queries']['report_margin_s']),) if ln == 'own' else ()):
                    items = [allq[k] for k in L.inside(D, allq, margin)]; si = np.full(len(items), j); tt = [t for _, t in items]; cand = L.targets(D, items)
                    cell[f'n_{sub}'] = len(items)
                    if sub == 'main':
                        S0 = clip_scores(L.forward(net, D, items, si, np.zeros(len(items)), amp), cand)
                        cell['d0_fixed'] = L.mrp_excl(S0, tt, excl); cell['d0_fixed_excl0'] = L.mrp_excl(S0, tt, 0)
                        base = L.j_offsets(A, s, items); SL = L.lag_stack(net, D, items, si, base, lag_list, cand, amp)
                        cell['j250_fixed'] = L.mrp_excl(SL[lag_list.index(0.0)], tt, excl)
                        for rule in rules: cell[f'j250_{rule}'] = L.mrp_excl(L.aggregate(SL, rule), tt, excl)
                        del SL
                    else:
                        offs = sorted({round(d + l, 2) for d in A['delta_grid_s'] for l in lag_list}); So = {}
                        for o in offs: So[o] = clip_scores(L.forward(net, D, items, si, np.full(len(items), o), amp), cand)
                        for d in A['delta_grid_s']:
                            cell[f'grid{d:+.2f}_fixed'] = L.mrp_excl(So[round(d, 2)], tt, excl)
                            SL = torch.stack([So[round(d + l, 2)] for l in lag_list])
                            for rule in rules: cell[f'grid{d:+.2f}_{rule}'] = L.mrp_excl(L.aggregate(SL, rule), tt, excl)
                        del So
                out['cells'][key] = cell; torch.cuda.empty_cache()
            print(f'sub-{s:03d} {ln} done ({time.time() - t0:.0f}s)', flush=True)
        with torch.no_grad(): W[j] = own[j]

    # negative-pair decomposition of the validation CLIP loss (own layers, B1 fixed validation batches as in c1_train)
    va = D.items['valid']; vr = np.random.default_rng(0); bs = rc['optim']['batch_size']; nvb = 2 if a.smoke else 20
    vb = [(vr.choice(len(va), bs, replace=False), vr.integers(0, nT, bs)) for _ in range(20)][:nvb]
    edges = A['negatives']['classes_s']; names = ['same_run_lt3', 'same_run_3to30', 'same_run_ge30', 'other_run']; acc = {k: [] for k in ['loss'] + [f'{n}_{q}' for n in names for q in ('count', 'mass', 'loss_without')]}
    with torch.no_grad():
        for idx, si in vb:
            it = [va[i] for i in idx]; X, P, Y = D.batch(it, si)
            with torch.autocast(**amp): est = net(X, P, torch.as_tensor(si, device=dev)).float()
            S = clip_scores(est, Y).double(); n = len(it); eye = torch.eye(n, dtype=torch.bool, device=dev)
            run = torch.as_tensor([x[0] for x in it], device=dev); tt = torch.as_tensor([x[1] for x in it], device=dev, dtype=torch.float64)
            same = run[:, None] == run[None]; dt = (tt[:, None] - tt[None]).abs()
            cls = {'same_run_lt3': same & (dt < edges[0]) & ~eye, 'same_run_3to30': same & (dt >= edges[0]) & (dt < edges[1]),
                   'same_run_ge30': same & (dt >= edges[1]), 'other_run': ~same}
            lse = torch.logsumexp(S, 1); loss = (lse - S.diagonal()); acc['loss'].append(float(loss.mean())); pr = torch.softmax(S, 1)
            for k, msk in cls.items():
                acc[f'{k}_count'].append(float(msk.sum(1).double().mean())); acc[f'{k}_mass'].append(float((pr * msk).sum(1).mean()))
                acc[f'{k}_loss_without'].append(float((torch.logsumexp(S.masked_fill(msk, float('-inf')), 1) - S.diagonal()).mean()))
    out['negatives'] = {k: float(np.mean(v)) for k, v in acc.items()}
    out['checks']['valid_loss_vs_shared_best'] = abs(out['negatives']['loss'] - M['summary']['shared_best_valid']) if nvb == 20 else None
    print(f'check validation CLIP loss reproduction: |diff| {out["checks"]["valid_loss_vs_shared_best"]}', flush=True)
    out['summary'].update(seconds=time.time() - t0, max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2 ** 30, smoke=a.smoke)
    name = time.strftime('%Y-%m-%d') + f"_a1_diag_{os.path.basename(md)}" + ('_smoke' if a.smoke else ''); od = os.path.join(ROOT, 'results/runs', name); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, 'metrics.json'), 'w'), indent=1); yaml.safe_dump(dict(a1=A, model=md), open(os.path.join(od, 'config.yaml'), 'w'))
    if a.smoke: print('SMOKE shapes', {k: {kk: (vv if kk.startswith('n_') else '-') for kk, vv in v.items()} for k, v in out['cells'].items()}, flush=True)
    print('SUMMARY', json.dumps(out['summary']), flush=True)


if __name__ == '__main__':
    main()
