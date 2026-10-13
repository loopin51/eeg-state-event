"""Stage B subject-layer calibration curve (D-090): one subject fold x run fold of the B1 design.

Loads the B1 shared network of (subject fold K, run fold F) (trained without any data of the targets) and, for every target,
evaluates the within-run ranking on the test runs under these subject layers:
  k0_population   mean of the training subjects' layers (no target data)
  other_adapter   the B1 adapter (14 runs) of each other target of the same fold (one cell set per other target)
  k14_b1          the target's own B1 adapter (re-evaluation; must equal the B1 numbers)
  k1, k2, k4, k8  the target's layer refitted from the mean initialisation on the first k runs of a fixed permutation of its
                  14 training runs (nested draws), with the B1 calibration settings and early stopping on its 2 validation runs
Per (target, condition, test run) it stores the within-run MRP, top-1, top-10 and query count; scripts/stageB_calib_report.py
pools them.

usage: python scripts/stageB_calib_curve.py configs/stageB_calib.yaml --subject-fold K --run-fold F [--timing]
       --timing: first target only, k in timing.ks, evaluation on the validation runs (never the test runs)
"""
import os, sys, glob, json, time, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.brain_clip import BrainModule, clip_loss, clip_scores
from stageB_b1_train import RunData, run_epochs


def draw_runs(seed, subject, run_fold, train_runs, k):
    """nested draw shared with the DE curve: first k runs of a fixed permutation of the 14 training runs (0-based)"""
    return sorted(int(r) for r in np.random.default_rng([seed, subject, run_fold]).permutation(sorted(train_runs))[:k])


def b1_run_dir(b1, k, f):
    d = sorted(glob.glob(os.path.join(ROOT, b1['out_root'], f"*_stageB_b1_{b1['decoder_setting']}_k{k}_f{f}")))
    assert len(d) == 1, d; return d[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--subject-fold', type=int, required=True)
    ap.add_argument('--run-fold', type=int, required=True); ap.add_argument('--timing', action='store_true'); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); b1 = yaml.safe_load(open(os.path.join(ROOT, c['b1_config']))); K, F = a.subject_fold, a.run_fold
    d = b1_run_dir(b1, K, F); rc = yaml.safe_load(open(os.path.join(d, 'config.yaml'))); r = rc['run']
    assert not r['smoke'] and not r['phaserand']
    trainers, all_targets = r['trainers'], r['targets']; nT = len(trainers); cal = rc['calibration']; dev = 'cuda'; t0 = time.time()
    targets = all_targets[:1] if a.timing else all_targets; ks = c['timing']['ks'] if a.timing else c['ks']
    eval_runs = r['valid_runs'] if a.timing else r['eval_runs']
    D = RunData(rc, trainers + all_targets, r['train_runs'], r['valid_runs'], eval_runs, r['source']); D.to_device(dev)
    m = rc['model']; net = BrainModule(out_dim=D.F.shape[0], n_subjects=nT + len(all_targets), merger_channels=m['merger_channels'],
                                       pos_dim=m['pos_dim'], merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'],
                                       hidden=m['hidden'], depth=m['depth'], subject_layers=True).to(dev)
    net.load_state_dict(torch.load(os.path.join(d, 'shared.pt'))); net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    W = net.subject.weights; mean_layer = W[:nT].mean(0).clone(); b1_layers = torch.load(os.path.join(d, 'target_layers.pt')).to(dev)
    name = time.strftime('%Y-%m-%d') + f"_stageB_calib_{b1['decoder_setting']}_k{K}_f{F}{'_timing' if a.timing else ''}"
    out = os.path.join(ROOT, c['out_root'], name); os.makedirs(out, exist_ok=True)
    yaml.safe_dump(dict(c, run=dict(b1_run=os.path.basename(d), subject_fold=K, run_fold=F, timing=a.timing, targets=targets, ks=ks,
                                    eval_runs=eval_runs, eval_split='valid' if a.timing else 'test')), open(os.path.join(out, 'config.yaml'), 'w'))
    tr_items, va_items, q = D.items['train'], D.items['valid'], D.items['eval']; qrun = np.array([x[0] for x in q])

    def fwd(items, si):
        X, P, Y = D.batch(items, si); return net(X, P, torch.as_tensor(np.asarray(si), device=dev)).float(), Y

    def evaluate(si):
        cells = {}
        with torch.no_grad():
            for rr in sorted(set(qrun.tolist())):
                qi = [q[i] for i in np.where(qrun == rr)[0]]; Y = torch.stack([D.Fg[:, D.feat_idx(x, t) + D.ar] for x, t in qi]).float()
                est = torch.cat([fwd(qi[b0:b0 + rc['eval']['score_chunk']], np.full(len(qi[b0:b0 + rc['eval']['score_chunk']]), si))[0]
                                 for b0 in range(0, len(qi), rc['eval']['score_chunk'])])
                S = clip_scores(est, Y); rk = (S > torch.diag(S)[:, None]).sum(1).cpu().numpy(); n = len(qi)
                cells[str(rr + 1)] = dict(mrp=float(np.mean(1 - rk / (n - 1))), n=n, top1=float((rk < 1).mean()), top10=float((rk < 10).mean()))
        return cells

    def set_layer(ti, layer):
        with torch.no_grad(): W[ti] = layer

    res, hist, timing = {}, [], {}
    for j, s in enumerate(targets):
        ti = nT + j; out_s = res[str(s)] = {}
        set_layer(ti, mean_layer); out_s['k0_population'] = evaluate(ti)
        set_layer(ti, b1_layers[j]); out_s['k14_b1'] = evaluate(ti)
        out_s['other_adapter'] = {}
        for jo, o in enumerate(all_targets):
            if o != s: set_layer(ti, b1_layers[jo]); out_s['other_adapter'][str(o)] = evaluate(ti)
        vr = np.random.default_rng([c['valid_seed'], s, F])
        cvb = [(vr.choice(len(va_items), cal['batch_size'], replace=False), np.full(cal['batch_size'], ti)) for _ in range(20)]

        def cvloss():
            vl = 0.0
            with torch.no_grad():
                for idx, si in cvb:
                    est, Y = fwd([va_items[i] for i in idx], si); vl += clip_loss(est, Y).item()
            return vl / len(cvb), W[ti].detach().clone()
        for k in ks:
            t1 = time.time(); runs_k = draw_runs(c['draw_seed'], s, F, r['train_runs'], k)
            items_k = [x for x in tr_items if x[0] in runs_k]; rng = np.random.default_rng([c['calib_seed'], s, F, k])
            set_layer(ti, mean_layer); W.requires_grad_(True); copt = torch.optim.Adam([W], lr=cal['lr'])

            def cstep():
                tl = 0.0
                for _ in range(cal['updates_per_epoch']):
                    idx = rng.choice(len(items_k), cal['batch_size'], replace=False)
                    est, Y = fwd([items_k[i] for i in idx], np.full(cal['batch_size'], ti)); loss = clip_loss(est, Y)
                    copt.zero_grad(set_to_none=True); loss.backward(); copt.step(); tl += loss.item()
                return tl / cal['updates_per_epoch']
            init_vl, _ = cvloss(); tag = f'cal_{s}_k{k}'; hist.append(dict(stage=tag, epoch=-1, train=None, valid=init_vl, runs=[x + 1 for x in runs_k]))
            best, row = run_epochs(cstep, cvloss, cal['max_epochs'], cal['patience'], hist, tag)
            if init_vl <= best: best, row = init_vl, mean_layer.clone()          # keep the initial layer if fitting never helped (as in B1)
            W.requires_grad_(False); set_layer(ti, row)
            out_s[f'k{k}'] = evaluate(ti); out_s[f'k{k}_meta'] = dict(runs=[x + 1 for x in runs_k], best_valid=best, init_valid=init_vl)
            timing[f'{s}_k{k}'] = time.time() - t1
            mrp = np.average([v['mrp'] for v in out_s[f'k{k}'].values()], weights=[v['n'] for v in out_s[f'k{k}'].values()])
            print(f'sub-{s:03d} k{k}: runs {[x + 1 for x in runs_k]} best valid {best:.4f} ({timing[f"{s}_k{k}"]:.0f}s)'
                  + (f' valid-run MRP {mrp:.4f}' if a.timing else ''), flush=True)
    json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
    json.dump(dict(summary=dict(eval_split='valid' if a.timing else 'test', eval_runs=[x + 1 for x in eval_runs], seconds=time.time() - t0,
                                calibration_seconds=timing), targets=res), open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    print('SUMMARY', json.dumps(dict(seconds=time.time() - t0, calibration_seconds=timing)), flush=True)


if __name__ == '__main__':
    main()
