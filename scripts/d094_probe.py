"""D-094 stage 2: validation-run probe of the B1 shared network inside its first 1,200 updates.

One B1 split (subject fold, run fold from the config) is trained exactly as scripts/stageB_b1_train.py step 1 (same seed,
same random draws, same fixed validation batches), but every `eval_every` updates (and before training) the script records
  - the mean training CLIP loss of the last interval,
  - the CLIP loss on B1's 20 fixed validation batches (training subjects, validation runs),
  - the within-run mean rank percentile (B1 definition: queries at word onsets, candidates = queries of the same run) on the
    validation runs for every training subject, and for every target subject with the uncalibrated mean subject layer
    (report only; never used to choose a candidate, D-094),
and once, the fraction of validation query windows that hit the clamp on a channel the model uses, per subject.
The test runs are never read (the query set is built from the validation runs). No checkpoint is written.

Experiments (configs/d094_probe.yaml): lr, exclude_bad (CAR and spatial attention also ignore the BIDS channels.tsv
`bad` marks, stored by stageA_prepare.py as eeg_meta.json `bad`; `flat` = IQR 0 is always excluded), phaserand source.

usage: python scripts/d094_probe.py configs/d094_probe.yaml --exp E0 [--smoke]
"""
import os, sys, json, time, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.brain_clip import BrainModule, clip_loss, clip_scores, INVALID
from stageB_b1_train import RunData


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--exp', required=True); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); P = yaml.safe_load(open(a.config)); X = dict(P['experiments'][a.exp])
    c = yaml.safe_load(open(os.path.join(ROOT, P['base_config']))); sp = c['split']; k, f = P['subject_fold'], P['run_fold']
    st = c['settings'][P['decoder_setting']]; c['eeg']['reference'] = st['reference']
    source = st['phaserand_source' if X.get('phaserand', False) else 'source']
    test_runs = [r - 1 for r in sp['run_folds'][f]]; valid_runs = [r - 1 for r in sp['valid_runs'][f]]
    train_runs = [r for r in range(20) if r not in test_runs + valid_runs]
    targets = sp['subject_folds'][k] + sp.get('extra_targets', {}).get(k, [])
    trainers = sorted(s for j, fs in enumerate(sp['subject_folds']) if j != k for s in fs)
    o = dict(c['optim'], lr=X['lr']); total, every = P['total_updates'], P['eval_every']
    if a.smoke:
        trainers = trainers[:3]; targets = targets[:2]; o.update(batch_size=32); total, every = 20, 10
    torch.manual_seed(c['seed']); rng = np.random.default_rng(c['seed']); dev = 'cuda'; t0 = time.time()
    subjects = trainers + targets; nT = len(trainers)
    D = RunData(c, subjects, train_runs, valid_runs, valid_runs, source)          # eval queries from the validation runs only
    if X.get('exclude_bad', False):
        meta = json.load(open(os.path.join(ROOT, c['cache_dir'], 'eeg_meta.json')))
        D.invalid = {s: np.array(meta['subjects'][str(s)]['flat']) | np.array(meta['subjects'][str(s)]['bad']) for s in subjects}
    D.to_device(dev)
    q = D.items['eval']; run = np.array([r for r, _ in q]); runs_v = sorted(set(run))
    print(f'{a.exp}: subject fold {k} run fold {f} trainers {trainers} targets {targets} source {source} lr {o["lr"]} '
          f'exclude_bad {X.get("exclude_bad", False)}; windows train {len(D.items["train"])} valid {len(D.items["valid"])} '
          f'valid queries {len(q)} ({time.time() - t0:.0f}s)', flush=True)
    m = c['model']; net = BrainModule(out_dim=D.F.shape[0], n_subjects=len(subjects), merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                                      merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'],
                                      depth=m['depth'], subject_layers=True).to(dev)
    tr_items, va_items = D.items['train'], D.items['valid']; vr = np.random.default_rng(0)

    def fwd(items, si):
        Xb, Pb, Y = D.batch(items, si); return net(Xb, Pb, torch.as_tensor(np.asarray(si), device=dev)).float(), Y

    opt = torch.optim.Adam(net.parameters(), lr=o['lr'])
    vb = [(vr.choice(len(va_items), o['batch_size'], replace=False), vr.integers(0, nT, o['batch_size'])) for _ in range(2 if a.smoke else 20)]
    Yc = torch.stack([D.Fg[:, D.feat_idx(r, t) + D.ar] for r, t in q])
    ch = c['eval']['score_chunk']

    # clamp fraction per subject on the validation query windows (depends on the reference only, not on training)
    clamp = {}
    with torch.no_grad():
        for j, s in enumerate(subjects):
            hit = 0
            for b0 in range(0, len(q), ch):
                Xb, Pb, _ = D.batch(q[b0:b0 + ch], np.full(len(q[b0:b0 + ch]), j)); used = ~(Pb == INVALID).all(-1)   # channels the model sees
                hit += int(((Xb.abs() >= c['eeg']['clamp']) & used[..., None]).flatten(1).any(1).sum())
            clamp[str(s)] = hit / len(q)

    def evaluate():
        net.eval(); out = {}
        with torch.no_grad():
            vl = float(np.mean([clip_loss(*fwd([va_items[i] for i in idx], si)).item() for idx, si in vb]))
            W = net.subject.weights; W[nT:] = W[:nT].mean(0, keepdim=True)        # uncalibrated mean layer for the targets
            for j, s in enumerate(subjects):
                rk = []
                for r in runs_v:                                                   # within-run scores, chunked (memory)
                    ir = np.where(run == r)[0]; S = torch.empty(len(ir), len(ir), device=dev)
                    for b0 in range(0, len(ir), ch):
                        ii = ir[b0:b0 + ch]; est, _ = fwd([q[i] for i in ii], np.full(len(ii), j))
                        S[b0:b0 + len(ii)] = torch.cat([clip_scores(est, Yc[ir[cc:cc + 256]].float()) for cc in range(0, len(ir), 256)], 1)
                    rk.append(((S > S.diag()[:, None]).sum(1).float() / (len(ir) - 1)).cpu().numpy())
                out[str(s)] = float(np.mean(1 - np.concatenate(rk)))
        tr = [out[str(s)] for s in trainers]; tg = [out[str(s)] for s in targets]
        return dict(valid_loss=vl, mrp_trainers=float(np.mean(tr)), mrp_targets_uncalibrated=float(np.mean(tg)), mrp=out)

    log = [dict(update=0, train_loss=None, **evaluate(), seconds=time.time() - t0)]
    print(f"update 0: valid {log[-1]['valid_loss']:.4f} mrp trainers {log[-1]['mrp_trainers']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    tl, n = 0.0, 0
    for u in range(1, total + 1):
        net.train(); idx = rng.choice(len(tr_items), o['batch_size'], replace=False); si = rng.integers(0, nT, o['batch_size'])
        loss = clip_loss(*fwd([tr_items[i] for i in idx], si)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tl += loss.item(); n += 1
        if u % every == 0:
            log.append(dict(update=u, train_loss=tl / n, **evaluate(), seconds=time.time() - t0)); tl, n = 0.0, 0
            print(f"update {u}: train {log[-1]['train_loss']:.4f} valid {log[-1]['valid_loss']:.4f} mrp trainers {log[-1]['mrp_trainers']:.4f} "
                  f"targets(uncal) {log[-1]['mrp_targets_uncalibrated']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    pts = log[1:] if len(log) > 1 else log
    i_loss = int(np.argmin([x['valid_loss'] for x in pts])); i_mrp = int(np.argmax([x['mrp_trainers'] for x in pts]))
    end = pts[-1]
    summary = dict(exp=a.exp, smoke=a.smoke, settings=X, source=source, trainers=trainers, targets=targets, valid_runs=[r + 1 for r in valid_runs],
                   n_valid_queries=len(q), best_by_valid_loss=dict(update=pts[i_loss]['update'], valid_loss=pts[i_loss]['valid_loss'],
                                                                   mrp_trainers=pts[i_loss]['mrp_trainers']),
                   best_by_mrp_trainers=dict(update=pts[i_mrp]['update'], mrp_trainers=pts[i_mrp]['mrp_trainers'],
                                             minus_end=pts[i_mrp]['mrp_trainers'] - end['mrp_trainers']),
                   end=dict(update=end['update'], valid_loss=end['valid_loss'], mrp_trainers=end['mrp_trainers']),
                   clamp_fraction=clamp, seconds=time.time() - t0)
    od = os.path.join(ROOT, P['out_dir']); os.makedirs(od, exist_ok=True)
    json.dump(dict(summary=summary, log=log), open(os.path.join(od, f"{a.exp}{'_smoke' if a.smoke else ''}.json"), 'w'), indent=1)
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
