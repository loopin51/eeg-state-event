"""D-092 A′ inspection after the stop rule (report only; the stop verdict does not change).

--part rescore: (a) loads best.pt of the A′ run, scores the same test queries as scripts/alice_train.py --mode aprime (fp32)
  and reports, per subject and averaged over subjects, top-10 (must equal the run's metrics.json), top-k with k = 10 x N/190,
  the mean rank percentile, and top-10 among random candidate sets of 190 test queries (fixed number of sets and seed from the
  config, the same sets for every subject; query and candidates come from the same set).
--part probe: (b) retrains A′ exactly as job 171 (same seed, random draws and fixed validation batches) for total_updates
  updates and, before training and every eval_every updates, records the mean training CLIP loss of the interval, the CLIP
  loss on the 20 fixed validation batches, and on the validation blocks (word-onset queries, candidates = all validation
  queries) the top-10 and mean rank percentile per subject. Record layout follows scripts/d094_probe.py. No test window is
  built and no checkpoint is written.
--smoke: 3 subjects, tiny batches; rescore then uses the validation queries (never the test data).

usage: python scripts/alice_aprime_check.py configs/alice_aprime_check.yaml --part rescore|probe [--smoke]
"""
import os, sys, json, time, argparse, yaml, numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from alice_train import Alice, build_net, pairs_for, Trainer, fixed_batches, score_rows


def setup(c, smoke):
    o = dict(c['aprime']); pre = dict(reference=o['reference'], invalid=o['invalid'], clamp=o['clamp'], baseline_s=o['baseline_s'])
    meta = json.load(open(os.path.join(ROOT, c['cache_dir'], 'eeg_meta.json'))); subjects = sorted(int(s) for s in meta['subjects'])
    if smoke: subjects = [1, 3, 26]; o.update(batch_size=32)
    D = Alice(c, o['source'], c['features']['aprime'], subjects, 'defossez', pre); B = pd.read_csv(os.path.join(ROOT, c['cache_dir'], 'blocks.tsv'), sep='\t')
    def spans(split, r):
        out_, b = [], B[B.run == r].sort_values('start')
        for _, x in b.iterrows():
            if x.split != split: continue
            if out_ and abs(out_[-1][1] - x.start) < 1e-6: out_[-1][1] = x.stop
            else: out_.append([x.start, x.stop])
        return out_
    D.feat_norm([(r, a0, b0) for r in range(D.nseg) for a0, b0 in spans('train', r)]); D.to_device('cuda')
    return o, D, subjects, spans


def rank_metrics(Sm):
    true = Sm.diagonal(); rk = (Sm > true[:, None]).sum(1).cpu().numpy(); return rk, len(true)


def score_subject(tr, D, q, si, Yc, chunk):
    keep = np.array([r not in D.missing[D.subjects[si]] for r, _ in q]); idx = np.where(keep)[0]
    Sm, ostd = score_rows(tr, [q[i] for i in idx], si, Yc[torch.as_tensor(idx, device=D.dev)], chunk); return Sm, idx, ostd


def rescore(c, C, a):
    R = C['rescore']; o, D, subjects, spans = setup(c, a.smoke); t0 = time.time()
    split = 'valid' if a.smoke else 'test'
    q = [it for r in range(D.nseg) for a0, b0 in spans(split, r) for it in D.onsets(r, a0, b0)]; n = len(q)
    Yc = torch.stack([D.Fg[:, D.feat_idx(r, t) + D.ar] for r, t in q])
    net = build_net(c, 'defossez', len(subjects), D, 'cuda')
    if not a.smoke: net.load_state_dict(torch.load(os.path.join(ROOT, c['out_root'], R['run'], 'best.pt')))
    net.eval(); tr = Trainer(net, D, dict(device_type='cuda', dtype=torch.bfloat16, enabled=False), None)
    m = min(R['subsample_n'], n); srng = np.random.default_rng(R['subsample_seed'])           # one generator for all sets
    sets = [np.sort(srng.choice(n, m, replace=False)) for _ in range(R['subsample_reps'])]
    k = R['top_k']; per = {}; sub_top10 = np.zeros((len(subjects), len(sets)))
    with torch.no_grad():
        for si, s in enumerate(subjects):
            Sm, idx, _ = score_subject(tr, D, q, si, Yc, c['eval']['score_chunk']); rk, nk = rank_metrics(Sm)
            pos = -np.ones(n, int); pos[idx] = np.arange(len(idx))
            for j, st in enumerate(sets):
                p = torch.as_tensor(pos[st][pos[st] >= 0], device='cuda'); S = Sm[p][:, p]; sub_top10[si, j] = float(((S > S.diagonal()[:, None]).sum(1) < 10).float().mean())
            per[str(s)] = dict(n_queries=nk, top10=float((rk < 10).mean()), **{f'top{k}': float((rk < k).mean())}, rank_pct=float((1 - rk / (nk - 1)).mean()),
                               top10_subsample190=float(sub_top10[si].mean()))
            print(f"S{s:02d}: top10 {per[str(s)]['top10']:.4f} top{k} {per[str(s)][f'top{k}']:.4f} rank_pct {per[str(s)]['rank_pct']:.4f} "
                  f"top10@190 {per[str(s)]['top10_subsample190']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    rep_means = sub_top10.mean(0)
    summ = dict(split=split, n_queries=n, smoke=a.smoke, run=R['run'], **{key: float(np.mean([v[key] for v in per.values()])) for key in ('top10', f'top{k}', 'rank_pct', 'top10_subsample190')},
                top10_subsample190_rep_pct=[float(x) for x in np.percentile(rep_means, [2.5, 50, 97.5])], chance_top10=10 / n, chance_topk=k / n,
                chance_top10_subsample190=10 / m, subsample=dict(n=m, reps=R['subsample_reps'], seed=R['subsample_seed']))
    if not a.smoke:
        ref = json.load(open(os.path.join(ROOT, c['out_root'], R['run'], 'metrics.json')))['fp32']; summ['top10_minus_run_metrics'] = summ['top10'] - ref['top10']
    od = os.path.join(ROOT, C['out_dir']); os.makedirs(od, exist_ok=True)
    json.dump(dict(summary=summ, subjects=per), open(os.path.join(od, f"rescore{'_smoke' if a.smoke else ''}.json"), 'w'), indent=1)
    print('SUMMARY', json.dumps(summ), flush=True)


def probe(c, C, a):
    P = C['probe']; total, every = P['total_updates'], P['eval_every']; t0 = time.time()
    o, D, subjects, spans = setup(c, a.smoke)
    if a.smoke: total, every = 20, 10
    items = {sp: [] for sp in ('train', 'valid')}
    for r in range(D.nseg):
        for sp in ('train', 'valid'): items[sp] += [it for a0, b0 in spans(sp, r) for it in D.grid(r, a0, b0)]
    q = [it for r in range(D.nseg) for a0, b0 in spans('valid', r) for it in D.onsets(r, a0, b0)]; nS = len(subjects)
    rng = np.random.default_rng(c['seed']); torch.manual_seed(c['seed'])          # same order of draws as alice_train.run_aprime
    trp, vap = pairs_for(D, items['train'], range(nS)), pairs_for(D, items['valid'], range(nS))
    print(f"A′ probe: subjects {nS}, windows train {len(items['train'])} valid {len(items['valid'])}, valid queries {len(q)}", flush=True)
    net = build_net(c, 'defossez', nS, D, 'cuda'); amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=o['precision'] == 'bf16')
    tr = Trainer(net, D, amp, rng); opt = torch.optim.Adam(net.parameters(), lr=o['lr'])
    vb = fixed_batches(vap, o['batch_size'], 2 if a.smoke else 20, np.random.default_rng(0))
    Yc = torch.stack([D.Fg[:, D.feat_idx(r, t) + D.ar] for r, t in q])

    def evaluate():
        vl, _ = tr.vloss(items['valid'], vb, lambda: None); net.eval(); mrp, top10, ostd = {}, {}, {}
        with torch.no_grad():
            for si, s in enumerate(subjects):
                Sm, idx, ostd[str(s)] = score_subject(tr, D, q, si, Yc, c['eval']['score_chunk']); rk, nk = rank_metrics(Sm)
                mrp[str(s)] = float((1 - rk / (nk - 1)).mean()); top10[str(s)] = float((rk < 10).mean())
        return dict(valid_loss=vl, mrp_mean=float(np.mean(list(mrp.values()))), top10_mean=float(np.mean(list(top10.values()))),
                    output_std_min=float(min(ostd.values())), mrp=mrp, top10=top10)

    log = [dict(update=0, train_loss=None, **evaluate(), seconds=time.time() - t0)]
    print(f"update 0: valid {log[-1]['valid_loss']:.4f} mrp {log[-1]['mrp_mean']:.4f} top10 {log[-1]['top10_mean']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    for u in range(every, total + 1, every):
        tl = tr.epoch(opt, items['train'], trp, o['batch_size'], every)
        log.append(dict(update=u, train_loss=tl, **evaluate(), seconds=time.time() - t0))
        print(f"update {u}: train {tl:.4f} valid {log[-1]['valid_loss']:.4f} mrp {log[-1]['mrp_mean']:.4f} top10 {log[-1]['top10_mean']:.4f} "
              f"out_std_min {log[-1]['output_std_min']:.3g} ({time.time() - t0:.0f}s)", flush=True)
    pts = log[1:]; end = pts[-1]
    il = int(np.argmin([x['valid_loss'] for x in pts])); im = int(np.argmax([x['mrp_mean'] for x in pts])); it = int(np.argmax([x['top10_mean'] for x in pts]))
    pick = lambda x: dict(update=x['update'], valid_loss=x['valid_loss'], mrp_mean=x['mrp_mean'], top10_mean=x['top10_mean'])
    summary = dict(smoke=a.smoke, total_updates=total, eval_every=every, n_valid_queries=len(q), chance_top10=10 / len(q),
                   best_by_valid_loss=pick(pts[il]), best_by_mrp=dict(**pick(pts[im]), minus_end=pts[im]['mrp_mean'] - end['mrp_mean']),
                   best_by_top10=dict(**pick(pts[it]), minus_end=pts[it]['top10_mean'] - end['top10_mean']), end=pick(end),
                   job171_epoch0_valid_loss=7.8163214206695555, seconds=time.time() - t0)
    od = os.path.join(ROOT, C['out_dir']); os.makedirs(od, exist_ok=True)
    json.dump(dict(summary=summary, log=log), open(os.path.join(od, f"probe{'_smoke' if a.smoke else ''}.json"), 'w'), indent=1)
    print('SUMMARY', json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--part', required=True, choices=['rescore', 'probe']); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); C = yaml.safe_load(open(os.path.join(ROOT, a.config) if not os.path.isabs(a.config) else a.config))
    c = yaml.safe_load(open(os.path.join(ROOT, C['base_config'])))
    (rescore if a.part == 'rescore' else probe)(c, C, a)


if __name__ == '__main__':
    main()
