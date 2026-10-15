"""C1 arm P2, step 1 (D-096, D-097, D-099): multi-dataset external pretraining of the Défossez brain module.

Data: the external listening datasets of D-099 (configs/c1_p2_pretrain.yaml `datasets`; ds004408 and DTU are never read),
through src/ese/listen/multi.py. One subject layer per external subject (dataset-specific subject layers); batches are
drawn per dataset with the configured mixing weights (`batch_mode: per_dataset`: all windows of a batch come from one dataset,
so CLIP negatives are windows of the same dataset) or per example (`mixed`). Same network, optimiser, window, EEG handling
(CAR without bad/flat channels, baseline, clamp) and w2v target as the P1 shared network (configs/c1_p1.yaml, D-097).
Evaluation every `eval_every` updates on fixed validation batches of every dataset (held-out stimulus groups); the combined
validation loss (mixing-weight average of the per-dataset losses) drives early stopping (patience in evaluation units, never
before min_updates). Output: pretrain.pt = the best state_dict (c1_train.py --init drops the subject layers), history.json
with every per-dataset validation loss, metrics.json, config.yaml in results/runs/<date>_c1_P2_pretrain_s<seed>[...]/.

Development candidates (D-097 11: at most 4): `candidates: {name: {training: {...}, weights: {dataset: w}}}` in the config;
--candidate NAME applies the overrides and writes to results/runs/<date>_c1_P2_pretrain_s<seed>_<NAME>[...]/pretrain.pt.

usage: python scripts/c1_pretrain.py configs/c1_p2_pretrain.yaml --seed S [--candidate NAME] [--smoke | --timing]
       python scripts/c1_pretrain.py configs/c1_p2_pretrain.yaml --seed 0 --weights
       --weights: no training; prints the training / validation EEG hours per dataset under the hold-out rule and the
       candidate weights w_d = H_d / sum H (hours) and sqrt(H_d) / sum sqrt(H) (sqrt, mixed); writes results/d099_ext/weights.json
       --smoke : 2 subjects x 2 recordings per dataset, 20 updates, batch 32
       --timing: full data, `timing_updates` updates, no checkpoint selection report
"""
import os, sys, json, time, copy, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.brain_clip import BrainModule, clip_loss
from ese.listen.multi import MultiData


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--candidate', default='')
    g = ap.add_mutually_exclusive_group(); g.add_argument('--smoke', action='store_true'); g.add_argument('--timing', action='store_true')
    g.add_argument('--weights', action='store_true')
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); b1 = yaml.safe_load(open(os.path.join(ROOT, cfg['b1_config'])))
    if a.weights: return weights(cfg)
    seed = a.seed; assert seed in cfg['seeds'] or a.smoke, f'seed {seed} not in {cfg["seeds"]}'
    if a.candidate:                                                       # development candidate overrides (D-097 11)
        cd = cfg['candidates'][a.candidate]; cfg['training'] = dict(cfg['training'], **cd.get('training', {}))
        miss = [d['name'] for d in cfg['datasets'] if d.get('enabled', True) and d['name'] not in cd.get('weights', {})]
        assert not miss or a.smoke, f'candidate {a.candidate}: no weight for {miss} (fill from ext_summary.py before any run)'
        for d in cfg['datasets']: d['weight'] = cd.get('weights', {}).get(d['name'], d['weight'])
    for k in ('window', 'model'): cfg.setdefault(k, b1[k])                 # P1 = B1 window and network unless overridden
    cfg['eeg'] = {**b1['eeg'], 'reference': 'car', **cfg.get('eeg', {})}
    o = dict(b1['optim'], **cfg.get('optim', {})); tr = dict(cfg['training']); nvb = tr['valid_batches_per_dataset']
    if a.smoke: o['batch_size'] = 32; tr.update(eval_every=10, max_updates=20, min_updates=0); nvb = 2
    if a.timing: tr.update(max_updates=tr['timing_updates'], min_updates=tr['timing_updates'])
    tag = (f'_{a.candidate}' if a.candidate else '') + ('_smoke' if a.smoke else '_timing' if a.timing else '')
    out = os.path.join(ROOT, cfg['out_root'], time.strftime('%Y-%m-%d') + f'_c1_P2_pretrain_s{seed}{tag}'); os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = 'cuda'; t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    D = MultiData(ROOT, cfg, smoke=a.smoke); names = [s.name for s in D.src]; wts = [d['weight'] for d in cfg['datasets'] if d.get('enabled', True)]
    summ_data = D.summary(); print('data', json.dumps(summ_data), f'({time.time() - t0:.0f}s)', flush=True)
    yaml.safe_dump(dict(cfg, optim=o, training=tr, run=dict(seed=seed, candidate=a.candidate, smoke=a.smoke, timing=a.timing, data=summ_data, n_subjects=D.n_subjects)),
                   open(os.path.join(out, 'config.yaml'), 'w'), allow_unicode=True)
    m = cfg['model']; net = BrainModule(out_dim=D.src[0].F.shape[0], n_subjects=D.n_subjects, merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                                        merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'],
                                        depth=m['depth'], subject_layers=True).to(dev)
    amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=o['precision'] == 'bf16'); opt = torch.optim.Adam(net.parameters(), lr=o['lr'])
    vr = np.random.default_rng(0)                                             # fixed validation batches, identical for every seed
    vb = {di: [D.sample(vr, 'valid', np.eye(len(names))[di], o['batch_size']) for _ in range(nvb)] for di in range(len(names))}
    hist = []

    def fwd(items):
        (X, P, S, *_), Y = D.batch(items, dev)
        with torch.autocast(**amp): return net(X, P, S).float(), Y

    def draw():
        if tr['batch_mode'] == 'per_dataset':
            di = rng.choice(len(wts), p=np.asarray(wts) / np.sum(wts)); return D.sample(rng, 'train', np.eye(len(names))[di], o['batch_size'])
        return D.sample(rng, 'train', wts, o['batch_size'])

    def step():
        net.train(); tl = 0.0
        for _ in range(tr['eval_every']):
            est, Y = fwd(draw()); loss = clip_loss(est, Y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item()
        return tl / tr['eval_every']

    def vloss():
        net.eval(); per = {}
        with torch.no_grad():
            for di, bs in vb.items(): per[names[di]] = float(np.mean([clip_loss(*fwd(b)).item() for b in bs]))
        comb = float(np.average([per[n] for n in names], weights=wts)); return comb, per

    best, bad, best_state, u = float('inf'), 0, None, 0
    while u < tr['max_updates']:
        t1 = time.time(); tl = step(); u += tr['eval_every']; vl, per = vloss()
        hist.append(dict(stage='pretrain', update=u, train=tl, valid=vl, valid_per_dataset=per, seconds=time.time() - t1))
        print(f'pretrain update {u}: train {tl:.4f} valid {vl:.4f} ' + ' '.join(f'{k} {v:.3f}' for k, v in per.items()) + f' ({time.time() - t1:.0f}s)', flush=True)
        if vl < best: best, bad, best_state = vl, 0, copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
        else:
            bad += 1
            if bad >= tr['patience_units'] and u >= tr['min_updates']: break
    torch.save(best_state, os.path.join(out, 'pretrain.pt')); json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
    bu = min(hist, key=lambda h: h['valid'])
    summ = dict(seed=seed, candidate=a.candidate, best_valid=best, best_update=bu['update'], best_valid_per_dataset=bu['valid_per_dataset'], updates=u,
                n_subjects=D.n_subjects, data=summ_data, device=torch.cuda.get_device_name(), seconds=time.time() - t0,
                max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2 ** 30, chance_valid=float(np.log(o['batch_size'])))
    json.dump(summ, open(os.path.join(out, 'metrics.json'), 'w'), indent=1); print('SUMMARY', json.dumps(summ), flush=True)


def weights(cfg):
    b1 = yaml.safe_load(open(os.path.join(ROOT, cfg['b1_config']))); cfg.setdefault('window', b1['window']); cfg['eeg'] = {**b1['eeg'], 'reference': 'car', **cfg.get('eeg', {})}
    D = MultiData(ROOT, cfg); summ = D.summary(); H = {k: v['train_hours'] for k, v in summ.items()}; tot = sum(H.values()); sq = sum(np.sqrt(list(H.values())))
    out = dict(data=summ, hours={k: round(v / tot, 4) for k, v in H.items()}, sqrt={k: round(float(np.sqrt(v) / sq), 4) for k, v in H.items()})
    os.makedirs(os.path.join(ROOT, 'results/d099_ext'), exist_ok=True); json.dump(out, open(os.path.join(ROOT, 'results/d099_ext/weights.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()
