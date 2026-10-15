"""D-100 amendment 1 (report only, added after one check cell was seen): fine shifts of +-1..4 samples (1/120 s) of the query
EEG window for the B1 decoders (25 main + 5 phase-randomised) and the target self-similarity baseline. Same queries
(common subset of scripts/d100_tau50.py), candidates (>= 30 s), null and cell layout; output data/cache/d100_tau50/fine/.
Not used for tau50, the R-e rule or any verdict.

usage: python scripts/d100_tau50_fine.py configs/d100_tau50.yaml
"""
import os, sys, glob, json, time, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.brain_clip import BrainModule, clip_scores
from d100_tau50 import common_queries, run_maps, excl_stats, target_windows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); c = yaml.safe_load(open(a.config)); t0 = time.time(); dev = 'cuda'
    b1 = yaml.safe_load(open(os.path.join(ROOT, c['b1_config']))); from stageB_b1_train import RunData
    sn = c['shift_null']; out_dir = os.path.join(ROOT, c['cache'], 'fine'); os.makedirs(out_dir, exist_ok=True); tgt_done = set()
    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, b1['out_root'], f"*_stageB_b1_{b1['decoder_setting']}_k?_f?")) + glob.glob(os.path.join(ROOT, b1['out_root'], f"*_stageB_b1_{b1['decoder_setting']}_k?_f?_phaserand"))
                  if os.path.exists(os.path.join(d, 'scores.npz')))
    assert len(dirs) == 30
    for d in dirs:
        name = os.path.basename(d); outp = os.path.join(out_dir, name + '.npz')
        if os.path.exists(outp): continue
        rc = yaml.safe_load(open(os.path.join(d, 'config.yaml'))); r = rc['run']; nT = len(r['trainers'])
        D = RunData(rc, r['trainers'] + r['targets'], r['train_runs'], r['valid_runs'], r['eval_runs'], r['source']); D.to_device(dev)
        m = rc['model']; net = BrainModule(out_dim=D.F.shape[0], n_subjects=nT + len(r['targets']), merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                                           merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'], depth=m['depth'], subject_layers=True).to(dev)
        net.load_state_dict(torch.load(os.path.join(d, 'shared.pt'))); net.eval()
        with torch.no_grad(): net.subject.weights[nT:] = torch.load(os.path.join(d, 'target_layers.pt')).to(dev)
        fine = [k / D.fs for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4)]
        q = D.items['eval']; kq = common_queries(c, b1, D.lens, D.fs, q); qrun = np.array([x[0] for x in q]); qt = np.array([x[1] for x in q])
        maps = run_maps({rr: qt[kq][qrun[kq] == rr] for rr in sorted(set(qrun.tolist()))}, sn['n_shift'], sn['min_shift_s'], sn['seed'], 0); arrays = {}
        for rr in sorted(set(qrun.tolist())):
            idx = [k for k in kq if qrun[k] == rr]; items = [q[k] for k in idx]; tt = qt[idx]
            cand = torch.stack([D.Fg[:, D.feat_idx(x, tq) + D.ar].float() for x, tq in items])
            if rr not in tgt_done and not name.endswith('phaserand'):
                for k, o in zip(range(-4, 5), fine):
                    o_, n_ = excl_stats(clip_scores(target_windows(D, items, shift=o), cand), tt, maps[rr], c['exclusion_s']); arrays[f'tgt_r{rr}_k{k:+d}'] = np.r_[o_, len(idx), n_]
                tgt_done.add(rr)
            for j, s in enumerate(r['targets']):
                ti = nT + j
                for k, o in zip(range(-4, 5), fine):
                    outs = []
                    with torch.no_grad():
                        for b0 in range(0, len(items), 128):
                            it = items[b0:b0 + 128]; X, P, _ = D.batch(it, np.full(len(it), ti), o); outs.append(net(X, P, torch.full((len(it),), ti, device=dev)).float())
                    S = torch.cat([torch.cat([clip_scores(x, cand[cc:cc + 512]) for cc in range(0, len(cand), 512)], 1) for x in outs])
                    o_, n_ = excl_stats(S, tt, maps[rr], c['exclusion_s']); arrays[f'main_s{s}_r{rr}_k{k:+d}'] = np.r_[o_, len(idx), n_]
        np.savez_compressed(outp, **{k: v.astype(np.float32) for k, v in arrays.items()}); print(f'{name}: {len(arrays)} cells ({time.time() - t0:.0f}s)', flush=True)
        del net, D; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
