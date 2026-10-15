"""D-101 (a): B1 models (25 main + 5 phase-randomised, frozen) scored on band-limited input (inference only).

Each target's continuous A4 cache (eeg_hp05 / eeg_hp05_phaserand at 120 Hz) is filtered run by run with a 4th-order zero-phase
Butterworth filter (sosfiltfilt) and then windowed exactly as in B1 (CAR without flat channels, 0.5-s baseline, clamp).
Metric: B1 primary within-run MRP (every same-run query is a candidate) with the B1 null (segment-order circular shift,
k_min 30 s, 1,000 draws, seed 20261012, runs drawn in ascending order as in stageB_b1_verdict.py); report: 30-s-excluded MRP
with the D-100 null. Per (model, target, run, band): observed, query count, 1,000 null means -> data/cache/d101_artifact/.

usage: python scripts/d101_band_input.py configs/d101_artifact.yaml [--only NAME] [--check]
       --check: first main model, first target, full band only; compares with the B1 metrics.json value
"""
import os, sys, glob, json, time, argparse, yaml, numpy as np, torch
from scipy.signal import butter, sosfiltfilt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.brain_clip import BrainModule, clip_scores
from stageB_b1_verdict import shift_maps
from d100_tau50 import run_maps, excl_stats


def b1_stats(S, maps, chunk=100):
    """observed B1 within-run MRP (all candidates) and null mean per draw (truth -> maps[b][i])"""
    n = S.shape[0]; ar = torch.arange(n, device=S.device); obs = (1 - (S > S.diagonal()[:, None]).sum(1).double() / (n - 1)).mean().item()
    M = torch.as_tensor(maps, device=S.device); out = []
    for b0 in range(0, len(M), chunk):
        J = M[b0:b0 + chunk]; v = S[ar[None], J]; rk = (S[None] > v[..., None]).sum(-1).double(); out.append((1 - rk / (n - 1)).mean(1).cpu().numpy())
    return obs, np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--only'); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); b1 = yaml.safe_load(open(os.path.join(ROOT, c['b1_config']))); bi = c['band_input']; t0 = time.time(); dev = 'cuda'
    from stageB_b1_train import RunData
    out_dir = os.path.join(ROOT, c['cache']); os.makedirs(out_dir, exist_ok=True); bands = bi['bands']
    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, b1['out_root'], f"*_stageB_b1_{b1['decoder_setting']}_k?_f?")) + glob.glob(os.path.join(ROOT, b1['out_root'], f"*_stageB_b1_{b1['decoder_setting']}_k?_f?_phaserand"))
                  if os.path.exists(os.path.join(d, 'scores.npz')))
    assert len(dirs) == 30, len(dirs)
    if a.only: dirs = [d for d in dirs if d.endswith(a.only)]
    if a.check: dirs = [d for d in dirs if not d.endswith('phaserand')][:1]; bands = {'full': None}
    meta = json.load(open(os.path.join(ROOT, b1['cache_dir'], 'eeg_meta.json'))); fs = meta['sample_rate']; offs, lens = meta['run_offsets'], meta['run_lengths']
    W = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True); w, e = b1['window'], b1['eeg']
    def hi(r): return lens[r] / fs - w['tmax'] - max(e['offset_s'], 0)
    onsets = {}
    for r in range(20):
        on = np.unique(W['onset'][W['run'] == r]); onsets[r] = on[(on >= -w['tmin']) & (on <= hi(r))]
    maps_b1 = shift_maps({r: {'onset': onsets[r]} for r in range(20)}, bi['shift_null']['n_shift'], bi['shift_null']['min_shift_s'], np.random.default_rng(bi['shift_null']['seed']))
    maps_ex = run_maps(onsets, bi['shift_null']['n_shift'], 30.0, 20261111, 0)
    def make_sos(v):
        if v is None: return None
        lo, hi_ = v; order = bi['filter']['order']
        if lo is None: return butter(order, hi_, btype='lowpass', fs=fs, output='sos')
        if hi_ is None: return butter(order, lo, btype='highpass', fs=fs, output='sos')
        return butter(order, [lo, hi_], btype='bandpass', fs=fs, output='sos')
    sos = {k: make_sos(v) for k, v in bands.items()}
    for d in dirs:
        name = os.path.basename(d); outp = os.path.join(out_dir, f'band_{name}.npz')
        if os.path.exists(outp) and not a.check: print('skip', name, flush=True); continue
        rc = yaml.safe_load(open(os.path.join(d, 'config.yaml'))); r_ = rc['run']; nT = len(r_['trainers'])
        D = RunData(rc, r_['trainers'] + r_['targets'], r_['train_runs'], r_['valid_runs'], r_['eval_runs'], r_['source']); D.to_device(dev)
        m = rc['model']; net = BrainModule(out_dim=D.F.shape[0], n_subjects=nT + len(r_['targets']), merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                                           merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'], depth=m['depth'], subject_layers=True).to(dev)
        net.load_state_dict(torch.load(os.path.join(d, 'shared.pt'))); net.eval()
        with torch.no_grad(): net.subject.weights[nT:] = torch.load(os.path.join(d, 'target_layers.pt')).to(dev)
        q = D.items['eval']; qrun = np.array([x[0] for x in q]); arrays = {}; targets = r_['targets'][:1] if a.check else r_['targets']
        for j, s in enumerate(targets):
            ti = nT + j; raw = D.eeg[s]
            for band, so in sos.items():
                if so is None: D.eeg[s] = raw
                else:
                    X = np.empty(raw.shape, np.float32)
                    for rr in range(20): X[:, offs[rr]:offs[rr] + lens[rr]] = sosfiltfilt(so, np.asarray(raw[:, offs[rr]:offs[rr] + lens[rr]], np.float64), axis=-1)
                    D.eeg[s] = X
                for rr in sorted(set(qrun.tolist())):
                    items = [q[k] for k in np.flatnonzero(qrun == rr)]; assert np.allclose([t for _, t in items], onsets[rr])
                    cand = torch.stack([D.Fg[:, D.feat_idx(x, t) + D.ar].float() for x, t in items]); outs = []
                    with torch.no_grad():
                        for b0 in range(0, len(items), 128):
                            it = items[b0:b0 + 128]; Xb, P, _ = D.batch(it, np.full(len(it), ti)); outs.append(net(Xb, P, torch.full((len(it),), ti, device=dev)).float())
                    S = torch.cat([torch.cat([clip_scores(o, cand[cc:cc + 512]) for cc in range(0, len(cand), 512)], 1) for o in outs])
                    o1, n1 = b1_stats(S, maps_b1[rr]); arrays[f'b1_s{s}_r{rr}_{band}'] = np.r_[o1, len(items), n1].astype(np.float32)
                    o2, n2 = excl_stats(S, onsets[rr], maps_ex[rr], 30.0); arrays[f'ex_s{s}_r{rr}_{band}'] = np.r_[o2, len(items), n2].astype(np.float32)
                print(f'{name} sub-{s:03d} {band}: B1 MRP {np.average([arrays[f"b1_s{s}_r{rr}_{band}"][0] for rr in sorted(set(qrun.tolist()))], weights=[arrays[f"b1_s{s}_r{rr}_{band}"][1] for rr in sorted(set(qrun.tolist()))]):.4f} ({time.time() - t0:.0f}s)', flush=True)
            D.eeg[s] = raw
        if a.check:
            mt = json.load(open(os.path.join(d, 'metrics.json')))['targets'][str(targets[0])]['main']['mrp_within']; print(f'check: B1 metrics.json mrp_within {mt:.4f}', flush=True); return
        np.savez_compressed(outp, **arrays); print(f'{name}: saved {len(arrays)} ({time.time() - t0:.0f}s)', flush=True)
        del net, D; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
