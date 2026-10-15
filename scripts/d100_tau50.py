"""D-100 (a): temporal-alignment tolerance of the B1 decoders (inference only; no training, no change to any B1 number).

For every B1 model (25 main + 5 phase-randomised; shared.pt and target_layers.pt frozen) and every target subject, the test
queries common to all perturbations are re-scored with the query EEG window perturbed:
  shift delta : EEG window [t - 0.5 + 0.15 + delta, t + 2.5 + 0.15 + delta)
  stretch r   : [a - 0.5 r, a + 2.5 r), a = t + 0.15, linear interpolation to the 3-s length, then CAR / baseline / clamp
Candidates are the truth and the same-run queries >= 30 s away (B1 report layer). Scores: CLIP (fixed alignment); report-only
delay search = max over l in [-0.25, 0.25] (step 0.05) of the delta + l scores. PRE: the same perturbations on the window
10 s earlier (queries and candidates restricted to those valid for every perturbation). The target self-similarity baseline
(no EEG) uses the perturbed true w2v window as the estimate. Null: B1 segment-order circular shift of the truth (k_min 30 s,
one k per run and draw, same for every subject and model), candidates >= 30 s from both the query and the shift partner;
pairs (i, draw) with the partner within 30 s are left out. Per (model, target, run, condition) the observed mean MRP, query
count and 1,000 null means go to data/cache/d100_tau50/; scripts/d100_tau50_report.py pools them.

usage: python scripts/d100_tau50.py configs/d100_tau50.yaml [--only NAME] [--check]
       --check: first main model, first target, first run only, prints the delta-0 value next to the B1 30-s-excluded value
"""
import os, sys, glob, json, time, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.brain_clip import BrainModule, clip_scores, INVALID


def offsets_grid(c):
    g = c['offset_grid_s']; return [round(x, 2) for x in np.arange(g['lo'], g['hi'] + 1e-9, g['step'])]


def common_queries(c, b1, lens, fs, items, pre=False):
    """indices of items (run, t) whose perturbed EEG windows (and target windows) stay inside the run for every perturbation"""
    w, e = b1['window'], b1['eeg']; m = max(abs(c['offset_grid_s']['lo']), c['offset_grid_s']['hi']); rmax = max(c['stretch_r'])
    base = -c['pre_shift_s'] if pre else 0.0; keep = []
    for k, (r, t) in enumerate(items):
        dur = lens[r] / fs
        lo = min(t + w['tmin'] + e['offset_s'] + base - m, t + e['offset_s'] + base + w['tmin'] * rmax, t + w['tmin'] - m)
        hi = max(t + w['tmax'] + e['offset_s'] + base + m, t + e['offset_s'] + base + w['tmax'] * rmax, t + w['tmax'] + m)
        if lo >= 0 and hi <= dur - 1.0 / fs: keep.append(k)
    return np.array(keep, int)


def run_maps(t_by_run, n_shift, min_shift, seed, tag):
    """B1 amendment-1 segment-order shift, one generator per run so that every model and subject gets the same draws"""
    maps = {}
    for r, on in t_by_run.items():
        rng = np.random.default_rng([seed, int(r), tag]); order = np.argsort(on); n = len(on); so = on[order]
        kmin = int(np.ceil(np.median(np.searchsorted(so, so + min_shift) - np.arange(n))))
        pos = np.empty(n, int); pos[order] = np.arange(n); ks = rng.integers(kmin, n - kmin + 1, n_shift); maps[r] = order[(pos[None] + ks[:, None]) % n]
    return maps


def excl_stats(S, t, maps, excl, chunk=50):
    """S [n, n] torch (rows queries, columns candidates = the same queries). Returns the observed mean of the 30-s-excluded
    MRP and the null mean per draw (truth -> shift partner j, candidates >= excl from both i and j)."""
    dev = S.device; n = S.shape[0]; tt = torch.as_tensor(t, device=dev, dtype=torch.float64)
    far = (tt[:, None] - tt[None]).abs() >= excl; ar = torch.arange(n, device=dev)
    above = (S > S.diagonal()[:, None]) & far; obs = (1 - above.sum(1).double() / far.sum(1).clamp(min=1).double()).mean().item()
    M = torch.as_tensor(maps, device=dev); out = []
    for b0 in range(0, len(M), chunk):
        J = M[b0:b0 + chunk]                                                   # [B, n]
        v = S[ar[None], J]                                                     # [B, n] score of the shifted truth
        cand = far[None] & far[J]                                              # [B, n, n] >= excl from i and from j
        rk = ((S[None] > v[..., None]) & cand).sum(-1).double(); cnt = cand.sum(-1).double()
        ok = far[ar[None], J] & (cnt > 0); mrp = torch.where(ok, 1 - rk / cnt.clamp(min=1), torch.zeros_like(rk))
        out.append((mrp.sum(1) / ok.sum(1).clamp(min=1)).cpu().numpy())
    return obs, np.concatenate(out)


def stretch_eeg(D, items, si, r_fac, base_shift=0.0):
    """B1 RunData.batch with a stretched EEG span [a - 0.5 r, a + 2.5 r) resampled to D.T samples"""
    e = D.c['eeg']; w = D.c['window']; Xs, runs = [], []; n_raw = int(np.ceil((w['tmax'] - w['tmin']) * r_fac * D.fs)) + 2
    for (r, t), s_i in zip(items, si):
        a = t + e['offset_s'] + base_shift; i = D.offs[r] + int(np.floor((a + w['tmin'] * r_fac) * D.fs))
        x = torch.from_numpy(np.asarray(D.eeg[D.subjects[s_i]][:, i:i + n_raw], np.float32))
        frac = (a + w['tmin'] * r_fac) * D.fs - np.floor((a + w['tmin'] * r_fac) * D.fs)
        pos = frac + np.arange(D.T) * r_fac; i0 = np.floor(pos).astype(int); f = torch.from_numpy((pos - i0).astype(np.float32))
        Xs.append(x[:, i0] * (1 - f) + x[:, i0 + 1] * f); runs.append(r)
    X = torch.stack(Xs).to(D.dev); S = torch.as_tensor(np.asarray(si), device=D.dev); bad = D.badg[S, torch.as_tensor(runs, device=D.dev)]
    if e['reference'] == 'car':
        ok = (~bad).float()[..., None]; X = X - (X * ok).sum(1, keepdim=True) / ok.sum(1, keepdim=True)
    X = X - X[..., : int(round(e['baseline_s'] * D.fs))].mean(-1, keepdim=True); X = X.clamp(-e['clamp'], e['clamp'])
    P = D.posg[None].expand(len(S), -1, -1).masked_fill(bad[..., None], INVALID)
    return X, P


def target_windows(D, items, shift=0.0, r_fac=None):
    """true w2v windows (standardised as in training) moved by shift or stretched around the onset"""
    out = []
    for (r, t) in items:
        if r_fac is None: out.append(D.Fg[:, D.feat_idx(r, t + shift) + D.ar].float())
        else:
            p0 = (t + D.c['window']['tmin'] * r_fac) * D.fs; pos = p0 + torch.arange(D.T, device=D.dev, dtype=torch.float64) * r_fac
            i0 = torch.floor(pos).long(); f = (pos - i0).float(); base = D.offs[r]
            out.append(D.Fg[:, base + i0].float() * (1 - f) + D.Fg[:, base + i0 + 1].float() * f)
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--only'); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); b1 = yaml.safe_load(open(os.path.join(ROOT, c['b1_config']))); t0 = time.time(); dev = 'cuda'
    from stageB_b1_train import RunData
    offs_grid = offsets_grid(c); deltas = [round(d, 2) for d in c['deltas_s']]; rs = c['stretch_r']; ds = c['delay_search_s']; excl = c['exclusion_s']
    sn = c['shift_null']; out_dir = os.path.join(ROOT, c['cache']); os.makedirs(out_dir, exist_ok=True)
    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, b1['out_root'], f"*_stageB_b1_{b1['decoder_setting']}_k?_f?")) + glob.glob(os.path.join(ROOT, b1['out_root'], f"*_stageB_b1_{b1['decoder_setting']}_k?_f?_phaserand"))
                  if os.path.exists(os.path.join(d, 'scores.npz')))
    assert len(dirs) == c['models']['main'] + c['models']['phaserand'], len(dirs)
    if a.only: dirs = [d for d in dirs if d.endswith(a.only)]
    if a.check: dirs = [d for d in dirs if not d.endswith('phaserand')][:1]
    tgt_done = set(json.load(open(os.path.join(out_dir, 'target_done.json')))) if os.path.exists(os.path.join(out_dir, 'target_done.json')) else set()
    for d in dirs:
        name = os.path.basename(d); outp = os.path.join(out_dir, name + '.npz')
        if os.path.exists(outp) and not a.check: print('skip', name, flush=True); continue
        rc = yaml.safe_load(open(os.path.join(d, 'config.yaml'))); r = rc['run']; nT = len(r['trainers'])
        D = RunData(rc, r['trainers'] + r['targets'], r['train_runs'], r['valid_runs'], r['eval_runs'], r['source']); D.to_device(dev)
        m = rc['model']; net = BrainModule(out_dim=D.F.shape[0], n_subjects=nT + len(r['targets']), merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                                           merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'], depth=m['depth'], subject_layers=True).to(dev)
        net.load_state_dict(torch.load(os.path.join(d, 'shared.pt'))); net.eval()
        with torch.no_grad(): net.subject.weights[nT:] = torch.load(os.path.join(d, 'target_layers.pt')).to(dev)
        q = D.items['eval']; kq = {'main': common_queries(c, b1, D.lens, D.fs, q), 'pre': common_queries(c, b1, D.lens, D.fs, q, pre=True)}
        qrun = np.array([x[0] for x in q]); qt = np.array([x[1] for x in q]); arrays = {}; summary = dict(n_eval=len(q), n_main=len(kq['main']), n_pre=len(kq['pre']))
        maps = {kind: run_maps({rr: qt[kq[kind]][qrun[kq[kind]] == rr] for rr in sorted(set(qrun.tolist()))}, sn['n_shift'], sn['min_shift_s'], sn['seed'], 0 if kind == 'main' else 1)
                for kind in kq}
        targets = r['targets'][:1] if a.check else r['targets']; runs_eval = sorted(set(qrun.tolist()))[:1] if a.check else sorted(set(qrun.tolist()))
        def fwd_offset(items, ti, shift):
            outs = []
            with torch.no_grad():
                for b0 in range(0, len(items), 128):
                    it = items[b0:b0 + 128]; X, P, _ = D.batch(it, np.full(len(it), ti), shift); outs.append(net(X, P, torch.full((len(it),), ti, device=dev)).float())
            return outs
        def fwd_stretch(items, ti, rf, base):
            outs = []
            with torch.no_grad():
                for b0 in range(0, len(items), 128):
                    it = items[b0:b0 + 128]; X, P = stretch_eeg(D, it, np.full(len(it), ti), rf, base); outs.append(net(X, P, torch.full((len(it),), ti, device=dev)).float())
            return outs
        def score(outs, cand):
            return torch.cat([torch.cat([clip_scores(o, cand[cc:cc + 512]) for cc in range(0, len(cand), 512)], 1) for o in outs])
        for rr in runs_eval:
            for kind in ('main', 'pre'):
                idx = [k for k in kq[kind] if qrun[k] == rr]; items = [q[k] for k in idx]; tt = qt[idx]; M = maps[kind][rr]; base = 0.0 if kind == 'main' else -c['pre_shift_s']
                cand = torch.stack([D.Fg[:, D.feat_idx(x, tq) + D.ar].float() for x, tq in items])
                # target self-similarity baseline (no EEG, once per run)
                if kind == 'main' and f'{rr}' not in tgt_done and not name.endswith('phaserand'):
                    Sg = {}
                    for o in offs_grid: Sg[o] = clip_scores(target_windows(D, items, shift=o), cand)
                    for dl in deltas:
                        o_, n_ = excl_stats(Sg[dl], tt, M, excl); arrays[f'tgt_r{rr}_d{dl:+.2f}'] = np.r_[o_, len(idx), n_]
                        Smax = torch.stack([Sg[round(dl + l, 2)] for l in np.arange(-ds, ds + 1e-9, 0.05).round(2)]).max(0).values
                        o_, n_ = excl_stats(Smax, tt, M, excl); arrays[f'tgt_r{rr}_ds{dl:+.2f}'] = np.r_[o_, len(idx), n_]
                    for rf in rs:
                        o_, n_ = excl_stats(clip_scores(target_windows(D, items, r_fac=rf), cand), tt, M, excl); arrays[f'tgt_r{rr}_s{rf:.2f}'] = np.r_[o_, len(idx), n_]
                    del Sg; tgt_done.add(f'{rr}')
                for j, s in enumerate(targets):
                    ti = nT + j; Sg = {}
                    grid = offs_grid if kind == 'main' else deltas
                    for o in grid: Sg[round(o, 2)] = score(fwd_offset(items, ti, base + o), cand)
                    for dl in deltas:
                        o_, n_ = excl_stats(Sg[dl], tt, M, excl); arrays[f'{kind}_s{s}_r{rr}_d{dl:+.2f}'] = np.r_[o_, len(idx), n_]
                        if kind == 'main':
                            Smax = torch.stack([Sg[round(dl + l, 2)] for l in np.arange(-ds, ds + 1e-9, 0.05).round(2)]).max(0).values
                            o_, n_ = excl_stats(Smax, tt, M, excl); arrays[f'{kind}_s{s}_r{rr}_ds{dl:+.2f}'] = np.r_[o_, len(idx), n_]
                    for rf in rs:
                        o_, n_ = excl_stats(score(fwd_stretch(items, ti, rf, base), cand), tt, M, excl); arrays[f'{kind}_s{s}_r{rr}_s{rf:.2f}'] = np.r_[o_, len(idx), n_]
                    del Sg
                    if a.check and kind == 'main':
                        Z = np.load(os.path.join(d, 'scores.npz')); Sb = Z[f'main_s{s}_r{rr}']; onb = Z['onset'][Z['run'] == rr]; dt = np.abs(onb[:, None] - onb[None]); np.fill_diagonal(dt, np.inf)
                        far = dt >= excl; b1v = np.mean(1 - ((Sb > np.diag(Sb)[:, None]) & far).sum(1) / np.maximum(far.sum(1), 1))
                        print(f'check sub-{s:03d} run {rr + 1}: delta0 on the common subset {arrays[f"main_s{s}_r{rr}_d+0.00"][0]:.4f} (n {len(idx)}), '
                              f'B1 30-s-excluded on all {len(onb)} queries {b1v:.4f}; null mean {arrays[f"main_s{s}_r{rr}_d+0.00"][2:].mean():.4f}', flush=True)
            print(f'{name} run {rr + 1} done ({time.time() - t0:.0f}s)', flush=True)
        if a.check: print(json.dumps({k: [round(float(v[0]), 4), int(v[1]), round(float(v[2:].mean()), 4)] for k, v in arrays.items()}), flush=True); return
        np.savez_compressed(outp, **{k: v.astype(np.float32) for k, v in arrays.items()}, summary=json.dumps(summary))
        json.dump(sorted(tgt_done), open(os.path.join(out_dir, 'target_done.json'), 'w'))
        print(f'{name}: saved {len(arrays)} cells, {summary} ({time.time() - t0:.0f}s)', flush=True)
        del net, D; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
