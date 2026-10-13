"""Stage A: train and evaluate the Défossez et al. (2023) decoder on ds004408 (D-065, faculty direction 1).

Variants: clip_w2v (paper model), clip_mel (CLIP loss, mel target), base_mel (MSE regression of mel, the paper "base"
model), clip_w2v_nosubj (subject layer removed). Training windows every 0.5 s fully inside one split's contiguous blocks
(-0.5..2.5 s, brain signal +0.15 s), per-window baseline over the first 0.5 s, clamp at 20. Validation CLIP / MSE loss
for early stopping. Test segments: windows starting 0.5 s before each word onset, fully inside test blocks; every
subject's test sample is ranked against the features of all test segments (CLIP scores; the regression model is scored
the same way). Reports per-subject top-1/5/10, mean rank percentile and the chance level 1/N, 10/N.

usage: python scripts/stageA_train.py configs/stageA_train.yaml [--variant V] [--seed S] [--tag T] [--smoke]
"""
import os, sys, json, time, argparse, yaml, numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.model.brain_clip import BrainModule, clip_loss, clip_scores, INVALID

VARIANTS = {'clip_w2v': ('w2v', 'clip', True), 'clip_mel': ('mel', 'clip', True), 'base_mel': ('mel', 'mse', True),
            'clip_w2v_nosubj': ('w2v', 'clip', False)}


def spans(B, split, run):
    """contiguous [start, stop] intervals of `split` blocks in `run` (seconds)."""
    b = B[(B.run == run)].sort_values('start'); out = []
    for _, r in b.iterrows():
        if r.split != split: continue
        if out and abs(out[-1][1] - r.start) < 1e-6: out[-1][1] = r.stop
        else: out.append([r.start, r.stop])
    return out


def windows(B, split, run, c, dur, word_onsets=None):
    w = c['window']; off = c['eeg']['offset_s']; t = []
    for a, b in spans(B, split, run):
        cand = word_onsets[(word_onsets >= a - w['tmin']) & (word_onsets <= b - w['tmax'])] if word_onsets is not None \
            else np.arange(a - w['tmin'], b - w['tmax'] + 1e-9, w['train_step_s'])
        t += [x for x in cand if x + w['tmin'] + off >= 0 and x + w['tmax'] + off <= dur]
    return np.array(t)


class Data:
    def __init__(self, c, feat_kind, smoke=False):
        d = os.path.join(ROOT, c['cache_dir']); self.c = c; self.meta = json.load(open(os.path.join(d, 'eeg_meta.json')))
        self.fs = self.meta['sample_rate']; self.offs = self.meta['run_offsets']; self.lens = self.meta['run_lengths']
        self.subjects = sorted(int(s) for s in self.meta['subjects'])[: 3 if smoke else None]
        src = c['eeg'].get('source', 'eeg')   # eeg | eeg_phaserand (pipeline control, scripts/stageA_phase_random.py)
        self.eeg = {s: np.load(os.path.join(d, f'{src}_sub-{s:03d}.npy'), mmap_mode='r') for s in self.subjects}
        self.pos = torch.from_numpy(np.load(os.path.join(d, 'positions.npy')))
        F = np.load(os.path.join(d, c['features'][feat_kind] + '.npy'), mmap_mode='r'); self.B = pd.read_csv(os.path.join(d, 'blocks.tsv'), sep='\t')
        W = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True)
        durs = [L / self.fs for L in self.lens]; self.T = int(round((c['window']['tmax'] - c['window']['tmin']) * self.fs))
        self.items = {}
        for split in ('train', 'valid', 'test'):
            rows = []
            for r in range(len(self.lens)):
                ons = np.unique(W['onset'][W['run'] == r]) if split == 'test' else None
                rows += [(r, t) for t in windows(self.B, split, r, c, durs[r], ons)]
            self.items[split] = rows
        tr = np.concatenate([np.arange(self.offs[r] + int(round(a * self.fs)), self.offs[r] + min(int(round(b * self.fs)), self.lens[r]))
                             for r in range(len(self.lens)) for a, b in spans(self.B, 'train', r)])
        sub = tr[:: max(1, len(tr) // 200000)]; Fs = np.asarray(F[:, np.sort(sub)], np.float32)
        self.fmean, self.fstd = Fs.mean(1, keepdims=True), Fs.std(1, keepdims=True) + 1e-6; self.F = F
        self.invalid = {s: np.array(self.meta['subjects'][str(s)]['flat']) for s in self.subjects}

    def feat(self, r, t):
        i = self.offs[r] + int(round((t + self.c['window']['tmin']) * self.fs))
        return (np.asarray(self.F[:, i:i + self.T], np.float32) - self.fmean) / self.fstd

    def brain(self, s, r, t):
        e = self.c['eeg']; i = self.offs[r] + int(round((t + self.c['window']['tmin'] + e['offset_s']) * self.fs))
        x = np.asarray(self.eeg[s][:, i:i + self.T], np.float32).copy(); bad = self.invalid[s][r]
        if e['reference'] == 'car': x -= x[~bad].mean(0, keepdims=True)
        x -= x[:, : int(round(e['baseline_s'] * self.fs))].mean(1, keepdims=True)
        return np.clip(x, -e['clamp'], e['clamp']), bad

    def to_device(self, dev):
        """fast path (D-078): standardised features on the GPU (float16), per-(subject, run) invalid-channel masks"""
        self.dev = dev; N = self.F.shape[1]; self.Fg = torch.empty(self.F.shape, dtype=torch.float16, device=dev)
        m, sd = torch.from_numpy(self.fmean).to(dev), torch.from_numpy(self.fstd).to(dev)
        for a in range(0, N, 32768):
            self.Fg[:, a:a + 32768] = ((torch.from_numpy(np.asarray(self.F[:, a:a + 32768], np.float32)).to(dev) - m) / sd).half()
        torch.cuda.empty_cache()
        self.badg = torch.from_numpy(np.stack([self.invalid[s] for s in self.subjects])).to(dev)   # [nS, runs, C]
        self.posg = self.pos.to(dev); self.ar = torch.arange(self.T, device=dev)

    def batch(self, split, idx, subj_idx):
        if not hasattr(self, 'Fg'): return self.batch_cpu(split, idx, subj_idx)
        e = self.c['eeg']; w = self.c['window']; X, fi, runs = [], [], []
        for k, si in zip(idx, subj_idx):
            r, t = self.items[split][k]; s = self.subjects[si]
            i = self.offs[r] + int(round((t + w['tmin'] + e['offset_s']) * self.fs)); X.append(self.eeg[s][:, i:i + self.T])
            fi.append(self.offs[r] + int(round((t + w['tmin']) * self.fs))); runs.append(r)
        X = torch.from_numpy(np.stack(X).astype(np.float32)).to(self.dev, non_blocking=True)
        S = torch.as_tensor(np.asarray(subj_idx), device=self.dev); bad = self.badg[S, torch.as_tensor(runs, device=self.dev)]   # [B, C]
        if e['reference'] == 'car':
            ok = (~bad).float()[..., None]; X = X - (X * ok).sum(1, keepdim=True) / ok.sum(1, keepdim=True)
        X = X - X[..., : int(round(e['baseline_s'] * self.fs))].mean(-1, keepdim=True); X = X.clamp(-e['clamp'], e['clamp'])
        P = self.posg[None].expand(len(S), -1, -1).masked_fill(bad[..., None], INVALID)
        Y = self.Fg[:, torch.as_tensor(fi, device=self.dev)[:, None] + self.ar].permute(1, 0, 2).float()   # [B, F, T]
        return X, P, S, Y

    def batch_cpu(self, split, idx, subj_idx):
        X, Y, P, S = [], [], [], []
        for k, si in zip(idx, subj_idx):
            r, t = self.items[split][k]; s = self.subjects[si]; x, bad = self.brain(s, r, t)
            p = self.pos.clone(); p[torch.from_numpy(bad)] = INVALID
            X.append(x); Y.append(self.feat(r, t)); P.append(p); S.append(si)
        return (torch.from_numpy(np.stack(X)), torch.stack(P), torch.tensor(S), torch.from_numpy(np.stack(Y)))


def loss_fn(kind, est, y):
    return clip_loss(est, y) if kind == 'clip' else torch.nn.functional.mse_loss(est, y)


def evaluate(net, D, dev, c, prec, split='test'):
    net.eval(); items = D.items[split]; n = len(items); amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=prec == 'bf16')
    Y = torch.from_numpy(np.stack([D.feat(r, t) for r, t in items])).to(dev, torch.float16)       # candidates [n, F, T]
    run = torch.tensor([r for r, _ in items], device=dev); n_run = (run[:, None] == run[None]).sum(1).cpu().numpy()
    res = {}
    with torch.no_grad():
        for si, s in enumerate(D.subjects):
            ranks, ranks_run = [], []
            for a in range(0, n, c['eval']['score_chunk']):
                idx = list(range(a, min(n, a + c['eval']['score_chunk'])))
                X, P, S, _ = D.batch(split, idx, [si] * len(idx))
                with torch.autocast(**amp): Z = net(X.to(dev), P.to(dev), S.to(dev)).float()
                sc = torch.cat([clip_scores(Z, Y[b:b + 512].float()) for b in range(0, n, 512)], 1)   # [chunk, n]
                true = sc[torch.arange(len(idx)), torch.tensor(idx)]; above = sc > true[:, None]
                ranks.append(above.sum(1).cpu()); ranks_run.append((above & (run[idx][:, None] == run[None])).sum(1).cpu())
            rk = torch.cat(ranks).numpy(); rr = torch.cat(ranks_run).numpy()   # within-run: candidates from the query's run only (D-077)
            res[s] = {**{f'top{k}': float((rk < k).mean()) for k in c['eval']['topk']}, 'rank_pct': float((1 - rk / (n - 1)).mean()),
                      **{f'run_top{k}': float((rr < k).mean()) for k in c['eval']['topk']}, 'run_rank_pct': float((1 - rr / (n_run - 1)).mean()),
                      'run_chance_top10': float(np.mean(np.minimum(10, n_run) / n_run))}
    return res, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--variant'); ap.add_argument('--seed', type=int)
    ap.add_argument('--tag', default=''); ap.add_argument('--eeg-source', default=None); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--reference', choices=['none', 'car']); ap.add_argument('--max-epochs', type=int)
    ap.add_argument('--valid-only', action='store_true', help='diagnostic run: evaluate on the validation split, never the test split'); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); c['variant'] = a.variant or c['variant']; c['seed'] = a.seed or c['seed']
    if a.eeg_source: c['eeg']['source'] = a.eeg_source
    if a.reference: c['eeg']['reference'] = a.reference
    if a.max_epochs: c['optim']['max_epochs'] = a.max_epochs
    valid_only = a.smoke or a.valid_only
    feat_kind, loss_kind, subj_layers = VARIANTS[c['variant']]; o = c['optim']
    if a.smoke: o.update(updates_per_epoch=20, max_epochs=2, batch_size=32)
    name = time.strftime('%Y-%m-%d') + f"_stageA_{c['variant']}_s{c['seed']}{a.tag}{'_smoke' if a.smoke else '_diag' if a.valid_only else ''}"
    out = os.path.join(ROOT, c['out_root'], name); os.makedirs(out, exist_ok=True); yaml.safe_dump(c, open(os.path.join(out, 'config.yaml'), 'w'))
    torch.manual_seed(c['seed']); rng = np.random.default_rng(c['seed']); dev = 'cuda'; t0 = time.time()
    D = Data(c, feat_kind, a.smoke); D.to_device(dev); nS = len(D.subjects)
    print(f"{c['variant']} seed {c['seed']}: subjects {nS}, windows train {len(D.items['train'])} valid {len(D.items['valid'])} "
          f"test segments {len(D.items['test'])} ({time.time() - t0:.0f}s)", flush=True)
    m = c['model']; net = BrainModule(out_dim=D.F.shape[0], n_subjects=nS, merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                                      merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'],
                                      depth=m['depth'], subject_layers=subj_layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=o['lr']); amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=o['precision'] == 'bf16')
    best, bad_epochs, hist = float('inf'), 0, []
    nv = len(D.items['valid']); vrng = np.random.default_rng(0)
    vbatches = [(vrng.choice(nv, o['batch_size'], replace=False), vrng.integers(0, nS, o['batch_size'])) for _ in range(20 if not a.smoke else 2)]
    for ep in range(o['max_epochs']):
        net.train(); tl = 0.0
        for _ in range(o['updates_per_epoch']):
            idx = rng.choice(len(D.items['train']), o['batch_size'], replace=False); si = rng.integers(0, nS, o['batch_size'])
            X, P, S, Y = D.batch('train', idx, si)
            with torch.autocast(**amp): est = net(X.to(dev), P.to(dev), S.to(dev)).float()
            loss = loss_fn(loss_kind, est, Y.to(dev)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item()
        net.eval(); vl = 0.0
        with torch.no_grad():
            for idx, si in vbatches:
                X, P, S, Y = D.batch('valid', idx, si)
                with torch.autocast(**amp): est = net(X.to(dev), P.to(dev), S.to(dev)).float()
                vl += loss_fn(loss_kind, est, Y.to(dev)).item()
        vl /= len(vbatches); tl /= o['updates_per_epoch']; hist.append(dict(epoch=ep, train=tl, valid=vl, t=time.time() - t0))
        print(f'epoch {ep}: train {tl:.4f} valid {vl:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if vl < best: best, bad_epochs = vl, 0; torch.save(net.state_dict(), os.path.join(out, 'best.pt'))
        else:
            bad_epochs += 1
            if bad_epochs >= o['patience']: break
        json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
    net.load_state_dict(torch.load(os.path.join(out, 'best.pt'))); res, n = evaluate(net, D, dev, c, o['precision'], 'valid' if valid_only else 'test')   # smoke / diagnostic runs never touch the test split
    summ = {k: float(np.mean([r[k] for r in res.values()])) for k in next(iter(res.values()))}
    summ.update(eval_split='valid' if valid_only else 'test', n_test_segments=n, chance_top1=1 / n, chance_top10=10 / n, best_valid=best, epochs=len(hist))
    json.dump(dict(summary=summ, subjects=res), open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    print('SUMMARY', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
