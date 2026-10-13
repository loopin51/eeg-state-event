"""Stage B B1 (D-085): one subject fold x run fold of the subject cross-fitted, run-split Défossez decoder on ds004408.

The shared network is trained on the training runs (14) of the pool subjects outside subject fold k, with early stopping on
their validation runs (2). Each target subject (fold k, plus the never-trained extra targets) then gets its own subject layer
fitted on its training runs with everything else frozen (eval mode), initialised at the mean of the training subjects'
layers and early-stopped on its validation runs. Test queries: 3-s windows starting 0.5 s before every word onset of the
test runs (4). For every target the script stores, per test run, the within-run CLIP score matrix (queries x candidates of
the same run, float32), and per query the rank of the true segment among all fold candidates and among other-run
candidates; the same for the PRE meter (same decoder on the EEG window pre_shift_s earlier). Verdicts are computed by
scripts/stageB_b1_verdict.py from these files.

usage: python scripts/stageB_b1_train.py configs/stageB_b1.yaml --subject-fold K --run-fold F [--phaserand] [--smoke]
       --smoke: tiny training, evaluation on the validation runs only (never the test runs)
"""
import os, sys, json, time, copy, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.model.brain_clip import BrainModule, clip_loss, clip_scores, INVALID


def check_setting(c):
    """D-085: A4 only if D-084 judged A4 'pass' (19 subjects); otherwise the registered setting."""
    v = os.path.join(ROOT, 'results/stageA_verdict/verdict_A4.json')
    a4_pass = os.path.exists(v) and json.load(open(v))['pools']['19']['D072'] == 'pass'
    want = 'A4' if a4_pass else 'registered'
    assert c['decoder_setting'] == want, f"D-085 rule: decoder_setting must be {want} (D-084 verdict file {'found' if os.path.exists(v) else 'missing'})"


class RunData:
    def __init__(self, c, subjects, train_runs, valid_runs, eval_runs, source):
        d = os.path.join(ROOT, c['cache_dir']); self.c = c; meta = json.load(open(os.path.join(d, 'eeg_meta.json')))
        self.fs = meta['sample_rate']; self.offs = meta['run_offsets']; self.lens = meta['run_lengths']; self.subjects = subjects
        self.eeg = {s: np.load(os.path.join(d, f'{source}_sub-{s:03d}.npy'), mmap_mode='r') for s in subjects}
        self.invalid = {s: np.array(meta['subjects'][str(s)]['flat']) for s in subjects}
        self.pos = torch.from_numpy(np.load(os.path.join(d, 'positions.npy')))
        self.F = np.load(os.path.join(d, c['features']['w2v'] + '.npy'), mmap_mode='r')
        w, e = c['window'], c['eeg']; self.T = int(round((w['tmax'] - w['tmin']) * self.fs))
        self.lo = -w['tmin']                                              # feature window and EEG window inside the run
        W = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True)
        def hi(r): return self.lens[r] / self.fs - w['tmax'] - max(e['offset_s'], 0)
        def grid(r): return np.arange(self.lo, hi(r) + 1e-9, w['train_step_s'])
        self.items = {'train': [(r, t) for r in train_runs for t in grid(r)], 'valid': [(r, t) for r in valid_runs for t in grid(r)]}
        q = []
        for r in eval_runs:
            on = np.unique(W['onset'][W['run'] == r]); q += [(r, float(t)) for t in on[(on >= self.lo) & (on <= hi(r))]]
        self.items['eval'] = q
        tr = np.concatenate([np.arange(self.offs[r], self.offs[r] + self.lens[r]) for r in train_runs])
        sub = np.sort(tr[:: max(1, len(tr) // 200000)]); Fs = np.asarray(self.F[:, sub], np.float32)
        self.fmean, self.fstd = Fs.mean(1, keepdims=True), Fs.std(1, keepdims=True) + 1e-6

    def to_device(self, dev):
        self.dev = dev; N = self.F.shape[1]; self.Fg = torch.empty(self.F.shape, dtype=torch.float16, device=dev)
        m, sd = torch.from_numpy(self.fmean).to(dev), torch.from_numpy(self.fstd).to(dev)
        for a in range(0, N, 32768):
            self.Fg[:, a:a + 32768] = ((torch.from_numpy(np.asarray(self.F[:, a:a + 32768], np.float32)).to(dev) - m) / sd).half()
        torch.cuda.empty_cache()
        self.badg = torch.from_numpy(np.stack([self.invalid[s] for s in self.subjects])).to(dev)
        self.posg = self.pos.to(dev); self.ar = torch.arange(self.T, device=dev)

    def feat_idx(self, r, t): return self.offs[r] + int(round((t + self.c['window']['tmin']) * self.fs))

    def batch(self, items, subj_idx, eeg_shift=0.0):
        """items: list of (run, t); subj_idx: index into self.subjects; eeg_shift moves only the EEG window (PRE meter)."""
        e = self.c['eeg']; w = self.c['window']; X, fi, runs = [], [], []
        for (r, t), si in zip(items, subj_idx):
            i = self.offs[r] + int(round((t + w['tmin'] + e['offset_s'] + eeg_shift) * self.fs))
            X.append(self.eeg[self.subjects[si]][:, i:i + self.T]); fi.append(self.feat_idx(r, t)); runs.append(r)
        X = torch.from_numpy(np.stack(X).astype(np.float32)).to(self.dev, non_blocking=True)
        S = torch.as_tensor(np.asarray(subj_idx), device=self.dev); bad = self.badg[S, torch.as_tensor(runs, device=self.dev)]
        if e['reference'] == 'car':
            ok = (~bad).float()[..., None]; X = X - (X * ok).sum(1, keepdim=True) / ok.sum(1, keepdim=True)
        X = X - X[..., : int(round(e['baseline_s'] * self.fs))].mean(-1, keepdim=True); X = X.clamp(-e['clamp'], e['clamp'])
        P = self.posg[None].expand(len(S), -1, -1).masked_fill(bad[..., None], INVALID)
        Y = self.Fg[:, torch.as_tensor(fi, device=self.dev)[:, None] + self.ar].permute(1, 0, 2).float()
        return X, P, Y


def run_epochs(step, valid_loss, max_epochs, patience, log, tag):
    """generic early-stopping loop; returns the best state snapshot taken by valid_loss's caller via `log` entries."""
    best, bad, best_state = float('inf'), 0, None
    for ep in range(max_epochs):
        tl = step(); vl, state = valid_loss(); log.append(dict(stage=tag, epoch=ep, train=tl, valid=vl))
        print(f'{tag} epoch {ep}: train {tl:.4f} valid {vl:.4f}', flush=True)
        if vl < best: best, bad, best_state = vl, 0, state
        else:
            bad += 1
            if bad >= patience: break
    return best, best_state


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--subject-fold', type=int, required=True)
    ap.add_argument('--run-fold', type=int, required=True); ap.add_argument('--phaserand', action='store_true'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); c = yaml.safe_load(open(a.config)); sp = c['split']; k, f = a.subject_fold, a.run_fold
    if not a.smoke: check_setting(c)
    st = c['settings'][c['decoder_setting']]; c['eeg']['reference'] = st['reference']; source = st['phaserand_source' if a.phaserand else 'source']
    test_runs = [r - 1 for r in sp['run_folds'][f]]; valid_runs = [r - 1 for r in sp['valid_runs'][f]]
    train_runs = [r for r in range(20) if r not in test_runs + valid_runs]
    targets = sp['subject_folds'][k] + sp.get('extra_targets', {}).get(k, [])
    trainers = sorted(s for j, fs in enumerate(sp['subject_folds']) if j != k for s in fs)
    o, cal = dict(c['optim']), dict(c['calibration'])
    if a.smoke:
        trainers = trainers[:3]; targets = targets[:2]; o.update(updates_per_epoch=20, max_epochs=2, batch_size=32)
        cal.update(updates_per_epoch=10, max_epochs=2, batch_size=32)
    eval_runs = valid_runs if a.smoke else test_runs                    # smoke runs never touch the test runs
    name = time.strftime('%Y-%m-%d') + f"_stageB_b1_{c['decoder_setting']}_k{k}_f{f}{'_phaserand' if a.phaserand else ''}{'_smoke' if a.smoke else ''}"
    out = os.path.join(ROOT, c['out_root'], name); os.makedirs(out, exist_ok=True)
    yaml.safe_dump(dict(c, run=dict(subject_fold=k, run_fold=f, phaserand=a.phaserand, smoke=a.smoke, source=source, trainers=trainers,
                                    targets=targets, train_runs=train_runs, valid_runs=valid_runs, eval_runs=eval_runs)), open(os.path.join(out, 'config.yaml'), 'w'))
    torch.manual_seed(c['seed']); rng = np.random.default_rng(c['seed']); dev = 'cuda'; t0 = time.time()
    subjects = trainers + targets; nT = len(trainers)
    D = RunData(c, subjects, train_runs, valid_runs, eval_runs, source); D.to_device(dev)
    print(f'subject fold {k} run fold {f}: trainers {trainers} targets {targets} source {source}; windows train {len(D.items["train"])} '
          f'valid {len(D.items["valid"])} eval queries {len(D.items["eval"])} ({time.time() - t0:.0f}s)', flush=True)
    m = c['model']; net = BrainModule(out_dim=D.F.shape[0], n_subjects=len(subjects), merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                                      merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'],
                                      depth=m['depth'], subject_layers=True).to(dev)
    amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=o['precision'] == 'bf16'); hist = []
    tr_items, va_items = D.items['train'], D.items['valid']; vr = np.random.default_rng(0)

    def fwd(items, si, shift=0.0):
        X, P, Y = D.batch(items, si, shift)
        with torch.autocast(**amp): return net(X, P, torch.as_tensor(np.asarray(si), device=dev)).float(), Y

    # 1. shared network on the trainers
    opt = torch.optim.Adam(net.parameters(), lr=o['lr'])
    vb = [(vr.choice(len(va_items), o['batch_size'], replace=False), vr.integers(0, nT, o['batch_size'])) for _ in range(2 if a.smoke else 20)]

    def step():
        net.train(); tl = 0.0
        for _ in range(o['updates_per_epoch']):
            idx = rng.choice(len(tr_items), o['batch_size'], replace=False); si = rng.integers(0, nT, o['batch_size'])
            est, Y = fwd([tr_items[i] for i in idx], si); loss = clip_loss(est, Y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item()
        return tl / o['updates_per_epoch']

    def vloss(batches):
        net.eval(); vl = 0.0
        with torch.no_grad():
            for idx, si in batches:
                est, Y = fwd([va_items[i] for i in idx], si); vl += clip_loss(est, Y).item()
        return vl / len(batches), copy.deepcopy({kk: v.detach().cpu() for kk, v in net.state_dict().items()})

    best, state = run_epochs(step, lambda: vloss(vb), o['max_epochs'], o['patience'], hist, 'shared')
    net.load_state_dict(state); torch.save(state, os.path.join(out, 'shared.pt'))
    print(f'shared done: best valid {best:.4f} ({time.time() - t0:.0f}s)', flush=True)

    # 2. target subject layers (everything else frozen, eval mode)
    for p in net.parameters(): p.requires_grad_(False)
    W = net.subject.weights; W.requires_grad_(True); cal_best = {}
    with torch.no_grad(): W[nT:] = W[:nT].mean(0, keepdim=True)
    for j, s in enumerate(targets):
        ti = nT + j; copt = torch.optim.Adam([W], lr=cal['lr'])
        cvb = [(vr.choice(len(va_items), cal['batch_size'], replace=False), np.full(cal['batch_size'], ti)) for _ in range(2 if a.smoke else 20)]

        def cstep():
            net.eval(); tl = 0.0
            for _ in range(cal['updates_per_epoch']):
                idx = rng.choice(len(tr_items), cal['batch_size'], replace=False)
                est, Y = fwd([tr_items[i] for i in idx], np.full(cal['batch_size'], ti)); loss = clip_loss(est, Y)
                copt.zero_grad(set_to_none=True); loss.backward(); copt.step(); tl += loss.item()
            return tl / cal['updates_per_epoch']

        def cvloss():
            vl, _ = vloss(cvb); return vl, W[ti].detach().clone()
        init_vl, _ = cvloss(); hist.append(dict(stage=f'cal_{s}', epoch=-1, train=None, valid=init_vl))
        b, row = run_epochs(cstep, cvloss, cal['max_epochs'], cal['patience'], hist, f'cal_{s}')
        if init_vl <= b: b, row = init_vl, W[ti].detach().clone()   # keep the initial layer if fitting never helped
        with torch.no_grad(): W[ti] = row
        cal_best[s] = b
    W.requires_grad_(False); torch.save(net.subject.weights.detach().cpu()[nT:], os.path.join(out, 'target_layers.pt'))
    json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)

    # 3. evaluation: score matrices against all fold candidates, main and PRE
    net.eval(); q = D.items['eval']; n = len(q); run = np.array([r for r, _ in q]); tq = np.array([t for _, t in q])
    Yc = torch.stack([D.Fg[:, D.feat_idx(r, t) + D.ar] for r, t in q])                                # [n, F, T] float16
    pre_s = c['eval']['pre_shift_s']; pre_ok = tq + c['window']['tmin'] + c['eeg']['offset_s'] - pre_s >= 0
    same = torch.as_tensor(run[:, None] == run[None], device=dev); res, arrays = {}, {'run': run, 'onset': tq, 'pre_ok': pre_ok}
    with torch.no_grad():
        for j, s in enumerate(targets):
            for kind, shift, rows in (('main', 0.0, np.arange(n)), ('pre', -pre_s, np.where(pre_ok)[0])):
                Sfull = torch.full((n, n), float('nan'), device=dev)
                for b0 in range(0, len(rows), c['eval']['score_chunk']):
                    ii = rows[b0:b0 + c['eval']['score_chunk']]; est, _ = fwd([q[i] for i in ii], np.full(len(ii), nT + j), shift)
                    Sfull[torch.as_tensor(ii, device=dev)] = torch.cat([clip_scores(est, Yc[cc:cc + 512].float()) for cc in range(0, n, 512)], 1)
                rr = torch.as_tensor(rows, device=dev); true = Sfull[rr, rr]; above = Sfull[rr] > true[:, None]
                rank_full = above.sum(1).cpu().numpy(); rank_cross = (above & ~same[rr]).sum(1).cpu().numpy()
                for r in sorted(set(run)):
                    ir = np.where(run == r)[0]; arrays[f'{kind}_s{s}_r{r}'] = Sfull[torch.as_tensor(ir, device=dev)][:, torch.as_tensor(ir, device=dev)].cpu().numpy()
                arrays[f'{kind}_s{s}_rank_full'] = np.full(n, -1); arrays[f'{kind}_s{s}_rank_full'][rows] = rank_full
                arrays[f'{kind}_s{s}_rank_cross'] = np.full(n, -1); arrays[f'{kind}_s{s}_rank_cross'][rows] = rank_cross
                within = []
                for r in sorted(set(run)):
                    Sr = arrays[f'{kind}_s{s}_r{r}']; ok = ~np.isnan(np.diag(Sr))
                    rk = (Sr[ok] > np.diag(Sr)[ok][:, None]).sum(1); within.append((rk, len(Sr)))
                rk = np.concatenate([x for x, _ in within]); nn_ = np.concatenate([np.full(len(x), m_) for x, m_ in within])
                res.setdefault(str(s), {})[kind] = {'mrp_within': float(np.mean(1 - rk / (nn_ - 1))), 'top1_within': float((rk < 1).mean()),
                                                    'top10_within': float((rk < 10).mean()), 'top10_full': float((rank_full < 10).mean()),
                                                    'n_queries': int(len(rk))}
            res[str(s)]['calibration_best_valid'] = cal_best[s]
            print(f'target sub-{s:03d}: {json.dumps(res[str(s)])} ({time.time() - t0:.0f}s)', flush=True)
    np.savez_compressed(os.path.join(out, 'scores.npz'), **arrays)
    summ = dict(eval_runs=[r + 1 for r in eval_runs], eval_split='valid' if a.smoke else 'test', n_queries=n, shared_best_valid=best,
                shared_epochs=sum(h['stage'] == 'shared' for h in hist), seconds=time.time() - t0)
    json.dump(dict(summary=summ, targets=res), open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    print('SUMMARY', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
