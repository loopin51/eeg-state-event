"""D-092 training on Alice EEG: technical gate A′ and one subject fold x segment fold of the cross-fitted readouts.

--mode aprime: Défossez clip_w2v on the paper-style sentence-block split (blocks.tsv), all 33 subjects, D-072 registered
  preprocessing (source eeg, no re-reference, flat channels invalid), fp32, early stopping on the validation blocks
  (evaluation unit and patience from the config; D-092 supplement 8). Test queries: windows from 0.5 s before each word onset fully inside a contiguous test span; each subject's
  query is ranked against all test candidates (top-1/5/10, mean rank percentile) and against the candidates of its segment.
  The best model is scored again under bf16 autocast (report only, D-092 supplement 2 d).
--mode main --arm defossez|reve --subject-fold K --segment-fold F: the shared network is trained on the training segments
  (8; the validation segment and the 3 test segments excluded; windows near a test-fold boundary dropped, guard_s) of the
  subjects outside subject fold K, early-stopped on their validation segment; each target subject (fold K) then fits only
  its subject layer on its own training segments (everything else frozen, eval mode), early-stopped on its own validation
  segment. Test queries: word-onset windows of the test segments. Per target and segment the script stores the
  within-segment score matrix (queries x candidates of the same segment) for the main readout and for PRE (same readout on
  the EEG window pre_shift_s earlier; queries and candidates restricted to the queries whose earlier window lies in the
  segment, D-085 amendment), the rank of each query among all fold candidates, and the std across queries of the readout
  output (collapse check, recorded, not a pass / fail rule). Verdicts: scripts/alice_verdict.py.
  --eval-split valid scores the validation segment instead of the test segments (encoder learning-rate selection, smoke).
  --phaserand uses the phase-randomised copy of the arm's cache (scripts/alice_phase_random.py).
  --frozen (REVE only): encoder without gradients (report).
--smoke: tiny training on a few subjects, evaluation on the validation segment / blocks only (never the test data).
--timing: full batch sizes and subjects, 20 updates per stage, one epoch, validation evaluation only (time and memory estimate).

usage: python scripts/alice_train.py configs/alice_train.yaml --mode aprime [--smoke]
       python scripts/alice_train.py configs/alice_train.yaml --mode main --arm reve --subject-fold 0 --segment-fold 0 [--encoder-lr 3e-5]
              [--eval-split valid] [--phaserand] [--frozen] [--smoke]
"""
import os, sys, json, time, copy, argparse, yaml, numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.model.brain_clip import BrainModule, clip_loss, clip_scores, INVALID


class Alice:
    """cache access: EEG of one source (one continuous recording per subject, segments stored back to back), w2v features."""
    def __init__(self, c, source, feat_kind, subjects, arm, pre):
        d = os.path.join(ROOT, c['cache_dir']); self.c, self.arm, self.pre = c, arm, pre
        meta = json.load(open(os.path.join(d, 'eeg_meta.json'))); base = source.replace('_phaserand', '')
        src = meta['sources'][base]; self.fs = src['sample_rate']; self.offs, self.lens = src['run_offsets'], src['run_lengths']
        am = json.load(open(os.path.join(d, 'audio_meta.json'))); self.ffs = am['sample_rate']; self.foffs, self.flens = am['run_offsets'], am['run_lengths']
        self.durs = [min(L / self.fs, Lf / self.ffs) for L, Lf in zip(self.lens, self.flens)]; self.nseg = len(self.lens)
        self.subjects = list(subjects); self.eeg = {s: np.load(os.path.join(d, f'{source}_sub-{s:03d}.npy'), mmap_mode='r') for s in subjects}
        self.missing = {s: {k - 1 for k in meta['subjects'][str(s)]['missing_segments']} for s in subjects}
        inv = {}
        for s in subjects:
            m = np.array(meta['subjects'][str(s)]['sources'][base]['flat'])
            if pre.get('invalid') == 'bad_or_flat': m = m | np.array(meta['subjects'][str(s)]['bad_mask'])
            inv[s] = m
        self.invalid = inv
        if arm == 'reve': self.pos = torch.from_numpy(np.load(os.path.join(d, c['arms']['reve']['positions'])))
        else: self.pos = torch.from_numpy(np.load(os.path.join(d, 'positions.npy')))
        w = c['window']; self.T = int(round((w['tmax'] - w['tmin']) * self.fs)); self.Tf = int(round((w['tmax'] - w['tmin']) * self.ffs))
        self.F = np.load(os.path.join(d, feat_kind + '.npy'), mmap_mode='r')
        self.words = np.load(os.path.join(d, 'words.npz'), allow_pickle=True)
        self.gstart = np.r_[0, np.cumsum(self.durs)][:-1]          # concatenated time of each segment start (boundary guard)

    def lo(self): return -self.c['window']['tmin']
    def hi(self, r): w = self.c['window']; return self.durs[r] - w['tmax'] - max(w['offset_s'], 0)

    def grid(self, r, a=0.0, b=None):
        """windows every train_step_s whose feature window lies in [a, b] and whose EEG window lies in the segment"""
        w = self.c['window']; b = self.durs[r] if b is None else b
        t0, t1 = max(self.lo(), a - w['tmin']), min(self.hi(r), b - w['tmax'])
        return [(r, float(t)) for t in np.arange(t0, t1 + 1e-9, w['train_step_s'])]

    def onsets(self, r, a=0.0, b=None):
        w = self.c['window']; b = self.durs[r] if b is None else b; on = np.unique(self.words['onset'][self.words['seg'] == r])
        return [(r, float(t)) for t in on[(on >= max(self.lo(), a - w['tmin'])) & (on <= min(self.hi(r), b - w['tmax']))]]

    def guard_ok(self, items, test_segs):
        """False for windows whose EEG span is within guard_s of the start or end of the test fold (concatenated time)"""
        if not items: return np.zeros(0, bool)
        w = self.c['window']; b0 = self.gstart[min(test_segs)]; b1 = self.gstart[max(test_segs)] + self.durs[max(test_segs)]
        s0 = np.array([self.gstart[r] + t + w['tmin'] + w['offset_s'] for r, t in items]); s1 = s0 + (w['tmax'] - w['tmin'])
        def dist(b): return np.where((s0 <= b) & (s1 >= b), 0.0, np.minimum(np.abs(s0 - b), np.abs(s1 - b)))
        return np.minimum(dist(b0), dist(b1)) >= self.c['guard_s']

    def feat_norm(self, segs_spans):
        """feature mean / std over the samples of [(segment, a, b)] spans"""
        idx = np.concatenate([np.arange(self.foffs[r] + int(round(a * self.ffs)), self.foffs[r] + min(int(round(b * self.ffs)), self.flens[r])) for r, a, b in segs_spans])
        sub = np.sort(idx[:: max(1, len(idx) // 200000)]); Fs = np.asarray(self.F[:, sub], np.float32)
        self.fmean, self.fstd = Fs.mean(1, keepdims=True), Fs.std(1, keepdims=True) + 1e-6

    def to_device(self, dev):
        self.dev = dev; N = self.F.shape[1]; self.Fg = torch.empty(self.F.shape, dtype=torch.float16, device=dev)
        m, sd = torch.from_numpy(self.fmean).to(dev), torch.from_numpy(self.fstd).to(dev)
        for a in range(0, N, 32768):
            self.Fg[:, a:a + 32768] = ((torch.from_numpy(np.asarray(self.F[:, a:a + 32768], np.float32)).to(dev) - m) / sd).half()
        torch.cuda.empty_cache()
        self.badg = torch.from_numpy(np.stack([self.invalid[s] for s in self.subjects])).to(dev); self.posg = self.pos.to(dev)
        self.ar = torch.arange(self.Tf, device=dev)

    def feat_idx(self, r, t): return self.foffs[r] + int(round((t + self.c['window']['tmin']) * self.ffs))

    def targets(self, items):
        fi = torch.as_tensor([self.feat_idx(r, t) for r, t in items], device=self.dev)
        return self.Fg[:, fi[:, None] + self.ar].permute(1, 0, 2).float()

    def batch(self, items, subj_idx, eeg_shift=0.0):
        w, p = self.c['window'], self.pre; X, fi = [], []
        for (r, t), si in zip(items, subj_idx):
            i = self.offs[r] + int(round((t + w['tmin'] + w['offset_s'] + eeg_shift) * self.fs))
            X.append(self.eeg[self.subjects[si]][:, i:i + self.T]); fi.append(self.feat_idx(r, t))
        X = torch.from_numpy(np.stack(X).astype(np.float32)).to(self.dev, non_blocking=True)
        S = torch.as_tensor(np.asarray(subj_idx), device=self.dev); Y = self.Fg[:, torch.as_tensor(fi, device=self.dev)[:, None] + self.ar].permute(1, 0, 2).float()
        if self.arm == 'reve': return X, self.posg[None].expand(len(S), -1, -1), Y
        bad = self.badg[S]
        if p['reference'] == 'car':
            ok = (~bad).float()[..., None]; X = X - (X * ok).sum(1, keepdim=True) / ok.sum(1, keepdim=True)
        X = X - X[..., : int(round(p['baseline_s'] * self.fs))].mean(-1, keepdim=True); X = X.clamp(-p['clamp'], p['clamp'])
        return X, self.posg[None].expand(len(S), -1, -1).masked_fill(bad[..., None], INVALID), Y


def build_net(c, arm, n_subjects, D, dev):
    if arm == 'reve':
        from ese.model.reve_readout import ReveReadout
        a = c['arms']['reve']; net = ReveReadout(a['encoder'], n_subjects, D.F.shape[0], D.Tf, D.T, D.fs, D.ffs, a['head_rank'], a['checkpointing'])
        net.load_encoder(os.path.join(ROOT, a['weights'])); return net.to(dev)
    m = c['defossez_model']
    return BrainModule(out_dim=D.F.shape[0], n_subjects=n_subjects, merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                       merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'], depth=m['depth'],
                       subject_layers=True, n_in=len(D.pos)).to(dev)


def run_epochs(step, valid_loss, max_epochs, patience, log, tag, min_epochs=0):
    """one 'epoch' = one evaluation unit (updates_per_epoch updates); stop after `patience` units without improvement, but not
    before min_epochs units (D-092 supplement 8: 150-update units, patience 4, no stop before 1,200 updates)"""
    best, bad, best_state = float('inf'), 0, None
    for ep in range(max_epochs):
        t0 = time.time(); tl = step(); vl, state = valid_loss(); log.append(dict(stage=tag, epoch=ep, train=tl, valid=vl, seconds=time.time() - t0))
        print(f'{tag} epoch {ep}: train {tl:.4f} valid {vl:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if vl < best: best, bad, best_state = vl, 0, state
        else:
            bad += 1
            if bad >= patience and ep + 1 >= min_epochs: break
    return best, best_state


def pairs_for(D, items, subj_idx_list):
    """(item, subject index) pairs whose segment exists for that subject"""
    return np.array([(i, si) for si in subj_idx_list for i, (r, _) in enumerate(items) if r not in D.missing[D.subjects[si]]], int).reshape(-1, 2)


class Trainer:
    def __init__(self, net, D, amp, rng):
        self.net, self.D, self.amp, self.rng = net, D, amp, rng

    def fwd(self, items, si, shift=0.0):
        X, P, Y = self.D.batch(items, si, shift)
        with torch.autocast(**self.amp): return self.net(X, P, torch.as_tensor(np.asarray(si), device=self.D.dev)).float(), Y

    def epoch(self, opt, items, pairs, bs, n_updates, train_mode=True):
        self.net.train(train_mode); tl = 0.0
        for _ in range(n_updates):
            pr = pairs[self.rng.choice(len(pairs), bs, replace=len(pairs) < bs)]
            est, Y = self.fwd([items[i] for i in pr[:, 0]], pr[:, 1]); loss = clip_loss(est, Y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item()
        return tl / n_updates

    def vloss(self, items, batches, snapshot):
        self.net.eval(); vl = 0.0
        with torch.no_grad():
            for pr in batches:
                est, Y = self.fwd([items[i] for i in pr[:, 0]], pr[:, 1]); vl += clip_loss(est, Y).item()
        return vl / len(batches), snapshot()


def fixed_batches(pairs, bs, n, rng):
    return [pairs[rng.choice(len(pairs), min(bs, len(pairs)), replace=False)] for _ in range(n)]


def score_rows(tr, items, si, Yc, chunk, shift=0.0):
    """scores [len(items), len(Yc)] and the raw outputs' std across queries (mean over units)"""
    rows, outs = [], []
    with torch.no_grad():
        for b in range(0, len(items), chunk):
            est, _ = tr.fwd(items[b:b + chunk], np.full(len(items[b:b + chunk]), si), shift)
            rows.append(torch.cat([clip_scores(est, Yc[cc:cc + 512].float()) for cc in range(0, len(Yc), 512)], 1)); outs.append(est.flatten(1).float())
    Z = torch.cat(outs); return torch.cat(rows), float(Z.std(0).mean()) if len(Z) > 1 else float('nan')


# ------------------------------------------------------------------------------------------------ A′
def run_aprime(c, a, out, dev):
    o = dict(c['aprime']); pre = dict(reference=o['reference'], invalid=o['invalid'], clamp=o['clamp'], baseline_s=o['baseline_s'])
    meta = json.load(open(os.path.join(ROOT, c['cache_dir'], 'eeg_meta.json'))); subjects = sorted(int(s) for s in meta['subjects'])
    if a.smoke: subjects = [1, 3, 26]; o.update(updates_per_epoch=20, max_epochs=2, batch_size=32, min_epochs=0)
    if a.timing: o.update(updates_per_epoch=20, max_epochs=1, min_epochs=0)
    D = Alice(c, o['source'], c['features']['aprime'], subjects, 'defossez', pre); B = pd.read_csv(os.path.join(ROOT, c['cache_dir'], 'blocks.tsv'), sep='\t')
    def spans(split, r):
        out_, b = [], B[B.run == r].sort_values('start')
        for _, x in b.iterrows():
            if x.split != split: continue
            if out_ and abs(out_[-1][1] - x.start) < 1e-6: out_[-1][1] = x.stop
            else: out_.append([x.start, x.stop])
        return out_
    items = {sp: [] for sp in ('train', 'valid', 'test')}
    for r in range(D.nseg):
        for sp in ('train', 'valid'): items[sp] += [it for a0, b0 in spans(sp, r) for it in D.grid(r, a0, b0)]
        items['test'] += [it for a0, b0 in spans('test', r) for it in D.onsets(r, a0, b0)]
    eval_split = 'valid' if a.smoke or a.timing else 'test'
    if eval_split == 'valid':   # word-onset queries of the validation blocks
        items['eval'] = [it for r in range(D.nseg) for a0, b0 in spans('valid', r) for it in D.onsets(r, a0, b0)]
    else: items['eval'] = items['test']
    D.feat_norm([(r, a0, b0) for r in range(D.nseg) for a0, b0 in spans('train', r)]); D.to_device(dev); nS = len(subjects)
    rng = np.random.default_rng(c['seed']); torch.manual_seed(c['seed'])
    trp, vap = pairs_for(D, items['train'], range(nS)), pairs_for(D, items['valid'], range(nS))
    print(f"A′: subjects {nS}, windows train {len(items['train'])} valid {len(items['valid'])}, {eval_split} queries {len(items['eval'])}", flush=True)
    net = build_net(c, 'defossez', nS, D, dev); amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=o['precision'] == 'bf16')
    tr = Trainer(net, D, amp, rng); opt = torch.optim.Adam(net.parameters(), lr=o['lr']); vb = fixed_batches(vap, o['batch_size'], 2 if a.smoke else 20, np.random.default_rng(0))
    hist = []; snap = lambda: copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
    best, state = run_epochs(lambda: tr.epoch(opt, items['train'], trp, o['batch_size'], o['updates_per_epoch']),
                             lambda: tr.vloss(items['valid'], vb, snap), o['max_epochs'], o['patience'], hist, 'shared', o.get('min_epochs', 0))
    net.load_state_dict(state); torch.save(state, os.path.join(out, 'best.pt')); json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
    q = items['eval']; n = len(q); seg = np.array([r for r, _ in q]); Yc = torch.stack([D.Fg[:, D.feat_idx(r, t) + D.ar] for r, t in q])
    res = {}
    for prec in ['fp32'] + (['bf16'] if o['report_bf16_eval'] else []):
        tr.amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=prec == 'bf16'); net.eval(); per = {}
        for si, s in enumerate(subjects):
            keep = np.array([r not in D.missing[s] for r in seg]); idx = np.where(keep)[0]
            Sm, ostd = score_rows(tr, [q[i] for i in idx], si, Yc[torch.as_tensor(idx, device=dev)], c['eval']['score_chunk'])
            true = Sm.diagonal(); above = Sm > true[:, None]; rk = above.sum(1).cpu().numpy(); nk = len(idx)
            same = torch.as_tensor(seg[idx][:, None] == seg[idx][None], device=dev); rr = (above & same).sum(1).cpu().numpy(); nr = same.sum(1).cpu().numpy()
            per[s] = dict(top1=float((rk < 1).mean()), top5=float((rk < 5).mean()), top10=float((rk < 10).mean()), rank_pct=float((1 - rk / (nk - 1)).mean()),
                          seg_top10=float((rr < 10).mean()), seg_rank_pct=float((1 - rr / np.maximum(nr - 1, 1)).mean()), n_queries=nk, output_std=ostd)
        res[prec] = dict(subjects=per, **{k: float(np.mean([v[k] for v in per.values()])) for k in ('top1', 'top5', 'top10', 'rank_pct', 'seg_top10', 'seg_rank_pct', 'output_std')})
        print(prec, json.dumps({k: v for k, v in res[prec].items() if k != 'subjects'}), flush=True)
    summ = dict(eval_split=eval_split, n_queries=n, best_valid=best, epochs=sum(h['stage'] == 'shared' for h in hist),
                best_epoch=int(np.argmin([h['valid'] for h in hist])), updates_per_unit=o['updates_per_epoch'],
                best_update=(int(np.argmin([h['valid'] for h in hist])) + 1) * o['updates_per_epoch'], stop_top10=o['stop_top10'], below_stop=res['fp32']['top10'] < o['stop_top10'])
    if 'bf16' in res: summ['bf16_minus_fp32_top10'] = res['bf16']['top10'] - res['fp32']['top10']
    json.dump(dict(summary=summ, **res), open(os.path.join(out, 'metrics.json'), 'w'), indent=1); print('SUMMARY', json.dumps(summ), flush=True)


# ------------------------------------------------------------------------------------------------ main split
def run_main(c, a, out, dev, run):
    arm = a.arm; ac = dict(c['arms'][arm]); o, cal = dict(c['main']), dict(c['calibration'])
    sp = json.load(open(os.path.join(ROOT, c['cache_dir'], 'split.json')))['main']; k, f = a.subject_fold, a.segment_fold
    test = [x - 1 for x in sp['segment_folds'][f]]; valid = sp['valid_segment'][f] - 1; train = [r for r in range(12) if r not in test + [valid]]
    targets = list(sp['subject_folds'][k]); trainers = sorted(s for j, fs in enumerate(sp['subject_folds']) if j != k for s in fs)
    if a.smoke:
        trainers, targets = trainers[:3], targets[:2]; o.update(updates_per_epoch=10, max_epochs=2, batch_size=16, valid_batches=2, min_epochs=0)
        cal.update(updates_per_epoch=5, max_epochs=2, batch_size=16, min_epochs=0)
    if a.timing: o.update(updates_per_epoch=20, max_epochs=1, min_epochs=0); cal.update(updates_per_epoch=20, max_epochs=1, min_epochs=0)
    eval_segs = [valid] if a.eval_split == 'valid' else test
    subjects = trainers + targets; nT = len(trainers); source = ac['source'] + ('_phaserand' if a.phaserand else '')
    D = Alice(c, source, c['features']['main'], subjects, arm, ac)
    def gitems(segs):
        it = [x for r in segs for x in D.grid(r)]; ok = D.guard_ok(it, test); return [x for x, g in zip(it, ok) if g]
    tr_items, va_items = gitems(train), gitems([valid])
    D.feat_norm([(r, 0.0, D.durs[r]) for r in train]); D.to_device(dev)
    run.update(trainers=trainers, targets=targets, train_segments=[r + 1 for r in train], valid_segment=valid + 1, test_segments=[r + 1 for r in test],
               eval_segments=[r + 1 for r in eval_segs], source=source)
    rng = np.random.default_rng(c['seed']); torch.manual_seed(c['seed']); vr = np.random.default_rng(0)
    trp, vap = pairs_for(D, tr_items, range(nT)), pairs_for(D, va_items, range(nT))
    print(f'{arm} subject fold {k} segment fold {f}: trainers {len(trainers)} targets {targets} source {source}; windows train {len(tr_items)} '
          f'valid {len(va_items)} (pairs {len(trp)}, {len(vap)}); eval segments {[r + 1 for r in eval_segs]}', flush=True)
    net = build_net(c, arm, len(subjects), D, dev); amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=o['precision'] == 'bf16')
    tr = Trainer(net, D, amp, rng); hist = []; t0 = time.time()
    if arm == 'reve':
        enc = list(net.enc.parameters()); rest = [p for n_, p in net.named_parameters() if not n_.startswith('enc.')]
        if a.frozen:
            for p in enc: p.requires_grad_(False)
            opt = torch.optim.Adam(rest, lr=ac['lr_head'])
        else:
            lr_enc = a.encoder_lr or ac['encoder_lr']; assert lr_enc, 'encoder_lr not set (D-092 selection)'; run['encoder_lr'] = lr_enc
            opt = torch.optim.Adam([dict(params=enc, lr=lr_enc), dict(params=rest, lr=ac['lr_head'])])
    else: opt = torch.optim.Adam(net.parameters(), lr=ac['lr'])
    yaml.safe_dump(dict(c, run=run), open(os.path.join(out, 'config.yaml'), 'w'))
    snap = lambda: copy.deepcopy({kk: v.detach().cpu() for kk, v in net.state_dict().items()})
    vb = fixed_batches(vap, o['batch_size'], o['valid_batches'], vr)
    best, state = run_epochs(lambda: tr.epoch(opt, tr_items, trp, o['batch_size'], o['updates_per_epoch']),
                             lambda: tr.vloss(va_items, vb, snap), o['max_epochs'], o['patience'], hist, 'shared', o.get('min_epochs', 0))
    net.load_state_dict(state); del state
    torch.save({kk: (v.half() if arm == 'reve' and v.is_floating_point() else v) for kk, v in net.state_dict().items()}, os.path.join(out, 'shared.pt'))
    print(f'shared done: best valid {best:.4f} ({time.time() - t0:.0f}s)', flush=True)
    # target subject layers
    for p in net.parameters(): p.requires_grad_(False)
    W = net.subject.weights; W.requires_grad_(True); cal_best, cal_valid = {}, {}
    with torch.no_grad(): W[nT:] = W[:nT].mean(0, keepdim=True)
    for j, s in enumerate(targets):
        ti = nT + j; v_seg = valid
        if valid in D.missing[s]:                                          # registered fallback (configs/alice_train.yaml)
            v_seg = next(r for r in [(valid + i) % 12 for i in range(1, 12)] if r not in test and r not in D.missing[s])
        own_tr = [x for x in tr_items if x[0] != v_seg and x[0] not in D.missing[s]]; own_va = gitems([v_seg]); cal_valid[s] = v_seg + 1
        ctp = np.array([(i, ti) for i in range(len(own_tr))]); cvb = fixed_batches(np.array([(i, ti) for i in range(len(own_va))]), cal['batch_size'], o['valid_batches'], vr)
        copt = torch.optim.Adam([W], lr=cal['lr'])
        if arm == 'reve':   # the encoder and pooling are frozen here: cache their output once (same values in eval mode)
            net.eval(); St = torch.full((1,), ti, device=dev)
            def zcache(its):
                out_ = []
                with torch.no_grad():
                    for b0 in range(0, len(its), c['eval']['score_chunk']):
                        X, P, _ = D.batch(its[b0:b0 + c['eval']['score_chunk']], np.full(len(its[b0:b0 + c['eval']['score_chunk']]), ti))
                        with torch.autocast(**amp): out_.append(net.pooled(X, P).float())
                return torch.cat(out_)
            Ztr, Zva = zcache(own_tr), zcache(own_va)
            def zloss(Zc, its, idx):
                with torch.autocast(**amp): est = net.decode(Zc[torch.as_tensor(idx, device=dev)], St.expand(len(idx))).float()
                return clip_loss(est, D.targets([its[i] for i in idx]))
            def cstep():
                tl = 0.0
                for _ in range(cal['updates_per_epoch']):
                    idx = rng.choice(len(own_tr), cal['batch_size'], replace=len(own_tr) < cal['batch_size']); loss = zloss(Ztr, own_tr, idx)
                    copt.zero_grad(set_to_none=True); loss.backward(); copt.step(); tl += loss.item()
                return tl / cal['updates_per_epoch']
            def cvloss():
                with torch.no_grad(): vl = sum(zloss(Zva, own_va, pr[:, 0]).item() for pr in cvb) / len(cvb)
                return vl, W[ti].detach().clone()
        else:
            cstep = lambda: tr.epoch(copt, own_tr, ctp, cal['batch_size'], cal['updates_per_epoch'], train_mode=False)
            def cvloss(): vl, _ = tr.vloss(own_va, cvb, lambda: None); return vl, W[ti].detach().clone()
        init_vl, _ = cvloss(); hist.append(dict(stage=f'cal_{s}', epoch=-1, train=None, valid=init_vl))
        b, row = run_epochs(cstep, cvloss, cal['max_epochs'], cal['patience'], hist, f'cal_{s}', cal.get('min_epochs', 0))
        if init_vl <= b: b, row = init_vl, W[ti].detach().clone()
        with torch.no_grad(): W[ti] = row
        cal_best[s] = b
    W.requires_grad_(False); torch.save(W.detach().cpu()[nT:], os.path.join(out, 'target_layers.pt')); json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
    # evaluation
    net.eval(); pre_s = c['eval']['pre_shift_s']; w = c['window']; arrays, res = {}, {}
    q_all = [x for r in eval_segs for x in D.onsets(r)]; seg_all = np.array([r for r, _ in q_all])
    Yc_all = torch.stack([D.Fg[:, D.feat_idx(r, t) + D.ar] for r, t in q_all])
    for r in eval_segs:
        m = seg_all == r; on = np.array([t for (rr, t) in q_all if rr == r]); arrays[f'onset_r{r + 1}'] = on
        arrays[f'pre_rows_r{r + 1}'] = np.where(on + w['tmin'] + w['offset_s'] - pre_s >= 0)[0]
    for j, s in enumerate(targets):
        ti = nT + j; segs = [r for r in eval_segs if r not in D.missing[s]]; res[str(s)] = dict(calibration_best_valid=cal_best[s], calibration_valid_segment=cal_valid[s], segments={})
        keep = np.where(np.isin(seg_all, segs))[0]; qk = [q_all[i] for i in keep]
        Sfull, ostd = score_rows(tr, qk, ti, Yc_all[torch.as_tensor(keep, device=dev)], c['eval']['score_chunk'])
        rank_full = (Sfull > Sfull.diagonal()[:, None]).sum(1).cpu().numpy(); arrays[f'main_s{s}_rank_full'] = rank_full; arrays[f'main_s{s}_segs'] = seg_all[keep] + 1
        for r in segs:
            ii = np.where(seg_all[keep] == r)[0]; it = torch.as_tensor(ii, device=dev); S = Sfull[it][:, it].cpu().numpy(); arrays[f'main_s{s}_r{r + 1}'] = S
            rows = arrays[f'pre_rows_r{r + 1}']; qr = [qk[i] for i in ii[rows]]; Yr = Yc_all[torch.as_tensor(keep[ii[rows]], device=dev)]
            P, pstd = score_rows(tr, qr, ti, Yr, c['eval']['score_chunk'], -pre_s); P = P.cpu().numpy(); arrays[f'pre_s{s}_r{r + 1}'] = P
            def summ(M):
                rk = (M > np.diag(M)[:, None]).sum(1); n = len(M)
                return dict(mrp=float(np.mean(1 - rk / (n - 1))), top1=float((rk < 1).mean()), top10=float((rk < 10).mean()), n=n,
                            score_col_std=float(M.std(0).mean()))
            res[str(s)]['segments'][str(r + 1)] = dict(main=summ(S), pre=summ(P))
        cells = res[str(s)]['segments'].values(); nn_ = sum(v['main']['n'] for v in cells)
        res[str(s)].update(mrp=sum(v['main']['mrp'] * v['main']['n'] for v in cells) / nn_, pre_mrp=sum(v['pre']['mrp'] * v['pre']['n'] for v in cells) / sum(v['pre']['n'] for v in cells),
                           top10_full=float((rank_full < 10).mean()), output_std=ostd)
        print(f'target S{s:02d}: mrp {res[str(s)]["mrp"]:.4f} pre {res[str(s)]["pre_mrp"]:.4f} output std {ostd:.2e} ({time.time() - t0:.0f}s)', flush=True)
    np.savez_compressed(os.path.join(out, 'scores.npz'), **arrays)
    summ = dict(eval_split=a.eval_split, eval_segments=[r + 1 for r in eval_segs], shared_best_valid=best, shared_epochs=sum(h['stage'] == 'shared' for h in hist),
                shared_best_epoch=int(np.argmin([h['valid'] for h in hist if h['stage'] == 'shared'])), updates_per_unit=o['updates_per_epoch'], seconds=time.time() - t0,
                peak_gpu_gb=torch.cuda.max_memory_reserved() / 2 ** 30)
    json.dump(dict(summary=summ, targets=res), open(os.path.join(out, 'metrics.json'), 'w'), indent=1); print('SUMMARY', json.dumps(summ), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--mode', required=True, choices=['aprime', 'main'])
    ap.add_argument('--arm', choices=['defossez', 'reve']); ap.add_argument('--subject-fold', type=int); ap.add_argument('--segment-fold', type=int)
    ap.add_argument('--encoder-lr', type=float); ap.add_argument('--eval-split', default='test', choices=['test', 'valid'])
    ap.add_argument('--phaserand', action='store_true'); ap.add_argument('--frozen', action='store_true'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--timing', action='store_true'); ap.add_argument('--gpu-budget-gb', type=float, default=10.5); a = ap.parse_args(); c = yaml.safe_load(open(a.config))
    if a.smoke or a.timing: a.eval_split = 'valid'                                      # smoke / timing runs never touch the test data
    dev = 'cuda'; tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30; torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_budget_gb / tot))
    if a.mode == 'aprime': name = f"{c['aprime'].get('run_name', 'alice_aprime')}_s{c['seed']}"
    else:
        assert a.frozen is False or a.arm == 'reve'
        lr = f"_lr{a.encoder_lr:g}" if a.encoder_lr else ''
        name = f"alice_{a.arm}_k{a.subject_fold}_f{a.segment_fold}{lr}{'_frozen' if a.frozen else ''}{'_phaserand' if a.phaserand else ''}{'_valid' if a.eval_split == 'valid' else ''}"
    out = os.path.join(ROOT, c['out_root'], time.strftime('%Y-%m-%d') + '_' + name + ('_smoke' if a.smoke else '_timing' if a.timing else '')); os.makedirs(out, exist_ok=True)
    run = dict(mode=a.mode, arm=a.arm, subject_fold=a.subject_fold, segment_fold=a.segment_fold, encoder_lr=a.encoder_lr, eval_split=a.eval_split,
               phaserand=a.phaserand, frozen=a.frozen, smoke=a.smoke, timing=a.timing)
    if a.mode == 'aprime': yaml.safe_dump(dict(c, run=run), open(os.path.join(out, 'config.yaml'), 'w')); run_aprime(c, a, out, dev)
    else: run_main(c, a, out, dev, run)


if __name__ == '__main__':
    main()
