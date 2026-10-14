"""Stage B B2 (D-091): one subject fold x run fold of the B1 design with the language target (e5 length-residual embedding
of the window's words minus its linear acoustic prediction for this run fold, items.npz R_f{f} from
scripts/stageB_b2_prepare.py, D-091 amendment 3).

Same model, target calibration and evaluation as scripts/stageB_b1_train.py, with two changes of D-091 amendment 4 (from the
validation-run probe D-094): the shared network is evaluated every `shared_training.eval_every` updates on the same fixed
validation batches and CLIP loss as B1, early stopping after `patience_units` evaluations without improvement but never
before `min_updates`; CAR (and the spatial-attention mask) also leave out the BIDS channels.tsv `bad` channels (flat
channels are always left out). The target is the standardised
768-d language vector held constant over the 3-s window (CLIP score = time sum); training windows without words are left
out. For every target the script stores the full fold score matrix (test queries x test queries, float16) of the decoder
and of the PRE meter (EEG window pre_shift_s earlier); scripts/stageB_b2_verdict.py ranks inside the pair sets.

usage: python scripts/stageB_b2_train.py configs/stageB_b2.yaml --subject-fold K --run-fold F [--phaserand] [--smoke]
       --smoke: tiny training, evaluation on the validation runs only (never the test runs)
"""
import os, sys, json, time, copy, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.brain_clip import BrainModule, clip_loss, clip_scores, INVALID
from stageB_b1_train import RunData, run_epochs, check_setting


def run_units(step, valid_loss, every, max_updates, min_updates, patience, log, tag):
    """early stopping on the validation loss evaluated every `every` updates; no stop before min_updates"""
    best, bad, best_state, u = float('inf'), 0, None, 0
    while u < max_updates:
        tl = step(); u += every; vl, state = valid_loss(); log.append(dict(stage=tag, epoch=u // every - 1, update=u, train=tl, valid=vl))
        print(f'{tag} update {u}: train {tl:.4f} valid {vl:.4f}', flush=True)
        if vl < best: best, bad, best_state = vl, 0, state
        else:
            bad += 1
            if bad >= patience and u >= min_updates: break
    return best, best_state


class LangData(RunData):
    """RunData with the language target: items keyed by (run, round(t, 4)) into items.npz"""
    def __init__(self, c, c2, subjects, train_runs, valid_runs, eval_runs, source, run_fold):
        super().__init__(c, subjects, train_runs, valid_runs, eval_runs, source)
        Z = np.load(os.path.join(ROOT, c2['cache'], 'items.npz'), allow_pickle=True); self.key = {}
        for i, (r, t, q) in enumerate(zip(Z['run'], Z['t'], Z['is_query'])): self.key[(int(r), round(float(t), 4), bool(q))] = i
        R, ht = Z[f'R_f{run_fold}'], Z['has_text']; self.R = R
        if c2['car_exclude_bad']:
            meta = json.load(open(os.path.join(ROOT, c['cache_dir'], 'eeg_meta.json')))
            self.invalid = {s: np.array(meta['subjects'][str(s)]['flat']) | np.array(meta['subjects'][str(s)]['bad']) for s in subjects}
        self.items['train'] = [x for x in self.items['train'] if ht[self.key[(x[0], round(x[1], 4), False)]]]
        self.items['valid'] = [x for x in self.items['valid'] if ht[self.key[(x[0], round(x[1], 4), False)]]]
        tr = [self.key[(x[0], round(x[1], 4), False)] for x in self.items['train']]
        self.rmean, self.rstd = R[tr].mean(0), R[tr].std(0) + 1e-6

    def to_device(self, dev):
        self.dev = dev; self.Rg = torch.from_numpy((self.R - self.rmean) / self.rstd).float().to(dev)
        self.badg = torch.from_numpy(np.stack([self.invalid[s] for s in self.subjects])).to(dev); self.posg = self.pos.to(dev)

    def idx(self, items, query): return [self.key[(r, round(t, 4), query)] for r, t in items]

    def batch(self, items, subj_idx, eeg_shift=0.0, query=False):
        e = self.c['eeg']; w = self.c['window']; X, runs = [], []
        for (r, t), si in zip(items, subj_idx):
            i = self.offs[r] + int(round((t + w['tmin'] + e['offset_s'] + eeg_shift) * self.fs))
            X.append(self.eeg[self.subjects[si]][:, i:i + self.T]); runs.append(r)
        X = torch.from_numpy(np.stack(X).astype(np.float32)).to(self.dev, non_blocking=True)
        S = torch.as_tensor(np.asarray(subj_idx), device=self.dev); bad = self.badg[S, torch.as_tensor(runs, device=self.dev)]
        if e['reference'] == 'car':
            ok = (~bad).float()[..., None]; X = X - (X * ok).sum(1, keepdim=True) / ok.sum(1, keepdim=True)
        X = X - X[..., : int(round(e['baseline_s'] * self.fs))].mean(-1, keepdim=True); X = X.clamp(-e['clamp'], e['clamp'])
        P = self.posg[None].expand(len(S), -1, -1).masked_fill(bad[..., None], INVALID)
        Y = self.Rg[torch.as_tensor(self.idx(items, query), device=self.dev)][:, :, None].expand(-1, -1, self.T)
        return X, P, Y


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--subject-fold', type=int, required=True)
    ap.add_argument('--run-fold', type=int, required=True); ap.add_argument('--phaserand', action='store_true'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--shared-only', action='store_true')   # diagnosis: shared stage only (validation runs), no calibration, test runs never read
    a = ap.parse_args(); c2 = yaml.safe_load(open(a.config)); c = yaml.safe_load(open(os.path.join(ROOT, c2['b1_config']))); sp = c['split']; k, f = a.subject_fold, a.run_fold
    if not a.smoke: check_setting(c)
    st = c['settings'][c['decoder_setting']]; c['eeg']['reference'] = st['reference']; source = st['phaserand_source' if a.phaserand else 'source']
    test_runs = [r - 1 for r in sp['run_folds'][f]]; valid_runs = [r - 1 for r in sp['valid_runs'][f]]
    train_runs = [r for r in range(20) if r not in test_runs + valid_runs]
    targets = sp['subject_folds'][k] + sp.get('extra_targets', {}).get(k, [])
    trainers = sorted(s for j, fs in enumerate(sp['subject_folds']) if j != k for s in fs)
    o, cal, sh = dict(c['optim']), dict(c['calibration']), dict(c2['shared_training'])
    o['updates_per_epoch'] = sh['eval_every']
    if a.smoke:
        trainers = trainers[:3]; targets = targets[:2]; o.update(updates_per_epoch=10, batch_size=32); sh.update(eval_every=10, max_updates=20, min_updates=0)
        cal.update(updates_per_epoch=10, max_epochs=2, batch_size=32)
    eval_runs = valid_runs if a.smoke or a.shared_only else test_runs
    name = time.strftime('%Y-%m-%d') + f"_stageB_b2_{c['decoder_setting']}_k{k}_f{f}{'_phaserand' if a.phaserand else ''}{'_smoke' if a.smoke else ''}"
    name += '_sharedonly' if a.shared_only else ''
    out = os.path.join(ROOT, c2['out_root'], name); os.makedirs(out, exist_ok=True)
    yaml.safe_dump(dict(c, b2=c2, run=dict(subject_fold=k, run_fold=f, phaserand=a.phaserand, smoke=a.smoke, source=source, trainers=trainers,
                                           targets=targets, train_runs=train_runs, valid_runs=valid_runs, eval_runs=eval_runs)), open(os.path.join(out, 'config.yaml'), 'w'))
    torch.manual_seed(c2['seed']); rng = np.random.default_rng(c2['seed']); dev = 'cuda'; t0 = time.time()
    subjects = trainers + targets; nT = len(trainers)
    D = LangData(c, c2, subjects, train_runs, valid_runs, eval_runs, source, f); D.to_device(dev)
    print(f'subject fold {k} run fold {f}: trainers {trainers} targets {targets} source {source}; windows train {len(D.items["train"])} '
          f'valid {len(D.items["valid"])} eval queries {len(D.items["eval"])} ({time.time() - t0:.0f}s)', flush=True)
    m = c['model']; net = BrainModule(out_dim=D.R.shape[1], n_subjects=len(subjects), merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                                      merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'],
                                      depth=m['depth'], subject_layers=True).to(dev)
    hist = []; tr_items, va_items = D.items['train'], D.items['valid']; vr = np.random.default_rng(0)

    def fwd(items, si, shift=0.0, query=False):
        X, P, Y = D.batch(items, si, shift, query); return net(X, P, torch.as_tensor(np.asarray(si), device=dev)).float(), Y

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

    best, state = run_units(step, lambda: vloss(vb), sh['eval_every'], sh['max_updates'], sh['min_updates'], sh['patience_units'], hist, 'shared')
    net.load_state_dict(state); torch.save(state, os.path.join(out, 'shared.pt'))
    print(f'shared done: best valid {best:.4f} ({time.time() - t0:.0f}s)', flush=True)
    if a.shared_only:
        hs = [h for h in hist if h['stage'] == 'shared']; json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
        json.dump(dict(summary=dict(shared_only=True, shared_best_valid=best, shared_best_update=min(hs, key=lambda h: h['valid'])['update'],
                                    shared_updates=hs[-1]['update'], car_exclude_bad=c2['car_exclude_bad'], device=torch.cuda.get_device_name(),
                                    seconds=time.time() - t0)), open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
        return

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
        if init_vl <= b: b, row = init_vl, W[ti].detach().clone()
        with torch.no_grad(): W[ti] = row
        cal_best[s] = b
    W.requires_grad_(False); torch.save(net.subject.weights.detach().cpu()[nT:], os.path.join(out, 'target_layers.pt'))
    json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)

    # 3. evaluation: full fold score matrices (main and PRE), float16
    net.eval(); q = D.items['eval']; n = len(q); run = np.array([r for r, _ in q]); tq = np.array([t for _, t in q])
    Yc = D.Rg[torch.as_tensor(D.idx(q, True), device=dev)][:, :, None]                                   # [n, 768, 1]
    pre_s = c2['eval']['pre_shift_s']; pre_ok = tq + c['window']['tmin'] + c['eeg']['offset_s'] - pre_s >= 0
    arrays = {'run': run, 'onset': tq, 'pre_ok': pre_ok, 'item_idx': np.array(D.idx(q, True))}; res = {}
    with torch.no_grad():
        for j, s in enumerate(targets):
            for kind, shift, rows in (('main', 0.0, np.arange(n)), ('pre', -pre_s, np.where(pre_ok)[0])):
                Sfull = torch.full((n, n), float('nan'), device=dev)
                for b0 in range(0, len(rows), c2['eval']['score_chunk']):
                    ii = rows[b0:b0 + c2['eval']['score_chunk']]; est, _ = fwd([q[i] for i in ii], np.full(len(ii), nT + j), shift, True)
                    est = est.sum(-1, keepdim=True)                                                       # constant target: time sum
                    Sfull[torch.as_tensor(ii, device=dev)] = clip_scores(est, Yc)
                arrays[f'{kind}_s{s}'] = Sfull.cpu().numpy().astype(np.float16)
            res[str(s)] = dict(calibration_best_valid=cal_best[s])
    np.savez_compressed(os.path.join(out, 'scores.npz'), **arrays)
    hs = [h for h in hist if h['stage'] == 'shared']
    summ = dict(eval_runs=[r + 1 for r in eval_runs], eval_split='valid' if a.smoke else 'test', n_queries=n, shared_best_valid=best,
                shared_best_update=min(hs, key=lambda h: h['valid'])['update'], shared_updates=hs[-1]['update'], car_exclude_bad=c2['car_exclude_bad'],
                seconds=time.time() - t0)
    json.dump(dict(summary=summ, targets=res), open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    print('SUMMARY', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
