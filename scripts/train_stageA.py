"""Stage A training (docs/04 §7). Usage: python scripts/train_stageA.py configs/stageA_sd.yaml [--out DIR]"""
import os
# Cap BLAS/OpenMP thread pools BEFORE numpy/torch import. The read-out battery runs thousands of small
# LDA / logistic fits; unconstrained OpenBLAS (MAX_THREADS=64) grabs every core and spends most of it
# spinning in thread-pool synchronisation. 8 threads is as fast and leaves the shared server usable.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import os, sys, json, time, math, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from torch.utils.data import DataLoader
from ese.data.dataset import MultiCache, collate
from ese.data.sampler import StageABatchSampler
from ese.model.ese_model import ESEModel


def split_subjects(meta, val_frac, seed, n_folds=None, fold=None):
    """Returns (train, val, test) subject dicts per dataset.
    Without folds: test = {} and val is the early-stopping hold-out (legacy behaviour).
    With folds (D-027): test = every n_folds-th subject of a fixed permutation (the OUTER fold, never seen by
    Stage A, not even for early stopping); val = val_frac of the REMAINING subjects (inner early-stopping split)."""
    rng = np.random.default_rng(seed); tr, va, te = {}, {}, {}
    for nm, g in meta.groupby('dataset'):
        subs = sorted(int(v) for v in g['subject_local'].unique()); rng.shuffle(subs)
        if n_folds:
            te[nm] = sorted(subs[fold::n_folds]); rest = [v for v in subs if v not in te[nm]]
        else:
            rest = subs
        n_val = max(1, round(len(rest) * val_frac)); va[nm] = sorted(rest[:n_val]); tr[nm] = sorted(rest[n_val:])
    return tr, va, te


class BatchIter:
    """Wraps sampler + dataset into an iterator of collated batches (with optional DataLoader workers)."""
    def __init__(self, ds, sampler, num_workers):
        self.ds, self.sampler, self.nw = ds, sampler, num_workers
    def __iter__(self):
        if self.nw == 0:
            while True:
                idx = self.sampler.sample(); items = [self.ds[i] for i, _ in idx]
                for it, (_, p) in zip(items, idx): it['pair_id'] = p
                yield collate(items)
        else:
            def gen():
                while True: yield self.sampler.sample()
            class _DS(torch.utils.data.IterableDataset):
                def __init__(s, ds, sampler): s.ds, s.sampler = ds, sampler
                def __iter__(s):
                    wi = torch.utils.data.get_worker_info(); rng = np.random.default_rng((wi.id if wi else 0) + 1000 * int(time.time()) % 100000)
                    s.sampler.rng = rng
                    while True:
                        idx = s.sampler.sample(); items = [s.ds[i] for i, _ in idx]
                        for it, (_, p) in zip(items, idx): it['pair_id'] = p
                        yield collate(items)
            for b in DataLoader(_DS(self.ds, self.sampler), batch_size=None, num_workers=self.nw, persistent_workers=True, prefetch_factor=4):
                yield b


def to(batch, dev): return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--out', default=None); ap.add_argument('--device', default=None)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    out = a.out or os.path.join(ROOT, 'results/runs', time.strftime('%Y%m%d_%H%M') + '_' + cfg['name']); os.makedirs(out, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out, 'config.yaml'), 'w'))
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    root = os.path.join(ROOT, cfg['cache_root'])
    full = MultiCache(root, cfg['datasets'], subjects=cfg.get('subjects'))
    tr_subs, va_subs, te_subs = split_subjects(full.meta, cfg['val_frac'], cfg['seed'], cfg.get('n_folds'), cfg.get('fold'))
    stageA_subs = dict(tr_subs)
    for nm, frac in (cfg.get('subsample_train') or {}).items():
        # H.1: Stage A sees only a fraction of the fold's training subjects of `nm`; split.json keeps the full
        # 'train' list so the read-out (and the DE baseline) is fitted on exactly the same subjects as before.
        k = max(2, int(round(len(tr_subs[nm]) * float(frac))))
        stageA_subs[nm] = sorted(int(v) for v in np.random.default_rng(cfg['seed'] + 7).permutation(tr_subs[nm])[:k])
    train = MultiCache(root, cfg['datasets'], subjects=stageA_subs); val = MultiCache(root, cfg['datasets'], subjects=va_subs)
    print(f"train windows {len(train)} val windows {len(val)} subjects_global {full.n_subjects_global} val_subs {va_subs}", flush=True)
    json.dump(dict(train=tr_subs, val=va_subs, test=te_subs, train_stageA=stageA_subs), open(os.path.join(out, 'split.json'), 'w'))
    b = cfg['batch']
    tr_it = iter(BatchIter(train, StageABatchSampler(train.meta, seed=cfg['seed'], **b), cfg['num_workers']))
    va_sampler = StageABatchSampler(val.meta, seed=cfg['seed'] + 1, **b)
    val_batches = []
    for _ in range(cfg['eval']['val_batches']):
        idx = va_sampler.sample(); items = [val[i] for i, _ in idx]
        for it, (_, p) in zip(items, idx): it['pair_id'] = p
        val_batches.append(collate(items))
    model = ESEModel(n_subjects=full.n_subjects_global, n_datasets=8, **cfg['model']).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f'params {n_par/1e6:.1f}M', flush=True)
    o = cfg['optim']; opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=o['lr'], weight_decay=o['weight_decay'], betas=tuple(o['betas']))
    bs = StageABatchSampler(train.meta, seed=0, **b).batch_size
    steps_per_epoch = max(1, len(train) // bs); total = cfg.get('max_steps') or steps_per_epoch * o['epochs']; warm = max(1, int(total * o['warmup_frac']))
    print(f'steps/epoch {steps_per_epoch} total {total} warmup {warm} batch {bs}', flush=True)
    use_amp = dev == 'cuda'
    # D-029: the early-stopping signal is the label-free validation loss of the ACTIVE terms only
    # (alpha-weighted coherence/align, plus JEPA when enabled). The legacy signal (coh + align regardless
    # of weights) is kept as eval.signal: coh_align for reproducing older runs.
    a1, a2 = cfg['model'].get('alphas', [1.0, 1.0, 0.1, 0.1])[:2]; use_pred = cfg['model'].get('use_predict', True)
    def val_signal(vp):
        if cfg['eval'].get('signal', 'active') == 'coh_align': return vp['coh'] + vp['align']
        return a1 * vp['coh'] + a2 * vp['align'] + (vp['pred'] if use_pred else 0.0)
    hist, best, best_step, bad, ema_val = [], float('inf'), 0, 0, None
    model.train(); t0 = time.time()
    for step in range(total):
        lr = o['lr'] * (step + 1) / warm if step < warm else 0.5 * o['lr'] * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        for g in opt.param_groups: g['lr'] = lr
        grl = 1 / (1 + math.exp(-10 * (min(1.0, step / max(1, o['grl_ramp_frac'] * total)) - 0.5)))
        m_ema = 1 - (1 - o['ema_start']) * (1 + math.cos(math.pi * step / total)) / 2
        batch = to(next(tr_it), dev)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            loss, parts = model(batch, grl_lambda=grl)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), o['grad_clip']); opt.step(); model.ema_update(m_ema)
        if step % cfg['eval']['every_steps'] == 0 or step == total - 1:
            model.eval(); vs, vca = [], []
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                for vb in val_batches:
                    _, vp = model(to(vb, dev), grl_lambda=0.0); vs.append(val_signal(vp)); vca.append(vp['coh'] + vp['align'])
            if all(x == 0 for x in vs): print('WARNING: active validation signal is 0 on every val batch (too few val subjects for the align loss?) - early stopping has no signal', flush=True)
            model.train(); v = float(np.mean(vs)); ema_val = v if ema_val is None else cfg['eval']['ema_beta'] * ema_val + (1 - cfg['eval']['ema_beta']) * v
            rec = dict(step=step, lr=lr, grl=grl, val_signal=v, val_coh_align=float(np.mean(vca)), val_ema=ema_val, sec=round(time.time() - t0), **{f'train_{k}': round(x, 4) for k, x in parts.items()})
            hist.append(rec); print(json.dumps(rec), flush=True)
            if step >= warm:
                if ema_val < best - 1e-4:
                    best, best_step, bad = ema_val, step, 0
                    torch.save(dict(model=model.state_dict(), cfg=cfg, step=step, n_subjects=full.n_subjects_global), os.path.join(out, 'best.pt'))
                else:
                    bad += 1
                    if bad >= cfg['eval']['patience']: print(f'EARLY STOP step {step} best {best:.4f} @ {best_step}', flush=True); break
            json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
        if cfg['ckpt_every_steps'] and step > 0 and step % cfg['ckpt_every_steps'] == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), cfg=cfg, step=step, n_subjects=full.n_subjects_global), os.path.join(out, 'last.pt'))
    torch.save(dict(model=model.state_dict(), cfg=cfg, step=step, n_subjects=full.n_subjects_global), os.path.join(out, 'last.pt'))
    if not os.path.exists(os.path.join(out, 'best.pt')): torch.save(dict(model=model.state_dict(), cfg=cfg, step=step, n_subjects=full.n_subjects_global), os.path.join(out, 'best.pt'))
    json.dump(dict(best_val=best, best_step=best_step, steps=step + 1, seconds=round(time.time() - t0), params_M=n_par / 1e6), open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print('STAGE_A_DONE', out, flush=True)

if __name__ == '__main__':
    main()
