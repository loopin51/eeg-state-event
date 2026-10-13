"""Phase 3.1 (D-036): supervised fine-tuning on BCI2020, cross-subject LOSO, aligned-pretrained vs scratch front end.
Model: L0 front end (tokenizer -> shared ChannelMerger -> Transformer) -> time mean -> Linear(512, 5).
Usage: python scripts/phase3_finetune.py OUT_JSON [--init pretrained|scratch|both] [--max-steps N] [--subjects 0,1]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.model.ese_model import ESEModel
from ese.eval.readout import bootstrap_ci

CKPT = os.path.join(ROOT, 'results/runs/20260922_2135_clisa_align_only/best.pt')


class Clf(nn.Module):
    def __init__(self, front):
        super().__init__(); self.front, self.head = front, nn.Linear(512, 5)
    def forward(self, x, pos, ds):
        return self.head(self.front(x, pos, ds).mean(1))


def build(init, dev):
    ck = torch.load(CKPT, map_location='cpu'); cfg = ck['cfg']
    m = ESEModel(n_subjects=ck['n_subjects'], n_datasets=8, **cfg['model'])
    if init == 'pretrained': m.load_state_dict(ck['model'])
    return Clf(m.front).to(dev)


def run_fold(init, W, P, y, tr, va, te, dev, seed, max_steps, epochs=30, bs=64):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); net = build(init, dev)
    opt = torch.optim.AdamW([{'params': net.front.parameters(), 'lr': 1e-4}, {'params': net.head.parameters(), 'lr': 1e-3}], weight_decay=0.05)
    use_amp = dev == 'cuda'
    def batch(idx):
        x = torch.from_numpy(np.asarray(W[idx], dtype=np.float32)).to(dev)
        return x, P.expand(len(idx), -1, -1), torch.zeros(len(idx), dtype=torch.long, device=dev)
    def pred(idx):
        net.eval(); o = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            for i in range(0, len(idx), 256): o.append(net(*batch(np.sort(idx[i:i + 256]))).argmax(1).cpu().numpy())
        return np.concatenate(o)
    best, state, bad, step = -1, None, 0, 0
    for ep in range(epochs):
        net.train(); order = rng.permutation(tr)
        for i in range(0, len(order), bs):
            b = np.sort(order[i:i + bs])
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss = F.cross_entropy(net(*batch(b)), torch.from_numpy(y[b]).to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); step += 1
            if max_steps and step >= max_steps: break
        acc = float((pred(va) == y[np.sort(va)]).mean())
        if acc > best + 1e-4: best, bad, state = acc, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else: bad += 1
        if bad >= 5 or (max_steps and step >= max_steps): break
    net.load_state_dict(state); te = np.sort(te)
    return float((pred(te) == y[te]).mean()), dict(val_acc=best, epochs=ep + 1, steps=step)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out_json'); ap.add_argument('--init', default='both')
    ap.add_argument('--max-steps', type=int, default=None); ap.add_argument('--subjects', default=None); ap.add_argument('--device', default=None)
    a = ap.parse_args(); dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    c = Cache(os.path.join(ROOT, 'data/cache'), 'bci2020'); m = c.meta
    y = m['label'].to_numpy(); subj = m['subject'].to_numpy(); P = torch.from_numpy(c.positions).float().to(dev)
    subs = np.unique(subj) if a.subjects is None else np.array([int(s) for s in a.subjects.split(',')])
    inits = ['pretrained', 'scratch'] if a.init == 'both' else [a.init]; res = {}
    for init in inits:
        acc, logs, t0 = {}, [], time.time()
        for s in subs:
            rest = np.unique(subj)[np.unique(subj) != s]; vs = np.random.default_rng(int(s)).choice(rest, 2, replace=False)
            tr = np.where(np.isin(subj, rest) & ~np.isin(subj, vs))[0]; va = np.where(np.isin(subj, vs))[0]; te = np.where(subj == s)[0]
            acc[int(s)], info = run_fold(init, c.windows, P, y, tr, va, te, dev, int(s), a.max_steps); logs.append(dict(subject=int(s), **info))
            print(f"[{init}] subject {s}: acc {acc[int(s)]:.3f} (val {info['val_acc']:.3f}, epochs {info['epochs']})", flush=True)
        arr = np.array(list(acc.values())); res[init] = dict(mean=float(arr.mean()), ci95=bootstrap_ci(arr), per_subject=acc, logs=logs, seconds=round(time.time() - t0))
        print(f"[{init}] cross-subject acc {arr.mean():.3f} {np.round(bootstrap_ci(arr), 3)}", flush=True)
    if len(inits) == 2:
        ks = sorted(res['pretrained']['per_subject']); d = np.array([res['pretrained']['per_subject'][k] - res['scratch']['per_subject'][k] for k in ks])
        res['pretrained_minus_scratch'] = dict(diff=float(d.mean()), ci95=bootstrap_ci(d), wins=int((d > 0).sum()), n=len(d))
        print(f"[P-S] {d.mean():+.3f} {np.round(bootstrap_ci(d), 3)} ({(d > 0).sum()}/{len(d)})", flush=True)
    json.dump(res, open(a.out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
