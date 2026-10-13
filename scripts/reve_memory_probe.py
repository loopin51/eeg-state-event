"""D-092 preparation: peak GPU memory and step time of a CLIP training step for (a) the REVE encoder front-end
(random initialisation, architecture from src/ese/third_party/reve, no pretrained weights) with a w2v-target head and
a subject layer, and (b) the Défossez front-end (BrainModule), at the Alice EEG input size. Random tensors stand in
for EEG and speech features; weight values do not change memory use. Each setting runs in a fresh model; an OOM is
recorded, not raised.

usage: python scripts/reve_memory_probe.py configs/reve_memory_probe.yaml
"""
import os, sys
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(v, os.environ.get('ESE_NUM_THREADS', '8'))
import json, time, resource, yaml, numpy as np, torch
from torch import nn
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.model.brain_clip import BrainModule, clip_loss
from ese.third_party.reve import Reve, ReveConfig


def checkpointed(backbone):
    """per-layer activation checkpointing for the vendored REVE TransformerBackbone, without editing its code."""
    from torch.utils.checkpoint import checkpoint
    def forward(x, return_out_layers=False):
        for attn, ff in backbone.layers:
            x = checkpoint(lambda y, a=attn, f=ff: (lambda z: f(z) + z)(a(y) + y), x, use_reentrant=False)
        return x
    backbone.forward = forward


class ReveReadout(nn.Module):
    """REVE encoder -> per-patch attention pooling over channels -> subject layer -> per-patch frames -> [F, T].
    head 'full': one linear E -> F x frames per patch; head 'light': E -> frames x rank, then rank -> F."""
    def __init__(self, enc_cfg, n_subjects, n_feat, T, head='full', rank=32, ckpt=False):
        super().__init__()
        self.enc = Reve(ReveConfig(**enc_cfg)); E = enc_cfg['embed_dim']; self.T, self.F = T, n_feat
        if ckpt: checkpointed(self.enc.transformer)
        self.score = nn.Linear(E, 1); self.subject = nn.Parameter(torch.eye(E)[None].repeat(n_subjects, 1, 1))
        self.out = None; self.E = E; self.head, self.rank = head, rank

    def forward(self, x, pos, subjects):
        h = self.enc(x, pos)                                                   # [B, C, P, E]
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)                    # over channels
        z = torch.einsum('bcp,bcpe->bpe', w, h)
        z = torch.einsum('bpe,bef->bpf', z, self.subject[subjects])
        if self.out is None:                                                   # built on first call (P known)
            self.P = z.shape[1]; K = self.T // self.P + 1
            self.out = (nn.Linear(self.E, self.F * K) if self.head == 'full' else
                        nn.Sequential(nn.Linear(self.E, K * self.rank), nn.Unflatten(-1, (K, self.rank)), nn.Linear(self.rank, self.F))).to(z.device)
        y = self.out(z)
        y = (y.view(len(z), self.P, self.F, -1).permute(0, 2, 1, 3) if self.head == 'full' else y.permute(0, 3, 1, 2)).flatten(2)
        return y[..., : self.T]


def measure(build, make_batch, steps, lr, prec, frozen_enc=False):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); net = build().cuda(); torch.manual_seed(0)
    X, P, S, Y = make_batch()
    with torch.no_grad(): net(X, P, S)                                         # materialise lazily built layers
    if frozen_enc:
        for p in net.enc.parameters(): p.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]; opt = torch.optim.Adam(params, lr=lr)
    amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=prec == 'bf16'); ts = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast(**amp): est = net(X, P, S).float()
        loss = clip_loss(est, Y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    out = dict(peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30, peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
               step_s=float(np.median(ts[2:])), n_params_m=sum(p.numel() for p in net.parameters()) / 1e6,
               n_trainable_m=sum(p.numel() for p in params) / 1e6, loss=float(loss))
    del net, opt, X, P, S, Y, est, loss; torch.cuda.empty_cache()
    return out


def main(cfg):
    c = yaml.safe_load(open(cfg)); dev = 'cuda'; I, Tg = c['input'], c['target']; C = I['n_channels']
    total = torch.cuda.get_device_properties(0).total_memory / 2**30   # cap below the gpuq system-GPU guard (job 143 was killed at 11.5 GB)
    torch.cuda.set_per_process_memory_fraction(min(1.0, c['gpu_budget_gb'] / total))
    T = int(round(I['seconds'] * Tg['sfreq'])); nR = int(round(I['seconds'] * I['reve_sfreq'])); nD = int(round(I['seconds'] * I['defossez_sfreq']))
    res = {'device': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'note': 'random initialisation, no pretrained weights', 'runs': []}
    print(res, flush=True)
    for B in c['batches']:
        def batch(n_t, pos_dim):
            return (torch.randn(B, C, n_t, device=dev), torch.rand(B, C, pos_dim, device=dev), torch.randint(0, c['n_subjects'], (B,), device=dev),
                    torch.randn(B, Tg['n_features'], T, device=dev))
        for prec in c['precisions']:
            jobs = [('defossez', 'finetune', lambda: BrainModule(out_dim=Tg['n_features'], n_subjects=c['n_subjects'], **c['defossez']), lambda: batch(nD, 2))] \
                if c.get('include_defossez', True) else []
            hv = c.get('head_variants', [{'head': 'full', 'ckpt': False}])
            jobs += [(f"reve_{e}_{h['head']}{'_ckpt' if h['ckpt'] else ''}", m,
                      (lambda e=e, h=h: ReveReadout(c['encoders'][e], c['n_subjects'], Tg['n_features'], T, h['head'], c.get('rank', 32), h['ckpt'])),
                      lambda: batch(nR, 3)) for e in c['encoders'] for m in c['modes'] for h in hv]
            for name, mode, build, mk in jobs:
                row = dict(model=name, mode=mode, batch=B, precision=prec)
                try: row.update(measure(build, mk, c['steps'], c['lr'], prec, frozen_enc=mode == 'frozen'))
                except torch.OutOfMemoryError: row['oom'] = True; torch.cuda.empty_cache()
                row['fits_budget'] = (not row.get('oom')) and row['peak_reserved_gb'] <= c['gpu_budget_gb']
                res['runs'].append(row); print(json.dumps(row), flush=True)
    res['host_maxrss_gb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    p = os.path.join(ROOT, c['out']); os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(res, open(p, 'w'), indent=1)
    print('host maxrss GB', round(res['host_maxrss_gb'], 2), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
