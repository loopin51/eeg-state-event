"""D-128 VERA-CPT stage 0 (docs/08 §14.2): label-free continued pretraining of the VERA encoder (REVE-Base + N1
band-wise patch embedding) on the public corpus of scripts/cpt_prep.py.

usage: python scripts/nc_pretrain.py configs/nc_pretrain.yaml --out <dir> [--corpus_dir <dir>] [--weights <file>]
                                     [--positions <file>] [--smoke] [--resume]

Objective (REVE, El Ouahidi et al. 2025, table 5): spatio-temporal block masking (55 %, spatial radius 3 cm, temporal
radius 3 s, channel dropout 10 % with radius 4 cm, position noise 0.25 cm); only the visible tokens go through the 22
encoder layers; a learned mask token plus the positional encoding fills the masked places and one new decoder layer
reconstructs the band-wise signal of the masked patches (6 bands x 200 samples, L1). Secondary loss: a global token
from attention pooling (REVE query token) over the outputs of every encoder layer reconstructs the masked patches
through a 2-layer FFN (L1, weight 0.1). Adversary: the position of the crop inside its recording (quartile bins) is
learned by a head on the detached global token; the encoder gets the confusion loss through a frozen copy of the head
(as the VERA adversaries, configs/nc_train.yaml). The last step is the checkpoint (no selection).

Writes <out>/: config.yaml, history.jsonl, state.pt (resumable, every ckpt_every steps), stats_<dataset>.npz,
vera_cpt.pt (enc.* and band_patch.* of the VERA model, read by scripts/nc_train.py --hp init=<file>), metrics.json.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, math, time, hashlib, argparse, datetime
import numpy as np, pandas as pd, yaml, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.model.reve_nc import ReveNC, band_split, frozen
from ese.third_party.reve.modeling_reve import FourierEmb4D, Attention, FeedForward, RMSNorm, FLASH_AVALIABLE


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(buf), b''): h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ corpus
class Corpus:
    """One dataset of scripts/cpt_prep.py: x.f32 [samples, C] time-major memmap (opened lazily, so workers each map it),
    recordings.parquet (offset, length, split), header.json (ch_names)."""
    def __init__(self, root, name, bank, L):
        self.name, self.dir, self.L = name, os.path.join(root, name), L
        self.h = json.load(open(os.path.join(self.dir, 'header.json'))); self.C = len(self.h['ch_names'])
        rec = pd.read_parquet(os.path.join(self.dir, 'recordings.parquet'))
        self.skipped_short = int((rec['length'] < L).sum()); self.rec = rec[rec['length'] >= L].reset_index(drop=True)
        self.off = self.rec['offset'].values.astype(np.int64); self.len = self.rec['length'].values.astype(np.int64)
        self.split = {s: np.where(self.rec['split'].values == s)[0] for s in ('train', 'val')}
        self.hours = {s: float(self.len[i].sum() / 200 / 3600) for s, i in self.split.items()}
        self.pos = np.array([bank[n] for n in self.h['ch_names']], np.float32)
        self.dist = np.linalg.norm(self.pos[:, None] - self.pos[None], axis=-1); self._x = None

    def __getstate__(self):                     # DataLoader workers: never pickle the memmap (it would copy the data)
        st = self.__dict__.copy(); st['_x'] = None; return st

    @property
    def x(self):
        if self._x is None:
            self._x = np.memmap(os.path.join(self.dir, 'x.f32'), dtype=np.float32, mode='r', shape=(int(self.h['n_samples']), self.C))
        return self._x

    def crop(self, r, s): return np.asarray(self.x[self.off[r] + s:self.off[r] + s + self.L]).T     # [C, L]

    def stats(self, c, dev, out):
        """Per recording: median channel SD (clip level) and band SD per channel after clipping (N1 scale), from up to
        stats_s seconds of evenly spaced crops; label-free. Cached in <out>/stats_<name>.npz."""
        p = os.path.join(out, f'stats_{self.name}.npz')
        if os.path.exists(p):
            z = np.load(p)
            if len(z['clip']) == len(self.rec): self.clip, self.bsd = z['clip'], z['bsd']; return
        I = c['input']; bands = [tuple(b) for b in I['bands']]; K0 = max(1, int(I['stats_s'] / I['crop_s']))
        clip = np.zeros(len(self.rec), np.float32); bsd = np.zeros((len(self.rec), self.C, len(bands)), np.float32)
        for r in range(len(self.rec)):
            K = int(min(K0, self.len[r] // self.L)); st = np.linspace(0, self.len[r] - self.L, K).astype(np.int64)
            X = np.stack([self.crop(r, s) for s in st]); med = float(np.median(X.std(-1))); clip[r] = med
            X = np.clip(X, -I['clip_sd'] * med, I['clip_sd'] * med)
            xb = band_split(torch.from_numpy(X).to(dev), c['sfreq'], bands)                 # [K, C, nb, L]
            bsd[r] = torch.sqrt((xb.double() ** 2).mean((0, 3))).float().cpu().numpy()
        np.savez(p, clip=clip, bsd=bsd); self.clip, self.bsd = clip, bsd


def block_mask(dist, P, m, rng, step_s):
    """[C*P] bool, True = masked, channel-major like the token order; exactly round(ratio * C * P) masked tokens.
    Channel dropout first (whole time series of the channels within dropout_radius of a random channel) up to
    dropout_ratio of the masked budget, then blocks (channels within spatial_radius x patches within temporal_radius)."""
    C = dist.shape[0]; N = C * P; Nm = int(round(m['ratio'] * N)); M = np.zeros((C, P), bool)
    rt = int(round(m['temporal_radius_s'] / step_s)); nd = int(round(m['dropout_ratio'] * Nm)); last = None
    while M.sum() < nd:
        c0 = rng.integers(C); new = np.zeros_like(M); new[dist[c0] <= m['dropout_radius_m'], :] = True
        last = new & ~M; M |= new
    while M.sum() < Nm:
        c0, t0 = rng.integers(C), rng.integers(P); new = np.zeros_like(M)
        new[np.ix_(dist[c0] <= m['spatial_radius_m'], np.arange(max(0, t0 - rt), min(P, t0 + rt + 1)))] = True
        last = new & ~M; M |= new
    extra = int(M.sum()) - Nm
    if extra > 0:                               # trim inside the last block (or anywhere if dropout alone overshot)
        pool = np.flatnonzero(last) if last is not None and last.sum() >= extra else np.flatnonzero(M)
        M.flat[rng.choice(pool, extra, replace=False)] = False
    return M.reshape(-1)


class Batches(torch.utils.data.IterableDataset):
    """Infinite training batches; every batch comes from one dataset (fixed channel set), datasets drawn in proportion to
    their training hours, recordings in proportion to their length, crop start uniform."""
    def __init__(self, corpora, c, B, seed):
        super().__init__(); self.cs, self.c, self.B, self.seed = corpora, c, B, seed
        h = np.array([k.hours['train'] for k in corpora]); self.p = h / h.sum()

    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); rng = np.random.default_rng([self.seed, wi.id if wi else 0])
        while True:
            yield make_batch(self.cs[rng.choice(len(self.cs), p=self.p)], self.c, self.B, rng, 'train')


def make_batch(k, c, B, rng, split, noise=True):
    L, m, e = k.L, c['mask'], c['encoder']; step = e['patch_size'] - e['patch_overlap']; P = (L - e['patch_size']) // step + 1
    idx = k.split[split]; w = (k.len[idx] - L + 1).astype(np.float64); r = rng.choice(idx, B, p=w / w.sum())
    s = rng.integers(0, k.len[r] - L + 1)
    x = np.stack([k.crop(ri, si) for ri, si in zip(r, s)]).astype(np.float32)
    frac = (s + L / 2) / k.len[r]; tbin = np.minimum(c['model']['adv_bins'] - 1, (frac * c['model']['adv_bins']).astype(np.int64))
    mask = np.stack([block_mask(k.dist, P, m, rng, step / c['sfreq']) for _ in range(B)])
    pos = np.repeat(k.pos[None], B, 0)
    if noise: pos = pos + rng.normal(0, c['input']['pos_noise_m'], pos.shape).astype(np.float32)
    return dict(x=torch.from_numpy(x), clip=torch.from_numpy(k.clip[r]), bsd=torch.from_numpy(k.bsd[r]),
                tbin=torch.from_numpy(tbin), mask=torch.from_numpy(mask), pos=torch.from_numpy(pos), ds=k.name)


# ------------------------------------------------------------------ model
class CPT(nn.Module):
    def __init__(self, c, weights):
        super().__init__(); e, I, mo = c['encoder'], c['input'], c['model']
        nc = dict(n1=True, n1_mode='embed', n2=False, n3=False, n4=False, bands=[tuple(b) for b in I['bands']],
                  band_gain=I['band_gain'], fm_scale=100.0, dropout=0.0, checkpoint=False)
        self.net = ReveNC(dict(nc=nc, encoder=e, sfreq=c['sfreq']), 2, 0, weights=weights)   # enc.* + band_patch.*
        self.net.band_emb.requires_grad_(False)          # unused in the 'embed' N1 mode
        E = e['embed_dim']; nb = len(I['bands']); self.out_dim = nb * e['patch_size']; self.ckpt = c['train']['activation_checkpoint']
        self.mask_token = nn.Parameter(torch.zeros(E))
        self.dec = nn.ModuleList([nn.ModuleList([Attention(E, heads=e['heads'], head_dim=e['head_dim'], use_flash=FLASH_AVALIABLE),
                                                 FeedForward(E, int(E * e['mlp_dim_ratio']), e['use_geglu'])]) for _ in range(mo['decoder_depth'])])
        self.dec_norm = RMSNorm(E); self.dec_out = nn.Linear(E, self.out_dim)
        self.sec = nn.Sequential(nn.Linear(E, E), nn.GELU(), nn.Linear(E, self.out_dim))
        h = mo['adv_hidden']; self.adv = nn.Sequential(nn.LayerNorm(E), nn.Linear(E, h), nn.GELU(), nn.Linear(h, mo['adv_bins']))
        for mod in (self.dec, self.dec_out, self.sec, self.adv):
            for n_, p in mod.named_parameters():
                if p.dim() >= 2: nn.init.normal_(p, std=mo['init_std'])
                elif n_.endswith('bias'): nn.init.zeros_(p)
        nn.init.normal_(self.mask_token, std=mo['init_std'])

    def new_modules(self): return [self.mask_token, *self.dec.parameters(), *self.dec_norm.parameters(), *self.dec_out.parameters(),
                                   *self.sec.parameters(), *self.adv.parameters()]

    def forward(self, x, pos, bsd, mask):
        """x [B, C, L] clipped microvolts, pos [B, C, 3], bsd [B, C, nb], mask [B, C*P] (True = masked, same count per row)."""
        net = self.net; enc = net.enc; B, C, L = x.shape; E = net.E
        xb = band_split(x, net.fs, net.bands) / bsd[..., None].clamp_min(1e-6) * net.cfg['band_gain']    # [B, C, nb, L]
        pt = xb.unfold(3, net.size, net.step); P = pt.shape[3]
        target = pt.permute(0, 1, 3, 2, 4).reshape(B, C * P, self.out_dim)                            # band patches, (c p) order
        p4 = FourierEmb4D.add_time_patch(pos, P); pe = enc.ln(enc.fourier4d(p4) + enc.mlp4d(p4))
        tok = net.band_patch(target) + pe
        order = torch.argsort(mask.to(torch.int8), dim=1, stable=True); Nm = int(mask[0].sum()); vis, msk = order[:, :C * P - Nm], order[:, C * P - Nm:]
        gat = lambda t, i: t.gather(1, i[..., None].expand(-1, -1, t.shape[-1]))
        h = gat(tok, vis); outs = []
        for attn, ff in enc.transformer.layers:
            if self.ckpt and h.requires_grad:
                h = checkpoint(lambda y, a=attn, f=ff: (lambda z: f(z) + z)(a(y) + y), h, use_reentrant=False)
            else:
                h = attn(h) + h; h = ff(h) + h
            outs.append(h)
        allh = torch.cat(outs, 1); q = enc.cls_query_token.expand(B, -1, -1)
        gtok = (torch.softmax(q @ allh.transpose(1, 2) / E ** 0.5, -1) @ allh).squeeze(1)               # [B, E]
        full = self.mask_token.to(h.dtype).expand(B, C * P, E).clone(); full.scatter_(1, vis[..., None].expand(-1, -1, E), h)
        full = full + pe
        for attn, ff in self.dec: full = attn(full) + full; full = ff(full) + full
        tg = gat(target, msk).float()
        l1 = (self.dec_out(self.dec_norm(gat(full, msk))).float() - tg).abs().mean()
        l2 = (self.sec(gtok[:, None] + gat(pe, msk)).float() - tg).abs().mean()
        return dict(l1=l1, l2=l2, adv=self.adv(gtok.detach()), conf=frozen(self.adv, gtok))

    def checkpoint_state(self):
        return {k: v.detach().float().cpu() for k, v in self.net.state_dict().items() if k.startswith(('enc.', 'band_patch.'))}


# ------------------------------------------------------------------ training
def lr_factor(step, T):
    S = T['steps']; wu = max(1, int(T['warmup_frac'] * S)); dc = int(T['decay_frac'] * S)
    if step < wu: return (step + 1) / wu
    if step >= S - dc: return max(0.0, (S - step) / max(1, dc))
    return 1.0


def to_dev(b, dev, c):
    x = b['x'].to(dev, non_blocking=True); clip = b['clip'].to(dev)[:, None, None] * c['input']['clip_sd']
    x = torch.maximum(torch.minimum(x, clip), -clip)
    return x, b['pos'].to(dev, non_blocking=True), b['bsd'].to(dev, non_blocking=True), b['mask'].to(dev, non_blocking=True), b['tbin'].to(dev)


def losses(model, b, dev, c, lam):
    x, pos, bsd, mask, tbin = to_dev(b, dev, c)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
        o = model(x, pos, bsd, mask)
    la = F.cross_entropy(o['adv'].float(), tbin); lc = -F.log_softmax(o['conf'].float(), -1).mean(-1).mean()
    total = o['l1'] + c['model']['secondary_weight'] * o['l2'] + la + lam * lc
    return total, dict(l1=o['l1'].item(), l2=o['l2'].item(), adv=la.item(), conf=lc.item())


@torch.no_grad()
def validate(model, val, dev, c):
    model.eval(); out = {}
    for b in val:
        _, d = losses(model, b, dev, c, 0.0)
        for k, v in d.items(): out.setdefault(b['ds'], {}).setdefault(k, []).append(v)
    model.train(); per = {ds: {k: float(np.mean(v)) for k, v in m.items()} for ds, m in out.items()}
    return dict(per_dataset=per, l1=float(np.mean([m['l1'] for m in per.values()])), l2=float(np.mean([m['l2'] for m in per.values()])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--out', required=True); ap.add_argument('--corpus_dir')
    ap.add_argument('--weights'); ap.add_argument('--positions'); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--datasets', help='comma list (smoke only)'); a = ap.parse_args()
    c = yaml.safe_load(open(os.path.join(ROOT, a.config))); T = c['train']
    if a.smoke: T.update(c['smoke'])
    rp = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
    corpus_dir = rp(a.corpus_dir or c['corpus_dir']); weights = rp(a.weights or c['weights']); positions = rp(a.positions or c['positions'])
    names = a.datasets.split(',') if a.datasets else c['datasets']
    assert a.smoke or names == c['datasets'], 'the registered run uses every dataset of the config'
    os.makedirs(a.out, exist_ok=True); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(c['seed']); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    bank = json.load(open(positions)); L = int(c['input']['crop_s'] * c['sfreq'])
    t0 = time.time(); corpora = [Corpus(corpus_dir, n, bank, L) for n in names]
    for k in corpora: k.stats(c, dev, a.out)
    stats_s = round(time.time() - t0, 1)
    corpus_info = {k.name: dict(channels=k.C, recordings=len(k.rec), skipped_short=k.skipped_short, hours=k.hours,
                                header_sha256=sha256(os.path.join(k.dir, 'header.json')), recordings_sha256=sha256(os.path.join(k.dir, 'recordings.parquet')))
                   for k in corpora}
    print(json.dumps(dict(corpus=corpus_info, stats_seconds=stats_s)), flush=True)
    yaml.safe_dump(dict(config=c, args=vars(a), corpus=corpus_info), open(os.path.join(a.out, 'config.yaml'), 'w'), sort_keys=False)

    from safetensors.torch import load_file
    model = CPT(c, None if a.weights == 'none' else load_file(weights)).to(dev)     # 'none': random init (code smoke only)
    assert a.smoke or a.weights != 'none'
    new = set(id(p) for p in model.new_modules())
    pre = [p for p in model.parameters() if p.requires_grad and id(p) not in new]
    def groups(ps, lr): return [dict(params=[p for p in ps if p.dim() >= 2], lr=lr, weight_decay=T['weight_decay']),
                                dict(params=[p for p in ps if p.dim() < 2], lr=lr, weight_decay=0.0)]
    opt = torch.optim.AdamW(groups(pre, T['lr_pretrained']) + groups(model.new_modules(), T['lr_new']), betas=tuple(T['betas']), eps=1e-8)
    for g in opt.param_groups: g['base_lr'] = g['lr']
    step, hist_path, state_path = 0, os.path.join(a.out, 'history.jsonl'), os.path.join(a.out, 'state.pt')
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location='cpu'); model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); step = st['step']
        print(json.dumps(dict(resumed_at=step)), flush=True)
    vb = min(T['batch'], 64); vrng = np.random.default_rng([c['seed'], 1])     # fixed validation crops and masks
    val = [make_batch(corpora[i % len(corpora)], c, vb, vrng, 'val', noise=False) for i in range(T['val_batches'])]
    loader = iter(torch.utils.data.DataLoader(Batches(corpora, c, T['batch'], c['seed'] + step), batch_size=None,
                                              num_workers=T['workers'], pin_memory=dev.type == 'cuda', persistent_workers=T['workers'] > 0,
                                              prefetch_factor=4 if T['workers'] > 0 else None))
    model.train(); skipped = 0; tw = time.time(); acc = {}
    while step < T['steps']:
        b = next(loader); f_ = lr_factor(step, T)
        for g in opt.param_groups: g['lr'] = g['base_lr'] * f_
        lam = c['model']['adv_lambda'] * (2 / (1 + math.exp(-10 * step / T['steps'])) - 1)
        loss, d = losses(model, b, dev, c, lam)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], T['clip']))
        if not math.isfinite(gn):
            skipped += 1; opt.zero_grad(set_to_none=True)
            if skipped > max(3, 0.01 * T['steps']): raise RuntimeError(f'non-finite gradients in {skipped} steps')
        else: opt.step()
        step += 1
        for k_, v in dict(d, gnorm=gn).items(): acc.setdefault(k_, []).append(v)
        if step % 50 == 0 or step == T['steps'] or a.smoke:
            el = time.time() - tw
            rec = dict(step=step, lr_factor=round(f_, 4), lam=round(lam, 4), skipped=skipped, crops_per_s=round(len(acc['l1']) * T['batch'] / max(el, 1e-6), 1),
                       **{k_: float(np.mean(v)) for k_, v in acc.items()},
                       max_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == 'cuda' else None)
            if step % T['val_every'] == 0 or step == T['steps']: rec['val'] = validate(model, val, dev, c)
            print(json.dumps(rec), flush=True); open(hist_path, 'a').write(json.dumps(rec) + '\n'); acc = {}; tw = time.time()
        if step % T['ckpt_every'] == 0 or step == T['steps']:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step), state_path + '.tmp'); os.replace(state_path + '.tmp', state_path)
    ck = os.path.join(a.out, 'vera_cpt.pt'); torch.save(model.checkpoint_state(), ck)
    hist = [json.loads(l) for l in open(hist_path)]
    metrics = dict(steps=step, batch=T['batch'], crops_seen=step * T['batch'], skipped_steps=skipped, corpus=corpus_info,
                   final_train={k_: hist[-1][k_] for k_ in ('l1', 'l2', 'adv', 'conf')}, final_val=hist[-1].get('val'),
                   checkpoint=dict(file='vera_cpt.pt', sha256=sha256(ck), keys=len(model.checkpoint_state())),
                   device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', torch=torch.__version__, flash_attn=FLASH_AVALIABLE,
                   date=datetime.date.today().isoformat(), smoke=a.smoke)
    json.dump(metrics, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1); print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
