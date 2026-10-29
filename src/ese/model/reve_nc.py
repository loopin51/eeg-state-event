"""REVE-NC (D-128, docs/08 §4.1): REVE encoder with
  N1 band-wise input     the signal is split into bands, each band z-scored per recording; n1_mode 'embed' (used): the
                         bands of a patch are concatenated and mapped by one Linear (nb x 200 -> E) initialised with REVE's
                         patch embedding in every band block, so the token count equals the broadband model; n1_mode
                         'tokens' (docs/08 first design, 6 x tokens, too slow - D-128 smoke): every band is its own token
  N2 noise stream        noise-reference channels (measured EOG/EMG + proxies) run through the same backbone in a separate
                         pass (signal tokens never attend to noise tokens); a bias expert predicts the label from the noise
                         representation only, and the part of the signal representation linearly predictable from the noise
                         representation is removed (running least squares, BatchNorm-style batch/running statistics)
  N3 null output + adversaries   K+1 outputs (null windows -> class K); adversary heads for session time, noise
                         high-frequency power (quantile bins), subject id, and the trial label read from the trial's own null
                         window. Heads learn the targets from detached features; the encoder is pushed towards uniform
                         adversary outputs through frozen copies of the heads (confusion loss; gradient reversal collapsed
                         the encoder in the development smoke, D-128 C11)
  N4 covariance pooling  per-channel attention pooling -> d x d covariance over channels -> matrix log (tangent space),
                         concatenated with the REVE query-token context; replaces the flatten-all-tokens linear head
N5 (split conformal abstention) is post hoc and lives with the evaluation code.

Every switch is in the config (`nc:` block) so the ablations (-N1 ... -N4, NC-scratch, NC-init) are config changes only.
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ese.third_party.reve import Reve, ReveConfig
from ese.third_party.reve.modeling_reve import FourierEmb4D, RMSNorm


# ------------------------------------------------------------------ signal helpers
def band_masks(n_times, fs, bands, half_width_hz=0.5, device=None):
    """[nb, F] raised-cosine pass bands on the rfft grid; neighbouring bands sharing an edge sum to 1 there."""
    f = torch.fft.rfftfreq(n_times, 1.0 / fs, device=device)
    def step(edge): return 0.5 - 0.5 * torch.cos(math.pi * ((f - (edge - half_width_hz)) / (2 * half_width_hz)).clamp(0, 1))
    return torch.stack([step(lo) * (1 - step(hi)) for lo, hi in bands])


def band_split(x, fs, bands, pad_s=0.5):
    """x [..., T] -> [..., nb, T]; reflection padding against circular wrap, zero-phase FFT masks."""
    T = x.shape[-1]; p = min(int(pad_s * fs), T - 1)
    xp = F.pad(x.reshape(-1, 1, T), (p, p), mode='reflect').reshape(*x.shape[:-1], T + 2 * p)
    M = band_masks(T + 2 * p, fs, bands, device=x.device).to(x.dtype)
    y = torch.fft.irfft(torch.fft.rfft(xp, dim=-1)[..., None, :] * M, n=T + 2 * p, dim=-1)
    return y[..., p:p + T]


def frozen(head, x):
    """head(x) with the head's parameters detached: gradients reach x only (confusion term)."""
    from torch.func import functional_call
    return functional_call(head, {k: v.detach() for k, v in head.named_parameters()}, (x,))


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)

    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None


def grl(x, lam): return GradReverse.apply(x, lam)


# ------------------------------------------------------------------ LoRA (QKVO, REVE official fine-tuning)
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__(); self.base, self.r, self.scale = base, r, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r)); self.drop = nn.Dropout(dropout)

    def forward(self, x): return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale


def add_lora(backbone, r, alpha, dropout=0.0):
    for attn, _ in backbone.layers:
        attn.to_qkv = LoRALinear(attn.to_qkv, r, alpha, dropout); attn.to_out = LoRALinear(attn.to_out, r, alpha, dropout)


def lora_parameters(module): return [p for n, p in module.named_parameters() if n.endswith('.A') or n.endswith('.B')]


def run_backbone(backbone, x, ckpt):
    for attn, ff in backbone.layers:
        if ckpt and x.requires_grad:
            x = checkpoint(lambda y, a=attn, f=ff: (lambda z: f(z) + z)(a(y) + y), x, use_reentrant=False)
        else:
            x = attn(x) + x; x = ff(x) + x
    return x


# ------------------------------------------------------------------ N2 residualisation
class NoiseResidual(nn.Module):
    """h_s - (h_n - mu_n) W, W = (S_nn + lam I)^-1 S_ns from exponential moving averages of the batch statistics.
    No gradient goes through W or into h_n here: the noise stream is shaped only by the bias head, and the signal path
    learns around a fixed linear projection. (Batch-fitted W with gradients through both streams, which share the
    backbone, collapsed training in the development smoke, D-128 C13.) Train and eval use the same running W."""
    def __init__(self, ds, dn, lam=1e-2, momentum=0.05):
        super().__init__(); self.lam, self.m = lam, momentum
        self.register_buffer('mu_n', torch.zeros(dn)); self.register_buffer('mu_s', torch.zeros(ds))
        self.register_buffer('S_nn', torch.eye(dn)); self.register_buffer('S_ns', torch.zeros(dn, ds))

    def forward(self, hs, hn):
        with torch.autocast(hs.device.type, enabled=False):         # least squares in float32 (no bf16 LU)
            hs, hn = hs.float(), hn.float().detach()
            if self.training and hs.shape[0] > 2:
                with torch.no_grad():
                    mn, ms = hn.mean(0), hs.detach().mean(0); a, b = hn - mn, hs.detach() - ms; n = hs.shape[0]
                    for buf, v in ((self.mu_n, mn), (self.mu_s, ms), (self.S_nn, a.t() @ a / n), (self.S_ns, a.t() @ b / n)): buf.lerp_(v, self.m)
            with torch.no_grad():
                I = torch.eye(self.S_nn.shape[0], device=hs.device)
                W = torch.linalg.solve(self.S_nn + self.lam * (self.S_nn.diagonal().mean() + 1.0) * I, self.S_ns)
            return hs - (hn - self.mu_n) @ W


# ------------------------------------------------------------------ N4 tangent-space pooling
def spd_log_vec(Z, eps=1e-3):
    """Z [B, C, d] -> upper triangle of logm(Z^T Z / C + eps tr/d I), off-diagonals x sqrt(2); float32, no autocast."""
    with torch.autocast(Z.device.type, enabled=False):
        Z = Z.float(); B, C, d = Z.shape; S = Z.transpose(1, 2) @ Z / C
        S = S + eps * (S.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] + 1e-6) * torch.eye(d, device=Z.device)
        w, V = torch.linalg.eigh(S); L = V @ torch.diag_embed(w.clamp_min(1e-8).log()) @ V.transpose(1, 2)
        iu = torch.triu_indices(d, d, device=Z.device); scale = torch.where(iu[0] == iu[1], 1.0, math.sqrt(2.0))
        return L[:, iu[0], iu[1]] * scale


class TangentHead(nn.Module):
    def __init__(self, E, d, out_dim):
        super().__init__(); self.q = nn.Parameter(torch.randn(E) * 0.02); self.proj = nn.Linear(E, d); self.out = out_dim
        self.norm = nn.LayerNorm(d * (d + 1) // 2)

    def forward(self, t):                       # t [B, C, S, E] (S = bands x patches of a channel)
        a = torch.softmax((t @ self.q) / t.shape[-1] ** 0.5, dim=-1)              # [B, C, S]
        z = self.proj((a[..., None] * t).sum(2))                                  # [B, C, d]
        return self.norm(spd_log_vec(z))


# ------------------------------------------------------------------ model
class ReveNC(nn.Module):
    """forward(x, pos, ref, scale) -> dict of heads; see `losses` in scripts/nc_train.py for how they are used.

    x [B, C, T] EEG microvolts, pos [B, C, 3] REVE bank positions, ref [B, R, T] noise reference (z-scored per recording),
    scale [B, C, nb] per-recording band standard deviations (N1) or None.
    """
    def __init__(self, cfg, n_classes, n_subjects=0, weights=None):
        super().__init__()
        nc = cfg['nc']; e = cfg['encoder']; self.cfg, self.K = nc, n_classes
        self.enc = Reve(ReveConfig(**e)); E = self.E = e['embed_dim']
        if weights is not None: self.enc.load_state_dict(weights, strict=True)
        self.size, self.step = e['patch_size'], e['patch_size'] - e['patch_overlap']; self.fs = cfg.get('sfreq', 200)
        self.bands = [tuple(b) for b in nc['bands']] if nc['n1'] else None
        nb = len(self.bands) if self.bands else 1; self.n1_mode = nc.get('n1_mode', 'embed')
        self.band_emb = nn.Parameter(torch.zeros(nb, E))
        if self.bands and self.n1_mode == 'embed':
            base = self.enc.to_patch_embedding[0]; self.band_patch = nn.Linear(nb * self.size, E)
            with torch.no_grad(): self.band_patch.weight.copy_(base.weight.repeat(1, nb)); self.band_patch.bias.copy_(base.bias)
        self.ckpt = nc.get('checkpoint', True)
        # N2
        if nc['n2']:
            self.noise_slot = nn.Parameter(torch.randn(nc['max_ref'], E) * 0.02)
            # LayerNorm fixes the scale of the noise representation: without it a noise stream that carries no label
            # information shrinks towards a constant, S_nn -> 0 and W explodes (development smoke divergence, D-128 C12)
            self.noise_proj = nn.Sequential(nn.Linear(E, nc['noise_dim']), nn.LayerNorm(nc['noise_dim'])); self.bias_head = nn.Linear(nc['noise_dim'], n_classes)
            self.resid = NoiseResidual(self._feat_dim(nc, E), nc['noise_dim'], lam=nc['resid_ridge'])
        # N4 / official head
        D = self._feat_dim(nc, E)
        if nc['n4']:
            self.tangent = TangentHead(E, nc['tangent_dim'], D)
            self.head = nn.Sequential(nn.LayerNorm(D), nn.Dropout(nc['dropout']), nn.Linear(D, nc['mlp_hidden']), nn.GELU(),
                                      nn.Dropout(nc['dropout']), nn.Linear(nc['mlp_hidden'], n_classes + int(nc['n3'])))
        else:
            self.head = None                    # lazily built: depends on the number of tokens (official flatten head)
        # N3 adversaries (on the residual representation)
        if nc['n3']:                            # on the full feature (N4) or on the residual query context (-N4)
            h, A, nbin = nc['adv_hidden'], (D if nc['n4'] else E), nc['adv_bins']
            def mlp(o): return nn.Sequential(nn.LayerNorm(A), nn.Linear(A, h), nn.GELU(), nn.Linear(h, o))
            self.adv_time, self.adv_noise, self.adv_null = mlp(nbin), mlp(nbin), mlp(n_classes)
            self.adv_subj = mlp(n_subjects) if n_subjects else None

    @staticmethod
    def _feat_dim(nc, E): return E + (nc['tangent_dim'] * (nc['tangent_dim'] + 1) // 2 if nc['n4'] else 0)

    # -------- tokens
    def _embed(self, x, pos, band_ids=None, slot=None):
        """x [B, C, T] -> tokens [B, C*P, E] with REVE patch embedding and 4D positional embedding."""
        enc = self.enc; patches = x.unfold(2, self.size, self.step); B, C, P, _ = patches.shape
        p4 = FourierEmb4D.add_time_patch(pos, P); pe = enc.ln(enc.fourier4d(p4) + enc.mlp4d(p4))
        t = enc.to_patch_embedding(patches).reshape(B, C * P, self.E) + pe
        if band_ids is not None: t = t + self.band_emb[band_ids].repeat_interleave(P, 0)[None]
        if slot is not None: t = t + slot.repeat_interleave(P, 0)[None]
        return t, P

    def encode_signal(self, x, pos, scale):
        B, C, T = x.shape
        if self.bands and self.n1_mode == 'embed':
            nb = len(self.bands); xb = band_split(x, self.fs, self.bands) / scale[..., None].clamp_min(1e-6) * self.cfg['band_gain']
            patches = xb.unfold(3, self.size, self.step)                                 # [B, C, nb, P, size]
            P = patches.shape[3]; patches = patches.permute(0, 1, 3, 2, 4).reshape(B, C, P, nb * self.size)
            enc = self.enc; p4 = FourierEmb4D.add_time_patch(pos, P); pe = enc.ln(enc.fourier4d(p4) + enc.mlp4d(p4))
            t = self.band_patch(patches).reshape(B, C * P, self.E) + pe
            t = run_backbone(self.enc.transformer, t, self.ckpt)
            return t.reshape(B, C, P, self.E)
        if self.bands:
            xb = band_split(x, self.fs, self.bands)                                  # [B, C, nb, T]
            xb = xb / scale[..., None].clamp_min(1e-6) * self.cfg['band_gain']
            nb = len(self.bands); xb = xb.reshape(B, C * nb, T)
            posb = pos.repeat_interleave(nb, 1); ids = torch.arange(nb, device=x.device).repeat(C)
            t, P = self._embed(xb, posb, band_ids=ids)
        else:
            nb = 1; t, P = self._embed(x / self.cfg['fm_scale'], pos)
        t = run_backbone(self.enc.transformer, t, self.ckpt)
        return t.reshape(B, C, nb * P, self.E)

    def encode_noise(self, ref):
        B, R, T = ref.shape; pos = torch.zeros(B, R, 3, device=ref.device, dtype=ref.dtype)
        t, P = self._embed(ref * self.cfg['band_gain'], pos, slot=self.noise_slot[:R])
        t = run_backbone(self.enc.transformer, t, self.ckpt)
        return self.noise_proj(t.mean(1))

    def features(self, x, pos, ref=None, scale=None):
        t = self.encode_signal(x, pos, scale); B = t.shape[0]; flat = t.reshape(B, -1, self.E)
        q = self.enc.cls_query_token.expand(B, -1, -1)
        ctx = (torch.softmax(q @ flat.transpose(1, 2) / self.E ** 0.5, -1) @ flat).squeeze(1)
        if self.cfg['n4']: h = torch.cat([ctx, self.tangent(t)], -1)
        else:
            if self.head is None:
                D = (flat.shape[1] + 1) * self.E
                self.head = nn.Sequential(RMSNorm(D), nn.Dropout(self.cfg['dropout']), nn.Linear(D, self.K + int(self.cfg['n3']))).to(x.device)
            h = torch.cat([ctx[:, None], flat], 1).flatten(1)
        out, a = {}, (h if self.cfg['n4'] else ctx)
        if self.cfg['n2'] and ref is not None:
            hn = self.encode_noise(ref); out['bias_logits'] = self.bias_head(hn)
            a = self.resid(a, hn)
            h = a if self.cfg['n4'] else torch.cat([a[:, None], flat], 1).flatten(1)
        out['h'], out['h_adv'] = h, a
        return out

    def forward(self, x, pos, ref=None, scale=None, adv_lambda=0.0):
        out = self.features(x, pos, ref, scale); h = out['h']; out['logits'] = self.head(h)
        if self.cfg['n3']:
            a, g = out['h_adv'], out['h_adv'].detach()
            for k in ('time', 'noise', 'null', 'subj'):
                hd = getattr(self, f'adv_{k}')
                if hd is not None: out[f'adv_{k}'], out[f'conf_{k}'] = hd(g), frozen(hd, a)
        return out
