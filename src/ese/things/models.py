"""F-V (D-111, D-112 section 4) encoders: EEG [B, 63, 250] -> image-embedding space [B, 1024], both with a per-subject layer.

  bm   : the repository's Défossez BrainModule (src/ese/model/brain_clip.py; spatial attention over 2D positions, per-subject
         270 x 270 layer, dilated conv stack) -> adaptive average pooling over time into `n_bins` bins -> linear projection.
  nice : NICE/ATM-type temporal-spatial convolution (Song et al. 2024, TSConv: conv (1, 25) -> avg pool (1, 51) stride 5 ->
         BN -> ELU -> conv (63, 1) -> BN -> ELU -> dropout -> 1x1 conv -> flatten; projection with a residual GELU block and
         LayerNorm as NICE's Proj_eeg), with a per-subject C x C channel layer in front (identity initialised).
Both expose `.subject` (a SubjectLayers-like module with `.weights` [n_subjects, cin, cout]) so the D-090 calibration code
refits the target's layer only. The loss is InfoNCE from EEG to the unique image targets of the batch with a learned logit
scale (CLIP initialisation 1 / 0.07), estimates and targets L2-normalised.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from ese.model.brain_clip import BrainModule, SubjectLayers


class BMEncoder(nn.Module):
    def __init__(self, n_subjects, out_dim=1024, n_times=250, m=None):
        super().__init__(); m = m or {}
        self.body = BrainModule(out_dim=m.get('body_out', 256), n_subjects=n_subjects, merger_channels=m.get('merger_channels', 270),
                                pos_dim=m.get('pos_dim', 2048), merger_dropout=m.get('merger_dropout', 0.2), initial_linear=m.get('initial_linear', 270),
                                hidden=m.get('hidden', 320), depth=m.get('depth', 10), subject_layers=True, n_in=m.get('n_in', 63))
        self.pool = nn.AdaptiveAvgPool1d(m.get('n_bins', 10)); self.drop = nn.Dropout(m.get('proj_dropout', 0.5))
        self.proj = nn.Linear(m.get('body_out', 256) * m.get('n_bins', 10), out_dim)
        self.subject = self.body.subject

    def forward(self, x, pos, subjects):
        return self.proj(self.drop(self.pool(self.body(x, pos, subjects)).flatten(1)))


class ChannelSubjectLayers(SubjectLayers):
    def __init__(self, c, n_subjects):
        super().__init__(c, c, n_subjects)
        with torch.no_grad(): self.weights.copy_(torch.eye(c).expand(n_subjects, c, c))


class NiceEncoder(nn.Module):
    def __init__(self, n_subjects, out_dim=1024, n_times=250, m=None, n_ch=63):
        super().__init__(); m = m or {}; k = m.get('k', 40)
        self.subject = ChannelSubjectLayers(n_ch, n_subjects)
        self.ts = nn.Sequential(nn.Conv2d(1, k, (1, m.get('m1', 25))), nn.AvgPool2d((1, m.get('m2', 51)), (1, m.get('s', 5))),
                                nn.BatchNorm2d(k), nn.ELU(), nn.Conv2d(k, k, (n_ch, 1)), nn.BatchNorm2d(k), nn.ELU(), nn.Dropout(m.get('dropout', 0.5)),
                                nn.Conv2d(k, k, 1))
        n_out = k * ((n_times - m.get('m1', 25) + 1 - m.get('m2', 51)) // m.get('s', 5) + 1)
        self.lin = nn.Linear(n_out, out_dim); self.res = nn.Sequential(nn.GELU(), nn.Linear(out_dim, out_dim), nn.Dropout(m.get('proj_dropout', 0.5)))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, pos, subjects):
        x = self.subject(x, subjects); h = self.lin(self.ts(x[:, None]).flatten(1)); return self.norm(h + self.res(h))


# ---- D-118 challenger arms (re-implemented from the papers' descriptions and public sources read only; no external code run)
class SubjectToken(nn.Module):
    """ATM subject token as a calibratable per-subject parameter: weights [n_subjects, 1, d] (row = the token)"""
    def __init__(self, d, n_subjects):
        super().__init__(); self.weights = nn.Parameter(torch.randn(n_subjects, 1, d) * 0.02)

    def forward(self, subjects): return self.weights[subjects]


class ATMEncoder(nn.Module):
    """E2 ATM (Li et al. 2024, ATM-S): channels as tokens (Linear(T, T) per channel time course, no positional encoding = the
    paper's headline setting), the subject token prepended, dropout 0.25 -> one post-norm transformer layer (4 heads of 62 dims,
    q/k/v Linear(250, 248), FFN conv1x1 250 -> 256 -> 250 GELU, dropout 0.25) -> final LayerNorm -> the 63 channel tokens
    (subject token dropped, as the authors' later fix) -> NICE TSConv (as NiceEncoder) -> Linear(1440, D) + residual GELU block
    (dropout 0.5) + LayerNorm. The subject token is the arm's subject layer (D-090 calibration refits the target's row)."""
    def __init__(self, n_subjects, out_dim=1024, n_times=250, m=None, n_ch=63):
        super().__init__(); m = m or {}; d = n_times; h = m.get('heads', 4); dh = d // h; k = m.get('k', 40)
        self.embed = nn.Linear(d, d); self.subject = SubjectToken(d, n_subjects); self.drop = nn.Dropout(m.get('dropout_attn', 0.25))
        self.q, self.k_, self.v = nn.Linear(d, h * dh), nn.Linear(d, h * dh), nn.Linear(d, h * dh); self.o = nn.Linear(h * dh, d); self.h, self.dh = h, dh
        self.ln1, self.ln2, self.lnf = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Conv1d(d, m.get('ffn', 256), 1), nn.GELU(), nn.Dropout(m.get('dropout_attn', 0.25)), nn.Conv1d(m.get('ffn', 256), d, 1))
        self.ts = nn.Sequential(nn.Conv2d(1, k, (1, m.get('m1', 25))), nn.AvgPool2d((1, m.get('m2', 51)), (1, m.get('s', 5))),
                                nn.BatchNorm2d(k), nn.ELU(), nn.Conv2d(k, k, (n_ch, 1)), nn.BatchNorm2d(k), nn.ELU(), nn.Dropout(m.get('dropout', 0.5)),
                                nn.Conv2d(k, k, 1))
        n_out = k * ((n_times - m.get('m1', 25) + 1 - m.get('m2', 51)) // m.get('s', 5) + 1)
        self.lin = nn.Linear(n_out, out_dim); self.res = nn.Sequential(nn.GELU(), nn.Linear(out_dim, out_dim), nn.Dropout(m.get('proj_dropout', 0.5)))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, pos, subjects):
        B, C, T = x.shape; t = torch.cat([self.subject(subjects), self.embed(x)], 1); t = self.drop(t)
        q = self.q(t).view(B, -1, self.h, self.dh).transpose(1, 2); k_ = self.k_(t).view(B, -1, self.h, self.dh).transpose(1, 2); v = self.v(t).view(B, -1, self.h, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k_, v, dropout_p=self.drop.p if self.training else 0.0).transpose(1, 2).reshape(B, -1, self.h * self.dh)
        t = self.ln1(t + self.drop(self.o(a))); t = self.ln2(t + self.drop(self.ff(t.transpose(1, 2)).transpose(1, 2))); t = self.lnf(t)[:, 1:]
        hh = self.lin(self.ts(t[:, None]).flatten(1)); return self.norm(hh + self.res(hh))


class MLPHead(nn.Module):
    """Brain-HIVE residual block: z = LayerNorm(z0 + W2 Dropout(GELU(W1 z0))), W1, W2 without bias, eps 1e-6"""
    def __init__(self, d, p):
        super().__init__(); self.w1 = nn.Linear(d, d, bias=False); self.w2 = nn.Linear(d, d, bias=False); self.drop = nn.Dropout(p); self.ln = nn.LayerNorm(d, eps=1e-6)

    def forward(self, z): return self.ln(z + self.w2(self.drop(F.gelu(self.w1(z)))))


class BrainMLP(nn.Module):
    """E3/E4 Brain MLP (Brain-HIVE MBP): a per-subject C x C channel layer (identity initialised; added so that every arm has a
    calibratable subject layer, D-118) on the selected channels -> flatten (channel-major) -> Linear(C*T, D) -> MLPHead"""
    def __init__(self, n_subjects, out_dim=1024, n_times=250, m=None, channels=None):
        super().__init__(); m = m or {}; self.register_buffer('ch', torch.as_tensor(channels, dtype=torch.long)); C = len(channels)
        self.subject = ChannelSubjectLayers(C, n_subjects); self.lin = nn.Linear(C * n_times, out_dim); self.head = MLPHead(out_dim, m.get('dropout', 0.1))

    def forward(self, x, pos, subjects):
        x = self.subject(x[:, self.ch], subjects); return self.head(self.lin(x.flatten(1)))


class FusionEncoder(nn.Module):
    """E4 Brain-HIVE fused target: z_bar = sum_k (W_k z_k + b_k) over the raw (unnormalised) part embeddings -> MLPHead; trained
    with the brain encoder (retrieval setting of the paper)"""
    def __init__(self, part_dims, out_dim=1024, p=0.1):
        super().__init__(); self.parts = nn.ModuleList(nn.Linear(d, out_dim) for d in part_dims); self.head = MLPHead(out_dim, p)

    def forward(self, zs): return self.head(sum(l(z) for l, z in zip(self.parts, zs)))


def build(name, n_subjects, cfg, channels=None):
    if name == 'bm': return BMEncoder(n_subjects, cfg['out_dim'], cfg['n_times'], cfg.get('bm'))
    if name == 'nice': return NiceEncoder(n_subjects, cfg['out_dim'], cfg['n_times'], cfg.get('nice'))
    if name == 'atm': return ATMEncoder(n_subjects, cfg['out_dim'], cfg['n_times'], cfg.get('atm'))
    if name in ('mlp', 'hive'): return BrainMLP(n_subjects, cfg['out_dim'], cfg['n_times'], cfg.get(name), channels if channels is not None else cfg[name]['channel_idx'])
    raise ValueError(name)


class LogitScale(nn.Module):
    def __init__(self, init=1 / 0.07, max_scale=100.0):
        super().__init__(); self.v = nn.Parameter(torch.tensor(math.log(init))); self.max = math.log(max_scale)

    def forward(self): return self.v.clamp(max=self.max).exp()


def info_nce(est, tgt, tgt_idx, scale):
    """est [B, D]; tgt [U, D] unique targets of the batch; tgt_idx [B] row of each estimate's target in tgt"""
    s = F.normalize(est, dim=-1) @ F.normalize(tgt, dim=-1).T * scale; return F.cross_entropy(s, tgt_idx)


def scores(est, cand):
    """cosine scores [B, K] between estimates and candidate embeddings"""
    return F.normalize(est.float(), dim=-1) @ F.normalize(cand.float(), dim=-1).T
