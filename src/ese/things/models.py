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


def build(name, n_subjects, cfg):
    if name == 'bm': return BMEncoder(n_subjects, cfg['out_dim'], cfg['n_times'], cfg.get('bm'))
    if name == 'nice': return NiceEncoder(n_subjects, cfg['out_dim'], cfg['n_times'], cfg.get('nice'))
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
