"""Brain module and CLIP loss of Défossez et al. (2023), for the stage A reproduction (D-065, faculty direction 1).

Ported from facebookresearch/brainmagick (bm/models/simpleconv.py, bm/models/common.py, bm/losses.py; CC BY-NC 4.0,
non-commercial research use) with the `clip_conv` configuration of the paper: spatial attention over 2D sensor positions
(Fourier embedding, 270 outputs, 0.2 radius spatial dropout) -> 1x1 conv -> per-subject linear layer -> 10 dilated
convs (320 ch, kernel 3, dilation 1..16 repeating every 5, residual when shapes match, BatchNorm + GELU), a GLU conv
(kernel 3) after every second conv -> 1x1 conv (640) + GELU -> 1x1 transposed conv to the target dimension.
Only the single-modality path is kept (no STFT, dual-path RNN, subject embedding or per-subject merger).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

INVALID = -0.1


class FourierEmb(nn.Module):
    """2D positions in [0,1] -> cos/sin of 2*pi*(kx*x + ky*y)/width over a [-margin, 1+margin] period."""
    def __init__(self, dimension=2048, margin=0.2):
        super().__init__()
        n = (dimension // 2) ** 0.5; assert int(n ** 2 * 2) == dimension
        self.n, self.margin = int(n), margin

    def forward(self, pos):
        *O, _ = pos.shape; f = torch.arange(self.n, device=pos.device, dtype=pos.dtype); width = 1 + 2 * self.margin
        p = pos + self.margin; loc = (p[..., None, None, 0] * (2 * math.pi * f[:, None] / width)
                                      + p[..., None, None, 1] * (2 * math.pi * f / width)).reshape(*O, -1)
        return torch.cat([loc.cos(), loc.sin()], -1)


class ChannelMerger(nn.Module):
    def __init__(self, chout=270, pos_dim=2048, dropout=0.2):
        super().__init__()
        self.heads = nn.Parameter(torch.randn(chout, pos_dim) / pos_dim ** 0.5); self.emb = FourierEmb(pos_dim); self.dropout = dropout

    def forward(self, x, pos):
        """x [B,C,T], pos [B,C,2] (INVALID for unusable channels) -> [B,chout,T]"""
        B, C, T = x.shape; invalid = (pos == INVALID).all(-1); off = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        off = off.masked_fill(invalid, float('-inf'))
        if self.training and self.dropout:
            centre = torch.rand(2, device=x.device); off = off.masked_fill((pos - centre).norm(dim=-1) <= self.dropout, float('-inf'))
        scores = torch.einsum('bcd,od->boc', self.emb(pos).to(x.dtype), self.heads.to(x.dtype)) + off[:, None]
        return torch.einsum('bct,boc->bot', x, scores.softmax(-1))


class SubjectLayers(nn.Module):
    def __init__(self, cin, cout, n_subjects):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, cin, cout) / cin ** 0.5)

    def forward(self, x, subjects):
        return torch.einsum('bct,bcd->bdt', x, self.weights[subjects])


class ConvSequence(nn.Module):
    def __init__(self, channels, kernel=3, dilation_growth=2, dilation_period=5, skip=True, glu=2, glu_context=1):
        super().__init__()
        self.seq, self.glus = nn.ModuleList(), nn.ModuleList(); dilation = 1
        for k, (cin, cout) in enumerate(zip(channels[:-1], channels[1:])):
            if dilation_period and k % dilation_period == 0: dilation = 1
            self.seq.append(nn.Sequential(nn.Conv1d(cin, cout, kernel, 1, kernel // 2 * dilation, dilation=dilation),
                                          nn.BatchNorm1d(cout), nn.GELU()))
            dilation *= dilation_growth
            self.glus.append(nn.Sequential(nn.Conv1d(cout, 2 * cout, 1 + 2 * glu_context, padding=glu_context), nn.GLU(dim=1))
                             if glu and (k + 1) % glu == 0 else None)
        self.skip = skip

    def forward(self, x):
        for m, g in zip(self.seq, self.glus):
            y = m(x); x = x + y if self.skip and y.shape == x.shape else y
            if g is not None: x = g(x)
        return x


class BrainModule(nn.Module):
    def __init__(self, out_dim=1024, n_subjects=19, merger_channels=270, pos_dim=2048, merger_dropout=0.2,
                 initial_linear=270, hidden=320, depth=10, subject_layers=True, spatial_attention=True, n_in=128):
        super().__init__()
        self.merger = ChannelMerger(merger_channels, pos_dim, merger_dropout) if spatial_attention else None
        cin = merger_channels if spatial_attention else n_in
        self.initial = nn.Conv1d(cin, initial_linear, 1)
        self.subject = SubjectLayers(initial_linear, initial_linear, n_subjects) if subject_layers else None
        self.encoder = ConvSequence([initial_linear] + [hidden] * depth)
        self.final = nn.Sequential(nn.Conv1d(hidden, 2 * hidden, 1), nn.GELU(), nn.ConvTranspose1d(2 * hidden, out_dim, 1))

    def forward(self, x, pos, subjects):
        if self.merger is not None: x = self.merger(x, pos)
        x = self.initial(x)
        if self.subject is not None: x = self.subject(x, subjects)
        return self.final(self.encoder(x))


def clip_scores(est, cand):
    """est [B,F,T], cand [B',F,T] -> [B,B'] scores; candidates L2-normalised over (F,T), estimates not (as in the paper code)."""
    return torch.einsum('bct,oct,o->bo', est, cand, 1 / (1e-8 + cand.flatten(1).norm(dim=1)))


def clip_loss(est, cand):
    s = clip_scores(est, cand); return F.cross_entropy(s, torch.arange(len(s), device=s.device))
