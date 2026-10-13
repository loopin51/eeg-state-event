"""L1 heads (docs/03 §2): fast event head, slow state head, JEPA predictor, GRL subject probes."""
import torch, torch.nn as nn, torch.nn.functional as F
from .encoder import Block


def lowpass(E, win=50):
    """Centred moving average over `win` patches (5 s at a 100 ms hop), edge-padded. [B,N,d] -> [B,N,d]."""
    x = E.transpose(1, 2)                                     # [B,d,N]
    pad = win // 2
    x = F.pad(x, (pad, win - 1 - pad), mode='replicate')
    return F.avg_pool1d(x, win, stride=1).transpose(1, 2)


class FastHead(nn.Module):
    """D-023: optionally sees only the HIGH-PASS part of E(t). Without this the window mean of z_event
    reproduces z_state (top canonical correlation 0.99), so the timescale split exists only in the loss."""
    def __init__(self, d=512, d_event=256, highpass=True, win=50):
        super().__init__()
        self.conv = nn.Conv1d(d, d, 3, padding=1); self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, d_event)
        self.highpass, self.win = highpass, win

    def forward(self, E):                       # [B,N_p,d] -> [B,N_p,d_event]
        if self.highpass: E = E - lowpass(E, self.win)
        h = F.gelu(self.conv(E.transpose(1, 2)).transpose(1, 2))
        return self.out(self.norm(h))


class SlowHead(nn.Module):
    """5 s window (50 patches), 1 s hop (10 patches), learned-query attention pooling -> 32-d bottleneck."""
    def __init__(self, d=512, d_state=32, win=50, hop=10, lowpass_input=True):
        super().__init__()
        self.win, self.hop, self.lowpass_input = win, hop, lowpass_input
        self.q = nn.Parameter(torch.randn(d) * 0.02); self.k = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, d_state), nn.LayerNorm(d_state))

    def forward(self, E):                       # [B,N_p,d] -> [B,N_s,d_state]
        if self.lowpass_input: E = lowpass(E, self.win)
        W = E.unfold(1, self.win, self.hop)     # [B,N_s,d,win]
        W = W.transpose(2, 3)                   # [B,N_s,win,d]
        att = (self.k(W) @ self.q) * (E.shape[-1] ** -0.5)   # [B,N_s,win]
        pooled = (att.softmax(-1).unsqueeze(-1) * W).sum(2)   # [B,N_s,d]
        return self.mlp(pooled)


class Predictor(nn.Module):
    """JEPA predictor: context E(t) [B,N_p,d] -> predicted z_event at every position."""
    def __init__(self, d=512, d_pred=256, d_event=256, layers=2, heads=4):
        super().__init__()
        self.inp = nn.Linear(d, d_pred)
        self.blocks = nn.ModuleList([Block(d_pred, heads, 4 * d_pred, 0.0, 0.0) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d_pred), nn.Linear(d_pred, d_event))

    def forward(self, E):
        h = self.inp(E)
        for b in self.blocks: h = b(h)
        return self.out(h)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None


class SubjectProbe(nn.Module):
    def __init__(self, d_in, n_subjects, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, n_subjects))

    def forward(self, z, lam):                  # z [B,d_in]
        return self.net(GradReverse.apply(z, lam))
