"""L0 ChannelMerger (docs/03 §1.3): coordinate Fourier PE as keys, K learnable virtual
channels as queries, soft attention over the (variable, padded) channel axis per patch.
Channels with INVALID position (< 0) are masked out. No per-subject parameters (D-004).
Ported in spirit from d'Ascoli et al.; weights are new.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class FourierPE(nn.Module):
    """[.., 2] coords in [0,1]^2 -> [.., 2*n_freq*n_freq] cos/sin of (kx*x + ky*y)."""
    def __init__(self, n_freq=16):
        super().__init__()
        k = torch.arange(1, n_freq + 1).float() * 2 * math.pi
        kx, ky = torch.meshgrid(k, k, indexing='ij')
        self.register_buffer('kx', kx.reshape(-1)); self.register_buffer('ky', ky.reshape(-1))
        self.dim = 2 * n_freq * n_freq

    def forward(self, pos):
        ang = pos[..., :1] * self.kx + pos[..., 1:2] * self.ky
        return torch.cat([ang.cos(), ang.sin()], -1)


class ChannelMerger(nn.Module):
    def __init__(self, d_tok=32, n_virtual=32, d_key=64, n_freq=16, d_out=512):
        super().__init__()
        self.pe = FourierPE(n_freq)
        self.key = nn.Linear(self.pe.dim + d_tok, d_key)
        self.query = nn.Parameter(torch.randn(n_virtual, d_key) * 0.02)
        self.value = nn.Linear(d_tok, d_tok)
        self.out = nn.Linear(n_virtual * d_tok, d_out)
        self.scale = d_key ** -0.5

    def forward(self, tokens, pos):
        """tokens [B,C,N_p,d_tok], pos [B,C,2] -> merged [B,N_p,d_out]"""
        B, C, Np, D = tokens.shape
        pe = self.pe(pos).unsqueeze(2).expand(B, C, Np, -1)                       # [B,C,N_p,pe]
        k = self.key(torch.cat([pe, tokens], -1))                                  # [B,C,N_p,d_key]
        v = self.value(tokens)                                                     # [B,C,N_p,D]
        att = torch.einsum('kd,bcpd->bpkc', self.query, k) * self.scale           # [B,N_p,K,C]
        valid = (pos[..., 0] >= 0)                                                 # [B,C]
        att = att.masked_fill(~valid[:, None, None, :], float('-inf'))
        att = att.softmax(-1)
        merged = torch.einsum('bpkc,bcpd->bpkd', att, v)                           # [B,N_p,K,D]
        return self.out(merged.reshape(B, Np, -1))
