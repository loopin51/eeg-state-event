"""L0 temporal encoder (docs/03 §1.4): pre-LN Transformer, RoPE, path drop; and the
FrontEnd that chains tokenizer -> merger -> (+dataset embedding) -> encoder -> E(t).
"""
import torch, torch.nn as nn, torch.nn.functional as F
from .tokenizer import PatchTokenizer
from .merger import ChannelMerger


def rope(x, base=10000.0):
    """x [B,H,T,Dh] -> rotated. Standard rotate-half RoPE over the time axis."""
    B, H, T, Dh = x.shape
    half = Dh // 2
    freqs = base ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * freqs[None, :]   # [T,half]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)


class Block(nn.Module):
    def __init__(self, d, heads, ffn, dropout=0.1, path_drop=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.h, self.drop, self.pd = heads, dropout, path_drop

    def _path(self, y):
        if not self.training or self.pd == 0: return y
        keep = (torch.rand(y.shape[0], 1, 1, device=y.device) > self.pd).to(y.dtype) / (1 - self.pd)
        return y * keep

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        p_drop = self.drop if (self.training and q.device.type != 'mps') else 0.0   # MPS SDPA has no dropout kernel
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=p_drop)
        x = x + self._path(self.o(a.transpose(1, 2).reshape(B, T, D)))
        return x + self._path(self.ff(self.n2(x)))


class FrontEnd(nn.Module):
    def __init__(self, d=512, layers=6, heads=8, ffn=2048, n_datasets=8, d_tok=32, n_virtual=32,
                 dropout=0.1, path_drop=0.1, sfreq=100):
        super().__init__()
        self.tok = PatchTokenizer(sfreq=sfreq, d_tok=d_tok)
        self.merger = ChannelMerger(d_tok=d_tok, n_virtual=n_virtual, d_out=d)
        self.ds_emb = nn.Embedding(n_datasets, d)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(d, heads, ffn, dropout, path_drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d); self.d = d

    def merge(self, x, pos, dataset_id):
        m = self.merger(self.tok(x), pos)                                          # [B,N_p,d]
        return m + self.ds_emb(dataset_id)[:, None, :]

    def encode(self, merged, mask=None):
        """merged [B,N_p,d]; mask [B,N_p] bool (True = replace by mask token) -> E(t)"""
        if mask is not None:
            merged = torch.where(mask[..., None], self.mask_token.to(merged.dtype), merged)
        for b in self.blocks: merged = b(merged)
        return self.norm(merged)

    def forward(self, x, pos, dataset_id, mask=None):
        return self.encode(self.merge(x, pos, dataset_id), mask)
