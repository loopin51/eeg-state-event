"""L0 tokenizer (docs/03 §1.2): 200 ms patches with 100 ms hop per channel ->
[waveform projection 24 | 5-band log-power 5 | log-variance 1] -> LayerNorm -> Linear 32.

Band powers come from 5 fixed FIR band-pass filters (scipy firwin) applied to the
whole 10 s window, then per-patch log-variance — far better frequency resolution
than an FFT of a 20-sample patch.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import firwin

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


class PatchTokenizer(nn.Module):
    def __init__(self, sfreq=100, patch=20, hop=10, d_wave=24, d_tok=32, fir_taps=101):
        super().__init__()
        self.patch, self.hop, self.sfreq = patch, hop, sfreq
        taps = np.stack([firwin(fir_taps, [lo, hi], pass_zero=False, fs=sfreq) for lo, hi in BANDS]).astype(np.float32)
        self.register_buffer('fir', torch.from_numpy(taps)[:, None, :])          # [5,1,taps]
        self.wave = nn.Linear(patch, d_wave)
        self.norm = nn.LayerNorm(d_wave + len(BANDS) + 1)
        self.proj = nn.Linear(d_wave + len(BANDS) + 1, d_tok)

    def n_patches(self, L): return (L - self.patch) // self.hop + 1

    def forward(self, x):
        """x: [B, C, L] -> tokens [B, C, N_p, d_tok]"""
        B, C, L = x.shape
        p = x.unfold(-1, self.patch, self.hop)                                     # [B,C,N_p,patch]
        wave = self.wave(p)
        bands = F.conv1d(x.reshape(B * C, 1, L), self.fir, padding=self.fir.shape[-1] // 2)   # [B*C,5,L]
        bp = bands.unfold(-1, self.patch, self.hop).pow(2).mean(-1)                # [B*C,5,N_p]
        bp = torch.log(bp + 1e-6).reshape(B, C, len(BANDS), -1).permute(0, 1, 3, 2)   # [B,C,N_p,5]
        lv = torch.log(p.var(-1, unbiased=False) + 1e-6).unsqueeze(-1)            # [B,C,N_p,1]
        return self.proj(self.norm(torch.cat([wave, bp, lv], -1)))
