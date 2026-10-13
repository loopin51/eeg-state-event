"""Affect encoder: d'Ascoli SimpleConv backbone (independent weights) + a light
self-supervised reconstruction head for Stage-1 pretraining.

Design (Doc II 2.4):
  - backbone = base SimpleConv, time_agg=None  -> keeps full H(t) [B, D, T]
  - NO sentence transformer (affect is within-window timing, not word order)
  - Stage-1: masked-time reconstruction (label-free)
  - Stage-2: freeze backbone, parameter-free mean-difference direction read-out
"""
import torch
import torch.nn as nn
from .simpleconv import SimpleConv, SimpleConvConfig


def build_affect_backbone(n_channels=62, n_subjects=15, hidden=160, depth=5,
                          out_dim=1024, merger_channels=270):
    cfg = SimpleConvConfig(
        hidden=hidden, depth=depth, kernel_size=3,
        dilation_growth=2, dilation_period=5,
        glu=2, glu_context=1, gelu=True, batch_norm=True,
        merger=True, merger_channels=merger_channels, merger_pos_dim=2048,
        merger_per_subject=True,
        initial_linear=512,
        subject_layers=True, subject_layers_dim="input",
        n_subjects=n_subjects,
        backbone_out_channels=out_dim,
        transformer_config=None,          # affect: no sentence transformer
    )
    return SimpleConv(in_channels=n_channels, out_channels=out_dim, config=cfg), cfg


class AffectPretrainer(nn.Module):
    """Backbone + a small conv decoder that reconstructs the input EEG.
    Self-supervised objective = reconstruct masked time spans (label-free)."""
    def __init__(self, n_channels=62, n_subjects=15, out_dim=1024):
        super().__init__()
        self.backbone, self.cfg = build_affect_backbone(
            n_channels=n_channels, n_subjects=n_subjects, out_dim=out_dim)
        self.n_channels = n_channels
        # lightweight decoder: H(t)[B,D,T] -> reconstructed EEG [B,C,T]
        self.decoder = nn.Sequential(
            nn.Conv1d(out_dim, 256, 1), nn.GELU(),
            nn.Conv1d(256, n_channels, 1),
        )

    def encode(self, x, subject_ids, channel_positions):
        """Returns H(t) [B, D, T]."""
        return self.backbone(x, subject_ids=subject_ids,
                             channel_positions=channel_positions)

    def forward(self, x, subject_ids, channel_positions):
        H = self.encode(x, subject_ids, channel_positions)
        recon = self.decoder(H)
        return recon, H


def make_time_mask(B, T, device, mask_ratio=0.5, span=10, generator=None):
    """Boolean mask [B,T]: True where masked. Contiguous spans of length `span`."""
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    n_spans = max(1, int((T * mask_ratio) / span))
    for b in range(B):
        for _ in range(n_spans):
            start = int(torch.randint(0, max(1, T - span), (1,), generator=generator).item())
            mask[b, start:start+span] = True
    return mask


def masked_recon_loss(model, x, subject_ids, channel_positions,
                      mask_ratio=0.5, span=10, generator=None):
    """Zero the masked input spans, reconstruct, MSE on masked positions only."""
    B, C, T = x.shape
    mask = make_time_mask(B, T, x.device, mask_ratio, span, generator)   # [B,T]
    x_in = x.clone()
    x_in[:, :, :] = torch.where(mask.unsqueeze(1), torch.zeros_like(x_in), x_in)
    recon, H = model(x_in, subject_ids, channel_positions)
    m = mask.unsqueeze(1).expand_as(x)                                    # [B,C,T]
    diff = (recon - x)[m]
    loss = (diff ** 2).mean()
    return loss, H
