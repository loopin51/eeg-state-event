"""REVE front-end readout for the D-092 comparison (Alice EEG): REVE encoder -> per-patch attention pooling over channels ->
subject layer (E x E) -> low-rank per-patch frame head -> [F, T] target grid.

Each patch outputs the target frames of the time it covers (patch p covers samples p*step .. p*step + patch_size); frames
covered by two patches are averaged and the frames after the last patch (0.2 s for a 3-s window at 200 Hz) are produced by
the last patch (D-092). Per-layer activation checkpointing of the transformer saves memory without changing the result
(D-092 supplement 1). When no encoder parameter requires a gradient (frozen encoder, subject-layer calibration), the
encoder runs without autograd. `pooled` and `decode` split the forward pass so that subject-layer calibration can reuse
the pooled encoder output (identical in eval mode).
"""
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint
from ese.third_party.reve import Reve, ReveConfig


def checkpointed(backbone):
    def forward(x, return_out_layers=False):
        for attn, ff in backbone.layers:
            x = checkpoint(lambda y, a=attn, f=ff: (lambda z: f(z) + z)(a(y) + y), x, use_reentrant=False)
        return x
    backbone.forward = forward


class SubjectMatrix(nn.Module):
    def __init__(self, dim, n_subjects):
        super().__init__(); self.weights = nn.Parameter(torch.eye(dim)[None].repeat(n_subjects, 1, 1))

    def forward(self, z, subjects): return torch.einsum('bpe,bef->bpf', z, self.weights[subjects])


class ReveReadout(nn.Module):
    def __init__(self, enc_cfg, n_subjects, n_feat, n_frames, n_samples, eeg_fs, feat_fs, rank=32, ckpt=True):
        super().__init__()
        self.enc = Reve(ReveConfig(**enc_cfg)); E = enc_cfg['embed_dim']
        if ckpt: checkpointed(self.enc.transformer)
        size, step = enc_cfg['patch_size'], enc_cfg['patch_size'] - enc_cfg['patch_overlap']; P = (n_samples - size) // step + 1
        spans = [(round(p * step / eeg_fs * feat_fs), round((p * step + size) / eeg_fs * feat_fs)) for p in range(P)]
        spans[-1] = (spans[-1][0], n_frames); self.spans = spans; K = max(b - a for a, b in spans)
        cnt = torch.zeros(n_frames)
        for a, b in spans: cnt[a:b] += 1
        assert (cnt > 0).all(); self.register_buffer('count', cnt)
        self.score = nn.Linear(E, 1); self.subject = SubjectMatrix(E, n_subjects)
        self.out = nn.Sequential(nn.Linear(E, K * rank), nn.Unflatten(-1, (K, rank)), nn.Linear(rank, n_feat))
        self.F, self.T = n_feat, n_frames

    def load_encoder(self, path):
        from safetensors.torch import load_file
        self.enc.load_state_dict(load_file(path), strict=True)

    def pooled(self, x, pos):
        """encoder + attention pooling over channels -> [B, P, E] (fixed while only subject layers are fitted)"""
        if any(p.requires_grad for p in self.enc.parameters()): h = self.enc(x, pos)          # [B, C, P, E]
        else:
            with torch.no_grad(): h = self.enc(x, pos)
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)                                        # over channels
        return torch.einsum('bcp,bcpe->bpe', w, h)

    def decode(self, z, subjects):
        y = self.out(self.subject(z, subjects))                                                    # [B, P, K, F]
        out = torch.zeros(len(z), self.F, self.T, device=z.device, dtype=y.dtype)
        for p, (a, b) in enumerate(self.spans): out[:, :, a:b] = out[:, :, a:b] + y[:, p, : b - a].transpose(1, 2)
        return out / self.count.to(out.dtype)

    def forward(self, x, pos, subjects): return self.decode(self.pooled(x, pos), subjects)
