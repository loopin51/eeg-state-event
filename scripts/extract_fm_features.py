"""Frozen features from public EEG foundation models (roadmap 2.9, docs/05 §2).

Loads braindecode's pretrained CBraMod or LaBraM, encodes every cached window, mean-pools the encoder output
over channels and patches, and saves results/features/<dataset>_<model>.npy ([N, D]). The read-out is then done
by scripts/eval_crossfit.py --extra-features <model> under the same cross-fit splits as the model and DE.

Input adaptation (documented, not tuned): the caches are 100 Hz, 0.5-45 Hz, per-subject/session robust
z-scored 10 s windows. Both models were pre-trained at 200 Hz with 1 s (200-sample) patches, so windows are
up-sampled 2x with a polyphase filter (no content above 45 Hz exists). Amplitude is left in z-score units
(std ~0.9), which is the same order of magnitude as the 100 uV units these models were trained on.

Usage: python scripts/extract_fm_features.py --model cbramod|labram [--datasets a,b] [--batch 64] [--limit N]
"""
import os, sys, json, argparse, time
import numpy as np, torch
from scipy.signal import resample_poly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache

HUB = {'cbramod': 'braindecode/cbramod-pretrained', 'labram': 'braindecode/labram-pretrained'}


def build(model, n_chans, ch_names, n_times, dev):
    import braindecode.models as M
    if model == 'cbramod':
        m = M.CBraMod.from_pretrained(HUB[model], return_encoder_output=True, n_chans=n_chans, n_times=n_times, sfreq=200)
    else:
        # The checkpoint's temporal embedding covers 15 patches (+cls). Original LaBraM uses its first
        # (n_patches + 1) rows for shorter inputs; do the same by loading the pretrained model at its native
        # length and copying all weights into a model built for our window length.
        full = M.Labram.from_pretrained(HUB[model], n_outputs=0, sfreq=200, n_chans=n_chans,
                                        chs_info=[{'ch_name': c} for c in ch_names])
        m = M.Labram(n_outputs=0, n_times=n_times, sfreq=200, n_chans=n_chans, chs_info=[{'ch_name': c} for c in ch_names])
        sd = full.state_dict(); tgt = m.state_dict()
        for k, v in sd.items():
            if k in tgt and tgt[k].shape != v.shape and k.endswith('temporal_embedding'):
                sd[k] = v[:, :tgt[k].shape[1]].clone()
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
    return m.to(dev).eval()


# LaBraM's channel vocabulary has no HEOR/HEOL; FACED attached them to the mastoids (see montage.py ALIAS).
LABRAM_ALIAS = {'HEOR': 'A2', 'HEOL': 'A1'}


@torch.no_grad()
def encode(m, model, x, ch_names):
    if model == 'cbramod':
        out = m(x)
    else:   # mean of the patch tokens (LaBraM's linear-probe pooling); cls token is not used
        out = m(x, return_features=True, ch_names=[LABRAM_ALIAS.get(c, c) for c in ch_names])['features']
    if isinstance(out, (tuple, list)): out = out[0]
    # pool every non-batch, non-feature axis -> [B, D]
    while out.dim() > 2: out = out.mean(1)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True, choices=list(HUB))
    ap.add_argument('--datasets', default='seediv,deap,faced,seedvii'); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--limit', type=int, default=None, help='only the first N windows (shape check)')
    ap.add_argument('--device', default=None)
    a = ap.parse_args(); dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.join(ROOT, 'results/features'), exist_ok=True)
    for nm in a.datasets.split(','):
        c = Cache(os.path.join(ROOT, 'data/cache'), nm); names = c.header['channel_names']
        n = len(c) if a.limit is None else min(a.limit, len(c)); n_times = 2 * c.header['length']
        m = build(a.model, c.header['channels'], names, n_times, dev); t0 = time.time(); feats = []
        for i in range(0, n, a.batch):
            x = np.asarray(c.windows[i:min(n, i + a.batch)], dtype=np.float32)
            x = resample_poly(x, 2, 1, axis=-1).astype(np.float32)                # 100 Hz -> 200 Hz
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
                feats.append(encode(m, a.model, torch.from_numpy(x).to(dev), names).float().cpu().numpy())
        F = np.concatenate(feats)
        out = os.path.join(ROOT, f'results/features/{nm}_{a.model}' + ('' if a.limit is None else '_check') + '.npy')
        np.save(out, F)
        print(f'[{nm}] {a.model} features {F.shape} in {time.time() - t0:.0f}s -> {out}  (finite: {np.isfinite(F).all()}, std {F.std():.3f})', flush=True)
        del m
        if dev == 'cuda': torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
