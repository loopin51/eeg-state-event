"""D-115 section 1 and amendment 1: features of blurred THINGS-EEG2 training images (k3s RTX 5080, data group fv_embed, no EEG).

Same pipeline as scripts/fv_embed.py stage image: the OpenCLIP ViT-H-14 preprocessing up to Normalize gives the 224 x 224 input in
[0, 1]; the blur is a per-channel scipy gaussian_filter with mode reflect on that input (fv_eegfree.yaml audit sigma = 8 px).
- AlexNet (torchvision IMAGENET1K_V1, fv_eegfree.yaml alexnet.weights) on the sigma-8 blurred input, ImageNet-normalised:
  fc7 = classifier[:6] output (second 4096 linear + ReLU, eval mode) and conv5 = features[:12] channel-wise spatial mean (256),
  for all 16,540 training images (held-out blur predictors H_b and H_c of D-115 amendment 1).
- CLIP image embeddings (L2-normalised) of the blurred development images (2,000) at blur_embed.clip_dev_sigmas_px; sigma 8 is a
  check against the existing clip_img_blur rows, sigma 4 is the report-only subset of D-115 section 1.
Development images: the 200 training concept folders first by sha256("{folder}|FV-dev") (ese.things.data.Meta.dev_concepts).
Images are never saved. Output: results/runs/<date>_fv_embed_blur/ with alex_blur8_fc7.npy [16540, 4096], alex_blur8_conv5.npy
[16540, 256], clip_img_blur<sigma>_dev.npy [2000, 1024], rows.json (training rows = fv_clip index rows, development rows), metrics.json.
usage (k3s): python scripts/fv_embed_blur.py configs/fv_pairs_build.yaml
"""
import os, sys, json, time
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'src'))
import fv_embed                                                          # sets the thread and HF_HUB_OFFLINE defaults
from ese.things.data import sha


def main():
    c = yaml.safe_load(open(os.path.join(ROOT, sys.argv[1]))); be = c['blur_embed']
    e = yaml.safe_load(open(os.path.join(ROOT, c['eegfree_config']))); tc = yaml.safe_load(open(os.path.join(ROOT, c['train_config'])))
    import torch, torchvision, open_clip
    import torchvision.transforms as T
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; assert torch.cuda.is_available()
    paths, conc, split = fv_embed.image_index(e); train_rows = [i for i, s in enumerate(split) if s == 'train']; assert len(train_rows) == 16540
    train = list(dict.fromkeys(conc[i] for i in train_rows))
    dev = set(sorted(train, key=lambda f: sha(f'{f}|{tc["dev"]["salt"]}'))[:tc['dev']['n']])
    dev_rows = [i for i in train_rows if conc[i] in dev]; assert len(dev_rows) == 2000; is_dev = set(dev_rows)
    out = os.path.join(ROOT, 'results', 'runs', time.strftime('%Y-%m-%d') + '_fv_embed_blur'); os.makedirs(out, exist_ok=True)
    model, pre, norm, clip_sha, _ = fv_embed.load_clip(e, torch, open_clip); model.to('cuda'); base = os.path.join(ROOT, e['images'])
    aw = os.path.join(ROOT, e['alexnet']['weights']); alex_sha = fv_embed.sha256(aw)
    alex = torchvision.models.alexnet(); alex.load_state_dict(torch.load(aw, map_location='cpu', weights_only=True)); alex.eval().to('cuda')
    inorm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]); s8 = e['audit']['blur_sigma_px']; A = be['alexnet']
    fc7, conv5, clip = [], [], {s: [] for s in be['clip_dev_sigmas_px']}; t0 = time.time(); B = e['clip']['batch']
    blur = lambda x, s: np.stack([gaussian_filter(ch, s, mode='reflect') for ch in x]).astype(np.float32)
    for b0 in range(0, len(train_rows), B):
        idx = train_rows[b0:b0 + B]; xs = [pre(Image.open(os.path.join(base, paths[i])).convert('RGB')).numpy() for i in idx]
        with torch.no_grad():
            xb = inorm(torch.from_numpy(np.stack([blur(x, s8) for x in xs]))).to('cuda')
            f = alex.features(xb); conv5.append(alex.features[:A['conv5_features']](xb).mean((2, 3)).float().cpu().numpy())
            fc7.append(alex.classifier[:A['fc7_classifier']](torch.flatten(alex.avgpool(f), 1)).float().cpu().numpy())
            dsel = [k for k, i in enumerate(idx) if i in is_dev]
            for s in clip:
                if dsel:
                    xd = norm(torch.from_numpy(np.stack([blur(xs[k], s) for k in dsel]))).to('cuda')
                    clip[s].append(torch.nn.functional.normalize(model.encode_image(xd).float(), dim=-1).cpu().numpy())
        if b0 // B % 40 == 0: print(f'{b0 + len(idx)}/{len(train_rows)} images {time.time() - t0:.0f}s', flush=True)
    arrs = dict(alex_blur8_fc7=np.concatenate(fc7), alex_blur8_conv5=np.concatenate(conv5), **{f'clip_img_blur{s}_dev': np.concatenate(v) for s, v in clip.items()})
    files = {}
    for k, a in arrs.items():
        a = a.astype(np.float32); assert np.isfinite(a).all(), k; np.save(os.path.join(out, k + '.npy'), a); files[k + '.npy'] = dict(sha256=fv_embed.sha256(os.path.join(out, k + '.npy')), shape=list(a.shape))
    json.dump(dict(train_rows=train_rows, dev_rows=dev_rows, dev_paths=[paths[i] for i in dev_rows]), open(os.path.join(out, 'rows.json'), 'w'))
    info = dict(n_train=len(train_rows), n_dev=len(dev_rows), blur_sigma_px=s8, clip_dev_sigmas_px=be['clip_dev_sigmas_px'], alexnet=A,
                clip_weights_sha256=clip_sha, alexnet_weights_sha256=alex_sha, seconds=round(time.time() - t0, 1),
                device=torch.cuda.get_device_name(0), torch=torch.__version__, torchvision=torchvision.__version__, open_clip=open_clip.__version__, files=files)
    json.dump(info, open(os.path.join(out, 'metrics.json'), 'w'), indent=1); print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
