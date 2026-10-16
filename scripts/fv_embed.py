"""F-V embeddings for the EEG-free gates and the decoder targets (D-111, D-112; k3s RTX 5080 per D-095).

stage image: OpenCLIP ViT-H-14 (laion2b_s32b_b79k) image embeddings of every THINGS-EEG2 image (training 16,540 then test 200),
             of its phase-randomized copy and of its strongly blurred copy, plus AlexNet conv1 / conv2 channel means.
stage text:  name text embeddings of all 1,854 concepts for prompt p1 and p2 from the name table (configs/fv_names.tsv).
stage smoke: imports, GPU, weight sha256 and one batch of 8 test images, or 8 synthetic images if the images are not
             staged yet (no arrays saved).

Images, thumbnails and the modified images are never written anywhere; only embedding arrays and an index are saved in
results/runs/<date>_fv_embed_<stage>/ (RUN_DIR on k3s). All embeddings are float32 and L2-normalized (AlexNet: raw means).

usage: python scripts/fv_embed.py configs/fv_eegfree.yaml --stage {smoke,image,text}
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '6'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import argparse, glob, hashlib, json, sys, time
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''): h.update(b)
    return h.hexdigest()


def image_index(c):
    """(relative paths, concept folder per image, split per image), training images first, each split in sorted order."""
    base = os.path.join(ROOT, c['images']); paths, conc, split = [], [], []
    for sp, key, (nc, ni) in (('train', 'training_images', c['n_train']), ('test', 'test_images', c['n_test'])):
        folders = sorted(d for d in os.listdir(os.path.join(base, key)) if os.path.isdir(os.path.join(base, key, d)))
        assert len(folders) == nc, f'{key}: {len(folders)} concept folders, expected {nc}'
        n0 = len(paths)
        for d in folders:
            for f in sorted(os.listdir(os.path.join(base, key, d))):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')): paths.append(f'{key}/{d}/{f}'); conc.append(d); split.append(sp)
        assert len(paths) - n0 == ni, f'{key}: {len(paths) - n0} images, expected {ni}'
    return paths, conc, split


def phase_randomize(x, seed, idx):
    """x: [3, H, W] in [0, 1]. Keep each channel's amplitude spectrum, use the phase of one real white-noise image for all
    channels (Hermitian symmetry kept), return the real part clipped to [0, 1]."""
    rng = np.random.default_rng([seed, idx])
    ph = np.angle(np.fft.fft2(rng.standard_normal(x.shape[1:])))
    amp = np.abs(np.fft.fft2(x, axes=(-2, -1)))
    return np.clip(np.real(np.fft.ifft2(amp * np.exp(1j * ph), axes=(-2, -1))), 0, 1).astype(np.float32)


def load_clip(c, torch, open_clip):
    w = os.path.join(ROOT, c['clip']['weights']); h = sha256(w)
    assert h == c['clip']['weights_sha256'], f'CLIP weights sha256 {h}'
    model, _, pre = open_clip.create_model_and_transforms(c['clip']['arch'], pretrained=w)
    tf = list(pre.transforms); norm = tf[-1]
    assert type(norm).__name__ == 'Normalize', tf
    import torchvision.transforms as T
    return model.eval(), T.Compose(tf[:-1]), norm, h, repr(pre)


def run_image(c, out, torch, open_clip, smoke=False):
    from PIL import Image
    import torchvision, torchvision.transforms as T
    from scipy.ndimage import gaussian_filter
    dev = 'cuda'; base = os.path.join(ROOT, c['images'])
    if smoke and not os.path.isdir(os.path.join(base, 'test_images')):   # environment check before the images arrive
        paths = [None] * 8; sel = list(range(8)); rng = np.random.default_rng(0)
        synth = [Image.fromarray(rng.integers(0, 256, (500, 500, 3), dtype=np.uint8)) for _ in sel]
    else:
        paths, conc, split = image_index(c); synth = None
        sel = [i for i, s in enumerate(split) if s == 'test'][:8] if smoke else list(range(len(paths)))
    model, pre, norm, clip_sha, pre_repr = load_clip(c, torch, open_clip); model.to(dev)
    aw = os.path.join(ROOT, c['alexnet']['weights']); alex_sha = sha256(aw)
    alex = torchvision.models.alexnet(); alex.load_state_dict(torch.load(aw, map_location='cpu', weights_only=True)); alex.eval().to(dev)
    alex_pre = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    L = c['alexnet']['layers']; a = c['audit']; B = c['clip']['batch']
    out_arr = {k: [] for k in ('clip_img', 'clip_img_phase', 'clip_img_blur', 'alex_conv1', 'alex_conv2')}
    t0 = time.time()
    for b0 in range(0, len(sel), B):
        idx = sel[b0:b0 + B]; xs, xp, xb, xa = [], [], [], []
        for i in idx:
            im = synth[i] if synth else Image.open(os.path.join(base, paths[i])).convert('RGB')
            x = pre(im).numpy()                                                   # [3, 224, 224] in [0, 1]
            xs.append(x); xp.append(phase_randomize(x, a['phase_seed'], i))
            xb.append(np.stack([gaussian_filter(ch, a['blur_sigma_px'], mode='reflect') for ch in x]).astype(np.float32))
            xa.append(alex_pre(im))
        with torch.no_grad():
            for key, arr in (('clip_img', xs), ('clip_img_phase', xp), ('clip_img_blur', xb)):
                e = model.encode_image(norm(torch.from_numpy(np.stack(arr))).to(dev)).float()
                out_arr[key].append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
            xa = torch.stack(xa).to(dev)
            for key, n in (('alex_conv1', L['conv1']), ('alex_conv2', L['conv2'])):
                out_arr[key].append(alex.features[:n](xa).mean((2, 3)).float().cpu().numpy())
        if b0 // B % 20 == 0: print(f'{b0 + len(idx)}/{len(sel)} images {time.time() - t0:.0f}s', flush=True)
    res = {k: np.concatenate(v).astype(np.float32) for k, v in out_arr.items()}
    for k, v in res.items():
        assert np.isfinite(v).all(), k
        if not smoke: np.save(os.path.join(out, k + '.npy'), v)
    info = dict(stage='smoke' if smoke else 'image', n_images=len(sel), synthetic=bool(synth), seconds=round(time.time() - t0, 1),
                clip_weights_sha256=clip_sha, alexnet_weights_sha256=alex_sha, clip_preprocess=pre_repr,
                shapes={k: list(v.shape) for k, v in res.items()})
    if smoke:   # same-image checks only: embeddings are unit norm and the three variants differ
        info['smoke'] = dict(norms=[float(np.linalg.norm(res['clip_img'][0]))],
                             cos_orig_phase=float((res['clip_img'][0] * res['clip_img_phase'][0]).sum()),
                             cos_orig_blur=float((res['clip_img'][0] * res['clip_img_blur'][0]).sum()))
    else:
        json.dump(dict(paths=paths, concept=conc, split=split, sha256=[sha256(os.path.join(base, p)) for p in paths]),
                  open(os.path.join(out, 'index.json'), 'w'))
    return info


def run_text(c, out, torch, open_clip):
    import csv
    from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES
    rows = list(csv.DictReader(open(os.path.join(ROOT, c['names']), encoding='utf-8'), delimiter='\t'))
    names_sha = sha256(os.path.join(ROOT, c['names']))
    model, _, _, clip_sha, _ = load_clip(c, torch, open_clip); model.to('cuda')
    tok = open_clip.get_tokenizer(c['clip']['arch']); col = 'prompt_name'
    assert col in rows[0], f'name table needs a {col} column, has {list(rows[0])}'
    names = [r[col] for r in rows]; t0 = time.time()

    def enc(texts):
        e = []
        with torch.no_grad():
            for b0 in range(0, len(texts), 256):
                e.append(torch.nn.functional.normalize(model.encode_text(tok(texts[b0:b0 + 256]).to('cuda')).float(), dim=-1).cpu().numpy())
        return np.concatenate(e)
    p1 = enc([c['prompts']['p1'].format(n) for n in names])
    T = len(OPENAI_IMAGENET_TEMPLATES)
    e2 = enc([t(n) for n in names for t in OPENAI_IMAGENET_TEMPLATES]).reshape(len(names), T, -1).mean(1)
    p2 = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
    np.save(os.path.join(out, 'clip_txt_p1.npy'), p1.astype(np.float32)); np.save(os.path.join(out, 'clip_txt_p2.npy'), p2.astype(np.float32))
    json.dump(dict(rows=rows), open(os.path.join(out, 'names.json'), 'w'))
    return dict(stage='text', n_names=len(names), n_templates_p2=T, names_sha256=names_sha, clip_weights_sha256=clip_sha,
                seconds=round(time.time() - t0, 1), shapes={'clip_txt_p1': list(p1.shape), 'clip_txt_p2': list(p2.shape)})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', required=True, choices=['smoke', 'image', 'text'])
    a = ap.parse_args(); c = yaml.safe_load(open(os.path.join(ROOT, a.config)))
    import torch, open_clip, torchvision
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    out = os.path.join(ROOT, 'results', 'runs', time.strftime('%Y-%m-%d') + f'_fv_embed_{a.stage}'); os.makedirs(out, exist_ok=True)
    info = run_text(c, out, torch, open_clip) if a.stage == 'text' else run_image(c, out, torch, open_clip, smoke=a.stage == 'smoke')
    info.update(device=torch.cuda.get_device_name(0), torch=torch.__version__, open_clip=open_clip.__version__,
                torchvision=torchvision.__version__, numpy=np.__version__, max_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                config=a.config)
    if a.stage != 'smoke':
        info['files'] = {f: sha256(os.path.join(out, f)) for f in sorted(os.listdir(out)) if f.endswith(('.npy', '.json')) and f != 'metrics.json'}
    json.dump(info, open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    yaml.safe_dump(c, open(os.path.join(out, 'config.yaml'), 'w'), sort_keys=False)
    print(json.dumps({k: v for k, v in info.items() if k not in ('files', 'clip_preprocess')}), flush=True)


if __name__ == '__main__':
    main()
