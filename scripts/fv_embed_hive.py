"""D-118 E4 (Brain-HIVE, amendment 2): raw part embeddings of every THINGS-EEG2 image (training 16,540 then test 200, the
fv_embed.py image order) for the fused target. No EEG. Images are never saved.

  rn50  OpenAI CLIP RN50 (timm/resnet50_clip.openai, open_clip RN50-quickgelu), encode_image (attention-pool projection), raw
  b32   OpenCLIP ViT-B-32 laion2b_s34b_b79k, encode_image (projected), raw (not L2-normalised)
  vae   SDXL VAE (stabilityai/stable-diffusion-xl-base-1.0 vae/): the image resized to 128 x 128 (bicubic, whole image), scaled to
        [-1, 1], latent_dist.sample(generator seeded 42 once for the whole run, as the paper's single stored draw) x scaling_factor
        -> 4 x 16 x 16 flattened (1,024), float32
CLIP inputs use each model's own open_clip preprocessing (resize, centre crop, normalisation).
k3s RTX 5080 (data group fv_hiveemb). Output: results/runs/<date>_fv_embed_hive/{rn50,b32,vae}.npy, info.json (weights sha256).

usage: python scripts/fv_embed_hive.py configs/fv_train.yaml
"""
import os, sys, json, time, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'src'))
import fv_embed
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def main():
    c = yaml.safe_load(open(P(sys.argv[1]))); hv = c['hive']; e = yaml.safe_load(open(P(hv['eegfree_config'])))
    import torch, open_clip, torchvision.transforms as T
    from PIL import Image
    from diffusers import AutoencoderKL
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    paths, conc, split = fv_embed.image_index(e); base = P(e['images']); w = hv['weights']; sh = {}
    for k, f in w['files'].items(): sh[k] = fv_embed.sha256(P(f)); assert sh[k] == w['sha256'][k], f'{k} weights sha256 {sh[k]}'
    rn, _, pre_rn = open_clip.create_model_and_transforms('RN50-quickgelu', pretrained=P(w['files']['rn50'])); rn.eval().cuda()
    b32, _, pre_b32 = open_clip.create_model_and_transforms('ViT-B-32', pretrained=P(w['files']['b32'])); b32.eval().cuda()
    vae = AutoencoderKL.from_pretrained(os.path.dirname(P(w['files']['vae']))).eval().cuda().float(); sf = vae.config.scaling_factor
    vpre = T.Compose([T.Resize((128, 128), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    gen = torch.Generator(device='cuda').manual_seed(42); out = {k: [] for k in ('rn50', 'b32', 'vae')}; t0 = time.time(); B = 64
    for b0 in range(0, len(paths), B):
        ims = [Image.open(os.path.join(base, p)).convert('RGB') for p in paths[b0:b0 + B]]
        with torch.no_grad():
            out['rn50'].append(rn.encode_image(torch.stack([pre_rn(i) for i in ims]).cuda()).float().cpu().numpy())
            out['b32'].append(b32.encode_image(torch.stack([pre_b32(i) for i in ims]).cuda()).float().cpu().numpy())
            lat = vae.encode(torch.stack([vpre(i) for i in ims]).cuda()).latent_dist.sample(generator=gen) * sf
            out['vae'].append(lat.flatten(1).float().cpu().numpy())
        if b0 // B % 40 == 0: print(f'{b0 + len(ims)}/{len(paths)} images {time.time() - t0:.0f}s', flush=True)
    res = {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
    d = os.path.join(ROOT, 'results', 'runs', time.strftime('%Y-%m-%d') + '_fv_embed_hive'); os.makedirs(d, exist_ok=True)
    for k, v in res.items(): assert np.isfinite(v).all() and len(v) == 16740, k; np.save(os.path.join(d, k + '.npy'), v)
    info = dict(n_images=len(paths), shapes={k: list(v.shape) for k, v in res.items()}, norms={k: float(np.linalg.norm(v, axis=1).mean()) for k, v in res.items()},
                weights_sha256=sh, vae_scaling_factor=sf, seconds=round(time.time() - t0, 1), device=torch.cuda.get_device_name(0),
                torch=torch.__version__, open_clip=open_clip.__version__, diffusers=__import__('diffusers').__version__,
                files_sha256={k: fv_embed.sha256(os.path.join(d, k + '.npy')) for k in res}, order='fv_embed image_index (training then test)')
    json.dump(info, open(os.path.join(d, 'info.json'), 'w'), indent=1); print(json.dumps(info))


if __name__ == '__main__':
    main()
