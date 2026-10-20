"""F-IX (D-119 amendment 1 A): colour labels of the THINGS-EEG2 images (no EEG). Server CPU (gpuq).

Per image: RGB (PIL convert), centre box [W/4, 3W/4) x [H/4, 3H/4) at the original resolution, every pixel named by the argmax of
the van de Weijer et al. (2009) w2c table (sRGB quantised by 8: index = R//8 + 32 G//8 + 1024 B//8, the rule of their im2c.m,
reimplemented; the MATLAB code is not run). Saves the 11 colour-name pixel fractions and the first colour (ties: earlier column)
for the 16,740 rows of data/cache/fv_clip/index.json. Pixels are not stored (THINGS images are not redistributed).
Output: results/fv_s/colour_labels.npz (frac [16740, 11] float32, top1 [16740] int8) and colour_labels.json (sha256, counts).
usage: python scripts/fv_s_color.py configs/fv_s.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '1'))
import sys, json, time, hashlib, yaml, numpy as np
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
LUT = None


def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def load_table(path):
    """w2c.txt rows: R G B (bin centres 3.5, 11.5, ...) then 11 probabilities -> argmax colour per quantised index [32768] int8"""
    T = np.loadtxt(path); assert T.shape == (32768, 14), T.shape
    q = np.floor(T[:, :3] / 8).astype(int); idx = q[:, 0] + 32 * q[:, 1] + 1024 * q[:, 2]; assert len(np.unique(idx)) == 32768
    lut = np.zeros(32768, np.int8); lut[idx] = np.argmax(T[:, 3:], 1); return lut


def init(lut):
    global LUT; LUT = lut


def one(path, frac=0.5):
    from PIL import Image
    im = np.asarray(Image.open(path).convert('RGB'), np.int64); H, W = im.shape[:2]; m = (1 - frac) / 2
    c = im[int(H * m):int(H * (1 - m)), int(W * m):int(W * (1 - m))].reshape(-1, 3) // 8
    lab = LUT[c[:, 0] + 32 * c[:, 1] + 1024 * c[:, 2]]; f = np.bincount(lab, minlength=11) / len(lab)
    return f.astype(np.float32), (W, H)


def main():
    c = yaml.safe_load(open(P(sys.argv[1]))); cc = c['colour']; t0 = time.time()
    assert fsha(cc['table']) == cc['table_sha256'], 'w2c.txt sha256'; lut = load_table(cc['table'])
    ix = json.load(open(P(c['clip_index']))); paths = [os.path.join(P(c['image_root']), p) for p in ix['paths']]; assert len(paths) == 16740
    with Pool(cc['workers'], initializer=init, initargs=(lut,)) as pool:
        res = pool.starmap(one, [(p, cc['centre_fraction']) for p in paths], chunksize=64)
    F = np.stack([r[0] for r in res]); top = np.argmax(F, 1).astype(np.int8); sizes = np.array([r[1] for r in res])
    out = P(c['out_dir']); os.makedirs(out, exist_ok=True); f = os.path.join(out, 'colour_labels.npz'); np.savez_compressed(f, frac=F, top1=top)
    nm = cc['names']; split = np.array(ix['split'])
    rep = dict(file=os.path.relpath(f, ROOT), sha256=fsha(f), table_sha256=cc['table_sha256'], tar_sha256=cc['tar_sha256'], n=int(len(F)), names=nm,
               centre_fraction=cc['centre_fraction'], image_size=dict(min=sizes.min(0).tolist(), max=sizes.max(0).tolist()),
               top1_counts={s: {nm[k]: int(((top == k) & (split == s)).sum()) for k in range(11)} for s in ('train', 'test')},
               achromatic_top1={s: float(np.isin(top[split == s], [0, 3, 9]).mean()) for s in ('train', 'test')},
               mean_top1_fraction={s: float(F[split == s].max(1).mean()) for s in ('train', 'test')}, seconds=round(time.time() - t0, 1))
    json.dump(rep, open(os.path.join(out, 'colour_labels.json'), 'w'), indent=1); print(json.dumps(rep))


if __name__ == '__main__':
    main()
