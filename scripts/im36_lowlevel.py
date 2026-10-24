"""D-124 EEG-free cue-photo predictor (meter L) and the low-level incongruent concept set. No EEG is read.
Each of the 36 cue photos (grey + alpha, 400 x 400) is composited on white (as shown), resized to `size`, and described by
16 x 16 intensities, a 4 x 4 x 8 gradient-orientation histogram and radial spatial-frequency energy (8 bands); features are
z-scored over the 36 photos. Leave-concept-out logistic regression (the same 6 folds as the EEG analysis) gives, for every concept,
the probability of its true category. Incongruent set = the `incongruent_per_category` concepts of each category with the lowest
probability. Output: results/im36/lowlevel.json (no pixels are stored).
usage: python scripts/im36_lowlevel.py configs/im36.yaml
"""
import os, sys, glob, json, yaml, numpy as np
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def folds_of(concepts, cats, k):
    f = {}
    for cat in (0, 1):
        for i, n in enumerate(sorted(x for x, cc in zip(concepts, cats) if cc == cat)):
            f[n] = i % k
    return f


def features(path, L):
    im = Image.open(path).convert('LA'); g = np.asarray(im, np.float64); a = g[..., 1] / 255.0
    x = g[..., 0] * a + 255.0 * (1 - a)                                         # composite on white
    x = np.asarray(Image.fromarray(x.astype(np.uint8)).resize((L['size'], L['size']), Image.BILINEAR), np.float64) / 255.0
    px = np.asarray(Image.fromarray((x * 255).astype(np.uint8)).resize((L['pixels'], L['pixels']), Image.BILINEAR), np.float64).ravel() / 255.0
    gy, gx = np.gradient(x); mag = np.hypot(gx, gy); ang = np.mod(np.arctan2(gy, gx), np.pi)
    ob = np.minimum((ang / np.pi * 8).astype(int), 7); cs = L['size'] // L['hog_cells']; hog = []
    for i in range(L['hog_cells']):
        for j in range(L['hog_cells']):
            sl = (slice(i * cs, (i + 1) * cs), slice(j * cs, (j + 1) * cs))
            hog.append(np.bincount(ob[sl].ravel(), weights=mag[sl].ravel(), minlength=8))
    hog = np.concatenate(hog); hog = hog / (np.linalg.norm(hog) + 1e-9)
    F = np.abs(np.fft.fftshift(np.fft.fft2(x - x.mean()))) ** 2; n = x.shape[0]; yy, xx = np.indices(F.shape); r = np.hypot(yy - n / 2, xx - n / 2)
    edges = np.linspace(1, n / 2, L['sf_bands'] + 1); sf = np.array([np.log(F[(r >= edges[k]) & (r < edges[k + 1])].sum() + 1e-9) for k in range(L['sf_bands'])])
    return np.concatenate([px, hog, sf])


def main():
    c = yaml.safe_load(open(P(sys.argv[1]))); L = c['lowlevel']; root = P(c['stimuli_root'])
    files = sorted(glob.glob(os.path.join(root, 'animals', '*.png'))) + sorted(glob.glob(os.path.join(root, 'tools', '*.png')))
    names = [os.path.basename(f).split('_PNG')[0] for f in files]; cats = [0 if '/animals/' in f else 1 for f in files]
    assert len(set(names)) == 36 and sum(cats) == 18
    X = np.stack([features(f, L) for f in files]); X = (X - X.mean(0)) / (X.std(0) + 1e-9); y = np.array(cats)
    fo = folds_of(names, cats, c['folds']); from sklearn.linear_model import LogisticRegression
    p_true = np.zeros(36)
    for k in range(c['folds']):
        te = np.array([fo[n] == k for n in names]); m = LogisticRegression(C=L['lr_C'], max_iter=5000).fit(X[~te], y[~te])
        pr = m.predict_proba(X[te]); p_true[te] = pr[np.arange(te.sum()), y[te]]
    inc = []
    for cat in (0, 1):
        idx = [i for i in range(36) if cats[i] == cat]; inc += [names[i] for i in sorted(idx, key=lambda i: p_true[i])[:L['incongruent_per_category']]]
    out = dict(concepts=names, category=cats, fold={n: fo[n] for n in names}, p_true=dict(zip(names, p_true.round(4).tolist())),
               accuracy=float(np.mean(p_true > 0.5)), balanced_accuracy=float(np.mean([np.mean(p_true[y == k] > 0.5) for k in (0, 1)])),
               incongruent=inc, n_misclassified=int((p_true < 0.5).sum()), feature_dims=int(X.shape[1]))
    os.makedirs(P(c['out_dir']), exist_ok=True); json.dump(out, open(os.path.join(P(c['out_dir']), 'lowlevel.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ('fold', 'concepts', 'category')}, indent=1))


if __name__ == '__main__':
    main()
