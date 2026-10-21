"""F-V EEG-free gates G0 and 0-V (D-111, D-112; definitions: reports/08_fv_stage/notes/fv_gate_g0_0v_draft_kr.md). No EEG.

stage lowlevel: low-level image features of every THINGS-EEG2 image (M1 CIELAB thumbnail, M2 CIELAB histogram, M3 L* mean
                and sd, M4 Gabor energy, A4 edge density) -> <embed_out>/lowlevel.npz, and the matched sets of the 200 test
                concepts (matching family M1-M4 + AlexNet conv1 from fv_embed) -> results/fv/matched_sets.json.
stage gate:     G0 (name table and text embeddings needed) and the 0-V audit measurers -> results/fv/gate_g0_0v.json.

Reads the fv_embed arrays from <embed_out> (index.json, clip_img*.npy, alex_conv*.npy, clip_txt_p*.npy, names.json).
usage: python scripts/fv_gate.py configs/fv_eegfree.yaml --stage {lowlevel,gate}
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import argparse, hashlib, json, sys, time
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    h = hashlib.sha256(); h.update(open(path, 'rb').read()); return h.hexdigest()


def srgb_to_lab(rgb):
    """rgb uint8 [..., 3] -> CIELAB (D65) float64 [..., 3]."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29; f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def gabor_bank(n, cycles, orients, k):
    """frequency-domain Gabor filters [len(cycles) * len(orients), n, n] (Gaussian centred at the carrier, peak 1)."""
    u = np.fft.fftfreq(n); U, V = np.meshgrid(u, u, indexing='xy'); out = []
    for cy in cycles:
        fp = cy / n; s = 1 / (2 * np.pi * (k / fp))
        for th in np.deg2rad(orients):
            out.append(np.exp(-((U - fp * np.cos(th)) ** 2 + (V - fp * np.sin(th)) ** 2) / (2 * s * s)))
    return np.stack(out)


def lowlevel_one(im, m, a, G):
    from PIL import Image
    from scipy.ndimage import sobel
    w, h = im.size; s = min(w, h); im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    thumb = srgb_to_lab(np.asarray(im.resize((m['thumb'], m['thumb']), Image.BOX))).ravel()
    lab64 = srgb_to_lab(np.asarray(im.resize((m['hist_size'],) * 2, Image.BOX))).reshape(-1, 3)
    e = m['hist_edges']; hist = np.histogramdd(np.clip(lab64, [0, -128, -128], [100, 127.999, 127.999]), bins=[e['L'], e['ab'], e['ab']])[0].ravel()
    hist = hist / hist.sum()
    L224 = srgb_to_lab(np.asarray(im.resize((224, 224), Image.BOX)))[..., 0] / 100
    lum = np.array([L224.mean() * 100, L224.std() * 100])
    g = srgb_to_lab(np.asarray(im.resize((m['gabor_size'],) * 2, Image.BOX)))[..., 0] / 100
    F = np.fft.fft2(g - g.mean()); n = m['gabor_size']
    gab = np.log((np.abs(F[None] * G) ** 2).sum((1, 2)) / n ** 4 + 1e-8)
    mag = np.hypot(sobel(L224, 0, mode='reflect'), sobel(L224, 1, mode='reflect')) / 8 >= a['edge_threshold']
    k = a['edge_grid']; cell = mag.reshape(k, 224 // k, k, 224 // k).mean((1, 3)).ravel()
    edge = np.concatenate([cell, [mag.mean()]])
    return thumb, hist, lum, gab, edge


def zblock(blocks, ref):
    """z-score each block over the rows in ref, divide by sqrt(dim), concatenate."""
    out = []
    for X in blocks:
        mu, sd = X[ref].mean(0), X[ref].std(0); sd[sd == 0] = 1
        out.append((X - mu) / sd / np.sqrt(X.shape[1]))
    return np.concatenate(out, 1)


def matched_sets(D, n, salt):
    """for each row i: the n nearest other rows by D[i], ties ordered by sha256('{i}|{j}|salt')."""
    out = []
    for i in range(len(D)):
        js = [j for j in range(len(D)) if j != i]
        js.sort(key=lambda j: (D[i, j], hashlib.sha256(f'{i}|{j}|{salt}'.encode()).hexdigest()))
        out.append(js[:n])
    return out


def ridge_fit(X, Y, alphas, groups, folds, salt):
    """standardize X (train stats), centre Y; alpha by concept-group k-fold CV (mean cosine). Returns predict(X) and info."""
    ug = sorted(set(groups), key=lambda g: hashlib.sha256(f'{g}|{salt}'.encode()).hexdigest())
    fold_of = {g: i % folds for i, g in enumerate(ug)}; f = np.array([fold_of[g] for g in groups])

    def solve(Xtr, Ytr):
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1; ym = Ytr.mean(0)
        U, S, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False); UtY = U.T @ (Ytr - ym)
        return lambda al: (lambda W: (lambda Z: (Z - mu) / sd @ W + ym))(Vt.T @ ((S / (S * S + al))[:, None] * UtY))

    def cos(P, Y):
        return float(((P / np.linalg.norm(P, axis=1, keepdims=True)) * (Y / np.linalg.norm(Y, axis=1, keepdims=True))).sum(1).mean())
    cv = np.zeros(len(alphas))
    for k in range(folds):
        tr, va = f != k, f == k; sol = solve(X[tr], Y[tr])
        for ai, al in enumerate(alphas): cv[ai] += cos(sol(al)(X[va]), Y[va]) / folds
    best = alphas[int(np.argmax(cv))]
    return solve(X, Y)(best), dict(alpha=best, cv_cos=[round(v, 5) for v in cv.tolist()])


def mrp_rows(P, T, sets):
    """MRP_i of the true candidate i among {i} + sets[i] (cosine of prediction row i with candidate text rows)."""
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True); r = np.empty(len(sets))
    for i, M in enumerate(sets):
        s = T[M] @ Pn[i]; s0 = T[i] @ Pn[i]; r[i] = ((s < s0).sum() + 0.5 * (s == s0).sum()) / len(M)
    return r


def null_rows(P, T, sets, n, seed):
    """per-concept mean MRP under a fake true candidate drawn uniformly from {i} + sets[i] (n draws)."""
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True); rng = np.random.default_rng(seed); mu = np.zeros(len(sets))
    R = []
    for i, M in enumerate(sets):     # R[i][k]: MRP if candidate k were the true one
        s = T[[i] + list(M)] @ Pn[i]
        R.append(((s[None, :] < s[:, None]).sum(1) + 0.5 * ((s[None, :] == s[:, None]).sum(1) - 1)) / (len(s) - 1))
    for _ in range(n):
        for i, r in enumerate(R): mu[i] += r[rng.integers(len(r))] / n
    return mu


def excess_upper(r, mu, n, seed):
    rng = np.random.default_rng(seed); d = r - mu
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(b, 0.90))


def load_embed(c):
    E = os.path.join(ROOT, c['embed_out']); idx = json.load(open(os.path.join(E, 'index.json')))
    return E, idx


def stage_lowlevel(c):
    from PIL import Image
    E, idx = load_embed(c); m, a = c['matching'], c['audit']; base = os.path.join(ROOT, c['images'])
    G = gabor_bank(m['gabor_size'], m['gabor_cycles'], m['gabor_orient_deg'], m['gabor_sigma_k'])
    feats = {k: [] for k in ('m1_thumb', 'm2_hist', 'm3_lum', 'm4_gabor', 'a4_edge')}; t0 = time.time()
    for n, p in enumerate(idx['paths']):
        for k, v in zip(feats, lowlevel_one(Image.open(os.path.join(base, p)).convert('RGB'), m, a, G)): feats[k].append(v)
        if n % 2000 == 0: print(f'{n}/{len(idx["paths"])} {time.time() - t0:.0f}s', flush=True)
    feats = {k: np.asarray(v, np.float32) for k, v in feats.items()}
    for k, v in feats.items(): assert np.isfinite(v).all(), k
    np.savez(os.path.join(E, 'lowlevel.npz'), **feats)
    te = np.array([s == 'test' for s in idx['split']]); conv1 = np.load(os.path.join(E, 'alex_conv1.npy'))
    Z = zblock([feats['m1_thumb'][te], feats['m2_hist'][te], feats['m3_lum'][te], feats['m4_gabor'][te], conv1[te]], np.arange(te.sum()))
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    tc = [cc for cc, s in zip(idx['concept'], idx['split']) if s == 'test']
    sets = matched_sets(D, m['n_matched'], m['tie_salt'])
    out = os.path.join(ROOT, c['out']); os.makedirs(out, exist_ok=True)
    res = dict(test_folders=tc, matched19=sets, matched9=[s[:m['n_matched_relaxed']] for s in sets],
               lowlevel_sha256=sha256(os.path.join(E, 'lowlevel.npz')), index_sha256=sha256(os.path.join(E, 'index.json')),
               alex_conv1_sha256=sha256(os.path.join(E, 'alex_conv1.npy')), seconds=round(time.time() - t0, 1),
               dims={k: int(v.shape[1]) for k, v in feats.items()} | {'alex_conv1': int(conv1.shape[1])})
    json.dump(res, open(os.path.join(out, 'matched_sets.json'), 'w'))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith(('matched', 'test_'))}), flush=True)


def stage_gate(c):
    E, idx = load_embed(c); g, rg = c['gate'], c['ridge']; out = os.path.join(ROOT, c['out']); t0 = time.time()
    rows = json.load(open(os.path.join(E, 'names.json')))['rows']
    key = {(r['split'], r['folder']): i for i, r in enumerate(rows)}
    img_row = np.array([key[(s, f)] for s, f in zip(idx['split'], idx['concept'])])
    tr = np.array([s == 'train' for s in idx['split']]); te = ~tr
    test_rows = img_row[te]; assert len(set(test_rows)) == 200
    I = np.load(os.path.join(E, 'clip_img.npy'))
    res = dict(names_sha256=sha256(os.path.join(ROOT, c['names'])), embed_files={f: sha256(os.path.join(E, f)) for f in sorted(os.listdir(E)) if f.endswith(('.npy', '.json', '.npz'))})
    mt = json.load(open(os.path.join(E, 'metrics_text.json')))
    assert mt['names_sha256'] == res['names_sha256'], 'text embeddings were made from another name table'
    # G0
    g0 = {}
    for p in ('p1', 'p2'):
        T = np.load(os.path.join(E, f'clip_txt_{p}.npy')); S = I[te] @ T[test_rows].T; Sall = I[te] @ T.T
        rank = (S > np.diag(S)[:, None]).sum(1); top1all = Sall.argmax(1)
        g0[p] = dict(top1=float((rank == 0).mean()), top5=float((rank < 5).mean()), mrp=float((199 - rank).mean() / 199),
                     open_top1=float((top1all == test_rows).mean()), open_top5=float(np.mean([t in np.argsort(-s)[:5] for t, s in zip(test_rows, Sall)])),
                     open_top1_train_share=float(np.mean([rows[k]['split'] == 'train' for k in top1all])))
        if g0['p1']['top1'] >= g['g0_threshold']: break
    use = 'p1' if g0['p1']['top1'] >= g['g0_threshold'] else 'p2' if g0.get('p2', {}).get('top1', 0) >= g['g0_threshold'] else None
    res['g0'] = dict(per_prompt=g0, pass_prompt=use, verdict='pass' if use else 'fail', threshold=g['g0_threshold'])
    if use is None:
        res['zero_v'] = 'not computed: G0 failed (output becomes image-candidate retrieval, D-111 3.3)'
        json.dump(res, open(os.path.join(out, 'gate_g0_0v.json'), 'w'), indent=1); print(json.dumps(res['g0'])); return
    T = np.load(os.path.join(E, f'clip_txt_{use}.npy')); Tt = T[test_rows]
    ms = json.load(open(os.path.join(out, 'matched_sets.json'))); low = np.load(os.path.join(E, 'lowlevel.npz'))
    feats = dict(a1_alex_conv2=np.load(os.path.join(E, 'alex_conv2.npy')), a2_clip_phase=np.load(os.path.join(E, 'clip_img_phase.npy')),
                 a3_clip_blur=np.load(os.path.join(E, 'clip_img_blur.npy')), a4_edge=low['a4_edge'])
    report = dict(m1_thumb=low['m1_thumb'], m2_hist=low['m2_hist'], m3_lum=low['m3_lum'], m4_gabor=low['m4_gabor'],
                  m5_alex_conv1=np.load(os.path.join(E, 'alex_conv1.npy')))
    targets = dict(T=T[img_row], I=I)
    groups = [f for f, s in zip(idx['concept'], idx['split']) if s == 'train']
    allsets = [[j for j in range(200) if j != i] for i in range(200)]

    def cells(fs, sets_list):
        cc = {}
        for fn, X in fs.items():
            for tn, Y in targets.items():
                pred, info = ridge_fit(X[tr].astype(np.float64), Y[tr].astype(np.float64), rg['alphas'], groups, rg['folds'], rg['fold_salt'])
                P = pred(X[te].astype(np.float64)); cell = dict(ridge=info)
                for nm, sets in sets_list.items():
                    r = mrp_rows(P, Tt, sets); mu = null_rows(P, Tt, sets, g['null_perm'], g['null_seed'])
                    ex, up = excess_upper(r, mu, g['boot'], g['boot_seed'])
                    cell[nm] = dict(mrp=float(r.mean()), null_mean=float(mu.mean()), excess=ex, upper90=up)
                cc[f'{fn}|{tn}'] = cell; print(fn, tn, {k: v for k, v in cell.items() if k != 'ridge'}, f'{time.time() - t0:.0f}s', flush=True)
        return cc
    audit = cells(feats, dict(m19=ms['matched19'], m9=ms['matched9'], unmatched=allsets))
    up19 = {k: v['m19']['upper90'] for k, v in audit.items()}; up9 = {k: v['m9']['upper90'] for k, v in audit.items()}
    p19 = all(u < g['upper_limit'] for u in up19.values()); p9 = all(u < g['upper_limit'] for u in up9.values())
    res['zero_v'] = dict(cells=audit, pass_m19=p19, pass_m9=None if p19 else p9,
                         verdict='pass (20 candidates)' if p19 else 'pass (10 candidates, relaxed)' if p9 else 'fail (H2 not tested; alpha to H3)',
                         upper_limit=g['upper_limit'])
    D = 1 - Tt @ Tt.T; ret = [np.median(D[i, ms['matched19'][i]]) / np.median(np.delete(D[i], i)) for i in range(200)]
    res['report'] = dict(semantic_retention=dict(median=float(np.median(ret)), mean=float(np.mean(ret))),
                         matching_family=cells(report, dict(m19=ms['matched19'], unmatched=allsets)))
    res['seconds'] = round(time.time() - t0, 1); res['matched_sets_sha256'] = sha256(os.path.join(out, 'matched_sets.json'))
    json.dump(res, open(os.path.join(out, 'gate_g0_0v.json'), 'w'), indent=1)
    print(json.dumps(dict(g0=res['g0']['verdict'], prompt=use, zero_v=res['zero_v']['verdict'])), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', required=True, choices=['lowlevel', 'gate'])
    a = ap.parse_args(); c = yaml.safe_load(open(os.path.join(ROOT, a.config)))
    stage_lowlevel(c) if a.stage == 'lowlevel' else stage_gate(c)


if __name__ == '__main__':
    main()
