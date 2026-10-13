"""Stage B B1 (D-085) linear references on the B1 run split (CPU): DE + ridge baseline and the ORDER meter.

Both predict the time-pooled wav2vec feature (w2v_run, standardised with training-run statistics of each run fold) of each
3-s window and rank every test query against the candidates of the same run by cosine (within-run mean rank percentile,
top-1, top-10), plus top-10 against all candidates of the fold.
  DE: log band power (1-4, 4-8, 8-13, 13-30, 30-45 Hz) x 128 channels of the EEG window after the decoder's reference,
      baseline and clamp; ridge fitted per subject on that subject's own 14 training runs only (calibration data, no other
      subject); penalty by grouped CV over training runs (7 groups of 2 runs). Every subject is a target (19).
  ORDER: no EEG; polynomial and Fourier terms of the relative position inside the run (no run one-hot: test runs unseen).
Computed for both decoder settings (the EEG reference and source differ); D-085 picks the one matching decoder_setting.

usage: python scripts/stageB_b1_meters.py configs/stageB_b1.yaml [--smoke]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np
from scipy.signal import welch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]; ALPHAS = np.logspace(-1, 6, 15)


def ridge(X, Y, a):
    U, s, Vt = np.linalg.svd(X, full_matrices=False); return Vt.T @ ((s / (s ** 2 + a))[:, None] * (U.T @ Y))


def unit(A): return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def fit_predict(Xtr, Ytr, Xte, groups):
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8; ym = Ytr.mean(0); Xtr = (Xtr - xm) / xs; Xte = (Xte - xm) / xs; sc = np.zeros(len(ALPHAS))
    for g in np.unique(groups):
        tr, va = groups != g, groups == g
        for j, a in enumerate(ALPHAS):
            P = Xtr[va] @ ridge(Xtr[tr], Ytr[tr] - ym, a); sc[j] += np.mean(np.sum(unit(P) * unit(Ytr[va] - ym), 1))
    a = ALPHAS[int(np.argmax(sc))]; return Xte @ ridge(Xtr, Ytr - ym, a), float(a)


def cells(P, Yq, run, onsets):
    """within-run ranking per run (1-based keys) and fold-wide top-10"""
    S = unit(P) @ unit(Yq).T; true = np.diag(S); above = S > true[:, None]; out = {}
    for r in np.unique(run):
        m = run == r; rk = above[np.ix_(m, m)].sum(1); n = int(m.sum())
        out[str(int(r) + 1)] = {'mrp': float(np.mean(1 - rk / (n - 1))), 'n': n, 'top1': float((rk < 1).mean()), 'top10': float((rk < 10).mean())}
    return out, float((above.sum(1) < 10).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); sp = c['split']; d = os.path.join(ROOT, c['cache_dir']); t0 = time.time()
    meta = json.load(open(os.path.join(d, 'eeg_meta.json'))); fs = meta['sample_rate']; offs = meta['run_offsets']; lens = meta['run_lengths']
    w, e = c['window'], c['eeg']; T = int(round((w['tmax'] - w['tmin']) * fs)); lo = -w['tmin']
    W = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True)
    def hi(r): return lens[r] / fs - w['tmax'] - max(e['offset_s'], 0)
    grid = [(r, float(t)) for r in range(20) for t in np.arange(lo, hi(r) + 1e-9, w['train_step_s'])]
    qs = []
    for r in range(20):
        on = np.unique(W['onset'][W['run'] == r]); qs += [(r, float(t)) for t in on[(on >= lo) & (on <= hi(r))]]
    items = grid + qs; ng = len(grid); run_all = np.array([r for r, _ in items]); t_all = np.array([t for _, t in items])
    F = np.asarray(np.load(os.path.join(ROOT, c['cache_dir'], c['features']['w2v'] + '.npy'), mmap_mode='r'), np.float32)
    fi = np.array([offs[r] + int(round((t + w['tmin']) * fs)) for r, t in items])
    pooled_raw = np.stack([F[:, i:i + T].mean(1) for i in fi])                                    # [items, 1024]
    rel = t_all / (np.array(lens)[run_all] / fs)
    POS = np.c_[rel, rel ** 2, rel ** 3, *[g(2 * np.pi * k * rel) for k in (1, 2, 4, 8) for g in (np.sin, np.cos)]]
    print(f'{ng} grid windows, {len(qs)} queries, features ready ({time.time() - t0:.0f}s)', flush=True)
    folds = range(1 if a.smoke else 5); fold_data = {}
    for f in folds:
        test = [r - 1 for r in sp['run_folds'][f]]; valid = [r - 1 for r in sp['valid_runs'][f]]; train = [r for r in range(20) if r not in test + valid]
        if a.smoke: test = valid                                                     # smoke: never the test runs
        tr_i = np.where(np.isin(run_all[:ng], train))[0]; te_i = ng + np.where(np.isin(run_all[ng:], test))[0]
        samp = np.concatenate([np.arange(offs[r], offs[r] + lens[r]) for r in train]); samp = np.sort(samp[:: max(1, len(samp) // 200000)])
        fm, fsd = F[:, samp].mean(1), F[:, samp].std(1) + 1e-6
        Y = (pooled_raw - fm) / fsd; grp = np.array([train.index(r) // 2 for r in run_all[tr_i]])
        fold_data[f] = (tr_i, te_i, Y, grp)
    del F
    res = {'folds': {}, 'ORDER': {}, 'DE': {}}
    for f, (tr_i, te_i, Y, grp) in fold_data.items():
        P, al = fit_predict(POS[tr_i], Y[tr_i], POS[te_i], grp); cl, full = cells(P, Y[te_i], run_all[te_i], t_all[te_i])
        res['ORDER'].update(cl); res['folds'][str(f)] = {'order_alpha': al, 'order_top10_full': full, 'n_queries': int(len(te_i))}
    print('ORDER', json.dumps({r: round(v['mrp'], 4) for r, v in res['ORDER'].items()}), flush=True)
    subjects = sp['subject_folds'][0][:2] if a.smoke else sorted(int(s) for s in meta['subjects'])   # smoke: the smoke decoder's targets
    for setting, st in c['settings'].items():
        out = res['DE'][setting] = {}
        for s in subjects:
            eeg = np.load(os.path.join(d, f"{st['source']}_sub-{s:03d}.npy"), mmap_mode='r'); flat = np.array(meta['subjects'][str(s)]['flat'])
            X = np.empty((len(items), 128 * len(BANDS)), np.float32)
            for b0 in range(0, len(items), 1000):
                xs = []
                for (r, t) in items[b0:b0 + 1000]:
                    i = offs[r] + int(round((t + w['tmin'] + e['offset_s']) * fs)); x = np.asarray(eeg[:, i:i + T], np.float32).copy()
                    if st['reference'] == 'car': x -= x[~flat[r]].mean(0, keepdims=True)
                    x -= x[:, : int(round(e['baseline_s'] * fs))].mean(1, keepdims=True); xs.append(np.clip(x, -e['clamp'], e['clamp']))
                fr, p = welch(np.stack(xs), fs=fs, nperseg=min(fs, T), axis=-1)
                X[b0:b0 + len(xs)] = np.log(np.stack([p[..., (fr >= l) & (fr < h)].mean(-1) for l, h in BANDS], -1) + 1e-8).reshape(len(xs), -1)
            out[str(s)] = {'cells': {}, 'top10_full': {}, 'alpha': {}}
            for f, (tr_i, te_i, Y, grp) in fold_data.items():
                P, al = fit_predict(X[tr_i], Y[tr_i], X[te_i], grp); cl, full = cells(P, Y[te_i], run_all[te_i], t_all[te_i])
                out[str(s)]['cells'].update(cl); out[str(s)]['top10_full'][str(f)] = full; out[str(s)]['alpha'][str(f)] = al
            m = np.mean([v['mrp'] for v in out[str(s)]['cells'].values()])
            print(f'{setting} DE sub-{s:03d}: within-run MRP {m:.4f} ({time.time() - t0:.0f}s)', flush=True)
    res['eval_split'] = 'valid' if a.smoke else 'test'
    p = os.path.join(ROOT, 'results/baselines', 'stageB_b1_de_order' + ('_smoke' if a.smoke else '') + '.json'); os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), indent=1); print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
