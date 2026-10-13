"""D-092 stage 0 (CPU): DE + ridge baseline and the ORDER meter on the Alice main split (segment 4-fold).

Both predict the time-pooled wav2vec feature (w2v_seg, standardised with training-segment statistics of each segment fold)
of each 3-s window and rank every test query against the candidates of the same segment by cosine (within-segment mean
rank percentile, top-1, top-10), plus top-10 against all candidates of the fold.
  Training windows: every train_step_s inside the 8 training segments of the fold (validation segment excluded), dropping
  windows whose EEG span is within guard_s of a test-fold boundary (D-092 boundary guard).
  DE: log band power (1-4, 4-8, 8-13, 13-30, 30-45 Hz) of the good channels (provider bad marks and flat channels removed)
      of the Défossez-arm EEG window (eeg_hp05, CAR over good channels, baseline, clamp); ridge fitted per subject on that
      subject's own training segments only; penalty by grouped CV over training segments.
  ORDER: no EEG; polynomial and Fourier terms of the relative position inside the segment.
Segments missing for a subject (no onset annotation) give no training windows and no queries for that subject.

usage: python scripts/alice_meters.py configs/alice_meters.yaml [--smoke]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np
from scipy.signal import welch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageB_b1_meters import ridge, unit, fit_predict, BANDS


def cells(P, Yq, seg):
    """within-segment ranking per segment (1-based keys) and fold-wide top-10"""
    S = unit(P) @ unit(Yq).T; true = np.diag(S); above = S > true[:, None]; out = {}
    for r in np.unique(seg):
        m = seg == r; rk = above[np.ix_(m, m)].sum(1); n = int(m.sum())
        out[str(int(r) + 1)] = {'mrp': float(np.mean(1 - rk / (n - 1))), 'n': n, 'top1': float((rk < 1).mean()), 'top10': float((rk < 10).mean())}
    return out, float((above.sum(1) < 10).mean())


def wmean(cl): return float(sum(v['mrp'] * v['n'] for v in cl.values()) / sum(v['n'] for v in cl.values()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); d = os.path.join(ROOT, c['cache_dir']); t0 = time.time()
    meta = json.load(open(os.path.join(d, 'eeg_meta.json'))); sp = json.load(open(os.path.join(d, 'split.json'))); main_ = sp['main']
    src = meta['sources'][c['source']]; fs = src['sample_rate']; offs, lens = src['run_offsets'], src['run_lengths']
    am = json.load(open(os.path.join(d, 'audio_meta.json'))); assert am['run_offsets'] == offs and am['sample_rate'] == fs
    w, e = c['window'], c['eeg']; T = int(round((w['tmax'] - w['tmin']) * fs)); lo = -w['tmin']; nseg = len(lens); durs = [L / fs for L in lens]
    Wd = np.load(os.path.join(d, 'words.npz'), allow_pickle=True)
    def hi(r): return durs[r] - w['tmax'] - max(w['offset_s'], 0)
    grid = [(r, float(t)) for r in range(nseg) for t in np.arange(lo, hi(r) + 1e-9, w['train_step_s'])]
    qs = []
    for r in range(nseg):
        on = np.unique(Wd['onset'][Wd['seg'] == r]); qs += [(r, float(t)) for t in on[(on >= lo) & (on <= hi(r))]]
    assert [sum(1 for r, _ in qs if r == k) for k in range(nseg)] == main_['queries_per_segment']
    items = grid + qs; ng = len(grid); seg_all = np.array([r for r, _ in items]); t_all = np.array([t for _, t in items])
    F = np.load(os.path.join(d, c['features']['w2v'] + '.npy'), mmap_mode='r')
    fi = np.array([offs[r] + int(round((t + w['tmin']) * fs)) for r, t in items])
    pooled_raw = np.stack([np.asarray(F[:, i:i + T], np.float32).mean(1) for i in fi])
    rel = t_all / np.array(durs)[seg_all]
    POS = np.c_[rel, rel ** 2, rel ** 3, *[g(2 * np.pi * k * rel) for k in (1, 2, 4, 8) for g in (np.sin, np.cos)]]
    # EEG span of each window in concatenated time, for the boundary guard
    g0 = np.array(offs[:-1]) / fs; span0 = g0[seg_all] + t_all + w['tmin'] + w['offset_s']; span1 = span0 + (w['tmax'] - w['tmin'])
    print(f'{ng} grid windows, {len(qs)} queries ({time.time() - t0:.0f}s)', flush=True)
    folds = range(1 if a.smoke else len(main_['segment_folds'])); fold_data = {}; guard = {}
    for f in folds:
        test = [k - 1 for k in main_['segment_folds'][f]]; valid = main_['valid_segment'][f] - 1; train = [r for r in range(nseg) if r not in test + [valid]]
        if a.smoke: test = [valid]                                                # smoke: never the test segments
        b0, b1 = g0[min(test)], g0[max(test)] + durs[max(test)]
        near = np.minimum(np.abs(span0 - b0), np.abs(span1 - b0)); near = np.where((span0 <= b0) & (span1 >= b0), 0, near)
        near1 = np.minimum(np.abs(span0 - b1), np.abs(span1 - b1)); near1 = np.where((span0 <= b1) & (span1 >= b1), 0, near1)
        ok = np.minimum(near, near1) >= c['guard_s']
        tr_all = np.where(np.isin(seg_all[:ng], train))[0]; tr_i = tr_all[ok[tr_all]]; te_i = ng + np.where(np.isin(seg_all[ng:], test))[0]
        guard[str(f)] = {'train_windows': int(len(tr_i)), 'dropped_by_guard': int(len(tr_all) - len(tr_i))}
        samp = np.concatenate([np.arange(offs[r], offs[r] + lens[r]) for r in train]); samp = np.sort(samp[:: max(1, len(samp) // 200000)])
        Fs = np.asarray(F[:, samp], np.float32); Y = (pooled_raw - Fs.mean(1)) / (Fs.std(1) + 1e-6)
        fold_data[f] = (tr_i, te_i, Y, seg_all[tr_i])
    res = {'folds': {}, 'guard': guard, 'ORDER': {}, 'DE': {}}
    for f, (tr_i, te_i, Y, grp) in fold_data.items():
        P, al = fit_predict(POS[tr_i], Y[tr_i], POS[te_i], grp); cl, full = cells(P, Y[te_i], seg_all[te_i])
        res['ORDER'].update(cl); res['folds'][str(f)] = {'order_alpha': al, 'order_top10_full': full, 'n_queries': int(len(te_i))}
    res['ORDER_mrp'] = wmean(res['ORDER']); print('ORDER', round(res['ORDER_mrp'], 4), json.dumps({r: round(v['mrp'], 4) for r, v in res['ORDER'].items()}), flush=True)
    subjects = sp['subjects'][:2] if a.smoke else sp['subjects']
    for s in subjects:
        info = meta['subjects'][str(s)]; miss = [k - 1 for k in info['missing_segments']]
        good = ~(np.array(info['bad_mask']) | np.array(info['sources'][c['source']]['flat']))
        eeg = np.load(os.path.join(d, f"{c['source']}_sub-{s:03d}.npy"), mmap_mode='r')
        X = np.empty((len(items), int(good.sum()) * len(BANDS)), np.float32)
        for b0_ in range(0, len(items), 1000):
            xs = []
            for (r, t) in items[b0_:b0_ + 1000]:
                i = offs[r] + int(round((t + w['tmin'] + w['offset_s']) * fs)); x = np.asarray(eeg[good, i:i + T], np.float32).copy()
                x -= x.mean(0, keepdims=True); x -= x[:, : int(round(e['baseline_s'] * fs))].mean(1, keepdims=True); xs.append(np.clip(x, -e['clamp'], e['clamp']))
            fr, p = welch(np.stack(xs), fs=fs, nperseg=min(fs, T), axis=-1)
            X[b0_:b0_ + len(xs)] = np.log(np.stack([p[..., (fr >= l) & (fr < h)].mean(-1) for l, h in BANDS], -1) + 1e-8).reshape(len(xs), -1)
        out = res['DE'][str(s)] = {'cells': {}, 'top10_full': {}, 'alpha': {}, 'n_good_channels': int(good.sum())}
        for f, (tr_i, te_i, Y, grp) in fold_data.items():
            tr = tr_i[~np.isin(seg_all[tr_i], miss)]; te = te_i[~np.isin(seg_all[te_i], miss)]
            if len(te) == 0: continue
            P, al = fit_predict(X[tr], Y[tr], X[te], seg_all[tr]); cl, full = cells(P, Y[te], seg_all[te])
            out['cells'].update(cl); out['top10_full'][str(f)] = full; out['alpha'][str(f)] = al
        out['mrp'] = wmean(out['cells'])
        print(f'DE S{s:02d}: within-segment MRP {out["mrp"]:.4f}, good channels {int(good.sum())} ({time.time() - t0:.0f}s)', flush=True)
    allc = [v for s in res['DE'].values() for v in s['cells'].values()]
    res['DE_mrp'] = float(sum(v['mrp'] * v['n'] for v in allc) / sum(v['n'] for v in allc))
    res['DE_top10'] = float(sum(v['top10'] * v['n'] for v in allc) / sum(v['n'] for v in allc))
    res['eval_split'] = 'valid' if a.smoke else 'test'; res['config'] = c
    p = os.path.join(ROOT, c['out'] + ('_smoke' if a.smoke else '') + '.json'); os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), indent=1); print('DE', round(res['DE_mrp'], 4), 'wrote', p, flush=True)


if __name__ == '__main__':
    main()
