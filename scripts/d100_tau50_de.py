"""D-100 (a) DE + ridge under the same perturbations (CPU features; the null uses torch on the available device).

The B1 baseline (scripts/stageB_b1_meters.py, A4 setting): log band power (1-4, 4-8, 8-13, 13-30, 30-45 Hz) x 128 channels
after CAR (flat channels excluded), 0.5-s baseline and clamp; ridge per subject on that subject's own 14 training runs of the
run fold (grouped CV, 7 groups of 2 runs) predicting the time-pooled w2v feature. Training windows are unperturbed; the test
query windows get the shift delta or the stretch r of scripts/d100_tau50.py. Scoring, candidates (>= 30 s) and the null are
those of scripts/d100_tau50.py. Output: data/cache/d100_tau50/DE.npz (key de_s{s}_r{run}_{cond}: observed, n, 1,000 null means).
The ridge (alpha choice by grouped CV and the final fit) depends only on the unperturbed training windows, so it is fitted
once per (subject, run fold) and applied to every condition (same numbers as stageB_b1_meters.fit_predict per condition).
Each subject is saved to de_parts/DE_s{s}.npz and DE.npz is written once all 17 parts exist (resumable, one job per subject).

usage: python scripts/d100_tau50_de.py configs/d100_tau50.yaml [--subjects 2 3 ...]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np, torch
from scipy.signal import welch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageB_b1_meters import BANDS, ALPHAS, ridge, unit
from d100_tau50 import common_queries, run_maps, excl_stats


def fit_once(Xtr, Ytr, groups):
    """stageB_b1_meters.fit_predict split into fit and predict: returns (xm, xs, weights); prediction = ((X - xm) / xs) @ weights"""
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8; ym = Ytr.mean(0); Xtr = (Xtr - xm) / xs; sc = np.zeros(len(ALPHAS))
    for g in np.unique(groups):
        tr, va = groups != g, groups == g
        for j, a in enumerate(ALPHAS):
            P = Xtr[va] @ ridge(Xtr[tr], Ytr[tr] - ym, a); sc[j] += np.mean(np.sum(unit(P) * unit(Ytr[va] - ym), 1))
    return xm, xs, ridge(Xtr, Ytr - ym, ALPHAS[int(np.argmax(sc))])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--subjects', type=int, nargs='*'); a = ap.parse_args(); c = yaml.safe_load(open(a.config)); t0 = time.time()
    b1 = yaml.safe_load(open(os.path.join(ROOT, c['b1_config']))); sp = b1['split']; d = os.path.join(ROOT, b1['cache_dir'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; st = b1['settings'][b1['decoder_setting']]
    meta = json.load(open(os.path.join(d, 'eeg_meta.json'))); fs = meta['sample_rate']; offs = meta['run_offsets']; lens = meta['run_lengths']
    w, e = b1['window'], b1['eeg']; T = int(round((w['tmax'] - w['tmin']) * fs)); lo = -w['tmin']
    W = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True)
    def hi(r): return lens[r] / fs - w['tmax'] - max(e['offset_s'], 0)
    grid = [(r, float(t)) for r in range(20) for t in np.arange(lo, hi(r) + 1e-9, w['train_step_s'])]
    qs = []
    for r in range(20):
        on = np.unique(W['onset'][W['run'] == r]); qs += [(r, float(t)) for t in on[(on >= lo) & (on <= hi(r))]]
    keep = common_queries(c, b1, lens, fs, qs); qk = [qs[k] for k in keep]; qrun = np.array([x[0] for x in qk]); qt = np.array([x[1] for x in qk])
    F = np.asarray(np.load(os.path.join(d, b1['features']['w2v'] + '.npy'), mmap_mode='r'), np.float32)
    def pooled(items): return np.stack([F[:, offs[r] + int(round((t + w['tmin']) * fs)):offs[r] + int(round((t + w['tmin']) * fs)) + T].mean(1) for r, t in items])
    Pg, Pq = pooled(grid), pooled(qk); grun = np.array([r for r, _ in grid])
    stats = {}
    for f in range(5):
        test = [r - 1 for r in sp['run_folds'][f]]; valid = [r - 1 for r in sp['valid_runs'][f]]; train = [r for r in range(20) if r not in test + valid]
        samp = np.concatenate([np.arange(offs[r], offs[r] + lens[r]) for r in train]); samp = np.sort(samp[:: max(1, len(samp) // 200000)])
        stats[f] = (train, test, F[:, samp].mean(1), F[:, samp].std(1) + 1e-6)
    del F
    conds = [('d', round(x, 2)) for x in c['deltas_s']] + [('s', x) for x in c['stretch_r']]
    maps = run_maps({rr: qt[qrun == rr] for rr in range(20)}, c['shift_null']['n_shift'], c['shift_null']['min_shift_s'], c['shift_null']['seed'], 0)
    pool = sorted(s for fs_ in sp['subject_folds'] for s in fs_); out = os.path.join(ROOT, c['cache']); parts = os.path.join(out, 'de_parts'); os.makedirs(parts, exist_ok=True)
    def window(eeg, flat, r, t, kind, val):
        if kind == 'd':
            i = offs[r] + int(round((t + w['tmin'] + e['offset_s'] + val) * fs)); x = np.asarray(eeg[:, i:i + T], np.float32).copy()
        else:
            a_ = t + e['offset_s']; p0 = (a_ + w['tmin'] * val) * fs; i = offs[r] + int(np.floor(p0)); n_raw = int(np.ceil((w['tmax'] - w['tmin']) * val * fs)) + 2
            raw = np.asarray(eeg[:, i:i + n_raw], np.float32); pos = (p0 - np.floor(p0)) + np.arange(T) * val; i0 = np.floor(pos).astype(int); fr = (pos - i0).astype(np.float32)
            x = raw[:, i0] * (1 - fr) + raw[:, i0 + 1] * fr
        if st['reference'] == 'car': x -= x[~flat[r]].mean(0, keepdims=True)
        x -= x[:, : int(round(e['baseline_s'] * fs))].mean(1, keepdims=True); return np.clip(x, -e['clamp'], e['clamp'])
    def de(xs):
        fr, p = welch(np.stack(xs), fs=fs, nperseg=min(fs, T), axis=-1)
        return np.log(np.stack([p[..., (fr >= l) & (fr < h)].mean(-1) for l, h in BANDS], -1) + 1e-8).reshape(len(xs), -1)
    for s in (a.subjects or pool):
        part = os.path.join(parts, f'DE_s{s}.npz')
        if os.path.exists(part): print('skip', s, flush=True); continue
        arrays = {}
        eeg = np.load(os.path.join(d, f"{st['source']}_sub-{s:03d}.npy"), mmap_mode='r'); flat = np.array(meta['subjects'][str(s)]['flat'])
        Xg = np.concatenate([de([window(eeg, flat, r, t, 'd', 0.0) for r, t in grid[b0:b0 + 1000]]) for b0 in range(0, len(grid), 1000)])
        Xq = {cd: np.concatenate([de([window(eeg, flat, r, t, *cd) for r, t in qk[b0:b0 + 1000]]) for b0 in range(0, len(qk), 1000)]) for cd in conds}
        for f, (train, test, fm, fsd) in stats.items():
            tr_i = np.where(np.isin(grun, train))[0]; grp = np.array([train.index(r) // 2 for r in grun[tr_i]]); Yg = (Pg - fm) / fsd; Yq = (Pq - fm) / fsd
            te = np.where(np.isin(qrun, test))[0]; xm, xs, Wr = fit_once(Xg[tr_i], Yg[tr_i], grp)
            for cd in conds:
                P = ((Xq[cd][te] - xm) / xs) @ Wr
                for rr in test:
                    m = qrun[te] == rr; Sm = torch.as_tensor(unit(P[m]) @ unit(Yq[te][m]).T, device=dev)
                    o_, n_ = excl_stats(Sm, qt[te][m], maps[rr], c['exclusion_s']); key = f"de_s{s}_r{rr}_{cd[0]}{cd[1]:+.2f}" if cd[0] == 'd' else f"de_s{s}_r{rr}_s{cd[1]:.2f}"
                    arrays[key] = np.r_[o_, int(m.sum()), n_].astype(np.float32)
        print(f'DE sub-{s:03d}: delta0 ' + f"{np.mean([arrays[f'de_s{s}_r{rr}_d+0.00'][0] for rr in range(20)]):.4f} ({time.time() - t0:.0f}s)", flush=True)
        np.savez_compressed(part, **arrays)
    if all(os.path.exists(os.path.join(parts, f'DE_s{s}.npz')) for s in pool):
        allv = {}
        for s in pool: Z = np.load(os.path.join(parts, f'DE_s{s}.npz')); allv.update({k: Z[k] for k in Z.files})
        np.savez_compressed(os.path.join(out, 'DE.npz'), **allv); print('saved DE.npz', len(allv), f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
