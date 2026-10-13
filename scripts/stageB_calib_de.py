"""Stage B calibration curve baseline (D-090, CPU): DE + ridge fitted on the same k runs as the decoder's subject layer.

Same features, targets and within-run ranking as scripts/stageB_b1_meters.py (decoder setting of configs/stageB_b1.yaml only).
For every subject, run fold and k, the ridge is fitted on the k runs given by stageB_calib_curve.draw_runs (penalty by
leave-one-run-out CV over those runs; for k = 1 the two halves of the run). k = 14 is the B1 baseline file.

usage: python scripts/stageB_calib_de.py configs/stageB_calib.yaml [--smoke]
       --smoke: subjects of subject fold 0 [:2], run fold 0, ranking on the validation runs (never the test runs)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np
from scipy.signal import welch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageB_b1_meters import BANDS, fit_predict, cells


def draw_runs(seed, subject, run_fold, train_runs, k):        # identical to stageB_calib_curve.draw_runs (no torch import here)
    return sorted(int(r) for r in np.random.default_rng([seed, subject, run_fold]).permutation(sorted(train_runs))[:k])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    cc = yaml.safe_load(open(a.config)); c = yaml.safe_load(open(os.path.join(ROOT, cc['b1_config']))); sp = c['split']; t0 = time.time()
    d = os.path.join(ROOT, c['cache_dir']); st = c['settings'][c['decoder_setting']]
    meta = json.load(open(os.path.join(d, 'eeg_meta.json'))); fs = meta['sample_rate']; offs = meta['run_offsets']; lens = meta['run_lengths']
    w, e = c['window'], c['eeg']; T = int(round((w['tmax'] - w['tmin']) * fs)); lo = -w['tmin']
    Wd = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True)
    def hi(r): return lens[r] / fs - w['tmax'] - max(e['offset_s'], 0)
    grid = [(r, float(t)) for r in range(20) for t in np.arange(lo, hi(r) + 1e-9, w['train_step_s'])]
    qs = []
    for r in range(20):
        on = np.unique(Wd['onset'][Wd['run'] == r]); qs += [(r, float(t)) for t in on[(on >= lo) & (on <= hi(r))]]
    items = grid + qs; ng = len(grid); run_all = np.array([r for r, _ in items]); t_all = np.array([t for _, t in items])
    F = np.asarray(np.load(os.path.join(d, c['features']['w2v'] + '.npy'), mmap_mode='r'), np.float32)
    fi = np.array([offs[r] + int(round((t + w['tmin']) * fs)) for r, t in items])
    pooled_raw = np.stack([F[:, i:i + T].mean(1) for i in fi])
    folds = [0] if a.smoke else range(5); fold_data = {}
    for f in folds:
        test = [r - 1 for r in sp['run_folds'][f]]; valid = [r - 1 for r in sp['valid_runs'][f]]; train = [r for r in range(20) if r not in test + valid]
        ev = valid if a.smoke else test
        samp = np.concatenate([np.arange(offs[r], offs[r] + lens[r]) for r in train]); samp = np.sort(samp[:: max(1, len(samp) // 200000)])
        fm, fsd = F[:, samp].mean(1), F[:, samp].std(1) + 1e-6
        fold_data[f] = (train, ng + np.where(np.isin(run_all[ng:], ev))[0], (pooled_raw - fm) / fsd)
    del F
    subjects = sp['subject_folds'][0][:2] if a.smoke else sorted(int(s) for s in meta['subjects'])
    res = {'setting': c['decoder_setting'], 'eval_split': 'valid' if a.smoke else 'test', 'subjects': {}}
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
        out = res['subjects'][str(s)] = {}
        for k in cc['ks']:
            ok = out[f'k{k}'] = {'cells': {}, 'alpha': {}, 'runs': {}}
            for f, (train, te_i, Y) in fold_data.items():
                runs_k = draw_runs(cc['draw_seed'], s, f, train, k); tr_i = np.where(np.isin(run_all[:ng], runs_k))[0]
                grp = (t_all[tr_i] > np.median(t_all[tr_i])).astype(int) if k == 1 else run_all[tr_i]
                P, al = fit_predict(X[tr_i], Y[tr_i], X[te_i], grp); cl, _ = cells(P, Y[te_i], run_all[te_i], t_all[te_i])
                ok['cells'].update(cl); ok['alpha'][str(f)] = al; ok['runs'][str(f)] = [r + 1 for r in runs_k]
            mm = np.average([v['mrp'] for v in ok['cells'].values()], weights=[v['n'] for v in ok['cells'].values()])
            print(f'sub-{s:03d} DE k{k}: within-run MRP {mm:.4f} ({time.time() - t0:.0f}s)', flush=True)
    p = os.path.join(ROOT, cc['de']['out'].replace('.json', '_smoke.json') if a.smoke else cc['de']['out']); os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), indent=1); print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
