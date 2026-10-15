"""C2 baseline (D-097 section 14, CPU): DE + ridge fitted on the same k calibration trials as the C2 decoders' subject layers.

scripts/c1_de.py on the DTU two-talker trials (attended stream): DE features (log Welch power in the stageB_b1_meters bands)
of each 3-s EEG window after CAR without bad and flat channels, baseline and clamp (the decoder's EEG handling); target =
window mean of the attended-stream w2v features standardised with the decoder's statistics of the job (DTUData of subject
fold K, trial fold f); ridge penalty chosen by leave-one-trial-out CV over the k trials (stageB_b1_meters.fit_predict); same
nested draws as the decoders (c2_dtu_split.draw, calibration_curve.draw_seed); within-trial ranking of the test-trial
queries (query_step_s grid, candidates = the query windows of the same trial), cells keyed by trial number.
The training grid (train_step_s) and the query grid (query_step_s) are the same 0.5-s grid here, so the features of every
trial are computed once per subject.

usage: python scripts/c2_de.py configs/c2_p1.yaml [--smoke]
       --smoke: two targets of subject fold 0, trial fold 0, ranking on the validation trials (never the test trials)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np
from scipy.signal import welch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'src'))
from stageB_b1_meters import BANDS, fit_predict, cells
from c2_dtu_split import DTUSplit, trials_from_cache, path
from c2_dtu_data import DTUData


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    cc = yaml.safe_load(open(path(a.config))); dd = cc['de']; c = yaml.safe_load(open(path(cc['b1_config']))); fc = yaml.safe_load(open(path(cc['folds_config'])))
    assert fc['query_step_s'] == c['window']['train_step_s'], 'one grid for training items and queries'
    c['eeg']['reference'] = c['settings'][c['decoder_setting']]['reference']; t0 = time.time()
    split = DTUSplit(fc, trials_from_cache(fc['cache_dir'])); w, e = c['window'], c['eeg']
    jobs = [(0, 0)] if a.smoke else [(K, f) for K in range(5) for f in range(5)]
    res = {'setting': c['decoder_setting'], 'car_exclude_bad': dd['car_exclude_bad'], 'ks': [str(k) for k in dd['ks']],
           'eval_split': 'valid' if a.smoke else 'test', 'subjects': {}}
    feats = {}                                                             # subject -> (items, X, raw pooled Y)
    for K, f in jobs:
        D = DTUData(c, fc, K, f, split=split); targets = D.targets[:2] if a.smoke else D.targets
        for s in targets:
            if s not in feats:
                recs = [r for r in split.trials[s]]; items = D.grid(s, recs, w['train_step_s']); fs = D.fs; T = D.T
                bad = {r['index']: np.array(D.recs[s][r['index']]['bad'], bool) | np.array(D.recs[s][r['index']]['flat'], bool) for r in recs}
                if not dd['car_exclude_bad']: bad = {i: np.zeros_like(b) for i, b in bad.items()}
                X = np.empty((len(items), D.eeg[s].shape[0] * len(BANDS)), np.float32); Yr = np.empty((len(items), D.F.shape[0]), np.float32)
                for b0 in range(0, len(items), 1000):
                    xs = []
                    for (_, i, t) in items[b0:b0 + 1000]:
                        r = D.recs[s][i]; j = r['offset'] + int(round((t + w['tmin'] + e['offset_s']) * fs)); assert r['offset'] <= j and j + T <= r['offset'] + r['length']
                        x = np.asarray(D.eeg[s][:, j:j + T], np.float32).copy()
                        if e['reference'] == 'car': x -= x[~bad[i]].mean(0, keepdims=True)
                        x -= x[:, : int(round(e['baseline_s'] * fs))].mean(1, keepdims=True); xs.append(np.clip(x, -e['clamp'], e['clamp']))
                    fr, p = welch(np.stack(xs), fs=fs, nperseg=min(int(fs), T), axis=-1)
                    X[b0:b0 + len(xs)] = np.log(np.stack([p[..., (fr >= l) & (fr < h)].mean(-1) for l, h in BANDS], -1) + 1e-8).reshape(len(xs), -1)
                for n, (_, i, t) in enumerate(items):
                    fi = D.feat_idx(s, i, t); Yr[n] = np.asarray(D.F[:, fi:fi + T], np.float32).mean(1)
                feats[s] = (items, X, Yr)
            items, X, Yr = feats[s]; Y = (Yr - D.fmean[:, 0]) / D.fstd[:, 0]
            trial = np.array([D.recs[s][i]['trial'] for _, i, _ in items]); rec = np.array([i for _, i, _ in items])
            ev = {r['index'] for r in D.sets[s]['valid' if a.smoke else 'test']}; te_i = np.where(np.isin(rec, list(ev)))[0]
            out = res['subjects'].setdefault(str(s), {'subject_fold': K})
            for k in dd['ks']:
                kn = 'kmax' if str(k) == 'max' else f'k{k}'; ok = out.setdefault(kn, {'cells': {}, 'alpha': {}, 'trials': {}})
                recs_k = split.draw(s, f, k, cc['calibration_curve']['draw_seed']); tr_i = np.where(np.isin(rec, [r['index'] for r in recs_k]))[0]
                P, al = fit_predict(X[tr_i], Y[tr_i], X[te_i], trial[tr_i]); cl, _ = cells(P, Y[te_i], trial[te_i] - 1, np.zeros(len(te_i)))
                ok['cells'].update(cl); ok['alpha'][str(f)] = al; ok['trials'][str(f)] = [r['trial'] for r in recs_k]
        print(f'subject fold {K} trial fold {f}: targets {targets} ({time.time() - t0:.0f}s)', flush=True)
    for s, out in res['subjects'].items():
        print(f'sub-{int(s):03d} DE ' + ' '.join(f'{kn} {np.average([v["mrp"] for v in out[kn]["cells"].values()], weights=[v["n"] for v in out[kn]["cells"].values()]):.4f}'
                                            for kn in out if kn.startswith('k')) + f' ({len(out["k1"]["cells"])} trials)', flush=True)
    p = path(dd['out'].replace('.json', '_smoke.json') if a.smoke else dd['out']); os.makedirs(os.path.dirname(p), exist_ok=True)
    res['seconds'] = time.time() - t0; json.dump(res, open(p, 'w'), indent=1); print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
