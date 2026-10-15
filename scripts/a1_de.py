"""F-III A1 baseline (D-105, CPU, before any A1 decoder result): DE + ridge under the J250 condition and the lag search.

scripts/c1_de.py (D-097) unchanged for the fit: 5-band log power x 128 channels, CAR without the BIDS bad and flat channels,
ridge fitted on the target's own k runs (c1 calibration_curve.draw_seed draws, delta 0 training windows), penalty by
leave-one-run-out CV (k = 1: the two halves of the run). Changes for A1 (configs/a1.yaml): the test queries are the word
onsets whose EEG windows moved by up to queries.margin_s stay inside the run; the query EEG window is moved by the
subject's J250 offset delta_i (+ each lag l); candidates are the truth and the same-run queries >= 30 s away; scores are the
cosine of the ridge prediction with the candidates' window-mean targets; every LS rule of configs/a1.yaml is computed (the
verdict uses the frozen one). Also delta 0 without search on the same queries (protection-condition definition).

usage: python scripts/a1_de.py configs/a1.yaml [--smoke]
       --smoke: subjects of subject fold 0 [:2], run fold 0, ranking on the validation runs (never the test runs)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np, torch
from scipy.signal import welch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageB_b1_meters import BANDS, fit_predict
torch.set_num_threads(int(os.environ.get('ESE_NUM_THREADS', '8')))


def draw_runs(seed, subject, run_fold, train_runs, k):        # identical to c1_de.draw_runs
    return sorted(int(r) for r in np.random.default_rng([seed, subject, run_fold]).permutation(sorted(train_runs))[:k])


def lags(a):
    g = a['lag_search']['lags_s']; return [round(x, 2) for x in np.arange(g['lo'], g['hi'] + 1e-9, g['step'])]


def j_offsets(a, subject, run, onsets_all):
    return np.random.default_rng([a['j250']['seed'], int(subject), int(run)]).uniform(-a['j250']['width_s'], a['j250']['width_s'], len(onsets_all))


def aggregate(S, rule):                                       # same as a1_lib.aggregate (torch, CPU)
    if rule == 'max': return S.max(0).values
    c = float(rule[3:]); tau = (c * S.permute(1, 0, 2).reshape(S.shape[1], -1).std(1)).clamp(min=1e-6)[None, :, None]
    return tau[0] * torch.logsumexp(S / tau, 0)


def mrp_excl(S, t, excl):                                     # same as a1_lib.mrp_excl
    tt = torch.as_tensor(np.asarray(t), dtype=torch.float64); far = (tt[:, None] - tt[None]).abs() >= excl
    above = (S > S.diagonal()[:, None]) & far; return float((1 - above.sum(1).double() / far.sum(1).clamp(min=1).double()).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    A = yaml.safe_load(open(a.config)); cc = yaml.safe_load(open(os.path.join(ROOT, A['c1_config']))); dd = cc['de']
    c = yaml.safe_load(open(os.path.join(ROOT, A['b1_config']))); sp = c['split']; t0 = time.time()
    d = os.path.join(ROOT, c['cache_dir']); st = c['settings'][c['decoder_setting']]
    meta = json.load(open(os.path.join(d, 'eeg_meta.json'))); fs = meta['sample_rate']; offs = meta['run_offsets']; lens = meta['run_lengths']
    w, e = c['window'], c['eeg']; T = int(round((w['tmax'] - w['tmin']) * fs)); lo = -w['tmin']; mg = A['queries']['margin_s']; excl = A['queries']['exclusion_s']
    Wd = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True); lag_list = lags(A); rules = A['lag_search']['rules']
    def hi(r): return lens[r] / fs - w['tmax'] - max(e['offset_s'], 0)
    grid = [(r, float(t)) for r in range(20) for t in np.arange(lo, hi(r) + 1e-9, w['train_step_s'])]
    onsets = {r: np.unique(Wd['onset'][Wd['run'] == r]) for r in range(20)}; qs = []
    for r in range(20):
        on = onsets[r]; on = on[(on >= lo) & (on <= hi(r))]
        ok = (on + w['tmin'] + e['offset_s'] - mg >= 0) & (on + w['tmax'] + e['offset_s'] + mg <= lens[r] / fs - 1.0 / fs)   # a1_lib.inside
        qs += [(r, float(t)) for t in on[ok]]
    items = grid + qs; ng = len(grid); run_all = np.array([r for r, _ in items])
    F = np.asarray(np.load(os.path.join(d, c['features']['w2v'] + '.npy'), mmap_mode='r'), np.float32)
    fi = np.array([offs[r] + int(round((t + w['tmin']) * fs)) for r, t in items]); pooled_raw = np.stack([F[:, i:i + T].mean(1) for i in fi])
    folds = [0] if a.smoke else range(5); fold_data = {}
    for f in folds:
        test = [r - 1 for r in sp['run_folds'][f]]; valid = [r - 1 for r in sp['valid_runs'][f]]; train = [r for r in range(20) if r not in test + valid]
        ev = valid if a.smoke else test
        samp = np.concatenate([np.arange(offs[r], offs[r] + lens[r]) for r in train]); samp = np.sort(samp[:: max(1, len(samp) // 200000)])
        fm, fsd = F[:, samp].mean(1), F[:, samp].std(1) + 1e-6
        fold_data[f] = (train, ng + np.where(np.isin(run_all[ng:], ev))[0], (pooled_raw - fm) / fsd)
    del F
    subjects = sp['subject_folds'][0][:2] if a.smoke else sorted(int(s) for s in meta['subjects'])
    res = dict(setting=c['decoder_setting'], car_exclude_bad=dd['car_exclude_bad'], ks=dd['ks'], eval_split='valid' if a.smoke else 'test', lags=lag_list, rules=rules,
               j250=A['j250'], queries=A['queries'], subjects={})

    def feats(eeg, flat, its, shifts):
        X = np.empty((len(its), 128 * len(BANDS)), np.float32)
        for b0 in range(0, len(its), 1000):
            xs = []
            for (r, t), sh in zip(its[b0:b0 + 1000], shifts[b0:b0 + 1000]):
                i = offs[r] + int(round((t + w['tmin'] + e['offset_s'] + sh) * fs)); x = np.asarray(eeg[:, i:i + T], np.float32).copy()
                if st['reference'] == 'car': x -= x[~flat[r]].mean(0, keepdims=True)
                x -= x[:, : int(round(e['baseline_s'] * fs))].mean(1, keepdims=True); xs.append(np.clip(x, -e['clamp'], e['clamp']))
            fr, p = welch(np.stack(xs), fs=fs, nperseg=min(fs, T), axis=-1)
            X[b0:b0 + len(xs)] = np.log(np.stack([p[..., (fr >= l) & (fr < h)].mean(-1) for l, h in BANDS], -1) + 1e-8).reshape(len(xs), -1)
        return X

    for s in subjects:
        eeg = np.load(os.path.join(d, f"{st['source']}_sub-{s:03d}.npy"), mmap_mode='r'); flat = np.array(meta['subjects'][str(s)]['flat'])
        if dd['car_exclude_bad']: flat = flat | np.array(meta['subjects'][str(s)]['bad'])
        Xg = feats(eeg, flat, grid, np.zeros(ng))                                        # training windows (true alignment)
        qd = {r: dict(zip(np.round(onsets[r], 6).tolist(), j_offsets(A, s, r, onsets[r]).tolist())) for r in range(20)}
        delta = np.array([qd[r][round(t, 6)] for r, t in qs])
        Xq = {'d0': feats(eeg, flat, qs, np.zeros(len(qs)))}
        for l in lag_list: Xq[l] = feats(eeg, flat, qs, delta + l)
        out = res['subjects'][str(s)] = {}
        for k in dd['ks']:
            ok = out[f'k{k}'] = {'cells': {}, 'alpha': {}, 'runs': {}}
            for f, (train, te_i, Y) in fold_data.items():
                runs_k = draw_runs(cc['calibration_curve']['draw_seed'], s, f, train, k); tr_i = np.where(np.isin(run_all[:ng], runs_k))[0]
                t_tr = np.array([t for _, t in grid])[tr_i]; grp = (t_tr > np.median(t_tr)).astype(int) if k == 1 else run_all[tr_i]
                qi = te_i - ng; keys = list(Xq)                                            # one ridge fit, applied to every query offset
                Pall, al = fit_predict(Xg[tr_i], Y[tr_i], np.concatenate([Xq[kk][qi] for kk in keys]), grp)
                P = {kk: Pall[j * len(qi):(j + 1) * len(qi)] for j, kk in enumerate(keys)}
                ok['alpha'][str(f)] = al; ok['runs'][str(f)] = [r + 1 for r in runs_k]
                Yq = Y[te_i]; rq = run_all[te_i]; tq = np.array([t for _, t in qs])[qi]
                def unit(z): z = np.asarray(z, np.float64); return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
                for rr in np.unique(rq):
                    m_ = rq == rr; C = torch.from_numpy(unit(Yq[m_]).T.copy()); tt = tq[m_]; cell = {'n': int(m_.sum())}
                    S0 = torch.from_numpy(unit(P['d0'][m_])) @ C; cell['d0_fixed'] = mrp_excl(S0, tt, excl)
                    SL = torch.stack([torch.from_numpy(unit(P[l][m_])) @ C for l in lag_list]); cell['j250_fixed'] = mrp_excl(SL[lag_list.index(0.0)], tt, excl)
                    for rule in rules: cell[f'j250_{rule}'] = mrp_excl(aggregate(SL, rule), tt, excl)
                    ok['cells'][str(int(rr) + 1)] = cell
        print(f'sub-{s:03d} done ({time.time() - t0:.0f}s)' + (f' smoke cells {list(out["k2"]["cells"])}' if a.smoke else ''), flush=True)
    p = os.path.join(ROOT, A['de']['out'].replace('.json', '_smoke.json') if a.smoke else A['de']['out']); os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), indent=1); print('wrote', p, f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
