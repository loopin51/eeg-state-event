"""D-127 (family F-XV follow-up): one-band-at-a-time decoding units on epochs with a wider low-pass (95 Hz, 256 / 250 Hz).

usage: python scripts/imband_verdict.py configs/imband.yaml --mode smoke|info0|verdict [--data TOL,...] [--subjects N]

Units (D-127 section 2): for each band b in bands.list, DE_b = channel log-variance of the band-passed window and RG_b = log-Euclidean
Ledoit-Wolf covariance of the same signal; DE_all = all band log-variances concatenated. Bands whose upper edge is above
bands.notch_above get 50 / 60 Hz IIR notches first. Reference units outside the max-T: REF_D2 (bands up to 40 Hz concatenated, the
D2 of D-126) and REF_D3 (1-40 Hz covariance, the D3 of D-126). Classifier, kappa, permutations, max-T and bootstrap are those of
scripts/imnoise_verdict.py (imported); IM36 / IM36A use the fold-balanced remaps (perm_mode_by).
Modes
  smoke    synthetic data through every path
  info0    section 6: random labels from each data set's permutation generator, their own permutations; writes <out_dir>/info0.json
  verdict  real labels: Q1 (max-T over the 13 units, Holm over the data sets), Q2 (gain over the better reference unit), Q3 (gamma
           units, A2 and L_pre of the same unit), meters L_pre / A1 / A2 on every unit; data sets that failed info0 are left out
"""
import os
os.environ.setdefault('ESE_NUM_THREADS', '1')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ['ESE_NUM_THREADS'])
import sys, glob, json, time, zlib, argparse, hashlib
import numpy as np, yaml
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imnoise_verdict as V                                  # noqa: E402

P = V.P
C = None


def setc(cfg):
    global C
    C = cfg; V.C = cfg


# -------------------------------------------------------------------------------------------------------------- units
def band_name(b):
    return f'{b[0]:g}-{b[1]:g}'


def band_filtered(X, t, fs, win):
    """band-passed copies cropped to the window, one per bands.list entry, and the 1-40 Hz copy for REF_D3 (filters on the whole epoch)"""
    bc = C['bands']; w = V.wsl(t, win); order = C['decoders']['butter_order']; out = []; Xn = None
    def bp(x, lo, hi):
        sos = butter(order, [lo, min(hi, 0.45 * fs)], btype='band', fs=fs, output='sos')
        return sosfiltfilt(sos, x, axis=-1)[..., w].astype(np.float32)
    for lo, hi in bc['list']:
        x = X
        if hi > bc['notch_above']:
            if Xn is None:
                Xn = X.astype(np.float64)
                for f0 in bc['notch_hz']:
                    if f0 < 0.45 * fs:
                        bn, an = iirnotch(f0, bc['notch_q'], fs=fs); Xn = filtfilt(bn, an, Xn, axis=-1)
            x = Xn
        out.append(bp(x, lo, hi))
    del Xn
    lo, hi = C['decoders']['rg_band']
    return out, bp(X, lo, hi)


def band_units(X, t, fs, win):
    """({13 units}, {2 reference units}) -> (features [n, d], pca dims)"""
    F, ref = band_filtered(X, t, fs, win); q = C['decoders']['pca']; U = {}
    de = [np.log(f.var(-1) + 1e-12) for f in F]
    for b, d_, f in zip(C['bands']['list'], de, F):
        U['DE' + band_name(b)] = (d_, q); U['RG' + band_name(b)] = (V.feat_rg(f), q)
    U['DEall'] = (np.concatenate(de, 1), q)
    n40 = sum(1 for b in C['bands']['list'] if b[1] <= 40)
    R = {'REF_D2': (np.concatenate(de[:n40], 1), q), 'REF_D3': (V.feat_rg(ref), q)}
    return U, R


# ------------------------------------------------------------------------------------------------------------ subject work
def subject_job(args):
    ds, f, mode, cfg = args
    setc(cfg); z = dict(np.load(f, allow_pickle=True)); sub = os.path.basename(f)[:-4]; W = C['windows'][ds]; B = C['n_perm']
    pm = C.get('perm_mode_by', {}).get(ds); crc = zlib.crc32(f'{ds}_{sub}'.encode()); seed = lambda key: [C['seeds'][key], crc]
    t0 = time.time(); res = dict(sub=sub, tests={})
    v = V.get_view(z, 'main', W['main']); spl = V.splits_of(v); n = len(v['y']); res['n'] = int(n); Kmax = int(max(v['K']))
    U, R = band_units(v['X'], v['t'], v['fs'], v['win']); res['units'] = list(U)
    tm = V.tested_mask(v, spl)
    if mode == 'info0':
        ic = C['info0']; rows = []; Yr = V.perm_labels(v, ic['reps'], seed('info0'), mode=pm)
        preps = {name: V.prepare(F, F, q, spl) for name, (F, q) in U.items()}
        for r in range(1, ic['reps'] + 1):
            vr = V.relabel(v, Yr[r].astype(int)); Ys = V.perm_labels(vr, ic['n_perm'], [C['seeds']['info0'], r, crc], mode=pm)
            rows.append({name: V.kappa_of(V.bacc(V.predict(pr, Ys, Kmax, n), Ys, tm, v['task'], v['K'])).tolist() for name, pr in preps.items()})
        res['info0'] = rows; res['seconds'] = round(time.time() - t0, 1); return res
    Y = V.perm_labels(v, B, seed('perm'), mode=pm)
    put = lambda name, sc: res['tests'].__setitem__(name, {u: V.kappa_of(r).tolist() for u, r in sc.items()})
    put('I', V.score_units(U, v, spl, Y)[0]); put('REF', V.score_units(R, v, spl, Y)[0]); del R
    if 'pre' in W:
        vp = V.get_view(z, 'pre', W['pre']); sp = V.splits_of(vp); Yp = V.perm_labels(vp, B, seed('perm'), mode=pm)
        Up, _ = band_units(vp['X'], vp['t'], vp['fs'], vp['win']); put('L_pre', V.score_units(Up, vp, sp, Yp)[0]); del Up
    E, A = V.artefact_split(z, v)
    Ua, _ = band_units(A, v['t'], v['fs'], v['win']); put('A1', V.score_units(Ua, v, spl, Y)[0]); del Ua
    Rr = V.resid_eeg(E, A); Ur, _ = band_units(Rr, v['t'], v['fs'], v['win']); put('A2', V.score_units(Ur, v, spl, Y)[0]); del Rr, Ur
    res['seconds'] = round(time.time() - t0, 1)
    return res


def run_dataset(ds, files, mode, cfg):
    from multiprocessing import Pool
    with Pool(cfg.get('workers_by', {}).get(ds, cfg['workers'])) as pool:
        return pool.map(subject_job, [(ds, f, mode, cfg) for f in files], chunksize=1)


# ------------------------------------------------------------------------------------------------------------ group stats
def add_padj(st):
    Tn = np.array(st['null']['T'])
    for u, x in st['units'].items(): x['p_adj'] = float((1 + (Tn >= x['z']).sum()) / (len(Tn) + 1))
    return st


def gain(subs, st_I, st_R):
    """Q2: per unit, subject bootstrap of kappa_u - kappa_ref (ref = reference unit with the larger group kappa)"""
    ref = max(st_R['units'], key=lambda u: st_R['units'][u]['kappa']); units_ = sorted(st_I['units']); S = len(subs)
    Kr = np.array([s['tests']['REF'][ref][0] for s in subs]); rng = np.random.default_rng(C['seeds']['boot']); bs = rng.integers(0, S, (C['n_boot'], S))
    q = 100 * C['alpha'] / len(units_); out = {}
    for u in units_:
        d = np.array([s['tests']['I'][u][0] for s in subs]) - Kr; bd = d[bs].mean(1)
        out[u] = dict(diff=float(d.mean()), lower_bonf=float(np.percentile(bd, q)), ci95=[float(x) for x in np.percentile(bd, [2.5, 97.5])])
    return ref, out


def statements(ds, r):
    a = C['alpha']; q1 = r['Q1_holm'] <= a; I = r['tests']['I']['units']
    best_gain = max(r['gain'].items(), key=lambda kv: kv[1]['lower_bonf'])
    q2 = '대역 분리로 정보가 커짐' if (q1 and best_gain[1]['lower_bonf'] > 0) else '대역 분리 이득 없음'
    gam = [u for u in I if u.endswith('-' + f"{C['bands']['list'][-1][1]:g}") and u != 'DEall' and u.startswith(('DE', 'RG'))]
    hits = [u for u in gam if I[u]['p_adj'] <= a] if q1 else []
    if not hits:
        q3 = '감마 정보 없음'
    else:
        msgs = []
        for u in hits:
            fails = []
            if r['tests']['A2']['units'][u]['p_unit'] > a: fails.append('근전도(A2)')
            lp = r['tests'].get('L_pre', {}).get('units', {}).get(u)
            if lp is not None and lp['p_unit'] <= a and lp['kappa'] >= C['meter_max']: fails.append('누설(L_pre)')
            msgs.append(f'{u}: ' + ('근전도·누설로 설명되지 않는 감마 정보' if not fails else '감마 정보, 경로를 배제하지 못함: ' + ', '.join(fails)))
        q3 = '; '.join(msgs)
    return dict(Q1='정보 있음' if q1 else '정보 없음(기각 못함)', Q2=q2, Q2_best=dict(unit=best_gain[0], **best_gain[1]), Q3=q3, gamma_units=gam)


# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--mode', required=True); ap.add_argument('--data', default='')
    ap.add_argument('--subjects', type=int, default=0); a = ap.parse_args(); cfg = yaml.safe_load(open(P(a.config))); setc(cfg)
    out = P(C['out_dir']); os.makedirs(out, exist_ok=True); dsl = a.data.split(',') if a.data else C['datasets']; t0 = time.time()
    if a.mode == 'smoke':
        sm = os.path.join(out, 'smoke'); os.makedirs(sm, exist_ok=True); C.update(n_perm=60, n_boot=100, workers=2); C['info0'].update(reps=3, n_perm=30)
        C['windows']['SYN'] = dict(main=[1.0, 3.0], pre=[-0.5, 0.0]); rep = {}
        for info in (0.0, 0.3):
            files = []
            for i, d in enumerate(V.synth(4, info, np.random.default_rng(1 + int(info * 10)))):
                f = os.path.join(sm, f'syn{info}_{i}.npz'); np.savez(f, **d); files.append(f)
            subs = run_dataset('SYN', files, 'verdict', C)
            st = {k: add_padj(V.maxT([s['tests'][k] for s in subs])) for k in subs[0]['tests']}
            ref, g = gain(subs, st['I'], st['REF'])
            rep[str(info)] = dict({k: (round(v['p_maxT'], 4), v['best_unit'], round(v['best_kappa'], 4)) for k, v in st.items()}, ref=ref,
                                  gain_best=max(g.items(), key=lambda kv: kv[1]['lower_bonf']), seconds=[s['seconds'] for s in subs],
                                  info0_rows=len(run_dataset('SYN', files, 'info0', C)[0]['info0']),
                                  statements=statements('SYN', dict(tests=st, gain=g, Q1_holm=st['I']['p_maxT'])))
            print(json.dumps(rep[str(info)], ensure_ascii=False), flush=True)
        json.dump(rep, open(os.path.join(sm, 'smoke.json'), 'w'), indent=1, ensure_ascii=False); return
    if a.mode == 'info0':
        from scipy.stats import binomtest, kstest
        res = {}
        for ds in dsl:
            files = sorted(glob.glob(os.path.join(P(C['cache_dir']), ds, 'sub-*.npz')))[:a.subjects or None]
            subs = run_dataset(ds, files, 'info0', C); ps = [V.maxT([s['info0'][r] for s in subs])['p_maxT'] for r in range(len(subs[0]['info0']))]
            k = int(np.sum(np.array(ps) <= C['alpha']))
            res[ds] = dict(p=ps, rate=k / len(ps), binom_p=float(binomtest(k, len(ps), C['alpha'], alternative='greater').pvalue),
                           ks_uniform_p=float(kstest(ps, 'uniform').pvalue), n_subjects=len(subs))
            print(ds, 'info0 rate', res[ds]['rate'], 'binom p', round(res[ds]['binom_p'], 4), f'{time.time() - t0:.0f}s', flush=True)
            json.dump(res, open(os.path.join(out, 'info0_partial.json'), 'w'))
        allp = np.concatenate([v['p'] for v in res.values()]); pooled = float(np.mean(allp <= C['alpha'])); lo, hi = C['info0']['pooled_range']
        summ = dict(datasets=res, pooled_rate=pooled, pooled_ok=bool(lo <= pooled <= hi),
                    passed={d: bool(v['binom_p'] > C['info0']['binom_min_p']) for d, v in res.items()})
        json.dump(summ, open(os.path.join(out, 'info0.json'), 'w'), indent=1)
        print(json.dumps({d: v['rate'] for d, v in res.items()}), 'pooled', pooled, summ['pooled_ok'], summ['passed']); return
    # verdict: gate by info0 (section 6)
    I0 = json.load(open(os.path.join(out, 'info0.json')))
    assert I0['pooled_ok'], 'info-0 pooled rate outside the registered range: verdict stopped (D-127 section 6)'
    excluded = [d for d in dsl if not I0['passed'].get(d, False)]; dsl = [d for d in dsl if d not in excluded]
    res = {}
    for ds in dsl:
        files = sorted(glob.glob(os.path.join(P(C['cache_dir']), ds, 'sub-*.npz')))[:a.subjects or None]
        subs = run_dataset(ds, files, 'verdict', C); print(ds, 'done', f'{time.time() - t0:.0f}s', [s['seconds'] for s in subs], flush=True)
        st = {k: add_padj(V.maxT([s['tests'][k] for s in subs])) for k in subs[0]['tests']}
        ref, g = gain(subs, st['I'], st['REF'])
        res[ds] = dict(tests=st, ref_unit=ref, gain=g, subjects=[s['sub'] for s in subs], n_trials=[s['n'] for s in subs], seconds=[s['seconds'] for s in subs])
        json.dump(res, open(os.path.join(out, 'verdict_partial.json'), 'w'))
    adj = V.holm([res[d]['tests']['I']['p_maxT'] for d in dsl])
    for d, pa in zip(dsl, adj):
        res[d]['Q1_holm'] = pa; res[d]['statements'] = statements(d, res[d]); I = res[d]['tests']['I']
        print(d, 'Q1 p', round(I['p_maxT'], 4), 'holm', round(pa, 4), 'best', I['best_unit'], round(I['best_kappa'], 4), I['best_ci'],
              '| ref', res[d]['ref_unit'], round(res[d]['tests']['REF']['units'][res[d]['ref_unit']]['kappa'], 4), '|', json.dumps(res[d]['statements'], ensure_ascii=False), flush=True)
    res['_meta'] = dict(config=a.config, seconds=round(time.time() - t0, 1), datasets=dsl, excluded_by_info0=excluded)
    f = os.path.join(out, 'verdict.json'); json.dump(res, open(f, 'w'), indent=1, ensure_ascii=False)
    print('sha256', hashlib.sha256(open(f, 'rb').read()).hexdigest())


if __name__ == '__main__':
    main()
