"""D-113 arms a and b (Gao et al. 2026 perception -> imagery, no alpha): features, decoders, meters, null, bootstrap, wording.

Features (D-113 5.1): band-pass of the cached continuous signal (theta 4-8, alpha 8-13, beta 13-30, gamma 30-45 Hz; MNE FIR,
zero phase), log variance per channel in each window (configs/fv_gao.yaml windows; the observation-anchored windows are moved
by the ERP-check shift of results/gao_imagery/erp_check.json). Decoder: training-side standardisation + shrinkage LDA (lsqr,
Ledoit-Wolf), one per category (task).
Per arm, subject and category (splits of ese.gao.design with the D-113 4.3 block exclusion):
  main      perception window (main features) -> imagery window; null: training labels permuted within training blocks
            (n, seed), refit; the same permuted fits also score the content control.
  control   perception -> perception (content positive control R-b2; excess 90% lower > 0 required).
  meters    late (late-perception window -> imagery), ADJ, block position, eye, muscle (gamma, all channels), temporal.
  report    rest window, imagery 1-s windows, occipital alpha, imagery -> imagery (arm b), no-exclusion split (main, ADJ, block
            position), sub-13 ses-02 OVI read as chair (main).
Metrics: balanced accuracy and MRP (true item's rank among the K items, 1 - rank / (K - 1); ties count half). Excess = metric -
null mean. Combined = per-subject mean over the categories the subject has, then subjects resampled (n, seed).
Wording (D-113 5.4, implementation note gao_ab_implementation_kr.md 4): control fails -> undecidable (positive control);
condition 1 lower <= 0 -> not detected; condition 2 or 3 fails -> undecidable (meters); all pass -> imagery transfer observed.
If the pipeline control R-b1 failed: 'not detected' becomes undecidable (R-b1 failed), and an observed transfer carries an R-b1 caveat.
--smoke: Gaussian features instead of EEG (code test; 20 null fits, first 4 keys; output in the cache folder, never reported).
--memcheck: real features, first key of the arm only, 20 null fits; prints the peak RSS only (no score is printed or saved).
usage (gpuq): env/bin/python -I scripts/fv_gao_decode.py configs/fv_gao.yaml --arm {b,a} [--smoke | --memcheck]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import argparse, glob, json, resource, sys, time, warnings
import numpy as np, yaml

warnings.filterwarnings('ignore')
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

mne.set_log_level('ERROR')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); P = lambda p: os.path.join(ROOT, p)
from ese.gao.design import load_trials, splits, meter_adj, meter_bpos


def features(c, shift):
    """{(run name, pos): array [n_windows, n_bands, 32]}, window and band names; cached as npz"""
    cache = P(c['cache']); fs = c['preprocess']['sfreq']; out = os.path.join(cache, f'features_shift{shift:+.3f}.npz')
    W = {k: (v[0] + shift, v[1] + shift) for k, v in c['windows']['obs_anchored'].items()}; W.update({k: tuple(v) for k, v in c['windows']['m_anchored'].items()})
    wn, bn = list(W), list(c['bands'])
    if os.path.exists(out):
        z = np.load(out, allow_pickle=False); FA = z['F']                # read the npz member once (each z['F'] re-reads it)
        return {(str(r), int(p)): FA[i] for i, (r, p) in enumerate(zip(z['run'], z['pos']))}, wn, bn, list(z['ch'])
    F, R, Q, ch0 = [], [], [], None
    for js in sorted(glob.glob(os.path.join(cache, 'sub-*.json'))):
        meta = json.load(open(js)); X = np.load(js[:-5] + '.npy').astype(np.float64); name = os.path.basename(js)[:-5]
        ch0 = ch0 or meta['ch_names']; assert meta['ch_names'] == ch0
        B = [mne.filter.filter_data(X, fs, lo, hi, method='fir', phase='zero', verbose=False) for lo, hi in c['bands'].values()]
        for p, m in enumerate(meta['m']):
            f = np.empty((len(wn), len(bn), X.shape[0]))
            for i, w in enumerate(wn):
                a, b = m + int(round(W[w][0] * fs)), m + int(round(W[w][1] * fs)); assert 0 <= a < b <= X.shape[1], (name, p, w)
                for j in range(len(bn)): f[i, j] = np.log(B[j][:, a:b].var(1))
            F.append(f); R.append(name); Q.append(p)
        print('features', name, flush=True)
    np.savez(out, F=np.array(F, np.float32), run=np.array(R), pos=np.array(Q), ch=np.array(ch0))
    return features(c, shift)


def lda(c): return LinearDiscriminantAnalysis(solver=c['decoder']['solver'], shrinkage=c['decoder']['shrinkage'])


class Fit:
    def __init__(self, c, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12; self.m = lda(c).fit((X - self.mu) / self.sd, y)

    def scores(self, X): return self.m.decision_function((X - self.mu) / self.sd)


def metrics(S, y, K):
    if S.ndim == 1: S = np.c_[-S, S]                                     # binary decision_function (never for K >= 3)
    pred = S.argmax(1); t = S[np.arange(len(y)), y][:, None]
    rank = (S > t).sum(1) + 0.5 * ((S == t).sum(1) - 1)
    return float(balanced_accuracy_score(y, pred)), float(np.mean(1 - rank / (K - 1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--arm', choices=['a', 'b'], required=True); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--memcheck', action='store_true')
    a = ap.parse_args(); c = yaml.safe_load(open(P(a.config))); t0 = time.time()
    T, runs = load_trials(P(c['bids']), c); Tr, runs_r = load_trials(P(c['bids']), c, relabel=True)
    if a.smoke:                                                          # code test only: Gaussian features, no EEG value is read
        shift = 0.0; wn = list(c['windows']['obs_anchored']) + list(c['windows']['m_anchored']); bn = list(c['bands'])
        ch = [l.split('\t')[0] for l in open(os.path.join(P(c['bids']), 'electrodes.tsv')).read().splitlines()[1:] if l]; g = np.random.default_rng(0)
        FT = g.standard_normal((len(T['label']), len(wn), len(bn), len(ch))); FTr = g.standard_normal((len(Tr['label']), len(wn), len(bn), len(ch)))
    else:
        erp = json.load(open(P(os.path.join(c['out'], 'erp_check.json')))); shift = float(erp['shift_s'])
        Fd, wn, bn, ch = features(c, shift)
        name = lambda T_, runs_, i: '%s_%s_%s' % runs_[T_['run'][i]]['key']
        FT = np.stack([Fd[(name(T, runs, i), int(T['pos'][i]))] for i in range(len(T['label']))])
        FTr = np.stack([Fd[(name(Tr, runs_r, i), int(Tr['pos'][i]))] for i in range(len(Tr['label']))])

    def fsel(F, setname, window):
        s = c['features'][setname]; bi = [bn.index(b) for b in s['bands']]; ci = list(range(len(ch))) if s['channels'] == 'all' else [ch.index(x) for x in s['channels']]
        return F[:, wn.index(window)][:, bi][:, :, ci].reshape(len(F), -1)

    rng = np.random.default_rng(c['perm_null']['seed']); n_null = 20 if a.smoke or a.memcheck else c['perm_null']['n']
    S = splits(T, c, exclude=True); S0 = splits(T, c, exclude=False); Sr = splits(Tr, c, exclude=True)
    keys = sorted(k for k in S if k[0] == a.arm)
    if a.smoke: keys = keys[:4]
    if a.memcheck: keys = keys[:1]
    lab = T['label']; rows = []
    for k in keys:
        _, s, task = k; tc = c['tasks'][task]; K = len(tc['items']); tr, te = S[k]; ytr, yte = lab[tr], lab[te]
        r = dict(arm=a.arm, subject=s, task=task, category=tc['category'], K=K, n_train=len(tr), n_test=len(te), chance=1 / K)
        Xp = fsel(FT, 'main', 'perception'); fit = Fit(c, Xp[tr], ytr)
        r['main'] = metrics(fit.scores(fsel(FT, 'main', 'imagery')[te]), yte, K)
        r['control'] = metrics(fit.scores(Xp[te]), yte, K)
        # null: permutation within training blocks, one refit scores main and control
        grp = {}
        for j, i in enumerate(tr): grp.setdefault((T['run'][i], T['block'][i]), []).append(j)
        nm, nc = [], []; Xi = fsel(FT, 'main', 'imagery')[te]
        for _ in range(n_null):
            yp = ytr.copy()
            for g in grp.values(): yp[g] = ytr[rng.permutation(g)]
            f = Fit(c, Xp[tr], yp); nm.append(metrics(f.scores(Xi), yte, K)); nc.append(metrics(f.scores(Xp[te]), yte, K))
        r['null_main_mean'] = np.mean(nm, 0).tolist(); r['null_control_mean'] = np.mean(nc, 0).tolist()
        # meters
        r['late'] = metrics(Fit(c, fsel(FT, 'main', 'late')[tr], ytr).scores(Xi), yte, K)
        for m in ('eye', 'muscle', 'temporal'):
            r[m] = metrics(Fit(c, fsel(FT, m, 'perception')[tr], ytr).scores(fsel(FT, m, 'imagery')[te]), yte, K)
        r['adj'] = metrics(meter_adj(T, tr, te, K)[1], yte, K); r['bpos'] = metrics(meter_bpos(T, tr, te, K, tc['block'])[1], yte, K)
        # report only
        for w in ('rest', 'im1', 'im2', 'im3'): r[w] = metrics(fit.scores(fsel(FT, 'main', w)[te]), yte, K)
        r['alpha_occ'] = metrics(Fit(c, fsel(FT, 'alpha_occ', 'perception')[tr], ytr).scores(fsel(FT, 'alpha_occ', 'imagery')[te]), yte, K)
        if a.arm == 'b':
            Xm = fsel(FT, 'main', 'imagery'); r['imag_to_imag'] = metrics(Fit(c, Xm[tr], ytr).scores(Xm[te]), yte, K)
        if k in S0:
            tr0, te0 = S0[k]; y0 = lab[te0]; f0 = Fit(c, Xp[tr0], lab[tr0])
            r['noexcl'] = dict(n_train=len(tr0), n_test=len(te0), main=metrics(f0.scores(fsel(FT, 'main', 'imagery')[te0]), y0, K),
                               adj=metrics(meter_adj(T, tr0, te0, K)[1], y0, K), bpos=metrics(meter_bpos(T, tr0, te0, K, tc['block'])[1], y0, K))
        if task == 'OVI' and k in Sr:
            trr, ter = Sr[k]; Xpr = fsel(FTr, 'main', 'perception')
            r['relabel_sub13'] = dict(n_train=len(trr), n_test=len(ter), main=metrics(Fit(c, Xpr[trr], Tr['label'][trr]).scores(fsel(FTr, 'main', 'imagery')[ter]), Tr['label'][ter], K))
        rows.append(r)
        if a.memcheck: continue
        print(k, 'main %.3f null %.3f control %.3f' % (r['main'][0], r['null_main_mean'][0], r['control'][0]), f'{time.time() - t0:.0f}s', flush=True)
    for k in sorted(set(Sr) - set(S)):
        if k[0] == a.arm and k[2] == 'OVI':                              # sub-13 enters arm b's object category only in the relabel sensitivity
            trr, ter = Sr[k]; Xpr = fsel(FTr, 'main', 'perception'); K = 4
            rows.append(dict(arm=a.arm, subject=k[1], task='OVI', category='object', K=K, sensitivity_only=True,
                             relabel_sub13=dict(n_train=len(trr), n_test=len(ter), main=metrics(Fit(c, Xpr[trr], Tr['label'][trr]).scores(fsel(FTr, 'main', 'imagery')[ter]), Tr['label'][ter], K))))
    rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20   # Linux: KiB -> GiB
    if a.memcheck:
        print(json.dumps(dict(memcheck=True, arm=a.arm, key=list(keys[0]), n_train=rows[0]['n_train'], n_test=rows[0]['n_test'], n_null=n_null,
                              peak_rss_gb=round(rss(), 3), seconds=round(time.time() - t0, 1)))); return
    # aggregation
    R = [r for r in rows if not r.get('sensitivity_only')]; subs = sorted({r['subject'] for r in R})
    def per_sub(fn):
        return np.array([np.mean([fn(r) for r in R if r['subject'] == s]) for s in subs])
    q = {'main_excess': per_sub(lambda r: r['main'][0] - r['null_main_mean'][0]), 'control_excess': per_sub(lambda r: r['control'][0] - r['null_control_mean'][0]),
         'main_mrp_excess': per_sub(lambda r: r['main'][1] - r['null_main_mean'][1]), 'main_acc': per_sub(lambda r: r['main'][0]), 'chance': per_sub(lambda r: r['chance'])}
    meters = ('late', 'bpos', 'eye', 'muscle', 'temporal')
    for m in meters + ('adj', 'rest', 'im1', 'im2', 'im3', 'alpha_occ') + (('imag_to_imag',) if a.arm == 'b' else ()):
        q[m] = per_sub(lambda r, m=m: r[m][0])
    br = np.random.default_rng(c['boot']['seed']); B = [br.integers(0, len(subs), len(subs)) for _ in range(c['boot']['n'])]
    pct = lambda v, p: float(np.percentile(v, p))
    boot = {k: np.array([v[b].mean() for b in B]) for k, v in q.items()}
    d2 = np.array([q['main_acc'][b].mean() - max(q[m][b].mean() for m in meters) for b in B])
    adj_ex = np.array([(q['adj'] - q['chance'])[b].mean() for b in B]); bpos_ex = np.array([(q['bpos'] - q['chance'])[b].mean() for b in B])
    summ = {k: dict(mean=float(v.mean()), p05=pct(boot[k], 5), p95=pct(boot[k], 95)) for k, v in q.items()}
    max_meter = max(meters, key=lambda m: q[m].mean())
    cond = dict(c1_main_excess_lower90=pct(boot['main_excess'], 5), c1=pct(boot['main_excess'], 5) > 0,
                c2_main_minus_max_meter_lower90=pct(d2, 5), c2_max_meter_point=max_meter, c2=pct(d2, 5) > 0,
                c3_adj_minus_chance_upper95=pct(adj_ex, 95), c3_bpos_minus_chance_upper95=pct(bpos_ex, 95),
                c3=pct(adj_ex, 95) < 0.02 and pct(bpos_ex, 95) < 0.02,
                control_excess_lower90=pct(boot['control_excess'], 5), control=pct(boot['control_excess'], 5) > 0)
    if not cond['control']: word = 'undecidable data (content positive control failed)'
    elif not cond['c1']: word = 'not detected'
    elif not (cond['c2'] and cond['c3']): word = 'undecidable data (meters)'
    else: word = 'imagery transfer observed (exploratory, no alpha)'
    rb1 = None if a.smoke else bool(erp['pipeline_control']['passed']); cond['pipeline_control_rb1'] = rb1
    if rb1 is False:                                                     # coordination session 2026-10-17 (implementation note 4)
        if word == 'not detected': word = 'undecidable data (pipeline control R-b1 failed)'
        elif word.startswith('imagery transfer observed'): word += ' [caveat: pipeline control R-b1 failed]'
    res = dict(arm=a.arm, smoke=a.smoke, shift_s=shift, n_null=n_null, n_boot=c['boot']['n'], subjects=subs, summary=summ, conditions=cond, wording=word,
               rows=rows, seconds=time.time() - t0, peak_rss_gb=rss())
    out = os.path.join(P(c['cache']), f'arm_{a.arm}_smoke.json') if a.smoke else P(os.path.join(c['out'], f'arm_{a.arm}.json')); json.dump(res, open(out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(json.dumps(dict(conditions=cond, wording=word), indent=1, default=str)); print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
