"""D-128 verdict (docs/08 §7; rules in configs/nc_verdict.yaml, registered before any evaluation score). Run once.

usage: env/bin/python scripts/nc_verdict.py configs/nc_verdict.yaml [--check]

--check only lists which registered runs are missing (no score is read). Every metric comes from the run folders'
preds.npz (test predictions of all folds; probabilities over the K classes, the null column of REVE-NC / EEGNet-NC is
ignored). Subject bootstrap: per run, one confusion matrix per test subject; a bootstrap draw is a multinomial count per
subject, applied to every run of the same task (and SB benchmark), so differences between models and conditions are
computed on the same resample. Metrics averaged over seeds use the same draw for every seed.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, glob, argparse
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from nc_grid import runs as grid_runs, folder


def load(rd, r):
    f = sorted(glob.glob(os.path.join(rd, '*' + folder(r), 'preds.npz')))
    if not f: return None
    z = np.load(f[-1]); m = json.load(open(os.path.join(os.path.dirname(f[-1]), 'metrics.json')))
    K = int(m['info']['n_classes']); y, p, s = z['y'].astype(int), z['probs'][:, :K].argmax(1), z['subject'].astype(int)
    S = int(s.max()) + 1; C = np.zeros((S, K, K))
    np.add.at(C, (s, y, p), 1)
    return dict(C=C, run=os.path.basename(os.path.dirname(f[-1])), K=K)


def metrics(C):
    """C [..., K, K] (rows: true) -> kappa, balanced accuracy (classes present in y)."""
    N = C.sum((-1, -2)); po = np.trace(C, axis1=-2, axis2=-1) / N
    pe = (C.sum(-1) * C.sum(-2)).sum(-1) / N ** 2; kap = (po - pe) / (1 - pe)
    rows = C.sum(-1); rec = np.diagonal(C, axis1=-2, axis2=-1) / np.where(rows > 0, rows, 1)
    bal = (rec * (rows > 0)).sum(-1) / (rows > 0).sum(-1)
    return kap, bal


class Boot:
    def __init__(self, n_subjects, B, seed):
        rng = np.random.default_rng(seed); S = n_subjects
        self.W = np.vstack([np.ones(S), rng.multinomial(S, np.full(S, 1 / S), size=B)])   # row 0 = the observed data

    def __call__(self, run):
        C = run['C']; W = self.W[:, :len(C)]
        return metrics(np.einsum('bs,skl->bkl', W, C))                                      # each [B+1]


def lower(x, alpha): return float(np.quantile(x[1:], alpha))
def interval(x, level): a = (1 - level) / 2; return [float(np.quantile(x[1:], a)), float(np.quantile(x[1:], 1 - a))]
def p_le(x, v): return float((1 + np.sum(x[1:] <= v)) / len(x))          # bootstrap one-sided p for H0: stat <= v


def holm(ps, alpha):
    o = np.argsort(ps); ok = [False] * len(ps)
    for k, i in enumerate(o):
        if ps[i] <= alpha / (len(ps) - k): ok[i] = True
        else: break
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    v = yaml.safe_load(open(os.path.join(ROOT, a.config))); g = yaml.safe_load(open(os.path.join(ROOT, v['grid'])))
    rd = os.path.join(ROOT, v['runs_dir']); R = grid_runs(g)
    missing = [folder(r)[1:] for r in R if not glob.glob(os.path.join(rd, '*' + folder(r), 'preds.npz'))]
    if a.check:
        print(json.dumps(dict(registered=len(R), missing=len(missing), first_missing=missing[:20]), indent=1)); return
    out_path = os.path.join(ROOT, v['out']); assert not os.path.exists(out_path), f'{out_path} exists: the verdict is run once'
    B, al = v['n_boot'], v['alpha']; cache = {}

    def get(task, arm, seed, sb=None, shuffle=False):
        key = (task, sb, arm, shuffle, seed)
        if key not in cache: cache[key] = load(rd, dict(task=task, sb=sb, arm=arm, shuffle=shuffle, seed=seed))
        return cache[key]

    boots = {}
    def boot_for(task, run):
        if task not in boots: boots[task] = Boot(len(run['C']), B, [v['boot_seed'], sum(map(ord, task))])
        return boots[task]

    def seed_mean(task, arm, seeds, sb=None, shuffle=False):
        rs = [get(task, arm, s, sb, shuffle) for s in seeds]
        if any(r is None for r in rs): return None
        ks, bs = zip(*[boot_for(task, r)(r) for r in rs])   # SB conditions share the PhysioNet-MI subjects
        return dict(kappa=np.mean(ks, 0), bal_acc=np.mean(bs, 0), runs=[r['run'] for r in rs])

    main_s, rep_s = g['seeds']['main'], g['seeds']['report']; res = dict(missing_runs=missing, n_boot=B)

    # ---------------- G0: label-shuffle runs
    g0 = {}
    for r in R:
        if not r['shuffle']: continue
        m = seed_mean(r['task'], r['arm'], [r['seed']], r['sb'], True)
        if m is None: continue
        lo, hi = interval(m['kappa'], v['g0']['level'])
        g0[f"{r['task']}|{r['sb'] or ''}|{r['arm']}"] = dict(kappa=float(m['kappa'][0]), ci99=[lo, hi], run=m['runs'][0],
                                                          passed=bool(lo <= 0 <= hi or hi < v['g0']['upper']))
    res['G0'] = g0
    def g0_ok(task, arm, sb):                    # sb: True = any SB condition, None = the task without SB
        sel = [x for k, x in g0.items() if k.split('|')[0] == task and k.split('|')[2] == arm and (bool(k.split('|')[1]) == bool(sb))]
        return bool(sel) and all(x['passed'] for x in sel)

    # ---------------- H1: shortcut reliance R on the SB benchmark
    h = v['h1']; T = 'physionet_mi'
    def R_of(arm, seeds, shortcut='c123'):
        parts = []
        for sig in h['signals']:
            cl, sc = seed_mean(T, arm, seeds, f'{sig}_clean'), seed_mean(T, arm, seeds, f'{sig}_{shortcut}')
            if cl is None or sc is None: return None
            parts.append((cl['kappa'], cl['kappa'] - sc['kappa']))
        return dict(kappa_clean=np.mean([p[0] for p in parts], 0), R=np.mean([p[1] for p in parts], 0),
                    kappa_clean_by_signal={sig: p[0] for sig, p in zip(h['signals'], parts)})
    nc = R_of(h['nc'], main_s); H1 = dict(R={}, tests={}, validity={})
    if nc is not None:
        H1['R'][h['nc']] = dict(R=float(nc['R'][0]), kappa_clean=float(nc['kappa_clean'][0]))
        ps, names = [], []
        for o in h['others']:
            oo = R_of(o, main_s)
            if oo is None: continue
            d_ = oo['R'] - nc['R']                                                         # > 0: REVE-NC relies less
            H1['R'][o] = dict(R=float(oo['R'][0]), kappa_clean=float(oo['kappa_clean'][0]))
            H1['tests'][o] = dict(diff=float(d_[0]), lower95=lower(d_, al), p=p_le(d_, 0.0)); ps.append(H1['tests'][o]['p']); names.append(o)
            H1['validity'][o] = dict(b_kappa_clean_gap=float(nc['kappa_clean'][0] - oo['kappa_clean'][0]),
                                     b_ok=bool(nc['kappa_clean'][0] >= oo['kappa_clean'][0] - h['validity_margin']))
        hol = holm(ps, al)
        for n, ok in zip(names, hol): H1['tests'][n]['holm_reject'] = bool(ok)
        va = {}
        for arm in [h['nc']] + h['others']:
            rr = R_of(arm, main_s)
            for sig, kb in rr['kappa_clean_by_signal'].items(): va[f'{arm}|{sig}'] = dict(kappa=float(kb[0]), lower95=lower(kb, al), ok=bool(lower(kb, al) > 0))
        H1['validity']['a_signal_learnable'] = va
        a_ok = all(x['ok'] for x in va.values()); b_ok = all(H1['validity'][o]['b_ok'] for o in names)
        g_ok = all(g0_ok(T, arm, True) for arm in [h['nc']] + h['others'])
        H1['verdict'] = ('판정 불가' if not (a_ok and b_ok and g_ok and len(names) == len(h['others']))
                         else '통과' if all(hol) else '불통과')
        # report-only
        rep = {}
        for arm in h['report_arms']:
            rr = R_of(arm, rep_s)
            if rr is not None: rep[arm] = dict(R=float(rr['R'][0]), interval95=interval(rr['R'], 0.95), kappa_clean=float(rr['kappa_clean'][0]))
        H1['report_arms_seed0'] = rep
        per = {}
        for arm in [h['nc']] + h['others']:
            for sc in h['report_shortcuts']:
                rr = R_of(arm, rep_s, sc)
                if rr is not None: per[f'{arm}|{sc}'] = dict(R=float(rr['R'][0]), interval95=interval(rr['R'], 0.95))
        H1['per_shortcut_seed0'] = per
        e_nc = R_of('eegnet_nc', main_s)
        if e_nc is not None:
            d_ = e_nc['R'] - nc['R']; H1['nc_minus_eegnet_nc'] = dict(diff=float(d_[0]), interval95=interval(d_, 0.95))
    res['H1'] = H1

    # ---------------- H2: non-inferiority to EEGNet (balanced accuracy)
    h = v['h2']; H2 = dict(tests={}, report={})
    ps, names = [], []
    for ds in h['datasets']:
        a_, b_ = seed_mean(ds, h['nc'], main_s), seed_mean(ds, h['ref'], main_s)
        if a_ is None or b_ is None: continue
        d_ = a_['bal_acc'] - b_['bal_acc']
        H2['tests'][ds] = dict(nc=float(a_['bal_acc'][0]), ref=float(b_['bal_acc'][0]), diff=float(d_[0]), lower95=lower(d_, al),
                               p=p_le(d_, h['margin'])); ps.append(H2['tests'][ds]['p']); names.append(ds)
        c_ = seed_mean(ds, h['report_ref'], main_s)
        if c_ is not None:
            e_ = a_['bal_acc'] - c_['bal_acc']
            H2['report'][f'{ds}|nc_minus_{h["report_ref"]}'] = dict(diff=float(e_[0]), lower95=lower(e_, al), noninferior=bool(lower(e_, al) >= h['margin']))
    for n, ok in zip(names, holm(ps, al)):
        H2['tests'][n]['holm_noninferior'] = bool(ok)
        H2['tests'][n]['g0_ok'] = bool(g0_ok(n, h['nc'], None) and g0_ok(n, h['ref'], None))
    for ds in h['interval_only']:
        a_, b_ = seed_mean(ds, h['nc'], main_s), seed_mean(ds, h['ref'], main_s)
        if a_ is not None and b_ is not None:
            d_ = a_['bal_acc'] - b_['bal_acc']; H2['report'][f'{ds}|nc_minus_{h["ref"]}'] = dict(diff=float(d_[0]), interval95=interval(d_, 0.95))
    H2['passed_both'] = bool(len(names) == len(h['datasets']) and all(H2['tests'][n]['holm_noninferior'] and H2['tests'][n]['g0_ok'] for n in names))
    res['H2'] = H2
    res['headline'] = bool(H1.get('verdict') == '통과' and H2['passed_both'])

    # ---------------- H3: descriptive
    h = v['h3']; H3 = {}
    for task in h['tasks']:
        row = {}
        for arm in ['eegnet', 'reve_official', 'reve_nc', 'reve_nc_m2']:
            seeds = main_s if arm != 'reve_nc_m2' and task != 'ugr_mode' else rep_s
            m = seed_mean(task, arm, seeds)
            if m is None: continue
            row[arm] = dict(kappa=float(m['kappa'][0]), kappa_ci95=interval(m['kappa'], h['level']), bal_acc=float(m['bal_acc'][0]))
        if task in h['pre']:
            for arm in ['eegnet', 'reve_official', 'reve_nc']:
                m = seed_mean(h['pre'][task], arm, rep_s)
                if m is not None:
                    row[f'pre|{arm}'] = dict(kappa=float(m['kappa'][0]), lower95=lower(m['kappa'], al), leak_flag=bool(lower(m['kappa'], al) > 0))
        if task in h['refs']:
            m = seed_mean(h['refs'][task], 'eegnet', rep_s)
            if m is not None: row['refs|eegnet'] = dict(kappa=float(m['kappa'][0]), kappa_ci95=interval(m['kappa'], h['level']))
        row['leak_flag_any_pre'] = any(x.get('leak_flag', False) for k, x in row.items() if isinstance(x, dict) and k.startswith('pre|'))
        H3[task] = row
    res['H3'] = H3
    os.makedirs(os.path.dirname(out_path), exist_ok=True); json.dump(res, open(out_path, 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(dict(H1=H1.get('verdict'), H2=H2['passed_both'], headline=res['headline'], missing=len(missing)), ensure_ascii=False))


if __name__ == '__main__':
    main()
