"""D-100 (a) report: pools the per-cell values of scripts/d100_tau50.py and scripts/d100_tau50_de.py.

Cell (target subject of the 17 pool, run) excess = observed 30-s-excluded MRP - mean of its 1,000 null means; pooled value =
query-weighted mean over runs of the subject mean (B1 stat). Group null per condition = the same pooling of each draw.
Retention R(|delta|) = [excess(+delta) + excess(-delta)] / [2 excess(0)]; tau50 = linear interpolation of R on the registered
grid at the first value below 0.5 ("> 1.0" if none). Interval: subject x run pigeonhole bootstrap, the same draws for every
condition (1,000, seed 20261112). Target baseline cells are runs only (no subject). Interpretation rules: D-100 (a).

usage: python scripts/d100_tau50_report.py configs/d100_tau50.yaml
"""
import os, sys, glob, json, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tau50(R, grid):
    """R: dict |delta| -> retention (R[0] = 1); returns the interpolated first crossing below 0.5 (np.inf if none)"""
    for a, b in zip(grid[:-1], grid[1:]):
        if R[b] < 0.5: return float(a + (R[a] - 0.5) / (R[a] - R[b]) * (b - a)) if R[a] != R[b] else float(b)
    return float('inf')


def tau_grid(R, grid):
    best = 0.0
    for x in grid[1:]:
        if R[x] >= 0.5: best = x
        else: break
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); c = yaml.safe_load(open(a.config))
    b1 = yaml.safe_load(open(os.path.join(ROOT, c['b1_config']))); sp = b1['split']; cache = os.path.join(ROOT, c['cache'])
    pool = sorted(s for f in sp['subject_folds'] for s in f); runs = list(range(20)); grid = c['tau_grid_s']
    V = {}
    for p in sorted(glob.glob(os.path.join(cache, '*.npz'))):
        Z = np.load(p); tag = 'phaserand' if 'phaserand' in os.path.basename(p) else 'DE' if os.path.basename(p) == 'DE.npz' else 'main'
        for k in Z.files:
            if k != 'summary': V[(tag, k)] = Z[k]
    deltas = [round(x, 2) for x in c['deltas_s']]; conds = [f'd{d:+.2f}' for d in deltas] + [f's{r:.2f}' for r in c['stretch_r']]
    rng = np.random.default_rng(c['bootstrap']['seed']); NB = c['bootstrap']['n_boot']
    SI = rng.integers(0, len(pool), (NB, len(pool))); RI = rng.integers(0, 20, (NB, 20))
    def matrix(tag, prefix, cond, subjects):
        obs = np.full((len(subjects), 20), np.nan); nm = np.full((len(subjects), 20), np.nan); nul = np.full((len(subjects), 20, c['shift_null']['n_shift']), np.nan); w = np.zeros(20)
        for i, s in enumerate(subjects):
            for r in runs:
                key = (tag, f'{prefix}_r{r}_{cond}' if s is None else f'{prefix}_s{s}_r{r}_{cond}'); v = V.get(key)
                if v is None: continue
                obs[i, r] = v[0]; w[r] = v[1]; nul[i, r] = v[2:]; nm[i, r] = v[2:].mean()
        return obs, nm, nul, w
    def pooled(M, w, si=None, ri=None):
        si = np.arange(M.shape[0]) if si is None else si; ri = np.arange(M.shape[1]) if ri is None else ri; X = M[np.ix_(si, ri)]; ww = w[ri]
        ok = ~np.isnan(X).all(0); return float((np.nanmean(X[:, ok], 0) * ww[ok]).sum() / ww[ok].sum())
    def family(tag, prefix, subjects, cond_list):
        out = {}
        for cd in cond_list:
            obs, nm, nul, w = matrix(tag, prefix, cd, subjects)
            if np.isnan(obs).all(): continue
            ex = obs - nm; gnull = np.array([pooled(nul[..., b], w) for b in range(nul.shape[-1])])
            si_all = SI if len(subjects) == len(pool) else rng.integers(0, len(subjects), (NB, len(subjects)))
            boot = np.array([pooled(ex, w, si_all[b] if subjects[0] is not None else np.zeros(1, int), RI[b]) for b in range(NB)])
            out[cd] = dict(mrp=pooled(obs, w), null_mean=pooled(nm, w), excess=pooled(ex, w), excess_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                           obs_above_null_p975=bool(pooled(obs, w) > np.percentile(gnull, 97.5)), n_queries=float(w.sum()), _boot=boot)
        return out
    def retention(fam, key_fn, boot_index=None):
        e0 = fam[key_fn(0.0)]['excess'] if boot_index is None else fam[key_fn(0.0)]['_boot'][boot_index]
        R = {0.0: 1.0}
        for x in grid[1:]:
            ep = fam[key_fn(x)]['excess'] if boot_index is None else fam[key_fn(x)]['_boot'][boot_index]
            em = fam[key_fn(-x)]['excess'] if boot_index is None else fam[key_fn(-x)]['_boot'][boot_index]
            R[x] = (ep + em) / (2 * e0) if e0 > 0 else float('nan')
        return R
    def one_sided(fam, sign, key_fn):
        e0 = fam[key_fn(0.0)]['excess']; R = {0.0: 1.0}
        for x in grid[1:]: R[x] = fam[key_fn(sign * x)]['excess'] / e0 if e0 > 0 else float('nan')
        return tau50(R, grid)
    def summarise(fam, prefix_key='d', min_excess=None):
        kf = (lambda d: f'{prefix_key}{d:+.2f}')
        if kf(0.0) not in fam: return None
        e0 = fam[kf(0.0)]['excess']
        if min_excess is not None and e0 <= min_excess: return dict(defined=False, excess0=e0)
        R = retention(fam, kf); t = tau50(R, grid)
        bt = np.array([tau50(retention(fam, kf, b), grid) for b in range(NB)])
        Rb = {x: np.array([retention(fam, kf, b)[x] for b in range(NB)]) for x in grid[1:]}
        return dict(defined=True, excess0=e0, retention={str(x): float(v) for x, v in R.items()},
                    retention_ci95={str(x): [float(np.nanpercentile(Rb[x], 2.5)), float(np.nanpercentile(Rb[x], 97.5))] for x in grid[1:]},
                    tau50=t, tau50_ci95=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))], tau50_grid=tau_grid(R, grid),
                    tau50_plus=one_sided(fam, 1, kf), tau50_minus=one_sided(fam, -1, kf), _Rb=Rb)
    res = {}
    fams = dict(decoder=family('main', 'main', pool, conds + [f'ds{d:+.2f}' for d in deltas]),
                pre=family('main', 'pre', pool, conds), phaserand=family('phaserand', 'main', sp['subject_folds'][0], conds),
                target=family('main', 'tgt', [None], conds + [f'ds{d:+.2f}' for d in deltas]), de=family('DE', 'de', pool, conds))
    for name, fam in fams.items():
        res[name] = dict(conditions={k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')} for k, v in fam.items()})
        s = summarise(fam, 'd', c['de_min_excess'] if name == 'de' else None)
        if s: res[name]['tau'] = {k: v for k, v in s.items() if not k.startswith('_')}
        if name in ('decoder', 'target'):
            s2 = summarise(fam, 'ds'); res[name]['tau_delay_search_report'] = {k: v for k, v in s2.items() if not k.startswith('_')} if s2 else None
        if f'd+0.00' in fam: res[name]['stretch_retention'] = {f'{r:.2f}': fam[f's{r:.2f}']['excess'] / fam['d+0.00']['excess'] for r in c['stretch_r'] if f's{r:.2f}' in fam}
    sd, stt = summarise(fams['decoder']), summarise(fams['target'])
    Dq = sd['_Rb'][0.25] - stt['_Rb'][0.25]; D = sd['retention']['0.25'] - stt['retention']['0.25']
    res['kim_rule'] = dict(D_025=D, D_025_ci95=[float(np.nanpercentile(Dq, 2.5)), float(np.nanpercentile(Dq, 97.5))], within_margin=bool(abs(D) <= c['kim_rule_margin']))
    res['r3_tau_path'] = dict(tau50=sd['tau50'], threshold=c['r3_tau_s'], met=bool(sd['tau50'] >= c['r3_tau_s']))
    # D-100 amendment 1 (report only): fine shifts k = -4..4 samples of 1/120 s; not used for tau50 or R-e
    FV = {}
    for p in sorted(glob.glob(os.path.join(cache, 'fine', '*.npz'))):
        Z = np.load(p); tag = 'phaserand' if 'phaserand' in os.path.basename(p) else 'main'
        for k in Z.files: FV[(tag, k)] = Z[k]
    if FV:
        V_save = V; V = FV; fine = {}
        for name, tag, prefix, subj in (('decoder', 'main', 'main', pool), ('phaserand', 'phaserand', 'main', sp['subject_folds'][0]), ('target', 'main', 'tgt', [None])):
            fam = family(tag, prefix, subj, [f'k{k:+d}' for k in range(-4, 5)])
            if 'k+0' not in fam: continue
            e0 = fam['k+0']['excess']
            fine[name] = {k: dict(shift_ms=round(int(k[1:]) * 1000 / 120, 2), excess=v['excess'], excess_ci95=v['excess_ci95'], retention=v['excess'] / e0 if e0 > 0 else None,
                                  retention_ci95=[float(np.nanpercentile(v['_boot'] / fam['k+0']['_boot'], q)) for q in (2.5, 97.5)]) for k, v in fam.items()}
        res['fine_shift_report'] = fine; V = V_save
    out = os.path.join(ROOT, c['out']); os.makedirs(out, exist_ok=True); json.dump(res, open(os.path.join(out, 'tau50.json'), 'w'), indent=1)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), facecolor='#EEF0EC')
    col = dict(decoder='#B5652D', target='#5B4A86', phaserand='#182129', pre='#2E6E6B', de='#888888')
    for name, fam in fams.items():
        xs = [d for d in sorted(deltas)]; ys = [fam[f'd{d:+.2f}']['excess'] for d in xs if f'd{d:+.2f}' in fam]
        if len(ys) == len(xs): ax[0].plot(xs, ys, 'o-', color=col[name], label=name, ms=3)
    ax[0].axhline(0, color='#182129', lw=0.5); ax[0].set_xlabel('EEG window shift δ (s)'); ax[0].set_ylabel('excess over own null (MRP)'); ax[0].legend(fontsize=7)
    for name in ('decoder', 'target'):
        t = res[name]['tau']; xs = grid; ax[1].plot(xs, [t['retention'][str(x)] for x in xs], 'o-', color=col[name], label=f"{name} τ50 {t['tau50']:.2f} s", ms=3)
    ax[1].axhline(0.5, color='#182129', lw=0.5, ls='--'); ax[1].set_xlabel('|δ| (s)'); ax[1].set_ylabel('retention'); ax[1].legend(fontsize=7)
    for x in ax: x.set_facecolor('#EEF0EC')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'tau50.png'), dpi=150)
    print(json.dumps(dict(decoder_tau=res['decoder']['tau'], target_tau=res['target']['tau'], kim=res['kim_rule'], r3=res['r3_tau_path'],
                          phaserand_excess={k: round(v['excess'], 4) for k, v in res['phaserand']['conditions'].items()},
                          pre_excess={k: round(v['excess'], 4) for k, v in res['pre']['conditions'].items()},
                          de=res['de'].get('tau'), stretch=res['decoder'].get('stretch_retention')), indent=1, default=str), flush=True)


if __name__ == '__main__':
    main()
