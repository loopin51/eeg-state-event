"""F-III A1 null simulation (code check; no EEG, no decoder output, no test cell is read).

Cell-level Monte Carlo of the A1 decision quantity with the registered interval (D-097 amendment 5 ③: seed-stratified
pigeonhole, sqrt(n/(n-1)) seed inflation, 5 seeds per arm; scripts/c1_stats.py) and categories (D-097 section 6) at both
chain-A levels (0.04, and 0.05 when the T1 stimulus gate fails first). Query counts per cell (17 pool subjects x 20 runs) are
those of the A1 query set (configs/a1.yaml queries.margin_s, word onsets and run lengths only); both arms have information 0,
plus an optional seed effect common to all cells; one power condition (true effect +0.03, seed sd 0.01).

usage: python scripts/a1_null_sim.py configs/a1_null_sim.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import c1_stats as CS


def counts(A, c):
    meta = json.load(open(os.path.join(ROOT, c['cache_dir'], 'eeg_meta.json'))); fs, lens = meta['sample_rate'], meta['run_lengths']
    w, e = c['window'], c['eeg']; m = A['queries']['margin_s']; Wd = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True)
    out = []
    for r in range(20):
        on = np.unique(Wd['onset'][Wd['run'] == r]); hi = lens[r] / fs - w['tmax'] - max(e['offset_s'], 0); on = on[(on >= -w['tmin']) & (on <= hi)]
        ok = (on + w['tmin'] + e['offset_s'] - m >= 0) & (on + w['tmax'] + e['offset_s'] + m <= lens[r] / fs - 1.0 / fs); out.append(int(ok.sum()))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); q = yaml.safe_load(open(a.config)); t0 = time.time()
    A = yaml.safe_load(open(os.path.join(ROOT, q['a1_config']))); c = yaml.safe_load(open(os.path.join(ROOT, A['b1_config'])))
    pool = sorted(s for fs in c['split']['subject_folds'] for s in fs); nr = counts(A, c); n = np.tile(np.array(nr, float), (len(pool), 1))
    rank_sd = np.sqrt(1 / 12 / n); res = dict(query_counts_per_run=nr, n_cells=int(n.size), n_queries=int(n.sum()), runs={})
    conds = [dict(seed_sd=t, effect=0.0) for t in q['seed_sd']] + q['power']
    for cd in conds:
        tau, eff = cd['seed_sd'], cd['effect']; rng = np.random.default_rng([q['seed'], int(round(tau * 1e4)), int(round(eff * 1e4))]); rows = []
        for _ in range(q['n_rep']):
            m = [0.5 + rng.normal(0, tau, (q['n_seeds'], 1, 1)) + rng.standard_normal((q['n_seeds'],) + n.shape) * np.sqrt(rank_sd ** 2 + q['cell_sd_extra'] ** 2) for _ in range(2)]
            m[1] = m[1] + eff; pt = CS.delta(m[0], m[1], n); iv = CS.interval(CS.seed_pigeonhole(m, n, q['n_boot'], rng, seed_inflate=True))
            rows.append({al: CS.category_welch(pt, iv[f'lower_{al}'], iv['p95'], 0.0, al, q['sesoi'])[0] for al in q['alphas']})
        for al in q['alphas']:
            cats = [r_[al] for r_ in rows]; key = f'alpha{al}_seedsd{tau}' + (f'_effect{eff}' if eff else '')
            res['runs'][key] = dict(alpha=al, seed_sd=tau, effect=eff, n_rep=q['n_rep'], reject=float(np.mean([x in CS.REJECTED for x in cats])),
                                    adopt=float(np.mean([x == 'adopt' for x in cats])), categories={k: float(np.mean([x == k for x in cats])) for k in CS.CATEGORIES})
            print(key, json.dumps(res['runs'][key]), flush=True)
    res['seconds'] = time.time() - t0; p = os.path.join(ROOT, q['out']); os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(res, open(p, 'w'), indent=1); print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
