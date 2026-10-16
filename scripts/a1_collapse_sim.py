"""Information-0 check of A1 prerequisite 5 (collapse model, D-105 section 6.5) with the real A1 query onsets (no EEG, no
decoder output). For 17 pool subjects x 20 runs, i.i.d. normal score matrices on the A1 query set of every run
(configs/a1.yaml queries.margin_s; word onsets and run lengths only) go through the verdict's 30-s-excluded shift null
(scripts/a1_verdict.null_cells, B1 amendment 1 maps, 1,000 draws, seed 20261012), once as they are (calibration of the null)
and once collapsed to their column means (prerequisite 5). Repeated n_rep times; reports the pooled p, the subject-p KS p,
the max |null - observed| and the subject p's of every repetition.

usage: python scripts/a1_collapse_sim.py configs/a1_null_sim.yaml [--n-rep R]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np
from scipy.stats import kstest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import c1_stats as CS
from a1_verdict import null_cells
from stageB_b1_verdict import shift_maps


def query_onsets(A, c):
    meta = json.load(open(os.path.join(ROOT, c['cache_dir'], 'eeg_meta.json'))); fs, lens = meta['sample_rate'], meta['run_lengths']
    w, e = c['window'], c['eeg']; m = A['queries']['margin_s']; Wd = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True)
    out = {}
    for r in range(20):
        on = np.unique(Wd['onset'][Wd['run'] == r]); hi = lens[r] / fs - w['tmax'] - max(e['offset_s'], 0); on = on[(on >= -w['tmin']) & (on <= hi)]
        ok = (on + w['tmin'] + e['offset_s'] - m >= 0) & (on + w['tmax'] + e['offset_s'] + m <= lens[r] / fs - 1.0 / fs); out[r + 1] = on[ok].astype(np.float64)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--n-rep', type=int, default=3); a = ap.parse_args(); t0 = time.time()
    q = yaml.safe_load(open(a.config)); A = yaml.safe_load(open(os.path.join(ROOT, q['a1_config']))); c = yaml.safe_load(open(os.path.join(ROOT, A['b1_config'])))
    pool = sorted(s for fs in c['split']['subject_folds'] for s in fs); ons = query_onsets(A, c); excl = A['queries']['exclusion_s']
    maps = shift_maps({r: {'onset': ons[r]} for r in sorted(ons)}, 1000, 30.0, np.random.default_rng(20261012)); rows = []
    for rep in range(a.n_rep):
        rng = np.random.default_rng([20261108, rep]); mats = {s: {r: rng.standard_normal((len(ons[r]), len(ons[r]))) for r in ons} for s in pool}
        row = {}
        for collapse in (False, True):
            subs, uu, obs, null = null_cells(mats, ons, maps, excl, collapse=collapse); n = np.array([[len(ons[u]) for u in uu] for _ in subs], float)
            T = CS.level(obs, n); g = np.array([CS.level(x, n) for x in null])
            ps = [CS.perm_p(CS.level(obs[j:j + 1], n[j:j + 1]), [CS.level(x[j:j + 1], n[j:j + 1]) for x in null]) for j in range(len(subs))]
            row['collapsed' if collapse else 'raw'] = dict(T=T, null_mean=float(g.mean()), null_sd=float(g.std(ddof=1)), p=CS.perm_p(T, g),
                                                          ks_p=float(kstest(ps, 'uniform').pvalue), max_abs_null_minus_obs=float(np.nanmax(np.abs(null - obs[None]))),
                                                          subject_p=ps, min_subject_p=float(min(ps)))
        rows.append(row); print(rep, json.dumps({k: {kk: x[kk] for kk in ('T', 'null_mean', 'null_sd', 'p', 'ks_p', 'max_abs_null_minus_obs', 'min_subject_p')} for k, x in row.items()}), flush=True)
    out = os.path.join(ROOT, 'results/a1/collapse_sim.json'); json.dump(dict(query_counts={str(r): len(x) for r, x in ons.items()}, rows=rows, seconds=time.time() - t0), open(out, 'w'), indent=1)
    print('wrote', out, f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
