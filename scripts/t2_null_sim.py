"""F-IV T2 information-0 simulation (D-109 section 10; before any test TTS score, not used in the verdict).

On the real Q, matched sets, shift maps and cell query counts (17 pool subjects x 20 runs), TTS and recorded scores are
i.i.d. standard normal; E_T (matched TTS MRP - 30-s-excluded shift-null mean) and D (recorded - TTS) go through the verdict's
statistics (t2_lib.cell_obs / cell_null / pigeonhole, c1_stats.category). n_rep repetitions (seed null_sim.seed). The null of
each simulated cell uses the first null_sim.n_shift draws of the registered maps (the verdict uses all 1,000) to keep the
2,000 repetitions within a CPU job; the count is written to the output. Reports: H_T2a rejection rate (lower > 0) and adoption
rate at alpha 0.05, the non-inferiority rate of D (95th percentile < 0.03), category rates. Flag: rejection rate above
0.05 + 2 x 0.005 -> tell the coordinator before any test score.

usage: python scripts/t2_null_sim.py configs/t2.yaml [--n-rep N] [--sets PATH --attrs PATH --out PATH]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, hashlib, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import t2_lib as T
import c1_stats as CS
from t2_lib import path, parse_key
from t2_mde import q_counts
from t2_io import read_sets, read_attrs, full


def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--n-rep', type=int); ap.add_argument('--sets'); ap.add_argument('--attrs'); ap.add_argument('--out')
    a = ap.parse_args(); v = yaml.safe_load(open(path(a.config))); q = v['null_sim']; vd = v['verdict']; t0 = time.time()
    setp, attp = path(a.sets or v['matched_sets']), path(a.attrs or v['match_attrs'])
    sets = full(read_sets(setp)[1]); attrs = read_attrs(attp)
    w = q_counts(v, setp, attp); b1 = yaml.safe_load(open(path(v['b1_config']))); meta = json.load(open(path(os.path.join(b1['cache_dir'], 'eeg_meta.json'))))
    fs, lens = meta['sample_rate'], meta['run_lengths']; wd, e = b1['window'], b1['eeg']; Tn = int(round((wd['tmax'] - wd['tmin']) * fs)); m = v['eeg_margin_s']
    def ok(t, r0): return int(round((t + wd['tmin'] + e['offset_s'] - m) * fs)) >= 0 and int(round((t + wd['tmin'] + e['offset_s'] + m) * fs)) + Tn <= lens[r0]
    rows = {r: sorted((k for k in sets if parse_key(k)[0] == r and ok(attrs[k][1], r)), key=lambda k: attrs[k][1]) for r in range(20)}
    geo = {}
    for r in range(20):
        cols = sorted(set(rows[r]) | {c for k in rows[r] for c in sets[k]['cands']}, key=lambda k: (attrs[k][0], attrs[k][1]))
        geo[r] = dict(C=T.cand_index(rows[r], cols, sets), m=len(cols), rt=np.array([attrs[k][1] for k in rows[r]]), ct=np.array([attrs[k][1] for k in cols]),
                      cs=np.array([parse_key(k)[0] == r for k in cols]))
    maps = T.run_maps({r: geo[r]['rt'] for r in range(20)}, v['shift_null']['n_shift'], v['shift_null']['min_shift_s'], v['shift_null']['seed'])
    ns = len(v['pool']); n_rep = a.n_rep or q['n_rep']; rng = np.random.default_rng(q['seed']); rows_out = []
    for i in range(n_rep):
        E = np.empty((ns, 20)); D = np.empty((ns, 20))
        for r in range(20):
            g = geo[r]; n = len(g['rt'])
            for s in range(ns):
                S = rng.standard_normal((n, g['m'])); mt = T.cell_obs(S, g['C']).mean()
                nl = T.cell_null(S, g['C'], g['rt'], g['ct'], g['cs'], maps[r][:q['n_shift']], v['exclusion_s'])[0].mean()
                mr = T.cell_obs(rng.standard_normal((n, g['m'])), g['C']).mean(); E[s, r] = mt - nl; D[s, r] = mr - mt
        seed_b = int(rng.integers(2 ** 31)); Eb = T.pigeonhole(E, w, v['bootstrap']['n_boot'], seed_b); Db = T.pigeonhole(D, w, v['bootstrap']['n_boot'], seed_b)
        pt = T.pooled(E, w); iv = CS.interval(Eb, alphas=(vd['alpha'],)); cat, _ = CS.category(pt, iv[f"lower_{vd['alpha']}"], iv['p95'], vd['sesoi'])
        rows_out.append(dict(E_T=pt, lower=iv[f"lower_{vd['alpha']}"], category=cat, D=T.pooled(D, w), D_upper95=float(np.percentile(Db, 95))))
        if i % 50 == 0: print(i, json.dumps(rows_out[-1]), f'({time.time() - t0:.0f}s)', flush=True)
    cats = [x['category'] for x in rows_out]; rej = float(np.mean([x['lower'] > 0 for x in rows_out]))
    res = dict(n_rep=n_rep, n_shift_sim=q['n_shift'], reject_rate=rej, adopt_rate=cats.count('adopt') / n_rep, category_rate={c: cats.count(c) / n_rep for c in sorted(set(cats))},
               noninferior_rate=float(np.mean([x['D_upper95'] < vd['ni_margin'] for x in rows_out])), flag=bool(rej > 0.05 + 2 * 0.005),
               queries_per_run=w.astype(int).tolist(), rows=rows_out, seconds=time.time() - t0, inputs=dict(matched_sets=sha(setp), match_attrs=sha(attp)),
               note='information 0; the non-inferiority rate is a check of D under information 0, not a power value')
    out = path(a.out or 'results/t2/null_sim.json'); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('n_rep', 'n_shift_sim', 'reject_rate', 'adopt_rate', 'category_rate', 'noninferior_rate', 'flag', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
