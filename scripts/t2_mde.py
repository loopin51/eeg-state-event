"""F-IV H_T2b MDE (D-109 section 12, amendment 1), recorded after the development stage and before any test TTS score.

Cell values D_{s,r} = rank noise of information-0 score matrices + sigma_subj u_s + sigma_cell e_{s,r} on the real test cell
structure (17 pool subjects x 20 runs, the run's Q query count from results/t2/matched_sets.json and the EEG margin).
Rank noise: with i.i.d. continuous scores the rank of the truth among its 19 candidates is uniform on 0..19, so each query's
MRP is drawn as U/19 independently for the recorded and the TTS term (the information-0 matrices of section 10 give the same
per-query law). sigma_subj, sigma_cell: development D components of results/t2/dev_stage.json (frozen rule). For each of
n_sim realisations the SE is the sd of the section-9 pigeonhole distribution (1,000); SE_0 = mean, MDE = 2.486 x SE_0.
MDE > 0.03 -> H_T2b "hold (MDE)". Reads no test score.

usage: python scripts/t2_mde.py configs/t2.yaml [--sets PATH --attrs PATH --dev PATH --out PATH]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, argparse, hashlib, datetime, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import t2_lib as T
from t2_lib import path, parse_key
from t2_io import read_sets, read_attrs, full


def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def q_counts(v, setp, attp):
    b1 = yaml.safe_load(open(path(v['b1_config']))); meta = json.load(open(path(os.path.join(b1['cache_dir'], 'eeg_meta.json'))))
    fs, lens = meta['sample_rate'], meta['run_lengths']; w, e = b1['window'], b1['eeg']; Tn = int(round((w['tmax'] - w['tmin']) * fs)); m = v['eeg_margin_s']
    sets = full(read_sets(setp)[1]); attrs = read_attrs(attp)
    def ok(t, r0): return int(round((t + w['tmin'] + e['offset_s'] - m) * fs)) >= 0 and int(round((t + w['tmin'] + e['offset_s'] + m) * fs)) + Tn <= lens[r0]
    return np.array([sum(1 for k in sets if parse_key(k)[0] == r and ok(attrs[k][1], r)) for r in range(20)], float)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--sets'); ap.add_argument('--attrs'); ap.add_argument('--dev'); ap.add_argument('--out')
    a = ap.parse_args(); v = yaml.safe_load(open(path(a.config))); q = v['mde']
    setp, attp = path(a.sets or v['matched_sets']), path(a.attrs or v['match_attrs']); devp = path(a.dev or v['verdict']['dev_stage'])
    dev = json.load(open(devp)); mc = dev['mde_components']; assert dev['gate']['passed'], 'MDE is recorded only after a passed development gate'
    w = q_counts(v, setp, attp); ns, nr = len(v['pool']), 20; rng = np.random.default_rng(q['seed']); se = []
    for i in range(q['n_sim']):
        noise = np.array([[(rng.integers(0, 20, int(w[r])).mean() - rng.integers(0, 20, int(w[r])).mean()) / 19.0 for r in range(nr)] for _ in range(ns)])
        D = noise + mc['sigma_subj'] * rng.standard_normal((ns, 1)) + mc['sigma_cell'] * rng.standard_normal((ns, nr))
        se.append(float(T.pigeonhole(D, w, v['bootstrap']['n_boot'], int(rng.integers(2 ** 31))).std(ddof=1)))
    SE0 = float(np.mean(se)); MDE = q['z_sum'] * SE0
    res = dict(statistic='H_T2b D = MRP_rec - MRP_TTS, 17 pool subjects x 20 test runs', SE_0=SE0, SE_sd=float(np.std(se, ddof=1)), MDE=MDE, threshold=q['threshold'],
               hold=bool(MDE > q['threshold']), components=mc, queries_per_run=w.astype(int).tolist(), n_sim=q['n_sim'], z_sum=q['z_sum'],
               inputs=dict(matched_sets=sha(setp), match_attrs=sha(attp), dev_stage=sha(devp)), written_at=datetime.datetime.now().isoformat(timespec='seconds'),
               note='recorded and committed before any test TTS score (D-109 sections 8-7, 12); components from development cells with the training subjects own layers')
    out = path(a.out or q['file']); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('SE_0', 'MDE', 'threshold', 'hold')}), flush=True)


if __name__ == '__main__':
    main()
