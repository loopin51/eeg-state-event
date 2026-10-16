"""Full-path check of the F-IV T2 verdict (D-109 section 10, second paragraph): fake inputs in the registered formats go
through scripts/t2_verdict.py unchanged. Not a registered computation and not used in any verdict.

Per repetition: fake score files of scripts/t2_infer.py (test, test_noise, phaserand; i.i.d. standard normal matrices on the
rows/columns that the real Q and matched-set files define, an added truth score `effect` for the TTS and recorded matrices of
the main mode), fake 0-T2, development, MDE, re-inference and DE records, then t2_verdict.py --sim. Records the categories,
the H_T2b verdict and the prerequisite results.

usage: python scripts/t2_verdict_sim.py configs/t2.yaml --sets PATH --attrs PATH --work DIR [--conditions info0:0:2 strong:1.5:2]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, shutil, hashlib, argparse, datetime, subprocess, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from t2_lib import path, parse_key, cand_index
from t2_io import read_sets, read_attrs, full


def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def one_rep(v, a, W, seed, eff):
    if os.path.exists(W): shutil.rmtree(W)
    sd = os.path.join(W, 'scores'); os.makedirs(sd); rng = np.random.default_rng(seed)
    b1 = yaml.safe_load(open(path(v['b1_config']))); sp = b1['split']; meta = json.load(open(path(os.path.join(b1['cache_dir'], 'eeg_meta.json'))))
    fs, lens = meta['sample_rate'], meta['run_lengths']; w, e = b1['window'], b1['eeg']; Tn = int(round((w['tmax'] - w['tmin']) * fs)); m = v['eeg_margin_s']
    sets = full(read_sets(path(a.sets))[1]); attrs = read_attrs(path(a.attrs))
    def ok(t, r0): return int(round((t + w['tmin'] + e['offset_s'] - m) * fs)) >= 0 and int(round((t + w['tmin'] + e['offset_s'] + m) * fs)) + Tn <= lens[r0]
    rows = {r: sorted((k for k in sets if parse_key(k)[0] == r and ok(attrs[k][1], r)), key=lambda k: attrs[k][1]) for r in range(20)}
    cols = {r: sorted(set(rows[r]) | {c for k in rows[r] for c in sets[k]['cands']}, key=lambda k: (attrs[k][0], attrs[k][1])) for r in range(20)}
    tgt = lambda K: sp['subject_folds'][K] + sp['extra_targets'].get(K, [])
    for mode, jobs, subs, ef in (('test', [(K, F) for K in range(5) for F in range(5)], tgt, eff), ('test_noise', [(K, F) for K in range(5) for F in range(5)], tgt, 0.0),
                                 ('phaserand', [(0, F) for F in range(5)], lambda K: sp['subject_folds'][0], 0.0)):
        summ = {}
        for K, F in jobs:
            arr = {}
            for r0 in [r - 1 for r in sp['run_folds'][F]]:
                arr[f'r{r0 + 1}_rows'] = np.array(rows[r0]); arr[f'r{r0 + 1}_cols'] = np.array(cols[r0]); ci = {k: j for j, k in enumerate(cols[r0])}
                tr = np.array([ci[k] for k in rows[r0]])
                for s in subs(K):
                    for kind in ('tts', 'rec'):
                        S = rng.standard_normal((len(rows[r0]), len(cols[r0]))).astype(np.float32); S[np.arange(len(tr)), tr] += ef; arr[f'{kind}_fixed_s{s}_r{r0 + 1}'] = S
            p = os.path.join(sd, f'{mode}_k{K}_f{F}.npz'); np.savez(p, **arr); summ[f'k{K}_f{F}'] = dict(sha256=sha(p))
        json.dump(dict(device='NVIDIA GeForce RTX 4070 (sim)', jobs=summ, matched_sets=dict(sha256=sha(path(a.sets))), match_attrs=dict(sha256=sha(path(a.attrs)))),
                  open(os.path.join(sd, f'{mode}_summary.json'), 'w'))
    DE = dict(subjects={})
    for s in sorted({s for K in range(5) for s in tgt(K)}):
        DE['subjects'][str(s)] = dict(cells={str(r + 1): dict(n=len(rows[r]), mrp_tts=float(0.5 + 0.02 * rng.standard_normal())) for r in range(20)})
    dep = os.path.join(W, 'de.json'); json.dump(DE, open(dep, 'w'))
    for name, obj in (('gate0T2.json', dict(passed=True)), ('dev_stage.json', dict(gate=dict(passed=True), rule_selection=dict(rule='fixed'), inputs=dict(device='NVIDIA GeForce RTX 4070 (sim)'))),
                      ('mde.json', dict(MDE=0.02)), ('reinfer.json', dict(device='NVIDIA GeForce RTX 4070 (sim)', passed=True, complete=True, max_abs_mrp_diff=0.0))):
        json.dump(obj, open(os.path.join(W, name), 'w'))
    V = dict(v); V['align'] = dict(v['align'], rule='fixed'); V['verdict'] = dict(v['verdict'], out=os.path.join(W, 'verdict.json'), gate0T2=os.path.join(W, 'gate0T2.json'),
                                                                                  dev_stage=os.path.join(W, 'dev_stage.json'))
    V['mde'] = dict(v['mde'], file=os.path.join(W, 'mde.json')); V['reinfer_check'] = dict(v['reinfer_check'], out=os.path.join(W, 'reinfer.json'))
    V['de_baseline'] = dict(file=dep, sha256=sha(dep), committed_at=datetime.datetime.now().isoformat(timespec='seconds'), commit='sim')
    vp = os.path.join(W, 't2_sim.yaml'); yaml.safe_dump(V, open(vp, 'w'))
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts/t2_verdict.py'), vp, '--scores', sd, '--sets', path(a.sets), '--attrs', path(a.attrs), '--sim'], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    R = json.load(open(os.path.join(W, 'verdict.json')))
    return dict(H_T2a=R['H_T2a'], H_T2b=R['H_T2b'], prerequisites={k: x['ok'] for k, x in R['prerequisites'].items()}, n_queries=R['n_queries'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--sets', required=True); ap.add_argument('--attrs', required=True)
    ap.add_argument('--work', required=True); ap.add_argument('--conditions', nargs='+', default=['info0:0:2', 'strong:1.5:2']); ap.add_argument('--out')
    a = ap.parse_args(); v = yaml.safe_load(open(path(a.config))); t0 = time.time(); res = {}
    for cd in a.conditions:
        name, eff, n = cd.split(':'); rows = []
        for i in range(int(n)):
            rows.append(one_rep(v, a, path(a.work), [20261319, int(float(eff) * 1000), i], float(eff))); print(name, i, json.dumps(rows[-1]), f'({time.time() - t0:.0f}s)', flush=True)
        res[name] = dict(effect=float(eff), rows=rows)
    shutil.rmtree(path(a.work), ignore_errors=True)
    if a.out: json.dump(dict(conditions=res, seconds=time.time() - t0), open(path(a.out), 'w'), indent=1)


if __name__ == '__main__':
    main()
