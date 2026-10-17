"""F-V verdict, report-only diagnostic (after V3, fad7e8a; the verdict is unchanged): are the label-shuffle outputs nearly
constant over inputs? Each test concept is the true answer once per subject, so an input-independent output gives MRP 0.5
exactly by design. No cosine against the name texts, no rank and no MRP is computed here.

Per subject x condition (ses123, k0) x kind (avg20 [200 rows], single [single trials]) and per arm
(shuffle model; seed-mean decoder = ese.things.verdict.seed_mean; each decoder seed alone, for a single-model comparison):
  (a) mean_cos_to_mean : rows L2-normalised, mean over rows of the cosine with the mean row vector (1 = constant direction)
  (b) total_var_unit   : sum over dimensions of the variance over rows of the L2-normalised rows (= 1 - |mean unit row|^2)
      total_var_raw    : the same on the raw outputs (single models only; the seed mean exists only normalised),
      with mean_sq_norm_raw (mean |row|^2) and the ratio total_var_raw / mean_sq_norm_raw.
Ridge meters (results/fv_verdict/ridge_test.npz holds only the V1 score matrices [200 test rows, 200 candidates], not the
embeddings): (b) on the score rows, raw and after removing each row's mean over candidates (rank-invariant), with mean_sq and
the ratio, for de, time, eye, pre.

usage: python scripts/fv_shuffle_constancy.py configs/fv_verdict.yaml --v2-dir results/runs/2026-10-17_fv_verdict_v2
       [--out results/fv_verdict/shuffle_constancy.json]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, hashlib, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.things import verdict as V
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
CONDS, KINDS = ('ses123', 'k0'), ('avg20', 'single')


def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def tv(X): return float(np.var(X, 0).sum())


def unit_stats(X):
    U = V.unit(np.asarray(X, np.float64)); m = U.mean(0); d = m / (np.linalg.norm(m) + 1e-12)
    return dict(mean_cos_to_mean=float((U @ d).mean()), total_var_unit=tv(U))


def raw_stats(X):
    X = np.asarray(X, np.float64); t = tv(X); ms = float((X ** 2).sum(1).mean()); return dict(total_var_raw=t, mean_sq_norm_raw=ms, ratio_raw=t / ms)


def summary(rows):
    keys = rows[0].keys(); return {k: dict(mean=float(np.mean([r[k] for r in rows])), min=float(np.min([r[k] for r in rows])), max=float(np.max([r[k] for r in rows]))) for k in keys}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--v2-dir', required=True); ap.add_argument('--out'); a = ap.parse_args()
    c = yaml.safe_load(open(P(a.config))); out = P(c['out_dir']); d2 = P(a.v2_dir); nseed = len(c['final']['seeds'])
    v1 = json.load(open(os.path.join(out, 'v1_manifest.json'))); v2 = json.load(open(os.path.join(d2, 'v2_manifest.json')))
    assert v2['inputs'] == v1['inputs'], 'V2 inputs differ from V1'
    for k, h in v2['files'].items(): assert fsha(os.path.join(d2, k)) == h, f'{k}: embedding sha256 differs from V2'
    assert fsha(P(v1['ridge_test']['file'])) == v1['ridge_test']['sha256'], 'ridge test file changed'
    per, agg = {}, {}
    for s in c['subjects']:
        Z = np.load(os.path.join(d2, f'z_emb_t{s:02d}.npz'))
        for cond in CONDS:
            for kind in KINDS:
                sd = [Z[f'seed{i}|{cond}|{kind}'] for i in range(nseed)]; sh = Z[f'shuffle|{cond}|{kind}']
                arms = {'shuffle': dict(unit_stats(sh), **raw_stats(sh), n_rows=int(len(sh))),
                        'decoder_seed_mean': dict(unit_stats(V.seed_mean(sd)), n_rows=int(len(sd[0])))}
                for i, E in enumerate(sd): arms[f'decoder_seed{i}'] = dict(unit_stats(E), **raw_stats(E), n_rows=int(len(E)))
                per[f't{s:02d}|{cond}|{kind}'] = arms
                for arm, r in arms.items(): agg.setdefault(f'{cond}|{kind}|{arm}', []).append(r)
    R = np.load(P(v1['ridge_test']['file'])); rid, ragg = {}, {}
    for s in c['subjects']:
        for kind in ('de', 'time', 'eye', 'pre'):
            S = np.asarray(R[f'{kind}_t{s:02d}'], np.float64); Sc = S - S.mean(1, keepdims=True)
            r = dict(total_var_raw=tv(S), mean_sq_raw=float((S ** 2).sum(1).mean()), total_var_rowcentred=tv(Sc), mean_sq_rowcentred=float((Sc ** 2).sum(1).mean()))
            r['ratio_raw'] = r['total_var_raw'] / r['mean_sq_raw']; r['ratio_rowcentred'] = r['total_var_rowcentred'] / r['mean_sq_rowcentred']
            rid[f't{s:02d}|{kind}'] = r; ragg.setdefault(kind, []).append(r)
    res = dict(note='report only, after V3 (fad7e8a); no name-text cosine, rank or MRP computed', v2_dir=os.path.relpath(d2, ROOT),
               summary_over_subjects={k: summary(v) for k, v in agg.items()}, ridge_score_rows_summary={k: summary(v) for k, v in ragg.items()},
               per_subject=per, ridge_per_subject=rid)
    o = P(a.out or os.path.join(c['out_dir'], 'shuffle_constancy.json')); json.dump(res, open(o, 'w'), indent=1)
    print(json.dumps({k: {m: round(v[m]['mean'], 4) for m in ('mean_cos_to_mean', 'total_var_unit')} for k, v in res['summary_over_subjects'].items()}), flush=True)


if __name__ == '__main__':
    main()
