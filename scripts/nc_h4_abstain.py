"""D-128 H4 prediction abstention (report-only, outside the verdict family; rule fixed 2026-11-05 before computing,
docs/DECISIONS.md D-128 "H4 예측 보류 계산 세부"). Reads only the saved test predictions of the registered grid runs.

usage: env/bin/python scripts/nc_h4_abstain.py configs/nc_h4.yaml

Per task x arm (VERA, EEGNet, REVE-Official; seeds of the grid) and its label-shuffle run (seed 0):
  confidence  max softmax over the K task classes; VERA's K+1 outputs are renormalised over the first K columns, and the
              null column is reported separately (mean probability, fraction of windows whose argmax over K+1 is null)
  AURC        windows sorted by confidence (highest first, ties in window order); selective risk at coverage i/n is the
              error rate of the top i; AURC is the mean over i = 1..n. Random-order expectation = overall error rate.
  abstention  fraction of windows with confidence < tau_K = 1/K + (1 - 1/K)/2
  aurc_tie_avg  sensitivity: the expected AURC over random order within windows of equal confidence (errors replaced by
              their tie-block mean); differs from AURC only where confidences tie (e.g. softmax saturated at 1.0 in float32)
Seed values are averaged. Subject bootstrap: one multinomial count per subject per draw ([boot_seed, sum(ord(task))],
row 0 = observed), shared by every run of the task and by every seed; a window enters a draw as often as its subject.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, glob, argparse
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from nc_grid import runs as grid_runs, folder

STATS = ('aurc', 'err', 'abstain', 'aurc_minus_err', 'aurc_tie_avg')


def load(rd, r, null_arm):
    f = sorted(glob.glob(os.path.join(rd, '*' + folder(r), 'preds.npz')))
    if not f: return None
    z = np.load(f[-1]); m = json.load(open(os.path.join(os.path.dirname(f[-1]), 'metrics.json')))
    K = int(m['info']['n_classes']); P = z['probs'].astype(np.float64); y = z['y'].astype(int); s = z['subject'].astype(int)
    assert P.shape[1] == K + int(null_arm), (folder(r), P.shape, K, null_arm)
    pk = P[:, :K] / P[:, :K].sum(1, keepdims=True); conf = pk.max(1); err = (pk.argmax(1) != y).astype(np.float64)
    order = np.argsort(-conf, kind='stable'); cs = conf[order]; blk = np.concatenate([[0], np.cumsum(cs[1:] != cs[:-1])])
    out = dict(run=os.path.basename(os.path.dirname(f[-1])), K=K, n=len(y), y=y, subj=s, conf=conf, err=err, order=order, blk=blk,
               n_conf_unique=int(len(np.unique(conf))), n_conf_max_tie=int(np.unique(conf, return_counts=True)[1].max()))
    if null_arm:
        out.update(null_prob_mean=float(P[:, K].mean()), null_argmax_frac=float((P.argmax(1) == K).mean()),
                   null_prob_median=float(np.median(P[:, K])))
        h = json.load(open(os.path.join(os.path.dirname(f[-1]), 'history.json')))      # null loss logged <=> null windows trained
        out['null_trained'] = all('null' in e for fold in h.values() for e in fold)
    return out


def stats(run, W, tau):
    """W [B+1, S] subject counts -> dict of [B+1] arrays."""
    o = run['order']; e_s, s_s = run['err'][o], run['subj'][o]; ab = (run['conf'] < tau).astype(np.float64); n_sub = run['subj']
    blk = run['blk']; R = {k: np.empty(len(W)) for k in STATS}
    for b, w in enumerate(W):
        wj = w[s_s]; e = np.repeat(e_s, wj); N = len(e); cov = np.arange(1, N + 1)
        R['aurc'][b] = float(np.mean(np.cumsum(e) / cov))
        sw = np.bincount(blk, wj); et = np.bincount(blk, wj * e_s) / np.where(sw > 0, sw, 1)
        R['aurc_tie_avg'][b] = float(np.mean(np.cumsum(np.repeat(et[blk], wj)) / cov))
        R['err'][b] = float(e.mean()); R['abstain'][b] = float((w[n_sub] * ab).sum() / N)
    R['aurc_minus_err'] = R['aurc'] - R['err']
    return R


def oracle_aurc(err):
    """AURC if every correct window ranked above every error (reference only)."""
    n = len(err); nc = int(n - err.sum()); i = np.arange(1, n + 1)
    return float(np.mean(np.where(i <= nc, 0.0, (i - nc) / i)))


def summary(runs, W, tau, level):
    per = [stats(r, W, tau) for r in runs]; a = (1 - level) / 2; out = {}
    for k in STATS:
        mean = np.mean([p[k] for p in per], axis=0)          # seed average on the same draw
        out[k] = float(mean[0]); out[k + '_ci'] = [float(np.quantile(mean[1:], a)), float(np.quantile(mean[1:], 1 - a))]
    out['per_seed'] = {r['run']: {**{k: float(p[k][0]) for k in STATS}, 'aurc_oracle': oracle_aurc(r['err']),
                                  'n_conf_unique': r['n_conf_unique'], 'n_conf_max_tie': r['n_conf_max_tie'],
                                  **{k: r[k] for k in ('null_prob_mean', 'null_prob_median', 'null_argmax_frac', 'null_trained') if k in r}}
                       for r, p in zip(runs, per)}
    out['aurc_oracle'] = float(np.mean([oracle_aurc(r['err']) for r in runs]))
    if 'null_prob_mean' in runs[0]:
        for k in ('null_prob_mean', 'null_prob_median', 'null_argmax_frac'): out[k] = float(np.mean([r[k] for r in runs]))
        out['null_trained'] = all(r['null_trained'] for r in runs)
    out['n_seeds'] = len(runs)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    v = yaml.safe_load(open(os.path.join(ROOT, a.config))); g = yaml.safe_load(open(os.path.join(ROOT, v['grid'])))
    rd = os.path.join(ROOT, v['runs_dir']); R = grid_runs(g); tasks = v['tasks']['h2'] + v['tasks']['h3']
    res = dict(rule=dict(confidence='max softmax over K task classes (VERA: first K of K+1 columns renormalised)',
                         tau='1/K + (1 - 1/K)/2', aurc='mean over coverage i/n, i=1..n, of top-i error rate; ties in window order',
                         bootstrap='subjects resampled with replacement (multinomial counts), same draw for all runs and seeds of a task',
                         aurc_oracle='reference only: AURC if all correct windows ranked first',
                         aurc_tie_avg='sensitivity only: expected AURC over random order within equal-confidence windows',
                         null_trained='VERA: null loss present in every logged epoch (task has null windows)'),
               n_boot=v['n_boot'], boot_seed=v['boot_seed'], level=v['level'], tasks={}, missing=[])
    for task in tasks:
        sel = [r for r in R if r['task'] == task and r['sb'] is None and r['arm'] in v['arms']]
        T = dict(group='h2' if task in v['tasks']['h2'] else 'h3', arms={}, shuffle={})
        W = None
        for arm in v['arms']:
            for shuf in (False, True):
                rr = sorted([r for r in sel if r['arm'] == arm and r['shuffle'] == shuf], key=lambda r: r['seed'])
                L = [load(rd, r, arm in v['null_arms']) for r in rr]
                res['missing'] += [folder(r)[1:] for r, l in zip(rr, L) if l is None]; L = [l for l in L if l is not None]
                if not L: continue
                K, n = L[0]['K'], L[0]['n']; S = int(max(l['subj'].max() for l in L)) + 1
                assert all(l['K'] == K and l['n'] == n and np.array_equal(l['subj'], L[0]['subj']) and np.array_equal(l['y'], L[0]['y']) for l in L)
                if W is None:
                    rng = np.random.default_rng([v['boot_seed'], sum(map(ord, task))])
                    W = np.vstack([np.ones(S, int), rng.multinomial(S, np.full(S, 1 / S), size=v['n_boot'])]); T.update(K=K, n_windows=n, n_subjects=S)
                    T['tau'] = 1 / K + (1 - 1 / K) / 2; T['y'] = L[0]['y']
                assert W.shape[1] == S and np.array_equal(T['y'], L[0]['y'])
                (T['shuffle'] if shuf else T['arms'])[arm] = dict(seeds=[r['seed'] for r in rr], **summary(L, W, T['tau'], v['level']))
                print(task, arm, 'shuf' if shuf else 'main', len(L), flush=True)
        T.pop('y', None); res['tasks'][task] = T
    out = os.path.join(ROOT, v['out']); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1); print(out, 'missing', len(res['missing']))


if __name__ == '__main__':
    main()
