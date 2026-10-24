"""D-125 verdict (server CPU, run once after every arm x task has finished).

usage: python scripts/reve_speech_verdict.py configs/reve_speech.yaml

Reads results/runs/*_rs_<arm>_<task>/metrics.json (non-smoke; one folder per arm x task) and the DE+LR baseline, and writes
results/reve_speech/verdict.json:
  - per arm x task: mean, SD and t 95% interval of the per-subject score (UGR accuracy, BCI2020 balanced accuracy), kappa,
    weighted F1, one-sided Wilcoxon against chance (paper 2.5);
  - primary comparison (D-125 4): REVE - EEGNet per task, two-sided Wilcoxon, Holm over the 5 tasks, subject bootstrap 95% CI;
  - report-only: REVE - LaBraM, REVE - DE+LR, REVE lr 1e-4;
  - paper check: |ours - paper| for EEGNet and LaBraM (> 5 pp -> "pipeline reproduction difference");
  - label-shuffle check: chance inside the t 95% interval of REVE label-shuffle.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, glob, argparse
import numpy as np, yaml
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ['eegnet', 'labram', 'reve', 'reve_shuffle', 'reve_lr1e4']


def run_metrics(c, arm, task):
    fs = sorted(f for f in glob.glob(os.path.join(ROOT, c['runs_dir'], f'*_rs_{arm}_{task}', 'metrics.json')))
    if not fs: return None
    assert len(fs) == 1, f'more than one run for {arm} {task}: {fs}'
    return json.load(open(fs[0]))['subjects'], os.path.relpath(os.path.dirname(fs[0]), ROOT)


def summary(per, key, chance):
    x = np.array([v[key] for v in per.values()]); n = len(x); m, sd = float(x.mean()), float(x.std(ddof=1)); h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    w = stats.wilcoxon(x - chance, alternative='greater') if np.any(x != chance) else None
    return dict(n=n, mean=m, sd=sd, ci95=[m - h, m + h], kappa=float(np.mean([v['kappa'] for v in per.values()])),
                wf1=float(np.mean([v['wf1'] for v in per.values()])), p_vs_chance=None if w is None else float(w.pvalue),
                best_epoch_mean=float(np.mean([v.get('best_epoch', np.nan) for v in per.values()])))


def paired(a, b, key, rng, B):
    subs = sorted(set(a) & set(b)); d = np.array([a[s][key] - b[s][key] for s in subs])
    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return dict(n=len(d), mean_diff=float(d.mean()), p_two_sided=p, boot_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                n_positive=int((d > 0).sum()))


def holm(ps, alpha):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(order): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    c = yaml.safe_load(open(os.path.join(ROOT, a.config))); v = c['verdict']; rng = np.random.default_rng(c['seed'])
    base = json.load(open(os.path.join(ROOT, v['baseline']))); out = dict(arms={}, runs={}, primary={}, report={}, paper_check={}, shuffle_check={})
    tasks = list(c['tasks'])
    for task in tasks:
        t = c['tasks'][task]; key = 'bal_acc' if t['data'] == 'bci2020' else 'acc'; ch = 1.0 / t['n_classes']; got = {}
        for arm in ARMS:
            r = run_metrics(c, arm, task)
            if r: got[arm] = r[0]; out['runs'][f'{arm}/{task}'] = r[1]; out['arms'].setdefault(arm, {})[task] = summary(r[0], key, ch)
        got['de_lr'] = base[task]['subjects']; out['arms'].setdefault('de_lr', {})[task] = summary(got['de_lr'], key, ch)
        if 'reve' in got and 'eegnet' in got: out['primary'][task] = paired(got['reve'], got['eegnet'], key, rng, v['bootstrap'])
        for other in ('labram', 'de_lr', 'reve_lr1e4'):
            if 'reve' in got and other in got: out['report'][f'reve-{other}/{task}'] = paired(got['reve'], got[other], key, rng, v['bootstrap'])
        for arm, name in (('eegnet', 'EEGNet'), ('labram', 'LaBraM')):
            if arm in got:
                ours = 100 * out['arms'][arm][task]['mean']; paper = c['paper'][task][name]
                out['paper_check'][f'{arm}/{task}'] = dict(ours=ours, paper=paper, diff_pp=ours - paper,
                                                           reproduction_difference=bool(abs(ours - paper) > c['paper']['reproduction_tolerance_pp']))
        if 'reve_shuffle' in got:
            s = out['arms']['reve_shuffle'][task]; out['shuffle_check'][task] = dict(chance=ch, ci95=s['ci95'], passed=bool(s['ci95'][0] <= ch <= s['ci95'][1]))
    done = [t for t in tasks if t in out['primary']]
    if len(done) == len(tasks):
        adj = holm(np.array([out['primary'][t]['p_two_sided'] for t in done]), v['holm_alpha'])
        for t, p in zip(done, adj):
            r = out['primary'][t]; r['p_holm'] = float(p)
            r['decision'] = ('REVE advantage' if r['mean_diff'] > 0 else 'EEGNet advantage') if p < v['holm_alpha'] else 'not distinguishable'
    else:
        out['incomplete'] = sorted(set(tasks) - set(done))
    path = os.path.join(ROOT, v['out']); os.makedirs(os.path.dirname(path), exist_ok=True); json.dump(out, open(path, 'w'), indent=1)
    print(json.dumps(dict(primary=out['primary'], paper_check=out['paper_check'], shuffle_check=out['shuffle_check']), indent=1))


if __name__ == '__main__':
    main()
