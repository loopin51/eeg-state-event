"""D-128 hyperparameter search on Cho2017 (configs/nc_hpsearch.yaml, docs/08 §4.2).

usage: python scripts/nc_hpsearch.py configs/nc_hpsearch.yaml jobs [--arms a,b] > jobs.tsv    # k3s_submit.py --jobs lines
       python scripts/nc_hpsearch.py configs/nc_hpsearch.yaml collect [--runs results/runs]   # pick, freeze

`jobs` draws n_configs random configurations per arm (sampler_seed, the same draws whatever arms are listed) and writes
results/nc/hpsearch/configs.json. `collect` reads the run folders <date>_nc_cho2017_<arm>_s0_hpNN, takes the pooled
balanced accuracy over the searched folds, keeps the best configuration per arm (ties: lower index) and writes
results/nc/hpsearch/summary.json and configs/nc_train_frozen.yaml (configs/nc_train.yaml with the chosen overrides merged
into each arm; ablations and NC-init inherit the REVE-NC values).
"""
import os, sys, json, glob, copy, argparse
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INHERIT = {'reve_nc': ['nc_init', 'reve_nc_m1', 'reve_nc_m2', 'reve_nc_m3', 'reve_nc_m4', 'reve_nc_large']}


def draw(space, rng):
    out = {}
    for k in sorted(space):
        v = space[k]
        if 'log_uniform' in v: a, b = v['log_uniform']; out[k] = float(np.exp(rng.uniform(np.log(a), np.log(b))))
        elif 'uniform' in v: a, b = v['uniform']; out[k] = float(rng.uniform(a, b))
        else: out[k] = v['choice'][int(rng.integers(len(v['choice'])))]
    return out


def configs(h):
    out = {}
    for ai, arm in enumerate(sorted(h['arms'])):
        rng = np.random.default_rng([h['sampler_seed'], ai]); out[arm] = [draw(h['arms'][arm], rng) for _ in range(h['n_configs'])]
    return out


def setv(node, key, v):
    ks = key.split('.')
    for q in ks[:-1]: node = node.setdefault(q, {})
    node[ks[-1]] = v


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('mode', choices=['jobs', 'collect'])
    ap.add_argument('--arms'); ap.add_argument('--runs', default='results/runs'); ap.add_argument('--data', default='nc_cho'); a = ap.parse_args()
    h = yaml.safe_load(open(os.path.join(ROOT, a.config))); C = configs(h); od = os.path.join(ROOT, 'results', 'nc', 'hpsearch'); os.makedirs(od, exist_ok=True)
    arms = a.arms.split(',') if a.arms else sorted(C)
    if a.mode == 'jobs':
        json.dump(C, open(os.path.join(od, 'configs.json'), 'w'), indent=1)
        for arm in arms:
            for i, cfg in enumerate(C[arm]):
                hp = ' '.join(f'{k}={v}' for k, v in sorted(cfg.items())) + f" train.max_epochs={h['max_epochs']}"
                cmd = (f"python scripts/nc_train.py {h['train_config']} --arm {arm} --task {h['task']} --seed {h['seed']} "
                       f"--folds {','.join(map(str, h['folds']))} --tag _hp{i:02d} --hp {hp}")
                print(f"nc_hp_{arm.replace('_', '-')}_{i:02d}\t{a.data}\t{cmd}")
        return
    summary = {}
    for arm in arms:
        rows = []
        for i in range(h['n_configs']):
            f = sorted(glob.glob(os.path.join(ROOT, a.runs, f"*_nc_{h['task']}_{arm}_s{h['seed']}_hp{i:02d}", 'metrics.json')))
            if not f: rows.append(None); continue
            m = json.load(open(f[-1])); rows.append(dict(index=i, run=os.path.basename(os.path.dirname(f[-1])), score=m['info']['pooled'][h['criterion']],
                                                         folds=sorted(m['folds']), hp=C[arm][i]))
        done = [r for r in rows if r]; best = max(done, key=lambda r: (r['score'], -r['index'])) if done else None
        summary[arm] = dict(n_done=len(done), n_configs=h['n_configs'], best=best, all=done)
    json.dump(summary, open(os.path.join(od, 'summary.json'), 'w'), indent=1)
    t = yaml.safe_load(open(os.path.join(ROOT, h['train_config'])))
    for arm, s in summary.items():
        if not s['best'] or s['n_done'] < s['n_configs']: continue
        for tgt in [arm] + INHERIT.get(arm, []):
            for k, v in s['best']['hp'].items(): setv(t['arms'][tgt], k, v)
            t['arms'][tgt]['frozen_from'] = f"{arm} hp{s['best']['index']:02d} ({s['best']['run']})"
    yaml.safe_dump(t, open(os.path.join(ROOT, 'configs', 'nc_train_frozen.yaml'), 'w'), sort_keys=False, allow_unicode=True)
    print(json.dumps({k: (v['n_done'], v['best'] and round(v['best']['score'], 4)) for k, v in summary.items()}))


if __name__ == '__main__':
    main()
