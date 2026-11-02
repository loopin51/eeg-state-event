"""D-128 registered grid (configs/nc_grid.yaml) -> runs of scripts/nc_train.py.

usage: python scripts/nc_grid.py configs/nc_grid.yaml list   [--group h1]               # every run: group, name, command
       python scripts/nc_grid.py configs/nc_grid.yaml k3s    --group h1 [--todo DIR]    # k3s_submit.py --jobs lines
       python scripts/nc_grid.py configs/nc_grid.yaml gpuq   --group h3 [--todo DIR]    # one gpuq submit line per run
       python scripts/nc_grid.py configs/nc_grid.yaml status [--runs results/runs]      # done / missing per group

A run is identified by (task, sb, arm, shuffle, seed); its folder is <date>_nc_<task>[_<sb>]_<arm>[_shuf]_s<seed>_g with
metrics.json once finished. --todo DIR leaves out runs that already have a finished folder in DIR. Runs are listed in
the order of the items (main seeds first), which is the submission order.
"""
import os, sys, glob, argparse, shlex
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = '_g'


def runs(g):
    out, seen = [], set()
    for gname, grp in g['groups'].items():
        for it in grp['items']:
            tasks = it['task'] if isinstance(it['task'], list) else [it['task']]
            for seed in g['seeds'][it['seeds']]:
                for task in tasks:
                    for sb in it.get('sb', [None]):
                        for arm in it['arms']:
                            key = (task, sb, arm, bool(it.get('shuffle')), seed)
                            if key in seen: continue
                            seen.add(key)
                            data = it.get('data', grp.get('data'))
                            data = data.get(task) if isinstance(data, dict) else data
                            out.append(dict(group=gname, task=task, sb=sb, arm=arm, shuffle=key[3], seed=seed, data=data))
    return out


def folder(r):
    return f"_nc_{r['task']}{'_' + r['sb'] if r['sb'] else ''}_{r['arm']}{'_shuf' if r['shuffle'] else ''}_s{r['seed']}{TAG}"


def done(r, runs_dir):
    return bool(glob.glob(os.path.join(runs_dir, '*' + folder(r), 'metrics.json')))


def command(g, r):
    return (f"python scripts/nc_train.py {g['train_config']} --arm {r['arm']} --task {r['task']} --seed {r['seed']}"
            f"{' --sb ' + r['sb'] if r['sb'] else ''}{' --shuffle' if r['shuffle'] else ''} --tag {TAG}")


def k8s_label(r):
    s = f"nc_g_{r['task']}{'_' + r['sb'] if r['sb'] else ''}_{r['arm']}{'_shuf' if r['shuffle'] else ''}_s{r['seed']}"
    return s.replace('physionet_mi', 'pmi').replace('reve_official', 'rofc').replace('reve_nc', 'rnc').replace('_', '-')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('mode', choices=['list', 'k3s', 'gpuq', 'status'])
    ap.add_argument('--group'); ap.add_argument('--todo'); ap.add_argument('--runs', default='results/runs'); a = ap.parse_args()
    g = yaml.safe_load(open(os.path.join(ROOT, a.config))); R = [r for r in runs(g) if not a.group or r['group'] == a.group]
    if a.todo: R = [r for r in R if not done(r, a.todo)]
    if a.mode == 'list':
        for r in R: print(f"{r['group']}\t{folder(r)[1:]}\t{command(g, r)}")
    elif a.mode == 'k3s':
        for r in R:
            assert g['groups'][r['group']]['runner'] == 'k3s'; print(f"{k8s_label(r)}\t{r['data']}\t{command(g, r)}")
    elif a.mode == 'gpuq':
        mem = g['groups'][a.group].get('mem_gb', 11)
        for r in R:
            assert g['groups'][r['group']]['runner'] == 'gpuq'
            print(f"~/.local/bin/gpuq submit -n {shlex.quote(folder(r)[4:])} -m {mem} -- env/bin/{command(g, r)}")
    else:
        rd = os.path.join(ROOT, a.runs)
        for gname in g['groups']:
            G = [r for r in R if r['group'] == gname]; n = sum(done(r, rd) for r in G); print(f'{gname}: {n}/{len(G)} done')


if __name__ == '__main__':
    main()
