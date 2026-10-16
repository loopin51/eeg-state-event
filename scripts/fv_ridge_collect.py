"""F-V (D-112 section 5.6): summary of the per-target ridge files (scripts/fv_ridge.py --mode dev) into results/baselines/things_de.json.

Per feature (de, time, eye, pre) and development set (A_ses4: gate D set; B_ses34: report only): per-target MRP / top-1 / top-5
and concept count, the concept-weighted mean over the development targets (dev.dev_targets) and over every target present,
the chosen penalties, and the sha256 of every source file. Development values only; test values are computed in the verdict job.

usage: python scripts/fv_ridge_collect.py configs/fv_train.yaml
"""
import os, sys, json, glob, hashlib, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    c = yaml.safe_load(open(os.path.join(ROOT, sys.argv[1]))); d = os.path.join(ROOT, 'results/baselines/fv_ridge_dev')
    R = {int(os.path.basename(p)[1:3]): json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, 't[0-9][0-9].json')))}
    assert R and all(not r['fake_targets'] for r in R.values()), 'no real-target ridge files'
    files = {os.path.relpath(p, ROOT): hashlib.sha256(open(p, 'rb').read()).hexdigest() for p in sorted(glob.glob(os.path.join(d, 't[0-9][0-9].*')))}
    out = dict(note='within-subject ridge, fit on sessions 1-3 non-development concepts, development concept names as candidates (D-112 5.4-5.6); development values only',
               targets=sorted(R), alpha_grid=R[min(R)].get('alpha_grid'), peak_rss_gb={str(t): R[t].get('peak_rss_gb') for t in sorted(R)}, dev_targets=c['dev']['dev_targets'], features={}, files=files)
    for f in R[min(R)]['features']:
        o = out['features'][f] = {}
        for s in ('A_ses4', 'B_ses34'):
            per = {str(t): R[t]['features'][f][s] for t in sorted(R)}
            def pooled(ts):
                ts = [t for t in ts if t in R]; n = np.array([R[t]['features'][f][s]['n_concepts'] for t in ts])
                return None if not ts else {k: float(np.average([R[t]['features'][f][s][k] for t in ts], weights=n)) for k in ('mrp', 'top1', 'top5')}
            o[s] = dict(per_target=per, dev_targets=pooled(c['dev']['dev_targets']), all_targets=pooled(sorted(R)))
        o['alpha'] = {str(t): R[t]['features'][f]['alpha'] for t in sorted(R)}
        grid = R[min(R)].get('alpha_grid'); o['alpha_at_grid_edge'] = None if grid is None else [str(t) for t in sorted(R) if R[t]['features'][f]['alpha'] in grid]
    p = os.path.join(ROOT, 'results/baselines/things_de.json'); json.dump(out, open(p, 'w'), indent=1); print('wrote', p, 'targets', sorted(R), flush=True)


if __name__ == '__main__':
    main()
