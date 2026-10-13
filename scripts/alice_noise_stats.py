"""D-092 supplement 4: per-segment noise statistics of the Alice caches, for the report-only pool (no decoder or baseline input).

For each subject, present segment and source: per good channel (provider bad marks and flat channels excluded) the ratio
SD / (IQR / 1.349) inside the segment (1 for Gaussian data; large when rare artifacts dominate the SD), summarised as the
median over channels, and the fraction of samples at the clamp (eeg_hp05, |x| >= 20) or clip (reve, |x| >= 15).
The subject statistic is the maximum over segments of the channel-median ratio.

usage: python scripts/alice_noise_stats.py configs/alice_prepare.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMIT = {'eeg_hp05': 20.0, 'reve': 15.0}


def main():
    c = yaml.safe_load(open(sys.argv[1])); d = os.path.join(ROOT, c['out_dir']); meta = json.load(open(os.path.join(d, 'eeg_meta.json'))); res = {}
    for s, info in meta['subjects'].items():
        miss = [k - 1 for k in info['missing_segments']]; row = res[s] = {}
        for src in c['noise_pool']['sources']:
            o = meta['sources'][src]['run_offsets']; Z = np.load(os.path.join(d, f'{src}_sub-{int(s):03d}.npy'), mmap_mode='r')
            good = ~(np.array(info['bad_mask']) | np.array(info['sources'][src]['flat'])) if src != 'reve' else ~np.array(info['sources'][src]['flat'])
            ratio, sat = {}, {}
            for k in range(len(o) - 1):
                if k in miss: continue
                x = np.asarray(Z[good, o[k]:o[k + 1]], np.float64); iqr = np.subtract(*np.percentile(x, [75, 25], axis=1)) / 1.349
                ratio[str(k + 1)] = float(np.median(x.std(1) / np.maximum(iqr, 1e-12))); sat[str(k + 1)] = float(np.mean(np.abs(x) >= LIMIT[src]))
            row[src] = dict(seg_ratio=ratio, seg_saturated=sat, max_ratio=max(ratio.values()), max_saturated=max(sat.values()))
    p = os.path.join(ROOT, c['noise_pool']['out']); os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(res, open(p, 'w'), indent=1)
    for s in sorted(res, key=lambda s: -res[s]['eeg_hp05']['max_ratio']):
        print(f"S{int(s):02d}", ' '.join(f"{src}: max ratio {v['max_ratio']:.2f} (seg {max(v['seg_ratio'], key=v['seg_ratio'].get)}), max sat {v['max_saturated']:.4f}" for src, v in res[s].items()))


if __name__ == '__main__':
    main()
