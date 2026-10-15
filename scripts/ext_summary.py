"""D-099: size table of the pretraining caches (subjects, EEG hours, unique stimulus hours, exclusions, clamp rate).

Reads only cache metadata (eeg_meta.json, stimuli.json) and, for the clamp fraction, a strided subsample of the EEG memmaps
(no decoder, no labels). Output: results/d099_ext/summary.json and a printed table.

usage: python scripts/ext_summary.py configs/c1_p2_pretrain.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); out = {}
    for d in cfg['datasets']:
        cache = os.path.join(ROOT, d['cache_dir'])
        if not os.path.exists(os.path.join(cache, 'eeg_meta.json')): out[d['name']] = dict(status='no cache'); continue
        meta = json.load(open(os.path.join(cache, 'eeg_meta.json')))
        if d['kind'] == 'alice':
            src = meta['sources'][d.get('source', 'eeg_hp05')]; fs = src['sample_rate']; L = src['run_lengths']
            n = {s: [k for k in range(len(L)) if k + 1 not in v['missing_segments']] for s, v in meta['subjects'].items()}
            eeg_s = sum(L[k] for ks in n.values() for k in ks) / fs; stim_s = sum(L) / fs; nrec = sum(len(v) for v in n.values())
            excl = {s: [f'segment {k}' for k in v['missing_segments']] for s, v in meta['subjects'].items() if v['missing_segments']}
            bad = float(np.mean([sum(v['bad_mask']) for v in meta['subjects'].values()]))
        else:
            fs = meta['sample_rate']; R = {s: v for s, v in meta['subjects'].items()}
            ok = [r for v in R.values() for r in v if 'excluded' not in r]; eeg_s = sum(r['length'] for r in ok) / fs; nrec = len(ok)
            st = json.load(open(os.path.join(cache, 'stimuli.json')))['stimuli'] if os.path.exists(os.path.join(cache, 'stimuli.json')) else {}
            used = {r['stim'] for r in ok}; stim_s = sum(st[k]['length'] for k in used if k in st) / fs if st else None
            excl = {s: [f"{r['rec']}: {r['excluded']}" for r in v if 'excluded' in r] for s, v in R.items() if any('excluded' in r for r in v)}
            bad = float(np.mean([np.mean([sum(r['bad']) for r in v if 'excluded' not in r]) for v in R.values()]))
            extra = dict(trigger_resid_max_ms=float(max((r.get('trigger_resid_max_ms', 0) for r in ok), default=0)),
                         drift_ppm_range=[float(min((r['clock_rate'] - 1) * 1e6 for r in ok)), float(max((r['clock_rate'] - 1) * 1e6 for r in ok))],
                         trigger_notes={r['rec']: r['trigger_note'] for r in ok if r.get('trigger_note')})
        clamp = []
        for s in list(meta['subjects'])[:: max(1, len(meta['subjects']) // 10)]:
            X = np.load(os.path.join(cache, f'{d.get("source", "eeg_hp05")}_sub-{int(s):03d}.npy'), mmap_mode='r'); clamp.append(float(np.mean(np.abs(np.asarray(X[:, ::97])) >= 20)))
        out[d['name']] = dict(subjects=len(meta['subjects']), recordings=nrec, eeg_hours=eeg_s / 3600, unique_stimulus_hours=None if stim_s is None else stim_s / 3600,
                              excluded=excl, mean_bad_channels=bad, clamp_fraction_sampled=float(np.mean(clamp)), sample_rate=fs,
                              **(extra if d['kind'] != 'alice' else {}))
        print(f"{d['name']:12s} subjects {out[d['name']]['subjects']:3d} recordings {nrec:5d} EEG {eeg_s / 3600:7.2f} h stimuli "
              f"{(stim_s or 0) / 3600:6.2f} h excluded {sum(len(v) for v in excl.values())} clamp {np.mean(clamp):.4f}", flush=True)
    os.makedirs(os.path.join(ROOT, 'results/d099_ext'), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, 'results/d099_ext/summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
