"""Pipeline control for the recording-layout caches of scripts/ext_prepare.py (D-099; first use: DTU for C2 prerequisite 1).

Same rule as scripts/stageA_phase_random.py (ds004408 A4 eeg_hp05_phaserand) and scripts/alice_phase_random.py: per subject
and per cached recording (DTU: one two-talker trial; the recording takes the place of the ds004408 run) the same random Fourier
phases are added to every channel, which keeps each channel's power spectrum and the cross-channel covariance but destroys any
time locking to the stimulus. One generator default_rng(seed) is drawn in subject order, then recording order; excluded
recordings hold no samples and draw nothing. CAR, baseline and clamp stay at load time, as for ds004408.
Output: <cache>/<source>_phaserand_sub-XXX.npy (float32, same shape as <source>_sub-XXX.npy).

usage: python scripts/ext_phase_random.py --cache data/cache/dtu [--source eeg_hp05] [--seed 20261015]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import json, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cache', required=True); ap.add_argument('--source', default='eeg_hp05')
    ap.add_argument('--seed', type=int, default=20261015); a = ap.parse_args()
    d = a.cache if os.path.isabs(a.cache) else os.path.join(ROOT, a.cache); meta = json.load(open(os.path.join(d, 'eeg_meta.json')))
    rng = np.random.default_rng(a.seed)
    for s in sorted(meta['subjects'], key=int):
        src = np.load(os.path.join(d, f'{a.source}_sub-{int(s):03d}.npy'), mmap_mode='r')
        dst = np.lib.format.open_memmap(os.path.join(d, f'{a.source}_phaserand_sub-{int(s):03d}.npy'), 'w+', np.float32, src.shape)
        n = 0
        for r in meta['subjects'][s]:
            if 'excluded' in r: continue
            o, L = r['offset'], r['length']; x = np.asarray(src[:, o:o + L], np.float64); X = np.fft.rfft(x, axis=1)
            ph = rng.uniform(0, 2 * np.pi, X.shape[1]); ph[0] = 0
            if L % 2 == 0: ph[-1] = 0
            dst[:, o:o + L] = np.fft.irfft(X * np.exp(1j * ph)[None], n=L, axis=1).astype(np.float32); n += L
        assert n == src.shape[1], f'sub-{int(s):03d}: recordings cover {n} of {src.shape[1]} samples'
        dst.flush(); print(f'sub-{int(s):03d} done', flush=True)
    json.dump(dict(rule='scripts/stageA_phase_random.py per cached recording', source=a.source, seed=a.seed),
              open(os.path.join(d, f'{a.source}_phaserand.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
