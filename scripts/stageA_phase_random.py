"""Stage A pipeline control: multivariate phase-randomised copy of the stage A EEG cache.

Per subject and run, the same random Fourier phases are added to every channel, which keeps each channel's power spectrum
and the cross-channel covariance but destroys any time locking to the stimulus. A decoder trained and tested on this copy
through the unchanged pipeline must stay at chance; otherwise the pipeline (split, windows, features) leaks.

usage: python scripts/stageA_phase_random.py configs/stageA_prepare.yaml [--source eeg_hp05]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(cfg_path, src_name='eeg'):
    c = yaml.safe_load(open(cfg_path)); d = os.path.join(ROOT, c['out_dir']); meta = json.load(open(os.path.join(d, 'eeg_meta.json')))
    offs = meta['run_offsets']; rng = np.random.default_rng(20261010)
    for s in c['subjects']:
        src = np.load(os.path.join(d, f'{src_name}_sub-{s:03d}.npy'), mmap_mode='r')
        dst = np.lib.format.open_memmap(os.path.join(d, f'{src_name}_phaserand_sub-{s:03d}.npy'), 'w+', np.float32, src.shape)
        for r in range(len(offs) - 1):
            x = np.asarray(src[:, offs[r]:offs[r + 1]], np.float64); X = np.fft.rfft(x, axis=1)
            ph = rng.uniform(0, 2 * np.pi, X.shape[1]); ph[0] = 0
            if x.shape[1] % 2 == 0: ph[-1] = 0
            dst[:, offs[r]:offs[r + 1]] = np.fft.irfft(X * np.exp(1j * ph)[None], n=x.shape[1], axis=1).astype(np.float32)
        dst.flush(); print(f'sub-{s:03d} done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[sys.argv.index('--source') + 1] if '--source' in sys.argv else 'eeg')
