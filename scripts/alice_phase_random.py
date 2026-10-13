"""D-092 pipeline control: phase-randomised copies of the Alice caches of both arms (eeg_hp05, reve).

Per subject and present segment the same random Fourier phases are added to every channel (power spectra and cross-channel
covariance kept, time locking to the stimulus destroyed), as scripts/stageA_phase_random.py. Missing segments stay zero.

usage: python scripts/alice_phase_random.py configs/alice_prepare.yaml [--sources eeg_hp05,reve]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(cfg, sources):
    c = yaml.safe_load(open(cfg)); d = os.path.join(ROOT, c['out_dir']); meta = json.load(open(os.path.join(d, 'eeg_meta.json')))
    for src in sources:
        offs = meta['sources'][src]['run_offsets']; rng = np.random.default_rng(20261012)
        for s, info in meta['subjects'].items():
            x0 = np.load(os.path.join(d, f'{src}_sub-{int(s):03d}.npy')); y = np.zeros_like(x0); miss = [k - 1 for k in info['missing_segments']]
            for r in range(len(offs) - 1):
                ph = None
                x = x0[:, offs[r]:offs[r + 1]].astype(np.float64); X = np.fft.rfft(x, axis=1)
                ph = rng.uniform(0, 2 * np.pi, X.shape[1]); ph[0] = 0
                if x.shape[1] % 2 == 0: ph[-1] = 0
                if r in miss: continue
                y[:, offs[r]:offs[r + 1]] = np.fft.irfft(X * np.exp(1j * ph)[None], n=x.shape[1], axis=1).astype(np.float32)
            np.save(os.path.join(d, f'{src}_phaserand_sub-{int(s):03d}.npy'), y)
        print(f'{src}: {len(meta["subjects"])} subjects done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[sys.argv.index('--sources') + 1].split(',') if '--sources' in sys.argv else ['eeg_hp05', 'reve'])
