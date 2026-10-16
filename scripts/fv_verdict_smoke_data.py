"""F-V verdict smoke (D-112 amendment 3): synthetic raw test files in the THINGS-EEG2 layout, for V1-V3 code tests only.

Writes OUT/raw/sub-XX/ses-YY/raw_eeg_test.npy (pickled dict like the real files: raw_eeg_data float64 [64, N], ch_names = the
cache's 63 channels + 'stim', ch_types, sfreq 1000, highpass, lowpass) with 200 test codes x `reps` repetitions plus catch
trials (99999) at a 200-ms SOA in random order, Gaussian EEG (40 uV), and OUT/seal.tsv (path, bytes, sha256) in the seal-table
format. No real test file is opened.

usage: python scripts/fv_verdict_smoke_data.py OUT --subjects 1 [--reps 20]
"""
import os, sys, json, argparse, hashlib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--subjects', type=int, nargs='+', required=True); ap.add_argument('--reps', type=int, default=20); a = ap.parse_args()
    ch = json.load(open(os.path.join(ROOT, 'data/cache/things_eeg2/channels.json')))['ch_names']; rows = ['path\tbytes\tsha256']
    for s in a.subjects:
        for e in (1, 2, 3, 4):
            rng = np.random.default_rng([s, e, 7]); codes = rng.permutation(np.r_[np.repeat(np.arange(1, 201), a.reps), np.full(100, 99999)])
            N = 2000 + 200 * len(codes) + 2000; x = rng.standard_normal((64, N)) * 4e-5; x[63] = 0
            for i, k in enumerate(codes): x[63, 2000 + 200 * i: 2000 + 200 * i + 10] = k
            d = os.path.join(a.out, 'raw', f'sub-{s:02d}', f'ses-{e:02d}'); os.makedirs(d, exist_ok=True); p = os.path.join(d, 'raw_eeg_test.npy')
            np.save(p, dict(raw_eeg_data=x, ch_names=ch + ['stim'], ch_types=['eeg'] * 63 + ['stim'], sfreq=1000, highpass=0.01, lowpass=100.0), allow_pickle=True)
            rows.append(f'things_eeg2/raw-eeg/sub-{s:02d}/ses-{e:02d}/raw_eeg_test.npy\t{os.path.getsize(p)}\t{hashlib.sha256(open(p, "rb").read()).hexdigest()}')
    open(os.path.join(a.out, 'seal.tsv'), 'w').write('\n'.join(rows) + '\n'); print('wrote', len(rows) - 1, 'synthetic test files', flush=True)


if __name__ == '__main__':
    main()
