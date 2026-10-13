"""Chisco waveform cache for the neural-network readouts (D-053). One folder per subject in data/cache/chisco_epochs/.

Same trial set and alignment as scripts/prepare_chisco_feats.py (imagine epochs of the preprocessed derivative, raw EDF
trigger within 2 samples, raw window -1.0..8.3 s inside the file); only raw_ok trials are kept, in the same order as
the feature npz, so trial i here is trial i of the feature file restricted to raw_ok.
Arrays (float16, 250 Hz):
  imag.npy   [n, 122, 825]  derivative imagine window 5.0-8.3 s, EEG channels, microvolts
  pre.npy    [n, 122, 250]  raw EDF, 1-100 Hz band-pass + 50 Hz notch, -1.0-0 s before reading onset, microvolts
  exg.npy    [n, 3, 825]    same raw processing, EMG + VEO + HEO, imagine window, float32, raw EDF units (these three
                            channels are already stored in microvolts, so they are not scaled; fp16 overflowed, D-056)
meta.npz: label, day, run.

usage: python scripts/prepare_chisco_epochs.py [--subjects 1 2 3 4 5]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, numpy as np
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_chisco_feats import RAW, NON_EEG, TRIAL_CODE, run_days

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, 'data/cache/chisco_epochs')


def subject(sub, textmap, out, only=None):
    import mne
    days = run_days(sub); eeg = None; n_tot = 0; parts = {'imag': [], 'pre': [], 'exg': []}; meta = {'label': [], 'day': [], 'run': []}
    for r in sorted(days, key=int):
        day, edf = days[r]
        fif = os.path.join(RAW, 'derivatives/preprocessed_fif', f'sub-{sub:02d}', 'eeg', f'sub-{sub:02d}_task-imagine_run-{r}_eeg.fif')
        if not os.path.exists(fif): continue
        e = mne.read_epochs(fif, verbose=False)
        if eeg is None: eeg = [c for c in e.ch_names if c not in NON_EEG]
        raw = mne.io.read_raw_edf(edf, preload=True, verbose=False); rfs = raw.info['sfreq']
        ev = mne.find_events(raw, stim_channel='Trigger', shortest_event=1, verbose=False); ev = ev[ev[:, 2] == TRIAL_CODE, 0]
        idx = np.array([np.argmin(np.abs(ev - 2 * s)) for s in e.events[:, 0]]) if len(ev) else np.zeros(len(e), int)
        onset = ev[idx] if len(ev) else np.zeros(len(e), int)
        ok = (np.abs(onset - 2 * e.events[:, 0]) <= 2) & (onset - int(1.0 * rfs) >= 0) & (onset + int(8.3 * rfs) <= raw.n_times)
        n_tot += len(e)
        if not ok.any(): continue
        x = e.get_data(picks=eeg)[ok] * 1e6
        parts['imag'].append(resample_poly(x, 1, 2, axis=-1).astype(np.float16))
        filt = raw.copy().pick(eeg + ['EMG', 'VEO', 'HEO']).filter(1, 100, verbose=False).notch_filter(50, verbose=False).get_data()
        filt[:len(eeg)] *= 1e6   # EEG volts -> microvolts; EMG/VEO/HEO already in microvolts
        on = onset[ok]; ne = len(eeg)
        seg = lambda a, t0, t1: np.stack([a[:, o + int(t0 * rfs): o + int(t1 * rfs)] for o in on])
        parts['pre'].append(resample_poly(seg(filt[:ne], -1.0, 0.0), 1, 4, axis=-1).astype(np.float16))
        parts['exg'].append(resample_poly(seg(filt[ne:], 5.0, 8.3), 1, 4, axis=-1).astype(np.float32))
        w = e.metadata['Word'].to_numpy()[ok]
        meta['label'].append(np.array([textmap[s] for s in w])); meta['day'].append(np.full(ok.sum(), day)); meta['run'].append(np.full(ok.sum(), int(r)))
        del raw, filt
    os.makedirs(out, exist_ok=True)
    for k, v in parts.items():
        if only is None or k in only: np.save(os.path.join(out, f'{k}.npy'), np.concatenate(v))
    M = {k: np.concatenate(v) for k, v in meta.items()}; np.savez(os.path.join(out, 'meta.npz'), eeg_channels=np.array(eeg), **M)
    return M, n_tot


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--subjects', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--only', nargs='+', choices=['imag', 'pre', 'exg'], default=None); a = ap.parse_args()
    textmap = json.load(open(os.path.join(RAW, 'json/textmaps.json'))); t0 = time.time()
    for sub in a.subjects:
        out = os.path.join(OUT, f'sub-{sub:02d}'); M, n_tot = subject(sub, textmap, out, a.only)
        F = np.load(os.path.join(ROOT, 'data/cache/chisco_feats', f'sub-{sub:02d}.npz'))   # must match the feature file's raw_ok trials
        same = bool(np.array_equal(F['label'][F['raw_ok']], M['label']) and np.array_equal(F['day'][F['raw_ok']], M['day']))
        assert np.isfinite(np.load(os.path.join(out, 'exg.npy'), mmap_mode='r')).all(), 'non-finite exg'
        print(f'sub-{sub:02d}: kept {len(M["label"])} of {n_tot}, matches feature file: {same}, '
              f'imag {np.load(os.path.join(out, "imag.npy"), mmap_mode="r").shape} ({time.time() - t0:.0f}s)', flush=True)
        assert same, 'trial set differs from the feature cache'


if __name__ == '__main__':
    main()
