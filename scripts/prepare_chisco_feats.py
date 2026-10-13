"""Chisco (Zhang et al., 2024, OpenNeuro ds005170): per-trial feature sets for the day-held-out calibration curve (D-047).

One npz per subject in data/cache/chisco_feats/. Trials = imagine epochs of the preprocessed derivative (fif, 500 Hz,
5.0-8.3 s after the reading onset); runs without an imagine fif are skipped. Each trial is matched to its raw EDF trigger
(1000 Hz, |raw - 2*fif| <= 2 samples) and needs the raw window -1.0..8.3 s inside the file. Trials failing either
(truncated raw EDF files, D-047 addendum) keep DE_IMAG / ORDER / LEN, get NaN raw-derived features and raw_ok = False.
Feature sets:
  DE_IMAG      derivative imagine window, 122 EEG ch, 6-band DE (primary)
  DE_IMAG_RAW  raw EDF, 1-100 Hz band-pass + 50 Hz notch, 5.0-8.3 s, 122 EEG ch
  DE_READ      same raw processing, reading window 0-5 s
  EMG / EOG    raw EMG (1 ch) / VEO+HEO, same processing, 5.0-8.3 s
  PRE          same raw processing, -1.0-0 s before reading onset (leak meter c)
  DC           unfiltered raw, 122 EEG ch mean and slope over -1.0-8.3 s (leak meter b)
  ORDER        trial onset time in run (s, from the derivative events), run position in day, and squares (leak meter a)
  LEN          sentence length with / without punctuation (confound reference)

usage: python scripts/prepare_chisco_feats.py [--subjects 1 2 3 4 5]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, glob, re, time, argparse, unicodedata, numpy as np
from scipy.signal import welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data/raw/chisco'); OUT = os.path.join(ROOT, 'data/cache/chisco_feats')
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45), (45, 100)]
NON_EEG = {'VEO', 'HEO', 'Trigger'}
TRIAL_CODE = 65380   # trial-onset trigger; two runs also carry one stray 1-sample code (sub-04 run 043, sub-05 run 025)


def de(x, fs):
    """x [n, C, T] -> [n, C*6] log band power."""
    f, P = welch(x, fs=fs, nperseg=min(x.shape[-1], int(fs)), axis=-1)
    return np.log(np.stack([P[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in BANDS], -1) + 1e-30).reshape(len(x), -1)


def run_days(sub):
    """run id string -> (day index, path of raw EDF)."""
    out = {}
    for di, d in enumerate(sorted(glob.glob(os.path.join(RAW, f'sub-{sub:02d}', 'ses-*')))):
        for f in glob.glob(os.path.join(d, 'eeg', '*_eeg.edf')):
            out[re.search(r'run-(\d+)', f).group(1)] = (di, f)
    return out


def n_chars(s, punct):
    return sum(1 for ch in s if punct or not unicodedata.category(ch).startswith('P'))


def subject(sub, textmap):
    import mne
    days = run_days(sub); feats = {k: [] for k in ['DE_IMAG', 'DE_IMAG_RAW', 'DE_READ', 'EMG', 'EOG', 'PRE', 'DC', 'ORDER', 'LEN']}
    meta = {k: [] for k in ['label', 'day', 'run', 'raw_ok']}; dropped = 0; skipped = []; eeg = None
    runs = sorted(days, key=int); pos_in_day = {}
    for r in runs: pos_in_day[r] = sum(1 for q in runs if days[q][0] == days[r][0] and int(q) < int(r))
    raw_sets = ['DE_IMAG_RAW', 'DE_READ', 'EMG', 'EOG', 'PRE', 'DC']
    for r in runs:
        day, edf = days[r]
        fif = os.path.join(RAW, 'derivatives/preprocessed_fif', f'sub-{sub:02d}', 'eeg', f'sub-{sub:02d}_task-imagine_run-{r}_eeg.fif')
        if not os.path.exists(fif): skipped.append(r); continue
        e = mne.read_epochs(fif, verbose=False)
        if eeg is None: eeg = [c for c in e.ch_names if c not in NON_EEG]
        assert [c for c in e.ch_names if c not in NON_EEG] == eeg, f'channel order differs in run {r}'
        x = e.get_data(picks=eeg); fs = e.info['sfreq']; words = e.metadata['Word'].to_numpy(); n = len(x)
        feats['DE_IMAG'].append(de(x, fs))
        t_on = e.events[:, 0] / fs; rpos = np.full(n, float(pos_in_day[r]))
        feats['ORDER'].append(np.c_[t_on, t_on ** 2, rpos, rpos ** 2])
        feats['LEN'].append(np.array([[n_chars(w, True), n_chars(w, False)] for w in words], dtype=float))
        raw = mne.io.read_raw_edf(edf, preload=True, verbose=False); rfs = raw.info['sfreq']
        ev = mne.find_events(raw, stim_channel='Trigger', shortest_event=1, verbose=False); ev = ev[ev[:, 2] == TRIAL_CODE, 0]
        idx = np.array([np.argmin(np.abs(ev - 2 * s)) for s in e.events[:, 0]]) if len(ev) else np.zeros(n, int)
        onset = ev[idx] if len(ev) else np.zeros(n, int)
        ok = (np.abs(onset - 2 * e.events[:, 0]) <= 2) & (onset - int(1.0 * rfs) >= 0) & (onset + int(8.3 * rfs) <= raw.n_times)
        dropped += int((~ok).sum())
        missing = [c for c in eeg + ['EMG', 'VEO', 'HEO'] if c not in raw.ch_names]
        assert not missing, f'raw run {r} lacks {missing}'
        blocks = {k: np.full((n, d), np.nan) for k, d in [('DE_IMAG_RAW', 6 * len(eeg)), ('DE_READ', 6 * len(eeg)), ('EMG', 6), ('EOG', 12), ('PRE', 6 * len(eeg)), ('DC', 2 * len(eeg))]}
        if ok.any():
            on = onset[ok]; ne = len(eeg)
            dc_raw = raw.get_data(picks=eeg)
            filt = raw.copy().pick(eeg + ['EMG', 'VEO', 'HEO']).filter(1, 100, verbose=False).notch_filter(50, verbose=False).get_data()

            def seg(a, t0, t1):
                return np.stack([a[:, o + int(t0 * rfs): o + int(t1 * rfs)] for o in on])
            imag = seg(filt, 5.0, 8.3)
            blocks['DE_IMAG_RAW'][ok] = de(imag[:, :ne], rfs); blocks['EMG'][ok] = de(imag[:, ne:ne + 1], rfs)
            blocks['EOG'][ok] = de(imag[:, ne + 1:], rfs); blocks['DE_READ'][ok] = de(seg(filt, 0.0, 5.0)[:, :ne], rfs)
            blocks['PRE'][ok] = de(seg(filt, -1.0, 0.0)[:, :ne], rfs)
            d = seg(dc_raw, -1.0, 8.3); tt = np.arange(d.shape[-1]) / rfs
            slope = np.polyfit(tt, d.transpose(2, 0, 1).reshape(len(tt), -1), 1)[0].reshape(len(d), ne)
            blocks['DC'][ok] = np.c_[d.mean(-1), slope]
            del dc_raw, filt
        for k in raw_sets: feats[k].append(blocks[k])
        meta['label'].append(np.array([textmap[w] for w in words])); meta['day'].append(np.full(n, day))
        meta['run'].append(np.full(n, int(r))); meta['raw_ok'].append(ok)
        del raw
    out = {k: np.concatenate(v).astype(np.float32) for k, v in feats.items()}
    out.update({k: np.concatenate(v) for k, v in meta.items()})
    return out, dropped, skipped, eeg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--subjects', type=int, nargs='+', default=[1, 2, 3, 4, 5]); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); textmap = json.load(open(os.path.join(RAW, 'json/textmaps.json'))); t0 = time.time()
    for sub in a.subjects:
        F, dropped, skipped, eeg = subject(sub, textmap)
        np.savez(os.path.join(OUT, f'sub-{sub:02d}.npz'), eeg_channels=np.array(eeg), dropped=dropped, skipped_runs=np.array(skipped), **F)
        n_day = np.bincount(F['day'])
        print(f'sub-{sub:02d}: trials {len(F["label"])}, raw_ok {int(F["raw_ok"].sum())}, dropped {dropped}, skipped runs {skipped}, days {n_day.tolist()}, classes {len(np.unique(F["label"]))}, '
              f'dims ' + ' '.join(f'{k}={F[k].shape[1]}' for k in ['DE_IMAG', 'DE_IMAG_RAW', 'DE_READ', 'EMG', 'EOG', 'PRE', 'DC', 'ORDER', 'LEN']) +
              f' ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
