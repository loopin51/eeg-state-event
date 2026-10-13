"""ds004408 (natural speech listening, Di Liberto 2015 / Broderick 2018): preprocessed EEG and stimulus features (D-059).

EEG per run: bad channels (channels.tsv status) interpolated on the biosemi128 montage, common-average reference,
1-8 Hz zero-phase band-pass, resampled to 64 Hz and to 20 Hz, cut to the audio length (recording start = audio start).
Stimulus per audio segment: broadband envelope (|hilbert| of the wav, 8 Hz low-pass) at 64 Hz and 20 Hz, 8 log-band
energies at 20 Hz (STFT, log-spaced 100-8000 Hz), word intervals from the TextGrid "words" tier.
Run XX is paired with stimuli/audioXX (README: one recording per audio segment, starts aligned).
Writes data/cache/ds004408/stim.npz and sub-XXX.npz (object arrays, one entry per run).

usage: python scripts/prepare_ds004408.py [--subjects 1 ... 19]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import re, glob, time, argparse, numpy as np, pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, hilbert, resample_poly, stft

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data/raw/ds004408'); OUT = os.path.join(ROOT, 'data/cache/ds004408'); N_RUNS = 20


def objarr(items):
    """1-D object array without numpy trying to broadcast equal leading dimensions."""
    out = np.empty(len(items), dtype=object)
    for i, x in enumerate(items): out[i] = x
    return out


def textgrid_words(path):
    """[(xmin, xmax, word)] of the 'words' tier, silences dropped."""
    txt = open(path, encoding='utf-8').read(); tiers = re.split(r'item \[\d+\]:', txt)[1:]
    tier = next(t for t in tiers if re.search(r'name = "words"', t))
    out = []
    for a, b, w in re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', tier):
        w = w.strip()
        if w and w.lower() not in ('sil', 'sp', '<sil>', ''): out.append((float(a), float(b), w))
    return out


def stimulus(k):
    fs, w = wavfile.read(os.path.join(RAW, 'stimuli', f'audio{k:02d}.wav')); w = w.astype(np.float64)
    if w.ndim > 1: w = w.mean(1)
    w /= np.abs(w).max() + 1e-12; dur = len(w) / fs
    env = np.abs(hilbert(w)); env = sosfiltfilt(butter(4, 8, btype='low', fs=fs, output='sos'), env)
    env64 = resample_poly(env, 64, fs); env20 = resample_poly(env, 20, fs)
    f, t, Z = stft(w, fs=fs, nperseg=int(0.05 * fs), noverlap=int(0.05 * fs) - int(fs / 20))
    edges = np.geomspace(100, min(8000, fs / 2 - 1), 9); P = np.abs(Z) ** 2
    bands = np.stack([np.log(P[(f >= lo) & (f < hi)].mean(0) + 1e-12) for lo, hi in zip(edges[:-1], edges[1:])])   # [8, frames] ~20 Hz
    words = textgrid_words(os.path.join(RAW, 'stimuli', f'audio{k:02d}.TextGrid'))
    return dict(dur=dur, env64=env64, env20=env20, bands20=bands, words=words)


def run_eeg(sub, k, dur):
    import mne
    base = os.path.join(RAW, f'sub-{sub:03d}', 'eeg', f'sub-{sub:03d}_task-listening_run-{k:02d}')
    raw = mne.io.read_raw_brainvision(base + '_eeg.vhdr', preload=True, verbose=False)
    ch = pd.read_csv(base + '_channels.tsv', sep='\t'); raw.info['bads'] = ch.loc[ch.status == 'bad', 'name'].tolist()
    raw.set_montage('biosemi128', on_missing='raise', verbose=False)
    if raw.info['bads']: raw.interpolate_bads(reset_bads=True, verbose=False)
    raw.set_eeg_reference('average', verbose=False)
    x = raw.get_data(); fs = raw.info['sfreq']; n = int(dur * fs)
    assert x.shape[1] >= n - 2, f'sub-{sub:03d} run-{k:02d}: EEG shorter than audio ({x.shape[1] / fs:.1f} < {dur:.1f} s)'
    x = sosfiltfilt(butter(4, [1, 8], btype='band', fs=fs, output='sos'), x[:, :n], axis=-1)
    return (resample_poly(x, 64, int(fs), axis=-1).astype(np.float32), resample_poly(x, 20, int(fs), axis=-1).astype(np.float32),
            int((ch.status == 'bad').sum()), raw.ch_names)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 20))); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    S = [stimulus(k) for k in range(1, N_RUNS + 1)]
    np.savez(os.path.join(OUT, 'stim.npz'), dur=np.array([s['dur'] for s in S]), env64=objarr([s['env64'] for s in S]),
             env20=objarr([s['env20'] for s in S]), bands20=objarr([s['bands20'] for s in S]), words=objarr([objarr(s['words']) for s in S]))
    print(f'stimuli: {N_RUNS} segments, {sum(s["dur"] for s in S) / 60:.1f} min, {sum(len(s["words"]) for s in S)} words ({time.time() - t0:.0f}s)', flush=True)
    for sub in a.subjects:
        e64, e20, nbad, names = [], [], [], None
        for k in range(1, N_RUNS + 1):
            x64, x20, b, nm = run_eeg(sub, k, S[k - 1]['dur']); e64.append(x64); e20.append(x20); nbad.append(b)
            assert names is None or nm == names, 'channel order differs between runs'; names = nm
        np.savez(os.path.join(OUT, f'sub-{sub:03d}.npz'), eeg64=objarr(e64), eeg20=objarr(e20), n_bad=np.array(nbad),
                 ch_names=np.array(names))
        print(f'sub-{sub:03d}: 20 runs, bad channels per run {nbad} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
