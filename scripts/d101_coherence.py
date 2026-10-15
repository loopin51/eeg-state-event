"""D-101 (b)(c): EEG-audio coherence, cross-correlation lags and post-silence onset responses on the raw 512-Hz ds004408 EEG.

No decoder and no EEG-based identification. EEG: BrainVision 512 Hz, cropped to the audio length (the recordings start with
the audio), CAR without BIDS-bad and flat channels. Audio: the run wav resampled to 512 Hz (waveform) and the Hilbert
envelope of the 16-kHz audio resampled to 512 Hz.
(b) magnitude-squared coherence (1-s Hann segments, 50 % overlap), pooled over runs per subject, averaged over subjects;
    null: the audio segment sequence circularly shifted by >= 30 s (100 draws, one per run and draw, same for every subject).
    cross-correlation (circular, normalised) of band-passed EEG and audio: waveform 20-60 and 70-250 Hz (lags -20..30 ms),
    envelope 1-60 Hz (lags -100..400 ms); null = the same circular cross-correlation read at lag + shift (>= 30 s).
(c) onsets of words preceded by >= 0.5 s of silence: 1-40 Hz evoked response and 70-250 Hz Hilbert amplitude, -50..300 ms;
    latency = first time >= 0 where the global value exceeds baseline mean + 3 SD for >= 5 ms.
Rules A_w / A_e / A_f and the decision table: docs/DECISIONS.md D-101 (applied by scripts/d101_report.py).

usage: python scripts/d101_coherence.py configs/d101_artifact.yaml [--subjects 2 3]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np, pandas as pd
from math import gcd
from scipy.io import wavfile
from scipy.signal import resample_poly, butter, sosfiltfilt, hilbert
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from d098_features import read_tier, SILENCE
FS = 512


def resamp(x, fs_in, fs_out): g = gcd(int(fs_in), int(fs_out)); return resample_poly(x, int(fs_out) // g, int(fs_in) // g, axis=-1)


def bp(x, band, fs=FS, order=4):
    lo, hi = band; hi = min(hi, fs / 2 * 0.98)
    return sosfiltfilt(butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos'), x, axis=-1)


def segments(x, n, hop):
    k = (x.shape[-1] - n) // hop + 1; idx = np.arange(n)[None] + hop * np.arange(k)[:, None]; w = np.hanning(n)
    return np.fft.rfft(x[..., idx] * w, axis=-1)                                     # [..., K, F]


def audio_signals(path):
    fs, w = wavfile.read(path); w = w.astype(np.float64); w = w.mean(1) if w.ndim > 1 else w
    wave = resamp(w, fs, FS); env = resamp(np.abs(hilbert(resamp(w, fs, 16000))), 16000, FS)
    return dict(wave=wave, env=env), len(w) / fs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--subjects', type=int, nargs='*'); ap.add_argument('--dry', action='store_true'); a = ap.parse_args(); t0 = time.time()
    c = yaml.safe_load(open(a.config)); b1 = yaml.safe_load(open(os.path.join(ROOT, c['b1_config']))); co = c['coherence']; oe = c['onset_erp']
    pool = a.subjects or sorted(s for f in b1['split']['subject_folds'] for s in f)
    meta = json.load(open(os.path.join(ROOT, b1['cache_dir'], 'eeg_meta.json'))); raw_root = os.path.join(ROOT, 'data/raw/ds004408')
    import mne; mne.set_log_level('ERROR')
    n, hop = co['nperseg'], co['nperseg'] // 2; freqs = np.fft.rfftfreq(n, 1 / FS); NB = co['n_shift']
    bands = co['bands_hz']; bmask = {k: (freqs >= v[0]) & (freqs <= v[1]) for k, v in bands.items()}
    xc = co['xcorr']; xbands = dict(**{f'wave_{k}': ('wave', v, xc['wave_lags_ms']) for k, v in xc['wave_bands_hz'].items()}, env_b1_60=('env', xc['env_band_hz'], xc['env_lags_ms']))
    lag_idx = {k: np.arange(int(np.floor(v[2][0] * FS / 1000)), int(np.ceil(v[2][1] * FS / 1000)) + 1) for k, v in xbands.items()}
    ew = [int(round(x * FS / 1000)) for x in oe['window_ms']]; et = np.arange(ew[0], ew[1] + 1)
    audio = {r: audio_signals(os.path.join(raw_root, 'stimuli', f'audio{r + 1:02d}.wav')) for r in range(20)}
    events = {}
    for r in range(20):
        iv = read_tier(os.path.join(raw_root, 'stimuli', f'audio{r + 1:02d}.TextGrid'), 'words'); ev = []
        for j, (x0, x1, t) in enumerate(iv):
            if t.strip() in SILENCE: continue
            gap_start = x0
            k = j - 1
            while k >= 0 and iv[k][2].strip() in SILENCE: gap_start = iv[k][0]; k -= 1
            if x0 - gap_start >= oe['min_gap_s'] and x0 * FS + ew[0] >= 0: ev.append(x0)
        events[r] = np.array(ev)
    print(f'audio and {sum(len(v) for v in events.values())} post-silence onsets ready ({time.time() - t0:.0f}s)', flush=True)
    shifts_seg, shifts_samp = {}, {}
    res = {'subjects': {}, 'freqs': freqs.tolist(), 'events': int(sum(len(v) for v in events.values()))}
    acc = {}; ch_names = None
    for s in pool:
        Sxa = {k: 0 for k in ('wave', 'env')}; Sxx = 0; Saa = {k: 0 for k in ('wave', 'env')}; Sxa_null = {k: 0 for k in ('wave', 'env')}
        XC = {k: 0 for k in xbands}; XCn = {k: 0 for k in xbands}; nrun = 0; ERP, HIGH, AENV = [], [], []
        bad_tsv = None
        for r in range(20):
            base = os.path.join(raw_root, f'sub-{s:03d}', 'eeg', f'sub-{s:03d}_task-listening_run-{r + 1:02d}')
            raw = mne.io.read_raw_brainvision(base + '_eeg.vhdr', preload=True); assert int(raw.info['sfreq']) == FS
            ch = pd.read_csv(base + '_channels.tsv', sep='\t'); bad = (ch['status'] == 'bad').to_numpy(); ch_names = list(ch['name']); assert raw.ch_names[:len(ch_names)] == ch_names and len(ch_names) == 128
            flat = np.array(meta['subjects'][str(s)]['flat'][r], bool); ok = ~(bad | flat)
            aud, dur = audio[r]; N = min(int(np.floor(dur * FS)), raw.n_times, len(aud['wave']), len(aud['env']))
            X = raw.get_data()[:, :N]; X = X - X[ok].mean(0, keepdims=True); X[~ok] = 0.0
            A = {k: aud[k][:N] for k in ('wave', 'env')}
            # (b) coherence
            Xs = segments(X, n, hop); K = Xs.shape[1]
            rng = np.random.default_rng([co['shift_seed'], r]); mseg = rng.integers(int(30 * FS / hop), K - int(30 * FS / hop), NB)
            Sxx = Sxx + (np.abs(Xs) ** 2).sum(1)
            for k in ('wave', 'env'):
                As = segments(A[k], n, hop); Sxa[k] = Sxa[k] + np.einsum('ckf,kf->cf', Xs, np.conj(As)); Saa[k] = Saa[k] + (np.abs(As) ** 2).sum(0)
                idx = (np.arange(K)[:, None] + mseg[None]) % K; Ar = np.conj(As[idx])          # [K, B, F]
                Sxa_null[k] = Sxa_null[k] + np.stack([Xs[:, :, f] @ Ar[:, :, f] for f in range(len(freqs))], -1).transpose(1, 0, 2)   # [B, C, F]
            # (b) cross-correlation lags
            rng2 = np.random.default_rng([co['shift_seed'], r, 1]); msamp = rng2.integers(int(30 * FS), N - int(30 * FS), NB)
            for key, (sig, band, _) in xbands.items():
                e = bp(X, band); e[~ok] = 0.0; au = bp(A[sig], band)
                full = np.fft.irfft(np.fft.rfft(e, axis=-1) * np.conj(np.fft.rfft(au))[None], n=N, axis=-1)
                full /= (np.linalg.norm(e, axis=-1, keepdims=True) * np.linalg.norm(au) + 1e-20)
                L = lag_idx[key]; XC[key] = XC[key] + full[:, L % N]; XCn[key] = XCn[key] + full[:, (L[None] + msamp[:, None]) % N].transpose(1, 0, 2)
            # (c) post-silence onset responses
            erp = bp(X, oe['bands_hz']['erp']); high = np.abs(hilbert(bp(X, oe['bands_hz']['high']), axis=-1)); envr = A['env']
            for t in events[r]:
                i = int(round(t * FS))
                if i + ew[1] >= N: continue
                ERP.append(erp[:, i + et]); HIGH.append(high[:, i + et]); AENV.append(envr[i + et])
            nrun += 1
            if a.dry:
                print('dry: shapes', {k: np.shape(v) for k, v in dict(Xs=Xs, Sxa_null=Sxa_null['wave'], XCn=XCn['env_b1_60'], ERP=np.shape(ERP)).items()}, f'N {N}, K {K}, events {len(ERP)} ({time.time() - t0:.0f}s)', flush=True); return
        coh = {k: np.abs(Sxa[k]) ** 2 / (Sxx * Saa[k][None] + 1e-30) for k in Sxa}
        coh_null = {k: np.abs(Sxa_null[k]) ** 2 / (Sxx[None] * Saa[k][None, None] + 1e-30) for k in Sxa}
        acc.setdefault('coh', []).append(coh); acc.setdefault('coh_null_band', []).append({k: np.stack([coh_null[k][..., m].mean(-1) for m in bmask.values()], -1) for k in coh})
        acc.setdefault('coh_null_spec_mean', []).append({k: coh_null[k].mean(1) for k in coh})
        acc.setdefault('xc', []).append({k: XC[k] / nrun for k in XC}); acc.setdefault('xc_null', []).append({k: XCn[k] / nrun for k in XCn})
        b0 = slice(0, -ew[0] + 1)
        acc.setdefault('erp', []).append(np.mean(ERP, 0)); acc.setdefault('high', []).append(np.mean(HIGH, 0)); acc.setdefault('aenv', []).append(np.mean(AENV, 0))
        print(f'sub-{s:03d}: {nrun} runs, {len(ERP)} onsets ({time.time() - t0:.0f}s)', flush=True)
    out_c = os.path.join(ROOT, c['cache']); os.makedirs(out_c, exist_ok=True)
    save = dict(ch_names=np.array(ch_names), freqs=freqs, subjects=np.array(pool), band_names=np.array(list(bands)), lag_ms={k: (lag_idx[k] * 1000 / FS).tolist() for k in xbands},
                erp_t_ms=(et * 1000 / FS))
    for k in ('wave', 'env'):
        save[f'coh_{k}'] = np.stack([x[k] for x in acc['coh']]); save[f'coh_null_band_{k}'] = np.stack([x[k] for x in acc['coh_null_band']])
        save[f'coh_null_spec_{k}'] = np.stack([x[k] for x in acc['coh_null_spec_mean']])
    for k in xbands: save[f'xc_{k}'] = np.stack([x[k] for x in acc['xc']]); save[f'xc_null_{k}'] = np.stack([x[k] for x in acc['xc_null']])
    save['erp'] = np.stack(acc['erp']); save['high'] = np.stack(acc['high']); save['aenv'] = np.stack(acc['aenv'])
    lagms = save.pop('lag_ms'); np.savez_compressed(os.path.join(out_c, 'coherence.npz'), **save)
    json.dump(dict(lag_ms=lagms, n_events=res['events'], subjects=pool, seconds=time.time() - t0), open(os.path.join(out_c, 'coherence_meta.json'), 'w'), indent=1)
    print(f'saved ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
