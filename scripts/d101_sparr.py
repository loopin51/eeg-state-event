"""D-101 amendment 1 (report only): the SparrKULee insert-phone artefact control recording (sub-002 ses-shortstories01
task-listeningActive run-01, stimulus audiobook_1_artefact) through the D-101 (b) coherence / lag / topography measures.

Input: the aligned file from scripts/d101_sparr_artefact.py (eeg-a4): 64 Biosemi EEG channels in V at 512 Hz on the stimulus
clock (EEG sample k = audio sample 93.75 k), the 48-kHz stimulus waveform. Same steps as scripts/d101_coherence.py: CAR over
all 64 channels (no status marks), audio waveform resampled to 512 Hz, Hilbert envelope of the 16-kHz audio at 512 Hz,
magnitude-squared coherence (1-s Hann, 50 %), 100 circular audio shifts >= 30 s, circular normalised cross-correlation of
band-passed signals. Only difference: the BDF signal is unreferenced (CMS/DRL offsets of tens of mV), so each channel is
linearly detrended over the whole recording and each 1-s segment is mean-removed before the Hann window.
Not used by rules 1-4; the result is a reference for what an acoustic artefact looks like on these measures.

usage: python scripts/d101_sparr.py configs/d101_artifact.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np
from scipy.signal import hilbert, detrend
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from d101_coherence import resamp, bp, FS


def segments(x, n, hop):
    k = (x.shape[-1] - n) // hop + 1; idx = np.arange(n)[None] + hop * np.arange(k)[:, None]; s = x[..., idx]
    return np.fft.rfft((s - s.mean(-1, keepdims=True)) * np.hanning(n), axis=-1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); t0 = time.time()
    c = yaml.safe_load(open(a.config)); co = c['coherence']; xc = co['xcorr']; inst = co['instantaneous_ms']
    src = os.path.join(ROOT, c['cache'], 'sparrkulee_artefact'); Z = np.load(os.path.join(src, 'sparrkulee_artefact.npz'))
    meta = json.load(open(os.path.join(src, 'sparrkulee_artefact.json'))); assert int(Z['fs']) == FS and abs(meta['audio_samples_per_eeg_sample'] - 93.75) < 1e-9
    ch = [str(x) for x in Z['ch_names']]; X = detrend(Z['eeg'].astype(np.float64), axis=-1); X = X - X.mean(0, keepdims=True)
    w = Z['audio'].astype(np.float64); w = w.mean(1) if w.ndim > 1 else w; afs = int(Z['audio_fs'])
    A = dict(wave=resamp(w, afs, FS), env=resamp(np.abs(hilbert(resamp(w, afs, 16000))), 16000, FS))
    N = min(X.shape[1], len(A['wave']), len(A['env'])); X = X[:, :N]; A = {k: v[:N] for k, v in A.items()}
    n, hop = co['nperseg'], co['nperseg'] // 2; freqs = np.fft.rfftfreq(n, 1 / FS); NB = co['n_shift']
    bmask = {k: (freqs >= v[0]) & (freqs <= v[1]) for k, v in co['bands_hz'].items()}
    Xs = segments(X, n, hop); K = Xs.shape[1]; Sxx = (np.abs(Xs) ** 2).sum(1)
    rng = np.random.default_rng([co['shift_seed'], 99]); mseg = rng.integers(int(30 * FS / hop), K - int(30 * FS / hop), NB)
    out = dict(recording=meta['recording'], stimulus=meta['stimulus'], seconds_used=N / FS, n_segments=int(K), n_channels=len(ch), coherence={}, xcorr={})
    import mne; pos = mne.channels.make_standard_montage('biosemi64').get_positions()['ch_pos']
    spec = {}
    for k in ('wave', 'env'):
        As = segments(A[k], n, hop); Saa = (np.abs(As) ** 2).sum(0); coh = np.abs(np.einsum('ckf,kf->cf', Xs, np.conj(As))) ** 2 / (Sxx * Saa[None] + 1e-30)
        Ar = np.conj(As[(np.arange(K)[:, None] + mseg[None]) % K])
        Sn = np.stack([Xs[:, :, f] @ Ar[:, :, f] for f in range(len(freqs))], -1).transpose(1, 0, 2); cn = np.abs(Sn) ** 2 / (Sxx[None] * Saa[None, None] + 1e-30)
        spec[k] = (coh.mean(0), cn.mean(1).mean(0)); out['coherence'][k] = {}
        for b, m in bmask.items():
            ob = coh[:, m].mean(1); nb = cn[:, :, m].mean(-1); exc = ob - nb.mean(0); top = np.argsort(-exc)[:10]
            out['coherence'][k][b] = dict(global_obs=float(ob.mean()), global_null_p95=float(np.percentile(nb.mean(1), 95)), global_null_mean=float(nb.mean()),
                                          chmax_obs=float(ob.max()), chmax_null_p95=float(np.percentile(nb.max(1), 95)), global_above_null=bool(ob.mean() > np.percentile(nb.mean(1), 95)),
                                          top10=[dict(ch=ch[i], excess=float(exc[i]), xyz_m=[round(float(v), 4) for v in pos[ch[i]]] if ch[i] in pos else None) for i in top])
    xbands = dict(**{f'wave_{k}': ('wave', v, xc['wave_lags_ms']) for k, v in xc['wave_bands_hz'].items()}, env_b1_60=('env', xc['env_band_hz'], xc['env_lags_ms']))
    rng2 = np.random.default_rng([co['shift_seed'], 99, 1]); msamp = rng2.integers(int(30 * FS), N - int(30 * FS), NB); curves = {}
    for key, (sig, band, lags) in xbands.items():
        L = np.arange(int(np.floor(lags[0] * FS / 1000)), int(np.ceil(lags[1] * FS / 1000)) + 1); lag_ms = L * 1000 / FS
        e = bp(X, band); au = bp(A[sig], band)
        full = np.fft.irfft(np.fft.rfft(e, axis=-1) * np.conj(np.fft.rfft(au))[None], n=N, axis=-1); full /= (np.linalg.norm(e, axis=-1, keepdims=True) * np.linalg.norm(au) + 1e-20)
        g = np.sqrt((full[:, L % N] ** 2).mean(0)); gn = np.sqrt((full[:, (L[None] + msamp[:, None]) % N] ** 2).mean(0))      # [L], [B, L]
        pk = int(np.argmax(g)); d = dict(peak_lag_ms=float(lag_ms[pk]), peak=float(g[pk]), null_peak_p95=float(np.percentile(gn.max(1), 95)))
        chpk = np.abs(full[:, L % N]).max(1); top = np.argsort(-chpk)[:10]; d['top10'] = [dict(ch=ch[i], peak_abs=float(chpk[i])) for i in top]
        if key.startswith('env'):
            z0 = np.abs(lag_ms) <= inst['env']; late = lag_ms >= 50
            d.update(v0=float(g[z0].max()), v0_null_p95=float(np.percentile(gn[:, z0].max(1), 95)), late_max=float(g[late].max()), late_peak_lag_ms=float(lag_ms[late][np.argmax(g[late])]))
        out['xcorr'][key] = d; curves[key] = (lag_ms, g)
    cw, xw = out['coherence']['wave']['b20_60'], out['xcorr']['wave_b20_60']; e = out['xcorr']['env_b1_60']
    out['flags_same_definition'] = dict(A_w=bool(cw['global_above_null'] and abs(xw['peak_lag_ms']) <= inst['wave'] and xw['peak'] > xw['null_peak_p95']),
                                        A_e=bool(e['v0'] > e['v0_null_p95'] and e['v0'] > e['late_max']),
                                        A_f=bool(out['coherence']['wave']['b70_250']['global_above_null'] and abs(out['xcorr']['wave_b70_250']['peak_lag_ms']) <= inst['wave']))
    od = os.path.join(ROOT, c['out']); os.makedirs(od, exist_ok=True); json.dump(out, open(os.path.join(od, 'sparrkulee_artefact.json'), 'w'), indent=1, ensure_ascii=False)
    np.savez_compressed(os.path.join(ROOT, c['cache'], 'sparr_coherence.npz'), freqs=freqs, **{f'coh_{k}': v[0] for k, v in spec.items()}, **{f'coh_null_{k}': v[1] for k, v in spec.items()},
                        **{f'xc_{k}': v[1] for k, v in curves.items()}, **{f'lag_{k}': v[0] for k, v in curves.items()})
    print(json.dumps(dict(flags=out['flags_same_definition'], coh={k: {b: (round(v['global_obs'], 5), round(v['global_null_p95'], 5), [t['ch'] for t in v['top10'][:5]]) for b, v in out['coherence'][k].items()} for k in out['coherence']},
                          xcorr={k: {kk: vv for kk, vv in v.items() if kk != 'top10'} for k, v in out['xcorr'].items()}), ensure_ascii=False, indent=1), flush=True)
    print(f'done ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
