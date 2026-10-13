"""ds007808 listeningcovert: are the covert-imagery windows silent? (diagnostic, not a decoding result)

For each run, band-limited RMS of the speaking microphone, the listening-audio monitor, upper/lower-lip EMG and EOG
(all bipolar pairs) is computed in covert, listening and overt windows and in rest windows (inter-trial gaps).
An overt-speech run (task-speechopen) of the same participant serves as the positive control.
Reports per-run medians, covert/rest and listening/rest ratios, the fraction of covert windows above the rest quantile,
and onset-aligned envelopes (figure). If the participant vocalised during "covert" trials, the microphone and lip EMG
rise in covert windows relative to rest while the listening-audio monitor stays at rest level.

usage: python scripts/ds007808_vocal_check.py configs/ds007808_vocal_check.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, yaml, numpy as np, pandas as pd, mne
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COPPER, TEAL, PURPLE, INK, PAPER = '#B5652D', '#2E6E6B', '#5B4A86', '#182129', '#EEF0EC'
SIGNALS = [('mic', 'mic'), ('stim', 'stim'), ('emg_upper', 'emg'), ('emg_lower', 'emg'), ('eog', 'eog')]


def band(x, fs, lo, hi, notch):
    if notch and hi > notch:
        for f0 in np.arange(notch, min(hi + notch, fs / 2), notch):
            b, a = iirnotch(f0, 30, fs); x = filtfilt(b, a, x)
    return sosfiltfilt(butter(4, [lo, hi], btype='band', fs=fs, output='sos'), x)


def windows(ev, cfg):
    c0, c1 = cfg['covert_window_s']; W, onsets = {}, {}
    for tt, g in ev.groupby('trial_type'):
        W[tt] = np.c_[g.onset + c0, g.onset + c1] if tt == 'covert' else np.c_[g.onset, g.onset + g.duration]
        onsets[tt] = g.onset.values
    span = np.concatenate(list(W.values())); span = span[np.argsort(span[:, 0])]
    r = cfg['rest']; rest = []
    for (_, e0), (s1, _) in zip(span[:-1], span[1:]):
        a, b = e0 + r['guard_s'], s1 - r['guard_s']
        if b - a >= r['min_s']: rest.append((a, min(b, a + r['max_s'])))
    W['rest'] = np.array(rest)
    return W, onsets


def rms(x, fs, W):
    return np.array([np.sqrt(np.mean(x[int(a * fs):int(b * fs)] ** 2)) for a, b in W])


def envelope(x, fs, onsets, pre, post, smooth):
    k = max(1, int(smooth * fs)); env = np.convolve(np.abs(x), np.ones(k) / k, mode='same')
    seg = [env[int((o - pre) * fs):int((o + post) * fs)] for o in onsets if o - pre > 0 and (o + post) * fs < len(env)]
    n = min(len(s) for s in seg); return np.median(np.stack([s[:n] for s in seg]), 0)


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path)); out = os.path.join(ROOT, cfg['out_dir']); os.makedirs(out, exist_ok=True)
    chs = sorted({c for v in cfg['channels'].values() for c in v}); q = cfg['threshold_quantile']; E = cfg['envelope']
    res = {}; fig, axes = plt.subplots(len(cfg['runs']), 3, figsize=(12, 3.2 * len(cfg['runs'])), squeeze=False)
    for ri, run in enumerate(cfg['runs']):
        base = os.path.join(ROOT, cfg['raw_root'], run)
        raw = mne.io.read_raw_edf(base + '_eeg.edf', preload=False, verbose=False); raw.pick(chs).load_data(verbose=False)
        fs = raw.info['sfreq']; X = dict(zip(raw.ch_names, raw.get_data()))
        ev = pd.read_csv(base + '_events.tsv', sep='\t'); W, onsets = windows(ev, cfg)
        name = run.split('/')[1] + ('_overt' if 'speechopen' in run else ''); R = {'fs': fs, 'n_windows': {k: len(v) for k, v in W.items()}, 'signals': {}}
        for si, (sig, bkey) in enumerate(SIGNALS):
            a, b = cfg['channels'][sig]; x = band(X[a] - X[b], fs, *cfg['bands'][bkey], cfg['notch_hz'])
            v = {k: rms(x, fs, w) for k, w in W.items()}; thr = np.quantile(v['rest'], q); med_rest = np.median(v['rest'])
            R['signals'][sig] = {
                'median_uV': {k: float(np.median(vv) * 1e6) for k, vv in v.items()},
                'ratio_to_rest': {k: float(np.median(vv) / med_rest) for k, vv in v.items() if k != 'rest'},
                f'frac_above_rest_q{q}': {k: float(np.mean(vv > thr)) for k, vv in v.items()}}
            col = {'mic': 0, 'emg_upper': 1, 'emg_lower': 2}.get(sig)
            if col is not None:
                ax = axes[ri, col]; t = None
                for tt, c in [('covert', COPPER), ('listening', TEAL), ('overt', PURPLE)]:
                    if tt not in onsets: continue
                    on, lab = onsets[tt], f'{tt} onset'
                    e = envelope(x, fs, on, E['pre_s'], E['post_s'], E['smooth_s']) / med_rest
                    t = np.arange(len(e)) / fs - E['pre_s']; ax.plot(t, e, color=c, lw=1.2, label=lab)
                ax.axhline(1, color=INK, lw=0.6, ls=':'); ax.axvspan(*cfg['covert_window_s'], color=COPPER, alpha=0.08)
                ax.set_yscale('log'); ax.set_title(f'{name} · {sig}', fontsize=9, color=INK)
                ax.set_xlabel('s from onset', fontsize=8); ax.set_ylabel('envelope / rest median', fontsize=8)
                if ri == 0 and col == 0: ax.legend(fontsize=7, frameon=False)
        res[name] = R
        print(name, R['n_windows'], {s: R['signals'][s]['ratio_to_rest'] for s in R['signals']}, flush=True)
    fig.patch.set_facecolor(PAPER); fig.tight_layout(); fig.savefig(os.path.join(out, 'envelopes.png'), dpi=130)
    json.dump({'config': cfg, 'runs': res}, open(os.path.join(out, 'metrics.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main(sys.argv[1])
