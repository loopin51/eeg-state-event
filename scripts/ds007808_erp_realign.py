"""D-062: ds007808 auditory ERP re-check, aligned to the acoustic onset instead of the events onset.

The acoustic onset of each listening segment is detected on the listening-audio monitor (EEG129 - EEG130). The events
onset lags the sound by a run-specific amount (0.15-0.67 s in spot checks), so onset-locked averages can misplace the N1.
Processing otherwise follows scripts/ds007808_erp_check.py (D-058): EEG 1-20 Hz, common-average reference over channels
with normal variance, 200 Hz, -0.2..0.6 s, baseline -0.2..0 s, top-10 % peak-amplitude trials dropped. The same trials
are also epoched at the events onset for comparison. First run of each recording day.

usage: python scripts/ds007808_erp_realign.py configs/ds007808_erp_realign.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, re, glob, json, yaml, numpy as np, pandas as pd, mne
from scipy.signal import butter, sosfiltfilt, hilbert
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COPPER, TEAL, INK, PAPER = '#B5652D', '#2E6E6B', '#182129', '#EEF0EC'


def acoustic_onsets(raw, lis, c):
    fs = raw.info['sfreq']; m = raw.get_data(picks=c['monitor']); x = m[0] - m[1]
    o = c['onset_detect']; x = sosfiltfilt(butter(4, o['band'], btype='band', fs=fs, output='sos'), x)
    k = max(1, int(o['smooth_s'] * fs)); env = np.convolve(np.abs(hilbert(x)), np.ones(k) / k, 'same')
    ac = np.full(len(lis), np.nan)
    for i, t in enumerate(lis.onset.values):
        s, e = int((t + o['search_s'][0]) * fs), int((t + o['search_s'][1]) * fs)
        p0, p1 = int((t + o['pre_s'][0]) * fs), int((t + o['pre_s'][1]) * fs)
        if p0 < 0 or e >= len(env): continue
        seg, pre = env[s:e], np.median(env[p0:p1]); thr = pre + o['frac'] * (np.quantile(seg, .95) - pre)
        if (seg > thr).any(): ac[i] = (s + np.argmax(seg > thr)) / fs
    return ac


def select(lis, ac, c):
    o = c['onset_detect']; lag = ac - lis.onset.values; med = np.nanmedian(lag)
    ok = np.isfinite(lag) & (np.abs(lag - med) <= o['max_dev_s'])
    end_prev = np.r_[-np.inf, (lis.onset.values + med + lis.duration.values)[:-1]]   # previous listening segment end
    ok &= (ac - end_prev) >= o['min_gap_s']
    return ok, float(med)


def run_set(spec, c):
    files = sorted(glob.glob(os.path.join(ROOT, c['raw_root'], spec['pattern'])))
    byday = {}
    for f in files: byday.setdefault(re.search(r'ses-(\d+)', f).group(1), f)
    e = c['erp']; days, erps = [], {'acoustic': [], 'onset': []}
    for d, f in sorted(byday.items()):
        raw = mne.io.read_raw_edf(f, preload=False, verbose=False); fs = raw.info['sfreq']
        raw.pick(raw.ch_names[:c['n_eeg']] + c['monitor']).load_data(verbose=False)
        ev = pd.read_csv(f.replace('_eeg.edf', '_events.tsv'), sep='\t')
        lis = ev[ev.trial_type == 'listening'].sort_values('onset').reset_index(drop=True)
        ac = acoustic_onsets(raw, lis, c); ok, lag = select(lis, ac, c)
        T = {'acoustic': ac[ok], 'onset': lis.onset.values[ok]}
        step = int(round(fs / e['fs_out'])); w0, w1 = e['window_s']; L = int(round((w1 - w0) * fs)); nt = len(range(0, L, step))
        sos = butter(4, e['band'], btype='band', fs=fs, output='sos'); n = c['n_eeg']
        valid = np.ones(ok.sum(), bool)
        for k in T: valid &= (T[k] + w0 > 0) & ((T[k] + w1) * fs < raw.n_times)
        T = {k: v[valid] for k, v in T.items()}
        EP = {k: np.zeros((len(T[k]), n, nt), np.float32) for k in T}; sd = np.zeros(n)
        chans = raw.ch_names[:n]
        for b in range(0, n, e['chan_block']):
            X = sosfiltfilt(sos, raw.get_data(picks=chans[b:b + e['chan_block']]), axis=-1)
            sd[b:b + X.shape[0]] = X.std(1)
            for k in T:
                i0 = np.round((T[k] + w0) * fs).astype(int)
                EP[k][:, b:b + X.shape[0]] = np.stack([X[:, i:i + L:step] for i in i0])
        med = np.median(sd); good = (sd < e['good_sd_ratio'] * med) & (sd > med / e['good_sd_ratio'])
        t = np.arange(nt) / e['fs_out'] + w0; g0, g1 = c['metrics']['gfp_window_s']; rec = {'day': d, 'file': os.path.basename(f),
              'n_listening': int(len(lis)), 'n_kept': int(valid.sum()), 'lag_median_s': lag, 'good_channels': int(good.sum())}
        for k, ep in EP.items():
            ep = ep[:, good] - ep[:, good].mean(1, keepdims=True)                          # common-average reference
            ep -= ep[:, :, t < 0].mean(-1, keepdims=True)
            amp = np.abs(ep).max((1, 2)); ep = ep[amp < np.quantile(amp, 1 - e['drop_top_frac'])]
            avg = ep.mean(0); gfp = avg.std(0)
            sh = np.corrcoef(ep[::2].mean(0).ravel(), ep[1::2].mean(0).ravel())[0, 1]
            full = np.zeros((n, nt)); full[good] = avg; erps[k].append(full)
            rec[k] = {'n_trials': int(len(ep)), 'split_half': float(sh), 'gfp_ratio': float(gfp[(t > g0) & (t < g1)].mean() / gfp[t < 0].mean()),
                      'gfp_peak_s': float(t[t > 0][np.argmax(gfp[t > 0])]), 'gfp': gfp.tolist()}
        days.append(rec)
        print(f"{spec['name']} {d} lag {lag:.3f}s kept {rec['n_kept']}/{rec['n_listening']} good {rec['good_channels']} | "
              + ' | '.join(f"{k}: sh {rec[k]['split_half']:.2f} gfp x{rec[k]['gfp_ratio']:.2f} peak {rec[k]['gfp_peak_s']:.3f}" for k in EP), flush=True)
    P = c['pass']; summary = {}
    for k in erps:
        E = np.stack(erps[k]); C = [np.corrcoef(E[i].ravel(), E[j].ravel())[0, 1] for i in range(len(E)) for j in range(i + 1, len(E))]
        shm = float(np.median([r[k]['split_half'] for r in days])); frac = float(np.mean([r[k]['gfp_ratio'] >= P['gfp_ratio'] for r in days]))
        bd = float(np.mean(C)) if C else float('nan')
        summary[k] = {'split_half_median': shm, 'gfp_ratio_day_frac': frac, 'between_day_mean': bd,
                      'between_day_min': float(np.min(C)) if C else float('nan'), 'n_days': len(days),
                      'pass': bool(shm >= P['split_half_median'] and frac >= P['gfp_ratio_day_frac'] and bd >= P['between_day_mean'])}
    return days, summary, t


def main(cfg_path):
    c = yaml.safe_load(open(cfg_path)); out = os.path.join(ROOT, c['out_dir']); os.makedirs(out, exist_ok=True)
    res = {}; fig, axes = plt.subplots(len(c['sets']), 2, figsize=(11, 3 * len(c['sets'])), squeeze=False)
    for si, spec in enumerate(c['sets']):
        days, summ, t = run_set(spec, c); res[spec['name']] = {'summary': summ, 'days': days}
        print(spec['name'], json.dumps(summ), flush=True)
        for ki, (k, col) in enumerate([('onset', TEAL), ('acoustic', COPPER)]):
            ax = axes[si, ki]
            for r in days: ax.plot(t, np.array(r[k]['gfp']) * 1e6, color=col, lw=0.6, alpha=0.5)
            ax.axvline(0, color=INK, lw=0.6, ls=':'); ax.set_title(f"{spec['name']} · {k}-aligned · pass={summ[k]['pass']}", fontsize=9)
            ax.set_xlabel('s'); ax.set_ylabel('GFP (µV)')
        json.dump({'config': c, 'results': res}, open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    fig.patch.set_facecolor(PAPER); fig.tight_layout(); fig.savefig(os.path.join(out, 'gfp.png'), dpi=130)


if __name__ == '__main__':
    main(sys.argv[1])
