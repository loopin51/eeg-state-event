"""ds007808 listening: is there a consistent onset-locked auditory response? (positive control for D-057, exploratory)
First run of each day, EEG 1-20 Hz, common-average reference over channels with normal variance, segments with
duration >= 2 s, -0.2..0.6 s, top-10 % peak-amplitude trials dropped. Reports within-run split-half and between-day
correlation of the average response. usage: python scripts/ds007808_erp_check.py (run from the project root)
"""
import numpy as np, pandas as pd, mne, glob, re
from scipy.signal import butter, sosfiltfilt, detrend
fs_out = 200
for sub in (1, 2):
    edfs = sorted(glob.glob(f'data/raw/ds007808/sub-{sub:02d}/ses-*/eeg/*task-listening_*_eeg.edf'))
    byday = {}
    for e in edfs: byday.setdefault(re.search(r'ses-(\d+)', e).group(1), e)   # first run of each day
    erps, info = [], []
    for d, e in sorted(byday.items()):
        raw = mne.io.read_raw_edf(e, preload=True, verbose=False); fs = raw.info['sfreq']; X = raw.get_data()[:128]
        X = sosfiltfilt(butter(4, [1, 20], btype='band', fs=fs, output='sos'), X, axis=-1)
        sd = X.std(1); med = np.median(sd); good = (sd < 5 * med) & (sd > med / 5)
        X = X[good] - X[good].mean(0, keepdims=True)                           # common-average reference over good channels
        Xg = np.zeros((128, X.shape[1])); Xg[good] = X
        ev = pd.read_csv(e.replace('_eeg.edf', '_events.tsv'), sep='\t'); seg = ev[ev.duration >= 2]
        step = int(fs / fs_out); L = int(.8 * fs); ep = np.stack([Xg[:, int(round((o - .2) * fs)): int(round((o - .2) * fs)) + L: step] for o in seg.onset if o > .2 and (o + .6) * fs < Xg.shape[1]])
        ep -= ep[:, :, :int(.2 * fs_out)].mean(-1, keepdims=True)
        # robust trial rejection: drop trials whose peak amplitude is in the top 10 %
        amp = np.abs(ep).max((1, 2)); ep = ep[amp < np.quantile(amp, .9)]
        erps.append(ep.mean(0)); sh = np.corrcoef(ep[::2].mean(0)[good].ravel(), ep[1::2].mean(0)[good].ravel())[0, 1]
        g = ep.mean(0)[good].std(0); t = np.arange(ep.shape[-1]) / fs_out - .2
        info.append(f'{d}: good ch {good.sum()}, n {len(ep)}, split-half {sh:.2f}, GFP pre {g[t<0].mean():.2g} 0.08-0.25 s {g[(t>.08)&(t<.25)].mean():.2g}, peak {t[np.argmax(g*(t>0))]:.3f}s')
    E = np.stack(erps); C = [np.corrcoef(E[i].ravel(), E[j].ravel())[0, 1] for i in range(len(E)) for j in range(i + 1, len(E))]
    print(f'sub-{sub:02d}'); [print('  ', s) for s in info]; print(f'   between-day ERP correlation (CAR, 1-20 Hz, -0.2..0.6 s): mean {np.mean(C):.3f} min {np.min(C):.3f} max {np.max(C):.3f}')
