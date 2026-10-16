"""D-113 5.1 preprocessing of the Gao et al. (2026) runs (arms a and b): one run at a time, continuous signal to the cache.

32 EEG channels (Status dropped), common average reference, FIR band-pass 1-45 Hz (MNE default firwin), 50 Hz notch,
resample to 250 Hz. Stored per run as float32 [32, n] (data/cache/gao_imagery/<sub>_<ses>_<task>.npy) with a json of channel
names, sfreq and the event samples m (round(onset x 250)). No label is used here. Code shipped with the data is not run.
usage (gpuq): env/bin/python -I scripts/fv_gao_prep.py configs/fv_gao.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import csv, glob, json, re, sys, time, warnings
import numpy as np, yaml

warnings.filterwarnings('ignore')
import mne

mne.set_log_level('ERROR')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P = lambda p: os.path.join(ROOT, p)
c = yaml.safe_load(open(P(sys.argv[1]))); pp = c['preprocess']; cache = P(c['cache']); os.makedirs(cache, exist_ok=True)
t0 = time.time(); log = []
for bdf in sorted(glob.glob(os.path.join(P(c['bids']), 'sub-*', 'ses-*', 'eeg', '*_eeg.bdf'))):
    sub, ses, task = re.search(r'(sub-\d+)_(ses-\d+)_task-([A-Z]+)_eeg\.bdf$', bdf).groups(); name = f'{sub}_{ses}_{task}'
    if os.path.exists(os.path.join(cache, name + '.json')): continue
    raw = mne.io.read_raw_bdf(bdf, preload=True, exclude=['Status'])
    assert len(raw.ch_names) == 32 and raw.info['sfreq'] == 1000, (raw.ch_names, raw.info['sfreq'])
    raw.set_eeg_reference(pp['reference'], projection=False)
    raw.filter(pp['l_freq'], pp['h_freq'], method='fir', phase='zero')
    raw.notch_filter(pp['notch'], method='fir', phase='zero')
    raw.resample(pp['sfreq'])
    X = raw.get_data().astype(np.float32)
    with open(bdf[:-len('_eeg.bdf')] + '_events.tsv', newline='') as f:
        on = [float(r['onset']) for r in csv.DictReader(f, delimiter='\t')]
    np.save(os.path.join(cache, name + '.npy'), X)
    json.dump(dict(ch_names=raw.ch_names, sfreq=raw.info['sfreq'], n=X.shape[1], m=[int(round(o * pp['sfreq'])) for o in on], source=os.path.relpath(bdf, ROOT)),
              open(os.path.join(cache, name + '.json'), 'w'))
    log.append(name); print(name, X.shape, f'{time.time() - t0:.0f}s', flush=True); del raw, X
print('done', len(log), f'{time.time() - t0:.0f}s')
