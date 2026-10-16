"""Gao et al. (2026): trigger codes in the BDF Status channel only (D-113 draft section 4.1; coordination-session approval).

Reads the Status channel of every *_eeg.bdf with mne (picks='Status'); EEG channels are never returned or computed on.
For each run: distinct trigger codes (transitions to a nonzero value, lower 16 bits) with counts, their match to the
events.tsv onsets (within 5 ms), and the offsets of the other codes relative to the nearest following events.tsv onset.
Run through gpuq on the server: env/bin/python -I scripts/fv_gao_status.py data/raw/gao_imagery/bids results/gao_imagery/status_codes.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import collections, csv, glob, json, re, sys, warnings

import numpy as np

warnings.filterwarnings('ignore')
import mne

mne.set_log_level('ERROR')
root, out = sys.argv[1], sys.argv[2]
runs = []
for bdf in sorted(glob.glob(os.path.join(root, 'sub-*', 'ses-*', 'eeg', '*_eeg.bdf'))):
    sub, ses, task = re.search(r'(sub-\d+)_(ses-\d+)_task-([A-Z]+)_eeg\.bdf$', bdf).groups()
    with open(bdf[:-len('_eeg.bdf')] + '_events.tsv', newline='') as f:
        ev = np.array([float(r['onset']) for r in csv.DictReader(f, delimiter='\t')])
    raw = mne.io.read_raw_bdf(bdf, preload=False)
    fs = raw.info['sfreq']
    st = raw.get_data(picks=['Status'])[0].astype(np.int64) & 0xFFFF
    raw.close()
    on = np.flatnonzero((st[1:] != st[:-1]) & (st[1:] != 0)) + 1
    codes, t = st[on], on / fs
    rec = dict(sub=sub, ses=ses, task=task, n_tsv=len(ev), status_values=sorted(map(int, np.unique(st))),
               code_counts={int(k): int(v) for k, v in sorted(collections.Counter(codes.tolist()).items())})
    match = np.array([np.min(np.abs(ev - x)) <= 0.005 for x in t]) if len(t) else np.zeros(0, bool)
    rec['n_codes_matching_tsv'] = int(match.sum())
    rec['tsv_onsets_with_code'] = int(sum(np.min(np.abs(t - x)) <= 0.005 for x in ev)) if len(t) else 0
    rel = collections.defaultdict(list)                       # non-matching codes: offset to the next tsv onset
    for c, x in zip(codes[~match], t[~match]):
        nxt = ev[ev > x]
        rel[int(c)].append(round(float(x - nxt[0]), 3) if len(nxt) else None)
    rec['other_codes_offset_to_next_event_s'] = {k: dict(n=len(v), values=sorted({a for a in v if a is not None})[:20]) for k, v in rel.items()}
    runs.append(rec)
    print(sub, ses, task, rec['code_counts'], 'match', rec['n_codes_matching_tsv'], '/', len(codes), 'tsv with code', rec['tsv_onsets_with_code'], '/', len(ev), flush=True)
json.dump(dict(runs=runs), open(out, 'w'), indent=1)
