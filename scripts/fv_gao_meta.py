"""Gao et al. (2026) visual imagery data: structure check without reading EEG values (D-111 decision 3).

Reads only the BIDS text sidecars (events.tsv, channels.tsv, eeg.json) and the BDF headers through
mne.io.read_raw_bdf(preload=False) (header and BDF+ annotations, no sample data). Does not import or run any
code shipped with the data. Writes one JSON summary.
Run on the server from the project folder: env/bin/python -I scripts/fv_gao_meta.py data/raw/gao_imagery/bids results/gao_imagery/meta.json
"""
import csv, glob, json, os, re, sys, warnings

import numpy as np

warnings.filterwarnings('ignore')
import mne

mne.set_log_level('ERROR')
root, out = sys.argv[1], sys.argv[2]


def tsv(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


runs = []
for bdf in sorted(glob.glob(os.path.join(root, 'sub-*', 'ses-*', 'eeg', '*_eeg.bdf'))):
    m = re.search(r'(sub-\d+)_(ses-\d+)_task-([A-Z]+)_eeg\.bdf$', bdf)
    sub, ses, task = m.groups()
    base = bdf[:-len('_eeg.bdf')]
    side = json.load(open(base + '_eeg.json'))
    chans = tsv(base + '_channels.tsv')
    ev = tsv(base + '_events.tsv')
    raw = mne.io.read_raw_bdf(bdf, preload=False)
    on = np.array([float(e['onset']) for e in ev])
    lat = np.array([int(e['latency']) for e in ev])
    lab = [e['type'] for e in ev]
    d = np.diff(on)
    gaps = np.where(d > 20)[0]  # block breaks: interval longer than one 17 s trial plus slack
    blocks = np.split(np.arange(len(ev)), gaps + 1)
    runlen, cur = [], 1
    for a, b in zip(lab, lab[1:]):
        cur = cur + 1 if a == b else (runlen.append(cur) or 1)
    runlen.append(cur)
    md = raw.info['meas_date']
    runs.append(dict(
        sub=sub, ses=ses, task=task,
        meas_date=md.isoformat() if md else None,
        sfreq=raw.info['sfreq'], n_times=raw.n_times, duration_s=raw.n_times / raw.info['sfreq'],
        header_ch_names=raw.ch_names, header_ch_types=sorted(set(raw.get_channel_types())),
        bdf_annotations=sorted(set(raw.annotations.description)) if len(raw.annotations) else [],
        n_bdf_annotations=len(raw.annotations),
        tsv_ch_names=[c['name'] for c in chans], tsv_bad=[c['name'] for c in chans if c['status'] != 'good'],
        sidecar={k: side.get(k) for k in ('SamplingFrequency', 'EEGChannelCount', 'EEGReference', 'EEGGround',
                                         'PowerLineFrequency', 'RecordingDuration', 'TaskDescription',
                                         'ElectrodeManufacturer', 'ElectrodeManufacturersModelName')},
        n_events=len(ev), label_counts={k: lab.count(k) for k in sorted(set(lab))},
        value_map=sorted({(e['type'], e['value']) for e in ev}), durations=sorted({e['duration'] for e in ev}),
        first_onset=float(on[0]), last_onset=float(on[-1]),
        latency_matches_onset_ms=bool(np.all(np.abs(lat - np.round(on * 1000)) <= 1)),
        trial_ids_sequential=[int(e['trial']) for e in ev] == list(range(1, len(ev) + 1)),
        interval_within_block_s=dict(min=float(np.delete(d, gaps).min()), median=float(np.median(np.delete(d, gaps))),
                                     max=float(np.delete(d, gaps).max())),
        block_sizes=[int(len(b)) for b in blocks], block_break_s=[round(float(d[g]), 3) for g in gaps],
        block_label_counts=[{k: [lab[i] for i in b].count(k) for k in sorted(set(lab))} for b in blocks],
        max_same_label_run=max(runlen),
        same_as_previous_rate=float(np.mean([a == b for a, b in zip(lab, lab[1:])])),
    ))
    raw.close()

# session order and gaps, from BDF header measurement dates
order = {}
for r in runs:
    order.setdefault(r['sub'], {}).setdefault(r['ses'], []).append((r['meas_date'], r['task']))
summary = {}
for s, ss in sorted(order.items()):
    summary[s] = {k: [t for _, t in sorted(v)] for k, v in sorted(ss.items())}
    starts = {k: min(dt for dt, _ in v) for k, v in ss.items() if all(dt for dt, _ in v)}
    if 'ses-01' in starts and 'ses-02' in starts:
        from datetime import datetime
        a, b = (datetime.fromisoformat(starts[k]) for k in ('ses-01', 'ses-02'))
        summary[s]['ses_gap_days'] = round((b - a).total_seconds() / 86400, 2)
json.dump(dict(n_runs=len(runs), runs=runs, task_order_and_gap=summary), open(out, 'w'), indent=1, ensure_ascii=False,
          default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('runs', len(runs))
