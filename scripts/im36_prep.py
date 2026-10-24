"""D-124 preprocessing and features for ds004514 (12 participants, 36 concepts, cue photo then silent naming / visual / auditory /
tactile imagery) and ds004517 (7 participants, auditory imagery only). No label-based score is computed here.
BioSemi .bdf at 2048 Hz: 64 EEG channels (A1..B32 renamed to 10-20 names from the README), filtered 0.1-40 Hz in channel chunks and
decimated to 128 Hz; bad channels (log-variance robust z > bad_mad, then > bad_epoch_frac of epochs over the ptp limit, at most
bad_epoch_max) interpolated; average reference. Events from events.tsv (sample column at 2048 Hz).
Per task epoch (task onset): DE in the imagery window, phase-randomised DE, frontal-proxy DE (meter E), GFP window means (meter G),
the cue DE of its trial (cue window after the photo onset, meter C), the pre-cue fixation DE (meter P), task name, position, delay.
Output: data/cache/im36/<ds>_<sub>.npz and prep.json.
usage: python scripts/im36_prep.py configs/im36.yaml [--only ds004514:sub-01] [--smoke]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, csv, glob, json, time, zlib, hashlib, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from sem_prep import de_feat, phase_rand, epochs, sl
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
BIOSEMI = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1',
           'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8',
           'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
TASKS = ['silent_naming', 'visual_imagery', 'auditory_imagery', 'tactile_imagery']


def concept_of(stim):
    return os.path.basename(stim).split('_PNG')[0]


def parse(evtsv):
    rows = list(csv.DictReader(open(evtsv, encoding='utf-8-sig'), delimiter='\t'))
    trials, cur = [], None
    for r in rows:
        tt = r['trial_type']
        if tt in ('animal_image', 'tool_image'):
            cur = dict(cue=int(float(r['sample'])), cue_onset=float(r['onset']), category=0 if tt.startswith('animal') else 1,
                       concept=concept_of(r['stim_file']), tasks=[]); trials.append(cur)
        elif cur is not None and tt.endswith('_task'):
            name = tt.split('_', 1)[1].replace('_task', '')
            cur['tasks'].append(dict(sample=int(float(r['sample'])), onset=float(r['onset']), task=TASKS.index(name)))
    return trials


def process(c, ds, sub, bdf, evtsv, smoke=False):
    import mne
    pc = c['prep']; fs_raw = pc['fs_raw']; dec = pc['decim']; fs = fs_raw / dec
    raw = mne.io.read_raw_bdf(bdf, preload=False, verbose=False)
    names = raw.ch_names[:64]; assert names in ([f'A{i}' for i in range(1, 33)] + [f'B{i}' for i in range(1, 33)], BIOSEMI), names[:5]   # ds004517 uses 10-20 names
    tr = parse(evtsv)
    if smoke:
        tr = tr[:12]
    stop = min(raw.n_times, max(t['tasks'][-1]['sample'] for t in tr if t['tasks']) + int(8 * fs_raw))
    E = np.empty((64, (stop + dec - 1) // dec))
    for i in range(0, 64, pc['chunk_channels']):
        x = raw.get_data(picks=list(range(i, min(64, i + pc['chunk_channels']))), stop=stop) * 1e6     # volts -> microvolts
        E[i:i + x.shape[0]] = mne.filter.filter_data(x, fs_raw, pc['band'][0], pc['band'][1], verbose=False)[:, ::dec]
    cues = np.array([t['cue'] for t in tr]) // dec
    tasks = [(ti, k, d['sample'] // dec, d['task'], d['onset'] - t['cue_onset']) for ti, t in enumerate(tr) for k, d in enumerate(t['tasks'])]
    ts = np.array([x[2] for x in tasks])
    wins = np.zeros(E.shape[1], bool)
    for t in tr:
        a = t['cue'] // dec + int(pc['cue_epoch'][0] * fs); b = (t['tasks'][-1]['sample'] if t['tasks'] else t['cue']) // dec + int(pc['task_epoch'][1] * fs)
        wins[max(a, 0):b] = True
    lv = np.log(E[:, wins].var(1)); mad = np.median(np.abs(lv - np.median(lv))) * 1.4826
    bad = [BIOSEMI[i] for i in np.flatnonzero(np.abs(lv - np.median(lv)) / (mad + 1e-12) > pc['bad_mad'])]
    E0 = E

    def clean(bads):
        r = mne.io.RawArray(E0.copy(), mne.create_info(BIOSEMI, fs, 'eeg'), verbose=False); r.set_montage(pc['montage'], verbose=False)
        r.info['bads'] = list(bads)
        if bads:
            r.interpolate_bads(reset_bads=True, verbose=False)
        r.set_eeg_reference('average', verbose=False); return r.get_data()
    E = clean(bad)
    frac = np.zeros(64); nep = 0
    for smp, ew in ((cues, pc['cue_epoch']), (ts, pc['task_epoch'])):
        ep_, _ = epochs(E, smp, ew[0], ew[1], fs); frac += ((ep_.max(-1) - ep_.min(-1)) > pc['reject_ptp_uv']).sum(0); nep += len(smp)
    bad2 = [BIOSEMI[i] for i in np.flatnonzero(frac / nep > pc['bad_epoch_frac']) if BIOSEMI[i] not in bad]
    skipped = []
    if len(bad2) > pc['bad_epoch_max']:
        skipped, bad2 = bad2, []
    if bad2:
        bad = bad + bad2; E = clean(bad)
    rng = np.random.default_rng([pc['phase_seed'], zlib.crc32(f'{ds}_{sub}'.encode())])
    front = [BIOSEMI.index(n) for n in pc['frontal_proxy']]
    # cue epochs (one per trial)
    ce, tc = epochs(E, cues, pc['cue_epoch'][0], pc['cue_epoch'][1], fs)
    c_ok = (ce.max(-1) - ce.min(-1)).max(1) <= pc['reject_ptp_uv']
    cue_de = de_feat(ce, fs, pc['de_bands'], sl(tc, pc['cue_window'])); pre_de = de_feat(ce, fs, pc['de_bands'], sl(tc, pc['pre_window']))
    # task epochs
    te, tt = epochs(E, ts, pc['task_epoch'][0], pc['task_epoch'][1], fs); w = sl(tt, pc['task_window'])
    t_ok = (te.max(-1) - te.min(-1)).max(1) <= pc['reject_ptp_uv']
    tb = te - te[..., sl(tt, pc['task_baseline'])].mean(-1, keepdims=True); gfp = tb.std(1)
    out = dict(trial=np.array([x[0] for x in tasks]), position=np.array([x[1] for x in tasks]), task=np.array([x[3] for x in tasks]),
               delay=np.array([x[4] for x in tasks]), onset=np.array([tr[x[0]]['cue_onset'] for x in tasks]),
               category=np.array([tr[x[0]]['category'] for x in tasks]), concept=np.array([tr[x[0]]['concept'] for x in tasks]),
               t_ok=t_ok, c_ok=c_ok[[x[0] for x in tasks]],
               de=de_feat(te, fs, pc['de_bands'], w).astype(np.float32), de_ph=de_feat(phase_rand(te, rng), fs, pc['de_bands'], w).astype(np.float32),
               front=de_feat(te[:, front], fs, pc['de_bands'], w).astype(np.float32),
               gfp=np.stack([gfp[:, k].mean(-1) for k in np.array_split(w, 4)], 1).astype(np.float32),
               cue=cue_de[[x[0] for x in tasks]].astype(np.float32), pre=pre_de[[x[0] for x in tasks]].astype(np.float32))
    meta = dict(ds=ds, sub=sub, n_trials=len(tr), n_tasks=len(tasks), bad=bad, bad_from_epochs=bad2, bad_epoch_rule_skipped=skipped,
                task_rejected=int((~t_ok).sum()), cue_rejected=int((~c_ok).sum()), concepts=len(set(t['concept'] for t in tr)))
    return out, meta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--only', default=''); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); c = yaml.safe_load(open(P(a.config))); od = P(c['cache_dir']); os.makedirs(od, exist_ok=True); log = []
    for ds, root in c['raw_roots'].items():
        for sd in sorted(glob.glob(os.path.join(P(root), 'sub-*'))):
            sub = os.path.basename(sd)
            if f'{ds}:{sub}' in c['exclude'] or (a.only and a.only != f'{ds}:{sub}'):
                continue
            bdf = glob.glob(os.path.join(sd, 'eeg', '*_eeg.bdf')); ev = glob.glob(os.path.join(sd, 'eeg', '*_events.tsv'))
            t0 = time.time(); out, meta = process(c, ds, sub, bdf[0], ev[0], a.smoke)
            f = os.path.join(od, f'{ds}_{sub}{"_smoke" if a.smoke else ""}.npz'); np.savez(f, **out)
            meta.update(file=os.path.relpath(f, ROOT), sha256=hashlib.sha256(open(f, 'rb').read()).hexdigest(), seconds=round(time.time() - t0, 1))
            log.append(meta); print(json.dumps(meta), flush=True)
            if a.smoke:
                return
    json.dump(dict(config=a.config, sessions=log), open(os.path.join(od, 'prep.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
