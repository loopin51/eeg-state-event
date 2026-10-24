"""F-XI (D-121) preprocessing and features for ds004306 (no label-based score is computed here).
Per session: the float32 .fdt is memory-mapped (channels x samples, EEGLAB layout), filtered 0.1-40 Hz in channel chunks and decimated
to 128 Hz; bad EEG channels (robust z of the log variance > bad_mad) are interpolated (spherical spline, standard_1005), EEG is
re-referenced to the average. Events come from the BIDS events.tsv (sample column at 1024 Hz).
Features (per trial): perception amplitude (mean in perc_window, baseline perc_baseline), band log-variance (DE), the pre-stimulus
amplitude (meter P, no baseline), GFP window means (meter G), EOG amplitude / DE (meter E), the same amplitude / DE after phase
randomisation (control), the perception epochs at 128 Hz (temporal generalisation, report only); imagination amplitude / DE / EOG /
phase-randomised versions. Output: data/cache/ds004306/<sub>_<ses>.npz and prep.json.
usage: python scripts/sem_prep.py configs/sem_ds004306.yaml [--smoke] [--only sub-010]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, re, csv, glob, json, time, zlib, hashlib, argparse, yaml, numpy as np
from scipy.signal import butter, sosfiltfilt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
EV = re.compile(r'^(Perception|Imagination)_?(?:audio_)?(a|t|image)_(penguin|guitar|flower)(?:_(.*))?$')


def parse_events(path, concepts, modalities):
    """Trials as (perception sample, imagination sample, modality, concept, exemplar, block); block = count of start_baseline before."""
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig'), delimiter='\t'))
    block, trials, pend = -1, [], None
    for r in rows:
        tt = r['trial_type'].split('###')[0]
        tt = tt.split(', ', 1)[1] if ', ' in tt else tt
        if tt == 'start_baseline':
            block += 1; continue
        m = EV.match(tt)
        if not m:
            continue
        s = int(float(r['sample']))
        if m[1] == 'Perception':
            pend = dict(sp=s, mod=modalities.index(m[2]), con=concepts.index(m[3]), ex=m[4] or '', block=max(block, 0), onset=float(r['onset']))
        elif pend is not None and modalities.index(m[2]) == pend['mod'] and concepts.index(m[3]) == pend['con']:
            pend['si'] = s; trials.append(pend); pend = None
    return trials


def filt_decim(fdt, nch, pnts, chans, band, fs, decim, chunk):
    import mne
    X = np.fromfile(fdt, dtype='<f4', count=pnts * nch).reshape(pnts, nch)      # EEGLAB .fdt: sample-major float32
    out = np.empty((len(chans), (pnts + decim - 1) // decim), np.float64)
    for i in range(0, len(chans), chunk):
        idx = chans[i:i + chunk]
        x = np.ascontiguousarray(X[:, idx].T, dtype=np.float64)
        x = mne.filter.filter_data(x, fs, band[0], band[1], verbose=False)
        out[i:i + len(idx)] = x[:, ::decim]
    return out


def de_feat(ep, fs, bands, win):
    """ep: trials x ch x t (whole epoch). Band log-variance in win (sample slice). -> trials x (ch * bands)."""
    f = []
    for lo, hi in bands:
        sos = butter(4, [lo, hi], btype='band', fs=fs, output='sos')
        y = sosfiltfilt(sos, ep, axis=-1)[..., win]
        f.append(np.log(y.var(-1) + 1e-12))
    return np.concatenate(f, axis=1)


def phase_rand(ep, rng):
    """Same random phase per frequency for every channel of a trial (keeps the spectrum and the channel covariance)."""
    F = np.fft.rfft(ep, axis=-1)
    ph = rng.uniform(0, 2 * np.pi, (ep.shape[0], 1, F.shape[-1])); ph[..., 0] = 0
    if ep.shape[-1] % 2 == 0:
        ph[..., -1] = 0
    return np.fft.irfft(F * np.exp(1j * ph), n=ep.shape[-1], axis=-1)


def epochs(data, samp, lo, hi, fs):
    a, b = int(round(lo * fs)), int(round(hi * fs))
    return np.stack([data[:, s + a:s + b] for s in samp]), np.arange(a, b) / fs


def sl(t, w):
    return np.flatnonzero((t >= w[0] - 1e-9) & (t < w[1] - 1e-9))


def process(c, sub, ses, fdt, chtsv, evtsv, smoke=False):
    import mne
    pc = c['prep']; fs = pc['fs_raw'] / pc['decim']
    names = [r['name'] for r in csv.DictReader(open(chtsv, encoding='utf-8-sig'), delimiter='\t')]
    types = [r['type'] for r in csv.DictReader(open(chtsv, encoding='utf-8-sig'), delimiter='\t')]
    nch = len(names); pnts = os.path.getsize(fdt) // (4 * nch); assert pnts * 4 * nch == os.path.getsize(fdt)
    eeg = [i for i, t in enumerate(types) if t == 'EEG']; eog = [i for i, t in enumerate(types) if t == 'EOG']
    tr = parse_events(evtsv, c['concepts'], c['modalities'])
    if smoke:
        pnts = min(pnts, int(tr[60]['si'] + 6 * pc['fs_raw'])); tr = tr[:60]
    t0 = time.time()
    D = filt_decim(fdt, nch, pnts, eeg + eog, pc['band'], pc['fs_raw'], pc['decim'], pc['chunk_channels'])
    E, O = D[:len(eeg)], D[len(eeg):]
    sp = np.array([t['sp'] for t in tr]) // pc['decim']; si = np.array([t['si'] for t in tr]) // pc['decim']
    # amendment 2 (D-121): flat raw signal (median EEG-channel ptp < flat_uv over the epoch) -> trial unusable (sub-010 is flat from 1464 s)
    Xr = np.memmap(fdt, dtype='<f4', mode='r', shape=(pnts, nch))
    def flat(s0, ew):
        a, b = s0 + int(ew[0] * pc['fs_raw']), s0 + int(ew[1] * pc['fs_raw'])
        seg = np.asarray(Xr[max(a, 0):b][:, eeg]); return np.median(seg.max(0) - seg.min(0)) < pc['flat_uv']
    p_flat = np.array([flat(t['sp'], pc['perc_epoch']) for t in tr]); i_flat = np.array([flat(t['si'], pc['imag_epoch']) for t in tr])
    live = ~(p_flat | i_flat); assert live.sum() > 0
    lo, hi = max(0, sp[live].min() - int(5 * fs)), min(E.shape[1], si[live].max() + int(5 * fs))
    wins = np.zeros(E.shape[1], bool)
    for a_, b_ in zip(sp[live], si[live]):
        wins[a_ + int(pc['perc_epoch'][0] * fs):b_ + int(pc['imag_epoch'][1] * fs)] = True
    lv = np.log(E[:, wins].var(1)); mad = np.median(np.abs(lv - np.median(lv))) * 1.4826
    bad = [names[eeg[i]] for i in np.flatnonzero(np.abs(lv - np.median(lv)) / (mad + 1e-12) > pc['bad_mad'])]
    E0 = E

    def clean(bads):
        info = mne.create_info([names[i] for i in eeg], fs, 'eeg'); raw = mne.io.RawArray(E0.copy(), info, verbose=False)
        raw.set_montage(pc['montage'], verbose=False); raw.info['bads'] = list(bads)
        if bads:
            raw.interpolate_bads(reset_bads=True, verbose=False)
        raw.set_eeg_reference('average', verbose=False); return raw.get_data()
    E = clean(bad)
    scale = 1.0 if np.median(np.abs(E[:, wins])) > 0.05 else 1e6          # to microvolts (EEGLAB data are usually in uV)
    # amendment 1 (D-121): channels exceeding the ptp limit in more than bad_epoch_frac of the signal-bearing epochs are added and
    # interpolated; amendment 2: at most bad_epoch_max channels, otherwise none are added (a global artefact, not a channel)
    frac = np.zeros(len(eeg)); nep = 0
    for smp, ew in ((sp[live], pc['perc_epoch']), (si[live], pc['imag_epoch'])):
        ep_, _ = epochs(E * scale, smp, ew[0], ew[1], fs); frac += ((ep_.max(-1) - ep_.min(-1)) > pc['reject_ptp_uv']).sum(0); nep += len(smp)
    bad2 = [names[eeg[i]] for i in np.flatnonzero(frac / nep > pc['bad_epoch_frac']) if names[eeg[i]] not in bad]
    bad2_skipped = []
    if len(bad2) > pc['bad_epoch_max']:
        bad2_skipped, bad2 = bad2, []
    if bad2:
        bad = bad + bad2; E = clean(bad)
    E = E * scale; O = O * scale
    rng = np.random.default_rng([pc['phase_seed'], zlib.crc32(f'{sub}_{ses}'.encode())])
    out = dict(modality=np.array([t['mod'] for t in tr]), concept=np.array([t['con'] for t in tr]), block=np.array([t['block'] for t in tr]),
               onset=np.array([t['onset'] for t in tr]), exemplar=np.array([t['ex'] for t in tr]))
    for cond, smp, ew, bw, ww in (('p', sp, pc['perc_epoch'], pc['perc_baseline'], pc['perc_window']),
                                  ('i', si, pc['imag_epoch'], pc['imag_baseline'], pc['imag_window'])):
        ep, t = epochs(E, smp, ew[0], ew[1], fs); eo, _ = epochs(O, smp, ew[0], ew[1], fs)
        ok = ((ep.max(-1) - ep.min(-1)).max(1) <= pc['reject_ptp_uv']) & ~(p_flat if cond == 'p' else i_flat)
        b = sl(t, bw); w = sl(t, ww)
        epb = ep - ep[..., b].mean(-1, keepdims=True); eob = eo - eo[..., b].mean(-1, keepdims=True)
        out[f'{cond}_ok'] = ok
        out[f'{cond}_amp'] = epb[..., w].mean(-1).astype(np.float32)
        out[f'{cond}_de'] = de_feat(ep, fs, pc['de_bands'], w).astype(np.float32)
        out[f'{cond}_eog_amp'] = eob[..., w].mean(-1).astype(np.float32)
        out[f'{cond}_eog_de'] = de_feat(eo, fs, pc['de_bands'], w).astype(np.float32)
        ph = phase_rand(ep, rng); phb = ph - ph[..., b].mean(-1, keepdims=True)
        out[f'{cond}_amp_ph'] = phb[..., w].mean(-1).astype(np.float32)
        out[f'{cond}_de_ph'] = de_feat(ph, fs, pc['de_bands'], w).astype(np.float32)
        if cond == 'p':
            out['p_pre'] = ep[..., sl(t, pc['pre_window'])].mean(-1).astype(np.float32)
            gfp = epb.std(1)
            out['p_gfp'] = np.stack([gfp[:, sl(t, g)].mean(-1) for g in pc['gfp_windows']], 1).astype(np.float32)
            out['p_ep'] = epb.astype(np.float32); out['p_t'] = t
            if pc.get('save_eog_epochs'):
                out['p_eog_ep'] = eob.astype(np.float32)       # D-123: window-specific ocular meter
    meta = dict(sub=sub, ses=ses, n_trials=len(tr), bad=bad, bad_from_epochs=bad2, bad_epoch_rule_skipped=bad2_skipped,
                p_flat=int(p_flat.sum()), i_flat=int(i_flat.sum()), scale=scale, pnts=int(pnts), fs=fs, seconds=round(time.time() - t0, 1),
                p_rejected=int((~out['p_ok']).sum()), i_rejected=int((~out['i_ok']).sum()),
                counts={f'{cd}_{m}': int(((out['modality'] == mi) & out[f'{cd}_ok']).sum()) for cd in 'pi' for mi, m in enumerate(c['modalities'])})
    return out, meta


def sessions(c):
    root = P(c['raw_root']); res = []
    for sd in sorted(glob.glob(os.path.join(root, 'sub-*', 'ses-*'))):
        sub, ses = sd.split(os.sep)[-2:]
        if int(sub.split('-')[1]) in [int(x.split('-')[1]) for x in c['exclude_subjects']]:
            continue
        ev = glob.glob(os.path.join(sd, 'eeg', '*_events.tsv')); ch = glob.glob(os.path.join(sd, 'eeg', '*_channels.tsv'))
        n = int(sub.split('-')[1]); s = int(ses.split('-')[1])
        fd = glob.glob(os.path.join(root, 'derivatives', 'fdt_files', f'sub{n:02d}_sess-{s:02d}', '*.fdt'))
        st = glob.glob(os.path.join(sd, 'eeg', '*_eeg.set'))
        if len(ev) == 1 and len(ch) == 1 and len(fd) == 1 and len(st) == 1:
            from scipy.io import loadmat                 # file names inside fdt_files are not reliable (sub13 holds 'sub12_sess01.fdt')
            h = loadmat(st[0], squeeze_me=True, variable_names=['nbchan', 'pnts'])
            assert int(h['nbchan']) * int(h['pnts']) * 4 == os.path.getsize(fd[0]), (sub, ses, fd[0])
            res.append((sub, ses, fd[0], ch[0], ev[0]))
        else:
            res.append((sub, ses, None, None, None))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--only', default='')
    a = ap.parse_args(); c = yaml.safe_load(open(P(a.config))); od = P(c['cache_dir']); os.makedirs(od, exist_ok=True)
    log = []
    for sub, ses, fd, ch, ev in sessions(c):
        if a.only and sub != a.only:
            continue
        if fd is None:
            log.append(dict(sub=sub, ses=ses, missing=True)); print('missing', sub, ses, flush=True); continue
        out, meta = process(c, sub, ses, fd, ch, ev, a.smoke)
        f = os.path.join(od, f'{sub}_{ses}{"_smoke" if a.smoke else ""}.npz'); np.savez(f, **out)
        meta['file'] = os.path.relpath(f, ROOT); meta['sha256'] = hashlib.sha256(open(f, 'rb').read()).hexdigest()
        log.append(meta); print(json.dumps(meta), flush=True)
        if a.smoke:
            break
    if not a.smoke:
        json.dump(dict(config=a.config, sessions=log), open(os.path.join(od, 'prep.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
