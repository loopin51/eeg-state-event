"""D-099: caches of the external listening EEG datasets for the C1 P2 pretraining (SparrKULee, KU Leuven AAD, ChineseEEG-2) and of
the C2 evaluation set DTU (two-talker trials, stim = attended stream, unattended = the other; never used for pretraining).

One cache per dataset in data/cache/<name>/ (configs/ext_prepare.yaml), in a common "recording" layout:
  eeg_hp05_sub-XXX.npy   float32 [C, N] memmap, the subject's recordings back to back. Recording sample j is stimulus
                         time stim_start_s + j / 120 s on the STIMULUS clock (drift-corrected where triggers allow it).
  eeg_meta.json          ch_names, sample_rate, positions fit, and per subject a list of recordings
                         {rec, offset, length, stim, stim_start_s, session, run, bad, flat, ...}; bad = the dataset's own
                         channel marks (never derived from labels, scores or decoder output), flat = IQR 0 in the recording.
  positions.npy          [C, 2] positions in the ds004408 cache frame (src/ese/listen/positions.py)
  stimuli.json           unique target stimuli {stim: {file, duration_s, offset, length}} (offset/length in w2v_stim.npy)
  w2v_stim.npy           float16 [1024, S] wav2vec2-large-xlsr-53 hidden-state mean over layers 14-18 at 120 Hz per stimulus
EEG chain (A4 of ds004408 / D-084): native rate -> (polyphase) about 500 Hz -> 0.5 Hz zero-phase FIR high-pass (MNE default)
-> polyphase 120 Hz -> onto the stimulus clock -> robust scaling per recording and channel ((x - median) / IQR). No
re-reference is stored: CAR (bad and flat channels left out) and the clamp at 20 are applied at load time, as for ds004408.
Audio: w2v on the whole stimulus when it is at most chunk_s long; longer stimuli in chunk_s windows every hop_s, each frame
taken from the window whose centre is nearest (context of at least (chunk_s - hop_s) / 2 on both sides).

usage: python scripts/ext_prepare.py configs/ext_prepare.yaml --dataset sparrkulee|kul_aad|dtu|chineseeeg2 [--stage eeg|audio|all]
       [--subjects 1,2] [--smoke]          (--smoke: two subjects, two recordings each, written to <out_dir>_smoke)
       python scripts/ext_prepare.py configs/ext_prepare.yaml --dataset alice --stage positions   (Alice: positions only)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, re, io, gzip, glob, json, time, shutil, argparse, yaml, numpy as np, pandas as pd
from scipy.signal import resample_poly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.listen import positions as LP


def rs(x, fs_in, fs_out, axis=-1):
    fs_in, fs_out = int(round(fs_in)), int(round(fs_out))
    if fs_in == fs_out: return x
    g = np.gcd(fs_in, fs_out); return resample_poly(x, fs_out // g, fs_in // g, axis=axis)


def eeg_chain(x, fs, e):
    """[C, T] at fs -> [C, T'] at 120 Hz on the EEG clock: polyphase to filter_rate, 0.5 Hz zero-phase FIR, polyphase to 120 Hz."""
    import mne
    fr = e['filter_rate'].get(int(round(fs)), int(round(fs))); x = rs(x, fs, fr)
    x = mne.filter.filter_data(x, fr, l_freq=e['highpass_hz'], h_freq=None, n_jobs=1, verbose=False)   # MNE default FIR, zero phase
    return rs(x, fr, e['sample_rate']), fr


def robust(y, q):
    med = np.median(y, 1, keepdims=True); iqr = np.subtract(*np.percentile(y, [q[1], q[0]], axis=1))[:, None]
    flat = iqr[:, 0] <= 1e-12; return ((y - med) / np.where(flat[:, None], 1.0, iqr)).astype(np.float32), flat


def to_stim_clock(y, fs, eeg_t0, rate, n_out):
    """y [C, T] at fs (EEG clock); stimulus time tau = k / fs (k < n_out) lies at EEG time eeg_t0 + tau * rate -> linear interpolation."""
    pos = (eeg_t0 + np.arange(n_out) / fs * rate) * fs; i = np.floor(pos).astype(int); w = (pos - i)[None].astype(np.float32)
    assert i[0] >= 0 and i[-1] + 1 < y.shape[1], f'EEG does not cover the stimulus ({i[0]}, {i[-1]}, {y.shape[1]})'
    return y[:, i] * (1 - w) + y[:, i + 1] * w


# ---------------------------------------------------------------- SparrKULee
def biosemi_trigger_onsets(status):
    """brain_pipe biosemi_trigger_processing_fn + AlignPeriodicBlockTriggers.get_trigger_indices (exporl/auditory-eeg-dataset)."""
    tr = status.astype(np.int64) & (2 ** 16 - 1); v, n = np.unique(tr, return_counts=True); ok = (v > 0) & (v < 256)
    base = v[ok][np.argsort(n[ok])[-1]]; on = np.where(tr != base)[0]
    return on[np.r_[True, np.diff(on) > 1]] if len(on) else on


def pulse_onsets(x):
    on = np.where(x > 0.5)[0]; return on[np.r_[True, np.diff(on) > 1]] if len(on) else on


def match_triggers(b, fs_b, s, fs_s):
    """brain_pipe default_drift_correction's count repair: one missing first / last EEG trigger, one spurious trigger at 0."""
    note = ''
    if len(b) + 1 < len(s): raise ValueError(f'EEG triggers {len(b)} < stimulus triggers {len(s)} - 1')
    if len(b) < len(s):
        lb, ls = (b[-1] - b[-2]) / fs_b, (s[-1] - s[-2]) / fs_s
        if 0.99 * lb < ls < 1.01 * lb: b = np.r_[b, b[-1] + int(round(ls * fs_b))]; note = 'last EEG trigger added'
        else: b = np.r_[b[0] - int(fs_b), b]; note = 'first EEG trigger added'
    elif len(b) == len(s) + 1 and b[0] == 0: b = b[1:]; note = 'spurious EEG trigger at 0 removed'
    if len(b) != len(s): raise ValueError(f'trigger counts differ: EEG {len(b)}, stimulus {len(s)}')
    return b, note


def sparrkulee_recordings(c, d, subjects):
    root = os.path.join(ROOT, d['raw_root']); recs = {}
    excl = set(d['exclude_recordings'])
    for sd in sorted(glob.glob(os.path.join(root, 'sub-*'))):
        sub = int(os.path.basename(sd)[4:])
        if (subjects and sub not in subjects) or sub in d.get('exclude_subjects', []): continue
        for ev in sorted(glob.glob(os.path.join(sd, 'ses-*', 'eeg', f'*_task-{d["task"]}_*events.tsv'))):
            bdf = ev.replace('_events.tsv', '_eeg.bdf.gz'); rel = os.path.relpath(bdf, root)
            if rel in excl or not os.path.exists(bdf): continue
            E = pd.read_csv(ev, sep='\t'); m = re.search(r'(ses-[^_]+)_task-[^_]+_run-(\d+)', ev)
            if len(E) != 1:                   # three recordings ship an events.tsv without a stimulus row
                recs.setdefault(sub, []).append(dict(rec=os.path.basename(ev)[:-len('_events.tsv')], stim='', skip=f'events.tsv has {len(E)} rows (no stimulus)'))
                continue
            e = E.iloc[0]
            recs.setdefault(sub, []).append(dict(rec=os.path.basename(ev)[:-len('_events.tsv')], session=m.group(1), run=int(m.group(2)),
                                                 eeg=bdf, stim=os.path.basename(str(e.stim_file)).split('.')[0],
                                                 stim_file=os.path.join(root, 'stimuli', str(e.stim_file)),
                                                 trigger_file=os.path.join(root, 'stimuli', str(e.trigger_file)),
                                                 condition=str(e.condition), snr=float(e.snr) if str(e.snr) != 'n/a' else None))
    return recs


def npz_gz(p):
    with gzip.open(p, 'rb') as f: z = dict(np.load(io.BytesIO(f.read())))
    return z['audio'].astype(np.float32), int(z['fs'])


def sparrkulee_eeg(c, d, r, tmp):
    import mne
    with gzip.open(r['eeg'], 'rb') as a, open(tmp, 'wb') as b: shutil.copyfileobj(a, b, 1 << 24)
    raw = mne.io.read_raw_bdf(tmp, preload=True, verbose=False); fs = raw.info['sfreq']; raw_names = raw.ch_names[:64]
    status = raw.get_data(picks=['Status'])[0]; x = raw.get_data(picks=raw_names) * 1e6; os.remove(tmp)
    alias = d.get('channel_alias', {}); names = [alias.get(n, n) for n in raw_names]          # header typos (same position)
    trig, fs_t = npz_gz(r['trigger_file']); s_on = pulse_onsets(trig); b_on = biosemi_trigger_onsets(status)
    b_on, note = match_triggers(b_on, fs, s_on, fs_t)
    tb, ts = b_on / fs, s_on / fs_t; A = np.c_[ts, np.ones_like(ts)]; (rate, t0), *_ = np.linalg.lstsq(A, tb, rcond=None)
    resid = tb - A @ np.r_[rate, t0]
    y, fr = eeg_chain(x, fs, c['eeg'])
    return y, names, dict(eeg_t0=float(t0), clock_rate=float(rate), n_triggers=int(len(b_on)), trigger_note=note,
                          renamed={a: b for a, b in zip(raw_names, names) if a != b},
                          trigger_resid_max_ms=float(np.abs(resid).max() * 1000), native_fs=float(fs), filter_rate=fr,
                          eeg_s=float(x.shape[1] / fs))


def sparrkulee_stim_duration(r):
    with gzip.open(r['stim_file'], 'rb') as f: z = np.load(io.BytesIO(f.read()))
    return len(z['audio']) / int(z['fs'])


# ---------------------------------------------------------------- KU Leuven AAD
def kul_recordings(c, d, subjects):
    import scipy.io as sio
    import hashlib
    root = os.path.join(ROOT, d['raw_root']); recs = {}
    md5 = dict(l.split() for l in open(os.path.join(root, '_list.txt')) if l.strip())   # Zenodo checksums (md5:<hex>)
    for f in sorted(glob.glob(os.path.join(root, 'S*.mat')), key=lambda p: int(re.search(r'S(\d+)', p).group(1))):
        sub = int(re.search(r'S(\d+)\.mat', f).group(1))
        if subjects and sub not in subjects: continue
        h = hashlib.md5(open(f, 'rb').read()).hexdigest()
        assert md5[os.path.basename(f)] == f'md5:{h}', f'{f}: md5 differs from Zenodo (download incomplete?)'
        T = sio.loadmat(f, squeeze_me=True, struct_as_record=False)['trials']
        for t in T:
            att = 0 if t.attended_ear == 'L' else 1; st = [str(x) for x in t.stimuli]
            dry = [re.sub(r'_hrtf\.wav$', '_dry.wav', x) for x in st]
            if not d.get('repetitions', True) and int(t.repetition): continue
            recs.setdefault(sub, []).append(dict(rec=f'S{sub}_trial{int(t.TrialID):02d}', session='ses-1', run=int(t.TrialID), mat=f,
                                                 trial_index=int(t.TrialID) - 1, stim=dry[att][:-4], unattended=dry[1 - att][:-4],
                                                 stim_file=os.path.join(root, d['stimuli_dir'], dry[att]), condition=str(t.condition),
                                                 experiment=int(t.experiment), part=int(t.part), repetition=int(t.repetition),
                                                 attended_ear=str(t.attended_ear), attended_track=int(t.attended_track)))
    return recs


def kul_eeg(c, d, r, tmp):
    import scipy.io as sio
    T = sio.loadmat(r['mat'], squeeze_me=True, struct_as_record=False)['trials']; t = T[r['trial_index']]
    assert int(t.TrialID) == r['run']; names = [str(ch.Label) for ch in t.FileHeader.Channels]; fs = float(t.FileHeader.SampleRate)
    x = np.asarray(t.RawData.EegData, np.float64).T; y, fr = eeg_chain(x, fs, c['eeg'])
    return y, names, dict(eeg_t0=0.0, clock_rate=1.0, native_fs=fs, filter_rate=fr, eeg_s=float(x.shape[1] / fs),
                          provider_highpass=f'{t.RawData.FilterImplementation} {float(t.RawData.HighPass)} Hz order {int(t.RawData.FilterOrder)}')


def wav_duration(p):
    from scipy.io import wavfile
    fs, w = wavfile.read(p, mmap=True); return len(w) / fs


# ---------------------------------------------------------------- DTU (C2 evaluation only, never pretraining; D-097 amendment 2)
# S<n>.mat (COCOHA struct): data.eeg [T, 73] (64 Biosemi 10-20 channels, EXG1-8, Status) at data.fsample.eeg = 512 Hz,
# data.event.eeg.sample / value: per trial a start event (value = expinfo trigger) and an end event. The MATLAB table expinfo
# (MCOS, not readable by scipy) is taken from results/d099_ext/dtu/expinfo_all.tsv (read from the same files, D-099). As in
# preproc_data.m (dataset code) the trial starts at its start event, where the audio starts; the EEG of the whole session is
# filtered once on the EEG clock and each trial is cut from it (no trigger inside a trial, so no drift correction: rate 1).
_DTU = {}


def dtu_subject(d, sub):
    """load S<sub>.mat once per subject: 64 EEG channels [C, T], names, fs, start / end event samples (structure checks only)."""
    import scipy.io as sio
    if _DTU.get('sub') == sub: return _DTU
    _DTU.clear(); m = sio.loadmat(os.path.join(ROOT, d['raw_root'], d['eeg_dir'], f'S{sub}.mat'), squeeze_me=True, struct_as_record=False)['data']
    names = [str(x) for x in m.dim.chan.eeg]; assert str(m.dim.eeg) == 'time_chan', m.dim.eeg
    assert names[64:] == [f'EXG{i}' for i in range(1, 9)] + ['Status'], f'S{sub}: unexpected channel list {names[60:]}'
    ev_s = np.asarray(m.event.eeg.sample, np.int64); ev_v = np.array([int(np.asarray(v).ravel()[0]) for v in m.event.eeg.value])
    assert len(ev_s) == 2 * d['n_trials'], f'S{sub}: {len(ev_s)} events, expected start and end of {d["n_trials"]} trials'
    assert np.all(np.diff(ev_s) > 0), f'S{sub}: events not in time order'
    _DTU.update(sub=sub, x=np.ascontiguousarray(np.asarray(m.eeg[:, :64], np.float64).T), names=names[:64], fs=float(m.fsample.eeg),
                start=ev_s[0::2], end=ev_s[1::2], start_value=ev_v[0::2], end_value=ev_v[1::2]); del m
    return _DTU


def dtu_recordings(c, d, subjects):
    T = pd.read_csv(os.path.join(ROOT, d['trials_table']), sep='\t'); recs = {}
    for sub, G in T.groupby('subject'):
        sub = int(sub)
        if subjects and sub not in subjects: continue
        assert sorted(G.trial) == list(range(1, d['n_trials'] + 1)), f'S{sub}: trial numbers'
        for e in G.sort_values('trial').itertuples():
            att = int(e.attend_mf); wav = [str(e.wavfile_male), str(e.wavfile_female) if isinstance(e.wavfile_female, str) else '']
            r = dict(rec=f'S{sub}_trial{int(e.trial):02d}', session='ses-1', run=int(e.trial), trial=int(e.trial), sub=sub,
                     trigger=int(e.trigger), attend_mf=att, attend_lr=int(e.attend_lr), acoustic_condition=int(e.acoustic_condition),
                     n_speakers=int(e.n_speakers), stim=wav[att - 1], unattended=wav[2 - att] if int(e.n_speakers) == 2 else '')
            if r['n_speakers'] != 2: r['skip'] = f'single-talker trial ({r["stim"]}, the same 50-s recording in all 10; not in C2, D-097 amendment 2)'
            r['stim_file'] = os.path.join(ROOT, d['raw_root'], d['audio_dir'], r['stim']); recs.setdefault(sub, []).append(r)
    return recs


def dtu_eeg(c, d, r, tmp):
    S = dtu_subject(d, r['sub']); k = r['trial'] - 1
    assert S['start_value'][k] == r['trigger'], f'{r["rec"]}: start event {S["start_value"][k]} != expinfo trigger {r["trigger"]}'
    if 'y' not in S: S['y'], S['fr'] = eeg_chain(S['x'], S['fs'], c['eeg'])      # whole session once, EEG clock
    fs = S['fs']
    return S['y'], S['names'], dict(eeg_t0=float(S['start'][k] / fs), clock_rate=1.0, native_fs=fs, filter_rate=S['fr'],
                                    eeg_s=float(S['x'].shape[1] / fs), stim_max_s=float((S['end'][k] - S['start'][k]) / fs),
                                    end_event_value=int(S['end_value'][k]), trigger_note='trial start event = audio onset (preproc_data.m)')


# ---------------------------------------------------------------- ChineseEEG-2 (filled in once the raw data are on the server)
def chineseeeg2_recordings(c, d, subjects):
    raise NotImplementedError('ChineseEEG-2 reader: written after the BIDS files and the reading-aloud audio are inspected (D-099)')


READERS = {'sparrkulee': (sparrkulee_recordings, sparrkulee_eeg, sparrkulee_stim_duration),
           'kul_aad': (kul_recordings, kul_eeg, lambda r: wav_duration(r['stim_file'])),
           'dtu': (dtu_recordings, dtu_eeg, lambda r: wav_duration(r['stim_file']))}


def positions_for(d, names):
    if d['montage'].startswith('biosemi'): return LP.biosemi_positions(names, d['montage'])
    if d['montage'].startswith('GSN'): return LP.egi_positions(names, d['montage'])
    raise ValueError(d['montage'])


DROP = ('eeg', 'mat', 'stim_file', 'trigger_file', 'trial_index', 'sub')   # reader-internal keys, not stored in eeg_meta.json


def eeg_stage(c, name, subjects, smoke):
    d = c['datasets'][name]; out = os.path.join(ROOT, c['out_root'], name + ('_smoke' if smoke else '')); os.makedirs(out, exist_ok=True)
    list_recs, read_eeg, stim_dur = READERS[name]; recs = list_recs(c, d, subjects)
    if smoke: recs = {s: v[:2] for s, v in list(recs.items())[:2]}
    meta_p = os.path.join(out, 'eeg_meta.json'); meta = json.load(open(meta_p)) if os.path.exists(meta_p) else dict(subjects={})
    e = c['eeg']; fs = e['sample_rate']; tmp = os.path.join(c['tmp_dir'], f'ext_prepare_{name}_{os.getpid()}.bdf'); names_all = meta.get('ch_names')
    durs = {}
    for sub, R in recs.items():
        t0 = time.time(); parts, rows, off = [], [], 0
        for r in R:
            if r.get('skip'):
                rows.append(dict({k: v for k, v in r.items() if k not in DROP + ('skip',)}, excluded=r['skip'])); print(f'  {r["rec"]}: excluded ({r["skip"]})', flush=True); continue
            hit = [x for x in d.get('exclude_stimuli', []) if re.search(x['pattern'], r['stim'])]
            if hit:
                rows.append(dict(rec=r['rec'], stim=r['stim'], excluded=hit[0]['reason'])); print(f'  {r["rec"]}: excluded ({hit[0]["reason"]})', flush=True); continue
            try:
                y, names, info = read_eeg(c, d, r, tmp)
            except (ValueError, AssertionError) as ex:
                rows.append(dict(rec=r['rec'], excluded=str(ex))); print(f'  {r["rec"]}: excluded ({ex})', flush=True); continue
            assert names_all is None or names == names_all, f'{r["rec"]}: channel names differ'; names_all = names
            if r['stim'] not in durs: durs[r['stim']] = stim_dur(r)
            n = int(np.floor(min(durs[r['stim']], info.get('stim_max_s', np.inf), (info['eeg_s'] - info['eeg_t0']) / info['clock_rate'] - 1.0 / fs) * fs))
            z = to_stim_clock(y, fs, info['eeg_t0'], info['clock_rate'], n); z, flat = robust(z, e['robust_quantiles'])
            bad = np.zeros(len(names), bool)   # none of the datasets ships channel status marks (D-099)
            parts.append(z); rows.append(dict({k: v for k, v in r.items() if k not in DROP},
                                              offset=off, length=n, stim_start_s=0.0, bad=bad.tolist(), flat=flat.tolist(),
                                              stim_duration_s=durs[r['stim']], **info))
            off += n
            if info.get('trigger_resid_max_ms', 0) > d.get('max_trigger_resid_ms', np.inf):
                rows[-1]['excluded'] = f'trigger residual {info["trigger_resid_max_ms"]:.1f} ms > {d["max_trigger_resid_ms"]} ms'
        if not parts: continue
        np.save(os.path.join(out, f'eeg_hp05_sub-{sub:03d}.npy'), np.concatenate(parts, 1))
        meta['subjects'][str(sub)] = rows
        print(f'sub-{sub:03d}: {len(parts)} recordings, {off / fs / 60:.1f} min, excluded {sum("excluded" in x for x in rows)} '
              f'({time.time() - t0:.0f}s)', flush=True)
        meta.update(ch_names=names_all, sample_rate=fs, eeg=e, dataset=name)
        json.dump(meta, open(meta_p, 'w'), indent=1)
    xy, fit = positions_for(d, names_all); np.save(os.path.join(out, 'positions.npy'), xy); meta['positions_fit'] = fit
    json.dump(meta, open(meta_p, 'w'), indent=1)


def stimuli_table(c, name, smoke):
    """every target stimulus file of the dataset (not only the cached recordings', so features never depend on QC or on the
    download order): SparrKULee stimuli/eeg/*.npz.gz without t_ / noise_ / triggers, KU Leuven stimuli/*_dry.wav, DTU audio/*.wav
    without dss.wav (attended and unattended segments both; DTU keys keep the .wav extension)."""
    d = c['datasets'][name]; out = os.path.join(ROOT, c['out_root'], name + ('_smoke' if smoke else '')); root = os.path.join(ROOT, d['raw_root'])
    os.makedirs(out, exist_ok=True); key = (lambda f: os.path.basename(f)) if name == 'dtu' else (lambda f: os.path.basename(f).split('.')[0])   # DTU: names with .wav (c2_dtu_data.py)
    if name == 'sparrkulee':
        files = [f for f in glob.glob(os.path.join(root, 'stimuli', 'eeg', '*.npz.gz')) if not re.match(r'(t_|noise_|triggers)', os.path.basename(f))]
    elif name == 'kul_aad': files = glob.glob(os.path.join(root, d['stimuli_dir'], '*_dry.wav'))
    elif name == 'dtu': files = [f for f in glob.glob(os.path.join(root, d['audio_dir'], '*.wav')) if os.path.basename(f) not in d['exclude_audio']]
    else: raise ValueError(name)
    if smoke:
        meta = json.load(open(os.path.join(out, 'eeg_meta.json'))); used = {r['stim'] for R in meta['subjects'].values() for r in R}
        files = [f for f in files if key(f) in used]
    return out, {key(f): dict(file=os.path.relpath(f, ROOT)) for f in sorted(files)}


def load_audio(p, fs_out):
    if p.endswith('.npz.gz'): w, fs = npz_gz(p)
    else:
        from scipy.io import wavfile
        fs, w = wavfile.read(p); w = w.astype(np.float64); w = w.mean(1) if w.ndim > 1 else w
    w = np.asarray(w, np.float64); w /= np.abs(w).max() + 1e-12; return rs(w, fs, fs_out).astype(np.float32), len(w) / fs


def audio_stage(c, name, smoke):
    import torch
    from transformers import Wav2Vec2Model
    a = c['audio']; fo = c['eeg']['sample_rate']; out, S = stimuli_table(c, name, smoke); dt = getattr(np, a['feature_dtype'])
    keys = sorted(S); lens = []
    for k in keys:
        w, dur = load_audio(os.path.join(ROOT, S[k]['file']), a['sample_rate']); S[k].update(duration_s=dur); lens.append(int(np.floor(dur * fo)))
    offs = np.r_[0, np.cumsum(lens)].astype(int)
    net = Wav2Vec2Model.from_pretrained(a['wav2vec_model']).cuda().eval(); layers = a['wav2vec_layers']

    def w2v(x, n_out):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            x = torch.from_numpy(x).cuda()[None]; x = (x - x.mean()) / (x.std() + 1e-7)
            h = torch.stack(net(x, output_hidden_states=True).hidden_states)[layers].float().mean(0)[0].T
        return torch.nn.functional.interpolate(h[None], n_out)[0].cpu().numpy()

    F = np.lib.format.open_memmap(os.path.join(out, 'w2v_stim.npy'), 'w+', dt, (1024, int(offs[-1]))); t0 = time.time()
    cs, hs = a['chunk_s'], a['hop_s']
    for k, L, o in zip(keys, lens, offs[:-1]):
        w, _ = load_audio(os.path.join(ROOT, S[k]['file']), a['sample_rate']); sr = a['sample_rate']; w = w[:int(L / fo * sr)]
        if L / fo <= cs: F[:, o:o + L] = w2v(w, L).astype(dt)
        else:
            starts = np.arange(0, max(L / fo - cs, 0) + hs, hs); starts[-1] = min(starts[-1], L / fo - cs)
            centre = starts + cs / 2; owner = np.argmin(np.abs(np.arange(L)[:, None] / fo - centre[None]), 1)
            for j, s0 in enumerate(starts):
                i0, i1 = int(round(s0 * fo)), min(int(round((s0 + cs) * fo)), L); h = w2v(w[int(i0 / fo * sr):int(i1 / fo * sr)], i1 - i0)
                sel = np.where(owner[i0:i1] == j)[0]; F[:, o + i0 + sel] = h[:, sel].astype(dt)
        S[k].update(offset=int(o), length=int(L)); print(f'{k}: {L / fo:.1f} s ({time.time() - t0:.0f}s)', flush=True)
    F.flush(); json.dump(dict(sample_rate=fo, audio=a, stimuli=S), open(os.path.join(out, 'stimuli.json'), 'w'), indent=1)


def alice_positions(c):
    """Alice (D-092 cache, untouched): file 3D positions -> ds004408 frame via the D-092 anchor channels (positions_common.npy)."""
    d = c['datasets']['alice']; cache = os.path.join(ROOT, d['cache_dir']); meta = json.load(open(os.path.join(cache, 'eeg_meta.json')))
    P = np.load(os.path.join(cache, 'positions_file.npy')); names = meta['ch_names']; anchors = meta['reve_fit']['anchors']
    xy, fit = LP.common_positions(names, P, anchors); np.save(os.path.join(cache, 'positions_common.npy'), xy)
    json.dump(fit, open(os.path.join(cache, 'positions_common_fit.json'), 'w'), indent=1); print('Alice', json.dumps(fit), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--dataset', required=True)
    ap.add_argument('--stage', default='all', choices=['eeg', 'audio', 'all', 'positions']); ap.add_argument('--subjects', default='')
    ap.add_argument('--smoke', action='store_true'); a = ap.parse_args(); c = yaml.safe_load(open(a.config))
    if a.dataset == 'alice': alice_positions(c); return
    subjects = [int(x) for x in a.subjects.split(',') if x]
    if a.stage in ('eeg', 'all'): eeg_stage(c, a.dataset, subjects, a.smoke)
    if a.stage in ('audio', 'all'): audio_stage(c, a.dataset, a.smoke)


if __name__ == '__main__':
    main()
