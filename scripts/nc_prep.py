"""D-128 step 1: label-free epoch caches for REVE-NC (docs/08 §3.1). No scores are computed.

usage: env/bin/python scripts/nc_prep.py configs/nc_prep.yaml --data physionet_mi|cho2017|bciciv2a|faced|bci2020|ugr|tol|ds004514 [--subjects a,b]

Every raw file is checked against the dataset sha256 list before it is read. Output: data/cache/nc/<dataset>/ in the
layout of src/ese/data/nc_cache.py. Channel names are mapped to the MNE standard_1005 spelling and must all be in the
REVE position bank (stop rule, as D-125).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, re, json, glob, time, pickle, hashlib, argparse
from fractions import Fraction
import numpy as np, yaml
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly, detrend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.nc_cache import NcCacheWriter


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(buf), b''): h.update(b)
    return h.hexdigest()


class ShaList:
    def __init__(self, listing, base):
        self.base = os.path.realpath(base); self.want = {l[66:].rstrip('\n'): l[:64] for l in open(listing)}; self.got = {}

    def check(self, path):
        rel = './' + os.path.relpath(os.path.realpath(path), self.base)
        assert rel in self.want, f'{rel} not in sha256 list'
        h = sha256(path); assert h == self.want[rel], f'sha256 mismatch {rel}'; self.got[rel] = h
        return h


def _std_montage():
    import mne
    try: return mne.channels.make_standard_montage('colin27_1005')            # MNE >= 1.13 name
    except ValueError: return mne.channels.make_standard_montage('standard_1005')


def canonical_names(names, bank):
    """Map names to the standard_1005 spelling (case-insensitive) and check the REVE bank."""
    import mne
    std = {n.upper(): n for n in _std_montage().ch_names}
    out = [std.get(n.strip('.').upper(), n.strip('.')) for n in names]
    miss = [n for n in out if n not in bank]
    assert not miss, f'channels missing from the REVE position bank: {miss}'
    return out


class Cleaner:
    """notch -> Butterworth band-pass -> 200 Hz, applied to [C, T] microvolt arrays (float64)."""
    def __init__(self, c, fs_in, line_hz):
        self.fs_in, self.fs = fs_in, c['sfreq']
        self.sos = butter(c['butter_order'], c['band_hz'], btype='bandpass', fs=fs_in, output='sos')
        self.notch = [iirnotch(f, c['notch_q'], fs=fs_in) for base in np.atleast_1d(line_hz)
                      for f in np.arange(base, min(fs_in / 2, c['band_hz'][1] + 5), base)]
        r = Fraction(self.fs, int(fs_in)).limit_denominator(1000); self.up, self.down = r.numerator, r.denominator
        assert abs(fs_in * self.up / self.down - self.fs) < 1e-9, (fs_in, self.up, self.down)

    def __call__(self, x, padlen=None):
        for b, a in self.notch: x = filtfilt(b, a, x, axis=-1, padlen=padlen)
        x = sosfiltfilt(self.sos, x, axis=-1, padlen=padlen)
        return resample_poly(x, self.up, self.down, axis=-1)


def car(x): return x - x.mean(0, keepdims=True)


def _interp(X, names, bad):
    import mne
    info = mne.create_info(list(names), 200.0, 'eeg'); info.set_montage(_std_montage(), verbose=False); info['bads'] = list(bad)
    ep = mne.EpochsArray(X[None] * 1e-6, info, verbose=False); ep.interpolate_bads(reset_bads=True, verbose=False)
    return ep.get_data(copy=False)[0] * 1e6


def fix_bad(c, X, names):
    """Bad-channel rule (per recording): on a median-referenced copy (a single wild channel cannot move the median;
    raw Biosemi data are reference-free, so channels next to CMS/DRL look small before referencing), a channel whose
    filtered SD is above bad_sd_hi x or below bad_sd_lo x the median channel SD is interpolated from the other channels
    by spherical splines (MNE, standard_1005 positions); the common average follows. Frontal-pole channels with blinks
    reach ~5-7 x (Cho2017), so the default 10 x leaves them alone. X: [..., C, T] microvolts (EEG). Returns (X, bad)."""
    import mne
    Xm = X - np.median(X, axis=-2, keepdims=True)
    sd = np.sqrt((Xm ** 2).reshape(-1, X.shape[-2], X.shape[-1]).mean((0, 2))); med = np.median(sd); del Xm
    bad = [names[i] for i in np.where((sd > c['bad_sd_hi'] * med) | (sd < c['bad_sd_lo'] * med))[0]]
    if not bad: return X, bad
    info = mne.create_info(list(names), 200.0, 'eeg'); info.set_montage(_std_montage(), verbose=False); info['bads'] = bad
    shp = X.shape; ep = mne.EpochsArray(X.reshape(-1, shp[-2], shp[-1]) * 1e-6, info, verbose=False)
    ep.interpolate_bads(reset_bads=True, verbose=False)
    return (ep.get_data(copy=False) * 1e6).reshape(shp), bad


def interp_batch(X, names, bad):
    """Interpolate the channels `bad` (found by fix_bad on the task data) in other windows of the same recording."""
    if not bad: return X
    import mne
    info = mne.create_info(list(names), 200.0, 'eeg'); info.set_montage(_std_montage(), verbose=False); info['bads'] = list(bad)
    ep = mne.EpochsArray(X * 1e-6, info, verbose=False); ep.interpolate_bads(reset_bads=True, verbose=False)
    return ep.get_data(copy=False) * 1e6


def pad_to(X, L):
    """Pre-cue windows shorter than the cache row length are stored at row positions 0..len, zeros after (windows `pre`)."""
    out = np.zeros(X.shape[:-1] + (L,), X.dtype); out[..., :X.shape[-1]] = X; return out


def runs_of(onsets, factor=3.0):
    """Run index from event times: a gap above factor x the median gap starts a new run (TOL, as D-126 imnoise_prep)."""
    g = np.diff(onsets); r = np.zeros(len(onsets), int)
    for i in np.flatnonzero(g > factor * np.median(g)): r[i + 1:] += 1
    return r


def slices(c, d, t0):
    """Config windows (seconds relative to the event) -> sample slices inside an epoch starting at t0 s."""
    fs, out = c['sfreq'], {}
    for k, w in d['windows'].items():
        if w is None: out[k] = None; continue
        s = w['slice_s']; s = s if isinstance(s[0], list) else [s]
        sl = [[int(round((a - t0) * fs)), int(round((b - t0) * fs))] for a, b in s]
        out[k] = dict(slice=sl[0] if len(sl) == 1 else sl, rows=w['rows'], seconds=w['slice_s'], **{q: v for q, v in w.items() if q not in ('slice_s', 'rows')})
    return out


# ------------------------------------------------------------------ PhysioNet-MI (EEGMMIDB)
def physionet_mi(c, d, bank, subjects):
    import mne
    mne.set_log_level('ERROR')
    base = os.path.join(ROOT, d['raw']); sl = ShaList(os.path.join(ROOT, d['sha256_list']), base)
    subs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(base, 'S[0-9][0-9][0-9]')))
    runs = {int(k): v for k, v in d['runs'].items()}; L = int(round((d['epoch_s'][1] - d['epoch_s'][0]) * c['sfreq']))
    # exclusion rule (docs/08 §3.1) recomputed from the headers; must reproduce the fixed list
    excl = []
    for s in subs:
        ok = True; n_img = 0
        for r in runs:
            raw = mne.io.read_raw_edf(os.path.join(base, s, f'{s}R{r:02d}.edf'), preload=False)
            ok &= raw.info['sfreq'] == d['exclude_rule']['sfreq']; n_img += sum(a in ('T1', 'T2') for a in raw.annotations.description)
        if not ok or n_img != d['exclude_rule']['n_imagery_events']: excl.append(s)
    assert excl == d['expect_excluded'], excl
    subs = [s for s in subs if s not in excl and (not subjects or s in subjects)]
    w = None; cl = Cleaner(c, 160.0, d['line_hz'])
    for si, s in enumerate(subs):
        t_sub = time.time()
        for ri, (r, kind) in enumerate(sorted(runs.items())):
            f = os.path.join(base, s, f'{s}R{r:02d}.edf'); sl.check(f)
            raw = mne.io.read_raw_edf(f, preload=True)
            names = canonical_names(raw.ch_names, bank)
            if w is None:
                w = NcCacheWriter(os.path.join(ROOT, c['cache_dir']), 'physionet_mi', names, L, extra=dict(
                    epoch_t0_s=d['epoch_s'][0], windows=slices(c, d, d['epoch_s'][0]), lowpass_hz=d['lowpass_hz'],
                    excluded_subjects=excl, preprocessing=dict(c=dict((k, c[k]) for k in ('band_hz', 'butter_order', 'notch_q', 'sfreq')),
                                                               line_hz=d['line_hz'], fs_in=160.0, reference='common average'),
                    labels=d['labels'], source='PhysioNet eegmmidb 1.0.0'))
            assert names == w.header['ch_names']
            x, bad = fix_bad(c, cl(raw.get_data() * 1e6), names); x = car(x); T = x.shape[1]
            if bad: w.header.setdefault('bad_channels', {})[f'{s}R{r:02d}'] = bad
            on = raw.annotations.onset; desc = raw.annotations.description
            xs, rows = [], []
            for j, (o, a) in enumerate(zip(on, desc)):
                i0 = int(round((o + d['epoch_s'][0]) * c['sfreq']))
                if i0 < 0 or i0 + L > T: continue
                is_null = a == d['null_event']
                xs.append(x[:, i0:i0 + L])
                rows.append(dict(subject=s, session=0, run=r, trial=j, kind='null' if is_null else 'task',
                                 label=-1 if is_null else int(d['labels'][kind][a]), event=a, clip_id=-1,
                                 t_onset=float(o), t_order=(ri + o / raw.times[-1]) / len(runs)))
            w.add(np.stack(xs), rows)
        print(json.dumps(dict(subject=s, n=w.n, seconds=round(time.time() - t_sub, 1))), flush=True)
    return w, sl


# ------------------------------------------------------------------ Cho2017 (GigaDB 100295)
def cho2017(c, d, bank, subjects):
    import mne
    from scipy.io import loadmat
    base = os.path.join(ROOT, d['raw']); sl = ShaList(os.path.join(ROOT, d['sha256_list']), base)
    files = sorted(glob.glob(os.path.join(base, 's[0-9][0-9].mat')))
    files = [f for f in files if not subjects or os.path.basename(f)[:-4] in subjects]
    names = canonical_names(d['eeg'], bank); fs_in = 512.0; cl = Cleaner(c, fs_in, d['line_hz'])
    seg = int(round((d['segment_s'][1] - d['segment_s'][0]) * fs_in)); L = int(round((d['segment_s'][1] - d['segment_s'][0]) * c['sfreq']))
    pad = int(d['segment_pad_s'] * fs_in); cue = int(round(-d['segment_s'][0] * fs_in))
    # channel order check: digitized senloc vs standard_1005 after orthogonal Procrustes; fraction of channels whose
    # nearest standard electrode has the same name. Biosemi channel order is fixed by the amplifier, so a low value for
    # one subject means a digitization slip (s10, s33: one block shifted by one row), not a data problem; the REVE bank
    # positions are used, never senloc.
    from scipy.linalg import orthogonal_procrustes
    std = _std_montage().get_positions()['ch_pos']
    P = np.array([std[n] for n in names]); P = P - P.mean(0); P /= np.linalg.norm(P)
    w = NcCacheWriter(os.path.join(ROOT, c['cache_dir']), 'cho2017', names, L, ref_names=d['emg'], extra=dict(
        epoch_t0_s=d['segment_s'][0], windows=slices(c, d, d['segment_s'][0]), lowpass_hz=d['lowpass_hz'],
        preprocessing=dict(c=dict((k, c[k]) for k in ('band_hz', 'butter_order', 'notch_q', 'sfreq')), line_hz=d['line_hz'],
                           fs_in=fs_in, segment_pad_s=d['segment_pad_s'], per_segment='linear detrend + filter with odd extension',
                           reference='common average (EEG only)'),
        labels={'left_hand': 0, 'right_hand': 1}, noise_types=d['noise_types'], source='GigaDB 10.5524/100295'))
    order_r = {}
    for f in files:
        t_sub = time.time(); s = os.path.basename(f)[:-4]; sl.check(f)
        e = loadmat(f, squeeze_me=True, struct_as_record=False)['eeg']
        assert float(e.srate) == fs_in and list(np.atleast_1d(e.frame)) == [-2000, 5000]
        Q = np.asarray(e.senloc, float); Q = Q - Q.mean(0); Q /= np.linalg.norm(Q); Q = Q @ orthogonal_procrustes(Q, P)[0]
        order_r[s] = round(float((((Q[:, None] - P[None]) ** 2).sum(-1).argmin(1) == np.arange(64)).mean()), 3)
        n = int(e.n_imagery_trials); ev = np.where(np.asarray(e.imagery_event) > 0)[0]
        assert len(ev) == n and np.all(ev == np.arange(n) * seg + cue - 1), (s, ev[:3])   # MATLAB sample 1024 -> index 1023
        bad_v = [set(np.atleast_1d(v).astype(int).tolist()) for v in np.atleast_1d(e.bad_trial_indices.bad_trial_idx_voltage)]
        bad_m = [set(np.atleast_1d(v).astype(int).tolist()) for v in np.atleast_1d(e.bad_trial_indices.bad_trial_idx_mi)]
        xs, refs, rows = [], [], []
        for lab, arr in ((0, e.imagery_left), (1, e.imagery_right)):
            A = np.asarray(arr, float); assert A.shape == (68, n * seg), A.shape
            for k in range(n):
                y = cl(detrend(A[:, k * seg:(k + 1) * seg], axis=-1), padlen=pad)
                xs.append(y[:64]); refs.append(y[64:])
                rows.append(dict(subject=s, session=0, run=0, trial=lab * n + k, kind='task', label=lab, clip_id=-1,
                                 t_onset=float(k * (seg / fs_in)), t_order=k / n,
                                 bad_voltage=(k + 1) in bad_v[lab], bad_mi=(k + 1) in bad_m[lab]))
        X, bad = fix_bad(c, np.stack(xs), names); X = X - X.mean(1, keepdims=True)
        if bad: w.header.setdefault('bad_channels', {})[s] = bad
        w.add(X, rows, ref=np.stack(refs)); del X, xs
        rest = cl(detrend(np.asarray(e.rest, float), axis=-1), padlen=pad)
        noise = [cl(detrend(np.asarray(z, float), axis=-1), padlen=pad) for z in np.atleast_1d(e.noise)]
        fixb = lambda z: car(_interp(z, names, bad)) if bad else car(z)      # non-task recordings reuse the subject's list
        w.add_aux(s, rest_eeg=fixb(rest[:64]).astype(np.float32), rest_ref=rest[64:].astype(np.float32),
                  noise_eeg=np.stack([fixb(z[:64]) for z in noise]).astype(np.float32),
                  noise_ref=np.stack([z[64:] for z in noise]).astype(np.float32), noise_types=np.array(d['noise_types']))
        print(json.dumps(dict(subject=s, n=w.n, senloc_nn=order_r[s], seconds=round(time.time() - t_sub, 1))), flush=True)
    w.header['senloc_nn_match'] = order_r
    if not subjects: assert np.mean([v >= 0.95 for v in order_r.values()]) >= 0.9, order_r
    return w, sl


# ------------------------------------------------------------------ BCI Competition IV 2a (BNCI 001-2014)
def bciciv2a(c, d, bank, subjects):
    from scipy.io import loadmat
    base = os.path.join(ROOT, d['raw']); sl = ShaList(os.path.join(ROOT, d['sha256_list']), base)
    names = canonical_names(d['eeg'], bank); fs_in = 250.0; cl = Cleaner(c, fs_in, d['line_hz'])
    L = int(round((d['epoch_s'][1] - d['epoch_s'][0]) * c['sfreq']))
    w = NcCacheWriter(os.path.join(ROOT, c['cache_dir']), 'bciciv2a', names, L, ref_names=d['eog'], extra=dict(
        epoch_t0_s=d['epoch_s'][0], windows=slices(c, d, d['epoch_s'][0]), lowpass_hz=d['lowpass_hz'],
        preprocessing=dict(c=dict((k, c[k]) for k in ('band_hz', 'butter_order', 'notch_q', 'sfreq')), line_hz=d['line_hz'],
                           fs_in=fs_in, reference='common average (EEG only); EOG left as recorded'),
        labels={'left hand': 0, 'right hand': 1, 'feet': 2, 'tongue': 3}, source='BNCI 001-2014 (TU Graz copy)'))
    for sub in range(1, 10):
        s = f'A{sub:02d}'
        if subjects and s not in subjects: continue
        for ses, tag in ((0, 'T'), (1, 'E')):
            t_sub = time.time(); f = os.path.join(base, f'{s}{tag}.mat'); sl.check(f)
            runs = np.atleast_1d(loadmat(f, squeeze_me=True, struct_as_record=False)['data'])
            task_runs = [r for r in runs if np.size(r.trial)]; aux = {}
            for ri, r in enumerate(runs):
                assert float(r.fs) == fs_in and r.X.shape[1] == 25
                if not np.size(r.trial): aux[f'calib{ri}_eeg'] = car(cl(r.X[:, :22].T.astype(float))).astype(np.float32)
            xs, refs, rows = [], [], []
            for ri, r in enumerate(task_runs):
                y = cl(r.X.T.astype(float)); ex, bad = fix_bad(c, y[:22], names); ex, rf = car(ex), y[22:]; T = y.shape[1]
                if bad: w.header.setdefault('bad_channels', {})[f'{s}{tag}_run{ri}'] = bad
                tr = np.atleast_1d(r.trial).astype(int) - 1; lab = np.atleast_1d(r.y).astype(int) - 1; art = np.atleast_1d(r.artifacts).astype(int)
                for j, t in enumerate(tr):
                    i0 = int(round((t / fs_in + d['cue_s'] + d['epoch_s'][0]) * c['sfreq'])); assert 0 <= i0 and i0 + L <= T, (s, tag, ri, j)
                    xs.append(ex[:, i0:i0 + L]); refs.append(rf[:, i0:i0 + L])
                    rows.append(dict(subject=s, session=ses, run=ri, trial=ri * len(tr) + j, kind='task', label=int(lab[j]), clip_id=-1,
                                     t_onset=float(t / fs_in + d['cue_s']), t_order=(ri + j / len(tr)) / len(task_runs), artifact=bool(art[j])))
            w.add(np.stack(xs), rows, ref=np.stack(refs)); w.add_aux(f'{s}{tag}', **aux)
            print(json.dumps(dict(subject=s, session=tag, n=w.n, seconds=round(time.time() - t_sub, 1))), flush=True)
    return w, sl


# ------------------------------------------------------------------ FACED (official Processed_data)
def faced(c, d, bank, subjects):
    from ese.data.montage import FACED32, ALIAS
    base = os.path.join(ROOT, d['raw']); sl = ShaList(os.path.join(ROOT, d['sha256_list']), os.path.join(ROOT, d['sha_base']))
    names = canonical_names([ALIAS.get(n, n) for n in FACED32], bank)
    EMO = [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 4 + [5] * 3 + [6] * 3 + [7] * 3 + [8] * 3
    VAL = [-1] * 12 + [0] * 4 + [1] * 12
    r = Fraction(c['sfreq'], d['sfreq_in']); L = int(30 * c['sfreq'])
    w = NcCacheWriter(os.path.join(ROOT, c['cache_dir']), 'faced', names, L, extra=dict(
        epoch_t0_s=0.0, windows=slices(c, d, 0.0), lowpass_hz=d['lowpass_hz'],
        preprocessing=dict(official='0.05-47 Hz, ICA, re-referenced, last 30 s of each clip (FACED Processed_data)',
                           here=f'resample_poly {r.numerator}/{r.denominator} only', fs_in=d['sfreq_in']),
        channel_note='HEOR/HEOL are mastoid leads (Electrode_Location.xlsx), mapped to A2/A1 as src/ese/data/montage.py',
        labels='9 emotions: 0 anger 1 disgust 2 fear 3 sadness 4 neutral 5 amusement 6 inspiration 7 joy 8 tenderness',
        source='FACED Processed_data (Kaggle copy on NAS)'))
    for f in sorted(glob.glob(os.path.join(base, 'sub*.pkl'))):
        s = os.path.basename(f)[:-4]
        if subjects and s not in subjects: continue
        sl.check(f); x = np.asarray(pickle.load(open(f, 'rb')), dtype=np.float64); assert x.shape == (28, 32, 7500), x.shape
        y = resample_poly(x, r.numerator, r.denominator, axis=-1)
        w.add(y, [dict(subject=s, session=0, run=0, trial=k, kind='task', label=EMO[k], clip_id=k + 1, t_onset=0.0, t_order=k / 28,
                       valence=VAL[k]) for k in range(28)])
        if int(s[3:]) % 20 == 0: print(json.dumps(dict(subject=s, n=w.n)), flush=True)
    return w, sl


# ------------------------------------------------------------------ BCI Competition 2020 Track 3 (imagined speech, 5 words)
def bci2020(c, d, bank, subjects):
    from scipy.io import loadmat
    base = os.path.join(ROOT, d['raw']); sl = ShaList(os.path.join(ROOT, d['sha256_list']), os.path.join(ROOT, d['sha_base']))
    fs_in = 256.0; cl = Cleaner(c, fs_in, d['line_hz']); n0 = int(round(-d['epoch_s'][0] * fs_in)); pad = int(d['pad_s'] * fs_in)
    w = None
    for sid in range(1, 16):
        s = f'sub-{sid:02d}'
        if subjects and s not in subjects: continue
        t_sub = time.time(); xs, pres, rows = [], [], []
        for ses, (folder, key) in enumerate((('Training set', 'epo_train'), ('Validation set', 'epo_validation'))):
            p = os.path.join(base, folder, f'Data_Sample{sid:02d}.mat'); sl.check(p)
            e = loadmat(p)[key][0, 0]; assert float(e['fs'][0, 0]) == fs_in
            x = np.transpose(e['x'], (2, 1, 0)).astype(np.float64); y = e['y'].argmax(0); n = len(x)
            names = canonical_names([str(q[0]) for q in e['clab'][0]], bank)
            xs.append(cl(x, padlen=pad))
            pres.append(cl(x[..., :n0], padlen=n0 - 1))      # pre-cue cut from the raw trial before filtering (D-126 amendment 5)
            rows += [dict(subject=s, session=ses, run=ses, trial=ses * 1000 + k, kind='task', label=int(y[k]), split=('train', 'test')[ses],
                          clip_id=-1, t_onset=float(k), t_order=k / n) for k in range(n)]
        X = np.concatenate(xs); Pre = np.concatenate(pres); L = X.shape[-1]
        if w is None:
            w = NcCacheWriter(os.path.join(ROOT, c['cache_dir']), 'bci2020', names, L, extra=dict(
                epoch_t0_s=d['epoch_s'][0], windows=slices(c, d, d['epoch_s'][0]), lowpass_hz=d['lowpass_hz'],
                preprocessing=dict(c=dict((k, c[k]) for k in ('band_hz', 'butter_order', 'notch_q', 'sfreq')), line_hz=d['line_hz'], fs_in=fs_in,
                                   per_trial='filtered alone with odd extension (pad_s)', pre='rows kind pre: raw [-0.5, 0) filtered alone, row positions 0-0.5 s',
                                   reference='common average'),
                labels=d['labels'], split='official: Training set -> train, Validation set -> test (Test set has no labels)',
                source='BCI Competition 2020 Track 3'))
        assert names == w.header['ch_names'] and L == w.header['length']
        X, bad = fix_bad(c, X, names); Pre = interp_batch(Pre, names, bad)
        if bad: w.header.setdefault('bad_channels', {})[s] = bad
        X = X - X.mean(1, keepdims=True); Pre = Pre - Pre.mean(1, keepdims=True)
        w.add(X, rows); w.add(pad_to(Pre, L), [dict(r, kind='pre') for r in rows])
        print(json.dumps(dict(subject=s, n=w.n, seconds=round(time.time() - t_sub, 1))), flush=True)
    return w, sl


# ------------------------------------------------------------------ UGR-MINDVOICE (OSF 6sh5d), raw EDF, no ICA
def ugr(c, d, bank, subjects):
    import mne
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    from reve_speech_prep import trials_from_events, word_table, norm
    base = os.path.join(ROOT, d['raw']); sl = ShaList(os.path.join(ROOT, d['sha256_list']), base)
    words, wcat, cats = word_table(os.path.join(ROOT, d['words'])); wn = {norm(v): i for i, v in enumerate(words)}; wf = {norm(v, False): i for i, v in enumerate(words)}
    edfs = sorted(glob.glob(os.path.join(base, d['edf_glob']))); subs = sorted({re.findall(r'(sub-\d+)', f)[0] for f in edfs})
    subs = [s for s in subs if s not in d['exclude_subjects'] and (not subjects or s in subjects)]
    fs0 = 1000.0; cl = Cleaner(c, fs0, d['line_hz']); fs = c['sfreq']; L = int(round((d['epoch_s'][1] - d['epoch_s'][0]) * fs)); a0 = int(round(d['epoch_s'][0] * fs))
    w = None
    for s in subs:
        for si, f in enumerate([f for f in edfs if f'/{s}/' in f]):
            t_sub = time.time(); ses = re.findall(r'(ses-\d+)', f)[0]; key = f'{s}_{ses}'; sl.check(f)
            ev = glob.glob(os.path.join(base, d['events_dir'], s, ses, 'eeg', '*_events.tsv')); assert len(ev) == 1, (key, ev)
            raw = mne.io.read_raw_edf(f, preload=False, verbose=False); assert raw.info['sfreq'] == fs0
            refs = d['ref_slots'].get(key, d['eog']); eeg = [n for n in raw.ch_names if n not in refs]
            names = canonical_names(eeg, bank)
            if w is None:
                w = NcCacheWriter(os.path.join(ROOT, c['cache_dir']), 'ugr', names, L, ref_names=d['eog'], extra=dict(
                    epoch_t0_s=d['epoch_s'][0], windows=slices(c, d, d['epoch_s'][0]), lowpass_hz=d['lowpass_hz'],
                    preprocessing=dict(c=dict((k, c[k]) for k in ('band_hz', 'butter_order', 'notch_q', 'sfreq')), line_hz=d['line_hz'], fs_in=fs0,
                                       ica='none (D-125 used ICA; N2 needs the ocular components left in)', reference='common average (EEG only)'),
                    ref_slots=d['ref_slots'], events='task rows: StartSpeech; null rows: StartStimulus - 1.0 s (rest, same trial)',
                    labels=dict(label='mode 0 Real 1 Silent', cat=cats, word=words), source='UGR-MINDVOICE (OSF 6sh5d)'))
            assert names == w.header['ch_names'], key
            X = np.concatenate([cl(raw.get_data(picks=eeg[i:i + 16]) * 1e6) for i in range(0, len(eeg), 16)])
            R = cl(raw.get_data(picks=refs) * 1e6); X, bad = fix_bad(c, X, names); X = car(X); T = X.shape[1]
            if bad: w.header.setdefault('bad_channels', {})[key] = bad
            tr = [t for t in trials_from_events(ev[0]) if t['part'] == 'Experiment' and 'Speech' in t['s'] and 'Stimulus' in t['s']]
            xs, rs, rows = [], [], []
            for k, t in enumerate(tr):
                i_s = int(round(t['s']['Speech'] / fs0 * fs)) + a0; i_r = int(round(t['s']['Stimulus'] / fs0 * fs + d['null_offset_s'] * fs)) + a0
                if min(i_s, i_r) < 0 or max(i_s, i_r) + L > T: continue
                wd = wn.get(norm(t['item'])) if t['kind'] == 'Words' else None
                if t['kind'] == 'Words' and wd is None: wd = wf.get(norm(t['item'], False))
                wd = -1 if wd is None else wd
                r0 = dict(subject=s, session=si, run=si, trial=k, label=0 if t['mode'] == 'Real' else 1, cat=wcat[wd] if wd >= 0 else -1, word=wd,
                          kind_item=t['kind'], cue=t['cue'], item=t['item'].strip(), clip_id=-1, t_onset=t['s']['Speech'] / fs0, t_order=k / len(tr))
                for kind, i0 in (('task', i_s), ('null', i_r)):
                    xs.append(X[:, i0:i0 + L]); rs.append(R[:, i0:i0 + L]); rows.append(dict(r0, kind=kind))
            w.add(np.stack(xs), rows, ref=np.stack(rs)); del X, R, raw
            print(json.dumps(dict(session=key, n=w.n, trials=len(tr), seconds=round(time.time() - t_sub, 1))), flush=True)
    return w, sl


# ------------------------------------------------------------------ Thinking out loud (ds003626, authors' epochs)
def tol(c, d, bank, subjects):
    import mne
    base = os.path.join(ROOT, d['raw']); sl = ShaList(os.path.join(ROOT, d['sha256_list']), base)
    fs_in = 256.0; cl = Cleaner(c, fs_in, d['line_hz']); pad = int(d['pad_s'] * fs_in); n0 = int(round(-d['epoch_s'][0] * fs_in))
    # 128 BioSemi electrodes (A1..D32, not 10-05 names) -> the 64 10-10 names of the BioSemi 64 cap: both MNE templates
    # (biosemi128, standard_1005) are moved to the unit sphere (least-squares sphere fit), aligned by alternating a one-to-one
    # minimum-distance assignment (Hungarian) and an orthogonal Procrustes rotation (10 rounds), and the final assignment
    # picks one electrode per name. Fixed and label-free; distances (percent of the head radius) go to the header.
    from scipy.optimize import linear_sum_assignment
    from scipy.linalg import orthogonal_procrustes
    def sphere(M):
        z = np.linalg.lstsq(np.c_[2 * M, np.ones(len(M))], (M ** 2).sum(1), rcond=None)[0]; ctr = z[:3]
        return (M - ctr) / np.sqrt(z[3] + (ctr ** 2).sum())
    b128 = mne.channels.make_standard_montage('biosemi128').get_positions()['ch_pos']; std = _std_montage().get_positions()['ch_pos']
    bn = list(b128); B = sphere(np.array([b128[n] for n in bn])); Q = sphere(np.array([std[n] for n in d['names64']]))
    for _ in range(10):
        qi, bj = linear_sum_assignment(np.linalg.norm(Q[:, None] - B[None], axis=-1)); B = B @ orthogonal_procrustes(B[bj], Q[qi])[0]
    D_ = np.linalg.norm(Q[:, None] - B[None], axis=-1); qi, bj = linear_sum_assignment(D_); o = np.argsort(qi)
    pick = [bn[j] for j in bj[o]]; dist = {n: round(float(D_[i, j]) * 100, 1) for n, i, j in zip(d['names64'], qi[o], bj[o])}
    names = canonical_names(d['names64'], bank); w = None
    for sd in sorted(glob.glob(os.path.join(base, 'derivatives', 'sub-*'))):
        s = os.path.basename(sd)
        if subjects and s not in subjects: continue
        t_sub = time.time(); xs, rs, pres, rows = [], [], [], []
        for si, dd in enumerate(sorted(glob.glob(os.path.join(sd, 'ses-*')))):
            p = os.path.join(dd, f'{s}_{os.path.basename(dd)}')
            for suf in ('_eeg-epo.fif', '_exg-epo.fif', '_events.dat'): sl.check(p + suf)
            ev = np.load(p + '_events.dat', allow_pickle=True)
            e = mne.read_epochs(p + '_eeg-epo.fif', verbose=False); x = mne.read_epochs(p + '_exg-epo.fif', verbose=False)
            assert len(e) == len(ev) == len(x) and e.info['sfreq'] == fs_in and abs(e.tmin - d['epoch_s'][0]) < 1e-6, p
            run = runs_of(ev[:, 0].astype(float)); keep = np.flatnonzero(ev[:, 2] == d['condition'])
            E = e.get_data(picks=pick)[keep] * 1e6; A = x.get_data()[keep] * 1e6
            xs.append(cl(E, padlen=pad)); rs.append(cl(A, padlen=pad)); pres.append(cl(E[..., :n0], padlen=n0 - 1))
            rows += [dict(subject=s, session=si, run=int(si * 100 + run[k]), trial=int(si * 1000 + k), kind='task', label=int(ev[k, 1]),
                          condition=int(ev[k, 2]), clip_id=-1, t_onset=float(ev[k, 0] / 1024.0), t_order=float(j / len(keep)))
                     for j, k in enumerate(keep)]
        X = np.concatenate(xs); R = np.concatenate(rs); Pre = np.concatenate(pres); L = X.shape[-1]
        if w is None:
            w = NcCacheWriter(os.path.join(ROOT, c['cache_dir']), 'tol', names, L, ref_names=[f'EXG{i}' for i in range(1, 9)], extra=dict(
                epoch_t0_s=d['epoch_s'][0], windows=slices(c, d, d['epoch_s'][0]), lowpass_hz=d['lowpass_hz'],
                preprocessing=dict(c=dict((k, c[k]) for k in ('band_hz', 'butter_order', 'notch_q', 'sfreq')), line_hz=d['line_hz'], fs_in=fs_in,
                                   source_epochs='authors derivatives (0.5-100 Hz, 50 Hz notch, -0.5-4 s); refiltered per epoch with odd extension',
                                   pre='rows kind pre: [-0.5, 0) cut from the derivative epoch, filtered alone', reference='common average (64 picked channels)'),
                channel_map=dict(zip(d['names64'], pick)), channel_map_mm=dist, condition=d['condition'],
                labels='0 arriba 1 abajo 2 derecha 3 izquierda (inner speech)', source='OpenNeuro ds003626 derivatives'))
        assert L == w.header['length']
        X, bad = fix_bad(c, X, names); Pre = interp_batch(Pre, names, bad)
        if bad: w.header.setdefault('bad_channels', {})[s] = bad
        X = X - X.mean(1, keepdims=True); Pre = Pre - Pre.mean(1, keepdims=True)
        w.add(X, rows, ref=R); w.add(pad_to(Pre, L), [dict(r, kind='pre') for r in rows], ref=np.zeros_like(R))
        print(json.dumps(dict(subject=s, n=w.n, seconds=round(time.time() - t_sub, 1))), flush=True)
    return w, sl


# ------------------------------------------------------------------ ds004514 (concept imagery, animals / tools)
def ds004514(c, d, bank, subjects):
    import csv, mne
    base = os.path.join(ROOT, d['raw']); sl = ShaList(os.path.join(ROOT, d['sha256_list']), base)
    fs0 = 2048.0; cl = Cleaner(c, fs0, d['line_hz']); fs = c['sfreq']; names = canonical_names(d['names64'], bank)
    L = int(round((d['epoch_s'][1] - d['epoch_s'][0]) * fs)); a0 = int(round(d['epoch_s'][0] * fs)); npre = int(round(d['pre_s'] * fs0))
    tasks = d['tasks']; w = NcCacheWriter(os.path.join(ROOT, c['cache_dir']), 'ds004514', names, L, extra=dict(
        epoch_t0_s=d['epoch_s'][0], windows=slices(c, d, d['epoch_s'][0]), lowpass_hz=d['lowpass_hz'],
        preprocessing=dict(c=dict((k, c[k]) for k in ('band_hz', 'butter_order', 'notch_q', 'sfreq')), line_hz=d['line_hz'], fs_in=fs0,
                           pre=f'rows kind pre: raw [-{d["pre_s"]}, 0) before the cue photo (fixation), filtered alone, row positions 0-{d["pre_s"]} s',
                           reference='common average'),
        excluded_subjects=d['exclude_subjects'], tasks=tasks, labels='category 0 animal 1 tool', source='OpenNeuro ds004514'))
    for f in sorted(glob.glob(os.path.join(base, 'sub-*', 'eeg', '*_eeg.bdf'))):
        s = re.findall(r'(sub-\d+)', f)[0]
        if s in d['exclude_subjects'] or (subjects and s not in subjects): continue
        t_sub = time.time(); sl.check(f); evf = f.replace('_eeg.bdf', '_events.tsv'); sl.check(evf)
        trials, cur = [], None
        for r in csv.DictReader(open(evf, encoding='utf-8-sig'), delimiter='\t'):
            tt = r['trial_type']
            if tt in ('animal_image', 'tool_image'):
                cur = dict(cue=int(float(r['sample'])), cat=0 if tt.startswith('animal') else 1, concept=os.path.basename(r['stim_file']).split('_PNG')[0], tasks=[]); trials.append(cur)
            elif cur is not None and tt.endswith('_task'): cur['tasks'].append((int(float(r['sample'])), tt.split('_', 1)[1].replace('_task', '')))
        raw = mne.io.read_raw_bdf(f, preload=False, verbose=False); assert raw.info['sfreq'] == fs0
        assert raw.ch_names[:64] == [f'A{i}' for i in range(1, 33)] + [f'B{i}' for i in range(1, 33)]
        stop = min(raw.n_times, max(t['tasks'][-1][0] for t in trials if t['tasks']) + int(8 * fs0))
        X, Pre = [], []
        for i in range(0, 64, 8):
            z = raw.get_data(picks=list(range(i, i + 8)), stop=stop) * 1e6
            Pre.append(np.stack([cl(z[:, t['cue'] - npre:t['cue']], padlen=npre - 1) for t in trials], 0)); X.append(cl(z)); del z
        X = np.concatenate(X); Pre = np.concatenate(Pre, 1); X, bad = fix_bad(c, X, names); X = car(X); T = X.shape[1]
        Pre = interp_batch(Pre, names, bad); Pre = Pre - Pre.mean(1, keepdims=True)
        if bad: w.header.setdefault('bad_channels', {})[s] = bad
        xs, rows, prow, pk = [], [], [], []
        for ti, t in enumerate(trials):
            base_r = dict(subject=s, session=0, run=0, label=t['cat'], concept=t['concept'], clip_id=-1, t_order=ti / len(trials))
            for k, (smp, name) in enumerate(t['tasks']):
                i0 = int(round(smp / fs0 * fs)) + a0
                if i0 < 0 or i0 + L > T: continue
                xs.append(X[:, i0:i0 + L]); rows.append(dict(base_r, trial=ti * 10 + k, kind='task', task=name, task_pos=k, t_onset=smp / fs0))
            prow.append(dict(base_r, trial=ti * 10, kind='pre', task='pre', task_pos=-1, t_onset=t['cue'] / fs0)); pk.append(ti)
        w.add(np.stack(xs), rows); w.add(pad_to(Pre[pk], L), prow); del X
        print(json.dumps(dict(subject=s, n=w.n, trials=len(trials), seconds=round(time.time() - t_sub, 1))), flush=True)
    return w, sl


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--data', required=True); ap.add_argument('--subjects')
    a = ap.parse_args(); c = yaml.safe_load(open(os.path.join(ROOT, a.config))); d = c['datasets'][a.data]
    bank = json.load(open(os.path.join(ROOT, c['positions']))); subjects = a.subjects.split(',') if a.subjects else None
    t0 = time.time(); w, sl = globals()[a.data](c, d, bank, subjects)
    out = w.close(raw_sha256=sl.got, n_raw_files_checked=len(sl.got), subjects_arg=subjects, total_seconds=round(time.time() - t0, 1))
    print(json.dumps(out, default=str)[:3000], flush=True)


if __name__ == '__main__':
    main()
