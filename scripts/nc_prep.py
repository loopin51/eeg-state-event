"""D-128 step 1: label-free epoch caches for REVE-NC (docs/08 §3.1). No scores are computed.

usage: env/bin/python scripts/nc_prep.py configs/nc_prep.yaml --data physionet_mi|cho2017|bciciv2a|faced [--subjects a,b]

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
        self.notch = [iirnotch(f, c['notch_q'], fs=fs_in) for f in np.arange(line_hz, min(fs_in / 2, c['band_hz'][1] + 5), line_hz)]
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


def slices(c, d, t0):
    """Config windows (seconds relative to the event) -> sample slices inside an epoch starting at t0 s."""
    fs, out = c['sfreq'], {}
    for k, w in d['windows'].items():
        if w is None: out[k] = None; continue
        s = w['slice_s']; s = s if isinstance(s[0], list) else [s]
        sl = [[int(round((a - t0) * fs)), int(round((b - t0) * fs))] for a, b in s]
        out[k] = dict(slice=sl[0] if len(sl) == 1 else sl, rows=w['rows'], seconds=w['slice_s'])
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--data', required=True); ap.add_argument('--subjects')
    a = ap.parse_args(); c = yaml.safe_load(open(os.path.join(ROOT, a.config))); d = c['datasets'][a.data]
    bank = json.load(open(os.path.join(ROOT, c['positions']))); subjects = a.subjects.split(',') if a.subjects else None
    t0 = time.time(); w, sl = globals()[a.data](c, d, bank, subjects)
    out = w.close(raw_sha256=sl.got, n_raw_files_checked=len(sl.got), subjects_arg=subjects, total_seconds=round(time.time() - t0, 1))
    print(json.dumps(out, default=str)[:3000], flush=True)


if __name__ == '__main__':
    main()
