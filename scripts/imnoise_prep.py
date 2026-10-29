"""D-126 epoch caches for the seven data sets (no label-based score is computed here).

usage: python scripts/imnoise_prep.py configs/imnoise.yaml --data TOL|BCI|UGR|SEM|IM36|IM36A|GAO [--only <sub>]

Output per subject: <cache_dir>/<ID>/<sub>.npz. A "view" is a set of epochs with its own trial table:
  main (no prefix)  X [n, C, T] float32 (microvolts), t [T], fs, ch [C], y, task, K [n_task], cvset, fold, role, pblock, tpos
  pc_*, pre_*       the same fields for the positive-control / pre-stimulus epochs when they are not windows of X
  art [n, Ca, T]    separate artefact channels of the main epochs (TOL EXG, SEM EOG); otherwise art_names = proxy EEG channels
Trial table: training of a test fold f uses role in {0, 2}, same cvset, fold != f; testing uses role in {1, 2}. Permutations
shuffle runs of equal labels inside pblock (array order = recording order inside each pblock). IM36 / IM36A also carry
concept_id and concept_cat (permutations remap concepts to categories) and the low-level incongruent flag.
Sources: TOL derivatives epochs (resampled 256 -> 128 Hz); BCI and UGR caches of D-125; SEM / IM36 / IM36A caches written by
scripts/sem_prep.py / im36_prep.py with configs/imnoise_sem.yaml / imnoise_im36.yaml (D-121 / D-124 rules, waveforms saved);
GAO continuous runs of scripts/fv_gao_prep.py (D-113 5.1), epochs [-9.5, 4.5] s around the imagery onset, 250 -> 125 Hz.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, time, hashlib, argparse
import numpy as np, yaml
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def runs_of(gaps_idx, n):
    """run index per trial from the positions of large gaps (gap after trial i starts a new run)"""
    r = np.zeros(n, int)
    for g in gaps_idx: r[g + 1:] += 1
    return r


def tol(c, s, only):
    import mne
    root = P(s['raw']); out = {}
    for sd in sorted(glob.glob(os.path.join(root, 'derivatives', 'sub-*'))):
        sub = os.path.basename(sd)
        if only and sub != only: continue
        X, A, y, run, ses, tpos = [], [], [], [], [], []; nrun = 0
        for si, d in enumerate(sorted(glob.glob(os.path.join(sd, 'ses-*')))):
            p = f'{sub}_{os.path.basename(d)}'
            ev = np.load(os.path.join(d, f'{p}_events.dat'), allow_pickle=True)
            e = mne.read_epochs(os.path.join(d, f'{p}_eeg-epo.fif'), verbose=False); x = mne.read_epochs(os.path.join(d, f'{p}_exg-epo.fif'), verbose=False)
            assert len(e) == len(ev) == len(x) and e.info['sfreq'] == 256 and abs(e.tmin + 0.5) < 1e-6, (p, len(e), len(ev))
            gaps = np.diff(ev[:, 0]); r = runs_of(np.flatnonzero(gaps > 3 * np.median(gaps)), len(ev))
            keep = ev[:, 2] == s['condition']
            for rr in np.unique(r[keep]): assert len(set(ev[r == rr, 2])) == 1, (p, rr)   # a run holds one condition
            ch = e.ch_names
            X.append(e.get_data()[keep] * 1e6); A.append(x.get_data()[keep] * 1e6); y.append(ev[keep, 1]); ses.append(np.full(keep.sum(), si))
            rk = r[keep]; _, rk = np.unique(rk, return_inverse=True); run.append(rk + nrun); nrun += rk.max() + 1
            tpos.append(ev[keep, 0] / 1024.0 + 1e5 * si)
        dn = int(round(256 / s.get('resample', 128)))                 # D-126: 256 -> 128 Hz; D-127: kept at 256 Hz
        X = resample_poly(np.concatenate(X), 1, dn, axis=-1).astype(np.float32); A = resample_poly(np.concatenate(A), 1, dn, axis=-1).astype(np.float32)
        run = np.concatenate(run); n = len(run); fs = 256.0 / dn
        out[sub] = dict(X=X, t=-0.5 + np.arange(X.shape[-1]) / fs, fs=fs, ch=np.array(ch), y=np.concatenate(y).astype(int),
                        task=np.zeros(n, int), K=np.array([4]), cvset=np.zeros(n, int), fold=run, role=np.full(n, 2), pblock=run,
                        tpos=np.concatenate(tpos), session=np.concatenate(ses), art=A, art_names=np.array(['EXG%d' % i for i in range(1, 9)]))
    return out


def bci(c, s, only):
    out = {}
    for f in sorted(glob.glob(os.path.join(P(s['cache']), 'sub-*.npz'))):
        sub = os.path.basename(f)[:-4]
        if only and sub != only: continue
        z = np.load(f); sp = z['split'].astype(int); n = len(sp)
        out[sub] = dict(X=z['x'].astype(np.float32), t=np.arange(z['x'].shape[-1]) / 200.0, fs=200.0, ch=z['ch_names'], y=z['y'].astype(int),
                        task=np.zeros(n, int), K=np.array([5]), cvset=np.zeros(n, int), fold=np.where(sp == 1, 0, -1), role=np.where(sp == 1, 1, 0),
                        pblock=sp, tpos=np.arange(n, dtype=float), art_names=np.array(c['artefact_channels']['BCI']))
        if s.get('pre_raw_config'):                                    # post-result diagnosis of D-126: pre-cue meter L for BCI
            pre = bci_pre(s, sub); assert np.array_equal(pre['y'], out[sub]['y']), sub
            out[sub].update({'pre_' + k: v for k, v in dict(out[sub], X=pre['X'], t=pre['t']).items() if k not in ('ch', 'art_names')})
    return out


def bci_pre(s, sub):
    """pre-cue epochs [-0.5, 0) s of the BCI2020 raw trials: the raw trial is cut first, then the D-125 chain (Butterworth band,
    notches, common average, 200 Hz), so no post-cue signal reaches the window through the zero-phase filters"""
    from scipy.io import loadmat
    from scipy.signal import butter, filtfilt, iirnotch
    b = yaml.safe_load(open(P(s['pre_raw_config'])))['bci2020']; base = P(b['raw']); sid = int(sub.split('-')[1]); xs, ys = [], []
    for folder, key in (('Training set', 'epo_train'), ('Validation set', 'epo_validation')):
        e = loadmat(os.path.join(base, folder, f'Data_Sample{sid:02d}.mat'))[key][0, 0]; x = np.transpose(e['x'], (2, 1, 0)).astype(np.float64)
        fs = float(e['fs'][0, 0]); x = x[..., :int(round(-b['epoch_start_s'] * fs))]
        bb, ab = butter(b['butter_order'], b['band_hz'], btype='band', fs=fs); x = filtfilt(bb, ab, x, axis=-1)
        for f0 in b['notch_hz']:
            bn, an = iirnotch(f0, b['notch_q'], fs=fs); x = filtfilt(bn, an, x, axis=-1)
        x = x - x.mean(1, keepdims=True)
        up, down = int(b['sfreq']), int(fs); g = np.gcd(up, down); x = resample_poly(x, up // g, down // g, axis=-1)
        xs.append(x.astype(np.float32)); ys.append(e['y'].argmax(0))
    X = np.concatenate(xs); return dict(X=X, t=b['epoch_start_s'] + np.arange(X.shape[-1]) / float(b['sfreq']), y=np.concatenate(ys).astype(int))


def ugr(c, s, only):
    out = {}; cues = ['Text', 'Audio', 'Pictures']
    for f in sorted(glob.glob(os.path.join(P(s['cache']), 'sub-*.npz'))):
        sub = os.path.basename(f)[:-4]
        if only and sub != only: continue
        z = np.load(f); sess = sorted(set(z['session'].tolist())); si = np.array([sess.index(v) for v in z['session']])
        cue = np.array([cues.index(v) for v in z['cue']]); t = -0.5 + np.arange(z['speech'].shape[-1]) / 200.0
        def view(sel):
            n = int(sel.sum())
            return dict(X=z['speech'][sel].astype(np.float32), t=t, fs=200.0, ch=z['ch_names'], y=z['cat'][sel].astype(int), task=np.zeros(n, int),
                        K=np.array([6]), cvset=si[sel], fold=cue[sel], role=np.full(n, 2), pblock=si[sel] * 3 + cue[sel],
                        tpos=np.flatnonzero(sel).astype(float), word=z['word'][sel].astype(int))
        main = (z['mode'] == 1) & (z['word'] >= 0); real = (z['mode'] == 0) & (z['word'] >= 0)
        d = view(main); d['art_names'] = np.array(c['artefact_channels']['UGR'])
        d.update({'pc_' + k: v for k, v in view(real).items()})
        pre = view(main); pre['X'] = z['rest'][main].astype(np.float32)                       # rest epoch: stimulus - 1.5 .. stimulus
        d.update({'pre_' + k: v for k, v in pre.items()})
        out[sub] = d
    return out


def sem(c, s, only):
    sc = yaml.safe_load(open(P(s['prep_config']))); files = sorted(glob.glob(os.path.join(P(sc['cache_dir']), 'sub-*_ses-*.npz')))
    fs = sc['prep']['fs_raw'] / sc['prep']['decim']
    subs = sorted({os.path.basename(f).split('_')[0] for f in files}); out = {}
    for sub in subs:
        if only and sub != only: continue
        parts = {'m': [], 'p': []}
        for k, f in enumerate([f for f in files if os.path.basename(f).startswith(sub + '_')]):
            z = np.load(f)
            for v, ok, X, tt in (('m', z['i_ok'], z['i_ep'], z['i_t']), ('p', z['p_ok'], z['p_ep'], z['p_t'])):
                n = int(ok.sum())
                parts[v].append(dict(X=X[ok], t=tt, y=z['concept'][ok], modality=z['modality'][ok], cvset=np.full(n, k), fold=z['block'][ok],
                                     pblock=k * 100 + z['block'][ok], tpos=z['onset'][ok] + 1e5 * k, art=z['i_eog_ep'][ok] if v == 'm' else None))
        def cat(v):
            L = parts[v]; n = sum(len(p['y']) for p in L)
            d = dict(X=np.concatenate([p['X'] for p in L]).astype(np.float32), t=L[0]['t'], fs=fs, ch=np.array(['e%d' % i for i in range(L[0]['X'].shape[1])]),
                     y=np.concatenate([p['y'] for p in L]).astype(int), task=np.zeros(n, int), K=np.array([3]), role=np.full(n, 2),
                     **{k: np.concatenate([p[k] for p in L]) for k in ('modality', 'cvset', 'fold', 'pblock', 'tpos')})
            if v == 'm': d['art'] = np.concatenate([p['art'] for p in L]).astype(np.float32); d['art_names'] = np.array(['HEOGR', 'HEOGL', 'VEOGU', 'VEOGL'])
            return d
        d = cat('m'); pc = cat('p'); d.update({'pc_' + k: v for k, v in pc.items()})
        out[sub] = d
    return out


def im36(c, s, only):
    ic = yaml.safe_load(open(P(s['prep_config']))); ds = s['ds']; fs = ic['prep']['fs_raw'] / ic['prep']['decim']
    low = json.load(open(P('results/im36/lowlevel.json'))); concepts = list(low['concepts']); ccat = np.array(low['category'])
    fold_of = low['fold'] if isinstance(low['fold'], dict) else dict(zip(concepts, low['fold'])); inc = set(low['incongruent'])
    out = {}
    for f in sorted(glob.glob(os.path.join(P(ic['cache_dir']), f'{ds}_sub-*.npz'))):
        sub = os.path.basename(f)[len(ds) + 1:-4]
        if only and sub != only: continue
        z = np.load(f); ok = z['t_ok']; cid = np.array([concepts.index(v) for v in z['concept']])
        assert (ccat[cid] == z['category']).all()
        def view(X, tt, okm, cidv, onset):
            n = int(okm.sum()); ci = cidv[okm]
            return dict(X=X[okm].astype(np.float32), t=tt, fs=fs, ch=z['ch_names'], y=ccat[ci], task=np.zeros(n, int), K=np.array([2]),
                        cvset=np.zeros(n, int), fold=np.array([fold_of[concepts[i]] for i in ci]), role=np.full(n, 2), pblock=np.zeros(n, int),
                        tpos=onset[okm], concept_id=ci, concept_cat=ccat, incongruent=np.array([concepts[i] in inc for i in ci]))
        d = view(z['task_ep'], z['task_t'], ok, cid, z['onset'] + z['delay']); d['art_names'] = np.array(c['artefact_channels'][s.get('id', 'IM36')])
        d['task_name'] = z['task'][ok]; d['delay'] = z['delay'][ok]
        tcid = np.array([concepts.index(v) for v in z['trial_concept']])
        pc = view(z['cue_ep'], z['cue_t'], z['cue_trial_ok'], tcid, z['trial_onset'])
        d.update({'pc_' + k: v for k, v in pc.items()})
        out[sub] = d
    return out


def gao(c, s, only):
    from ese.gao.design import load_trials
    gc = yaml.safe_load(open(P(s['prep_config']))); T, runs = load_trials(P(gc['bids']), gc); cache = P(gc['cache'])
    lo, hi = s['epoch']; out = {}; tasks = list(gc['tasks'])
    for sub in sorted(set(T['sub'])):
        if only and sub != only: continue
        X, keep = [], []
        for ri in sorted(set(T['run'][T['sub'] == sub])):
            key = runs[ri]['key']; name = '%s_%s_%s' % key
            meta = json.load(open(os.path.join(cache, name + '.json'))); D = np.load(os.path.join(cache, name + '.npy'), mmap_mode='r'); fs = meta['sfreq']
            idx = np.flatnonzero((T['run'] == ri) & (T['label'] >= 0)); m = np.array(meta['m'])[T['pos'][idx]]
            a, b = int(round(lo * fs)), int(round(hi * fs)); good = (m + a >= 0) & (m + b <= D.shape[1])
            X.append(np.stack([np.asarray(D[:, mm + a:mm + b]) for mm in m[good]]) * 1e6); keep.append(idx[good])
        rs = s.get('resample', 125); g = int(np.gcd(int(fs), int(rs)))                # D-126: 250 -> 125 Hz; D-127: kept at 250 Hz
        keep = np.concatenate(keep); X = resample_poly(np.concatenate(X), int(rs) // g, int(fs) // g, axis=-1).astype(np.float32)
        n = len(keep); run = T['run'][keep]
        out[sub] = dict(X=X, t=lo + np.arange(X.shape[-1]) / float(rs), fs=float(rs), ch=np.array(meta['ch_names']), y=T['label'][keep].astype(int),
                        task=np.array([tasks.index(v) for v in T['task'][keep]]), K=np.array([len(gc['tasks'][t]['items']) for t in tasks]),
                        cvset=run, fold=T['block'][keep], role=np.full(n, 2), pblock=run * 10 + T['block'][keep], tpos=T['onset'][keep] + 1e4 * run,
                        art_names=np.array(c['artefact_channels']['GAO']))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--data', required=True); ap.add_argument('--only', default='')
    a = ap.parse_args(); c = yaml.safe_load(open(P(a.config))); s = dict(c['sources'][a.data]); s['id'] = a.data
    fn = dict(TOL=tol, BCI=bci, UGR=ugr, SEM=sem, IM36=im36, IM36A=im36, GAO=gao)[a.data]
    od = os.path.join(P(c['cache_dir']), a.data); os.makedirs(od, exist_ok=True); t0 = time.time(); log = []
    for sub, d in fn(c, s, a.only).items():
        f = os.path.join(od, f'{sub}.npz'); np.savez(f, **d)
        meta = dict(sub=sub, n=int(len(d['y'])), shape=list(d['X'].shape), labels=np.bincount(d['y']).tolist(), folds=int(len(set(d['fold'][d['role'] > 0].tolist()))),
                    pc=int(len(d['pc_y'])) if 'pc_y' in d else None, sha256=hashlib.sha256(open(f, 'rb').read()).hexdigest(), seconds=round(time.time() - t0, 1))
        log.append(meta); print(json.dumps(meta), flush=True)
    if not a.only:
        json.dump(dict(config=a.config, data=a.data, subjects=log), open(os.path.join(od, 'prep.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
