"""D-113 arm c inputs (server CPU, gpuq): the 30-channel pipeline control of section 6.4 and the model copies for k3s.

--part things: for every target of --targets, the development set A inputs (fv_train.py mode dev, set A_ses4: per development
  concept present in the target's session 4, the mean over all its images x repetitions of the whitened image means).
    X63  : the 63-channel input as the F-V model sees it (whitened with the session's 63-channel W).
    X30  : the same concepts as 30-channel input (D-113 6.2, 6.4): the 63-channel whitening is undone with sigma^(1/2)
           (X_raw = sigma^(1/2) X_white, sigma symmetric), the 30 channels of channels30 are kept, and W30 is estimated
           from those 30 channels of the session's training-run single epochs with the F-V MVNN (ese.things.preprocess.mvnn_fit,
           "epochs" variant, Ledoit-Wolf, condition mean then grand mean; no label beyond the image code). The 30 channels sit
           in their 63-channel slots, the other 33 slots are 0 and their positions INVALID (bm masks them; nice cannot take it).
  No test file is read (training-run caches only; data.session_mean refuses anything else).
--part gao (D-113 6.2): per subject x session, the Gao bdf runs (1000 Hz, Status dropped; code shipped with the data is not
  run) -> Fz reference (x - x_Fz) -> the 30 channels -> causal first-order Butterworth high-pass at 0.01 Hz (as the THINGS
  online filter; initial state matched to the first sample) -> the F-V FIR low-pass (ese.things.preprocess.fir_lowpass) ->
  epochs -0.2..1.0 s on the 4-ms grid at tau, baseline -0.2..0 s. tau = m - 6 s (perception, D-113 4.1; ERP shift of
  erp_check.json added) and m + 1, 2, 3 s (imagery). W30 = mvnn_fit over every epoch of the session (all windows and runs),
  each epoch its own condition (no label). Model input 0..1 s, stored compact [n, 30, 250] float16 (the infer job puts them in
  the 63-channel slots). Labels come from ese.gao.design.load_trials (bad runs kept in the file with bad = 1; they are left
  out at scoring). Report-only item means (D-113 6.2 last item): per (task, item) of the session, the mean of the whitened
  epochs of the non-bad trials, per window.
--part models --runs DIR [DIR ...]: copies shared.pt, layers.pt and config.yaml of each run folder into stage_dir/models/<run>/.
Every call rewrites stage_dir/manifest.json (sha256 and bytes of every input npz and every file under stage_dir/models/).
usage (gpuq): env/bin/python -I scripts/fv_gao_c_prep.py configs/fv_gao_c.yaml --part things [--targets 1 5 9]
              env/bin/python -I scripts/fv_gao_c_prep.py configs/fv_gao_c.yaml --part gao [--subjects sub-01 ...]
              env/bin/python -I scripts/fv_gao_c_prep.py configs/fv_gao_c.yaml --part models --runs results/runs/<run> ...
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import argparse, hashlib, json, resource, shutil, sys, time
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
from ese.things.data import Meta, subject_mean, concept_mean
from ese.things.preprocess import mvnn_fit, fir_lowpass, grid


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def sqrtm_sym(S):
    w, V = np.linalg.eigh(S); assert w.min() > 0, f'sigma not positive definite ({w.min()})'
    return (V * w ** 0.5) @ V.T


def apply(M, X, out=None):
    """M [D, C] on every item of X [n, C, T] in chunks (float64 product, float32 result)"""
    out = np.empty((len(X), M.shape[0], X.shape[2]), np.float32) if out is None else out
    for a in range(0, len(X), 1024): out[a:a + 1024] = np.einsum('dc,nct->ndt', M, np.asarray(X[a:a + 1024], np.float64))
    return out


def things(c, tc, pc, targets, log):
    cache = P(pc['cache_dir']); meta = Meta(P(tc['meta'])); dev_c = meta.dev_concepts(tc['dev']['salt'], tc['dev']['n'])
    dci = np.array([meta.train_concepts.index(x) for x in dev_c]); dev_codes = meta.codes_of(dev_c)
    ch63 = json.load(open(os.path.join(cache, 'channels.json')))['ch_names']; pos = np.load(P(tc['positions']))
    ch30 = c['channels30']; ci = np.array([ch63.index(x) for x in ch30]); assert len(set(ci)) == 30
    pos30 = np.full_like(pos, c['invalid']); pos30[ci] = pos[ci]; s = c['pipeline_control']['session']; ws = c['pipeline_control']['w30_session']
    out = []
    for t in targets:
        t0 = time.time(); d = os.path.join(cache, f'sub-{t:02d}')
        m = np.load(os.path.join(d, f'ses-{ws:02d}_mvnn.npz')); assert list(m['ch_names']) == ch63, 'mvnn channel order differs'
        R = sqrtm_sym(m['sigma']); assert np.allclose(R @ m['W'], np.eye(63), atol=1e-6), 'sigma^(1/2) W != I'
        ev = np.load(os.path.join(d, f'ses-{ws:02d}_training_events.npz')); E = np.load(os.path.join(d, f'ses-{ws:02d}_training.npy'), mmap_mode='r')
        raw30 = apply(R[ci], E); S30, W30 = mvnn_fit(raw30, ev['code'], m['times'], pc); del raw30
        assert ws == s, 'W30 must come from the evaluated session (D-113 6.4)'
        codes, X, nrep = subject_mean(cache, t, [s], codes=dev_codes); keep, Xc = concept_mean(codes, X, nrep, meta.img_concept, dci); del X
        true = np.array([list(dci).index(j) for j in keep])
        X30 = np.zeros_like(Xc); X30[:, ci] = apply(W30 @ R[ci], Xc)
        f = os.path.join(P(c['stage_dir']), f'things_t{t:02d}.npz')
        np.savez(f, target=t, session=s, concepts=keep, true=true, X63=Xc.astype(np.float16), X30=X30.astype(np.float16), pos63=pos.astype(np.float32),
                 pos30=pos30.astype(np.float32), ch30_index=ci, W30=W30, sigma30=S30)
        r = dict(target=t, file=os.path.relpath(f, ROOT), n_concepts=int(len(keep)), n_missing=int(len(dci) - len(keep)), n_epochs_w30=int(len(ev['code'])),
                 n_conditions_w30=int(len(np.unique(ev['code']))), w30_min_eig=float(np.linalg.eigvalsh(S30).min()), seconds=time.time() - t0)
        log(json.dumps(r)); out.append(r)
    return out


def gao(c, tc, pc, subjects, log):
    import csv, glob, re, warnings
    import mne
    from scipy import signal
    from ese.gao.design import load_trials
    warnings.filterwarnings('ignore'); mne.set_log_level('ERROR')
    g = yaml.safe_load(open(P(c['gao']['config']))); bids = P(g['bids']); bad = {tuple(r) for r in g['bad_runs']}
    shift = float(json.load(open(P(os.path.join(g['out'], 'erp_check.json'))))['shift_s'])
    T, runs = load_trials(bids, dict(g, bad_runs=[])); rk = {r['key']: i for i, r in enumerate(runs)}   # every run, labels of all trials
    cache = P(pc['cache_dir']); ch63 = json.load(open(os.path.join(cache, 'channels.json')))['ch_names']; pos = np.load(P(tc['positions']))
    ch30 = c['channels30']; ci = np.array([ch63.index(x) for x in ch30]); pos30 = np.full_like(pos, c['invalid']); pos30[ci] = pos[ci]
    wins = dict(perc=c['gao']['perception_offset_s'] + shift, **{f'im{k + 1}': o for k, o in enumerate(c['gao']['imagery_offsets_s'])})
    bdfs = sorted(glob.glob(os.path.join(bids, 'sub-*', 'ses-*', 'eeg', '*_eeg.bdf'))); out = []
    sess = sorted({re.search(r'(sub-\d+)_(ses-\d+)_', os.path.basename(b)).groups() for b in bdfs})
    for sub, ses in sess:
        if subjects and sub not in subjects: continue
        t0 = time.time(); E = {w: [] for w in wins}; rows = []
        for bdf in [b for b in bdfs if f'{sub}_{ses}_' in os.path.basename(b)]:
            task = re.search(r'task-([A-Z]+)_eeg', bdf).group(1); key = (sub, ses, task)
            raw = mne.io.read_raw_bdf(bdf, preload=True, exclude=['Status']); fs = raw.info['sfreq']; assert fs == 1000 and len(raw.ch_names) == 32
            x = raw.get_data().astype(np.float64); names = raw.ch_names; del raw
            x = x - x[names.index('Fz')][None]; x = x[[names.index(ch) for ch in ch30]]
            sos = signal.butter(1, c['gao']['highpass_hz'], 'highpass', fs=fs, output='sos'); zi = signal.sosfilt_zi(sos)
            x = np.stack([signal.sosfilt(sos, ch, zi=zi * ch[0])[0] for ch in x])
            h = fir_lowpass(pc, fs); x = np.stack([signal.oaconvolve(ch, h, mode='same') for ch in x])
            off, times = grid(pc, fs)
            with open(bdf[:-len('_eeg.bdf')] + '_events.tsv', newline='') as f: m = np.array([int(round(float(r['onset']) * fs)) for r in csv.DictReader(f, delimiter='\t')])
            ii = np.flatnonzero(T['run'] == rk[key]); assert len(ii) == len(m)
            for w, o in wins.items():
                idx = (m + int(round(o * fs)))[:, None] + off[None]; assert idx.min() >= 0 and idx.max() < x.shape[1], (key, w)
                ep = x[:, idx].transpose(1, 0, 2); nb = int((times < 0).sum()); E[w].append((ep - ep[..., :nb].mean(-1, keepdims=True)).astype(np.float32))
            rows += [dict(task=task, pos=int(T['pos'][i]), label=int(T['label'][i]), block=int(T['block'][i]), bad=int(key in bad)) for i in ii]
            del x
        E = {w: np.concatenate(v) for w, v in E.items()}; allE = np.concatenate(list(E.values()))
        S30, W30 = mvnn_fit(allE, np.arange(len(allE)), times, pc); del allE
        t_in = times >= pc['epoch']['input_tmin'] - 1e-9; res = dict(subject=sub, session=ses, ch30_index=ci, pos30=pos30.astype(np.float32), W30=W30, sigma30=S30,
                                                                     task=np.array([r['task'] for r in rows]), pos=np.array([r['pos'] for r in rows]),
                                                                     label=np.array([r['label'] for r in rows]), block=np.array([r['block'] for r in rows]), bad=np.array([r['bad'] for r in rows]))
        for w in wins:
            Xw = np.einsum('dc,nct->ndt', W30, E[w][..., t_in].astype(np.float64)); res[f'X30_{w}'] = Xw.astype(np.float16)
            keys = sorted({(r['task'], r['label']) for r, b in zip(rows, res['bad']) if not b and r['label'] >= 0})
            res[f'X30_{w}_itemavg'] = np.stack([Xw[[j for j, r in enumerate(rows) if not r['bad'] and (r['task'], r['label']) == k]].mean(0) for k in keys]).astype(np.float16)
            res['itemavg_task'] = np.array([k[0] for k in keys]); res['itemavg_label'] = np.array([k[1] for k in keys])
            res['itemavg_n'] = np.array([sum(1 for r in rows if not r['bad'] and (r['task'], r['label']) == k) for k in keys])
        f = os.path.join(P(c['stage_dir']), f'gao_{sub}_{ses}.npz'); np.savez(f, **res)
        r = dict(subject=sub, session=ses, file=os.path.relpath(f, ROOT), n_trials=len(rows), n_bad=int(res['bad'].sum()), n_items=int(len(res['itemavg_n'])),
                 w30_min_eig=float(np.linalg.eigvalsh(S30).min()), whitened_sd=float(res['X30_perc'].astype(np.float32).std()), seconds=time.time() - t0)
        log(json.dumps(r)); out.append(r)
    return out


def models(c, runs, log):
    out = []
    for r in runs:
        r = P(r).rstrip('/'); dst = os.path.join(P(c['stage_dir']), 'models', os.path.basename(r)); os.makedirs(dst, exist_ok=True)
        for f in ('shared.pt', 'layers.pt', 'config.yaml'): shutil.copy2(os.path.join(r, f), os.path.join(dst, f))
        run = yaml.safe_load(open(os.path.join(dst, 'config.yaml')))['run']; out.append(dict(run=os.path.basename(r), **{k: run[k] for k in ('target', 'seed', 'candidate', 'mode', 'shuffle', 'smoke')}))
        log(json.dumps(out[-1]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--part', choices=['things', 'gao', 'models'], required=True)
    ap.add_argument('--targets', type=int, nargs='+'); ap.add_argument('--runs', nargs='+'); ap.add_argument('--subjects', nargs='+'); a = ap.parse_args()
    c = yaml.safe_load(open(P(a.config))); tc = yaml.safe_load(open(P(c['train_config']))); pc = yaml.safe_load(open(P(tc['prep_config'])))
    sd = P(c['stage_dir']); os.makedirs(sd, exist_ok=True); t0 = time.time(); log = lambda s: print(s, flush=True)
    if a.part == 'things': res = things(c, tc, pc, a.targets or tc['dev']['dev_targets'], log)
    elif a.part == 'gao': res = gao(c, tc, pc, a.subjects, log)
    else: res = models(c, a.runs, log)
    files = sorted(os.path.relpath(os.path.join(dp, f), sd) for dp, _, fs in os.walk(sd) for f in fs if f.endswith('.npz') or os.path.relpath(dp, sd).split(os.sep)[0] == 'models')
    man = dict(files={f: dict(sha256=sha256(os.path.join(sd, f)), bytes=os.path.getsize(os.path.join(sd, f))) for f in files})
    json.dump(man, open(os.path.join(sd, 'manifest.json'), 'w'), indent=1)
    print(json.dumps(dict(part=a.part, items=len(res), files=len(files), seconds=round(time.time() - t0, 1), peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3))))


if __name__ == '__main__':
    main()
