"""D-120 section 1: THINGS-EEG1 (ds003825 v1.2.0) epochs of the EEG2 training photos, per-subject standardisation, group mean,
group whitening. Server gpuq CPU, one job for all subjects (one subject in memory at a time).

Per subject (participants.tsv exclude == 0): the authors' derivative (eeglab .set/.fdt: 0.1-100 Hz FIR, average reference,
250 Hz, 64 channels); the k-th 'E  1' marker <-> the k-th events.tsv row (counts equal, |latency - 1 - onset / 4| < 1 sample,
else the subject is dropped and recorded); main-part rows (isteststim == 0) whose stimname equals an EEG2 training photo file
name are epoched -0.1 .. 1.0 s (275 samples), baseline -0.1 .. 0 s subtracted, standardised per channel with the mean / sd
over the subject's training + validation concept epochs (all samples). Development (= test) photos of the subject go to
dev/subjects_dev.npy [S, 2000, 64, 250] float16 (input window) with a presence mask; every photo is added to the group sum.
Group: mean over the subjects that saw the photo; W = (mean per-epoch Ledoit-Wolf covariance of the TRAINING-concept group
epochs, -0.1 .. 1.0 s)^(-1/2). Outputs (cache_dir): group_trainvalid.npy float16 [n, 64, 250] whitened + codes, W.npy, stats,
positions.npy, channels.json, dev/ (group_dev.npy whitened, subjects_dev.npy, mask, rows), prep.json; results/mi_eeg1/prep.json
and files_sha256.tsv. Development-photo EEG is written but never summarised here (no statistic over it is printed or stored).
usage: python scripts/mi_eeg1_prep.py configs/mi_eeg1.yaml [--smoke]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, csv, json, time, hashlib, argparse, resource, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.things.data import Meta
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def positions(names):
    from ese.listen.positions import template_3d, common_positions
    ref = template_3d('standard_1005'); low = {k.lower(): k for k in ref}; b64 = set(template_3d('biosemi64'))
    miss = [n for n in names if n.lower() not in low]; assert not miss, f'channels without a standard_1005 position: {miss}'
    pos3d = np.array([ref[low[n.lower()]] for n in names]); anchors = {n: b for n in names for b in b64 if b.lower() == n.lower()}
    return common_positions(names, pos3d, anchors)


def splits(tc, meta):
    dev_c = meta.dev_concepts(tc['dev']['salt'], tc['dev']['n']); valid_c = meta.valid_concepts(dev_c, tc['valid']['salt'], tc['valid']['fraction'])
    train_c = [x for x in meta.train_concepts if x not in set(dev_c) and x not in set(valid_c)]
    rows = lambda cs: np.flatnonzero(np.isin(np.array(meta.train_img_concept), cs))
    return rows(train_c), rows(valid_c), rows(dev_c), (len(train_c), len(valid_c), len(dev_c))


def read_subject(raw_root, s, c):
    import scipy.io as sio
    base = os.path.join(raw_root, 'derivatives', 'eeglab', f'sub-{s:02d}_task-rsvp_continuous')
    E = sio.loadmat(base + '.set', squeeze_me=True, struct_as_record=False)['EEG']
    assert int(E.srate) == c['epoch']['sfreq'] and E.ref == 'averef', (E.srate, E.ref)
    X = np.fromfile(base + '.fdt', dtype='<f4'); C, T = int(E.nbchan), int(E.pnts); assert X.size == C * T, (X.size, C, T)
    X = X.reshape(T, C).T                                                      # EEGLAB column-major [channels x points]
    names = [ch.labels for ch in E.chanlocs]
    lat = np.array([e.latency for e in E.event if e.type == c['events']['onset_marker']], float)
    rows = list(csv.DictReader(open(os.path.join(raw_root, f'sub-{s:02d}', 'eeg', f'sub-{s:02d}_task-rsvp_events.tsv')), delimiter='\t'))
    files = [base + '.set', base + '.fdt', os.path.join(raw_root, f'sub-{s:02d}', 'eeg', f'sub-{s:02d}_task-rsvp_events.tsv')]
    return X, names, lat, rows, files


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    c = yaml.safe_load(open(P(a.config))); tc = yaml.safe_load(open(P(c['train_config']))); t0 = time.time(); raw = P(c['raw_root'])
    cache = P(c['cache_dir']) + ('_smoke' if a.smoke else ''); os.makedirs(os.path.join(cache, 'dev'), exist_ok=True)
    out = P(c['out_dir']) + ('_smoke' if a.smoke else ''); os.makedirs(out, exist_ok=True)
    meta = Meta(P(tc['meta'])); base = [f.split('/')[-1] for f in meta.train_files]; assert len(set(base)) == len(base) == 16540
    row_of = {b: k for k, b in enumerate(base)}
    tr, va, dv, ncon = splits(tc, meta); tv = np.sort(np.r_[tr, va]); is_tv = np.zeros(16540, bool); is_tv[tv] = True; is_tr = np.zeros(16540, bool); is_tr[tr] = True
    dpos = {int(r): k for k, r in enumerate(dv)}
    part = list(csv.DictReader(open(os.path.join(raw, 'participants.tsv')), delimiter='\t'))
    subs = [int(r['participant_id'][4:]) for r in part if r[c['subjects_exclude_column']].strip() == '0']
    excluded = {r['participant_id']: r.get('note_IZ', '') for r in part if r[c['subjects_exclude_column']].strip() != '0'}
    if a.smoke: subs = c['smoke_subjects']
    ep = c['epoch']; sf = ep['sfreq']; o0 = int(round(ep['tmin'] * sf)); n_ep = int(round((ep['tmax'] - ep['tmin']) * sf)); i0 = int(round((ep['input_tmin'] - ep['tmin']) * sf)); nt = ep['n_times']
    assert i0 + nt <= n_ep
    S = len(subs); Gsum = None; Gn = np.zeros(16540, np.int64); log = {}; ch_ref = None; shas = []
    SD = np.lib.format.open_memmap(os.path.join(cache, 'dev', 'subjects_dev.npy'), mode='w+', dtype=np.float16, shape=(S, len(dv), 64, nt))
    mask = np.zeros((S, len(dv)), bool); kept = []
    for si, s in enumerate(subs):
        X, names, lat, rows, files = read_subject(raw, s, c); shas += [(os.path.relpath(f, raw), os.path.getsize(f), fsha(f)) for f in files]
        if ch_ref is None: ch_ref = names
        rec = dict(n_markers=int(len(lat)), n_rows=len(rows))
        if names != ch_ref:                                                    # same channel set in another order -> reorder by name; another set -> drop
            if sorted(names) != sorted(ch_ref): rec['dropped'] = 'channel set differs'; log[f'sub-{s:02d}'] = rec; print(f'sub-{s:02d}: dropped ({rec})', flush=True); continue
            X = X[[names.index(n) for n in ch_ref]]; rec['channels_reordered'] = True
        if len(lat) != len(rows): rec['dropped'] = 'marker / row count mismatch'; log[f'sub-{s:02d}'] = rec; print(f'sub-{s:02d}: dropped ({rec})', flush=True); continue
        off = np.abs(lat - 1 - np.array([float(r['onset']) for r in rows]) / 4.0); rec['max_offset_samples'] = float(off.max())
        if off.max() >= c['events']['max_offset_samples']: rec['dropped'] = 'onset offset'; log[f'sub-{s:02d}'] = rec; print(f'sub-{s:02d}: dropped ({rec})', flush=True); continue
        sel = [(k, row_of[r['stimname']]) for k, r in enumerate(rows) if r['isteststim'] == '0' and r['stimname'] in row_of]
        rec['n_main_rows'] = int(sum(r['isteststim'] == '0' for r in rows)); rec['n_matched'] = len(sel)
        code_rows = np.array([q for _, q in sel]); dup = len(code_rows) - len(np.unique(code_rows)); rec['n_duplicate_photos'] = int(dup); assert dup == 0, f'sub-{s:02d}: a photo shown twice'
        on = np.round(lat[[k for k, _ in sel]] - 1).astype(np.int64) + o0; ok = (on >= 0) & (on + n_ep <= X.shape[1]); rec['n_out_of_range'] = int((~ok).sum())
        on, code_rows = on[ok], code_rows[ok]
        Ep = np.stack([X[:, t:t + n_ep] for t in on]).astype(np.float32); del X
        Ep -= Ep[:, :, :-o0].mean(2, keepdims=True) if o0 < 0 else 0                   # baseline -0.1 .. 0 s
        m_tv = is_tv[code_rows]; mu = Ep[m_tv].mean((0, 2)); sd = Ep[m_tv].std((0, 2)); Ep = (Ep - mu[None, :, None]) / sd[None, :, None]
        if Gsum is None: Gsum = np.zeros((16540, 64, n_ep), np.float32)
        Gsum[code_rows] += Ep; Gn[code_rows] += 1
        md = np.isin(code_rows, dv)
        for r_, e_ in zip(code_rows[md], Ep[md]): SD[si, dpos[int(r_)]] = e_[:, i0:i0 + nt].astype(np.float16); mask[si, dpos[int(r_)]] = True
        rec.update(n_epochs=int(len(code_rows)), n_trainvalid=int(m_tv.sum()), n_dev=int(md.sum()), sd_min=float(sd.min()), sd_max=float(sd.max())); kept.append(s)
        log[f'sub-{s:02d}'] = rec; del Ep
        print(f'sub-{s:02d}: {rec["n_epochs"]} epochs ({time.time() - t0:.0f}s, peak {peak():.2f} GB)', flush=True)
    SD.flush(); np.save(os.path.join(cache, 'dev', 'mask.npy'), mask); np.save(os.path.join(cache, 'dev', 'rows.npy'), dv)
    np.save(os.path.join(cache, 'dev', 'subjects.npy'), np.array(subs)); np.save(os.path.join(cache, 'dev', 'kept.npy'), np.isin(subs, kept))   # rows of subjects_dev; dropped rows stay empty
    have = Gn > 0; G = Gsum / np.maximum(Gn, 1)[:, None, None]; del Gsum
    # whitening from the training-concept group epochs (full -0.1 .. 1.0 s)
    from sklearn.covariance import ledoit_wolf
    trh = np.flatnonzero(is_tr & have); cov = np.zeros((64, 64))
    for r in trh: cov += ledoit_wolf(G[r].T, assume_centered=False)[0]
    cov /= len(trh); ev, U = np.linalg.eigh(cov); W = (U * ev ** -0.5) @ U.T
    Wg = lambda A: np.einsum('ij,njt->nit', W, A[:, :, i0:i0 + nt]).astype(np.float16)
    tvh = np.flatnonzero(is_tv & have); np.save(os.path.join(cache, 'group_trainvalid.npy'), Wg(G[tvh])); np.save(os.path.join(cache, 'group_trainvalid_rows.npy'), tvh)
    np.save(os.path.join(cache, 'group_trainvalid_n.npy'), Gn[tvh]); np.save(os.path.join(cache, 'dev', 'group_dev.npy'), Wg(G[dv])); np.save(os.path.join(cache, 'dev', 'group_dev_n.npy'), Gn[dv])
    np.save(os.path.join(cache, 'W.npy'), W.astype(np.float32)); np.save(os.path.join(cache, 'cov.npy'), cov)
    xy, fit = positions(ch_ref); np.save(os.path.join(cache, 'positions.npy'), xy); json.dump(dict(ch_names=ch_ref, fit=fit), open(os.path.join(cache, 'channels.json'), 'w'), indent=1)
    res = dict(stage='prep', smoke=a.smoke, subjects=kept, dropped=sorted(set(subs) - set(kept)), excluded=excluded, n_subjects=len(kept), concepts=dict(train=ncon[0], valid=ncon[1], dev=ncon[2]),
               photos=dict(train=int(len(tr)), valid=int(len(va)), dev=int(len(dv)), trainvalid_with_eeg=int(len(tvh)), dev_with_eeg=int(have[dv].sum())),
               subjects_per_photo_trainvalid=dict(min=int(Gn[tvh].min()), mean=float(Gn[tvh].mean()), max=int(Gn[tvh].max())),
               whitening=dict(n_epochs=int(len(trh)), eig_min=float(ev.min()), eig_max=float(ev.max())), per_subject=log,
               outputs={f: fsha(os.path.join(cache, f)) for f in ('group_trainvalid.npy', 'group_trainvalid_rows.npy', 'W.npy', 'positions.npy', 'dev/subjects_dev.npy', 'dev/group_dev.npy', 'dev/mask.npy', 'dev/rows.npy', 'dev/kept.npy')},
               seconds=time.time() - t0, peak_rss_gb=peak())
    json.dump(res, open(os.path.join(out, 'prep.json'), 'w'), indent=1)
    with open(os.path.join(out, 'files_sha256.tsv'), 'w') as f:
        for p_, n_, h_ in shas: f.write(f'{p_}\t{n_}\t{h_}\n')
    print(json.dumps({k: res[k] for k in ('n_subjects', 'concepts', 'photos', 'subjects_per_photo_trainvalid', 'whitening')}), flush=True)


if __name__ == '__main__':
    main()
