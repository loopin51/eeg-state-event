"""VERA-CPT (docs/08 §14.1): label-free continued-pretraining corpus, one dataset per run. No scores are computed.

usage: env/bin/python scripts/cpt_prep.py configs/cpt_prep.yaml --data sparrkulee|alice|ds004408|things_eeg1|things_eeg2 [--smoke]
       env/bin/python scripts/cpt_prep.py configs/cpt_prep.yaml --summary     (results/nc/cpt/corpus_summary.json, sha256 of the caches)

Whole recordings, no labels, events, triggers or stimuli. Chain = the VERA evaluation caches (scripts/nc_prep.py: Cleaner,
fix_bad, car, canonical_names; parameters from configs/nc_prep.yaml). Output in <cache_dir>/<name>/ (--smoke: 2 recordings
in <smoke_dir>/<name>/):
  x.f32               float32 [n_samples, C] time-major, recordings back to back, microvolts, 200 Hz (raw binary, little endian)
  recordings.parquet  rec, subject, session, run, offset, length, raw_file, raw_sha256, n_interpolated, interpolated, split, ...
  header.json         ch_names (REVE bank), sfreq, dtype, n_samples, hours, line_hz (+ source), chain, channel_map, excluded, peak RSS
Recordings are processed by a worker pool (`workers` per dataset, one recording per worker at a time); the parent writes them
in listing order, so the output does not depend on the number of workers.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, re, json, glob, gzip, time, shutil, hashlib, argparse, resource, traceback
import multiprocessing as mp
import numpy as np, pandas as pd, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import nc_prep as NP                  # Cleaner, fix_bad, car, canonical_names, _std_montage (the VERA chain)

P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
peak_gb = lambda who=resource.RUSAGE_SELF: resource.getrusage(who).ru_maxrss / 1e6      # Linux: kB


def hashes(path, md5=False, buf=1 << 24):
    h, m = hashlib.sha256(), hashlib.md5() if md5 else None
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(buf), b''):
            h.update(b)
            if m: m.update(b)
    return h.hexdigest(), (m.hexdigest() if m else None)


def bids_line(path):
    """PowerLineFrequency of a BIDS *_eeg.json next to the recording (None if absent or n/a)"""
    if not os.path.exists(path): return None
    v = json.load(open(path)).get('PowerLineFrequency')
    return float(v) if isinstance(v, (int, float)) else None


# ------------------------------------------------------------------ electrode name assignment (TOL rule, nc_prep.tol)
def sphere(M):
    z = np.linalg.lstsq(np.c_[2 * M, np.ones(len(M))], (M ** 2).sum(1), rcond=None)[0]; ctr = z[:3]
    return (M - ctr) / np.sqrt(z[3] + (ctr ** 2).sum())


def sphere_match(Q, B, rounds=10):
    """scripts/nc_prep.py tol(): rows Q and columns B moved to the unit sphere (least-squares sphere fit), aligned by alternating
    a one-to-one minimum-distance assignment (Hungarian) and an orthogonal Procrustes rotation of B; the final assignment gives
    one column per row. Returns column index per row (row order) and the distance in percent of the head radius."""
    from scipy.optimize import linear_sum_assignment
    from scipy.linalg import orthogonal_procrustes
    Q, B = sphere(Q), sphere(B)
    for _ in range(rounds):
        qi, bj = linear_sum_assignment(np.linalg.norm(Q[:, None] - B[None], axis=-1)); B = B @ orthogonal_procrustes(B[bj], Q[qi])[0]
    D = np.linalg.norm(Q[:, None] - B[None], axis=-1); qi, bj = linear_sum_assignment(D); o = np.argsort(qi)
    return bj[o], [round(float(D[i, j]) * 100, 1) for i, j in zip(qi[o], bj[o])]


def biosemi128_to_64(c):
    """10-10 name of the BioSemi 64 cap -> BioSemi 128 electrode (A1..D32), MNE templates biosemi128 and standard_1005"""
    import mne
    b128 = mne.channels.make_standard_montage('biosemi128').get_positions()['ch_pos']; std = NP._std_montage().get_positions()['ch_pos']
    bn = list(b128); j, dist = sphere_match(np.array([std[n] for n in c['names64']]), np.array([b128[n] for n in bn]))
    cmap = dict(zip(c['names64'], [bn[k] for k in j])); dmap = dict(zip(c['names64'], dist))
    ref = P(c['biosemi128_check'])
    if os.path.exists(ref):
        assert json.load(open(ref))['channel_map'] == cmap, 'the 128 -> 64 map differs from the TOL evaluation cache'
    return cmap, dmap, os.path.exists(ref)


OLD_NAMES = {'T3', 'T4', 'T5', 'T6'}     # standard_1005 duplicates of T7, T8, P7, P8


def nearest_1005(names, pos, bank):
    """file electrodes with their own digitized positions -> one standard_1005 name each (REVE bank names only, old 10-20
    duplicates left out), by the TOL rule with the file electrodes as rows"""
    sp = NP._std_montage().get_positions()['ch_pos']; cand = [n for n in sp if n in bank and n not in OLD_NAMES]
    j, dist = sphere_match(np.asarray(pos, float), np.array([sp[n] for n in cand]))
    return dict(zip(names, [cand[k] for k in j])), dict(zip(names, dist))


# ------------------------------------------------------------------ dataset listings (parent) and readers (workers)
# listing -> specs (picklable dicts, one per file); reader(spec) -> sources, one per recording:
#   dict(rec_key, subject, session, run, raw_file, names (output order, bank names), fs, get(idx) -> [len(idx), T] uV float64,
#        close(), info)
def _excl_subject(s, d): return int(re.search(r'\d+', s).group()) in d.get('exclude_subjects', [])


def list_sparrkulee(c, d):
    base = P(d['raw']); md5 = {}
    for l in open(os.path.join(base, d['md5_list'])):
        f = l.rstrip('\n').split('\t')
        if len(f) >= 3: md5[f[1]] = f[2]
    specs, excl = [], []
    for f in sorted(glob.glob(os.path.join(base, d['glob']))):
        rel = os.path.relpath(f, base); s = rel.split('/')[0]
        if _excl_subject(s, d): continue
        if rel in d['exclude_recordings']: excl.append(dict(raw_file=rel, reason=d['exclude_reason'])); continue
        m = re.search(r'(ses-[^_]+)_task-([^_]+)_run-(\d+)', rel)
        specs.append(dict(path=f, rel=rel, subject=s, session=m.group(1), run=f'task-{m.group(2)}_run-{m.group(3)}', md5=md5.get(rel)))
    subs = sorted({x['subject'] for x in specs}); lf = bids_line(os.path.join(base, d['bids_json']))
    assert lf == d['line_hz'], f'{d["bids_json"]}: PowerLineFrequency {lf} != {d["line_hz"]}'
    excl.insert(0, dict(raw_file='sub-' + ',sub-'.join(f'{k:03d}' for k in d['exclude_subjects']), reason='restricted on RDR (not public), configs/ext_prepare.yaml'))
    return specs, excl, dict(n_subjects=len(subs))


def read_sparrkulee(c, d, spec, bank):
    import mne
    sha, md5 = hashes(spec['path'], md5=True)
    assert spec['md5'] and md5 == spec['md5'], f'{spec["rel"]}: md5 {md5} differs from the RDR list {spec["md5"]}'
    tmp = os.path.join(c['tmp_dir'], f'cpt_prep_{os.getpid()}.bdf')
    with gzip.open(spec['path'], 'rb') as a, open(tmp, 'wb') as b: shutil.copyfileobj(a, b, 1 << 24)
    raw = mne.io.read_raw_bdf(tmp, preload=False, verbose=False)
    eeg = [n for n in raw.ch_names if not re.match(c['non_eeg'], n)]; drop = [n for n in raw.ch_names if n not in eeg]
    assert len(eeg) == d['n_eeg'], f'{spec["rel"]}: {len(eeg)} EEG channels'
    alias = d.get('channel_alias', {}); names = NP.canonical_names([alias.get(n, n) for n in eeg], bank)
    yield dict(rec_key=spec['rel'], subject=spec['subject'], session=spec['session'], run=spec['run'], raw_file=spec['rel'], raw_sha256=sha,
               names=names, fs=float(raw.info['sfreq']), get=lambda idx: raw.get_data(picks=[eeg[i] for i in idx]) * 1e6,
               close=lambda: os.remove(tmp), info=dict(dropped=','.join(drop), md5_checked=True,
                                                       renamed=','.join(f'{a}->{b}' for a, b in alias.items() if a in eeg)))


def list_alice(c, d):
    base = P(d['raw']); fs = sorted(glob.glob(os.path.join(base, d['glob'])))
    specs = [dict(path=f, rel=os.path.relpath(f, base), subject=os.path.basename(os.path.dirname(f))) for f in fs]
    return specs, [], dict(n_subjects=len(specs))


def alice_positions(d, raw):
    cp = raw.get_montage().get_positions()['ch_pos']; eeg = sorted([n for n in raw.ch_names if re.match(d['eeg_names'], n)], key=int)
    return eeg, np.array([cp[n] for n in eeg])


def alice_map(d, specs, bank):
    """file channel names '1'..'61' -> standard_1005 names from the digitized positions of the first file (parent, once);
    every reader checks that its file has the same electrodes at the same positions"""
    import mne
    eeg, pos = alice_positions(d, mne.io.read_raw_fif(specs[0]['path'], preload=False, verbose=False))
    m, dist = nearest_1005(eeg, pos, bank); return dict(eeg=eeg, pos=pos.tolist(), map=m, dist=dist)


def read_alice(c, d, spec, bank):
    import mne
    sha, _ = hashes(spec['path']); raw = mne.io.read_raw_fif(spec['path'], preload=False, verbose=False)
    eeg, pos = alice_positions(d, raw); A = d['_map']; m = A['map']; assert len(eeg) == d['n_eeg'], (spec['rel'], len(eeg))
    assert eeg == A['eeg'] and np.allclose(pos, A['pos']), f'{spec["rel"]}: electrode positions differ from the first file'
    names = NP.canonical_names([m[n] for n in eeg], bank); drop = [n for n in raw.ch_names if n not in eeg]
    yield dict(rec_key=spec['rel'], subject=spec['subject'], session='ses-1', run='alice', raw_file=spec['rel'], raw_sha256=sha, names=names,
               fs=float(raw.info['sfreq']), get=lambda idx: raw.get_data(picks=[eeg[i] for i in idx]) * 1e6, close=lambda: None,
               info=dict(dropped=','.join(drop), provider_bads=','.join(raw.info['bads'])))


def list_ds004408(c, d):
    base = P(d['raw']); fs = sorted(glob.glob(os.path.join(base, d['glob'])))
    specs = [dict(path=f, rel=os.path.relpath(f, base), subject=os.path.relpath(f, base).split('/')[0],
                  run=re.search(r'(run-\d+)', f).group(1)) for f in fs]
    return specs, [], dict(n_subjects=len({x['subject'] for x in specs}))


def read_ds004408(c, d, spec, bank):
    import mne
    cmap = d['_cmap']; data = spec['path'][:-len('.vhdr')] + '.eeg'
    sha, _ = hashes(data); sha_h, _ = hashes(spec['path'])
    lf = bids_line(spec['path'].replace('_eeg.vhdr', '_eeg.json')); assert lf in (None, d['line_hz']), (spec['rel'], lf)
    ch = pd.read_csv(spec['path'].replace('_eeg.vhdr', '_channels.tsv'), sep='\t')
    raw = mne.io.read_raw_brainvision(spec['path'], preload=False, verbose=False)
    assert len([n for n in raw.ch_names if not re.match(c['non_eeg'], n)]) == d['n_eeg'], spec['rel']
    pick = [cmap[n] for n in c['names64']]; names = NP.canonical_names(c['names64'], bank)
    yield dict(rec_key=spec['rel'], subject=spec['subject'], session='ses-1', run=f'task-listening_{spec["run"]}', raw_file=os.path.relpath(data, P(d['raw'])),
               raw_sha256=sha, names=names, fs=float(raw.info['sfreq']), get=lambda idx: raw.get_data(picks=[pick[i] for i in idx]) * d['scale_to_uv'],
               close=lambda: None, info=dict(header_sha256=sha_h, provider_bads=','.join(ch.loc[ch.status == 'bad', 'name'].astype(str))))


def list_things_eeg1(c, d):
    base = P(d['raw']); fs = sorted(glob.glob(os.path.join(base, d['glob'])))
    specs = [dict(path=f, rel=os.path.relpath(f, base), subject=re.search(r'(sub-\d+)', f).group(1)) for f in fs]
    return specs, [], dict(n_subjects=len(specs))


def read_things_eeg1(c, d, spec, bank):
    import scipy.io as sio
    E = sio.loadmat(spec['path'], squeeze_me=True, struct_as_record=False, variable_names=['EEG'])['EEG']
    fdt = os.path.join(os.path.dirname(spec['path']), str(E.data)); assert os.path.exists(fdt) and fdt.endswith('.fdt'), (spec['rel'], E.data)
    C, T = int(E.nbchan), int(E.pnts); X = np.memmap(fdt, '<f4', 'r'); assert X.size == C * T, (spec['rel'], X.size, C, T)
    X = X.reshape(T, C)                                                       # EEGLAB column-major [channels x points]
    raw_names = [str(ch.labels) for ch in E.chanlocs]; assert C == d['n_eeg'], (spec['rel'], C)
    names = NP.canonical_names(raw_names, bank); s = spec['subject']
    lf = bids_line(os.path.join(P(d['raw']), s, 'eeg', f'{s}_task-rsvp_eeg.json')); assert lf == d['line_hz'], (s, lf)
    sha, _ = hashes(fdt); sha_set, _ = hashes(spec['path'])
    yield dict(rec_key=spec['rel'], subject=s, session='ses-1', run='task-rsvp', raw_file=os.path.relpath(fdt, P(d['raw'])), raw_sha256=sha,
               names=names, fs=float(E.srate), get=lambda idx: np.asarray(X[:, idx], np.float64).T, close=lambda: None,
               info=dict(set_sha256=sha_set, provider='authors derivative (eeglab continuous: 0.1-100 Hz FIR, average reference, 250 Hz)',
                         provider_ref=str(E.ref)))


def list_things_eeg2(c, d):
    base = P(d['raw']); fs = sorted(glob.glob(os.path.join(base, d['glob'])))
    specs = [dict(path=f, rel=os.path.relpath(f, base), subject=f.split('/')[-3], session=f.split('/')[-2]) for f in fs]
    ex = d.get('excluded_files')
    excl = [dict(raw_file=os.path.relpath(f, base), reason=ex['reason']) for f in sorted(glob.glob(os.path.join(base, ex['glob'])))] if ex else []
    return specs, excl, dict(n_subjects=len({x['subject'] for x in specs}))


def read_things_eeg2(c, d, spec, bank):
    from ese.things import rawio
    sha = rawio.sha256(spec['path']); D = rawio.load_dict(spec['path'], 'raw')         # restricted unpickler; stops on the test file
    x = D['raw_eeg_data']; eeg = [i for i, t in enumerate(D['ch_types']) if t == 'eeg']
    drop = [str(D['ch_names'][i]) for i in range(len(D['ch_names'])) if i not in eeg]; assert len(eeg) == d['n_eeg'], (spec['rel'], len(eeg))
    names = NP.canonical_names([str(D['ch_names'][i]) for i in eeg], bank)
    yield dict(rec_key=spec['rel'], subject=spec['subject'], session=spec['session'], run='raw_eeg_training', raw_file=spec['rel'], raw_sha256=sha,
               names=names, fs=float(D['sfreq']), get=lambda idx: x[[eeg[i] for i in idx]] * 1e6, close=lambda: None,
               info=dict(dropped=','.join(drop), provider_filter=f'highpass {D.get("highpass")} lowpass {D.get("lowpass")}'))


# ------------------------------------------------------------------ one file -> processed recordings (worker)
def process(job):
    c, name, d, spec, bank = job; out = []
    try:
        for src in globals()['read_' + name](c, d, spec, bank):
            t0 = time.time(); cc = dict(c['chain'], band_hz=d.get('band_hz', c['chain']['band_hz']))
            cl = NP.Cleaner(cc, src['fs'], d['line_hz']); C = len(src['names']); k = c['chunk_channels']
            x = np.concatenate([cl(src['get'](list(range(i, min(i + k, C))))) for i in range(0, C, k)])
            src['close'](); x, bad = NP.fix_bad(c['chain'], x, src['names']); x = NP.car(x).astype(np.float32)
            assert np.isfinite(x).all(), f'{src["rec_key"]}: non-finite samples after the chain'
            sd = x.std(1)
            row = dict(subject=src['subject'], session=src['session'], run=src['run'], length=int(x.shape[1]), raw_file=src['raw_file'],
                       raw_sha256=src['raw_sha256'], n_interpolated=len(bad), interpolated=','.join(bad), fs_in=src['fs'],
                       sd_median_uv=float(np.median(sd)), sd_min_uv=float(sd.min()), sd_max_uv=float(sd.max()),
                       notch_hz=','.join(f'{f:g}' for base in np.atleast_1d(d['line_hz'])
                                         for f in np.arange(base, min(src['fs'] / 2, cc['band_hz'][1] + 5), base)),
                       seconds=round(time.time() - t0, 1), worker_peak_gb=round(peak_gb(), 2), reordered=False, **src['info'])
            out.append((src['rec_key'], src['names'], row, np.ascontiguousarray(x.T)))
    except Exception as e:                                                    # recorded as an excluded file, never silently
        return spec['rel'], None, f'{type(e).__name__}: {e}', traceback.format_exc(limit=3)
    finally:
        tmp = os.path.join(c['tmp_dir'], f'cpt_prep_{os.getpid()}.bdf')
        if os.path.exists(tmp): os.remove(tmp)
    return spec['rel'], out, None, None


def split_of(n, s):
    k = max(s['min_val'], int(round(s['fraction'] * n))); val = set(np.random.default_rng(s['seed']).choice(n, size=min(k, n), replace=False).tolist())
    return ['val' if i in val else 'train' for i in range(n)]


def summary(c):
    """results/nc/cpt/corpus_summary.json + headers/<name>.json from the finished caches (sha256 of x.f32 and recordings.parquet);
    no statistic of the signals beyond what header.json already holds"""
    res = os.path.join(ROOT, c['summary_dir']); os.makedirs(os.path.join(res, 'headers'), exist_ok=True); out = {}
    for name in c['datasets']:
        o = os.path.join(P(c['cache_dir']), name)
        if not os.path.exists(os.path.join(o, 'header.json')): out[name] = dict(missing=True); continue
        h = json.load(open(os.path.join(o, 'header.json'))); shutil.copy(os.path.join(o, 'header.json'), os.path.join(res, 'headers', f'{name}.json'))
        R = pd.read_parquet(os.path.join(o, 'recordings.parquet'))
        files = {f: dict(bytes=os.path.getsize(os.path.join(o, f)), sha256=hashes(os.path.join(o, f))[0]) for f in ('x.f32', 'recordings.parquet', 'header.json')}
        out[name] = dict(n_recordings=h['n_recordings'], n_subjects=h['n_subjects'], n_val_recordings=h['n_val_recordings'], hours=h['hours'],
                         n_channels=h['n_channels'], ch_names=h['ch_names'], n_samples=h['n_samples'], fs_in=h['fs_in'], line_hz=h['line_hz'],
                         n_excluded=len(h['excluded']), recordings_with_interpolation=h['recordings_with_interpolation'],
                         n_interpolated_total=h['n_interpolated_total'], peak_rss_gb=h['peak_rss_gb'], seconds=h['seconds'],
                         min_recording_s=float(R.length.min() / h['sfreq']), files=files, cache=os.path.relpath(o, ROOT))
        print(name, json.dumps({k: out[name][k] for k in ('n_recordings', 'hours', 'n_channels')}), flush=True)
    ok = [v for v in out.values() if not v.get('missing')]
    tot = dict(n_datasets=len(ok), n_recordings=sum(v['n_recordings'] for v in ok), hours=sum(v['hours']['total'] for v in ok),
               hours_train=sum(v['hours']['train'] for v in ok), hours_val=sum(v['hours']['val'] for v in ok),
               bytes=sum(sum(f['bytes'] for f in v['files'].values()) for v in ok))
    json.dump(dict(plan='docs/08 §14.1 (VERA-CPT pretraining corpus)', config=os.path.relpath(c['_config'], ROOT), total=tot, datasets=out,
                   excluded_datasets=c.get('excluded_datasets', {})),
              open(os.path.join(res, 'corpus_summary.json'), 'w'), indent=1)
    print(json.dumps(tot), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--data'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--summary', action='store_true'); a = ap.parse_args(); c = yaml.safe_load(open(P(a.config))); c['_config'] = P(a.config)
    if a.summary: summary(c); return
    d = c['datasets'][a.data]; t_all = time.time()
    nc = yaml.safe_load(open(P(c['chain_config'])))
    c['chain'] = {k: nc[k] for k in ('sfreq', 'band_hz', 'butter_order', 'notch_q', 'bad_sd_hi', 'bad_sd_lo')}
    bank = json.load(open(P(nc['positions'])))
    out = os.path.join(P(c['smoke_dir'] if a.smoke else c['cache_dir']), a.data); os.makedirs(out, exist_ok=True)
    for f in ('x.f32', 'x.f32.part', 'recordings.parquet', 'header.json'):
        if os.path.exists(os.path.join(out, f)): os.remove(os.path.join(out, f))
    specs, excluded, listing = globals()['list_' + a.data](c, d)
    if a.smoke:
        specs = [specs[0], specs[-1]] if len(specs) > 1 else specs
    print(json.dumps(dict(dataset=a.data, files=len(specs), excluded=len(excluded), workers=d.get('workers', 1), **listing)), flush=True)
    extra = {}
    if a.data == 'alice':
        A = alice_map(d, specs, bank); d['_map'] = A; v = np.array(list(A['dist'].values()))
        extra = dict(channel_map=A['map'], channel_map_dist_pct_radius=A['dist'], channel_map_dist_median=float(np.median(v)),
                     channel_map_dist_max=float(v.max()), channel_map_rule='file channels 1..61 (digitized positions of the fif, equal in all files, '
                     'checked) -> standard_1005 names in the REVE bank (T3-T6 left out), rule of scripts/nc_prep.py tol() with the file electrodes as rows')
    if a.data == 'ds004408':
        cmap, dmap, checked = biosemi128_to_64(c); v = np.array(list(dmap.values())); d['_cmap'] = cmap
        extra = dict(channel_map=cmap, channel_map_dist_pct_radius=dmap, channel_map_rule='scripts/nc_prep.py tol(): BioSemi 64 10-10 names <- '
                     'nearest BioSemi 128 electrode (MNE biosemi128 vs standard_1005, unit sphere, Hungarian + Procrustes x 10; no electrode file in ds004408)',
                     channel_map_equals_tol_cache=checked, channel_map_dist_median=float(np.median(v)), channel_map_dist_max=float(v.max()))
    rows, names_all, n_total, t_last = [], None, 0, time.time()
    jobs = [(c, a.data, d, s, bank) for s in specs]; nw = min(d.get('workers', 1), len(jobs))
    pool = mp.get_context('fork').Pool(nw, maxtasksperchild=1) if nw > 1 else None
    it = pool.imap(process, jobs, chunksize=1) if pool else map(process, jobs)
    with open(os.path.join(out, 'x.f32.part'), 'wb') as fx:
        for rel, recs, err, tb in it:
            if err:
                excluded.append(dict(raw_file=rel, reason=err)); print(json.dumps(dict(excluded=rel, reason=err)), '\n', tb, flush=True); continue
            for key, names, row, xt in recs:
                if names_all is None: names_all = names
                if names != names_all:
                    if sorted(names) != sorted(names_all):
                        excluded.append(dict(raw_file=key, reason=f'channel set differs: {sorted(set(names) ^ set(names_all))}')); continue
                    xt = xt[:, [names.index(n) for n in names_all]]; row['reordered'] = True
                xt.astype('<f4').tofile(fx); row.update(rec=len(rows), offset=n_total, rec_key=key); n_total += xt.shape[0]; rows.append(row)
                print(json.dumps(dict(rec=row['rec'], key=key, s=round(row['length'] / 200, 1), interp=row['interpolated'], sd_med=round(row['sd_median_uv'], 2),
                                      sd_min=round(row['sd_min_uv'], 2), sd_max=round(row['sd_max_uv'], 2), sec=row['seconds'],
                                      worker_gb=row['worker_peak_gb'], parent_gb=round(peak_gb(), 2), elapsed=round(time.time() - t_all))), flush=True)
    if pool: pool.close(); pool.join()
    assert rows, 'no recording written'
    os.replace(os.path.join(out, 'x.f32.part'), os.path.join(out, 'x.f32'))
    R = pd.DataFrame(rows); R['split'] = split_of(len(R), c['split'])
    first = ['rec', 'subject', 'session', 'run', 'offset', 'length', 'raw_file', 'raw_sha256', 'n_interpolated', 'interpolated', 'split']
    R = R[first + [k for k in R.columns if k not in first]]
    for k in R.columns:
        if R[k].dtype == object: R[k] = R[k].map(lambda v: '' if v is None or (isinstance(v, float) and np.isnan(v)) else v)
    R.to_parquet(os.path.join(out, 'recordings.parquet'), index=False)
    fs = c['chain']['sfreq']; hrs = {k: float(R.length[R.split == k].sum() / fs / 3600) for k in ('train', 'val')}
    assert os.path.getsize(os.path.join(out, 'x.f32')) == n_total * len(names_all) * 4
    header = dict(dataset=a.data, smoke=a.smoke, ch_names=names_all, n_channels=len(names_all), sfreq=fs, dtype='float32', byte_order='little',
                  layout='time-major [n_samples, n_channels], recordings back to back (recordings.parquet offset / length in samples)',
                  unit='microvolt', n_samples=int(n_total), n_recordings=int(len(R)), n_subjects=int(R.subject.nunique()),
                  hours=dict(train=hrs['train'], val=hrs['val'], total=hrs['train'] + hrs['val']),
                  n_val_recordings=int((R.split == 'val').sum()), split=c['split'],
                  line_hz=d['line_hz'], line_hz_source=d['line_hz_source'], fs_in=sorted(set(R.fs_in.tolist())),
                  chain=dict(c['chain'], band_hz=d.get('band_hz', c['chain']['band_hz']), source=c['chain_config'],
                             order='notch (line + harmonics below band_hz[1] + 5 Hz, iirnotch Q, filtfilt) -> Butterworth band-pass (sosfiltfilt) '
                                   '-> resample_poly to sfreq -> bad channels on a median-referenced copy (SD > bad_sd_hi x or < bad_sd_lo x the median '
                                   'channel SD over the whole recording; MNE spherical spline, standard_1005 positions; scripts/nc_prep.py fix_bad) '
                                   '-> common average over the EEG channels (scripts/nc_prep.py car)',
                             chunk_channels=c['chunk_channels']),
                  input=d.get('input', 'raw recordings'), scale_to_uv=d.get('scale_to_uv'), scale_note=d.get('scale_note'), positions=nc['positions'], excluded=excluded, n_interpolated_total=int(R.n_interpolated.sum()),
                  recordings_with_interpolation=int((R.n_interpolated > 0).sum()),
                  peak_rss_gb=dict(parent=round(peak_gb(), 2), max_child=round(peak_gb(resource.RUSAGE_CHILDREN), 2),
                                   max_worker_reported=float(R.worker_peak_gb.max()), workers=nw),
                  seconds=round(time.time() - t_all, 1), listing=listing, config=a.config, **extra)
    json.dump(header, open(os.path.join(out, 'header.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: header[k] for k in ('dataset', 'n_recordings', 'n_subjects', 'n_samples', 'hours', 'n_channels', 'peak_rss_gb', 'seconds')}
                     | dict(excluded=len(excluded)), default=str), flush=True)


if __name__ == '__main__':
    main()
