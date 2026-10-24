"""D-125 preprocessing: UGR-MINDVOICE (table 2) and BCI2020 Track 3 (table 3) epoch caches for the REVE speech benchmark.

usage: python scripts/reve_speech_prep.py configs/reve_speech.yaml --data ugr|bci2020 [--subjects sub-01,...]

UGR (per session, D-125 1.1): 64-ch EDF at 1 kHz -> 50 Hz notch -> 0.1-75 Hz zero-phase FIR -> 200 Hz -> ICA (fastica, fit
on a 1 Hz high-passed copy, EOG components from EOG1/EOG2 removed) -> drop EOG1/EOG2 -> common average -> epochs.
Speech epoch [StartSpeech - 0.5, + 1.0) s and rest epoch [StartStimulus - 1.5, StartStimulus) s, both baseline-corrected on
their first 0.5 s, in microvolts. Practice trials are dropped. The Silent* seal of D-110 is lifted by D-125.
Output per subject: <cache>/ugr/<sub>.npz with speech, rest [n, 62, 300] float32 and per-trial labels.

BCI2020 (per trial, D-125 1.2): Butterworth 0.5-75 Hz (order 4, filtfilt), 50/60 Hz notches, common average, 200 Hz,
imagery window 0-2 s. Output <cache>/bci2020/<sub>.npz with x [350, C, 400] float32, y, split (0 Training, 1 Validation).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, re, csv, json, glob, time, hashlib, argparse, unicodedata
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIAL = re.compile(r'(Real|Silent)(Syllables|Words)(Experiment|Practice)(Start|End)(\w+)$')


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(buf), b''): h.update(b)
    return h.hexdigest()


def check_sha(listing, base, path):
    """Compare a raw file with the NAS sha256 list (paths relative to the dataset folder, './' prefix)."""
    rel = './' + os.path.relpath(os.path.realpath(path), os.path.realpath(base))
    want = {l[66:].rstrip('\n'): l[:64] for l in open(listing)}
    assert rel in want, f'{rel} not in {listing}'
    got = sha256(path); assert got == want[rel], f'sha256 mismatch {rel}'
    return got


def norm(s, accents=True):
    s = unicodedata.normalize('NFC', s.strip().lower())
    if not accents: s = ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')
    return s


def word_table(path):
    rows = [r for r in csv.DictReader((l for l in open(path, encoding='utf-8') if not l.startswith('#')), delimiter='\t')]
    cats = sorted({r['category'] for r in rows}); assert len(rows) == 60 and len(cats) == 6
    return [r['word'] for r in rows], [cats.index(r['category']) for r in rows], cats


def trials_from_events(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig'), delimiter='\t')); out, cur = [], None
    for r in rows:
        a, _, b = r['trial_type'].partition(':'); m = TRIAL.match(a)
        if not m: continue
        mode, kind, part, se, phase = m.groups()
        if se == 'Start' and phase == 'Fixation':
            cue, _, item = b.partition('_'); cur = dict(mode=mode, kind=kind, part=part, cue=cue, item=item, s={}); out.append(cur)
        if cur is not None and se == 'Start': cur['s'][phase] = int(r['sample'])
    return out


def ugr_session(c, edf, events, words):
    import mne
    u = c['ugr']; t0 = time.time()
    raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
    raw.set_channel_types({e: 'eog' for e in u['eog']}, verbose=False)
    fs0 = raw.info['sfreq']; assert fs0 == 1000.0, fs0
    raw.notch_filter(u['notch_hz'], picks='all', verbose=False)
    raw.filter(*u['band_hz'], picks='all', phase='zero', fir_design='firwin', verbose=False)
    raw.resample(u['sfreq'], verbose=False); fs = raw.info['sfreq']
    ic = u['ica']; hp = raw.copy().filter(ic['highpass_hz'], None, picks='all', verbose=False)
    # D-125 amendment 1: channels with SD > k x median are left out of the ICA (kept unchanged), 2 s segments with
    # peak-to-peak above the limit are left out of the fit, fixed number of components
    eeg = mne.pick_types(hp.info, eeg=True); sd = hp.get_data(picks=eeg).std(1); med = np.median(sd)
    picks = [int(i) for i, v in zip(eeg, sd) if v <= ic['channel_sd_factor'] * med]
    log = dict(edf=os.path.basename(edf), ica_left_out=[hp.ch_names[i] for i in eeg if i not in picks])
    ica = mne.preprocessing.ICA(n_components=min(ic['n_components'], len(picks) - 1), method=ic['method'], random_state=c['seed'], max_iter='auto')
    try:
        ica.fit(hp, picks=picks, reject=dict(eeg=ic['reject_p2p_uv'] * 1e-6), tstep=ic['tstep_s'], verbose=False)
        used = float(ica.n_samples_) / hp.info['sfreq']; log['ica_fit_seconds'] = round(used, 1)
        if used < ic['min_fit_s']: raise RuntimeError(f'only {used:.0f} s left for the ICA fit')
        bad, scores = ica.find_bads_eog(hp, ch_name=u['eog'], threshold=ic['eog_threshold'], verbose=False)
        ica.exclude = bad; log.update(ica_components=int(ica.n_components_), ica_excluded=[int(i) for i in bad])
        if bad: ica.apply(raw, verbose=False)
        else: log['ica_note'] = 'no EOG component selected; ICA not applied'
    except Exception as e:                                  # D-125 1.1: keep the session without ICA
        log.update(ica_error=repr(e))
    del hp
    raw.drop_channels(u['eog']); raw.set_eeg_reference('average', projection=False, verbose=False)
    X = (raw.get_data() * 1e6).astype(np.float32); names = list(raw.ch_names)
    pre, post = int(round(u['pre_s'] * fs)), int(round(u['post_s'] * fs)); off = int(round(u['rest_offset_s'] * fs)); r = fs0 / fs
    wnorm = {norm(w): i for i, w in enumerate(words)}; wflat = {norm(w, False): i for i, w in enumerate(words)}
    keep = dict(speech=[], rest=[], mode=[], kind=[], cue=[], item=[], word=[]); skipped = 0
    for t in trials_from_events(events):
        if t['part'] != 'Experiment': continue
        if 'Speech' not in t['s'] or 'Stimulus' not in t['s']: skipped += 1; continue
        ss, sr = int(round(t['s']['Speech'] / r)), int(round(t['s']['Stimulus'] / r)) + off
        if ss - pre < 0 or ss + post > X.shape[1] or sr - pre < 0 or sr + post > X.shape[1]: skipped += 1; continue
        for key, s0 in (('speech', ss), ('rest', sr)):
            e = X[:, s0 - pre:s0 + post]; keep[key].append(e - e[:, :pre].mean(1, keepdims=True))
        keep['mode'].append(0 if t['mode'] == 'Real' else 1); keep['kind'].append(t['kind']); keep['cue'].append(t['cue'])
        keep['item'].append(t['item'].strip())
        w = wnorm.get(norm(t['item'])) if t['kind'] == 'Words' else None
        if t['kind'] == 'Words' and w is None: w = wflat.get(norm(t['item'], False))
        keep['word'].append(-1 if w is None else w)
    log.update(n_trials=len(keep['mode']), skipped=skipped, seconds=round(time.time() - t0, 1))
    return keep, names, log


def ugr(c, subjects):
    import mne  # noqa: F401  (fail early if missing)
    u = c['ugr']; base = os.path.join(ROOT, u['raw']); out = os.path.join(ROOT, c['cache_dir'], 'ugr'); os.makedirs(out, exist_ok=True)
    words, wcat, cats = word_table(os.path.join(ROOT, u['words']))
    edfs = sorted(glob.glob(os.path.join(base, u['edf_glob']))); subs = sorted({re.findall(r'(sub-\d+)', f)[0] for f in edfs})
    subs = [s for s in subs if s not in u['exclude_subjects'] and (not subjects or s in subjects)]
    logs = {}
    for s in subs:
        parts, names0 = [], None
        for f in [f for f in edfs if f'/{s}/' in f]:
            ses = re.findall(r'(ses-\d+)', f)[0]
            ev = glob.glob(os.path.join(base, u['events_dir'], s, ses, 'eeg', '*_events.tsv')); assert len(ev) == 1, (s, ses, ev)
            h = check_sha(os.path.join(ROOT, u['sha256_list']), base, f)
            keep, names, log = ugr_session(c, f, ev[0], words); log['sha256'] = h; logs[f'{s}_{ses}'] = log
            assert names0 is None or names == names0; names0 = names; keep['session'] = [ses] * len(keep['mode']); parts.append(keep)
            print(json.dumps({f'{s}_{ses}': log}), flush=True)
        cat = lambda k: [v for p in parts for v in p[k]]
        word = np.array(cat('word'), int); covered = sorted(set(word[word >= 0].tolist()))
        np.savez(os.path.join(out, f'{s}.npz'), speech=np.stack(cat('speech')), rest=np.stack(cat('rest')),
                 mode=np.array(cat('mode'), np.int8), kind=np.array(cat('kind')), cue=np.array(cat('cue')), item=np.array(cat('item')),
                 word=word, cat=np.array([wcat[w] if w >= 0 else -1 for w in word], int), session=np.array(cat('session')),
                 ch_names=np.array(names0))
        logs[s] = dict(n=len(word), words_covered=len(covered))
        assert len(covered) == 60, f'{s}: only {len(covered)} of the 60 words matched (D-125 1.3 stop rule)'
    json.dump(dict(categories=cats, words=words, sessions=logs), open(os.path.join(out, 'prep_log.json'), 'w'), indent=1, ensure_ascii=False)


def bci2020(c, subjects):
    from scipy.io import loadmat
    from scipy.signal import butter, filtfilt, iirnotch, resample_poly
    b = c['bci2020']; base = os.path.join(ROOT, b['raw']); out = os.path.join(ROOT, c['cache_dir'], 'bci2020'); os.makedirs(out, exist_ok=True)
    logs = {}
    for sid in range(1, 16):
        s = f'sub-{sid:02d}'
        if subjects and s not in subjects: continue
        xs, ys, sp = [], [], []
        for k, (folder, key) in enumerate((('Training set', 'epo_train'), ('Validation set', 'epo_validation'))):
            p = os.path.join(base, folder, f'Data_Sample{sid:02d}.mat'); h = check_sha(os.path.join(ROOT, b['sha256_list']), os.path.join(ROOT, b['sha_base']), p)
            e = loadmat(p)[key][0, 0]; x = np.transpose(e['x'], (2, 1, 0)).astype(np.float64); fs = float(e['fs'][0, 0])
            names = [str(q[0]) for q in e['clab'][0]]
            bb, ab = butter(b['butter_order'], b['band_hz'], btype='band', fs=fs); x = filtfilt(bb, ab, x, axis=-1)
            for f0 in b['notch_hz']:
                bn, an = iirnotch(f0, b['notch_q'], fs=fs); x = filtfilt(bn, an, x, axis=-1)
            x = x - x.mean(1, keepdims=True)
            up, down = int(b['sfreq']), int(fs); g = np.gcd(up, down); x = resample_poly(x, up // g, down // g, axis=-1)
            i0 = int(round((b['window_s'][0] - b['epoch_start_s']) * b['sfreq'])); i1 = i0 + int(round((b['window_s'][1] - b['window_s'][0]) * b['sfreq']))
            assert x.shape[-1] >= i1, (x.shape, i1)
            xs.append(x[..., i0:i1].astype(np.float32)); ys.append(e['y'].argmax(0)); sp.append(np.full(len(x), k, np.int8))
            logs.setdefault(s, {})[folder] = dict(sha256=h, fs=fs, n=len(x), raw_std=float(np.std(e['x'])))
        np.savez(os.path.join(out, f'{s}.npz'), x=np.concatenate(xs), y=np.concatenate(ys), split=np.concatenate(sp), ch_names=np.array(names))
        print(json.dumps({s: logs[s]}), flush=True)
    json.dump(logs, open(os.path.join(out, 'prep_log.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--data', required=True, choices=['ugr', 'bci2020'])
    ap.add_argument('--subjects', default=''); a = ap.parse_args()
    c = yaml.safe_load(open(os.path.join(ROOT, a.config))); subs = [s for s in a.subjects.split(',') if s]
    (ugr if a.data == 'ugr' else bci2020)(c, subs)


if __name__ == '__main__':
    main()
