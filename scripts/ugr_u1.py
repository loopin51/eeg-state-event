"""D-110 section 4: U1 (R-b1) auditory N1 on UGR-MINDVOICE, session split. EEG is read only through the sealed reader.

Epochs (per R block, never across blocks): 64 EEG channels decoded by scripts/ugr_edf.py -> EOG channels dropped (EOG1, EOG2;
TP9, TP10 at those places in sub-13 ses-01) -> flat channels (IQR 0 inside the block; the dataset marks no bad channel) left out
-> 0.5-30 Hz zero-phase FIR (MNE default design, reflect padding inside the block) -> every 4th sample (250 Hz) -> CAR over the
remaining channels -> trials [-0.2, 0.5) s, baseline [-0.2, 0). Real main trials only, item 'silence' left out (no cue wav; interpretation record of D-110). U1 alignment: audio cue = StartStimulus +
speech onset of its wav (D-110 3.1 VAD), text / picture cue = StartStimulus. P100 alignment: StartStimulus for every cue.
ROI = mean of the non-flat channels among Fz, F1, F2, FCz, FC1, FC2, Cz, C1, C2; trials with ROI peak-to-peak > 200 uV are left
out (counted). Cached per session in <cache>/u1/<sub>_<ses>.npz.
Test: per target, L = most negative time of the ses-01 (audio - text) ROI difference wave in [80, 200] ms; D = ses-02 (text -
audio) mean ROI amplitude in [L - 25, L + 25] ms; statistic = sum of D over the six targets; null = cue labels (audio / text)
permuted within block in ses-02 (1,000, seed 20261403), one-sided p = (1 + #null >= observed) / 1001. Pass p < 0.05.
Report only: per-target p and k of 6, single-session subjects (L from odd blocks, D from even blocks), P100 (visual - audio,
O1 Oz O2, positive peak in [80, 140] ms), sensitivity with the excluded sub-13 ses-02 block, and (not registered, coordination
session) the same test without the first and last trial of every block (FIR edge effect within ~3.3 s of block edges).

usage: python scripts/ugr_u1.py configs/ugr_u.yaml [--stage epochs|stats|all]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, csv, glob, json, time, hashlib, argparse, unicodedata, yaml, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ugr_edf import read_session, filter_input_check
from ugr_u0 import wav_name, cue_features
from d106_ugr import trials_from, PAT
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sessions(c):
    s = c['subjects']; out = [f'{t}/{x}' for t in s['targets'] for x in ('ses-01', 'ses-02')]
    return out + [f'{t}/ses-01' for t in s['pool_single']] + list(s['pool_extra_sessions'])


def materials_wavs(c):
    """NFC stem -> path; some Materials names are stored NFD (n + combining tilde)"""
    M = os.path.join(c['data_root'], c['materials'])
    return {unicodedata.normalize('NFC', f[:-4]): os.path.join(M, f) for f in os.listdir(M) if f.lower().endswith('.wav')}


def speech_onset(c, kind, item, cache):
    k = (kind, item)
    if k not in cache:
        if '_wavs' not in cache: cache['_wavs'] = materials_wavs(c)
        n = unicodedata.normalize('NFC', wav_name(item, kind)); assert n in cache['_wavs'], n
        cache[k] = cue_features(cache['_wavs'][n], c['u0']['vad'])['onset_s']
    return cache[k]


def label_checked_rows(path):
    """event rows whose label differs from the open trial (StartFixation) are left out; an orphan StartStimulus at the end of
    sub-06 ses-02 (6000.068 s, no StartFixation) otherwise overwrites the cue onset of the preceding trial in trials_from"""
    rows, out, cur = list(csv.DictReader(open(path, encoding='utf-8-sig'), delimiter='\t')), [], None
    for r in rows:
        a, _, b = r['trial_type'].partition(':'); m = PAT.match(a)
        if not m: out.append(r); continue
        if m.group(4) == 'Start' and m.group(5) == 'Fixation': cur = (a[:-len('StartFixation')], b)
        elif cur is None or b != cur[1] or not a.startswith(cur[0]): continue
        out.append(r)
    return out


def session_trials(c, session):
    ev = glob.glob(os.path.join(c['data_root'], c['events_glob'].replace('sub-*/ses-*', session))); assert len(ev) == 1
    trials = trials_from(label_checked_rows(ev[0]), 'trial_type'); assert all(x['part'] == 'Experiment' for x in trials)
    real_all = [x for x in trials if x['mode'] == 'Real']
    return real_all, [x for x in real_all if x['item'] not in c['u1']['exclude_items']]   # 'silence': no cue wav, an audio-cue trial plays no sound


def epochs_session(c, session, onsets):
    import mne
    u = c['u1']; fs = u['sample_rate']; step = 1000 // fs; pre = int(round(-u['epoch_s'][0] * fs)); T = int(round((u['epoch_s'][1] - u['epoch_s'][0]) * fs))
    real_all, real = session_trials(c, session)
    from ugr_edf import header, edf_path
    labels = [l for l in header(edf_path(c, session))['labels'] if l != 'EDF Annotations']; assert len(labels) == 64
    keep = [l for l in labels if l not in c['preprocess']['drop_channels']]; assert len(keep) == 62, (session, len(keep))
    blocks, decoded, h = read_session(c, session, keep)
    t0 = {}
    for x in real:
        st = int(round(x['ph']['Stimulus']['Start'] * 1000)); t0[id(x)] = st
    out = dict(u1=[], cue=[], meta=[]); flat_log = []
    for bi, b in enumerate(blocks):
        x = b['data'].astype(np.float64); q = np.percentile(x, [25, 75], axis=1); flat = (q[1] - q[0]) <= 1e-12; flat_log.append([keep[i] for i in np.where(flat)[0]])
        filter_input_check(x, b, decoded)
        y = mne.filter.filter_data(x, 1000.0, l_freq=u['band_hz'][0], h_freq=u['band_hz'][1], pad='reflect', n_jobs=1, verbose=False)[:, ::step]
        good = ~flat; y = y - y[good].mean(0, keepdims=True); y[flat] = np.nan
        for x_ in real:
            st = t0[id(x_)]
            if not (b['start'] <= st < b['end']): continue
            on = st + int(round(1000 * speech_onset(c, x_['kind'], x_['item'], onsets))) if x_['cue'] == 'Audio' else st
            rec = []
            for o in (on, st):
                i = int(round((o - b['start']) / step)) - pre
                assert 0 <= i and i + T <= y.shape[1], f'{session}: epoch outside its R block'
                e = y[:, i:i + T]; rec.append((e - e[:, :pre].mean(1, keepdims=True)).astype(np.float32))
            out['u1'].append(rec[0]); out['cue'].append(rec[1])
            out['meta'].append(dict(cue=x_['cue'], kind=x_['kind'], item=x_['item'], block=x_['block'], pos=x_['pos'], block_from_end=x_['block_from_end'],
                                    rblock=bi, stim_sample=st, onset_sample=on))
    assert len(out['meta']) == len(real), f'{session}: {len(real) - len(out["meta"])} Real trials outside the R blocks'
    bl = {}
    for x in real_all: bl.setdefault(x['block'], []).append(x['pos'])   # block edges from every Real trial, excluded items included
    for m in out['meta']: m['first_or_last'] = m['pos'] in (min(bl[m['block']]), max(bl[m['block']]))
    return dict(u1=np.stack(out['u1']), cue=np.stack(out['cue']), meta=out['meta'], channels=keep, flat=flat_log,
                decoded=[list(r) for r in decoded], n_blocks=len(blocks), excluded_items=len(real_all) - len(real),
                by_cue={k: dict(kept=sum(x['cue'] == k for x in real), excluded=sum(x['cue'] == k for x in real_all) - sum(x['cue'] == k for x in real))
                        for k in ('Audio', 'Text', 'Pictures')})


def roi_mean(E, chans, roi):
    idx = [chans.index(r) for r in roi]; return np.nanmean(E[:, idx, :], axis=1)


def load(c, session):
    d = np.load(os.path.join(ROOT, c['cache'], 'u1', session.replace('/', '_') + '.npz'), allow_pickle=True)
    return dict(u1=d['u1'], cue=d['cue'], meta=list(d['meta']), channels=list(d['channels']))


def prepare(c, session, which, roi, keep_fn):
    """ROI series [n, T] of the kept trials, cue labels, blocks; ROI peak-to-peak rejection"""
    S = load(c, session); R = roi_mean(S[which], S['channels'], roi); m = S['meta']
    ok = np.array([keep_fn(x) for x in m]); ptp = R.max(1) - R.min(1); rej = ptp > c['u1']['reject_ptp_uv']
    sel = ok & ~rej
    return R[sel], np.array([x['cue'] for x in m])[sel], np.array([x['block'] for x in m])[sel], int((ok & rej).sum()), int(ok.sum())


def latency(R, cue, a, b, times, win, sign):
    d = R[cue == a].mean(0) - R[cue == b].mean(0); w = (times >= win[0] - 1e-9) & (times <= win[1] + 1e-9)
    j = np.where(w)[0][np.argmin(sign * d[w])]; return float(times[j])


def amp(R, times, L, hw):
    w = (times >= L - hw - 1e-9) & (times <= L + hw + 1e-9); return R[:, w].mean(1)


def test(units, n, seed, pos, neg):
    """units: list of (amplitudes, cue, block); D = mean(pos cues) - mean(neg cues); null = cue labels permuted within block"""
    rng = np.random.default_rng(seed); obs = []; null = np.zeros((n, len(units)))
    for k, (a, cue, blk) in enumerate(units):
        isp = np.isin(cue, pos); obs.append(a[isp].mean() - a[~isp].mean()); groups = [np.where(blk == g)[0] for g in np.unique(blk)]
        for r in range(n):
            lab = isp.copy()
            for g in groups: lab[g] = isp[g][rng.permutation(len(g))]
            null[r, k] = a[lab].mean() - a[~lab].mean()
    obs = np.array(obs); tot = obs.sum(); nt = null.sum(1)
    return dict(D=obs.tolist(), sum=float(tot), p=float((1 + (nt >= tot).sum()) / (n + 1)),
                p_each=[float((1 + (null[:, k] >= obs[k]).sum()) / (n + 1)) for k in range(len(units))])


def stats(c):
    u = c['u1']; fs = u['sample_rate']; times = np.arange(int(round((u['epoch_s'][1] - u['epoch_s'][0]) * fs))) / fs + u['epoch_s'][0]
    tg = c['subjects']['targets']; excl = c['subjects']['primary_exclude_blocks']; res = dict(times_s=[float(times[0]), float(times[-1])])

    def run(name, kind, keep_extra=lambda x: True, include_excluded=False):
        aud = (['Audio'], ['Text']) if kind == 'n1' else (['Text', 'Pictures'], ['Audio'])
        roi = u['roi'] if kind == 'n1' else u['p100']['roi']; which = 'u1' if kind == 'n1' else 'cue'
        win = u['peak_search_s'] if kind == 'n1' else u['p100']['peak_search_s']; sign = 1 if kind == 'n1' else -1
        seed = u['null_perm']['seed'] if kind == 'n1' else u['p100']['seed']; units, per, counts = [], {}, {}
        cues = aud[0] + aud[1]
        for s in tg:
            def k1(x): return x['cue'] in cues and keep_extra(x)
            def k2(x, s=s):
                drop = (not include_excluded) and any(x['block_from_end'] == -k for k in excl.get(f'{s}/ses-02', []))
                return x['cue'] in cues and keep_extra(x) and not drop
            R1, c1, b1, rj1, n1 = prepare(c, f'{s}/ses-01', which, roi, k1); R2, c2, b2, rj2, n2 = prepare(c, f'{s}/ses-02', which, roi, k2)
            g1 = np.where(np.isin(c1, aud[0]), 'P', 'N'); L = latency(R1, g1, 'P', 'N', times, win, sign)
            units.append((amp(R2, times, L, u['half_width_s']), np.where(np.isin(c2, aud[0]), 'P', 'N'), b2))
            per[s] = dict(L_ms=1000 * L, n_ses01=int(len(c1)), n_ses02=int(len(c2)), rejected=[rj1, rj2], considered=[n1, n2])
        # D = mean(second group) - mean(first group): N1 -> text - audio (positive), P100 -> visual - audio (positive)
        units = [(a, np.where(g == 'P', 'N', 'P') if kind == 'n1' else g, b) for a, g, b in units]
        t = test(units, u['null_perm']['n'], seed, ['P'], ['N'])
        for k, s in enumerate(tg): per[s].update(D_uv=t['D'][k], p=t['p_each'][k])
        out = dict(kind=kind, sum_D_uv=t['sum'], p=t['p'], per_target=per, k_of_6=int(sum(p < u['pass_p'] for p in t['p_each'])))
        if name == 'primary': out['passed'] = bool(t['p'] < u['pass_p'])
        res[name] = out

    run('primary', 'n1')
    run('report_include_sub13_excluded_block', 'n1', include_excluded=True)
    run('report_p100', 'p100')
    run('report_unregistered_drop_block_first_last', 'n1', keep_extra=lambda x: not x['first_or_last'])
    single = {}
    for sess in [f'{t}/ses-01' for t in c['subjects']['pool_single']] + list(c['subjects']['pool_extra_sessions']):
        R, cu, bl, rj, n = prepare(c, sess, 'u1', u['roi'], lambda x: x['cue'] in ('Audio', 'Text'))
        rb = {b: i for i, b in enumerate(sorted(set(bl)))}; odd = np.array([rb[b] % 2 == 1 for b in bl])
        g = np.where(cu == 'Audio', 'P', 'N'); L = latency(R[odd], g[odd], 'P', 'N', times, u['peak_search_s'], 1)
        t = test([(amp(R[~odd], times, L, u['half_width_s']), np.where(g[~odd] == 'P', 'N', 'P'), bl[~odd])], u['null_perm']['n'], u['null_perm']['seed'], ['P'], ['N'])
        single[sess] = dict(L_ms=1000 * L, D_uv=t['D'][0], p=t['p'], rejected=rj, considered=n)
    res['report_single_session_odd_even'] = single
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', default='all'); a = ap.parse_args(); t0 = time.time()
    c = yaml.safe_load(open(os.path.join(ROOT, a.config))); cd = os.path.join(ROOT, c['cache'], 'u1'); os.makedirs(cd, exist_ok=True); log = {}
    if a.stage in ('epochs', 'all'):
        onsets = {}
        for s in sessions(c):   # every audio cue wav is resolved before any EDF is opened
            for x in session_trials(c, s)[1]:
                if x['cue'] == 'Audio': speech_onset(c, x['kind'], x['item'], onsets)
        print('speech onsets resolved for', len(onsets) - 1, 'audio cue items', flush=True)
        for s in sessions(c):
            E = epochs_session(c, s, onsets)
            np.savez(os.path.join(cd, s.replace('/', '_') + '.npz'), u1=E['u1'], cue=E['cue'], meta=np.array(E['meta'], dtype=object), channels=np.array(E['channels']))
            log[s] = dict(trials=len(E['meta']), blocks=E['n_blocks'], flat=E['flat'], decoded_ranges=E['decoded'], discarded_outside_R=0, excluded_items=E['excluded_items'], by_cue=E['by_cue'])
            print(s, len(E['meta']), 'trials', E['excluded_items'], 'excluded items', E['n_blocks'], 'blocks', 'flat', sorted({f for fl in E['flat'] for f in fl}), f'{time.time() - t0:.0f}s', flush=True)
        on = [v for k, v in onsets.items() if k != '_wavs']; log['speech_onset_s'] = dict(min=min(on), max=max(on), n=len(on))
        json.dump(log, open(os.path.join(cd, 'epochs_log.json'), 'w'), indent=1)
    if a.stage in ('stats', 'all'):
        res = stats(c); res['u0_note'] = 'U0 re-judged after its result with corrected meter definitions (D-110 amendment 1): cannot be judged (too few stimuli)'
        p = os.path.join(ROOT, c['out'], 'u1.json'); json.dump(res, open(p, 'w'), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != 'report_single_session_odd_even'})[:4000])
        print('sha256', hashlib.sha256(open(p, 'rb').read()).hexdigest(), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
