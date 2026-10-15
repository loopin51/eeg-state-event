"""D-106 section 3: UGR-MINDVOICE checks without EEG.

R-a meters (events of the eeg folder only): rules built from ses-01 are applied to the ses-02 covert-word trials of the
primary subjects (sensitivity: + sub-15). ADJ_copy = item of the preceding main trial; ADJ_table = modal next item after the
preceding item in ses-01 (fallback: ses-01 modal covert-word item); BLOCKPOS = item at the same (block number, position in
block) in ses-01 (same fallback). Pass per meter: one-sided Clopper-Pearson 95% upper of the pooled accuracy < 1/K + 0.02
(K = distinct items among the test trials). sub-13 ses-02 third-to-last block is left out of the primary set (D-100 rule 2).
u_overt (session audio anonymized.wav and audio events): 16 kHz mono, RMS dB in 25-ms windows every 10 ms; per trial the
threshold is the median + 5 x 1.4826 x MAD of the frames in the 0.5 s before StartSpeech; onset = first frame in
[StartSpeech, EndSpeech) from which the RMS stays >= threshold for >= 50 ms; u_overt = onset - StartSpeech. Overt trials give
u_overt; covert trials give the false-detection rate. Descriptive only (not R-e).

usage: python scripts/d106_ugr.py configs/d106_gates.yaml [--part ra|u|all]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import re, csv, glob, json, math, time, argparse, collections, yaml, numpy as np
from scipy.stats import beta
from scipy.signal import resample_poly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r'(Real|Silent)(Syllables|Words)(Experiment|Practice)(Start|End)(\w+)$')


def trials_from(rows, col):
    trials, cur = [], None
    for r in rows:
        a, _, b = r[col].partition(':'); m = PAT.match(a)
        if not m: continue
        mode, kind, part, se, phase = m.groups(); t = float(r['onset'])
        if se == 'Start' and phase == 'Fixation':
            cur = dict(mode=mode, kind=kind, part=part, cue=b.split('_')[0], item='_'.join(b.split('_')[1:]), ph={}); trials.append(cur)
        if cur is not None: cur['ph'].setdefault(phase, {})[se] = t
    main = [x for x in trials if x['part'] == 'Experiment']; blk, pos, prev = -1, 0, None
    for x in main:
        k = x['mode'] + x['kind']
        if k != prev: blk += 1; pos = 0
        else: pos += 1
        x['block'], x['pos'], prev = blk, pos, k
    nb = blk + 1
    for x in main: x['block_from_end'] = nb - x['block']
    return main


def read(path, col):
    return trials_from(list(csv.DictReader(open(path, encoding='utf-8-sig'), delimiter='\t')), col)


def cp_upper(k, n, a=0.05): return 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))


def part_ra(c, res):
    u = c['ugr']; D = u['meta_root']; out = {}
    def ev(s, ses): return read(glob.glob(os.path.join(D, f'sub-{s:02d}', f'ses-{ses:02d}', 'eeg', '*_events.tsv'))[0], 'trial_type')
    for setname, subs in (('primary', u['primary_subjects']), ('sensitivity_with_sub15', u['sensitivity_subjects'])):
        for label in (('Silent', 'Words'), ('Silent', 'Syllables'), ('Real', 'Words')):
            hits = collections.Counter(); n = 0; per = {}; items_test = collections.Counter()
            for s in subs:
                A, B = ev(s, 1), ev(s, 2)
                fallback = collections.Counter(x['item'] for x in A if (x['mode'], x['kind']) == label).most_common(1)[0][0]
                table = collections.defaultdict(collections.Counter)
                for p, q in zip(A[:-1], A[1:]): table[p['item']][q['item']] += 1
                bp = {(x['block'], x['pos']): x['item'] for x in A}
                test = [(i, x) for i, x in enumerate(B) if (x['mode'], x['kind']) == label and not (setname == 'primary' and s == 13 and x['block_from_end'] == 3)]
                h = collections.Counter()
                for i, x in test:
                    prev = B[i - 1]['item'] if i > 0 else None
                    pred = dict(ADJ_copy=prev,
                                ADJ_table=(min(table[prev].items(), key=lambda kv: (-kv[1], kv[0]))[0] if prev in table and table[prev] else fallback),
                                BLOCKPOS=bp.get((x['block'], x['pos']), fallback))
                    for k, v in pred.items(): h[k] += int(v == x['item'])
                    items_test[x['item']] += 1
                per[f'sub-{s:02d}'] = dict(n=len(test), **{k: h[k] / max(1, len(test)) for k in ('ADJ_copy', 'ADJ_table', 'BLOCKPOS')}); hits += h; n += len(test)
            K = len(items_test); maj = max(items_test.values()) / n if n else float('nan')
            meters = {k: dict(acc=hits[k] / n, cp95_upper=cp_upper(hits[k], n), threshold=1 / K + 0.02, passed=bool(cp_upper(hits[k], n) < 1 / K + 0.02)) for k in ('ADJ_copy', 'ADJ_table', 'BLOCKPOS')}
            out[f'{setname}/{label[0]}{label[1]}'] = dict(n=n, K=K, chance=1 / K, majority_rate=maj, meters=meters, per_subject=per,
                                                          all_passed=bool(all(m['passed'] for m in meters.values())))
    res['ra'] = out


def part_u(c, res):
    import soundfile as sf
    u = c['ugr']; D = u['meta_root']; A = u['audio_root']; sessions = {}; rows = []
    for d in sorted(glob.glob(os.path.join(D, 'sub-*', 'ses-*'))):
        sub, ses = re.findall(r'sub-(\d+)', d)[0], re.findall(r'ses-(\d+)', d)[0]; key = f'sub-{sub}_ses-{ses}'
        evp = glob.glob(os.path.join(d, 'audio', '*_events.tsv')); wav = os.path.join(A, f'sub-{sub}', f'ses-{ses}', 'audio', 'anonymized.wav')
        if not evp: sessions[key] = dict(status='no audio events'); continue
        if not os.path.exists(wav): sessions[key] = dict(status='no audio file'); continue
        T = read(evp[0], 'description'); x, fs = sf.read(wav, dtype='float32', always_2d=True); x = x.mean(1)
        g = math.gcd(int(fs), 16000); x = resample_poly(x, 16000 // g, int(fs) // g).astype(np.float32); fs = 16000
        win, hop = int(u['rms']['win_s'] * fs), int(u['rms']['hop_s'] * fs); nfr = 1 + (len(x) - win) // hop
        c2 = np.r_[0.0, np.cumsum(x.astype(np.float64) ** 2)]; st = hop * np.arange(nfr); rms = 20 * np.log10(np.sqrt(np.maximum(c2[st + win] - c2[st], 0) / win) + 1e-10)
        tf = (np.arange(nfr) * hop + win / 2) / fs; need = int(math.ceil(u['min_run_s'] / u['rms']['hop_s']))
        n_det = collections.Counter(); n_all = collections.Counter(); lat = collections.defaultdict(list)
        for t in T:
            sp = t['ph'].get('Speech', {})
            if 'Start' not in sp or 'End' not in sp: continue
            s0, s1 = sp['Start'], sp['End']; bm = (tf >= s0 - u['baseline_s']) & (tf < s0)
            if bm.sum() < 5: continue
            b = rms[bm]; thr = np.median(b) + u['mad_k'] * 1.4826 * np.median(np.abs(b - np.median(b)))
            seg = np.where((tf >= s0) & (tf < s1))[0]; above = rms[seg] >= thr; onset = None
            for j in range(len(seg) - need + 1):
                if above[j:j + need].all(): onset = tf[seg[j]]; break
            cond = t['mode'] + t['kind']; excl = (sub == '15' and ses == '02') or (sub == '13' and ses == '02' and t['block_from_end'] == 3)
            n_all[cond] += 1; rows.append(dict(session=key, cond=cond, latency=None if onset is None else onset - s0, primary=not excl))
            if onset is not None: n_det[cond] += 1; lat[cond].append(onset - s0)
        sessions[key] = dict(status='ok', fs_in=int(sf.info(wav).samplerate), detected={k: n_det[k] / n_all[k] for k in n_all}, n={k: n_all[k] for k in n_all},
                             median_latency={k: float(np.median(v)) for k, v in lat.items() if v})
        print(key, json.dumps(sessions[key]['detected']), flush=True)
    def summ(sel):
        out = {}
        for cond in ('RealWords', 'RealSyllables', 'SilentWords', 'SilentSyllables'):
            rr = [r for r in sel if r['cond'] == cond]; L = np.array([r['latency'] for r in rr if r['latency'] is not None])
            out[cond] = dict(n=len(rr), detected_rate=len(L) / max(1, len(rr)))
            if len(L): out[cond].update(median=float(np.median(L)), iqr=[float(np.percentile(L, 25)), float(np.percentile(L, 75))],
                                        p10=float(np.percentile(L, 10)), p90=float(np.percentile(L, 90)), sd=float(L.std()))
        return out
    res['u_overt'] = dict(sessions=sessions, primary=summ([r for r in rows if r['primary']]), all_sessions=summ(rows))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--part', default='all'); a = ap.parse_args(); c = yaml.safe_load(open(a.config)); t0 = time.time()
    out = os.path.join(ROOT, c['out']); os.makedirs(out, exist_ok=True); path = os.path.join(out, 'ugr.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    if a.part in ('ra', 'all'): part_ra(c, res); print(json.dumps({k: {m: round(v['acc'], 4) for m, v in x['meters'].items()} for k, x in res['ra'].items()}, indent=0), flush=True)
    if a.part in ('u', 'all'): part_u(c, res); print(json.dumps(res['u_overt']['primary'], indent=0), flush=True)
    res['seconds'] = time.time() - t0; json.dump(res, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()
