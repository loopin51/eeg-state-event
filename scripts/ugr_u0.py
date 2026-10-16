"""D-110 section 3: U0 stimulus-meter gate for UGR-MINDVOICE (no EEG).

Query set Q = ses-02 Real main-trial audio-cue word trials of the six targets (D-100 primary exclusions). Candidate set C(w) =
words with the same Syl_num and stressEnd as w and cue speech duration within dur_tol of w (w included); queries of words with
|C(w)| < min_candidates leave Q. Meters score every candidate from query-side information fixed to the original trial; the rank
of the true word is (below + 0.5 ties) / (|C| - 1). Null: ses-02 labels permuted within (target, block) among the Q trials,
truth and C follow the permuted label, query-side information stays. Excess = observed MRP - null mean (query-weighted over the
six targets); 90% interval = bootstrap of queries within target. Gate: every gate meter's 90% upper < pass_excess_upper90 and
at least min_words_with_min_candidates words with |C| >= min_candidates; on failure the strata are narrowed once.

usage: python scripts/ugr_u0.py configs/ugr_u.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import re, sys, glob, json, math, zipfile, hashlib, argparse, unicodedata, collections, yaml, numpy as np
from scipy.io import wavfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d106_ugr import read as read_events          # D-106 trial / block / position definition (main trials only)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nfc(s): return unicodedata.normalize('NFC', s.strip().lower())


def wav_name(item, kind):
    """D-110 2.4: words drop accents and turn n-tilde into n; syllables keep n-tilde."""
    s = nfc(item)
    if kind == 'Words': return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    return ''.join('\0' if ch == 'ñ' else ch for ch in s).encode('ascii', 'ignore').decode().replace('\0', 'ñ') if 'ñ' in s else \
        unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()


def xlsx_rows(path, sheet='xl/worksheets/sheet1.xml'):
    z = zipfile.ZipFile(path); ss = []
    if 'xl/sharedStrings.xml' in z.namelist():
        x = z.read('xl/sharedStrings.xml').decode('utf8'); ss = [re.sub(r'<[^>]+>', '', m) for m in re.findall(r'<si>(.*?)</si>', x, re.S)]
    rows = []
    for r in re.findall(r'<row[^>]*>(.*?)</row>', z.read(sheet).decode('utf8'), re.S):
        d = {}
        for attr, body in re.findall(r'<c ([^>]*?)(?:/>|>(.*?)</c>)', r, re.S):
            col = re.search(r'r="([A-Z]+)\d+"', attr).group(1); v = re.search(r'<v>(.*?)</v>', body or '')
            d[col] = (ss[int(v.group(1))] if 't="s"' in attr else v.group(1)) if v else ''
        rows.append(d)
    hdr = rows[0]; return [{hdr[k]: v for k, v in r.items() if k in hdr} for r in rows[1:]]


def cue_features(path, vad):
    fs, x = wavfile.read(path); x = x.astype(np.float64)
    if x.ndim > 1: x = x.mean(1)
    win, hop = int(vad['frame_ms'] * fs / 1000), int(vad['hop_ms'] * fs / 1000); n = 1 + (len(x) - win) // hop
    c2 = np.r_[0.0, np.cumsum(x ** 2)]; st = hop * np.arange(n); db = 10 * np.log10(np.maximum(c2[st + win] - c2[st], 0) / win + 1e-12)
    thr = db.max() - vad['thresh_db_below_peak']; need = int(math.ceil(vad['min_run_ms'] / vad['hop_ms'])); above = db >= thr
    runs = [j for j in range(n - need + 1) if above[j:j + need].all()]
    on, off = runs[0], runs[-1] + need - 1
    dur = ((off * hop + win) - on * hop) / fs; seg = db[on:off + 1]
    env = np.array([b.mean() for b in np.array_split(seg, 10)]); env = env - env.mean()
    return dict(dur=float(dur), onset_s=float(on * hop / fs), env10=env.tolist(), fs=int(fs))


def ranks(S, T, Cm):
    """S: [Q, V] scores, T: [Q] true word index, Cm: [V, V] candidate mask (row = truth). Normalised rank of the truth."""
    st = S[np.arange(len(T)), T][:, None]; m = Cm[T].copy(); m[np.arange(len(T)), T] = False
    return ((S < st) & m).sum(1) + 0.5 * ((S == st) & m).sum(1), m.sum(1)


def mrp(S, T, Cm):
    b, k = ranks(S, T, Cm); return b / k


def run_gate(c, V, feat, char, Q, strata, rng_seed_perm, rng_seed_boot, label):
    u0 = c['u0']; nV = len(V); idx = {w: i for i, w in enumerate(V)}
    dur = np.array([feat[w]['dur'] for w in V]); syl = np.array([char[w]['Syl_num'] for w in V]); stress = np.array([char[w]['stressEnd'] for w in V])
    lex = np.array([char[w]['lexical'] for w in V])
    Cm = (syl[:, None] == syl[None, :]) & (stress[:, None] == stress[None, :]) & (np.abs(dur[None, :] - dur[:, None]) <= strata['dur_tol'] * dur[:, None])
    if 'lexical' in strata.get('add_same', []): Cm &= lex[:, None] == lex[None, :]
    size = Cm.sum(1); okw = size >= u0['strata']['min_candidates']
    q = [x for x in Q if okw[idx[x['word']]]]
    T = np.array([idx[x['word']] for x in q]); subj = np.array([x['subject'] for x in q]); blk = [(x['subject'], x['block']) for x in q]
    nlet = np.array([float(char[w]['n_letters']) for w in V]); fon = np.array([char[w]['fon_ini'] for w in V]); env = np.array([feat[w]['env10'] for w in V])
    S = {}
    S['DUR'] = -np.abs(dur[None, :] - dur[T][:, None]); S['NLET'] = -np.abs(nlet[None, :] - nlet[T][:, None])
    S['FREQ'] = np.array([[x['freq'].get(w, 0) for w in V] for x in q], float)
    S['BLOCKPOS'] = np.array([[float(x['blockpos'] == w) for w in V] for x in q])
    S['ADJ_table'] = np.array([[x['adj'].get(w, 0) for w in V] for x in q], float)
    S['ADJ_copy'] = np.array([[float(x['prev'] == w) for w in V] for x in q])
    S['SYL'] = -(syl[None, :] != syl[T][:, None]).astype(float); S['STRESS'] = -(stress[None, :] != stress[T][:, None]).astype(float)
    S['FONINI'] = (fon[None, :] == fon[T][:, None]).astype(float); S['LEXICAL'] = (lex[None, :] == lex[T][:, None]).astype(float)
    S['ENV10'] = -np.sqrt(((env[None, :, :] - env[T][:, None, :]) ** 2).sum(-1))
    groups = collections.defaultdict(list)
    for i, g in enumerate(blk): groups[g].append(i)
    rp = np.random.default_rng(rng_seed_perm); perms = []
    for _ in range(u0['null_perm']['n']):
        Tp = T.copy()
        for ii in groups.values(): Tp[ii] = T[rp.permutation(ii)]
        perms.append(Tp)
    rb = np.random.default_rng(rng_seed_boot); bysub = {s: np.where(subj == s)[0] for s in sorted(set(subj))}
    boots = [np.concatenate([rb.choice(ii, len(ii)) for ii in bysub.values()]) for _ in range(u0['bootstrap']['n'])]
    out = {}
    for name in u0['gate_meters'] + u0['report_meters']:
        r = mrp(S[name], T, Cm); null = np.array([mrp(S[name], Tp, Cm).mean() for Tp in perms])
        ex = r.mean() - null.mean(); bt = np.array([r[b].mean() for b in boots]) - null.mean()
        out[name] = dict(gate=name in u0['gate_meters'], mrp=float(r.mean()), null_mean=float(null.mean()), excess=float(ex),
                         ci90=[float(np.percentile(bt, 5)), float(np.percentile(bt, 95))],
                         per_subject={s: float(r[ii].mean() - np.mean([mrp(S[name], Tp, Cm)[ii].mean() for Tp in perms[:200]])) for s, ii in bysub.items()})
    nwords = int(okw.sum()); gate_ok = all(out[m]['ci90'][1] < u0['pass_excess_upper90'] for m in u0['gate_meters'])
    words_ok = nwords >= u0['strata']['min_words_with_min_candidates']
    K = Cm[T].sum(1); p0 = float(np.mean(1.0 / K)); n = len(T); rd = c['u0']['rd']
    p = p0 + rd['z'] * math.sqrt(p0 * (1 - p0) / n); Km = float(K.mean())
    bits = math.log2(Km) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (Km - 1))
    res = dict(label=label, strata=strata, n_queries=n, n_queries_before_size_filter=len(Q), queries_per_subject={s: int(len(ii)) for s, ii in bysub.items()},
               words_with_min_candidates=nwords, candidate_size=dict(min=int(size[okw].min()), median=float(np.median(size[okw])), max=int(size[okw].max())),
               meters=out, gate_meters_pass=bool(gate_ok), words_ok=bool(words_ok), passed=bool(gate_ok and words_ok),
               rd=dict(p0=p0, n=n, p_mde=p, mean_K=Km, bits_per_trial=bits, bits_per_min=bits * rd['trials_per_min'], meets_0p3=bool(bits * rd['trials_per_min'] <= rd['max_bits_per_min'])))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    c = yaml.safe_load(open(os.path.join(ROOT, a.config))); D = c['data_root']; M = os.path.join(D, c['materials'])
    char = {}
    for r in xlsx_rows(os.path.join(M, 'Materials characteristics.xlsx')):
        w = r.get('Spanish word', '')
        if not w or r.get('Category') in ('Practice trial', 'Instruction word') or not r.get('Syl_num'): continue
        char[nfc(w)] = dict(Syl_num=int(float(r['Syl_num'])), stressEnd=int(float(r['stressEnd'])), n_letters=int(float(r['n_letters'])),
                            fon_ini=nfc(r.get('fon_ini', '')), lexical='pseudo' if r['Category'] == 'Pseudoword' else 'real')
    ev = {}
    for f in glob.glob(os.path.join(D, c['events_glob'])):
        ev['/'.join(f.split('/')[-4:-2])] = read_events(f, 'trial_type')
    words = sorted({nfc(x['item']) for T in ev.values() for x in T if x['kind'] == 'Words'})
    unmapped = sorted(w for w in words if w not in char)
    V = [w for w in words if w in char]
    feat = {w: cue_features(os.path.join(M, wav_name(w, 'Words') + '.wav'), c['u0']['vad']) for w in V}
    Q = []; excl = c['subjects']['primary_exclude_blocks']
    for s in c['subjects']['targets']:
        A, B = ev[f'{s}/ses-01'], ev[f'{s}/ses-02']
        freq = collections.Counter(nfc(x['item']) for x in A if x['kind'] == 'Words')
        table = collections.defaultdict(collections.Counter)
        for p, q in zip(A[:-1], A[1:]): table[nfc(p['item'])][nfc(q['item'])] += 1
        bp = {(x['block'], x['pos']): nfc(x['item']) for x in A}
        for i, x in enumerate(B):
            if not (x['mode'] == 'Real' and x['kind'] == 'Words' and x['cue'] == 'Audio'): continue
            if any(x['block_from_end'] == -k for k in excl.get(f'{s}/ses-02', [])): continue
            w = nfc(x['item'])
            if w not in char: continue
            prev = nfc(B[i - 1]['item']) if i > 0 else None
            Q.append(dict(subject=s, block=x['block'], word=w, prev=prev, freq=dict(freq), adj=dict(table.get(prev, {})), blockpos=bp.get((x['block'], x['pos']))))
    u0 = c['u0']; res = dict(n_vocab=len(words), unmapped_items=unmapped, cue=dict(dur_s=dict(min=min(f['dur'] for f in feat.values()), median=float(np.median([f['dur'] for f in feat.values()])),
                                                                                                max=max(f['dur'] for f in feat.values()))))
    first = run_gate(c, V, feat, char, Q, dict(same=u0['strata']['same'], dur_tol=u0['strata']['dur_tol']), u0['null_perm']['seed'], u0['bootstrap']['seed'], 'L0')
    res['L0'] = first; res['verdict'] = 'pass' if first['passed'] else None
    if not first['passed']:
        nar = dict(same=u0['strata']['same'], dur_tol=u0['strata']['narrow_once']['dur_tol'], add_same=u0['strata']['narrow_once']['add_same'])
        second = run_gate(c, V, feat, char, Q, nar, u0['null_perm']['seed'], u0['bootstrap']['seed'], 'L1')
        res['L1'] = second; res['verdict'] = 'pass (narrowed once)' if second['passed'] else 'fail: U2 closed ("cannot be asked with these stimuli")'
    res['cue_features'] = feat
    os.makedirs(os.path.join(ROOT, c['out']), exist_ok=True); p = os.path.join(ROOT, c['out'], 'u0.json')
    json.dump(res, open(p, 'w'), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != 'cue_features'}, ensure_ascii=False)[:6000])
    print('sha256', hashlib.sha256(open(p, 'rb').read()).hexdigest())


if __name__ == '__main__':
    main()
