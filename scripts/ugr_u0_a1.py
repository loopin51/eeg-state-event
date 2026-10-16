"""D-110 amendment 1 (post-result change approved by the user 2026-10-16): U0 stimulus-meter gate with predictive meters (no EEG).

Unchanged from D-110 section 3 (scripts/ugr_u0.py): query set Q, strata L0 and the single narrowing L1, rank (below + 0.5 ties) /
(|C| - 1), within-block label permutation null (seed 20261401), query bootstrap within target (seed 20261402), gate 90% upper
< 0.02, word-count rule, R-d. BLOCKPOS, ADJ_table and ADJ_copy keep their scores and null.
Changed:
  DUR, NLET (and the report meters SYL, STRESS, FONINI, LEXICAL, ENV10, LOWDIM) follow the D-098 gate 0 / 0-T / 0-T2 form: the
  attributes of the query's true word are mapped by ridge (stageB_b1_meters.fit_predict, ALPHAS, inner CV over the other word
  folds) to the time-mean cue target (D-110 2.4 clip, w2v layers 14-18, standardised with the fold's training words); the score
  of candidate v is the cosine between that prediction and v's standardised target. Word-level folds: words ordered by
  sha256(word + salt), fold = rank % n; the pair (w, v) is scored by the model fitted without the folds of both w and v (fold
  pairs), because a candidate inside the fit is pulled towards the prediction of its stratum neighbours (first info-0 check).
  FREQ keeps its score (ses-01 count of v); its null is item-uniform: the truth uniform over C(w), i.e. the null MRP of a query is
  the mean rank over v in C(w) (exactly 0.5).
Modes:
  --info0  targets replaced by N(0, 1) (seed info0.seed + 10 k for draw k) for the ridge meters; FREQ truths drawn uniformly
           from C(w) (seed + 1). BLOCKPOS / ADJ are not computed. Expected excess ~ 0. Writes <out stem>_info0.json.
  --plant  targets = N(0, 1) + plant_snr x z(dur) x g (g ~ N(0, 1), seed info0.plant_seed + 10 k); ridge meters only.
  --reps R repeat the info0 / plant draw R times; the file keeps draw 0 in full and, per level and meter, the mean and sd of
           the excess, and for both interval methods the mean 90% upper and the share of draws with 90% upper >= the gate
           threshold (info0: gate failure share; plant: detection share).
Intervals: query bootstrap within target (registered, ci90) and word (cluster) bootstrap (ci90_word: words of the query set
resampled with replacement, all queries of a drawn word kept, seed word_bootstrap.seed). The verdict uses ci_method, fixed from
the info-0 draws before any real-target value (selection rule of the coordination session, D-110 amendment 1).
  (none)   real cue targets, all meters, the registered verdict. Writes u0_amend1.json.

usage: python scripts/ugr_u0_a1.py configs/ugr_u.yaml [--info0 | --plant] [--reps R]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, math, hashlib, argparse, collections, yaml, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ugr_u0 import nfc, wav_name, xlsx_rows, cue_features, ranks, mrp, read_events
from stageB_b1_meters import fit_predict, unit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(c):
    """vocabulary, characteristics, cue features and Q exactly as scripts/ugr_u0.py main()"""
    D = c['data_root']; M = os.path.join(D, c['materials']); char = {}
    for r in xlsx_rows(os.path.join(M, 'Materials characteristics.xlsx')):
        w = r.get('Spanish word', '')
        if not w or r.get('Category') in ('Practice trial', 'Instruction word') or not r.get('Syl_num'): continue
        char[nfc(w)] = dict(Syl_num=int(float(r['Syl_num'])), stressEnd=int(float(r['stressEnd'])), n_letters=int(float(r['n_letters'])),
                            fon_ini=nfc(r.get('fon_ini', '')), lexical='pseudo' if r['Category'] == 'Pseudoword' else 'real')
    ev = {'/'.join(f.split('/')[-4:-2]): read_events(f, 'trial_type') for f in glob.glob(os.path.join(D, c['events_glob']))}
    words = sorted({nfc(x['item']) for T in ev.values() for x in T if x['kind'] == 'Words'}); V = [w for w in words if w in char]
    feat = {w: cue_features(os.path.join(M, wav_name(w, 'Words') + '.wav'), c['u0']['vad']) for w in V}
    Q = []; excl = c['subjects']['primary_exclude_blocks']
    for s in c['subjects']['targets']:
        A, B = ev[f'{s}/ses-01'], ev[f'{s}/ses-02']
        freq = collections.Counter(nfc(x['item']) for x in A if x['kind'] == 'Words'); table = collections.defaultdict(collections.Counter)
        for p, q in zip(A[:-1], A[1:]): table[nfc(p['item'])][nfc(q['item'])] += 1
        bp = {(x['block'], x['pos']): nfc(x['item']) for x in A}
        for i, x in enumerate(B):
            if not (x['mode'] == 'Real' and x['kind'] == 'Words' and x['cue'] == 'Audio'): continue
            if any(x['block_from_end'] == -k for k in excl.get(f'{s}/ses-02', [])): continue
            w = nfc(x['item'])
            if w not in char: continue
            prev = nfc(B[i - 1]['item']) if i > 0 else None
            Q.append(dict(subject=s, block=x['block'], word=w, prev=prev, freq=dict(freq), adj=dict(table.get(prev, {})), blockpos=bp.get((x['block'], x['pos']))))
    return words, V, char, feat, Q


def attributes(V, char, feat):
    """per-word attribute blocks (rows follow V)"""
    def onehot(vals):
        cats = sorted(set(vals)); return np.array([[float(v == k) for k in cats] for v in vals])
    A = dict(dur=np.array([[feat[w]['dur']] for w in V]), n_letters=np.array([[float(char[w]['n_letters'])] for w in V]),
             Syl_num=np.array([[float(char[w]['Syl_num'])] for w in V]), stressEnd_onehot=onehot([char[w]['stressEnd'] for w in V]),
             fon_ini_onehot=onehot([char[w]['fon_ini'] for w in V]), lexical=np.array([[float(char[w]['lexical'] == 'pseudo')] for w in V]),
             env10=np.array([feat[w]['env10'] for w in V]))
    return A


def word_scores(X, Y, fold):
    """S[w, v] = cosine(ridge prediction for w, standardised target of v), both from the model fitted without the folds of w and v
    (fold pair f <= g; inner CV over the remaining folds). Neither the query word nor the candidate enters the fit, so under
    random targets the truth and its candidates are exchangeable."""
    S = np.full((len(Y), len(Y)), np.nan); alphas = {}; F = np.unique(fold)
    for i, f in enumerate(F):
        for g in F[i:]:
            tr = (fold != f) & (fold != g); te = ~tr; fm, fsd = Y[tr].mean(0), Y[tr].std(0) + 1e-8; Ys = (Y - fm) / fsd
            P, a = fit_predict(X[tr], Ys[tr], X[te], fold[tr]); Sp = unit(P) @ unit(Ys[te]).T; ti = np.where(te)[0]
            pair = (fold[ti][:, None] == f) & (fold[ti][None, :] == g) | (fold[ti][:, None] == g) & (fold[ti][None, :] == f)
            S[np.ix_(ti, ti)] = np.where(pair, Sp, S[np.ix_(ti, ti)]); alphas[f'{f}{g}'] = a
    assert not np.isnan(S).any()
    return S, alphas


CI_KEY = dict(query='ci90', word='ci90_word')


def rank_in(s, t, cs):
    """normalised rank of t among the candidate indices cs (t in cs) for the score row s"""
    o = cs[cs != t]; return float(((s[o] < s[t]).sum() + 0.5 * (s[o] == s[t]).sum()) / len(o))


def run_gate(c, V, char, feat, Q, strata, S_word, label, mode, freq_seed=None):
    u0 = c['u0']; am = c['u0_amend1']; idx = {w: i for i, w in enumerate(V)}
    dur = np.array([feat[w]['dur'] for w in V]); syl = np.array([char[w]['Syl_num'] for w in V]); stress = np.array([char[w]['stressEnd'] for w in V])
    lex = np.array([char[w]['lexical'] for w in V])
    Cm = (syl[:, None] == syl[None, :]) & (stress[:, None] == stress[None, :]) & (np.abs(dur[None, :] - dur[:, None]) <= strata['dur_tol'] * dur[:, None])
    if 'lexical' in strata.get('add_same', []): Cm &= lex[:, None] == lex[None, :]
    size = Cm.sum(1); okw = size >= u0['strata']['min_candidates']
    q = [x for x in Q if okw[idx[x['word']]]]
    T = np.array([idx[x['word']] for x in q]); subj = np.array([x['subject'] for x in q]); blk = [(x['subject'], x['block']) for x in q]
    S = {name: Sw[T] for name, Sw in S_word.items()}                      # query-side information = the original true word
    item_uniform = set()
    if mode != 'plant':
        S['FREQ'] = np.array([[x['freq'].get(w, 0) for w in V] for x in q], float); item_uniform.add('FREQ')
    if mode == 'real':
        S['BLOCKPOS'] = np.array([[float(x['blockpos'] == w) for w in V] for x in q])
        S['ADJ_table'] = np.array([[x['adj'].get(w, 0) for w in V] for x in q], float)
        S['ADJ_copy'] = np.array([[float(x['prev'] == w) for w in V] for x in q])
    T_freq = T
    if mode == 'info0':                                                      # FREQ under info 0: truth uniform over C(w)
        rf = np.random.default_rng(freq_seed); T_freq = np.array([rf.choice(np.where(Cm[t])[0]) for t in T])
    groups = collections.defaultdict(list)
    for i, g in enumerate(blk): groups[g].append(i)
    rp = np.random.default_rng(u0['null_perm']['seed']); perms = []
    for _ in range(u0['null_perm']['n']):
        Tp = T.copy()
        for ii in groups.values(): Tp[ii] = T[rp.permutation(ii)]
        perms.append(Tp)
    rb = np.random.default_rng(u0['bootstrap']['seed']); bysub = {s: np.where(subj == s)[0] for s in sorted(set(subj))}
    boots = [np.concatenate([rb.choice(ii, len(ii)) for ii in bysub.values()]) for _ in range(u0['bootstrap']['n'])]
    rw = np.random.default_rng(am['word_bootstrap']['seed']); uw = np.unique(T); byword = {w: np.where(T == w)[0] for w in uw}
    wboots = [np.concatenate([byword[w] for w in rw.choice(uw, len(uw))]) for _ in range(am['word_bootstrap']['n'])]
    gate_names = [m for m in u0['gate_meters'] if m in S]; out = {}
    for name in gate_names + [m for m in am['ridge_report_meters'] if m in S]:
        if name in item_uniform:                                             # candidate set stays C(w) of the query's original word
            r, nq = np.zeros(len(T)), np.zeros(len(T))
            for i, (w0, t) in enumerate(zip(T, T_freq)):
                cs = np.where(Cm[w0])[0]; r[i] = rank_in(S[name][i], t, cs); nq[i] = np.mean([rank_in(S[name][i], v, cs) for v in cs])
            null_kind = 'item_uniform'; d = r - nq; ex = float(d.mean()); bt = np.array([d[b].mean() for b in boots]); bw = np.array([d[b].mean() for b in wboots])
            per = {s: float(d[ii].mean()) for s, ii in bysub.items()}; nm = float(nq.mean())
        else:
            r = mrp(S[name], T, Cm); null = np.array([mrp(S[name], Tp, Cm).mean() for Tp in perms]); null_kind = 'within_block_permutation'
            nm = float(null.mean()); ex = float(r.mean() - nm); bt = np.array([r[b].mean() for b in boots]) - nm; bw = np.array([r[b].mean() for b in wboots]) - nm
            per = {s: float(r[ii].mean() - np.mean([mrp(S[name], Tp, Cm)[ii].mean() for Tp in perms[:200]])) for s, ii in bysub.items()}
        out[name] = dict(gate=name in u0['gate_meters'], null=null_kind, mrp=float(r.mean()), null_mean=nm, excess=ex,
                         ci90=[float(np.percentile(bt, 5)), float(np.percentile(bt, 95))], ci90_word=[float(np.percentile(bw, 5)), float(np.percentile(bw, 95))],
                         per_subject=per)
    nwords = int(okw.sum()); res = dict(label=label, strata=strata, n_queries=len(T), n_queries_before_size_filter=len(Q),
                                        queries_per_subject={s: int(len(ii)) for s, ii in bysub.items()}, words_with_min_candidates=nwords,
                                        candidate_size=dict(min=int(size[okw].min()), median=float(np.median(size[okw])), max=int(size[okw].max())), meters=out)
    if mode == 'real':
        ck = CI_KEY[am['ci_method']]; gate_ok = all(out[m][ck][1] < u0['pass_excess_upper90'] for m in u0['gate_meters'])
        words_ok = nwords >= u0['strata']['min_words_with_min_candidates']
        K = Cm[T].sum(1); p0 = float(np.mean(1.0 / K)); n = len(T); rd = u0['rd']; p = p0 + rd['z'] * math.sqrt(p0 * (1 - p0) / n); Km = float(K.mean())
        bits = math.log2(Km) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (Km - 1))
        res.update(ci_method=am['ci_method'], gate_meters_pass=bool(gate_ok), words_ok=bool(words_ok), passed=bool(gate_ok and words_ok),
                   rd=dict(p0=p0, n=n, p_mde=p, mean_K=Km, bits_per_trial=bits, bits_per_min=bits * rd['trials_per_min'], meets_0p3=bool(bits * rd['trials_per_min'] <= rd['max_bits_per_min'])))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); g = ap.add_mutually_exclusive_group(); g.add_argument('--info0', action='store_true'); g.add_argument('--plant', action='store_true')
    ap.add_argument('--reps', type=int, default=1); a = ap.parse_args(); mode = 'info0' if a.info0 else 'plant' if a.plant else 'real'
    c = yaml.safe_load(open(os.path.join(ROOT, a.config))); u0 = c['u0']; am = c['u0_amend1']; i0 = am['info0']
    words, V, char, feat, Q = build(c)
    tj = json.load(open(os.path.join(ROOT, c['cache'], 'cue_targets.json'))); TF = np.load(os.path.join(ROOT, c['cache'], 'cue_targets.npy'), mmap_mode='r')
    row = {s: i for i, s in enumerate(tj['stems'])}; Y_real = np.stack([TF[row[wav_name(w, 'Words')]].astype(np.float64).mean(0) for w in V])
    order = sorted(V, key=lambda w: hashlib.sha256((w + am['word_folds']['salt']).encode()).hexdigest())
    fold = np.array([order.index(w) % am['word_folds']['n'] for w in V]); A = attributes(V, char, feat)
    L0 = dict(same=u0['strata']['same'], dur_tol=u0['strata']['dur_tol'])
    nar = dict(same=u0['strata']['same'], dur_tol=u0['strata']['narrow_once']['dur_tol'], add_same=u0['strata']['narrow_once']['add_same'])

    def evaluate(Y, freq_seed):
        S_word, alphas = {}, {}
        for name, blocks in list(am['ridge_meters'].items()) + list(am['ridge_report_meters'].items()):
            S_word[name], alphas[name] = word_scores(np.hstack([A[b] for b in blocks]), Y, fold)
        res = dict(mode=mode, amendment='D-110 amendment 1 (post-result change, user approval 2026-10-16)', n_vocab=len(words), n_words=len(V),
                   word_fold_sizes=np.bincount(fold).tolist(), ridge_alpha=alphas, target_rows=list(Y.shape))
        first = run_gate(c, V, char, feat, Q, L0, S_word, 'L0', mode, freq_seed); res['L0'] = first
        if mode != 'real':
            res['L1'] = run_gate(c, V, char, feat, Q, nar, S_word, 'L1', mode, freq_seed)
        else:
            res['verdict'] = 'pass' if first['passed'] else None
            if not first['passed']:
                second = run_gate(c, V, char, feat, Q, nar, S_word, 'L1', mode); res['L1'] = second
                res['verdict'] = 'pass (narrowed once)' if second['passed'] else 'fail: U2 closed ("cannot be asked with these stimuli")'
        return res

    if mode == 'real':
        res = evaluate(Y_real, None)
    else:
        d = np.array([feat[w]['dur'] for w in V]); z = (d - d.mean()) / d.std(); draws = []
        for k in range(a.reps):
            if mode == 'info0': Y = np.random.default_rng(i0['seed'] + 10 * k).normal(size=Y_real.shape); fs = i0['seed'] + 10 * k + 1
            else:
                rg = np.random.default_rng(i0['plant_seed'] + 10 * k); fs = None
                Y = rg.normal(size=Y_real.shape) + i0['plant_snr'] * z[:, None] * rg.normal(size=(1, Y_real.shape[1]))
            draws.append(evaluate(Y, fs))
        res = draws[0]; thr = u0['pass_excess_upper90']; summ = {}
        for L in ('L0', 'L1'):
            gm = [m for m in u0['gate_meters'] if m in res[L]['meters']]
            for m in res[L]['meters']:
                ex = np.array([r[L]['meters'][m]['excess'] for r in draws]); e = dict(mean_excess=float(ex.mean()), sd_excess=float(ex.std(ddof=1)) if len(ex) > 1 else None)
                for meth, ck in CI_KEY.items():
                    up = np.array([r[L]['meters'][m][ck][1] for r in draws]); e[f'{meth}_mean_upper90'] = float(up.mean()); e[f'{meth}_upper_ge_thr_share'] = float((up >= thr).mean())
                summ[f'{L}/{m}'] = e
            summ[f'{L}/any_gate_meter'] = dict(meters=gm, **{f'{meth}_upper_ge_thr_share': float(np.mean([any(r[L]['meters'][m][ck][1] >= thr for m in gm) for r in draws])) for meth, ck in CI_KEY.items()})
        res['draws'] = a.reps; res['summary_over_draws'] = summ
    stem = am['out'][:-5]; p = os.path.join(ROOT, am['out'] if mode == 'real' else f'{stem}_{mode}.json'); os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), indent=1, ensure_ascii=False)
    if mode == 'real': print('written', p)
    else: print(json.dumps(res['summary_over_draws'], ensure_ascii=False))
    print('sha256', hashlib.sha256(open(p, 'rb').read()).hexdigest())


if __name__ == '__main__':
    main()
