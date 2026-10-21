"""D-108 / D-109 T2 stimulus gate 0-T2 (no EEG; configs/t2_gate0T2.yaml, rules reports/07_fiv_stage/notes/t2_gate0T2_draft_kr.md).

Queries are the B1 test queries with a valid af_heart TTS window (as 0-T). The text unit U(q) is the words with heard onsets in
[t - 0.5, t + 2.5). Matching attributes: word count, letters, syllables (n_syl) and the TTS speech time of U (VAD frames between
the TTS onsets of U's first word and of the next word). Levels L0-L3 are relaxed only in order; the level is the first one
where >= 80% of the queries have >= 19 eligible candidates (else the highest coverage, ties -> stricter). 19 candidates per query
by sha256 order; queries with fewer are left out.
(i) heard w2v window -> the 20 TTS windows (clip_scores, MRP (below + 0.5 ties)/19, run bootstrap 90% lower >= 0.90).
(ii) six meters (query-side heard + TTS attributes of the true text) -> ridge to the time-pooled target of the candidate domain
(TTS windows / heard windows), cosine ranking among the same 20 candidates, excess over the circular shift null, 90% upper < 0.02
in both domains. (iii) coverage >= 0.50. Report only: ENV_T, level coverages, attribute residuals, unmatched values, delay search.

usage: python scripts/t2_gate0T2.py configs/t2_gate0T2.yaml [--smoke]
  --smoke: run fold 0 only, print array shapes and element counts (no coverage, MRP or meter value), write nothing.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, hashlib, argparse, math, yaml, numpy as np, torch, soundfile as sf
from scipy.signal import resample_poly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageB_b1_meters import fit_predict, unit
from d100_tau50 import run_maps

SENT, COMMA, OTHER = '.?', ',', ':-'


def sha(s): return hashlib.sha256(s.encode()).hexdigest()


def load_audio(path, sr_out):
    x, sr = sf.read(path, dtype='float64'); x = x.mean(1) if x.ndim == 2 else x; g = math.gcd(int(sr), int(sr_out))
    return resample_poly(x, sr_out // g, int(sr) // g) if sr != sr_out else x


def vad(x, sr, v):
    """speech flag per frame k (frame start k * hop): 20 log10(RMS) >= run p95 - drop"""
    n, h = int(round(v['win_s'] * sr)), int(round(v['hop_s'] * sr)); c = np.r_[0.0, np.cumsum(x ** 2)]
    k = np.arange(0, (len(x) - n) // h + 1) * h; db = 20 * np.log10(np.sqrt((c[k + n] - c[k]) / n) + 1e-10)
    return db >= np.percentile(db, v['ref_percentile']) - v['drop_db']


def envelope(x, sr, fs_env, win):
    """RMS envelope sampled at k / fs_env (window centred, win s)"""
    c = np.r_[0.0, np.cumsum(x ** 2)]; ctr = np.arange(int(np.floor(len(x) / sr * fs_env))) / fs_env
    a = np.clip(np.round((ctr - win / 2) * sr).astype(int), 0, len(x)); b = np.clip(np.round((ctr + win / 2) * sr).astype(int), 0, len(x))
    return np.sqrt((c[b] - c[a]) / np.maximum(b - a, 1))


def env_window(e, t0, fs, T):
    i = int(round(t0 * fs)); out = np.zeros(T); lo, hi = max(i, 0), min(i + T, len(e)); out[lo - i:hi - i] = e[lo:hi]; return out


def modspec(w, fs, bands, total):
    x = (w - w.mean()) * np.hanning(len(w)); P = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(w), 1 / fs)
    sel = lambda lo, hi, last: (f >= lo) & ((f <= hi) if last else (f < hi))
    return [np.log(P[sel(lo, hi, k == len(bands) - 1)].sum() + 1e-12) for k, (lo, hi) in enumerate(bands)] + [np.log(P[sel(*total, True)].sum() + 1e-12)]


def nonspeech_runs(flags, min_len):
    """count and total length (frames) of False runs >= min_len"""
    d = np.diff(np.r_[1, flags.astype(int), 1]); s, e = np.where(d == -1)[0], np.where(d == 1)[0]; L = e - s; L = L[L >= min_len]
    return len(L), int(L.sum())


def posterms(p): return [p, p ** 2, p ** 3] + [g(2 * np.pi * k * p) for k in (1, 2, 4, 8) for g in (np.sin, np.cos)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args(); t0 = time.time()
    c = yaml.safe_load(open(a.config)); c1 = yaml.safe_load(open(os.path.join(ROOT, c['t1_config']))); b1 = yaml.safe_load(open(os.path.join(ROOT, c1['b1_config'])))
    sp = b1['split']; g = c['gate']; mt = c['matching']; vv = c['vad']; rt = g['rate']; dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = os.path.join(ROOT, b1['cache_dir']); meta = json.load(open(os.path.join(d, 'eeg_meta.json'))); fs = meta['sample_rate']; offs = meta['run_offsets']; lens = meta['run_lengths']
    w, e = b1['window'], b1['eeg']; T = int(round((w['tmax'] - w['tmin']) * fs)); lo = -w['tmin']
    def hi(r): return lens[r] / fs - w['tmax'] - max(e['offset_s'], 0)
    Wd = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True)
    F = np.load(os.path.join(d, b1['features']['w2v'] + '.npy'), mmap_mode='r')
    vd = os.path.join(ROOT, c1['cache'], c['voice']); tm = json.load(open(os.path.join(vd, 'meta.json')))
    TF = np.load(os.path.join(vd, 'w2v_run.npy'), mmap_mode='r'); toffs, tlens = tm['run_offsets'], tm['run_lengths']
    js = [json.load(open(os.path.join(vd, f'run{r:02d}.json'))) for r in range(20)]
    fold_of = {r - 1: f for f in range(5) for r in sp['run_folds'][f]}; folds = [0] if a.smoke else list(range(5))
    runs_used = sorted(r - 1 for f in folds for r in sp['run_folds'][f])
    stats = {}
    for f in range(5):                                                       # as 0-T
        test = [r - 1 for r in sp['run_folds'][f]]; valid = [r - 1 for r in sp['valid_runs'][f]]; train = [r for r in range(20) if r not in test + valid]
        samp = np.concatenate([np.arange(offs[r], offs[r] + lens[r]) for r in train]); samp = np.sort(samp[:: max(1, len(samp) // 200000)])
        Fs = np.asarray(F[:, samp], np.float32); stats[f] = (train, Fs.mean(1, keepdims=True), Fs.std(1, keepdims=True) + 1e-6)
    story0 = np.r_[0, np.cumsum([lens[r] / fs for r in range(20)])]
    # ---- per-run words, audio, VAD, envelopes
    WR, Qs = {}, []
    for r in range(20):
        m = np.where(Wd['run'] == r)[0]; on, off = Wd['onset'][m], Wd['offset'][m]; uq, first = np.unique(on, return_index=True)
        keep = (uq >= lo) & (uq <= hi(r)); ton = np.asarray(js[r]['onsets_s'], float)
        xt = load_audio(os.path.join(vd, f'run{r:02d}.wav'), rt['audio_fs']); xa = load_audio(os.path.join(ROOT, c['stimuli'], f'audio{r + 1:02d}.wav'), rt['audio_fs'])
        sp_flags = vad(xt, rt['audio_fs'], vv); csp = np.r_[0, np.cumsum(sp_flags)]
        WR[r] = dict(on=on, off=off, ton=ton, nlet=Wd['n_letters'][m], nsyl=Wd['n_syl'][m], punct=Wd['punct'][m], word=[str(x).lower() for x in Wd['word'][m]],
                     sp=sp_flags, csp=csp, tdur=js[r]['duration_s'], env_a=envelope(xa, rt['audio_fs'], rt['env_fs'], rt['rms_win_s']),
                     env_t=envelope(xt, rt['audio_fs'], rt['env_fs'], rt['rms_win_s']))
        for t, k in zip(uq[keep], first[keep]):
            tau = ton[k]; i0 = int(round((tau + w['tmin']) * fs))
            if i0 >= 0 and i0 + T <= tlens[r]: Qs.append((r, float(t), int(k), float(tau)))
    print('queries valid', len(Qs), f'({time.time() - t0:.0f}s)', flush=True)
    hop = vv['hop_s']
    def speech_s(r, a_, b_):
        R = WR[r]; i, j = int(np.ceil(a_ / hop - 1e-9)), int(np.ceil(b_ / hop - 1e-9)); i, j = np.clip([i, j], 0, len(R['sp'])); return (R['csp'][j] - R['csp'][i]) * hop
    # ---- attributes and meter inputs per valid query
    A = dict(nw=[], nL=[], nS=[], DT=[], seq=[]); X = {k: [] for k in g['meters']}; ENVA, ENVT = [], []
    for r, t, k, tau in Qs:
        R = WR[r]; on = R['on']; U = np.where((on >= t + w['tmin']) & (on < t + w['tmax']))[0]; k0, k1 = U[0], U[-1]
        end = R['ton'][k1 + 1] if k1 + 1 < len(on) else R['tdur']; DT = speech_s(r, R['ton'][k0], end)
        TW = np.where((R['ton'] >= tau + w['tmin']) & (R['ton'] < tau + w['tmax']))[0]
        nL, nw, nS = float(R['nlet'][U].sum()), float(len(U)), float(R['nsyl'][U].sum()); tL, tw, tS = float(R['nlet'][TW].sum()), float(len(TW)), float(R['nsyl'][TW].sum())
        A['nw'].append(nw); A['nL'].append(nL); A['nS'].append(nS); A['DT'].append(DT); A['seq'].append(' '.join(R['word'][i] for i in U))
        X['LOWONLY_PLUS'].append([nL, nw, nS, tL, tw, tS] + [v ** 2 for v in (nL, nw, nS, tL, tw, tS)])
        tws = speech_s(r, tau + w['tmin'], tau + w['tmax']); hs = float(np.sum(np.clip(np.minimum(R['off'][U], t + w['tmax']) - np.maximum(on[U], t + w['tmin']), 0, None)))
        X['TTS_DUR'].append([DT, tws, hs, DT ** 2, tws ** 2, hs ** 2])
        gaps = on[U[1:]] - R['off'][U[:-1]]; gp = gaps[gaps >= vv['pause_min_s']]
        i0 = int(np.ceil((tau + w['tmin']) / hop - 1e-9)); i1 = int(np.ceil((tau + w['tmax']) / hop - 1e-9)); npz, nlen = nonspeech_runs(R['sp'][max(i0, 0):i1], int(round(vv['pause_min_s'] / hop)))
        pc = lambda idx: [sum(R['punct'][i] in SENT for i in idx), sum(R['punct'][i] in COMMA for i in idx), sum(R['punct'][i] in OTHER for i in idx)]
        X['PAUSE_PUNCT'].append([len(gp), float(gp.sum()), npz, nlen * hop] + pc(U) + pc(TW))
        ea = env_window(R['env_a'], t + w['tmin'], rt['env_fs'], T); et = env_window(R['env_t'], tau + w['tmin'], rt['env_fs'], T)
        X['RATE'].append(modspec(ea, rt['env_fs'], rt['bands_hz'], rt['total_hz']) + modspec(et, rt['env_fs'], rt['bands_hz'], rt['total_hz']))
        ENVA.append(ea); ENVT.append(et)
        P_ = R['punct']; prevS = [i for i in range(k0) if P_[i] in SENT]; prevP = [i for i in range(k0) if P_[i] != '0']
        fS = [i for i in U if P_[i] in SENT]
        X['SENTPOS'].append([np.log1p(k0 - 1 - prevS[-1] if prevS else k0), np.log1p(k0 - 1 - prevP[-1] if prevP else k0), len(fS), int(sum(P_[i] != '0' for i in U)),
                             (R['off'][fS[0]] - (t + w['tmin'])) / (w['tmax'] - w['tmin']) if fS else -1.0, sum(P_[i] in SENT for i in TW)])
        X['ORDER'].append(posterms(t / (lens[r] / fs)) + posterms((story0[r] + t) / story0[-1]))
    A = {k: np.array(v) for k, v in A.items()}; X = {k: np.asarray(v, float) for k, v in X.items()}; ENVA, ENVT = np.array(ENVA), np.array(ENVT)
    qr = np.array([q[0] for q in Qs]); qt = np.array([q[1] for q in Qs]); qk = np.array([q[2] for q in Qs]); qtau = np.array([q[3] for q in Qs]); N = len(Qs)
    keys = [f'r{r + 1}_w{k}' for r, k in zip(qr, qk)]
    print('attributes', {k: v.shape for k, v in X.items()}, f'({time.time() - t0:.0f}s)', flush=True)
    # ---- matching
    def eligible(q, lv):
        pool = np.where(qr == qr[q])[0] if lv['pool'] == 'same_run' else np.where(np.isin(qr, [r - 1 for r in sp['run_folds'][fold_of[qr[q]]]]))[0]
        ok = (qr[pool] != qr[q]) | (np.abs(qt[pool] - qt[q]) >= mt['min_sep_s'])
        ok &= A['nw'][pool] == A['nw'][q]; ok &= np.abs(A['nL'][pool] - A['nL'][q]) <= lv['letters_rel'] * A['nL'][q] + 1e-9
        ok &= np.abs(A['nS'][pool] - A['nS'][q]) <= lv['syl_abs']; ok &= np.abs(A['DT'][pool] - A['DT'][q]) <= lv['dur_rel'] * A['DT'][q] + 1e-9
        ok &= A['seq'][pool] != A['seq'][q]; ok &= pool != q; return pool[ok]
    scope = np.where(np.isin(qr, runs_used))[0]
    cover = {}
    for lv in mt['levels']:
        cover[lv['name']] = float(np.mean([len(eligible(q, lv)) >= mt['n_cand'] for q in scope]))
    if a.smoke: print('smoke: levels evaluated', len(cover), 'scope queries', len(scope), flush=True)
    names = [lv['name'] for lv in mt['levels']]; hit = [n for n in names if cover[n] >= mt['level_coverage_target']]
    level = hit[0] if hit else max(names, key=lambda n: (cover[n], -names.index(n))); LV = mt['levels'][names.index(level)]
    cand = {}
    for q in scope:
        el = eligible(q, LV)
        if len(el) >= mt['n_cand']: cand[q] = np.r_[q, sorted(el, key=lambda j: sha(keys[q] + '|' + keys[j] + mt['salt']))[:mt['n_cand']]]
    keep = np.array(sorted(cand)); coverage = len(keep) / len(scope)
    LVs = dict(LV, pool='same_run'); cand_same = {}
    for q in keep:
        el = eligible(q, LVs); cand_same[q] = sorted(el, key=lambda j: sha(keys[q] + '|' + keys[j] + mt['salt']))[:mt['n_cand']] if len(el) >= mt['n_cand'] else []
    print('matching done', f'({time.time() - t0:.0f}s)', flush=True)
    # ---- score matrices per fold: rows = retained queries of the fold's test runs, columns = all valid queries of those runs
    sn = g['shift_null']; rng = np.random.default_rng(g['bootstrap']['seed']); RB = rng.integers(0, 20, (g['bootstrap']['n'], 20))
    retained_by_run = {r: keep[qr[keep] == r] for r in runs_used}
    maps = run_maps({r: qt[retained_by_run[r]] for r in runs_used if len(retained_by_run[r]) >= 2}, sn['n_shift'], sn['min_shift_s'], sn['seed'], 0)
    def stats_from(S, rows, cand_q, cand_rows, run):
        """S [len(rows), ncol] (row order = rows = retained queries of the run), cand_rows [n, K+1] column positions (truth first),
        cand_q the same candidates as query indices. Shift null: truth and candidates of i fixed, estimate -> partner j; same-run
        candidates within min_sep of t_j are left out, (i, draw) pairs with |t_j - t_i| < min_sep are left out (as D-109 section 5)"""
        St = torch.as_tensor(S, device=dev); C = torch.as_tensor(cand_rows, device=dev); n, K1 = C.shape; ar = torch.arange(n, device=dev)
        sc = St[ar[:, None], C]; v = sc[:, :1]; obs = (((sc[:, 1:] < v).sum(1) + 0.5 * (sc[:, 1:] == v).sum(1)).double() / (K1 - 1)).cpu().numpy()
        CQ = np.asarray(cand_q)[:, 1:]; CT = torch.as_tensor(qt[CQ], device=dev); CS = torch.as_tensor(qr[CQ] == run, device=dev)
        tr = torch.as_tensor(qt[rows], device=dev); sep = mt['min_sep_s']
        M = torch.as_tensor(maps[run], device=dev); nul, dropped_pairs, dropped_c = [], 0.0, 0.0
        for b0 in range(0, len(M), 100):
            J = M[b0:b0 + 100]; sj = St[J[..., None], C[None]]; vj = sj[..., :1]; tj = tr[J]               # [B, n]
            okc = ~(CS[None] & ((CT[None] - tj[..., None]).abs() < sep)); okp = (tj - tr[None]).abs() >= sep  # [B, n, K], [B, n]
            k = okc.sum(-1).double(); m = (((sj[..., 1:] < vj) & okc).sum(-1) + 0.5 * ((sj[..., 1:] == vj) & okc).sum(-1)).double() / k.clamp(min=1)
            use = okp & (k > 0); nul.append(((m * use).sum(1) / use.sum(1).clamp(min=1)).cpu().numpy())
            dropped_pairs += float((~use).double().sum()); dropped_c += float(((~okc) & okp[..., None]).double().sum())
        nl = np.concatenate(nul); return dict(n=int(n), mrp=float(obs.mean()), null_mean=float(nl.mean()), null=nl, per_query=obs,
                                              null_dropped_pair_share=dropped_pairs / (len(M) * n), null_dropped_cand_share=dropped_c / (len(M) * n * (K1 - 1)))
    def pool(cells, key):
        rs = sorted(cells); n = np.array([cells[r]['n'] for r in rs], float); v = np.array([cells[r][key] for r in rs]); pos = {r: i for i, r in enumerate(rs)}
        pt = float((v * n).sum() / n.sum()); RBs = RB if len(rs) == 20 else rng.integers(0, len(rs), (g['bootstrap']['n'], len(rs)))
        bs = np.array([(v[b] * n[b]).sum() / n[b].sum() for b in RBs]); return pt, bs
    def summarise(cells):
        for r in cells: cells[r]['excess'] = cells[r]['mrp'] - cells[r]['null_mean']
        mrp, bm = pool(cells, 'mrp'); ex, be = pool(cells, 'excess'); rs = sorted(cells)
        nl = np.array([cells[r]['null'] for r in rs]); n = np.array([cells[r]['n'] for r in rs], float); gnull = (nl * n[:, None]).sum(0) / n.sum()
        return dict(mrp=mrp, mrp_ci90=[float(np.percentile(bm, 5)), float(np.percentile(bm, 95))], excess=ex, excess_ci90=[float(np.percentile(be, 5)), float(np.percentile(be, 95))],
                    null_p975=float(np.percentile(gnull, 97.5)),
                    null_dropped_pair_share=float(sum(cells[r]['null_dropped_pair_share'] * cells[r]['n'] for r in rs) / n.sum()),
                    null_dropped_cand_share=float(sum(cells[r]['null_dropped_cand_share'] * cells[r]['n'] for r in rs) / n.sum()), per_run={str(r + 1): dict(n=cells[r]['n'], mrp=cells[r]['mrp'], excess=cells[r]['excess']) for r in rs})
    def run_arrays(Farr, offs_, lens_, r, fm, fsd):
        return (torch.from_numpy(np.asarray(Farr[:, offs_[r]:offs_[r] + lens_[r]], np.float32)).to(dev) - torch.from_numpy(fm).to(dev)) / torch.from_numpy(fsd).to(dev)
    def windows(Frun, starts):
        idx = torch.as_tensor(starts, device=dev)[:, None] + torch.arange(T, device=dev)[None]; return Frun[:, idx].permute(1, 0, 2).reshape(len(starts), -1)
    lags = np.round(np.arange(g['report_only']['delay_search_lags_s'][0], g['report_only']['delay_search_lags_s'][1] + 1e-9, g['report_only']['delay_search_lags_s'][2]), 2)
    cells = {k: {} for k in ('main', 'delay_search', 'unmatched', 'env_t0', 'env_t_lag')}; mcells = {(m, dm): {} for m in g['meters'] for dm in ('tts', 'audio')}
    mcells[('LOWONLY_PLUS', 'unmatched_tts')] = {}
    Ypool = {'tts': np.zeros((N, 1024), np.float32), 'audio': np.zeros((N, 1024), np.float32)}
    for r in range(20):                                                      # pooled raw targets via cumulative sums
        for dm, Farr, o_, l_, tt in (('tts', TF, toffs, tlens, qtau), ('audio', F, offs, lens, qt)):
            ix = np.where(qr == r)[0]
            if not len(ix): continue
            cs = np.concatenate([np.zeros((1024, 1)), np.cumsum(np.asarray(Farr[:, o_[r]:o_[r] + l_[r]], np.float64), 1)], 1)
            st = np.round((tt[ix] + w['tmin']) * fs).astype(int) if dm == 'audio' else np.round((tt[ix] + w['tmin']) * fs).astype(int)
            Ypool[dm][ix] = ((cs[:, st + T] - cs[:, st]) / T).T
    for f in folds:
        test = [r - 1 for r in sp['run_folds'][f]]; train, fm, fsd = stats[f]
        cols = np.where(np.isin(qr, test))[0]; colpos = {q: i for i, q in enumerate(cols)}; rows = keep[np.isin(qr[keep], test)]; rowpos = {q: i for i, q in enumerate(rows)}
        Freal = {r: run_arrays(F, offs, lens, r, fm, fsd) for r in test}; Ftts = {r: run_arrays(TF, toffs, tlens, r, fm, fsd) for r in test}
        def score_matrix(lag):
            S = torch.zeros(len(rows), len(cols), device=dev)
            for ci in range(0, len(cols), 384):
                cc = cols[ci:ci + 384]
                C = torch.cat([windows(Ftts[r], np.clip(np.round((qtau[cc][qr[cc] == r] + lag + w['tmin']) * fs).astype(int), 0, tlens[r] - T)) for r in test if (qr[cc] == r).any()])
                order = np.concatenate([np.where(qr[cc] == r)[0] for r in test if (qr[cc] == r).any()]); Cn = C / (1e-8 + C.norm(dim=1, keepdim=True))
                for ri in range(0, len(rows), 384):
                    rr = rows[ri:ri + 384]
                    R_ = torch.cat([windows(Freal[r], np.round((qt[rr][qr[rr] == r] + w['tmin']) * fs).astype(int)) for r in test if (qr[rr] == r).any()])
                    rorder = np.concatenate([np.where(qr[rr] == r)[0] for r in test if (qr[rr] == r).any()])
                    blk = R_ @ Cn.T; S[ri + torch.as_tensor(rorder, device=dev)[:, None], ci + torch.as_tensor(order, device=dev)[None]] = blk
                    del R_, blk
                del C, Cn
            return S.cpu().numpy()
        S0 = score_matrix(0.0); SD = np.max([score_matrix(l) for l in lags], 0)
        if a.smoke: print('smoke: fold', f, 'S columns', S0.shape[1], 'rows present', S0.shape[0] > 0, 'finite', bool(np.isfinite(S0).all() and np.isfinite(SD).all()), f'({time.time() - t0:.0f}s)', flush=True)
        # envelope correlations (TTS domain, report only)
        za = (ENVA - ENVA.mean(1, keepdims=True)) / (ENVA.std(1, keepdims=True) + 1e-12)
        def env_tts(lag):
            out = []
            for q in cols:
                ew = env_window(WR[qr[q]]['env_t'], qtau[q] + lag + w['tmin'], rt['env_fs'], T); out.append((ew - ew.mean()) / (ew.std() + 1e-12))
            return np.array(out)
        E0 = za[rows] @ env_tts(0.0).T / T; EL = np.max([za[rows] @ env_tts(l).T / T for l in lags], 0)
        # meters: ridge on the fold's training-run queries, predictions for the test-run queries
        tr_i = np.where(np.isin(qr, train))[0]; grp = np.array([train.index(r) // 2 for r in qr[tr_i]]); P = {}
        for m in g['meters']:
            for dm in ('tts', 'audio'):
                Y = (Ypool[dm] - fm[:, 0]) / fsd[:, 0]; Pm, _ = fit_predict(X[m][tr_i], Y[tr_i], X[m][rows], grp); P[(m, dm)] = (unit(Pm) @ unit(Y[cols]).T)
        for r in test:
            rr = retained_by_run[r]
            if len(rr) < 2: continue
            ri = np.array([rowpos[q] for q in rr]); Cm = np.array([[colpos[j] for j in cand[q]] for q in rr])
            Cq = np.array([cand[q] for q in rr])
            cells['main'][r] = stats_from(S0[ri], rr, Cq, Cm, r); cells['delay_search'][r] = stats_from(SD[ri], rr, Cq, Cm, r)
            cells['env_t0'][r] = stats_from(E0[ri], rr, Cq, Cm, r); cells['env_t_lag'][r] = stats_from(EL[ri], rr, Cq, Cm, r)
            same = np.where(qr[cols] == r)[0]; Cu = np.array([np.r_[colpos[q], same[same != colpos[q]]] for q in rr]); Cuq = cols[Cu]
            cells['unmatched'][r] = stats_from(S0[ri], rr, Cuq, Cu, r)
            for (m, dm) in P: mcells[(m, dm)][r] = stats_from(P[(m, dm)][ri], rr, Cq, Cm, r)
            mcells[('LOWONLY_PLUS', 'unmatched_tts')][r] = stats_from(P[('LOWONLY_PLUS', 'tts')][ri], rr, Cuq, Cu, r)
        print('fold', f, f'({time.time() - t0:.0f}s)', flush=True)
        del Freal, Ftts; torch.cuda.empty_cache() if dev == 'cuda' else None
    if a.smoke:
        print('smoke done: run cells', {k: len(v) for k, v in cells.items()}, 'meter keys', len(mcells), 'X dims', {k: v.shape[1] for k, v in X.items()}, f'({time.time() - t0:.0f}s)'); return
    # ---- summaries and verdict
    res = dict(level=level, level_coverage=cover, n_queries_valid=N, n_queries_retained=int(len(keep)), coverage=coverage,
               coverage_per_run={str(r + 1): float(np.mean([q in cand for q in np.where(qr == r)[0]])) for r in range(20)})
    res['candidates_other_run_share'] = float(np.mean([qr[j] != qr[q] for q in keep for j in cand[q][1:]]))
    res['same_run_lists_full'] = int(sum(len(cand_same[q]) == mt['n_cand'] for q in keep))
    rel = lambda key: np.concatenate([np.abs(A[key][cand[q][1:]] - A[key][q]) / max(A[key][q], 1e-9) for q in keep])
    absd = lambda key: np.concatenate([np.abs(A[key][cand[q][1:]] - A[key][q]) for q in keep])
    tw_n = X['LOWONLY_PLUS'][:, 4]; tw_L = X['LOWONLY_PLUS'][:, 3]
    res['residuals'] = {k: dict(median=float(np.median(v)), p90=float(np.percentile(v, 90))) for k, v in
                        (('letters_rel', rel('nL')), ('syllables_abs', absd('nS')), ('tts_dur_rel', rel('DT')),
                         ('tts_window_words_abs', np.concatenate([np.abs(tw_n[cand[q][1:]] - tw_n[q]) for q in keep])),
                         ('tts_window_letters_rel', np.concatenate([np.abs(tw_L[cand[q][1:]] - tw_L[q]) / max(tw_L[q], 1) for q in keep])))}
    res['i_information'] = summarise(cells['main'])
    res['meters'] = {f'{m}/{dm}': summarise(mcells[(m, dm)]) for m in g['meters'] for dm in ('tts', 'audio')}
    res['report'] = dict(delay_search=summarise(cells['delay_search']), unmatched_i=summarise(cells['unmatched']),
                         unmatched_LOWONLY_PLUS_tts=summarise(mcells[('LOWONLY_PLUS', 'unmatched_tts')]), ENV_T_lag0=summarise(cells['env_t0']), ENV_T_lagmax=summarise(cells['env_t_lag']))
    ok_i = res['i_information']['mrp_ci90'][0] >= g['pass_raw_mrp_lower90']
    ok_ii = all(v['excess_ci90'][1] < g['pass_excess_upper90'] for v in res['meters'].values()); ok_iii = coverage >= g['pass_coverage']
    res['conditions'] = dict(i_information=bool(ok_i), ii_meters=bool(ok_ii), iii_coverage=bool(ok_iii)); res['verdict'] = 'pass' if ok_i and ok_ii and ok_iii else 'fail'
    res['seconds'] = time.time() - t0; out = os.path.join(ROOT, c['out']); os.makedirs(out, exist_ok=True)
    ms = dict(level=level, salt=mt['salt'], key_format='r{run 1-based}_w{word index within the run, 0-based}', sets=[dict(q=keys[q], cands=[keys[j] for j in cand[q][1:]], cands_same_run=[keys[j] for j in cand_same[q]]) for q in keep],
              note_same_run='cands_same_run: same-run eligible candidates at the chosen level tolerances, same sha256 order, 19 or empty (D-109 sections 7 and 13)')
    json.dump(ms, open(os.path.join(out, 'matched_sets.json'), 'w'), indent=0)
    attrs = dict(key_format=ms['key_format'], fields=['run', 't', 'tau', 'n_w', 'n_L', 'n_S', 'D_T', 'seq'],
                 note='all valid queries (TTS window inside the TTS run); stimulus attributes only, no EEG (D-109 supplement 1)',
                 queries={keys[q]: [int(qr[q]) + 1, float(qt[q]), float(qtau[q]), int(A['nw'][q]), int(A['nL'][q]), int(A['nS'][q]), float(A['DT'][q]), str(A['seq'][q])] for q in range(N)})
    json.dump(attrs, open(os.path.join(out, 'match_attrs.json'), 'w'), indent=0)
    res['match_attrs_sha256'] = hashlib.sha256(open(os.path.join(out, 'match_attrs.json'), 'rb').read()).hexdigest()
    res['matched_sets_sha256'] = hashlib.sha256(open(os.path.join(out, 'matched_sets.json'), 'rb').read()).hexdigest()
    json.dump(res, open(os.path.join(out, 'gate0T2.json'), 'w'), indent=1, default=float)
    print(json.dumps(dict(verdict=res['verdict'], conditions=res['conditions'], level=level, coverage=coverage, mrp=res['i_information']['mrp'], ci90=res['i_information']['mrp_ci90'],
                          meters={k: (round(v['excess'], 4), [round(x, 4) for x in v['excess_ci90']]) for k, v in res['meters'].items()}), indent=1, default=float))


if __name__ == '__main__':
    main()
