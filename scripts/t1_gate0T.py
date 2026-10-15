"""D-104 T1 stimulus gate 0-T (no EEG; configs/t1_tts.yaml gate_0T).

(i) For every B1 test query (word onset t of a run, every run being a test run of its fold), the estimate is the heard-audio
w2v window [t - 0.5, t + 2.5) and the candidates are the natural-length TTS windows [tau - 0.5, tau + 2.5) of all valid
queries of the same run (tau = TTS onset of the same word). Score = clip_scores, within-run MRP (below + 0.5 ties)/(n - 1),
query-weighted over runs; 90% lower bound from a 1,000-draw test-run bootstrap; excess over the B1 segment-order circular
shift (k_min 30 s, truth -> shift partner j, candidates = the other queries). Features standardised with the run fold's
training-run statistics of the real w2v_run (as RunData).
(ii) ORDER (relative position) and LOWONLY (letters and word count of the words with real onsets inside the window, and their
squares): ridge to the time-pooled TTS window on the fold's training-run queries (grouped CV 7 x 2 runs), cosine ranking of
the pooled TTS candidates; excess over their own shift null; 90% upper bound from the run bootstrap.
Report only: onset differences, duration ratios, delay search (max over 11 lags), stretched TTS (piecewise-linear warp of the
TTS features onto the TextGrid onsets), the report voice.

usage: python scripts/t1_gate0T.py configs/t1_tts.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageB_b1_meters import fit_predict, unit
from d100_tau50 import run_maps


def mrp_stats(S, maps):
    """S [n, n] torch (rows estimates, columns candidates; diagonal = truth). Observed per-query MRP and the null mean per draw."""
    n = S.shape[0]; d = S.diagonal()[:, None]
    obs = ((S < d).sum(1) + 0.5 * ((S == d).sum(1) - 1)).double() / (n - 1)
    ar = torch.arange(n, device=S.device); M = torch.as_tensor(maps, device=S.device); nul = []
    for b0 in range(0, len(M), 100):
        J = M[b0:b0 + 100]; v = S[ar[None], J][..., None]                    # [B, n, 1] score of the shift partner
        below = (S[None] < v).sum(-1).double(); ties = (S[None] == v).sum(-1).double() - 1
        nul.append(((below + 0.5 * ties) / (n - 1)).mean(1).cpu().numpy())
    return obs.cpu().numpy(), np.concatenate(nul)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); t0 = time.time()
    c = yaml.safe_load(open(a.config)); g = c['gate_0T']; b1 = yaml.safe_load(open(os.path.join(ROOT, c['b1_config']))); sp = b1['split']
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = os.path.join(ROOT, b1['cache_dir']); meta = json.load(open(os.path.join(d, 'eeg_meta.json'))); fs = meta['sample_rate']; offs = meta['run_offsets']; lens = meta['run_lengths']
    w, e = b1['window'], b1['eeg']; T = int(round((w['tmax'] - w['tmin']) * fs)); lo = -w['tmin']
    def hi(r): return lens[r] / fs - w['tmax'] - max(e['offset_s'], 0)
    Wd = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True)
    F = np.load(os.path.join(d, b1['features']['w2v'] + '.npy'), mmap_mode='r')
    voices = [c['tts']['voice'], c['tts']['voice_report']]; tts = {}
    for v in voices:
        vd = os.path.join(ROOT, c['cache'], v)
        if not os.path.exists(os.path.join(vd, 'meta.json')): continue
        tm = json.load(open(os.path.join(vd, 'meta.json'))); js = [json.load(open(os.path.join(vd, f'run{r:02d}.json'))) for r in range(20)]
        tts[v] = dict(F=np.load(os.path.join(vd, 'w2v_run.npy'), mmap_mode='r'), offs=tm['run_offsets'], lens=tm['run_lengths'], js=js)
    assert voices[0] in tts, 'primary voice features missing'
    fold_of = {r - 1: f for f in range(5) for r in sp['run_folds'][f]}
    stats = {}
    for f in range(5):
        test = [r - 1 for r in sp['run_folds'][f]]; valid = [r - 1 for r in sp['valid_runs'][f]]; train = [r for r in range(20) if r not in test + valid]
        samp = np.concatenate([np.arange(offs[r], offs[r] + lens[r]) for r in train]); samp = np.sort(samp[:: max(1, len(samp) // 200000)])
        Fs = np.asarray(F[:, samp], np.float32); stats[f] = (train, Fs.mean(1, keepdims=True), Fs.std(1, keepdims=True) + 1e-6)
    # queries per run with word index of the onset word, real and TTS onsets of every word of the run
    Q = {}
    for r in range(20):
        m = np.where(Wd['run'] == r)[0]; on = Wd['onset'][m]; uq, first = np.unique(on, return_index=True)
        keep = (uq >= lo) & (uq <= hi(r)); Q[r] = dict(t=uq[keep], widx=first[keep], on=on, nlet=Wd['n_letters'][m])
    def tts_valid(v, r):
        tau = np.asarray(tts[v]['js'][r]['onsets_s'])[Q[r]['widx']]; i0 = np.round((tau + w['tmin']) * fs).astype(int)
        return (i0 >= 0) & (i0 + T <= tts[v]['lens'][r]), tau, i0
    res = dict(voice=voices[0], n_queries_b1=int(sum(len(Q[r]['t']) for r in range(20))), runs={}, meters={}, report={})
    # common valid set for the primary voice
    sel = {r: tts_valid(voices[0], r)[0] for r in range(20)}
    res['n_queries_valid'] = int(sum(s.sum() for s in sel.values())); res['n_dropped'] = res['n_queries_b1'] - res['n_queries_valid']
    def windows_real(r, idx):
        f = fold_of[r]; _, fm, fsd = stats[f]; t = Q[r]['t'][idx]; i = offs[r] + np.round((t + w['tmin']) * fs).astype(int)
        return torch.from_numpy(np.stack([(np.asarray(F[:, j:j + T], np.float32) - fm) / fsd for j in i])).to(dev)
    def windows_tts(v, r, idx, lag=0.0, stretch=False):
        f = fold_of[r]; _, fm, fsd = stats[f]; TF = tts[v]['F']; base = tts[v]['offs'][r]; L = tts[v]['lens'][r]
        if not stretch:
            tau = np.asarray(tts[v]['js'][r]['onsets_s'])[Q[r]['widx'][idx]]; i0 = np.clip(np.round((tau + lag + w['tmin']) * fs).astype(int), 0, L - T)
            return torch.from_numpy(np.stack([(np.asarray(TF[:, base + j:base + j + T], np.float32) - fm) / fsd for j in i0])).to(dev)
        on_real = Q[r]['on']; on_tts = np.asarray(tts[v]['js'][r]['onsets_s']); out = []
        for t in Q[r]['t'][idx]:
            tr_ = t + w['tmin'] + np.arange(T) / fs; tt = np.interp(tr_, on_real, on_tts); j = np.clip(np.round(tt * fs).astype(int), 0, L - 1)
            out.append((np.asarray(TF[:, base + j], np.float32) - fm) / fsd)
        return torch.from_numpy(np.stack(out)).to(dev)
    def clip(R, C): return torch.einsum('bct,oct,o->bo', R, C, 1 / (1e-8 + C.flatten(1).norm(dim=1)))
    sn = g['shift_null']; maps = run_maps({r: Q[r]['t'][sel[r]] for r in range(20)}, sn['n_shift'], sn['min_shift_s'], sn['seed'], 0)
    def gate_run(r, v, mode):
        idx = np.where(sel[r])[0]; R = windows_real(r, idx)
        if mode == 'main': S = clip(R, windows_tts(v, r, idx))
        elif mode == 'stretched': S = clip(R, windows_tts(v, r, idx, stretch=True))
        elif mode == 'delay_search':
            S = torch.stack([clip(R, windows_tts(v, r, idx, lag=l)) for l in np.round(np.arange(-0.25, 0.2501, 0.05), 2)]).amax(0)
        elif mode == 'audio_self': S = clip(R, R)
        o, nl = mrp_stats(S, maps[r]); return dict(n=int(len(idx)), mrp=float(o.mean()), null_mean=float(nl.mean()), null=nl)
    rng = np.random.default_rng(g['bootstrap']['seed']); RB = rng.integers(0, 20, (g['bootstrap']['n'], 20))
    def pool(cells, key):
        n = np.array([cells[r]['n'] for r in range(20)], float); v = np.array([cells[r][key] for r in range(20)])
        pt = float((v * n).sum() / n.sum()); bs = np.array([(v[b] * n[b]).sum() / n[b].sum() for b in RB]); return pt, bs
    def summarise(cells):
        for r in cells: cells[r]['excess'] = cells[r]['mrp'] - cells[r]['null_mean']
        mrp, bm = pool(cells, 'mrp'); ex, be = pool(cells, 'excess')
        nl = np.array([cells[r]['null'] for r in range(20)]); n = np.array([cells[r]['n'] for r in range(20)], float)
        gnull = (nl * n[:, None]).sum(0) / n.sum()
        return dict(mrp=mrp, mrp_ci90=[float(np.percentile(bm, 5)), float(np.percentile(bm, 95))], excess=ex, excess_ci90=[float(np.percentile(be, 5)), float(np.percentile(be, 95))],
                    null_p975=float(np.percentile(gnull, 97.5)), per_run={str(r + 1): dict(n=cells[r]['n'], mrp=cells[r]['mrp'], excess=cells[r]['excess']) for r in range(20)})
    for mode in ('main', 'audio_self', 'delay_search', 'stretched'):
        cells = {r: gate_run(r, voices[0], mode) for r in range(20)}; res['runs' if mode == 'main' else 'report'][mode] = summarise(cells)
        print(mode, round(res['runs' if mode == 'main' else 'report'][mode]['mrp'], 4), f'({time.time() - t0:.0f}s)', flush=True)
    res['runs'] = res['runs']['main']
    if voices[1] in tts:
        sel_main = sel; sel = {r: tts_valid(voices[0], r)[0] & tts_valid(voices[1], r)[0] for r in range(20)}
        maps = run_maps({r: Q[r]['t'][sel[r]] for r in range(20)}, sn['n_shift'], sn['min_shift_s'], sn['seed'], 0)
        res['report']['voice_report'] = dict(voice=voices[1], primary_same_subset=summarise({r: gate_run(r, voices[0], 'main') for r in range(20)}),
                                             report_voice=summarise({r: gate_run(r, voices[1], 'main') for r in range(20)}))
        sel = sel_main; maps = run_maps({r: Q[r]['t'][sel[r]] for r in range(20)}, sn['n_shift'], sn['min_shift_s'], sn['seed'], 0)
    # meters on the primary voice
    v = voices[0]; TF = tts[v]['F']
    allq = [(r, i) for r in range(20) for i in np.where(sel[r])[0]]
    def pooled_tts(r, i):
        tau = tts[v]['js'][r]['onsets_s'][Q[r]['widx'][i]]; j = tts[v]['offs'][r] + int(round((tau + w['tmin']) * fs)); return np.asarray(TF[:, j:j + T], np.float32).mean(1)
    Ytts = np.stack([pooled_tts(r, i) for r, i in allq]); qr = np.array([r for r, _ in allq])
    rel = np.array([Q[r]['t'][i] / (lens[r] / fs) for r, i in allq])
    POS = np.c_[rel, rel ** 2, rel ** 3, *[gg(2 * np.pi * k * rel) for k in (1, 2, 4, 8) for gg in (np.sin, np.cos)]]
    def low(r, i):
        t = Q[r]['t'][i]; m = (Q[r]['on'] >= t + w['tmin']) & (Q[r]['on'] < t + w['tmax']); return [Q[r]['nlet'][m].sum(), m.sum()]
    LOW = np.array([low(r, i) for r, i in allq], float); LOW = np.c_[LOW, LOW ** 2]
    for name, X in (('ORDER', POS), ('LOWONLY', LOW)):
        cells = {}
        for f in range(5):
            train, fm, fsd = stats[f]; test = [r - 1 for r in sp['run_folds'][f]]
            Y = (Ytts - fm[:, 0]) / fsd[:, 0]; tr_i = np.where(np.isin(qr, train))[0]; te_i = np.where(np.isin(qr, test))[0]
            grp = np.array([train.index(r) // 2 for r in qr[tr_i]]); P, al = fit_predict(X[tr_i], Y[tr_i], X[te_i], grp)
            for r in test:
                m = qr[te_i] == r; S = torch.from_numpy(unit(P[m]) @ unit(Y[te_i][m]).T).to(dev)
                o, nl = mrp_stats(S, maps[r]); cells[r] = dict(n=int(m.sum()), mrp=float(o.mean()), null_mean=float(nl.mean()), null=nl, alpha=al)
        res['meters'][name] = summarise(cells); print(name, round(res['meters'][name]['excess'], 4), f'({time.time() - t0:.0f}s)', flush=True)
    # onset differences and duration ratios
    diffs = []
    for r in range(20):
        on_tts = np.asarray(tts[v]['js'][r]['onsets_s'])
        for i in np.where(sel[r])[0]:
            t = Q[r]['t'][i]; k = Q[r]['widx'][i]; m = (Q[r]['on'] >= t + w['tmin']) & (Q[r]['on'] < t + w['tmax'])
            diffs.append((on_tts[m] - on_tts[k]) - (Q[r]['on'][m] - t))
    diffs = np.abs(np.concatenate(diffs))
    res['report']['onset_difference_abs_s'] = {str(p): float(np.percentile(diffs, p)) for p in (50, 90, 99)}
    res['report']['duration_ratio'] = {str(r + 1): tts[v]['js'][r]['duration_s'] / (lens[r] / fs) for r in range(20)}
    res['report']['filled_words'] = int(sum(tts[v]['js'][r]['filled_words'] for r in range(20)))
    gi = res['runs']; passed_i = gi['mrp_ci90'][0] >= g['pass_raw_mrp_lower90']
    passed_ii = all(res['meters'][k]['excess_ci90'][1] < g['meters']['pass_excess_upper90'] for k in ('ORDER', 'LOWONLY'))
    res['verdict'] = 'pass' if passed_i and passed_ii else 'fail'; res['conditions'] = dict(i_information=passed_i, ii_meters=passed_ii); res['seconds'] = time.time() - t0
    out = os.path.join(ROOT, c['out']); os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, 'gate0T.json'), 'w'), indent=1, default=float)
    print(json.dumps(dict(verdict=res['verdict'], mrp=gi['mrp'], ci90=gi['mrp_ci90'], excess=gi['excess'], meters={k: (round(x['excess'], 4), [round(y, 4) for y in x['excess_ci90']]) for k, x in res['meters'].items()},
                          report={k: (round(x['mrp'], 4) if isinstance(x, dict) and 'mrp' in x else x) for k, x in res['report'].items() if k != 'duration_ratio'}), indent=1, default=float))


if __name__ == '__main__':
    main()
