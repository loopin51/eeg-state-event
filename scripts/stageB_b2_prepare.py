"""Stage B B2 (D-091) stimulus side, no EEG: language targets, acoustic features, acoustic pair sets and their statistics.

Items are the B1 windows (0.5-s grid windows of every run for training, word-onset queries for evaluation), each a 3-s
window [t + tmin, t + tmax).
  language target: e5 length-residual embedding (D-075/D-076) of the words whose onset lies in the window, in order.
  acoustic features: mel (time mean of the stage A log-mel, 120), envelope (stimulus 64-Hz envelope, 30 bins of 0.1 s),
                     F0 (YIN: per bin the median log2 F0 of voiced frames and the voiced fraction). D-091 amendment 2:
                     the primary F0 block runs YIN on the audio resampled to 16 kHz and low-passed at 1 kHz (Butterworth,
                     zero phase); the registered original (16 kHz, no low-pass) is kept as the f0raw block.
                     A frame is unvoiced if no CMNDF dip below the threshold or its RMS is more than min_rms_db below the
                     run's loudest frame. Bins without voiced frames take the window's voiced mean (run mean if none).
  pairs: for each query of a fold pool (the test runs of run fold f; also the validation runs for smoke checks), the 19
         eligible queries nearest in the mean of the z-scored block distances. Pools (D-091 amendment 1): within (primary;
         same run, >= 30 s apart; rows of queries with < 19 eligible candidates are -1), cross (other test runs also
         eligible), mel_within (mel block only, within-run), within_f0raw (mel, env and the f0raw block, within-run).
Language target per run fold (D-091 amendment 3): the language vector standardised with the training-run grid windows (with
words) minus its ridge prediction from the 210 primary pairing features (mel, env, F0, voiced fraction; fitted on those
windows, penalty by grouped CV over 7 groups of 2 training runs), re-standardised with the same windows -> R_f{f}.
Statistics (stimulus only): pair vs random acoustic distance, language-target cosine true vs pair vs random, and per query
the Spearman correlation of language cosine and acoustic distance over its 19 pairs, for the original target and for the
fold's residual target; the share of target variance explained by the acoustic ridge (training windows, test queries).

usage: python scripts/stageB_b2_prepare.py configs/stageB_b2.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np, pandas as pd, torch, soundfile as sf
from math import gcd
from scipy.signal import resample_poly, butter, sosfiltfilt
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from d3_metric_gate import len_feats
from stageB_b1_meters import fit_predict


def items_of(c, lens, fs, W):
    w, e = c['window'], c['eeg']; lo = -w['tmin']
    def hi(r): return lens[r] / fs - w['tmax'] - max(e['offset_s'], 0)
    run, t, isq = [], [], []
    for r in range(20):
        g = np.arange(lo, hi(r) + 1e-9, w['train_step_s']); run += [r] * len(g); t += g.tolist(); isq += [False] * len(g)
        on = np.unique(W['onset'][W['run'] == r]); on = on[(on >= lo) & (on <= hi(r))]; run += [r] * len(on); t += on.tolist(); isq += [True] * len(on)
    return np.array(run), np.array(t), np.array(isq)


def yin(x, sr, p, dev):
    """frame-wise F0 (Hz, nan if unvoiced) at hop_s; frame centre times = i * hop + frame / 2"""
    fl, hop = int(round(p['frame_s'] * sr)), int(round(p['hop_s'] * sr)); tmin, tmax = int(sr / p['fmax']), int(sr / p['fmin'])
    X = torch.as_tensor(x, dtype=torch.float32, device=dev).unfold(0, fl, hop); Wi = fl - tmax
    d = torch.stack([((X[:, :Wi] - X[:, tau:tau + Wi]) ** 2).sum(1) for tau in range(tmax + 1)], 1)       # [frames, tmax + 1]
    cm = d[:, 1:] * torch.arange(1, tmax + 1, device=dev) / (torch.cumsum(d[:, 1:], 1) + 1e-12)              # CMNDF at tau = 1..tmax
    cm = cm[:, tmin - 1:]; below = cm < p['threshold']
    has = below.any(1); first = torch.where(has, below.float().argmax(1), torch.zeros_like(has, dtype=torch.long))
    idx = first.clone()                                                                       # follow the dip to its local minimum
    for _ in range(tmax):
        nxt = torch.clamp(idx + 1, max=cm.shape[1] - 1); go = has & (cm.gather(1, nxt[:, None])[:, 0] < cm.gather(1, idx[:, None])[:, 0]) & (nxt > idx)
        if not go.any(): break
        idx = torch.where(go, nxt, idx)
    tau = (idx + tmin).float(); rms = 10 * torch.log10((X ** 2).mean(1) + 1e-12); loud = rms > rms.max() - (-p['min_rms_db'])
    f0 = torch.where(has & loud, sr / tau, torch.full_like(tau, float('nan')))
    return f0.cpu().numpy(), (np.arange(len(f0)) * hop + fl / 2) / sr


def bins_mean(v, times, starts, nb, bs, fn):
    """per window start s and bin b: fn over values whose time is in [s + b bs, s + (b+1) bs)"""
    out = np.full((len(starts), nb), np.nan)
    lo = np.searchsorted(times, starts[:, None] + np.arange(nb) * bs); hi = np.searchsorted(times, starts[:, None] + (np.arange(nb) + 1) * bs)
    for i in range(len(starts)):
        for b in range(nb):
            seg = v[lo[i, b]:hi[i, b]]
            if len(seg): out[i, b] = fn(seg)
    return out


def pair_sets(Z_blocks, run, t, n_pairs, min_same, within):
    """Z_blocks: list of [n, d] z-scored blocks over the pool -> indices [n, n_pairs] of the nearest eligible items (-1 rows
    for queries with fewer than n_pairs eligible candidates)"""
    D = np.zeros((len(run), len(run)))
    for Z in Z_blocks:
        g = Z @ Z.T; sq = np.diag(g); D += (sq[:, None] + sq[None] - 2 * g) / Z.shape[1]
    D /= len(Z_blocks)
    far = np.abs(t[:, None] - t[None]) >= min_same; same = run[:, None] == run[None]
    ok = (same & far) if within else (~same | far); np.fill_diagonal(ok, False)
    Dm = np.where(ok, D, np.inf); P = np.argpartition(Dm, n_pairs, axis=1)[:, :n_pairs]
    P = np.take_along_axis(P, np.argsort(np.take_along_axis(Dm, P, 1), 1), 1); P[ok.sum(1) < n_pairs] = -1
    return P, D, ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); t0 = time.time()
    c2 = yaml.safe_load(open(a.config)); c = yaml.safe_load(open(os.path.join(ROOT, c2['b1_config']))); sp = c['split']; dev = 'cuda'
    d = os.path.join(ROOT, c['cache_dir']); meta = json.load(open(os.path.join(d, 'eeg_meta.json'))); fs = meta['sample_rate']
    offs, lens = meta['run_offsets'], meta['run_lengths']; w = c['window']; T = int(round((w['tmax'] - w['tmin']) * fs))
    W = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True)
    run, t, isq = items_of(c, lens, fs, W); n = len(run); start = t + w['tmin']
    # language targets
    lt = c2['language_target']; texts = []
    for r, s0 in zip(run, start):
        m = (W['run'] == r) & (W['onset'] >= s0) & (W['onset'] < s0 + (w['tmax'] - w['tmin'])); o = np.argsort(W['onset'][m])
        texts.append(' '.join(str(x) for x in W['word'][m][o]))
    has_text = np.array([len(x) > 0 for x in texts])
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(lt['embedder']['model'], device=dev); pre = lt['embedder']['prefix']
    def emb(tx): return model.encode([pre + x for x in tx], batch_size=lt['batch_size'], normalize_embeddings=True, convert_to_numpy=True)
    fit = list(pd.read_csv(os.path.join(ROOT, lt['resid_fit']), sep='\t', keep_default_na=False).text)
    Wlen = np.linalg.lstsq(len_feats(fit), emb(fit), rcond=None)[0]
    uniq = sorted(set(x for x in texts if x)); E = dict(zip(uniq, emb(uniq)))
    R = np.zeros((n, Wlen.shape[1]), np.float32)
    R[has_text] = np.stack([E[x] for x in np.array(texts)[has_text]]) - len_feats(list(np.array(texts)[has_text])) @ Wlen
    del model; torch.cuda.empty_cache(); print(f'{n} items, {len(uniq)} distinct texts, {int((~has_text).sum())} without words ({time.time() - t0:.0f}s)', flush=True)
    # acoustic features
    ac = c2['acoustic']; nb = int(round((w['tmax'] - w['tmin']) / ac['bin_s'])); bs = ac['bin_s']
    mel = np.load(os.path.join(d, 'mel_run.npy'), mmap_mode='r'); S = np.load(os.path.join(ROOT, 'data/cache/ds004408/stim.npz'), allow_pickle=True)
    MEL = np.zeros((n, mel.shape[0]), np.float32); ENV = np.zeros((n, nb)); F0s = {k: np.zeros((n, nb)) for k in ('f0', 'f0raw')}; VFs = {k: np.zeros((n, nb)) for k in ('f0', 'f0raw')}
    f0_stats = {'f0': {}, 'f0raw': {}}
    for r in range(20):
        ir = np.where(run == r)[0]
        for i in ir: j = offs[r] + int(round(start[i] * fs)); MEL[i] = np.asarray(mel[:, j:j + T], np.float32).mean(1)
        env = np.asarray(S['env64'][r], float); te = np.arange(len(env)) / 64.0
        ENV[ir] = bins_mean(env, te, start[ir], nb, bs, np.mean)
        x, sr = sf.read(os.path.join(ROOT, 'data/raw/ds004408/stimuli', f'audio{r + 1:02d}.wav')); x = x if x.ndim == 1 else x.mean(1)
        lp = ac['yin']['lowpass']; g = gcd(int(lp['resample_hz']), int(sr)); xr = resample_poly(x, int(lp['resample_hz']) // g, int(sr) // g)
        xl = sosfiltfilt(butter(lp['order'], lp['cutoff_hz'], fs=lp['resample_hz'], output='sos'), xr).copy()
        for key, xx in (('f0', xl), ('f0raw', xr)):
            f0, tf = yin(xx, lp['resample_hz'], ac['yin'], dev); voiced = ~np.isnan(f0); lf = np.log2(np.where(voiced, f0, 1.0)); F0, VF = F0s[key], VFs[key]
            VF[ir] = bins_mean(voiced.astype(float), tf, start[ir], nb, bs, np.mean)
            F0[ir] = bins_mean(np.where(voiced, lf, np.nan), tf, start[ir], nb, bs, lambda s: np.nan if np.isnan(s).all() else np.nanmedian(s))
            run_mean = float(lf[voiced].mean())
            for i in ir:
                row = F0[i]; fill = np.nanmean(row) if (~np.isnan(row)).any() else run_mean; F0[i] = np.where(np.isnan(row), fill, row)
            f0_stats[key][str(r + 1)] = dict(voiced_frac=float(voiced.mean()), median_hz=float(np.median(f0[voiced])))
    assert not np.isnan(ENV).any() and not any(np.isnan(v).any() for v in VFs.values())
    print(f'acoustic features ready ({time.time() - t0:.0f}s): voiced fraction', {k: round(np.mean([x["voiced_frac"] for x in v.values()]), 3) for k, v in f0_stats.items()}, flush=True)
    blocks = {'mel': MEL, 'env': ENV, 'f0': np.c_[F0s['f0'], VFs['f0']], 'f0raw': np.c_[F0s['f0raw'], VFs['f0raw']]}
    # residual language targets per run fold (amendment 3)
    out_c = os.path.join(ROOT, c2['cache']); os.makedirs(out_c, exist_ok=True); arrays, stats = {}, {'f0': f0_stats, 'folds': {}, 'residual': {}}
    feats = {'mel': MEL, 'env': ENV, 'f0': F0s['f0'], 'vf': VFs['f0']}; AC = np.concatenate([feats[k] for k in lt['acoustic_residual']['features']], 1)
    RF = {}
    for f in range(5):
        test = [r - 1 for r in sp['run_folds'][f]]; valid = [r - 1 for r in sp['valid_runs'][f]]; train = [r for r in range(20) if r not in test + valid]
        tr_i = np.where(~isq & has_text & np.isin(run, train))[0]; grp = np.array([train.index(r) // 2 for r in run[tr_i]])
        Y = (R - R[tr_i].mean(0)) / (R[tr_i].std(0) + 1e-6); P, al = fit_predict(AC[tr_i], Y[tr_i], AC, grp); E = Y - P
        te_q = np.where(isq & np.isin(run, test))[0]
        def r2(ix): Yc = Y[ix] - Y[ix].mean(0); return float(1 - ((E[ix] - E[ix].mean(0)) ** 2).sum() / (Yc ** 2).sum())
        RF[f] = ((E - E[tr_i].mean(0)) / (E[tr_i].std(0) + 1e-6)).astype(np.float32)
        stats['residual'][f'f{f}'] = dict(alpha=al, explained_train=r2(tr_i), explained_test_queries=r2(te_q))
        print(f'fold {f}: acoustic ridge alpha {al:g}, explained variance train {r2(tr_i):.4f} test queries {r2(te_q):.4f} ({time.time() - t0:.0f}s)', flush=True)
    # pair sets and statistics
    rng = np.random.default_rng(20261019); pc = c2['pairs']
    for f in range(5):
        for split, rs in (('test', [r - 1 for r in sp['run_folds'][f]]), ('valid', [r - 1 for r in sp['valid_runs'][f]])):
            pool = np.where(isq & np.isin(run, rs))[0]
            Zb = {k: (lambda X: (X - X.mean(0)) / (X.std(0) + 1e-8))(np.asarray(v[pool], float)) for k, v in blocks.items()}
            for kind, names, within in (('within', ac['blocks'], True), ('cross', ac['blocks'], False), ('mel_within', ac['sensitivity_blocks'], True),
                                        ('within_f0raw', [x if x != 'f0' else 'f0raw' for x in ac['blocks']], True)):
                P, D, ok = pair_sets([Zb[k] for k in names], run[pool], t[pool], pc['n_pairs'], pc['min_same_run_s'], within)
                arrays[f'pairs_{kind}_{split}_f{f}'] = np.where(P >= 0, pool[np.maximum(P, 0)], -1); arrays[f'pool_{split}_f{f}'] = pool
                if split == 'test':
                    v = P[:, 0] >= 0; Pv = P[v]; Dv, okv = D[v], ok[v]
                    rand = np.array([rng.choice(np.where(okv[i])[0], pc['n_pairs'], replace=False) for i in range(len(Pv))])
                    st = stats['folds'][f'{kind}_f{f}'] = dict(
                        n_queries=int(len(pool)), n_dropped=int((~v).sum()),
                        acoustic_dist_pairs=float(np.take_along_axis(Dv, Pv, 1).mean()), acoustic_dist_random=float(np.take_along_axis(Dv, rand, 1).mean()),
                        same_run_pair_frac=float((run[pool][Pv] == run[pool][v][:, None]).mean()))
                    for tag, RR in (('', R), ('resid_', RF[f])):
                        Rn = RR[pool] / (np.linalg.norm(RR[pool], axis=1, keepdims=True) + 1e-12); C = (Rn @ Rn.T)[v]
                        rho = np.array([spearmanr(C[i, Pv[i]], -Dv[i, Pv[i]])[0] for i in range(len(Pv))])
                        st.update({f'{tag}lang_cos_pairs': float(np.take_along_axis(C, Pv, 1).mean()), f'{tag}lang_cos_random': float(np.take_along_axis(C, rand, 1).mean()),
                                   f'{tag}spearman_lang_vs_acoustic_in_pairs': dict(mean=float(np.nanmean(rho)), q25=float(np.nanpercentile(rho, 25)), q75=float(np.nanpercentile(rho, 75)))})
            print(f'fold {f} {split}: {len(pool)} queries ({time.time() - t0:.0f}s)', flush=True)
    np.savez(os.path.join(out_c, 'items.npz'), run=run, t=t, is_query=isq, has_text=has_text, text=np.array(texts), R=R, mel=MEL, env=ENV,
             f0=F0s['f0'], vf=VFs['f0'], f0raw=F0s['f0raw'], vfraw=VFs['f0raw'], **{f'R_f{f}': RF[f] for f in range(5)})
    np.savez(os.path.join(out_c, 'pairs.npz'), **arrays)
    o = os.path.join(ROOT, c2['out']); os.makedirs(o, exist_ok=True); json.dump(stats, open(os.path.join(o, 'stimulus_stats.json'), 'w'), indent=1)
    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in stats['folds'].items()}), flush=True)
    print(json.dumps(stats['residual']), flush=True)


if __name__ == '__main__':
    main()
