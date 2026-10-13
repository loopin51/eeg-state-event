"""ds004408 P1: identification of the heard 3-s chunk among its 19 nearest chunks in the same run (D-059).

Chunks: consecutive non-overlapping 3.0 s windows of each audio segment; text = words whose onset falls in the window
(chunks with < 4 words dropped). EEG feature = 1-8 Hz EEG at 20 Hz, window delayed by 0.1 s (128 ch x 60 samples),
z-scored per run. Target TXT = intfloat/multilingual-e5-base embedding of the chunk text ("query: ", mean pooling, L2).
Acoustic features AUDF = stimulus envelope (60 samples) + mean of 8 log-band energies over the window.
Rows reported per subject: EEG->TXT (primary), PRE->TXT, ORDER->TXT (meters), AUDF->TXT, AUDF+EEG->TXT (semantic
increment), EEG->ENV (acoustic identification, for reference). Ridge via SVD, penalty by 4-fold grouped CV over training
runs (mean cosine); 5 folds of 4 consecutive runs. Chance: 100 shuffles of the calibration pairs, 95th percentile.

usage: python scripts/ds004408_identify.py OUT_JSON [--perms 100] [--subjects 1 ... 19]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import json, time, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CACHE = os.path.join(ROOT, 'data/cache/ds004408')
FS = 20; WIN = 3.0; DELAY = 0.1; PRE = 0.5; MIN_WORDS = 4; N_RUNS = 20; N_LOCAL = 20
FOLDS = [list(range(i, i + 4)) for i in range(0, 20, 4)]; ALPHAS = np.logspace(0, 7, 15)


def chunks(stim):
    """per run: list of (start_s, text) for eligible chunks."""
    out = []
    for r in range(N_RUNS):
        dur = float(stim['dur'][r]); words = list(stim['words'][r]); c = []
        k = 1   # first chunk starts at 3 s so that the 0.5 s pre-window exists
        while (k * WIN + WIN + DELAY) * FS < len(stim['env20'][r]) - 1 and k * WIN + WIN + DELAY < dur:
            t0 = k * WIN; w = [x[2] for x in words if t0 <= x[0] < t0 + WIN]
            if len(w) >= MIN_WORDS: c.append((t0, ' '.join(w).lower()))
            k += 1
        out.append(c)
    return out


def embed(texts):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base'); net = AutoModel.from_pretrained('intfloat/multilingual-e5-base').cuda().eval(); out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            b = tok([f'query: {t}' for t in texts[i:i + 256]], padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
            h = net(**b).last_hidden_state; m = b['attention_mask'].unsqueeze(-1).float(); e = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def svd(X): U, s, Vt = np.linalg.svd(X, full_matrices=False); return U, s, Vt
def ridge_fit(U, s, Vt, Y, a): return Vt.T @ ((s / (s ** 2 + a))[:, None] * (U.T @ Y))
def cos_rows(P, Y): return (P * Y).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12)


def pick_alpha(X, Y, run, train_runs):
    groups = [train_runs[i::4] for i in range(4)]; score = np.zeros(len(ALPHAS))
    for g in groups:
        tr, va = ~np.isin(run, g), np.isin(run, g); xm, ym = X[tr].mean(0), Y[tr].mean(0); U, s, Vt = svd(X[tr] - xm)
        for j, a in enumerate(ALPHAS): score[j] += cos_rows((X[va] - xm) @ ridge_fit(U, s, Vt, Y[tr] - ym, a), Y[va] - ym).mean()
    return float(ALPHAS[int(np.argmax(score))])


def local_candidates(run, t, idx):
    C = []
    for i in idx:
        same = idx[(run[idx] == run[i]) & (idx != i)]
        if len(same) >= N_LOCAL - 1: C.append(np.r_[i, same[np.argsort(np.abs(t[same] - t[i]))[:N_LOCAL - 1]]])
    return np.array(C)


def hits(P, Yn, C, pos):
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12); sim = np.einsum('md,mkd->mk', Pn[pos[C[:, 0]]], Yn[C])
    return (sim.argmax(1) == 0).astype(float)


def evaluate(X, Y, run, t, perms, rng, standardise_x):
    """5-fold run-grouped evaluation; returns per-chunk hits (in candidate order) and permutation top-1 values."""
    H, null, n_null = [], np.zeros(perms), np.zeros(perms)
    for f in FOLDS:
        tr_runs = [r for r in range(N_RUNS) if r not in f]; tr, te = np.where(~np.isin(run, f))[0], np.where(np.isin(run, f))[0]
        xm = X[tr].mean(0); xs = X[tr].std(0) + 1e-8 if standardise_x else 1.0; Xtr = (X[tr] - xm) / xs; Xte = (X[te] - xm) / xs
        ym = Y[tr].mean(0); Ytr = Y[tr] - ym; Yall = Y - ym; Yn = Yall / (np.linalg.norm(Yall, axis=1, keepdims=True) + 1e-12)
        a = pick_alpha(Xtr, Ytr, run[tr], tr_runs); U, s, Vt = svd(Xtr); pos = np.full(len(run), -1); pos[te] = np.arange(len(te))
        C = local_candidates(run, t, te); H.append(hits(Xte @ ridge_fit(U, s, Vt, Ytr, a), Yn, C, pos))
        for q in range(perms):
            h = hits(Xte @ ridge_fit(U, s, Vt, Ytr[rng.permutation(len(tr))], a), Yn, C, pos); null[q] += h.sum(); n_null[q] += len(h)
    return np.concatenate(H), null / np.maximum(n_null, 1)


def boot(v, n=1000, seed=0):
    rng = np.random.default_rng(seed); b = [rng.choice(v, len(v)).mean() for _ in range(n)]; return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out_json'); ap.add_argument('--perms', type=int, default=100)
    ap.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 20))); a = ap.parse_args(); t0 = time.time()
    S = np.load(os.path.join(CACHE, 'stim.npz'), allow_pickle=True); CH = chunks(S)
    run = np.concatenate([[r] * len(c) for r, c in enumerate(CH)]); tstart = np.concatenate([[x[0] for x in c] for c in CH]); texts = [x[1] for c in CH for x in c]
    TXT = embed(texts); n0 = int(round(DELAY * FS)); nw = int(WIN * FS); npre = int(PRE * FS)
    i0 = np.round(tstart * FS).astype(int)
    ENV = np.stack([np.asarray(S['env20'][r], np.float64)[i:i + nw] for r, i in zip(run, i0)])
    BAND = np.stack([np.asarray(S['bands20'][r], np.float64)[:, i:i + nw].mean(1) for r, i in zip(run, i0)])
    AUDF = np.c_[ENV, BAND]; ORDER = np.c_[tstart, tstart ** 2]
    res = dict(n_chunks=int(len(run)), n_per_run=[len(c) for c in CH], subjects={}); print(f'{len(run)} chunks, embedded ({time.time() - t0:.0f}s)', flush=True)
    rows = {}
    for sub in a.subjects:
        E = np.load(os.path.join(CACHE, f'sub-{sub:03d}.npz'), allow_pickle=True)
        eeg = [np.asarray(x, np.float64) for x in E['eeg20']]; eeg = [(x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-12) for x in eeg]
        EEG = np.stack([eeg[r][:, i + n0:i + n0 + nw].ravel() for r, i in zip(run, i0)]); PREF = np.stack([eeg[r][:, i - npre:i].ravel() for r, i in zip(run, i0)])
        rng = np.random.default_rng(sub); R = {}
        for name, X, Y, st in [('EEG->TXT', EEG, TXT, False), ('PRE->TXT', PREF, TXT, False), ('ORDER->TXT', ORDER, TXT, True),
                               ('AUDF->TXT', AUDF, TXT, True), ('AUDF+EEG->TXT', np.c_[(AUDF - AUDF.mean(0)) / (AUDF.std(0) + 1e-8), EEG], TXT, False),
                               ('EEG->ENV', EEG, (ENV - ENV.mean(0)) / (ENV.std(0) + 1e-8), False)]:
            h, null = evaluate(X, Y, run, tstart, a.perms if name in ('EEG->TXT', 'PRE->TXT', 'ORDER->TXT', 'EEG->ENV') else 0, rng, st)
            R[name] = dict(top1=float(h.mean()), n=int(len(h)), null_p95=float(np.quantile(null, 0.95)) if len(null) else None); rows[(sub, name)] = h
        R['increment'] = float((rows[(sub, 'AUDF+EEG->TXT')] - rows[(sub, 'AUDF->TXT')]).mean())
        res['subjects'][sub] = R; json.dump(res, open(a.out_json, 'w'), indent=1)
        print(f'sub-{sub:03d}: ' + ' | '.join(f'{k} {v["top1"]:.3f}' + (f' (p95 {v["null_p95"]:.3f})' if v['null_p95'] else '') for k, v in R.items() if isinstance(v, dict)) +
              f' | increment {R["increment"]:+.4f} ({time.time() - t0:.0f}s)', flush=True)
    subs = list(res['subjects'].values()); summ = {}
    for k in ['EEG->TXT', 'PRE->TXT', 'ORDER->TXT', 'AUDF->TXT', 'AUDF+EEG->TXT', 'EEG->ENV']:
        v = np.array([s[k]['top1'] for s in subs]); thr = [s[k]['null_p95'] for s in subs if s[k]['null_p95'] is not None]
        summ[k] = dict(mean=float(v.mean()), ci95=boot(v), chance_threshold=float(np.mean(thr)) if thr else None, per_subject=v.round(4).tolist())
    inc = np.array([s['increment'] for s in subs]); summ['increment'] = dict(mean=float(inc.mean()), ci95=boot(inc))
    p = summ['EEG->TXT']
    if p['ci95'][0] <= p['chance_threshold']: verdict = 'heard chunk not identified'
    elif any(summ[m]['mean'] > summ[m]['chance_threshold'] for m in ('PRE->TXT', 'ORDER->TXT')): verdict = 'withheld (leak meter above chance)'
    else: verdict = 'heard chunk identified; semantic increment ' + ('passed' if summ['increment']['ci95'][0] > 0 else 'not passed')
    res['summary'] = summ; res['verdict'] = verdict; json.dump(res, open(a.out_json, 'w'), indent=1)
    for k, v in summ.items(): print(k, v, flush=True)
    print('VERDICT (valid only if P0-a passed):', verdict, flush=True)


if __name__ == '__main__':
    main()
