"""ds004408 positive-control gates (D-059).

P0-a  envelope tracking: backward ridge model, EEG (128 ch, lags 0-250 ms at 64 Hz) -> speech envelope. Pearson r on
      held-out runs; per-subject chance = 95th percentile over 100 circular shifts (>= 5 s) of the held-out envelope.
      Pass: subject-bootstrap lower bound of the mean r > 0 and >= 75 % of subjects above their own threshold.
P0-b  semantic dissimilarity (Broderick 2018): forward TRF (lags -100..700 ms) to 16 channels nearest Cz.
      Model A = envelope + word-onset impulses; B = A + impulses scaled by semantic dissimilarity
      (1 - cos(GloVe vector, mean of the previous 10 content-word vectors in the run)). Pass: bootstrap lower bound of
      the mean held-out r gain (B - A) > 0. Also reports the share of subjects whose dissimilarity TRF is negative at 300-500 ms.
Splits: 5 folds of 4 consecutive runs; ridge penalty chosen by 4-fold grouped CV inside the training runs.

usage: python scripts/ds004408_p0.py OUT_JSON [--subjects 1 ... 19]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import re, json, time, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CACHE = os.path.join(ROOT, 'data/cache/ds004408')
FS = 64; N_RUNS = 20; FOLDS = [list(range(i, i + 4)) for i in range(0, 20, 4)]; LAMS = 10.0 ** np.arange(-3, 5)


def z(x, axis=-1):
    return (x - x.mean(axis, keepdims=True)) / (x.std(axis, keepdims=True) + 1e-12)


def lagged(X, lags):
    """X [T, C] -> [T, C * len(lags)], column block j = X shifted so row t holds X[t + lag_j] (zero padded)."""
    T, C = X.shape; out = np.zeros((T, C * len(lags)), np.float32)
    for j, L in enumerate(lags):
        if L >= 0: out[:T - L, j * C:(j + 1) * C] = X[L:]
        else: out[-L:, j * C:(j + 1) * C] = X[:T + L]
    return out


def ridge(C, b, lam):
    d = np.mean(np.diag(C)); return np.linalg.solve(C + lam * d * np.eye(len(C)), b)


def corr(a, b):
    a = a - a.mean(0); b = b - b.mean(0); return (a * b).sum(0) / (np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + 1e-12)


def fit_eval(Xs, Ys, train, test):
    """Ridge with the penalty picked by grouped CV over the training runs; returns predictions for the test runs."""
    C = {r: Xs[r].T @ Xs[r] for r in train}; B = {r: Xs[r].T @ Ys[r] for r in train}
    groups = [train[i::4] for i in range(4)]; score = np.zeros(len(LAMS))
    for g in groups:
        tr = [r for r in train if r not in g]; Ct = sum(C[r] for r in tr); Bt = sum(B[r] for r in tr)
        for j, lam in enumerate(LAMS):
            W = ridge(Ct, Bt, lam); score[j] += np.mean([corr(Xs[r] @ W, Ys[r]).mean() for r in g])
    lam = LAMS[int(np.argmax(score))]; W = ridge(sum(C.values()), sum(B.values()), lam)
    return {r: Xs[r] @ W for r in test}, W, float(lam)


def boot(v, n=1000, seed=0):
    rng = np.random.default_rng(seed); b = [rng.choice(v, len(v)).mean() for _ in range(n)]; return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def dissimilarity(words):
    """per-word (onset, value, is_content); value = 1 - cos(word, mean of previous 10 content words) for content words."""
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    model = dissimilarity.model = getattr(dissimilarity, 'model', None) or SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d', device='cpu')
    toks = [re.sub(r"[^a-z']", '', w.lower()) for _, _, w in words]; vec = model.encode(toks, convert_to_numpy=True)
    out, ctx = [], []
    for (on, _, _), t, v in zip(words, toks, vec):
        content = bool(t) and t not in ENGLISH_STOP_WORDS and np.linalg.norm(v) > 0
        val = 0.0
        if content and ctx:
            m = np.mean(ctx[-10:], 0); val = 1 - float(v @ m / (np.linalg.norm(v) * np.linalg.norm(m) + 1e-12))
        if content: ctx.append(v)
        out.append((on, val, content))
    return out


def cz_channels(names, k=16):
    import mne
    bs = mne.channels.make_standard_montage('biosemi128').get_positions()['ch_pos']
    cz = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']['Cz']; cz = cz / np.linalg.norm(cz)
    ang = {n: np.arccos(np.clip(bs[n] @ cz / np.linalg.norm(bs[n]), -1, 1)) for n in names}
    near = sorted(ang, key=ang.get)[:k]; return [list(names).index(n) for n in near], near


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out_json'); ap.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 20)))
    a = ap.parse_args(); t0 = time.time(); S = np.load(os.path.join(CACHE, 'stim.npz'), allow_pickle=True)
    env = [z(np.asarray(e, np.float64)) for e in S['env64']]; words = [list(w) for w in S['words']]
    dis = [dissimilarity(w) for w in words]; rng = np.random.default_rng(0)
    res = dict(subjects={}, n_content=[int(sum(c for _, _, c in d)) for d in dis]); print(f'dissimilarity ready ({time.time() - t0:.0f}s)', flush=True)
    lag_b = list(range(0, int(0.25 * FS) + 1)); lag_f = list(range(int(-0.1 * FS), int(0.7 * FS) + 1))
    for sub in a.subjects:
        E = np.load(os.path.join(CACHE, f'sub-{sub:03d}.npz'), allow_pickle=True); eeg = [z(np.asarray(x, np.float64)).T for x in E['eeg64']]   # [T, 128]
        T = [min(len(eeg[r]), len(env[r])) for r in range(N_RUNS)]; eeg = [eeg[r][:T[r]] for r in range(N_RUNS)]; en = [env[r][:T[r]] for r in range(N_RUNS)]
        # P0-a
        Xb = [lagged(eeg[r], lag_b) for r in range(N_RUNS)]; Yb = [en[r][:, None] for r in range(N_RUNS)]; rs, null = [], np.zeros(100)
        for f in FOLDS:
            P, _, _ = fit_eval(Xb, Yb, [r for r in range(N_RUNS) if r not in f], f)
            for r in f:
                p = P[r][:, 0]; rs.append(corr(p, en[r])); n = len(p)
                for q in range(100): null[q] += corr(p, np.roll(en[r], int(rng.integers(5 * FS, n - 5 * FS)))) / N_RUNS
        r_env = float(np.mean(rs)); thr = float(np.quantile(null, 0.95))
        # P0-b
        idx, near = cz_channels(E['ch_names']); Yf = [eeg[r][:, idx] for r in range(N_RUNS)]
        def design(r, with_dis):
            on = np.zeros(T[r]); dv = np.zeros(T[r])
            for (t, v, c) in dis[r]:
                i = int(round(t * FS))
                if i < T[r]: on[i] = 1.0; dv[i] = v
            cols = [en[r], on] + ([dv] if with_dis else []); return lagged(np.stack(cols, 1), lag_f)
        gains, trf = [], []
        XA = [design(r, False) for r in range(N_RUNS)]; XB = [design(r, True) for r in range(N_RUNS)]; rA, rB = [], []
        for f in FOLDS:
            tr = [r for r in range(N_RUNS) if r not in f]
            PA, _, _ = fit_eval(XA, Yf, tr, f); PB, WB, _ = fit_eval(XB, Yf, tr, f)
            rA += [corr(PA[r], Yf[r]).mean() for r in f]; rB += [corr(PB[r], Yf[r]).mean() for r in f]
            w = WB.reshape(len(lag_f), 3, len(idx))[:, 2, :].mean(1); trf.append(w)
        trf = np.mean(trf, 0); win = [(L / FS >= 0.3) and (L / FS <= 0.5) for L in lag_f]
        res['subjects'][sub] = dict(r_env=r_env, r_env_null_p95=thr, rA=float(np.mean(rA)), rB=float(np.mean(rB)), gain=float(np.mean(rB) - np.mean(rA)),
                                    trf_dis_300_500=float(trf[win].mean()), trf_dis=trf.tolist(), cz_channels=near)
        s = res['subjects'][sub]
        print(f'sub-{sub:03d}: envelope r {r_env:.4f} (null p95 {thr:.4f}) | TRF r A {s["rA"]:.4f} B {s["rB"]:.4f} gain {s["gain"]:+.5f} '
              f'| dissimilarity TRF 300-500 ms {s["trf_dis_300_500"]:+.4f} ({time.time() - t0:.0f}s)', flush=True)
        json.dump(res, open(a.out_json, 'w'), indent=1)
    v = np.array([s['r_env'] for s in res['subjects'].values()]); above = np.mean([s['r_env'] > s['r_env_null_p95'] for s in res['subjects'].values()])
    g = np.array([s['gain'] for s in res['subjects'].values()]); neg = np.mean([s['trf_dis_300_500'] < 0 for s in res['subjects'].values()])
    res['P0a'] = dict(mean_r=float(v.mean()), ci95=boot(v), share_above_own_null=float(above), passed=bool(boot(v)[0] > 0 and above >= 0.75))
    res['P0b'] = dict(mean_gain=float(g.mean()), ci95=boot(g), share_negative_300_500=float(neg), passed=bool(boot(g)[0] > 0))
    json.dump(res, open(a.out_json, 'w'), indent=1); print('P0a', res['P0a'], '\nP0b', res['P0b'], flush=True)


if __name__ == '__main__':
    main()
