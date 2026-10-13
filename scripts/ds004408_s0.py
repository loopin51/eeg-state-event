"""ds004408 S0: Broderick et al. (2018) semantic-dissimilarity TRF with sentence context (D-069, registered before results).

Dissimilarity of each content word = 1 - Pearson r(GloVe vector, mean vector of the preceding content words in the same
sentence); the first content word of a sentence uses the mean of the previous sentence's content words; the first content
word of a run gets 0; function words get 0. Sentence ends come from restored punctuation (primary) or pauses >= 0.30 s
(sensitivity). Controls: context shuffle (context = content words of a random other sentence of the same run; registered)
and value shuffle (values permuted among content words of each run; report only).
Per subject, a 64 Hz forward TRF (lags -100..800 ms, 128 channels) on [envelope, word onset, dissimilarity] is fitted on
all 20 runs with the ridge penalty picked by 5-fold run CV. Group test: sign-flip cluster permutation test (negative tail,
cluster-forming one-sided p 0.05, biosemi128 adjacency) on the dissimilarity TRF over 200-600 ms x 128 channels.
Pass: smallest-p negative cluster p < 0.05 with >= 50 % of its channels among the 32 channels nearest CPz, and the
context-shuffle control not passing the same rule.

usage: python scripts/ds004408_s0.py configs/ds004408_s0_trf.yaml [--subjects 2 3 4] [--n-perm 1000]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ds004408_p0 import lagged, ridge, corr, z
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CACHE = os.path.join(ROOT, 'data/cache/ds004408')
N_RUNS = 20; FOLDS = [list(range(i, i + 4)) for i in range(0, 20, 4)]


def load_words():
    return dict(np.load(os.path.join(CACHE, 'word_features.npz')))


def sentences(run, end):
    """per run: list of word-index arrays, one per sentence (split after words flagged as sentence ends)."""
    out = []
    for r in range(N_RUNS):
        idx = np.flatnonzero(run == r); cut = np.flatnonzero(end[idx]) + 1
        out.append([s for s in np.split(idx, cut) if len(s)])
    return out


def dissimilarity(F, end_key='sent_end_punct', mode='sentence', seed=0):
    """per-word dissimilarity values (0 for function words) and the content mask."""
    V = F['glove'].astype(np.float64); content = F['content'] & (np.linalg.norm(V, axis=1) > 0)
    val = np.zeros(len(V)); rng = np.random.default_rng(seed)
    dis = lambda v, m: 1.0 - float(np.corrcoef(v, m)[0, 1])
    for sents in sentences(F['run'], F[end_key]):
        cont = [s[content[s]] for s in sents]
        for k, c in enumerate(cont):
            if mode == 'context_shuffle':
                others = [j for j in range(len(cont)) if j != k and len(cont[j])]
                src = cont[others[int(rng.integers(len(others)))]] if others else None
            for p, i in enumerate(c):
                if mode == 'context_shuffle': ctx = src
                elif p > 0: ctx = c[:p]
                elif k > 0 and len(cont[k - 1]): ctx = cont[k - 1]
                else: ctx = None
                if ctx is not None: val[i] = dis(V[i], V[ctx].mean(0))
    if mode == 'value_shuffle':
        for r in range(N_RUNS):
            idx = np.flatnonzero((F['run'] == r) & content); val[idx] = val[rng.permutation(idx)]
    return val, content


def fit_subject(Xs, Ys, lams):
    C = [X.T @ X for X in Xs]; B = [X.T @ Y for X, Y in zip(Xs, Ys)]; score = np.zeros(len(lams))
    for f in FOLDS:
        tr = [r for r in range(N_RUNS) if r not in f]; Ct = sum(C[r] for r in tr); Bt = sum(B[r] for r in tr)
        for j, lam in enumerate(lams):
            W = ridge(Ct, Bt, lam); score[j] += np.mean([corr(Xs[r] @ W, Ys[r]).mean() for r in f])
    lam = float(lams[int(np.argmax(score))]); return ridge(sum(C), sum(B), lam), lam


def nearest(names, center, k):
    import mne
    bs = mne.channels.make_standard_montage('biosemi128').get_positions()['ch_pos']
    c = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos'][center]; c = c / np.linalg.norm(c)
    ang = np.array([np.arccos(np.clip(bs[n] @ c / np.linalg.norm(bs[n]), -1, 1)) for n in names]); return np.argsort(ang)[:k]


def cluster_test(trf, lag_ms, cfg, adj, roi, n_perm, seed=0):
    import mne
    from scipy.stats import t as tdist
    win = cfg['test_window_ms']; sel = (lag_ms >= win[0]) & (lag_ms <= win[1]); X = trf[:, sel, :]; n = len(X)
    thr = tdist.ppf(cfg['cluster']['threshold_p'], n - 1)
    _, clusters, pv, _ = mne.stats.permutation_cluster_1samp_test(X, threshold=thr, n_permutations=n_perm, tail=-1, adjacency=adj,
                                                                  seed=seed, out_type='mask', verbose=False)
    if not len(clusters): return dict(p=1.0, n_clusters=0, roi_share=0.0, passed=False)
    k = int(np.argmin(pv)); m = clusters[k]; ch = np.flatnonzero(m.any(0)); share = float(np.isin(ch, roi).mean()); tt = lag_ms[sel][m.any(1)]
    return dict(p=float(pv[k]), n_clusters=int(len(pv)), all_p=sorted(float(x) for x in pv)[:5], n_channels=int(len(ch)), roi_share=share,
                t_range_ms=[float(tt.min()), float(tt.max())], passed=bool(pv[k] < 0.05 and share >= cfg['cluster']['roi_share']))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--subjects', type=int, nargs='+'); ap.add_argument('--n-perm', type=int)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); c0 = cfg['s0']; t0 = time.time()
    subjects = a.subjects or cfg['subjects']; n_perm = a.n_perm or c0['cluster']['n_permutations']; fs = c0['fs']
    out_dir = os.path.join(ROOT, cfg['out_dir']); os.makedirs(out_dir, exist_ok=True)
    import mne
    F = load_words(); S = np.load(os.path.join(CACHE, 'stim.npz'), allow_pickle=True)
    assert sum(len(w) for w in S['words']) == len(F['run']), 'word lists of stim.npz and word_features.npz differ'
    variants = {'sentence_punct': ('sent_end_punct', 'sentence'), 'context_shuffle': ('sent_end_punct', 'context_shuffle'),
                'value_shuffle': ('sent_end_punct', 'value_shuffle'), 'sentence_pause': ('sent_end_pause', 'sentence')}
    dis = {k: dissimilarity(F, e, m, cfg['seed'])[0] for k, (e, m) in variants.items()}
    lags = list(range(int(round(c0['lags_ms'][0] / 1000 * fs)), int(round(c0['lags_ms'][1] / 1000 * fs)) + 1)); lag_ms = np.array(lags) / fs * 1000
    lams = 10.0 ** np.arange(cfg['lambdas_log10'][0], cfg['lambdas_log10'][1] + 1)
    env = [z(np.asarray(e, np.float64)) for e in S['env64']]
    trf = {k: [] for k in variants}; lam = {k: [] for k in variants}; names = None
    for sub in subjects:
        E = np.load(os.path.join(CACHE, f'sub-{sub:03d}.npz'), allow_pickle=True); names = [str(n) for n in E['ch_names']]
        T = [min(len(env[r]), E['eeg64'][r].shape[1]) for r in range(N_RUNS)]
        Y = [z(np.asarray(E['eeg64'][r][:, :T[r]], np.float64)).T for r in range(N_RUNS)]
        for k in variants:
            Xs = []
            for r in range(N_RUNS):
                w = np.flatnonzero(F['run'] == r); pos = np.round(F['onset'][w] * fs).astype(int); ok = pos < T[r]
                on = np.zeros(T[r]); dv = np.zeros(T[r]); np.add.at(on, pos[ok], 1.0); np.add.at(dv, pos[ok], dis[k][w][ok])
                Xs.append(lagged(np.stack([env[r][:T[r]], on, dv], 1), lags))
            W, l = fit_subject(Xs, Y, lams); trf[k].append(W.reshape(len(lags), 3, -1)[:, 2, :]); lam[k].append(l)
        print(f'sub-{sub:03d} done ({time.time() - t0:.0f}s)', flush=True)
    info = mne.create_info(names, fs, 'eeg'); info.set_montage(mne.channels.make_standard_montage('biosemi128'))
    adj, _ = mne.channels.find_ch_adjacency(info, 'eeg'); roi = nearest(names, c0['roi']['center'], c0['roi']['n_channels'])
    pool = [i for i, s in enumerate(subjects) if s not in cfg['pool_exclude']]
    res = dict(config=cfg['s0'], subjects=subjects, lag_ms=lag_ms.tolist(), roi=[names[i] for i in roi], lambdas=lam,
               n_content=int(F['content'].sum()), tests={})
    for k in variants:
        A = np.stack(trf[k]); res['tests'][k] = dict(all=cluster_test(A, lag_ms, c0, adj, roi, n_perm, cfg['seed']))
        if len(pool) < len(subjects): res['tests'][k]['pool17'] = cluster_test(A[pool], lag_ms, c0, adj, roi, n_perm, cfg['seed'])
        rm = A[:, :, roi].mean(2); res['tests'][k]['roi_trf_mean'] = rm.mean(0).tolist(); res['tests'][k]['roi_trf_se'] = (rm.std(0, ddof=1) / np.sqrt(len(rm))).tolist()
        sel = (lag_ms >= 300) & (lag_ms <= 500); res['tests'][k]['share_negative_300_500_roi'] = float(np.mean(rm[:, sel].mean(1) < 0))
        print(k, {kk: v for kk, v in res['tests'][k]['all'].items()}, flush=True)
    prim, ctrl = res['tests']['sentence_punct']['all'], res['tests']['context_shuffle']['all']
    res['verdict'] = ('S0 통과' if prim['passed'] and not ctrl['passed'] else 'S0 구분 불가(대조도 통과)' if prim['passed'] else 'S0 미통과')
    np.savez(os.path.join(out_dir, 's0_trf.npz'), lag_ms=lag_ms, subjects=np.array(subjects), **{k: np.stack(v) for k, v in trf.items()})
    json.dump(res, open(os.path.join(out_dir, 's0.json'), 'w'), indent=1, ensure_ascii=False)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for k, col in (('sentence_punct', '#B5652D'), ('context_shuffle', '#2E6E6B'), ('value_shuffle', '#5B4A86')):
        m, se = np.array(res['tests'][k]['roi_trf_mean']), np.array(res['tests'][k]['roi_trf_se'])
        ax.plot(lag_ms, m, color=col, label=f"{k} (p={res['tests'][k]['all']['p']:.3f})"); ax.fill_between(lag_ms, m - se, m + se, color=col, alpha=.2)
    ax.axvspan(*c0['test_window_ms'], color='#182129', alpha=.06); ax.axhline(0, color='#182129', lw=.5)
    ax.set_xlabel('lag (ms)'); ax.set_ylabel('TRF weight, CPz ROI (a.u.)'); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 's0_trf.png'), dpi=150)
    print('verdict:', res['verdict'], f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
