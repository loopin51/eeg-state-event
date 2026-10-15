"""D-106 shared stimulus features of the ds004408 words (no EEG): WORD_AC, WIN_AC, FORM inputs, W2V_LOW, positions."""
import os, sys, re, json, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from d098_features import read_tier, SILENCE


def load_words():
    return np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True)


def b1_meta():
    b1 = yaml.safe_load(open(os.path.join(ROOT, 'configs/stageB_b1.yaml'))); d = os.path.join(ROOT, b1['cache_dir'])
    meta = json.load(open(os.path.join(d, 'eeg_meta.json'))); return b1, d, meta


def word_ac(W, span_s=0.8, bins=8):
    """mel_run (120 bands, 120 Hz) over [onset, onset + span) in `bins` bins -> [n, 120 * bins] and log duration; valid mask"""
    b1, d, meta = b1_meta(); fs = meta['sample_rate']; offs, lens = meta['run_offsets'], meta['run_lengths']
    M = np.load(os.path.join(d, 'mel_run.npy'), mmap_mode='r'); L = int(round(span_s * fs)); n = len(W['word'])
    X = np.zeros((n, M.shape[0] * bins), np.float32); ok = np.zeros(n, bool)
    for i in range(n):
        r = int(W['run'][i]); s = int(round(W['onset'][i] * fs))
        if s + L <= lens[r]:
            seg = np.asarray(M[:, offs[r] + s: offs[r] + s + L], np.float32); X[i] = seg.reshape(M.shape[0], bins, L // bins).mean(2).ravel(); ok[i] = True
    return np.c_[X, np.log(np.asarray(W['duration'], float) + 1e-3)], ok


def win_ac(W):
    """D-098 gate-0 window acoustic features (380-d) of the B2 query item with the same (run, onset); valid mask"""
    I = np.load(os.path.join(ROOT, 'data/cache/stageB_b2/items.npz'), allow_pickle=True)
    Fz = np.load(os.path.join(ROOT, 'data/cache/d098/features.npz'), allow_pickle=True); reg = list(Fz['reg']); names = list(Fz['reg_names']); fs = 20
    run, t, isq = I['run'], I['t'], I['is_query']; L = 60; s0 = np.round((t - 0.5) * fs).astype(int)
    ie = names.index('env_edge'); ib = [names.index(f'band_edge{b}') for b in range(8)]; iw, ip = names.index('word_onset'), names.index('phone_onset')
    def binned(x, nb): return x.reshape(nb, L // nb, -1).mean(1).T.ravel()
    key = {(int(r), round(float(tt), 4)): k for k, (r, tt, q) in enumerate(zip(run, t, isq)) if q}
    n = len(W['word']); X = np.zeros((n, 380)); ok = np.zeros(n, bool)
    for i in range(n):
        k = key.get((int(W['run'][i]), round(float(W['onset'][i]), 4)))
        if k is None or s0[k] < 0 or s0[k] + L > len(reg[run[k]]): continue
        R = reg[run[k]][s0[k]:s0[k] + L]
        ex = np.r_[binned(R[:, [ie]], 30), binned(R[:, ib], 10), binned(R[:, [iw]], 30) * (L // 30), binned(R[:, [ip]], 30) * (L // 30)]
        X[i] = np.r_[I['mel'][k], I['env'][k], I['f0'][k], I['vf'][k], ex]; ok[i] = True
    return X, ok


def phones(W):
    """TextGrid phones (stress digits removed) whose onset lies inside each word interval"""
    out = [[] for _ in range(len(W['word']))]
    for r in range(20):
        tier = [(a, b, re.sub(r'\d', '', x)) for a, b, x in read_tier(os.path.join(ROOT, f'data/raw/ds004408/stimuli/audio{r + 1:02d}.TextGrid'), 'phones') if x not in SILENCE]
        on = np.array([a for a, _, _ in tier])
        for i in np.where(W['run'] == r)[0]:
            j0, j1 = np.searchsorted(on, W['onset'][i] - 1e-6), np.searchsorted(on, W['offset'][i] - 1e-6); out[i] = [x for _, _, x in tier[j0:j1]]
    return out


def ngram_counts(ph_lists, train_mask, min_count=2):
    grams = []
    for p in ph_lists:
        s = ['#'] + p + ['#']; grams.append([' '.join(s[i:i + k]) for k in (1, 2, 3) for i in range(len(s) - k + 1)])
    from collections import Counter
    cnt = Counter(g for i in np.where(train_mask)[0] for g in grams[i]); vocab = {g: j for j, g in enumerate(sorted(g for g, c in cnt.items() if c >= min_count))}
    X = np.zeros((len(ph_lists), len(vocab)), np.float32)
    for i, gs in enumerate(grams):
        for g in gs:
            j = vocab.get(g)
            if j is not None: X[i, j] += 1
    return X


def w2v_low(W, cache):
    b1, d, meta = b1_meta(); fs = meta['sample_rate']; offs = meta['run_offsets']; F = np.load(os.path.join(cache, 'w2v_low.npy'), mmap_mode='r')
    X = np.zeros((len(W['word']), F.shape[0]), np.float32)
    for i in range(len(W['word'])):
        r = int(W['run'][i]); a = offs[r] + int(round(W['onset'][i] * fs)); b = max(a + 1, offs[r] + int(round(W['offset'][i] * fs)))
        X[i] = np.asarray(F[:, a:b], np.float32).mean(1)
    return X


def positions(W):
    b1, d, meta = b1_meta(); fs = meta['sample_rate']; lens = np.array(meta['run_lengths']) / fs
    run = np.asarray(W['run']); rel = np.asarray(W['onset']) / lens[run]; glob = (np.r_[0, np.cumsum(lens)][run] + np.asarray(W['onset'])) / lens.sum()
    return rel, glob, np.c_[rel, rel ** 2, rel ** 3, *[g(2 * np.pi * k * rel) for k in (1, 2, 4, 8) for g in (np.sin, np.cos)]]


def occurrence_index(keys):
    seen = {}; out = np.zeros(len(keys), int)
    for i, k in enumerate(keys): seen[k] = seen.get(k, 0) + 1; out[i] = seen[k]
    return out
