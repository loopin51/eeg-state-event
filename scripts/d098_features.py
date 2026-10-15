"""D-098 stimulus features (no EEG): phoneme-level lexical variables and the 20-Hz regressors of gates J and 0.

Phones: the phones tier of the ds004408 TextGrids (Prosodylab-Aligner), stress digits removed; silence labels ('', sil, sp)
skipped. A word of word_features.npz is matched to a TextGrid word interval of the same run whose onset is within 10 ms; its
phones are the phones whose onset lies inside that interval.
Cohort model (Brodbeck et al. 2018): lexicon = CMUdict pronunciations (variants split the word count equally), word count =
10 ** max(wordfreq zipf, floor); a story word's aligned pronunciation is added as a variant when missing. At the k-th phone of
a word, C_k = lexicon entries whose first k phones match; phoneme surprisal = -log2 F(C_k) / F(C_{k-1}) (C_0 = lexicon);
cohort entropy = entropy (bits) of the count distribution over C_k.
Regressors per run at 20 Hz (length = min(env20, bands20)): env (1), 8 log bands, env edge (half-wave rectified first
difference, 1), band edges (8), word onset impulse, phone onset impulse; env/bands/edges are z-scored per run.

usage: python scripts/d098_features.py configs/d098_gates.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import re, sys, json, time, bisect, hashlib, argparse, yaml, numpy as np
from wordfreq import zipf_frequency
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILENCE = {'', 'sil', 'sp', 'SIL', 'SP'}


def read_tier(path, name):
    """intervals (xmin, xmax, text) of the named IntervalTier of a Praat long-format TextGrid"""
    txt = open(path, encoding='utf-8', errors='replace').read()
    parts = re.split(r'\n\s*item \[\d+\]:', txt)
    for p in parts:
        if re.search(r'name = "%s"' % name, p):
            return [(float(a), float(b), c) for a, b, c in re.findall(r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "(.*)"', p)]
    raise KeyError(name)


def strip(ph): return re.sub(r'\d', '', ph).upper()


def load_cmudict(path):
    lex = {}
    for line in open(path, encoding='utf-8', errors='replace'):
        line = line.split('#')[0].strip()
        if not line: continue
        w, *ph = line.split(); w = re.sub(r'\(\d+\)$', '', w).lower()
        lex.setdefault(w, set()).add(tuple(strip(p) for p in ph))
    return lex


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); c = yaml.safe_load(open(a.config)); t0 = time.time()
    fs = c['fs']; lx = c['lexicon']; cpath = os.path.join(ROOT, lx['cmudict'])
    sha = hashlib.sha256(open(cpath, 'rb').read()).hexdigest(); assert sha == lx['cmudict_sha256'], sha
    W = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True)
    S = np.load(os.path.join(ROOT, 'data/cache/ds004408/stim.npz'), allow_pickle=True)
    # 1. phones of every story word
    wrun, won = W['run'], W['onset']; nW = len(wrun); wphones = [None] * nW; ph_run, ph_on, ph_word, ph_lab, all_ph = [], [], [], [], []
    unmatched = 0
    for r in range(20):
        tg = os.path.join(ROOT, c['textgrid_dir'], f'audio{r + 1:02d}.TextGrid')
        P = [(x0, x1, strip(t)) for x0, x1, t in read_tier(tg, 'phones') if t.strip() not in SILENCE]
        Wt = [(x0, x1, t) for x0, x1, t in read_tier(tg, 'words') if t.strip() not in SILENCE]
        all_ph.append(np.array([p[0] for p in P])); pst = np.array([p[0] for p in P]); wst = np.array([w[0] for w in Wt])
        for i in np.flatnonzero(wrun == r):
            j = int(np.argmin(np.abs(wst - won[i]))) if len(wst) else -1
            if j < 0 or abs(wst[j] - won[i]) > 0.010: unmatched += 1; continue
            k = np.flatnonzero((pst >= Wt[j][0] - 1e-6) & (pst < Wt[j][1] - 1e-6))
            wphones[i] = tuple(P[q][2] for q in k)
            for pos, q in enumerate(k): ph_run.append(r); ph_on.append(P[q][0]); ph_word.append(i); ph_lab.append(pos)
    print(f'{nW} words, {unmatched} unmatched, {len(ph_on)} phones in matched words ({time.time() - t0:.0f}s)', flush=True)
    # 2. lexicon and cohort variables
    lex = load_cmudict(cpath); added = 0; zf = lx['zipf_floor']
    story_words = [str(x).lower() for x in W['word']]
    for i, w in enumerate(story_words):
        if wphones[i] and wphones[i] not in lex.get(w, set()): lex.setdefault(w, set()).add(wphones[i]); added += 1
    entries = []
    for w, prons in lex.items():
        cnt = 10.0 ** max(zipf_frequency(w, 'en'), zf) / len(prons)
        entries += [(p, cnt) for p in prons]
    entries.sort(key=lambda e: e[0]); keys = [e[0] for e in entries]; cnt = np.array([e[1] for e in entries])
    cs = np.r_[0.0, np.cumsum(cnt)]; cl = np.r_[0.0, np.cumsum(cnt * np.log2(cnt))]
    def cohort(prefix):
        lo = bisect.bisect_left(keys, prefix); hi = bisect.bisect_left(keys, prefix + ('~',)); return lo, hi
    sur = np.zeros(len(ph_on)); ent = np.zeros(len(ph_on))
    for q, (i, pos) in enumerate(zip(ph_word, ph_lab)):
        pr = wphones[i]; lo0, hi0 = cohort(pr[:pos]); lo1, hi1 = cohort(pr[:pos + 1])
        F0, F1 = cs[hi0] - cs[lo0], cs[hi1] - cs[lo1]; assert F1 > 0
        sur[q] = -np.log2(F1 / F0); ent[q] = np.log2(F1) - (cl[hi1] - cl[lo1]) / F1
    print(f'lexicon {len(entries)} entries ({added} aligned variants added); phoneme surprisal mean {sur.mean():.3f}, '
          f'cohort entropy mean {ent.mean():.3f} ({time.time() - t0:.0f}s)', flush=True)
    # 3. 20-Hz regressors
    def zs(x): return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-12)
    reg = []; names = ['env'] + [f'band{b}' for b in range(8)] + ['env_edge'] + [f'band_edge{b}' for b in range(8)] + ['word_onset', 'phone_onset']
    ph_run, ph_on = np.array(ph_run), np.array(ph_on)
    for r in range(20):
        env = np.asarray(S['env20'][r], float); bands = np.asarray(S['bands20'][r], float); T = min(len(env), bands.shape[1]); env, bands = env[:T], bands[:, :T]
        ee = np.maximum(np.diff(env, prepend=env[0]), 0); be = np.maximum(np.diff(bands, axis=1, prepend=bands[:, :1]), 0)
        wo = np.zeros(T); p = np.round(won[wrun == r] * fs).astype(int); np.add.at(wo, p[p < T], 1.0)
        po = np.zeros(T); p = np.round(all_ph[r] * fs).astype(int); np.add.at(po, p[p < T], 1.0)
        reg.append(np.vstack([zs(env), zs(bands), zs(ee), zs(be), wo, po]).T.astype(np.float32))
    out = os.path.join(ROOT, c['features_cache']); os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, 'features.npz'), reg=np.array(reg, dtype=object), reg_names=np.array(names), ph_run=ph_run, ph_onset=ph_on,
             ph_word=np.array(ph_word), ph_pos=np.array(ph_lab), ph_surprisal=sur, cohort_entropy=ent,
             phone_onsets=np.array(all_ph, dtype=object), word_matched=np.array([x is not None for x in wphones]))
    stats = dict(n_words=int(nW), unmatched_words=int(unmatched), n_phones_in_words=int(len(ph_on)), n_phones_all=int(sum(len(x) for x in all_ph)),
                 lexicon_entries=len(entries), aligned_variants_added=int(added), cmudict_sha256=sha,
                 ph_surprisal=dict(mean=float(sur.mean()), sd=float(sur.std())), cohort_entropy=dict(mean=float(ent.mean()), sd=float(ent.std())),
                 corr_surprisal_entropy=float(np.corrcoef(sur, ent)[0, 1]))
    o = os.path.join(ROOT, c['out']); os.makedirs(o, exist_ok=True); json.dump(stats, open(os.path.join(o, 'features_stats.json'), 'w'), indent=1)
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
