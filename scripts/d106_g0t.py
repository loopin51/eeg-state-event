"""D-106 section 1: G0-T lemma zero-shot semantic retrieval gate on the ds004408 stimulus (no EEG, alpha-free, pipeline check).

Tokens: content words (word_features content) with a coarse POS in {NOUN (incl. PROPN), VERB, ADJ, ADV} from d106_lemma.py,
whose lemma is in the GloVe 6B 300-d vocabulary, and whose meter inputs exist (word interval and window inside the run; the
number left out is reported). Families (Snowball stem of the lemma) go to 5 folds by sha256(family + salt) order. Per test
lemma: 19 candidates from the same fold's test lemmas with the same coarse POS (lemma majority POS), |dZipf| <= 0.5 and
|dletters| <= 2, nearest by sqrt((dZipf/0.5)^2 + (dletters/2)^2) (ties by sha256). Minimum: 100 lemmas in total and 10 per fold.
Meters (token inputs -> lemma GloVe, ridge, penalty by grouped CV over 5 groups of training families): WORD_AC, WIN_AC, AC_NL
(RFF of WORD_AC, 5 seeds, lemma MRP averaged over seeds), FORM (phone 1-3-grams), W2V_LOW (xlsr-53 layers 1-4), ORDER_POS.
Lemma prediction = mean of its token predictions; MRP among the 20 candidates by cosine ((below + 0.5 ties) / 19). Null: per
fold random permutation of the lemma predictions, 1,000 draws; per-lemma excess = MRP - its null mean; 90% interval from a
1,000-draw lemma bootstrap. Pass: every meter's excess 90% upper < 0.02.

usage: python scripts/d106_g0t.py configs/d106_gates.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, hashlib, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from d098_gate0 import fit_predict, rff
from stageB_b1_meters import unit
import d106_feats as DF


def sha(s): return hashlib.sha256(s.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); c = yaml.safe_load(open(a.config)); g = c['g0t']; t0 = time.time()
    cache = os.path.join(ROOT, c['cache']); W = DF.load_words(); Lz = np.load(os.path.join(cache, 'lemma.npz'))
    lemma, pos, fam = Lz['lemma'], np.where(Lz['pos'] == 'PROPN', 'NOUN', Lz['pos']), Lz['family']
    from sentence_transformers import SentenceTransformer
    from wordfreq import zipf_frequency
    we = SentenceTransformer(g['glove_model'], device='cpu')[0]; vocab, E = we.tokenizer.word2idx, we.emb_layer.weight.detach().cpu().numpy()
    XW, okw = DF.word_ac(W, g['word_ac']['span_s'], g['word_ac']['bins']); XA, oka = DF.win_ac(W)
    tok = np.asarray(W['content'], bool) & np.isin(pos, [p for p in g['pos_keep'] if p != 'PROPN']) & np.array([l in vocab for l in lemma])
    n_content = int(tok.sum()); tok &= okw & oka; idx = np.where(tok)[0]
    res = dict(tokens_content_pos_glove=n_content, tokens=int(len(idx)), tokens_dropped_window=n_content - int(len(idx)))
    lem_list = sorted(set(lemma[idx].tolist())); L = len(lem_list); li = {l: k for k, l in enumerate(lem_list)}; tl = np.array([li[l] for l in lemma[idx]])
    lpos = np.array([max(set(pos[idx][tl == k].tolist()), key=lambda p: (list(pos[idx][tl == k]).count(p), p)) for k in range(L)])
    lfam = np.array([fam[idx][tl == k][0] for k in range(L)]); lz = np.array([zipf_frequency(l, 'en') for l in lem_list]); llet = np.array([len(l) for l in lem_list])
    families = sorted(set(lfam.tolist()), key=lambda f: sha(f + g['fold_salt'])); ffold = {f: k % g['n_folds'] for k, f in enumerate(families)}
    lfold = np.array([ffold[f] for f in lfam]); Y = np.stack([E[vocab[l]] for l in lem_list])
    # candidate sets
    cand = {}
    for k in range(L):
        m = np.where((lfold == lfold[k]) & (lpos == lpos[k]) & (np.abs(lz - lz[k]) <= g['candidates']['zipf_tol']) & (np.abs(llet - llet[k]) <= g['candidates']['letters_tol']))[0]
        m = m[m != k]
        if len(m) < g['candidates']['n']: continue
        dist = np.sqrt(((lz[m] - lz[k]) / g['candidates']['zipf_tol']) ** 2 + ((llet[m] - llet[k]) / g['candidates']['letters_tol']) ** 2)
        order = sorted(range(len(m)), key=lambda j: (dist[j], sha(lem_list[m[j]] + '|' + lem_list[k]))); cand[k] = np.r_[k, m[order[:g['candidates']['n']]]]
    ev = np.array(sorted(cand)); per_fold = {f: int((lfold[ev] == f).sum()) for f in range(g['n_folds'])}
    res.update(lemmas=L, families=len(families), lemmas_evaluated=int(len(ev)), lemmas_per_fold=per_fold, pos_counts={p: int((lpos[ev] == p).sum()) for p in sorted(set(lpos[ev].tolist()))})
    print(json.dumps({k: v for k, v in res.items()}), flush=True)
    if len(ev) < g['min_lemmas_total'] or min(per_fold.values()) < g['min_lemmas_per_fold']:
        res['verdict'] = 'undecidable (too few lemmas)'
    else:
        ph = DF.phones(W); rel, glob, POS = DF.positions(W); occ = np.log(DF.occurrence_index(list(lemma)))
        first = (DF.occurrence_index(list(lemma)) == 1).astype(float)
        WL = DF.w2v_low(W, cache)
        base = dict(WORD_AC=XW, WIN_AC=XA, W2V_LOW=WL, ORDER_POS=np.c_[POS, glob, occ, first])
        rng_null = np.random.default_rng(g['null']['seed']); perms = {f: None for f in range(g['n_folds'])}
        ev_by_fold = {f: ev[lfold[ev] == f] for f in range(g['n_folds'])}
        for f in range(g['n_folds']): perms[f] = np.stack([rng_null.permutation(len(ev_by_fold[f])) for _ in range(g['null']['n'])])
        def lemma_mrp(P_lem):
            """P_lem [L, 300] predictions for the test lemmas (rows for ev); per-lemma observed MRP and null mean"""
            obs = {}; nul = {}
            for f, evf in ev_by_fold.items():
                C = np.stack([cand[k] for k in evf]); Yc = unit(Y[C])                     # [m, 20, 300]
                def mrp(Pm):
                    s = np.einsum('md,mcd->mc', unit(Pm), Yc); v = s[:, :1]
                    return ((s[:, 1:] < v).sum(1) + 0.5 * (s[:, 1:] == v).sum(1)) / g['candidates']['n']
                o = mrp(P_lem[evf]); nl = np.mean([mrp(P_lem[evf][pi]) for pi in perms[f]], 0)
                for k, x, y in zip(evf, o, nl): obs[k] = x; nul[k] = y
            return np.array([obs[k] for k in ev]), np.array([nul[k] for k in ev])
        def run_meter(X, seeds=None):
            P_tok = np.zeros((len(idx), Y.shape[1]))
            Xi = X[idx]
            outs = []
            for s in (seeds or [None]):
                for f in range(g['n_folds']):
                    te = lfold[tl] == f; tr = ~te
                    trf = sorted(set(lfam[tl[tr]].tolist()), key=lambda q: sha(q + g['fold_salt'])); gmap = {q: k % g['cv_groups'] for k, q in enumerate(trf)}
                    grp = np.array([gmap[q] for q in lfam[tl[tr]]]); Xtr, Xte = Xi[tr], Xi[te]
                    if s is not None:
                        xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8; Xtr, Xte = rff((Xtr - xm) / xs, (Xte - xm) / xs, g['rff']['n'], s)
                    P, _ = fit_predict(Xtr, Y[tl[tr]], Xte, grp); P_tok[te] = P
                P_lem = np.zeros((L, Y.shape[1]))
                for k in range(L): P_lem[k] = P_tok[tl == k].mean(0)
                outs.append(lemma_mrp(P_lem))
            o = np.mean([x[0] for x in outs], 0); nl = np.mean([x[1] for x in outs], 0); return o, nl
        rng_b = np.random.default_rng(g['bootstrap']['seed']); B = rng_b.integers(0, len(ev), (g['bootstrap']['n'], len(ev)))
        meters = {}
        for name, X, seeds in (('WORD_AC', base['WORD_AC'], None), ('WIN_AC', base['WIN_AC'], None), ('AC_NL', base['WORD_AC'], g['rff']['seeds']),
                               ('FORM', None, None), ('W2V_LOW', base['W2V_LOW'], None), ('ORDER_POS', base['ORDER_POS'], None)):
            if name == 'FORM':
                o_all, n_all = [], []
                P_tok = np.zeros((len(idx), Y.shape[1]))
                for f in range(g['n_folds']):
                    te = lfold[tl] == f; tr = ~te; mask = np.zeros(len(W['word']), bool); mask[idx[tr]] = True
                    XF = DF.ngram_counts(ph, mask)[idx]
                    trf = sorted(set(lfam[tl[tr]].tolist()), key=lambda q: sha(q + g['fold_salt'])); gmap = {q: k % g['cv_groups'] for k, q in enumerate(trf)}
                    P, _ = fit_predict(XF[tr], Y[tl[tr]], XF[te], np.array([gmap[q] for q in lfam[tl[tr]]])); P_tok[te] = P
                P_lem = np.stack([P_tok[tl == k].mean(0) for k in range(L)]); o, nl = lemma_mrp(P_lem)
            else: o, nl = run_meter(X, seeds)
            ex = o - nl; bs = np.array([ex[b].mean() for b in B])
            meters[name] = dict(mrp=float(o.mean()), null_mean=float(nl.mean()), excess=float(ex.mean()), excess_ci90=[float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
                                per_fold={str(f): float(ex[lfold[ev] == f].mean()) for f in range(g['n_folds'])})
            print(name, round(meters[name]['excess'], 4), [round(x, 4) for x in meters[name]['excess_ci90']], f'({time.time() - t0:.0f}s)', flush=True)
        res['meters'] = meters
        res['verdict'] = 'pass' if all(m['excess_ci90'][1] < g['pass_excess_upper90'] for m in meters.values()) else 'fail'
    res['seconds'] = time.time() - t0; out = os.path.join(ROOT, c['out']); os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, 'g0t.json'), 'w'), indent=1); print('verdict', res['verdict'], flush=True)


if __name__ == '__main__':
    main()
