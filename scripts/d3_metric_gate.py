"""Direction 3, first gate: validation of the similarity metrics of the faithfulness device (D-075).

Three tests per metric, on pair lists built and hashed before any metric is computed (--stage pairs):
  T1 paraphrase vs length-matched unrelated: MRPC test paraphrases (a, b) against (a, c), c a sentence of another MRPC
     test pair with b's word count +- tol. AUC with a pair bootstrap.
  T2 same meaning vs same topic, different meaning: STS-B test pairs with gold >= 4 against gold 1-2. AUC with a pair
     bootstrap; Spearman with the gold score over the whole test set is reported.
  T3 length bias: for DailyDialog test utterances, the mean similarity to random utterances of other dialogues, Spearman
     with the utterance's word count. Item bootstrap.
Metrics: cosine of e5 / mpnet / bge sentence embeddings, raw and length-residualised (L2-normalised embedding minus its
linear prediction from [chars, words, log chars, log words], fitted on DailyDialog train utterances), and bidirectional NLI
entailment (min of P(entail) in both directions, cross-encoder/nli-deberta-v3-base).

usage: python scripts/d3_metric_gate.py configs/d3_metric_gate.yaml --stage pairs|score
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, time, hashlib, argparse, yaml, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_test(d):
    fs = [f for f in glob.glob(os.path.join(ROOT, d, '**', '*'), recursive=True)
          if 'test' in os.path.basename(f) and f.endswith(('.parquet', '.csv', '.csv.gz', '.tsv')) and '/.cache/' not in f]
    assert len(fs) == 1, fs; f = fs[0]
    return pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f, sep='\t' if f.endswith('.tsv') else ',')


def nwords(s): return len(str(s).split())


def dd_utterances(c, split):
    d = pd.read_parquet(os.path.join(ROOT, c['data']['dailydialog_dir'], split, '0000.parquet'))
    return [(i, u.strip()) for i, dl in enumerate(d['dialog']) for u in dl if u.strip()]


def build_pairs(c):
    out = os.path.join(ROOT, c['out_dir']); os.makedirs(out, exist_ok=True); rng = np.random.default_rng(c['seed']); p = c['pairs']
    m = find_test(c['data']['mrpc_dir']); pos = m[m.label == 1].reset_index(drop=True)
    pool = list(m.sentence2) + list(m.sentence1); pool_src = np.r_[m.index, m.index]; pool_len = np.array([nwords(x) for x in pool]); rows = []
    for k, r in pos.iterrows():
        L = nwords(r.sentence2); src = m.index[(m.sentence1 == r.sentence1) & (m.sentence2 == r.sentence2)][0]
        cand = np.where((np.abs(pool_len - L) <= p['t1_len_tol_words']) & (pool_src != src))[0]
        c_ = pool[cand[rng.integers(len(cand))]]
        rows += [dict(group=k, a=r.sentence1, b=r.sentence2, y=1), dict(group=k, a=r.sentence1, b=c_, y=0)]
    t1 = pd.DataFrame(rows)
    s = find_test(c['data']['stsb_dir']); s = s.rename(columns={'score': 'gold'}) if 'score' in s else s
    lo, hi = p['t2_neg_range']; t2 = s[(s.gold >= p['t2_pos_min']) | ((s.gold >= lo) & (s.gold <= hi))].copy()
    t2['y'] = (t2.gold >= p['t2_pos_min']).astype(int); t2 = t2.rename(columns={'sentence1': 'a', 'sentence2': 'b'})[['a', 'b', 'gold', 'y']]
    stsb_all = s.rename(columns={'sentence1': 'a', 'sentence2': 'b'})[['a', 'b', 'gold']]
    U = dd_utterances(c, 'test'); dlg = np.array([d for d, _ in U]); items = rng.choice(len(U), p['t3_n_items'], replace=False); rows = []
    for k, i in enumerate(items):
        others = np.where(dlg != dlg[i])[0]; refs = rng.choice(others, p['t3_n_refs'], replace=False)
        rows += [dict(item=k, a=U[i][1], b=U[j][1], len_a=nwords(U[i][1])) for j in refs]
    t3 = pd.DataFrame(rows)
    tr = dd_utterances(c, 'train'); fit = [tr[i][1] for i in rng.choice(len(tr), p['resid_fit_n'], replace=False)]
    files = {'t1.tsv': t1, 't2.tsv': t2, 'stsb_all.tsv': stsb_all, 't3.tsv': t3, 'resid_fit.tsv': pd.DataFrame({'text': fit})}
    hashes = {}
    for f, df in files.items():
        path = os.path.join(out, f); df.to_csv(path, sep='\t', index=False); hashes[f] = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    summ = dict(t1_groups=int(t1.group.nunique()), t2_pos=int(t2.y.sum()), t2_neg=int((1 - t2.y).sum()), stsb_all=len(stsb_all),
                t3_items=int(t3.item.nunique()), t3_pairs=len(t3), resid_fit=len(fit),
                t1_len_diff_pos=float(np.mean([abs(nwords(a) - nwords(b)) for a, b, y in zip(t1.a, t1.b, t1.y) if y])),
                t1_len_diff_neg=float(np.mean([abs(nwords(a) - nwords(b)) for a, b, y in zip(t1.a, t1.b, t1.y) if not y])), sha256=hashes)
    json.dump(summ, open(os.path.join(out, 'pairs_manifest.json'), 'w'), indent=1); print(json.dumps(summ, indent=1))


def len_feats(texts):
    ch = np.array([len(t) for t in texts], float); w = np.array([nwords(t) for t in texts], float)
    return np.c_[np.ones_like(ch), ch, w, np.log1p(ch), np.log1p(w)]


class Embedder:
    def __init__(self, spec, bs, fit_texts):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(spec['model'], device='cuda'); self.prefix = spec['prefix']; self.bs = bs; self.cache = {}
        E = self.raw(fit_texts); X = len_feats(fit_texts); self.W = np.linalg.lstsq(X, E, rcond=None)[0]

    def raw(self, texts):
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            E = self.m.encode([self.prefix + t for t in new], batch_size=self.bs, normalize_embeddings=True, convert_to_numpy=True)
            self.cache.update(zip(new, E))
        return np.stack([self.cache[t] for t in texts])

    def scores(self, a, b):
        Ea, Eb = self.raw(a), self.raw(b); Ra, Rb = Ea - len_feats(a) @ self.W, Eb - len_feats(b) @ self.W
        cos = lambda X, Y: np.sum(X * Y, 1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12)
        return {'raw': cos(Ea, Eb), 'resid': cos(Ra, Rb)}


def nli_scores(spec, bs, a, b):
    from sentence_transformers import CrossEncoder
    m = CrossEncoder(spec['model'], device='cuda'); lab = {v.lower(): int(k) for k, v in m.model.config.id2label.items()}['entailment']
    def ent(x, y):
        logit = m.predict(list(zip(x, y)), batch_size=bs, convert_to_numpy=True, show_progress_bar=False)
        e = np.exp(logit - logit.max(1, keepdims=True)); return (e / e.sum(1, keepdims=True))[:, lab]
    return np.minimum(ent(a, b), ent(b, a))


def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def judge(c, t1, t2, sts, t3, S):
    from scipy.stats import spearmanr
    g = c['gate']; rng = np.random.default_rng(c['seed'] + 1); B = g['n_boot']; res = {}
    y1, grp = t1.y.values, t1.group.values; ug = np.unique(grp); gi = {k: np.where(grp == k)[0] for k in ug}
    y2 = t2.y.values; it = t3.item.values; ui = np.unique(it); L = t3.groupby('item').len_a.first().reindex(ui).values
    for name, (s1, s2, s_all, s3) in S.items():
        a1 = auc(s1, y1); bs1 = [auc(s1[ix], y1[ix]) for ix in (np.concatenate([gi[k] for k in rng.choice(ug, len(ug))]) for _ in range(B))]
        a2 = auc(s2, y2); bs2 = []
        for _ in range(B):
            ix = rng.choice(len(y2), len(y2)); bs2.append(auc(s2[ix], y2[ix]))
        m3 = pd.Series(s3).groupby(it).mean().reindex(ui).values; r3 = spearmanr(m3, L)[0]
        bs3 = [spearmanr(m3[ix], L[ix])[0] for ix in (rng.choice(len(ui), len(ui)) for _ in range(B))]
        q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        r = dict(t1_auc=a1, t1_ci=q(bs1), t2_auc=a2, t2_ci=q(bs2), stsb_spearman=float(spearmanr(s_all, sts.gold)[0]), t3_rho=float(r3), t3_ci=q(bs3))
        r['pass'] = dict(t1=r['t1_ci'][0] >= g['t1_auc_lower'], t2=r['t2_ci'][0] >= g['t2_auc_lower'],
                         t3=-g['t3_abs_rho'] <= r['t3_ci'][0] and r['t3_ci'][1] <= g['t3_abs_rho'])
        r['pass']['all'] = all(r['pass'].values()); res[name] = r; print(name, json.dumps(r), flush=True)
    emb = {k: v for k, v in res.items() if k != 'nli' and not k.startswith('e5_raw') and v['pass']['all']}
    sel = dict(primary_embedding=max(emb, key=lambda k: emb[k]['t2_auc']) if emb else None, primary_nli='nli' if res['nli']['pass']['all'] else None)
    return res, sel


def score(c):
    out = os.path.join(ROOT, c['out_dir']); man = json.load(open(os.path.join(out, 'pairs_manifest.json')))
    for f, h in man['sha256'].items(): assert hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() == h, f'{f} changed'
    rd = lambda f: pd.read_csv(os.path.join(out, f), sep='\t', keep_default_na=False)
    t1, t2, sts, t3, fit = rd('t1.tsv'), rd('t2.tsv'), rd('stsb_all.tsv'), rd('t3.tsv'), list(rd('resid_fit.tsv').text)
    sets = [(list(t1.a), list(t1.b)), (list(t2.a), list(t2.b)), (list(sts.a), list(sts.b)), (list(t3.a), list(t3.b))]
    S = {}; t0 = time.time(); bs = c['metrics']['batch_size']
    for name, spec in c['metrics']['embedders'].items():
        E = Embedder(spec, bs, fit); sc = [E.scores(a, b) for a, b in sets]
        for kind in ('raw', 'resid'): S[f'{name}_{kind}'] = tuple(x[kind] for x in sc)
        print(f'{name} done ({time.time() - t0:.0f}s)', flush=True); del E
    S['nli'] = tuple(nli_scores(c['metrics']['nli'], bs, a, b) for a, b in sets); print(f'nli done ({time.time() - t0:.0f}s)', flush=True)
    res, sel = judge(c, t1, t2, sts, t3, S)
    rdir = os.path.join(ROOT, 'results/d3_metric_gate'); os.makedirs(rdir, exist_ok=True)
    json.dump(dict(config=c, pairs=man, metrics=res, selection=sel), open(os.path.join(rdir, 'metrics.json'), 'w'), indent=1)
    print('SELECTION', json.dumps(sel), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', choices=['pairs', 'score'], required=True); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); build_pairs(c) if a.stage == 'pairs' else score(c)
