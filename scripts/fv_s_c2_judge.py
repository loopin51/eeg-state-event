"""F-IX (D-119 amendment 1 F, I) judge of the C2 outputs (server CPU, gpuq, env_nlp: spaCy judge of D-117).

Per output: colour word = the first basic colour word (black blue brown grey/gray green orange pink purple red white yellow) attached to
the central object (Reader.colour; background colours are not counted);
central object = ese.things.lang.Vocab.central -> the THINGS concepts of that term (its Word or synonyms) -> union of their parent
categories. Per cell (subject, concept k): Q4 = (colour hit - permutation) + (category hit - permutation); colour hit = the word is the
first colour of the image (development concept rows: the share of the concept's 10 photos whose first colour is the word), category
hit = the categories meet the concept's set (concepts with a category only, else 0). Permutation = mean over the other concepts.
No colour word / no THINGS central object -> 0 for that part (its permutation is 0 too).

  --stage dev  : c2_dev.jsonl of the train run -> development Q4 per candidate step, the chosen step (largest; ties -> fewer steps),
                 development gate (the smallest candidate's excess >= dev_gate, review 2 item 2). -> results/fv_s/c2_dev.json
  --stage test : c2_test.jsonl of the test run -> results/fv_s/c2_cells.npz (q4 main [10, 200], controls, dev_gate_open) + c2_test.json
usage: env_nlp/bin/python scripts/fv_s_c2_judge.py configs/fv_s.yaml --stage {dev,test} --run DIR
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, re, json, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
from fv_s import Setup, fsha


class Reader:
    def __init__(self, st):
        import spacy
        from ese.things.lang import Vocab
        lc = st.lc; assert spacy.__version__ == lc['parser']['spacy']; self.V = Vocab(spacy.load('en_core_web_lg'), P(lc['names_table']), P(lc['things_meta']))
        self.st = st; self.cw = {w: i for i, w in enumerate(st.cols)}; self.cw['gray'] = st.cols.index('grey'); self.cache = {}

    def colour(self, s, term):
        """first basic colour word attached to the central object (review 2 item 6): an amod / compound modifier of the central noun
        (also through conj between modifiers), or an acomp of the sentence root whose subject is the central noun; a colour word inside
        the central term itself ('orange' as the object) is not a colour. -1 when none."""
        if term is None: return -1
        from ese.things.lang import norm_text
        doc = self.V.nlp(norm_text(s)); sents = list(doc.sents)
        if not sents: return -1
        last = term[-1]; heads = [t for t in sents[0] if t.pos_ in ('NOUN', 'PROPN') and (t.lemma_.lower() == last or t.lower_ == last)]
        if not heads: return -1
        h = heads[0]; cand = []
        todo = [ch for ch in h.children if ch.dep_ in ('amod', 'compound')]
        while todo:
            x = todo.pop(0); cand.append(x); todo += [ch for ch in x.children if ch.dep_ in ('conj', 'amod', 'compound')]
        if h.dep_ in ('nsubj', 'nsubjpass'): cand += [ch for ch in h.head.children if ch.dep_ in ('acomp', 'oprd')] + [g for ch in h.head.children if ch.dep_ == 'acomp' for g in ch.children if g.dep_ == 'conj']
        cand = sorted(cand, key=lambda t: t.i); inside = set(term)
        return next((self.cw[t.lower_] for t in cand if t.lower_ in self.cw and t.lower_ not in inside), -1)

    def __call__(self, s):
        if s not in self.cache:
            term = self.V.central(s)[0]; col = self.colour(s, term); cats = set()
            if term is not None:
                for i in self.V.name_of.get(term, set()) | {i for i, S in enumerate(self.V.S) if term in S}: cats |= set(self.st.catset[i])
            self.cache[s] = (col, cats)
        return self.cache[s]


def cells(rd, outs, colour_share, catsets, hascat):
    """outs [ns][nk] strings; colour_share [nk, 11] (share of first colour per concept); catsets [nk] sets -> q4, parts [ns, nk]"""
    ns, nk = len(outs), len(outs[0]); Q = np.zeros((ns, nk)); ch = np.zeros((ns, nk)); kh = np.zeros((ns, nk)); words = np.zeros((ns, nk), int) - 1
    cs_sum = colour_share.sum(0); hc = np.asarray(hascat, bool)
    for i in range(ns):
        for k in range(nk):
            col, cats = rd(outs[i][k]); words[i, k] = col; c_part = k_part = 0.0
            if col >= 0: ch[i, k] = colour_share[k, col]; c_part = ch[i, k] - (cs_sum[col] - colour_share[k, col]) / (nk - 1)
            if cats and hc[k]:
                hits = np.array([bool(cats & catsets[j]) for j in range(nk)]) & hc; kh[i, k] = hits[k]; k_part = hits[k] - (hits.sum() - hits[k]) / (hc.sum() - 1)
            Q[i, k] = c_part + k_part
    return Q, ch, kh, words


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', choices=['dev', 'test'], required=True); ap.add_argument('--run', required=True); a = ap.parse_args()
    c = yaml.safe_load(open(P(a.config))); st = Setup(c); rd = Reader(st); out = P(c['out_dir']); subs = st.subjects
    if a.stage == 'dev':
        rows = [json.loads(l) for l in open(os.path.join(P(a.run), 'c2_dev.jsonl'))]
        share = np.stack([np.bincount(st.top1[st.dev_rows][st.dev_concept == k], minlength=11) / 10 for k in range(len(st.dev_c))])
        cats = [set(st.catset[n]) for n in st.dev_name]; hc = st.has_cat[st.dev_name]; res = dict(stage='dev', run=os.path.basename(P(a.run)), c2_dev_sha256=fsha(os.path.join(P(a.run), 'c2_dev.jsonl')), steps={})
        for step in sorted({r['step'] for r in rows}):
            o = {(r['subject'], r['concept']): r['output'] for r in rows if r['step'] == step}; outs = [[o[(s, k)] for k in range(len(st.dev_c))] for s in subs]
            Q, ch, kh, w = cells(rd, outs, share, cats, hc)
            res['steps'][str(step)] = dict(excess=float(Q.mean()), colour_word_rate=float((w >= 0).mean()), n_colour_words=int(len(np.unique(w[w >= 0]))),
                                           n_distinct_outputs=int(len({x for r_ in outs for x in r_})), example=outs[0][:3])
        best = max(res['steps'], key=lambda s: (round(res['steps'][s]['excess'], 12), -int(s))); res['chosen_step'] = int(best)
        lo = min(v['excess'] for v in res['steps'].values())                     # review 2 item 2: the smallest candidate, not the chosen one
        res['dev_gate'] = dict(chosen_excess=res['steps'][best]['excess'], min_candidate_excess=lo, min=c['rules']['dev_gate'], open=bool(lo >= c['rules']['dev_gate']))
        json.dump(res, open(os.path.join(out, 'c2_dev.json'), 'w'), indent=1); print(json.dumps(res), flush=True); return
    dv = json.load(open(os.path.join(out, 'c2_dev.json'))); rows = [json.loads(l) for l in open(os.path.join(P(a.run), 'c2_test.jsonl'))]
    meta = json.load(open(os.path.join(P(a.run), 'metrics.json'))); assert meta['step'] == dv['chosen_step'], 'test run step differs from the development choice'
    k_ = np.arange(200); share = np.eye(11)[st.top1[st.test_rows]]; cats = [set(st.catset[n]) for n in st.test_name]; hc = st.has_cat[st.test_name]; arrs, rep = {}, {}
    for cn in sorted({r['cond'] for r in rows}):
        o = {(r['subject'], r['row']): r['output'] for r in rows if r['cond'] == cn}; outs = [[o.get((s, k), o.get((s, -1))) for k in k_] for s in subs]
        Q, ch, kh, w = cells(rd, outs, share, cats, hc); key = 'q4' if cn == 'dec+ses123+avg20' else 'q4_' + cn.replace('+', '_')
        arrs[key] = Q; rep[cn] = dict(excess=float(Q.mean()), colour_word_rate=float((w >= 0).mean()), colour_hit=float(ch[w >= 0].mean()) if (w >= 0).any() else None,
                                      category_hit=float(kh[:, hc].mean()), n_distinct_outputs=int(len({x for r_ in outs for x in r_})))
    np.savez(os.path.join(out, 'c2_cells.npz'), dev_gate_open=dv['dev_gate']['open'], step=dv['chosen_step'], **arrs)
    json.dump(dict(stage='test', run=os.path.basename(P(a.run)), c2_test_sha256=fsha(os.path.join(P(a.run), 'c2_test.jsonl')), conditions=rep,
                   cells_sha256=fsha(os.path.join(out, 'c2_cells.npz'))), open(os.path.join(out, 'c2_test.json'), 'w'), indent=1)
    print('c2 test cells written (values are read by fv_s.py --stage verdict):', sorted(arrs), flush=True)


if __name__ == '__main__':
    main()
