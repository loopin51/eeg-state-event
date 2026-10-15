"""D-106 G0-T: lemma, coarse POS and morphological family of every ds004408 word (env_tts, CPU).

spaCy en_core_web_sm 3.8.0 is run on each run's text written exactly as for the T1 rendering (restored punctuation,
contractions joined, scripts/t1_tts_render.run_text); spaCy tokens are aligned in order to the written tokens by normalised
strings and every TextGrid word gets the lemma / POS of the spaCy token that starts its written token (a contraction fragment
gets 'FRAG'). Family = Snowball English stem of the lower-cased lemma. Output: data/cache/d106/lemma.npz.

usage: env_tts/bin/python scripts/d106_lemma.py configs/d106_gates.yaml
"""
import os, sys, re, json, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from t1_tts_render import run_text, norm, FRAG


def written_tokens(words, puncts):
    """the written tokens of run_text (same rules), one per owner index"""
    toks = []
    for w, p in zip(words, puncts):
        if w in FRAG and toks: toks[-1] = toks[-1] + "'" + w
        else: toks.append(w)
        if p != '0': toks[-1] = toks[-1] + (' —' if p == '-' else p)
    return toks


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); c = yaml.safe_load(open(a.config))
    import spacy, snowballstemmer
    nlp = spacy.load('en_core_web_sm'); assert nlp.meta['version'] == '3.8.0', nlp.meta['version']; stem = snowballstemmer.stemmer('english')
    W = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True); n = len(W['word'])
    lemma = np.array([''] * n, dtype=object); pos = np.array([''] * n, dtype=object); unmatched = 0
    for r in range(20):
        idx = np.where(W['run'] == r)[0]; words = [str(x) for x in W['word'][idx]]; text, owner = run_text(words, [str(x) for x in W['punct'][idx]])
        written = written_tokens(words, [str(x) for x in W['punct'][idx]]); assert ' '.join(written) == text; doc = nlp(text); toks = [t for t in doc if norm(t.text)]; k = 0
        first = {}
        for wi, o in enumerate(owner):
            if o not in first: first[o] = wi
        for o, wtxt in enumerate(written):
            key = norm(wtxt)
            if not key: continue
            acc, j, head = '', k, None
            while j < len(toks) and len(acc) < len(key):
                if head is None: head = toks[j]
                acc += norm(toks[j].text); j += 1
            wi = first.get(o)
            if wi is None: continue
            if acc == key and head is not None:
                k = j; lemma[idx[wi]] = head.lemma_.lower(); pos[idx[wi]] = head.pos_
            else: unmatched += 1
        for wi, w in enumerate(words):
            if w in FRAG and wi > 0 and owner[wi] == owner[wi - 1]: pos[idx[wi]] = 'FRAG'
    fam = np.array([stem.stemWord(l) if l else '' for l in lemma], dtype=object)
    out = os.path.join(ROOT, c['cache']); os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, 'lemma.npz'), lemma=lemma.astype(str), pos=pos.astype(str), family=fam.astype(str), spacy_version=nlp.meta['version'])
    print(json.dumps(dict(words=n, unmatched_written=unmatched, no_lemma=int((lemma == '').sum()),
                          pos_counts={p: int((pos == p).sum()) for p in sorted(set(pos.tolist()))})), flush=True)


if __name__ == '__main__':
    main()
