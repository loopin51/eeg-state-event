"""F-VIII (D-117 amendments 1-5) vocabulary rules and the central-object judge for the language outputs.

Fixed before the sentence pool is generated (D-117 amendment 1: rules, generic list and parser version are committed first).

Vocabulary (1,854 THINGS concepts, configs/fv_names.tsv order = things_id order):
  name terms     N(c) = {THINGS 'Word' of c}                      (error vocabulary: names of the other concepts)
  synonym terms  S(c) = lemmas of c's own synset ('WordNet Synonyms' column of concepts-metadata_things.tsv, '_' -> ' ');
                 no hypernyms or hyponyms. Hit vocabulary of the true concept t: N(t) | S(t).
  terms are lower-cased and tokenised with the parser's tokenizer; a term matches tokens i..j when every term token equals the
  sentence token's lower-cased text or its lemma, and the last matched token is a NOUN or PROPN (nouns only). Spans are found
  greedily left to right, longest term first, without overlap (token units: cup does not match cupboard, ant does not match plant).
Span classification for true concept t (in this order):
  hit      span in N(t) | S(t)                (a homonym Word shared by several concepts is a hit when it is t's Word)
  generic  span in GENERIC                    (category words, never an error)
  error    span in N(c) for some c != t       (another THINGS concept's name)
  neutral  anything else                      (including synonyms of other concepts that are not their Word)
Central object of a sentence (first sentence of the text only):
  1. the root's nominal subject (nsubj / nsubjpass) or attribute (attr, 'There is a ...') when it is a NOUN/PROPN (a pronoun
     subject without a noun subject: none); else the root itself when it is a NOUN/PROPN; else the first noun chunk's root
     (else the first noun); else none.
  2. while that noun's lemma is a frame or quantity noun (FRAME) with a child 'of' that has a pobj NOUN/PROPN, move to the pobj
     ('A photo of a dog' -> dog, 'a pair of scissors' -> scissors).
  3. the central span is the matched term span that contains this token; without one, the right-most span inside the token's
     compound modifiers ('ice cream melting' when 'melting' is tagged as the head noun); else the token's lower-cased lemma.
Method B variant filter (truth-free, `single_object`): central object is a THINGS name or synonym, has no noun conjunct, and the
sentence has at most two THINGS-name spans; CLIP's 77-token limit is checked by the caller.
Row outcomes: strict M1 hit = central span is a hit; M2 error = central span is an error; otherwise no THINGS name (abstain).
'close-up' / 'close up' are normalised to 'closeup' before parsing.
"""
import csv, re, hashlib, json
from collections import defaultdict

GENERIC = ('object', 'objects', 'thing', 'things', 'item', 'items', 'stuff', 'something', 'tool', 'tools', 'animal', 'animals',
           'creature', 'container', 'device', 'equipment', 'gadget', 'artifact', 'food', 'piece')
FRAME = ('photo', 'photograph', 'picture', 'image', 'closeup', 'view', 'shot', 'snapshot', 'depiction', 'illustration',
         'pair', 'bunch', 'group', 'pile', 'set', 'stack', 'slice', 'couple', 'handful', 'row', 'collection', 'cluster', 'heap',
         'bundle', 'piece', 'kind', 'type', 'variety', 'assortment', 'array', 'selection', 'display', 'arrangement')
ABSTAIN = 'A photo of an object.'
NOUNS = ('NOUN', 'PROPN')


def norm_text(s):
    return re.sub(r'\bclose[- ]up\b', 'closeup', s, flags=re.I)


class Vocab:
    def __init__(self, nlp, names_tsv, meta_tsv):
        self.nlp = nlp
        rows = list(csv.DictReader(open(names_tsv, encoding='utf-8'), delimiter='\t'))
        meta = {r['uniqueID'].strip().replace(' ', '_'): r for r in csv.DictReader(open(meta_tsv, encoding='utf-8'), delimiter='\t')}
        self.concepts = [r['concept'] for r in rows]; self.split = [r['split'] for r in rows]; self.word = [r['display_name'] for r in rows]
        self.index = {c: i for i, c in enumerate(self.concepts)}
        tok = lambda s: tuple(t.text.lower() for t in nlp.tokenizer(s.lower().replace('_', ' ').strip()))
        self.N, self.S = [], []
        for r in rows:
            m = meta[r['concept']]; self.N.append({tok(m['Word'])})
            self.S.append({tok(x) for x in m['WordNet Synonyms'].split(',') if x.strip()})
        self.name_of = defaultdict(set)                       # term tokens -> concepts whose Word it is
        for i, n in enumerate(self.N):
            for t in n: self.name_of[t].add(i)
        self.generic = {tok(g) for g in GENERIC}
        terms = set(self.name_of) | set().union(*self.S)
        self.by_len = sorted(terms, key=lambda t: (-len(t), t)); self.first = defaultdict(list)
        for t in self.by_len: self.first[t[0]].append(t)
        self.sha = hashlib.sha256(json.dumps([self.concepts, [sorted(n) for n in self.N], [sorted(s) for s in self.S],
                                              GENERIC, FRAME], sort_keys=True).encode()).hexdigest()

    def spans(self, doc):
        """greedy longest-first non-overlapping term spans [(start, end, term tokens)] whose last token is a noun"""
        low = [t.text.lower() for t in doc]; lem = [t.lemma_.lower() for t in doc]; out, i = [], 0
        while i < len(doc):
            cands = set(self.first.get(low[i], [])) | set(self.first.get(lem[i], [])); hit = None
            for t in sorted(cands, key=lambda t: (-len(t), t)):
                j = i + len(t)
                if j <= len(doc) and all(t[k] in (low[i + k], lem[i + k]) for k in range(len(t))) and doc[j - 1].pos_ in NOUNS:
                    hit = (i, j, t); break
            if hit: out.append(hit); i = hit[1]
            else: i += 1
        return out

    def classify(self, term, t):
        """term tokens -> 'hit' | 'generic' | 'error' | 'neutral' for true concept index t (None: no true concept)"""
        if t is not None and (term in self.N[t] or term in self.S[t]): return 'hit'
        if term in self.generic: return 'generic'
        if self.name_of.get(term, set()) - ({t} if t is not None else set()): return 'error'
        return 'neutral'

    def central(self, text):
        """-> (central term tokens or None, all spans) for the first sentence of text"""
        doc = self.nlp(norm_text(text)); sents = list(doc.sents)
        self.last_conj = False
        if not sents: return None, []
        s = sents[0]; doc = s.as_doc(); root = doc[s.root.i - s.start]
        subj = [ch for ch in root.children if ch.dep_ in ('nsubj', 'nsubjpass', 'attr')]
        if any(ch.pos_ == 'PRON' for ch in subj) and not any(ch.pos_ in NOUNS for ch in subj): self.last_conj = False; return None, self.spans(doc)
        c = next((ch for ch in subj if ch.pos_ in NOUNS), None)
        if c is None and root.pos_ in NOUNS: c = root
        if c is None:
            ch = list(doc.noun_chunks); c = ch[0].root if ch and ch[0].root.pos_ in NOUNS else next((t for t in doc if t.pos_ in NOUNS), None)
        self.last_conj = False
        if c is None: return None, self.spans(doc)
        for _ in range(4):
            if c.lemma_.lower() not in FRAME and c.text.lower() not in FRAME: break
            of = next((ch for ch in c.children if ch.dep_ == 'prep' and ch.lower_ == 'of'), None)
            po = next((ch for ch in of.children if ch.dep_ == 'pobj' and ch.pos_ in NOUNS), None) if of is not None else None
            if po is None: break
            c = po
        sp = self.spans(doc); self.last_conj = any(ch.dep_ == 'conj' and ch.pos_ in NOUNS for ch in c.children)
        for a, b, term in sp:
            if a <= c.i < b: return term, sp
        comp, todo = set(), [c]                                   # 'ice cream melting' (head mis-tagged): the compound modifiers' span
        while todo:
            x = todo.pop(); kids = [ch for ch in x.children if ch.dep_ == 'compound']; comp |= {k.i for k in kids}; todo += kids
        inner = [(a, b, term) for a, b, term in sp if any(a <= k < b for k in comp)]
        if inner: return max(inner, key=lambda x: x[1])[2], sp
        return (c.lemma_.lower(),), sp

    def judge(self, text, t):
        """row outcome for true concept index t: dict(central, central_class, central_concepts, mentions)"""
        term, sp = self.central(text)
        cls = self.classify(term, t) if term is not None else 'none'
        ments = [(' '.join(x[2]), self.classify(x[2], t)) for x in sp]
        return dict(central=None if term is None else ' '.join(term), cls=cls, conj=self.last_conj,
                    concepts=sorted(self.name_of.get(term, set()) | ({t} if cls == 'hit' else set())) if term is not None else [],
                    mentions=ments, n_things=sum(k in ('hit', 'error') for _, k in ments))

    def single_object(self, text):
        """truth-free check of method B variants (D-117 amendment 1): a central object that is a THINGS name or synonym, no noun
        conjunct on it ('a cup and a plate'), and at most two THINGS-name spans in the sentence"""
        term, sp = self.central(text)
        if term is None or self.last_conj or term in self.generic: return False
        if not (term in self.name_of or any(term in s for s in self.S)): return False
        return sum(1 for _, _, t in sp if t in self.name_of and t not in self.generic) <= 2
