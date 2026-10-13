"""ds004408 word-level features for S0 and the nested TRF (direction 2, D-068). GPU for two small language models.

Per word (TextGrid "words" tier, run order 1..20 = story order):
  onset, offset, duration (s); n_letters; n_syllables (stressed vowels of the TextGrid "phones" tier inside the word);
  zipf frequency (wordfreq, English); content flag (not in sklearn ENGLISH_STOP_WORDS, not a contraction fragment such as
  "s"/"ll"/"ve" that the TextGrid splits off, not one of the auxiliaries in EXTRA_FUNCTION);
  punctuation restored with oliverguhr/fullstop-punctuation-multilang-large over the whole story (sentence end = . ? !);
  pause-based sentence ends (gap to the next word >= 0.30 s) as a sensitivity alternative;
  GPT-2 small (lower-cased words with restored punctuation, sliding window of 1024 tokens over the whole story):
    surprisal = -log2 P(word | previous text) summed over its sub-word tokens; entropy (bits) of the next-token
    distribution before the word's first token; layer-6 hidden state averaged over its tokens (768-d contextual vector);
  context-shuffle control (D-069): the restored sentences of each run are put in a random order (seed 0), GPT-2 is run
    on that text, and each word's layer-6 vector is mapped back to its original position (ctx_l6_shuf: same word,
    wrong preceding context);
  GloVe 6B 300-d static vector (vocabulary of sentence-transformers/average_word_embeddings_glove.6B.300d looked up
  directly, because that model's tokenizer drops stop words; zeros if the word is not in the vocabulary).
Writes data/cache/ds004408/word_features.npz.

usage: python scripts/ds004408_word_features.py
"""
import os, re, json, time, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data/raw/ds004408'); OUT = os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'); N_RUNS = 20
EXTRA_FUNCTION = {'s', 't', 'll', 've', 'd', 're', 'm', 'did', 'does', 'doing', 'just'}


def tiers(path):
    txt = open(path, encoding='utf-8').read(); out = {}
    for t in re.split(r'item \[\d+\]:', txt)[1:]:
        name = re.search(r'name = "([^"]+)"', t).group(1)
        out[name] = [(float(a), float(b), w.strip()) for a, b, w in re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', t)]
    return out


def restore_punct(words, dev):
    """punctuation label after each word ('0', '.', ',', '?', '-', ':'); windows of 120 words with stride 100, keeping
    only the middle 100 words of each window (10 words of context on both sides; full edges at the story ends)."""
    from transformers import pipeline
    nlp = pipeline('token-classification', model='oliverguhr/fullstop-punctuation-multilang-large', aggregation_strategy='none', device=0 if dev == 'cuda' else -1)
    n = len(words); lab = ['0'] * n; W, S, M = 120, 100, 10
    for s in range(0, n, S):
        chunk = words[s:s + W]; starts = np.cumsum([0] + [len(w) + 1 for w in chunk[:-1]])
        lo = 0 if s == 0 else M; hi = len(chunk) if s + W >= n else M + S
        for r in nlp(' '.join(chunk)):           # sub-word tokens in order: the last token of a word sets its label
            k = int(np.searchsorted(starts, r['start'], side='right') - 1)
            if lo <= k < hi: lab[s + k] = r['entity']
        if s + W >= n: break
    return lab


def gpt2_features(tokens_text, words, dev):
    import torch
    from transformers import GPT2TokenizerFast, GPT2LMHeadModel
    tok = GPT2TokenizerFast.from_pretrained('gpt2'); lm = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True).to(dev).eval()
    ids, owner = [tok.bos_token_id], [-1]
    for i, w in enumerate(tokens_text):
        t = tok.encode((' ' if i else '') + w); ids += t; owner += [i] * len(t)
    ids = np.array(ids); owner = np.array(owner); n = len(words)
    logp = np.full(len(ids), np.nan); ent = np.full(len(ids), np.nan); hid = np.zeros((len(ids), 768), np.float32)
    CTX, STEP = 1024, 512; s = 0
    with torch.no_grad():
        while True:
            e = min(s + CTX, len(ids)); x = torch.tensor(ids[s:e])[None].to(dev); o = lm(x)
            lp = torch.log_softmax(o.logits[0].float(), -1); p = lp.exp()
            first = 1 if s == 0 else CTX - STEP   # positions whose context is complete enough (first window: all)
            for j in range(first, e - s):
                pos = s + j; logp[pos] = lp[j - 1, ids[pos]].item() / np.log(2) if j > 0 else np.nan
                ent[pos] = -(p[j - 1] * lp[j - 1]).sum().item() / np.log(2) if j > 0 else np.nan
            hid[s + first:e] = o.hidden_states[6][0, first:].float().cpu().numpy()
            if e == len(ids): break
            s += STEP
    surpr = np.array([-np.nansum(logp[owner == i]) for i in range(n)])
    first_tok = np.array([np.flatnonzero(owner == i)[0] for i in range(n)])
    entropy = ent[first_tok]; ctx = np.stack([hid[owner == i].mean(0) for i in range(n)])
    return surpr, entropy, ctx


def main():
    import torch
    from wordfreq import zipf_frequency
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from sentence_transformers import SentenceTransformer
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; t0 = time.time(); rows = []
    for r in range(1, N_RUNS + 1):
        T = tiers(os.path.join(RAW, 'stimuli', f'audio{r:02d}.TextGrid'))
        phones = [(a, b, p) for a, b, p in T['phones'] if p and p.lower() not in ('sil', 'sp')]
        for a, b, w in T['words']:
            if not w or w.lower() in ('sil', 'sp', '<sil>'): continue
            syl = sum(1 for pa, pb, p in phones if pa >= a - 1e-6 and pb <= b + 1e-6 and p[-1:].isdigit())
            rows.append(dict(run=r - 1, onset=a, offset=b, word=w, clean=re.sub(r"[^a-z']", '', w.lower()), n_syl=max(syl, 1)))
    words = [x['clean'] or x['word'].lower() for x in rows]; n = len(words)
    print(f'{n} words ({time.time() - t0:.0f}s)', flush=True)
    punct = restore_punct(words, dev); print(f'punctuation restored: {sum(p in ".?!" for p in punct)} sentence ends ({time.time() - t0:.0f}s)', flush=True)
    tokens_text = [w + (p if p in '.,?!:' else '') for w, p in zip(words, punct)]
    surpr, entropy, ctx = gpt2_features(tokens_text, words, dev); print(f'GPT-2 done ({time.time() - t0:.0f}s)', flush=True)
    run_ = np.array([x['run'] for x in rows]); end = np.array([p in '.?!' for p in punct]); rng = np.random.default_rng(0); order = []
    for r in range(N_RUNS):
        idx = np.flatnonzero(run_ == r); cut = np.flatnonzero(end[idx]) + 1
        sents = [s_ for s_ in np.split(idx, cut) if len(s_)]; order += [i for k in rng.permutation(len(sents)) for i in sents[k]]
    order = np.array(order); _, _, ctx_s = gpt2_features([tokens_text[i] for i in order], words, dev)
    ctx_shuf = np.zeros_like(ctx); ctx_shuf[order] = ctx_s
    print(f'shuffled-context GPT-2 done, cos(ctx, ctx_shuf) mean {np.mean(np.sum(ctx * ctx_shuf, 1) / (np.linalg.norm(ctx, axis=1) * np.linalg.norm(ctx_shuf, axis=1))):.3f} ({time.time() - t0:.0f}s)', flush=True)
    we = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d', device='cpu')[0]
    vocab, E = we.tokenizer.word2idx, we.emb_layer.weight.detach().cpu().numpy()
    glove = np.stack([E[vocab[w]] if w in vocab else np.zeros(E.shape[1], np.float32) for w in words])
    run = np.array([x['run'] for x in rows]); onset = np.array([x['onset'] for x in rows]); offset = np.array([x['offset'] for x in rows])
    nxt_gap = np.r_[onset[1:] - offset[:-1], np.inf]; nxt_gap[np.r_[run[1:] != run[:-1], True]] = np.inf
    np.savez(OUT, run=run, onset=onset, offset=offset, duration=offset - onset, word=np.array(words), n_letters=np.array([len(w) for w in words]),
             n_syl=np.array([x['n_syl'] for x in rows]), zipf=np.array([zipf_frequency(w, 'en') for w in words]),
             content=np.array([bool(w) and w not in ENGLISH_STOP_WORDS and w not in EXTRA_FUNCTION for w in words]), punct=np.array(punct),
             sent_end_punct=np.array([p in '.?!' for p in punct]), sent_end_pause=nxt_gap >= 0.30,
             surprisal=surpr, entropy=entropy, ctx_l6=ctx.astype(np.float32), ctx_l6_shuf=ctx_shuf.astype(np.float32), glove=glove.astype(np.float32))
    print(f'saved {OUT}: surprisal mean {np.nanmean(surpr):.2f} bits, entropy mean {np.nanmean(entropy):.2f} bits, '
          f'pause sentence ends {int((nxt_gap >= 0.30).sum())}, unknown GloVe {int((np.linalg.norm(glove, axis=1) == 0).sum())} '
          f'(content words {int(sum(bool(w) and w not in ENGLISH_STOP_WORDS and w not in EXTRA_FUNCTION for w in words))}) ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
