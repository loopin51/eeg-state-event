"""D-104 T1: natural-length TTS rendering of the ds004408 run texts (Kokoro-82M v1.0, configs/t1_tts.yaml). Runs in env_tts on CPU.

Text per run: TextGrid words with the restored punctuation of word_features.npz. Two text normalisations (fixed before any
0-T number): split contractions of the TextGrid ("don t", "boy s") are written with an apostrophe ("don't", "boy's"), and
the restored punct code '-' (a dash pause) is written as an em dash. The pipeline's own chunking is used and the chunks are
concatenated without added silence. TTS word onsets: Kokoro tokens (start_ts) are aligned in order to the run's words by
normalised strings; a token covering several words gives its span in proportion to their letter counts; words left without
a time are linearly interpolated (count reported).
Output: data/cache/t1_tts/<voice>/run{r:02d}.wav (24 kHz, int16) and run{r:02d}.json (word onsets in TTS time, counts).

usage: env_tts/bin/python scripts/t1_tts_render.py configs/t1_tts.yaml [--voice af_heart] [--runs 0 1 ...]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import re, sys, json, time, hashlib, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAG = {'s', 'll', 'd', 've', 're', 't', 'm'}
SR = 24000


def norm(s): return re.sub(r'[^a-z0-9]', '', s.lower())


def run_text(words, puncts):
    """returns the TTS string and, for every TextGrid word, the index of the written token (several words may share one)"""
    toks, owner = [], []
    for w, p in zip(words, puncts):
        if w in FRAG and toks: toks[-1] = toks[-1] + "'" + w; owner.append(len(toks) - 1)
        else: toks.append(w); owner.append(len(toks) - 1)
        if p != '0': toks[-1] = toks[-1] + (' —' if p == '-' else p)
    return ' '.join(toks), owner


def align(words, owner, tts_tokens):
    """tts_tokens: list of (text, start, end). Word onsets (s) in TTS time; NaN where not matched (filled later)."""
    written = {}
    for i, o in enumerate(owner): written.setdefault(o, []).append(i)
    targets = [(o, norm(''.join(words[i] for i in idx)), idx) for o, idx in sorted(written.items())]
    ons = np.full(len(words), np.nan); k = 0; acc, acc_t0 = '', None; matched = 0
    for o, key, idx in targets:
        acc, t0, t1, j = '', None, None, k
        while j < len(tts_tokens) and len(acc) < len(key):
            txt, s, e = tts_tokens[j]; n = norm(txt)
            if n == '': j += 1; continue
            if s is not None and t0 is None: t0 = s
            if e is not None: t1 = e
            acc += n; j += 1
        if acc == key and t0 is not None:
            k = j; matched += 1; L = np.array([len(words[i]) for i in idx], float); cum = np.r_[0, np.cumsum(L)[:-1]] / L.sum()
            span = (t1 if t1 is not None else t0) - t0
            for i, c in zip(idx, cum): ons[i] = t0 + c * span
        # no match: leave NaN and do not advance (the next written token re-anchors the search)
    nan = np.isnan(ons); filled = int(nan.sum())
    if filled:
        x = np.arange(len(ons)); ons[nan] = np.interp(x[nan], x[~nan], ons[~nan])
    return ons, matched, filled


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--voice'); ap.add_argument('--runs', type=int, nargs='*'); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); t = c['tts']; voice = a.voice or t['voice']; t0 = time.time()
    import torch; torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    from huggingface_hub import snapshot_download
    d = snapshot_download(t['model_repo'], revision=t['model_revision'], allow_patterns=['config.json', t['model_file'], f'voices/{voice}.pt'])
    sha = hashlib.sha256(open(os.path.join(d, t['model_file']), 'rb').read()).hexdigest(); assert sha == t['model_sha256'], sha
    import kokoro, misaki; assert kokoro.__version__ == '0.9.4' and misaki.__version__ == '0.9.4', (kokoro.__version__, misaki.__version__)
    from kokoro import KModel, KPipeline
    model = KModel(repo_id=t['model_repo'], config=os.path.join(d, 'config.json'), model=os.path.join(d, t['model_file'])).eval()
    pipe = KPipeline(lang_code=t['lang_code'], repo_id=t['model_repo'], model=model); vpath = os.path.join(d, 'voices', f'{voice}.pt')
    W = np.load(os.path.join(ROOT, c['text']['source']), allow_pickle=True)
    out = os.path.join(ROOT, c['cache'], voice); os.makedirs(out, exist_ok=True)
    import soundfile as sf
    for r in (a.runs if a.runs is not None else range(20)):
        m = W['run'] == r; words = [str(x) for x in W['word'][m]]; text, owner = run_text(words, [str(x) for x in W['punct'][m]])
        audio, toks, off = [], [], 0.0
        with torch.no_grad():
            for res in pipe(text, voice=vpath, speed=t['speed'], split_pattern=t['split_pattern']):
                au = res.audio.numpy() if hasattr(res.audio, 'numpy') else np.asarray(res.audio)
                for tk in res.tokens: toks.append((tk.text, None if tk.start_ts is None else off + tk.start_ts, None if tk.end_ts is None else off + tk.end_ts))
                audio.append(au); off += len(au) / SR
        audio = np.concatenate(audio); ons, matched, filled = align(words, owner, toks)
        sf.write(os.path.join(out, f'run{r:02d}.wav'), audio, SR, subtype='PCM_16')
        json.dump(dict(run=r, voice=voice, text=text, n_words=len(words), n_written=int(max(owner) + 1), matched_written=matched, filled_words=filled,
                       duration_s=len(audio) / SR, onsets_s=[float(x) for x in ons], wav_sha256=hashlib.sha256(open(os.path.join(out, f'run{r:02d}.wav'), 'rb').read()).hexdigest()),
                  open(os.path.join(out, f'run{r:02d}.json'), 'w'))
        print(f'{voice} run {r}: {len(audio) / SR:.1f} s, written {max(owner) + 1}, matched {matched}, filled words {filled} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
