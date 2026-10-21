"""Fact-check reports/07_fiv_stage/notes/cocktail_party_audit_kr.md against the non-EEG files on the server.

Re-runs the audit's own reads (same code as the audit session): comprehension scores, content-word counts and rates,
run lengths, and the difflib order match of content words against four Gutenberg texts. No EEG file exists here.
Run on the server: env/bin/python -I scripts/fiv_verify_cocktail.py /mnt/data/eeg_datasets/broderick_cocktail
"""
import difflib, hashlib, os, re, sys

import numpy as np
import scipy.io as sio

root = sys.argv[1]
for dp, _, fn in sorted(os.walk(root)):
    for f in sorted(fn):
        p = os.path.join(dp, f)
        print('sha256', hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8], os.path.getsize(p), os.path.relpath(p, root))

base = os.path.join(root, 'cnsp_meta/LalorCocktailParty')
s = sio.loadmat(os.path.join(base, 'Behavioural Data/Comprehension Scores.mat'), squeeze_me=True)['score'].astype(float)
print('score shape', s.shape, 'nan share', np.isnan(s).mean())
for g, sl in (('sub1-17 (attend 20000)', slice(0, 17)), ('sub18-33 (attend Journey)', slice(17, 33))):
    print(g, 'attended %.4f unattended %.4f' % (np.nanmean(s[sl, :, 0]), np.nanmean(s[sl, :, 1])))

m = sio.loadmat(os.path.join(base, 'dataCND/dataStim.mat'), squeeze_me=True, struct_as_record=False)
for st_name in ('stim', 'stim2'):
    st = m[st_name]
    lens = [np.size(x) / st.fs for x in st.data[0]]
    print(st_name, 'fs', st.fs, 'runs', len(lens), 'run length min %.1f max %.1f total %.1f s = %.2f min' % (min(lens), max(lens), sum(lens), sum(lens) / 60))
run_total_min = sum(np.size(x) / m['stim'].fs for x in m['stim'].data[0]) / 60
books = {n: re.findall(r"[a-z]+", open(os.path.join(root, f'gutenberg/pg{n}.txt'), encoding='utf-8', errors='ignore').read().lower())
         for n in (164, 2488, 18857, 3748)}
for nm, story in (('unprocessedWord2Vec', '20000'), ('unprocessedWord2Vec2', 'Journey')):
    d = m[nm].data
    nw = sum(len(d[r].wordVec) for r in range(30))
    last = sum(d[r].offset_time[-1] for r in range(30)) / 60
    print(story, 'content words', nw, 'per min: last-offset denominator %.2f (%.2f min), run-length denominator %.2f (%.2f min)' % (
        nw / last, last, nw / run_total_min, run_total_min))
    runs = [[(re.findall(r"[a-z]+", w.lower()) or [''])[0] for w in d[r].wordVec] for r in range(30)]
    allw = [w for ws in runs for w in ws]
    vocab = set(allw)
    bounds = np.cumsum([0] + [len(ws) for ws in runs])
    for n, toks in books.items():
        fb = [t for t in toks if t in vocab]
        sm = difflib.SequenceMatcher(None, fb, allw, autojunk=False)
        hit = np.zeros(len(allw), bool)
        for a, b, size in sm.get_matching_blocks():
            hit[b:b + size] = True
        per = [hit[bounds[r]:bounds[r + 1]].mean() for r in range(30)]
        print(story, 'pg%d' % n, 'matched %.4f' % hit.mean(), 'run min %.3f median %.3f' % (min(per), np.median(per)), flush=True)
