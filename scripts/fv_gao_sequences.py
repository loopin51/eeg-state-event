"""Gao et al. (2026): label-sequence reuse across runs, from events.tsv only (no EEG values).

The same trial order (and the same 30/40-trial blocks) recurs across subjects and sometimes across the two
sessions of one subject. Any EEG feature that tracks time in run or time in block can then predict labels
across those runs. This script reports, per task: distinct sequences and blocks, ses-01/ses-02 position
agreement, shared blocks between sessions, and for each subject how many other-subject runs share no block.
Run on the server: env/bin/python -I scripts/fv_gao_sequences.py data/raw/gao_imagery/bids > results/gao_imagery/label_sequences.txt
"""
import collections, csv, glob, itertools, os, sys

root = sys.argv[1]
S = {}
for f in sorted(glob.glob(os.path.join(root, 'sub-*/ses-*/eeg/*_events.tsv'))):
    b = os.path.basename(f).split('_')
    with open(f, newline='') as fh:
        S[(b[0], b[1], b[2][5:])] = [r['type'] for r in csv.DictReader(fh, delimiter='\t')]
BAD = {('sub-08', 'ses-01', 'OVI'), ('sub-13', 'ses-02', 'OVI')}  # label anomalies (gao_structure_kr.md 3)
print('excluded runs (label anomalies):', sorted(BAD))
for t in ('AVI', 'FVI', 'OVI'):
    L = 40 if t == 'OVI' else 30
    keys = [k for k in S if k[2] == t and k not in BAD]
    blk = {k: [tuple(S[k][i:i + L]) for i in range(0, len(S[k]), L)] for k in keys}
    cnt = collections.Counter(b for k in keys for b in blk[k])
    print('\n## %s: runs %d, distinct sequences %d, distinct blocks %d (of %d), max reuse of one block %d' % (
        t, len(keys), len({tuple(S[k]) for k in keys}), len(cnt), sum(map(len, blk.values())), max(cnt.values())))
    groups = collections.defaultdict(list)
    for k in keys:
        groups[tuple(S[k])].append('%s/%s' % (k[0], k[1][-2:]))
    for i, g in enumerate(sorted(groups.values(), key=len, reverse=True)):
        if len(g) > 1:
            print('  identical sequence group %d (%d runs): %s' % (i + 1, len(g), ' '.join(g)))
    print('  session pairs (subject, position agreement, blocks of ses-02 that also occur in ses-01):')
    for s in sorted({k[0] for k in keys}):
        a, b = (s, 'ses-01', t), (s, 'ses-02', t)
        if a in blk and b in blk:
            ag = sum(x == y for x, y in zip(S[a], S[b])) / len(S[a])
            print('    %s %.3f %d' % (s, ag, sum(x in set(blk[a]) for x in blk[b])))
    ag = [sum(x == y for x, y in zip(S[a], S[b])) / len(S[a]) for a, b in itertools.combinations(keys, 2) if S[a] != S[b]]
    print('  non-identical run pairs: position agreement mean %.3f, min %.3f, max %.3f' % (sum(ag) / len(ag), min(ag), max(ag)))
    print('  cross-subject (test subject: other-subject runs, of which share no block with any test run, blocks kept if shared blocks are dropped):')
    for s in sorted({k[0] for k in keys}):
        mine = {b for k in keys if k[0] == s for b in blk[k]}
        others = [k for k in keys if k[0] != s]
        kept_blocks = sum(b not in mine for k in others for b in blk[k])
        print('    %s %d %d %d/%d' % (s, len(others), sum(not (set(blk[k]) & mine) for k in others), kept_blocks,
                                    sum(len(blk[k]) for k in others)))
