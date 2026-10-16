"""D-113 design (EEG-free): trial table of the Gao et al. (2026) runs, the arm a / arm b splits with the shared-block
exclusion (D-113 4.3), and the EEG-free meters ADJ and block position (D-113 5.3 items 2-3).

Trial table (one row per trial, all runs in sorted file order): sub, ses, task, run (index), pos (index in run), block
(index in run), bpos (index in block), onset (s), label (item index in cfg tasks[task].items; -1 if not an item), and
bkey (the block's label sequence as a tuple of type strings; two blocks are "the same" when their bkey is equal).
"""
import csv, glob, os
import numpy as np


def load_trials(bids, cfg, relabel=False):
    bad = {tuple(r) for r in cfg['bad_runs']}
    rl = cfg.get('sensitivity_relabel') if relabel else None
    if rl: bad = bad - {tuple(rl['run'])}
    rows, runs = [], []
    for f in sorted(glob.glob(os.path.join(bids, 'sub-*', 'ses-*', 'eeg', '*_events.tsv'))):
        b = os.path.basename(f).split('_'); key = (b[0], b[1], b[2][5:])
        if key in bad: continue
        with open(f, newline='') as fh:
            ev = list(csv.DictReader(fh, delimiter='\t'))
        types = [e['type'] for e in ev]
        if rl and key == tuple(rl['run']): types = [rl['to'] if t == rl['from'] else t for t in types]
        items = cfg['tasks'][key[2]]['items']; L = cfg['tasks'][key[2]]['block']
        bkeys = [tuple(types[i:i + L]) for i in range(0, len(types), L)]
        ri = len(runs); runs.append(dict(key=key, events=f, n=len(ev)))
        for i, (e, t) in enumerate(zip(ev, types)):
            rows.append(dict(sub=key[0], ses=key[1], task=key[2], run=ri, pos=i, block=i // L, bpos=i % L, onset=float(e['onset']),
                             label=items.index(t) if t in items else -1, bkey=bkeys[i // L]))
    T = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != 'bkey'}
    T['bkey'] = [r['bkey'] for r in rows]
    return T, runs


def splits(T, cfg, exclude=True):
    """{(arm, sub, task): (train_idx, test_idx)} for arm 'b' (own ses-01 -> own ses-02) and arm 'a' (other subjects ->
    the subject, all sessions). exclude=True drops blocks with the same label sequence (D-113 4.3): arm b drops test blocks
    that also occur in the training session; arm a drops training blocks that also occur in any test block."""
    sub, ses, task, bk = T['sub'], T['ses'], T['task'], T['bkey']
    out = {}
    for t in cfg['tasks']:
        for s in sorted(set(sub)):
            # arm b
            tr = np.flatnonzero((sub == s) & (task == t) & (ses == 'ses-01')); te = np.flatnonzero((sub == s) & (task == t) & (ses == 'ses-02'))
            if len(tr) and len(te):
                if exclude:
                    seen = {bk[i] for i in tr}; te = np.array([i for i in te if bk[i] not in seen], int)
                if len(te): out[('b', s, t)] = (tr, te)
            # arm a
            te = np.flatnonzero((sub == s) & (task == t)); tr = np.flatnonzero((sub != s) & (task == t))
            if len(te):
                if exclude:
                    mine = {bk[i] for i in te}; tr = np.array([i for i in tr if bk[i] not in mine], int)
                out[('a', s, t)] = (tr, te)
    return out


def _argmax_rows(C, marg):
    """argmax per row of count matrix C, ties broken by the training marginal, then the lowest index"""
    S = C + 1e-6 * marg[None] / max(marg.sum(), 1) - 1e-9 * np.arange(C.shape[1])[None]
    return S.argmax(1), S


def meter_adj(T, tr, te, K):
    """ADJ: label from the previous trial's label (same run), transition counts of the training trials. The first trial of a
    run is predicted from the training marginal. Returns predictions and scores [n_test, K]."""
    lab, run, pos = T['label'], T['run'], T['pos']
    prev = {(run[i], pos[i]): lab[i] for i in range(len(lab))}
    C = np.zeros((K + 1, K)); marg = np.bincount(lab[tr], minlength=K).astype(float)
    for i in tr:
        p = prev.get((run[i], pos[i] - 1), -1); C[p if p >= 0 else K, lab[i]] += 1
    C[K] = marg
    rows = np.array([prev.get((run[i], pos[i] - 1), -1) for i in te]); rows[rows < 0] = K
    _, S = _argmax_rows(C, marg); S = S[rows]
    return S.argmax(1), S


def meter_bpos(T, tr, te, K, L):
    """block position: label from the trial's index in its block, position-label counts of the training trials"""
    lab, bpos = T['label'], T['bpos']
    C = np.zeros((L, K)); marg = np.bincount(lab[tr], minlength=K).astype(float)
    for i in tr: C[bpos[i], lab[i]] += 1
    _, S = _argmax_rows(C, marg); S = S[bpos[te]]
    return S.argmax(1), S
