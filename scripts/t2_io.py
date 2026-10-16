"""Readers of the 0-T2 files (D-109 amendment 2; eeg-dir3 327aca9 formats), no torch.

matched_sets.json: {level, salt, key_format, sets: [{q, cands: [19 keys], cands_same_run: [19 keys or empty]}], ...}
match_attrs.json : {key_format, fields, queries: {key: [run, t, tau, n_w, n_L, n_S, D_T, seq]}}
"""
import json


def read_sets(p):
    """(level, {query key: dict(cands, cands_same_run)}) for every listed query"""
    M = json.load(open(p))
    return M['level'], {x['q']: dict(cands=list(x['cands']), cands_same_run=list(x.get('cands_same_run') or [])) for x in M['sets']}


def read_attrs(p):
    return json.load(open(p))['queries']


def full(sets, key='cands'):
    """queries whose candidate list has 19 entries"""
    return {k: dict(cands=x[key]) for k, x in sets.items() if len(x[key]) == 19}
