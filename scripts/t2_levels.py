"""Matched sets of every 0-T2 level from the stimulus attributes only (D-109 section 13 report item "values per matching level";
computed after the 0-T2 and density-audit results because the item was missed in the implementation, not used in any verdict).

The matching of scripts/t2_gate0T2.py (eeg-dir3, 327aca9) is rebuilt from results/t2/match_attrs.json (run, t, n_w, n_L, n_S,
D_T, seq of all 11,220 valid queries; no EEG, no audio) with the levels, salt, n_cand and min_sep of configs/t2_gate0T2.yaml:
same `eligible` rule, same sha256 order, queries with fewer than 19 eligible candidates left out, cands_same_run at the level
tolerances. Check first: the rebuilt chosen level must equal results/t2/matched_sets.json (bytes, else query and candidate lists).
Only if it does, the files of the other levels are written (same format, out_dir/matched_sets_L{n}.json).

usage: python scripts/t2_levels.py configs/t2_gate0T2.yaml [--attrs PATH] [--sets PATH] [--out-dir DIR]
"""
import os, sys, json, hashlib, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha(s): return hashlib.sha256(s.encode()).hexdigest()


def fsha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--attrs', default='results/t2/match_attrs.json')
    ap.add_argument('--sets', default='results/t2/matched_sets.json'); ap.add_argument('--out-dir', default='results/t2/levels'); a = ap.parse_args()
    P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
    c = yaml.safe_load(open(P(a.config))); c1 = yaml.safe_load(open(P(c['t1_config']))); b1 = yaml.safe_load(open(P(c1['b1_config'])))
    sp = b1['split']; mt = c['matching']; fold_of = {r - 1: f for f in range(5) for r in sp['run_folds'][f]}
    att = json.load(open(P(a.attrs))); keys = list(att['queries']); V = [att['queries'][k] for k in keys]   # insertion order = 0-T2 query order
    qr = np.array([v[0] - 1 for v in V]); qt = np.array([v[1] for v in V], float)
    A = dict(nw=np.array([v[3] for v in V], float), nL=np.array([v[4] for v in V], float), nS=np.array([v[5] for v in V], float),
             DT=np.array([v[6] for v in V], float), seq=np.array([v[7] for v in V]))

    def eligible(q, lv):                                                    # verbatim rule of scripts/t2_gate0T2.py
        pool = np.where(qr == qr[q])[0] if lv['pool'] == 'same_run' else np.where(np.isin(qr, [r - 1 for r in sp['run_folds'][fold_of[qr[q]]]]))[0]
        ok = (qr[pool] != qr[q]) | (np.abs(qt[pool] - qt[q]) >= mt['min_sep_s'])
        ok &= A['nw'][pool] == A['nw'][q]; ok &= np.abs(A['nL'][pool] - A['nL'][q]) <= lv['letters_rel'] * A['nL'][q] + 1e-9
        ok &= np.abs(A['nS'][pool] - A['nS'][q]) <= lv['syl_abs']; ok &= np.abs(A['DT'][pool] - A['DT'][q]) <= lv['dur_rel'] * A['DT'][q] + 1e-9
        ok &= A['seq'][pool] != A['seq'][q]; ok &= pool != q; return pool[ok]

    def build(LV):
        cand, same = {}, {}
        for q in range(len(keys)):
            el = eligible(q, LV)
            if len(el) >= mt['n_cand']: cand[q] = sorted(el, key=lambda j: sha(keys[q] + '|' + keys[j] + mt['salt']))[:mt['n_cand']]
        LVs = dict(LV, pool='same_run')
        for q in sorted(cand):
            el = eligible(q, LVs); same[q] = sorted(el, key=lambda j: sha(keys[q] + '|' + keys[j] + mt['salt']))[:mt['n_cand']] if len(el) >= mt['n_cand'] else []
        return dict(level=LV['name'], salt=mt['salt'], key_format=att['key_format'],
                    sets=[dict(q=keys[q], cands=[keys[j] for j in cand[q]], cands_same_run=[keys[j] for j in same[q]]) for q in sorted(cand)],
                    note_same_run='cands_same_run: same-run eligible candidates at the chosen level tolerances, same sha256 order, 19 or empty (D-109 sections 7 and 13)')

    ref = json.load(open(P(a.sets))); names = [lv['name'] for lv in mt['levels']]; od = P(a.out_dir); os.makedirs(od, exist_ok=True)
    M = build(mt['levels'][names.index(ref['level'])]); tmp = os.path.join(od, f"rebuild_{ref['level']}.json"); json.dump(M, open(tmp, 'w'), indent=0)
    same_bytes = fsha(tmp) == fsha(P(a.sets)); same_lists = [(x['q'], x['cands'], x['cands_same_run']) for x in M['sets']] == [(x['q'], x['cands'], x['cands_same_run']) for x in ref['sets']]
    res = dict(chosen_level=ref['level'], n_queries_valid=len(keys), reference=dict(file=a.sets, sha256=fsha(P(a.sets))), attrs=dict(file=a.attrs, sha256=fsha(P(a.attrs))),
               rebuild=dict(file=os.path.relpath(tmp, ROOT), sha256=fsha(tmp), n_sets=len(M['sets']), same_bytes=bool(same_bytes), same_lists=bool(same_lists)), levels={})
    print('rebuild', ref['level'], json.dumps(res['rebuild']), flush=True)
    if same_bytes or same_lists:
        for lv in mt['levels']:
            if lv['name'] == ref['level']: continue
            L = build(lv); p = os.path.join(od, f"matched_sets_{lv['name']}.json"); json.dump(L, open(p, 'w'), indent=0)
            res['levels'][lv['name']] = dict(file=os.path.relpath(p, ROOT), sha256=fsha(p), n_sets=len(L['sets']), coverage=len(L['sets']) / len(keys),
                                             same_run_lists_full=sum(len(x['cands_same_run']) == mt['n_cand'] for x in L['sets']))
            print(lv['name'], json.dumps(res['levels'][lv['name']]), flush=True)
    else:
        res['note'] = 'rebuild differs from the committed matched sets: other levels not written (not reproducible, not computed)'
    json.dump(res, open(os.path.join(od, 'levels_rebuild.json'), 'w'), indent=1); print(json.dumps({k: res[k] for k in ('chosen_level', 'rebuild')}), flush=True)


if __name__ == '__main__':
    main()
