"""F-IV B1 density audit (D-109 section 13, amendments 1-2; alpha-free, descriptive; "B1 test matrices re-opened").

R_dens = (MRP_matched - 0.5) / (MRP_full - 0.5) on the T2 query set Q (results/t2/matched_sets.json, EEG margin of section 4)
for the 17 pool subjects x 20 test runs, query-weighted, with the subject x run pigeonhole 90% interval (1,000, seed
density.seed). MRP_full: B1 definition (all B1 test queries of the run, exclusion 0) from the stored scores.npz main matrices.
MRP_matched: truth + the 19 matched candidates, (below + 0.5 ties)/19, recorded w2v candidates, fixed alignment.
  level L0                 : stored main matrices only;
  level L1 or above        : recorded scores of the re-inference path (scripts/t2_infer.py --mode test --recorded-only),
                             only after the re-inference check passed (results/t2/reinfer_check.json);
  L1+ and the check failed : stored matrices with cands_same_run ("differs from the T2 candidate set").
Report: per-subject and per-run R_dens, 19-subject value, MRP_unmatched (19 random same-run Q queries >= 30 s away with a
different word sequence, default_rng([unmatched_seed, run, i])) and the N_H condition (matched recorded MRP at least 0.03 below
unmatched). The matched recorded MRP is the same data as the MRP_rec term of H_T2b (section 9).

Report items added after the results (D-109 section 13, missed in the implementation, not used in any verdict):
  --same-run          : L1+ and the check passed -> also the stored-matrix value with cands_same_run (queries with 19 same-run
                       candidates only; MRP_full on the same queries);
  --sets / --scores-dir: another matched-set file (results/t2/levels/, scripts/t2_levels.py) and its recorded-score folder.

usage: python scripts/t2_density.py configs/t2.yaml [--same-run] [--sets PATH --scores-dir DIR] [--out PATH]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, argparse, hashlib, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import t2_lib as T
from t2_lib import path, parse_key
from t2_io import read_sets, read_attrs, full


def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--out'); ap.add_argument('--same-run', action='store_true'); ap.add_argument('--sets'); ap.add_argument('--scores-dir')
    a = ap.parse_args(); v = yaml.safe_load(open(path(a.config)))
    b1 = yaml.safe_load(open(path(v['b1_config']))); sp = b1['split']; dq = v['density']
    setp = path(a.sets or v['matched_sets']); level, sets_all = read_sets(setp); attrs = read_attrs(path(v['match_attrs']))
    assert level in ('L0', 'L1', 'L2', 'L3'), f'matched-set level unknown: {level}'
    source = 'stored'
    if level != 'L0':
        chk = json.load(open(path(v['reinfer_check']['out']))); assert chk['complete'], 'the registered re-inference check has not been run'
        source = 'reinfer' if chk['passed'] else 'stored_same_run'
        if a.same_run: assert chk['passed']; source = 'stored_same_run'
    cand_key = 'cands_same_run' if source == 'stored_same_run' else 'cands'
    sets = full(sets_all, cand_key)
    meta = json.load(open(path(os.path.join(b1['cache_dir'], 'eeg_meta.json')))); fs, lens = meta['sample_rate'], meta['run_lengths']; w, e = b1['window'], b1['eeg']
    def eeg_ok(t, r0):
        m = v['eeg_margin_s']; lo = int(round((t + w['tmin'] + e['offset_s'] - m) * fs)); hi = int(round((t + w['tmin'] + e['offset_s'] + m) * fs)) + int(round((w['tmax'] - w['tmin']) * fs))
        return lo >= 0 and hi <= lens[r0]
    subs19 = sorted(s for f in sp['subject_folds'] for s in f) + sorted(s for x in sp['extra_targets'].values() for s in x); pool = v['pool']
    fullm, matched, unm = {}, {}, {}; wq = np.zeros(20); files = {}
    for K in range(5):
        for F in range(5):
            d = T.b1_folder(b1, K, F); Z = np.load(os.path.join(d, 'scores.npz')); files[f'k{K}_f{F}'] = sha(os.path.join(d, 'scores.npz'))
            R = np.load(path(os.path.join(a.scores_dir or v['infer_out'], f'test_rec_k{K}_f{F}.npz'))) if source == 'reinfer' else None
            targets = sp['subject_folds'][K] + sp['extra_targets'].get(K, [])
            for r0 in [r - 1 for r in sp['run_folds'][F]]:
                on = Z['onset'][Z['run'] == r0]; keys_b1 = {round(float(t), 6): i for i, t in enumerate(on)}
                rows = sorted((k for k in sets if parse_key(k)[0] == r0 and eeg_ok(attrs[k][1], r0)), key=lambda k: attrs[k][1]); wq[r0] = len(rows)
                if not rows:                                               # a level file may leave a run empty: weight 0, per-run value None
                    for s in targets: fullm[(s, r0)] = matched[(s, r0)] = unm[(s, r0)] = 0.0
                    continue
                rt = np.array([attrs[k][1] for k in rows]); um = T.unmatched(rows, attrs, rt, dq['unmatched_seed'], r0 + 1)
                if source == 'reinfer':
                    cols = list(R[f'r{r0 + 1}_cols']); assert list(R[f'r{r0 + 1}_rows']) == rows
                else:
                    cols = sorted({k for k in rows} | {c for k in rows for c in sets[k]['cands']}, key=lambda k: attrs[k][1])
                    assert all(parse_key(c)[0] == r0 for c in cols)
                C = T.cand_index(rows, cols, sets); ku = [k for k in rows if um[k] is not None]; Cu = T.cand_index(ku, cols, sets, extra=um)
                bi = np.array([keys_b1[round(float(attrs[k][1]), 6)] for k in rows]); bc = np.array([keys_b1.get(round(float(attrs[k][1]), 6), -1) for k in cols])
                for s in targets:
                    S = Z[f'main_s{s}_r{r0}'].astype(np.float64)
                    rk = (S[bi] > S[bi, bi][:, None]).sum(1); fullm[(s, r0)] = float(np.mean(1 - rk / (len(S) - 1)))
                    if source == 'reinfer': Sm = R[f'rec_fixed_s{s}_r{r0 + 1}'].astype(np.float64)
                    else: assert (bc >= 0).all(); Sm = S[np.ix_(bi, bc)]
                    matched[(s, r0)] = float(T.cell_obs(Sm, C).mean()); ui = [rows.index(k) for k in ku]
                    unm[(s, r0)] = float(T.cell_obs(Sm[ui], Cu).mean()) if ku else float('nan')   # level files: a run may have no unmatched draw
    def mat(dct, subs): return np.array([[dct[(s, r)] for r in range(20)] for s in subs])
    Fm, Mm, Um = mat(fullm, pool), mat(matched, pool), mat(unm, pool)
    rng = np.random.default_rng(dq['seed']); ns, nr = Fm.shape; Rb = np.empty(1000)
    for b in range(1000):
        si = rng.integers(0, ns, ns); ri = rng.integers(0, nr, nr)
        Rb[b] = (T.pooled(Mm[np.ix_(si, ri)], wq[ri]) - 0.5) / (T.pooled(Fm[np.ix_(si, ri)], wq[ri]) - 0.5)
    pf, pm = T.pooled(Fm, wq), T.pooled(Mm, wq); pu = T.pooled(Um, wq) if np.isfinite(Um).all() else None; Rd = (pm - 0.5) / (pf - 0.5); lo, hi = np.percentile(Rb, [5, 95])
    desc = ('more than half of the B1 MRP excess remains after density matching' if lo >= 0.5 else
            'more than half of the B1 MRP excess relies on density differences' if hi < 0.5 else 'undecidable')
    per_sub = {str(s): (np.average([matched[(s, r)] for r in range(20)], weights=wq) - 0.5) / (np.average([fullm[(s, r)] for r in range(20)], weights=wq) - 0.5) for s in subs19}
    per_run = {str(r + 1): (np.mean([matched[(s, r)] for s in pool]) - 0.5) / (np.mean([fullm[(s, r)] for s in pool]) - 0.5) if wq[r] else None for r in range(20)}
    F19, M19 = mat(fullm, subs19), mat(matched, subs19)
    res = dict(level=level, source=source, candidate_set='T2 candidate set' if source != 'stored_same_run' else 'differs from the T2 candidate set',
               n_queries=int(wq.sum()), queries_per_run=wq.astype(int).tolist(), MRP_full=pf, MRP_matched=pm, MRP_unmatched=pu, R_dens=Rd, R_dens_ci90=[float(lo), float(hi)],
               description=desc, N_H_condition=bool(pm <= pu - 0.03) if pu is not None else None, unmatched_missing_cells=int((~np.isfinite(Um)).sum()), R_dens_19=(T.pooled(M19, wq) - 0.5) / (T.pooled(F19, wq) - 0.5), per_subject=per_sub, per_run=per_run,
               note='B1 test matrices re-opened, descriptive (no alpha). MRP_matched (recorded, fixed) is the same data as the MRP_rec term of H_T2b (D-109 sections 9, 13).',
               same_run_report=bool(a.same_run), scores_dir=a.scores_dir or v['infer_out'],
               inputs=dict(matched_sets=sha(setp), match_attrs=sha(path(v['match_attrs'])), b1_scores=files))
    out = path(a.out or 'results/t2/density_audit.json'); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('level', 'source', 'n_queries', 'MRP_full', 'MRP_matched', 'MRP_unmatched', 'R_dens', 'R_dens_ci90', 'description', 'N_H_condition')}), flush=True)


if __name__ == '__main__':
    main()
