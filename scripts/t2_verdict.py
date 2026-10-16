"""F-IV T2 verdict (D-109 sections 8-9, 14). The only reader of the test score matrices of scripts/t2_infer.py.

Not opened (verdict.json says why, nothing else is read): 0-T2 failed, the development gate did not pass, the frozen rule in
configs/t2.yaml differs from the development stage, or the MDE record is missing.
H_T2a: E_T = query-weighted sum over the 17 x 20 cells of (matched TTS MRP - 30-s-excluded shift-null mean), frozen rule,
subject x run pigeonhole (1,000, bootstrap.seed); the five D-097 categories at alpha 0.05 (lower = 5th percentile).
H_T2b (only when H_T2a is "adopt"): D = recorded - TTS MRP, non-inferior if the 95th percentile < 0.03 (0.02 reported);
"hold (MDE)" if the recorded MDE > 0.03. Prerequisites 1-7 of section 8; reports of section 14 that come from the same files.

usage: python scripts/t2_verdict.py configs/t2.yaml [--scores DIR --sets PATH --attrs PATH --out PATH --sim]
       --sim: fake inputs of scripts/t2_verdict_sim.py (skips the commit-time checks that need git)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, argparse, hashlib, datetime, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import t2_lib as T
import c1_stats as CS
from t2_lib import path, parse_key
from t2_io import read_sets, read_attrs, full


def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load_cells(v, sd, kind_tag, jobs, subjects_of, rows, cols, sets, attrs, maps, rule, with_null=True):
    """per (subject, run0): dict(tts, null, rec) for the frozen rule from the npz files of one mode"""
    out = {}
    for K, F in jobs:
        Z = np.load(os.path.join(sd, f'{kind_tag}_k{K}_f{F}.npz'))
        for r0 in [r - 1 for r in v['_sp']['run_folds'][F]]:
            if f'r{r0 + 1}_rows' not in Z.files: continue
            assert list(Z[f'r{r0 + 1}_rows']) == rows[r0] and list(Z[f'r{r0 + 1}_cols']) == cols[r0], f'{kind_tag} k{K}_f{F} r{r0 + 1}: rows or columns differ'
            C = T.cand_index(rows[r0], cols[r0], sets); ct = np.array([attrs[k][1] for k in cols[r0]]); cs = np.array([parse_key(k)[0] == r0 for k in cols[r0]])
            rt = np.array([attrs[k][1] for k in rows[r0]])
            for s in subjects_of(K):
                S = Z[f'tts_{rule}_s{s}_r{r0 + 1}'].astype(np.float64); c = dict(tts=float(T.cell_obs(S, C).mean()), n=len(rows[r0]))
                if with_null: nl, kp, kc = T.cell_null(S, C, rt, ct, cs, maps[r0], v['exclusion_s']); c.update(null=float(nl.mean()), null_draws=nl, kept_pairs=kp, kept_cands=kc)
                if f'rec_{rule}_s{s}_r{r0 + 1}' in Z.files: c['rec'] = float(T.cell_obs(Z[f'rec_{rule}_s{s}_r{r0 + 1}'].astype(np.float64), C).mean())
                if with_null and 'rec' in c: c['rec_null'] = float(T.cell_null(Z[f'rec_{rule}_s{s}_r{r0 + 1}'].astype(np.float64), C, rt, ct, cs, maps[r0], v['exclusion_s'])[0].mean())
                out[(s, r0)] = c
    return out


def mat(cells, subs, key): return np.array([[cells[(s, r)][key] for r in range(20)] for s in subs])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--scores'); ap.add_argument('--sets'); ap.add_argument('--attrs'); ap.add_argument('--out')
    ap.add_argument('--sim', action='store_true'); a = ap.parse_args(); v = yaml.safe_load(open(path(a.config))); vd = v['verdict']; rule = v['align']['rule']
    b1 = yaml.safe_load(open(path(v['b1_config']))); sp = b1['split']; v['_sp'] = sp; out = path(a.out or vd['out']); os.makedirs(os.path.dirname(out), exist_ok=True)
    sd = path(a.scores or v['infer_out']); setp, attp = path(a.sets or v['matched_sets']), path(a.attrs or v['match_attrs'])
    g0 = json.load(open(path(vd['gate0T2']))); dev = json.load(open(path(vd['dev_stage'])))
    why = []
    if not g0.get('passed', g0.get('verdict') == 'pass'): why.append('0-T2 failed')
    if not dev['gate']['passed']: why.append('development gate not passed')
    if rule != dev['rule_selection']['rule']: why.append(f'configs/t2.yaml align.rule {rule} != development selection {dev["rule_selection"]["rule"]}')
    if not os.path.exists(path(v['mde']['file'])): why.append('MDE record missing')
    if why:
        json.dump(dict(opened=False, why=why, written_at=datetime.datetime.now().isoformat(timespec='seconds')), open(out, 'w'), indent=1); print('NOT OPENED', why); return
    sets = full(read_sets(setp)[1]); attrs = read_attrs(attp)
    summ = {m: json.load(open(os.path.join(sd, f'{m}_summary.json'))) for m in ('test', 'test_noise', 'phaserand')}
    for m, x in summ.items(): assert x['matched_sets']['sha256'] == sha(setp) and x['match_attrs']['sha256'] == sha(attp), f'{m}: other matched-set file'
    rows, cols = {}, {}
    for K in range(5):
        for F in range(5):
            Z = np.load(os.path.join(sd, f'test_k{K}_f{F}.npz'))
            for r0 in [r - 1 for r in sp['run_folds'][F]]:
                if f'r{r0 + 1}_rows' in Z.files: rows.setdefault(r0, list(Z[f'r{r0 + 1}_rows'])); cols.setdefault(r0, list(Z[f'r{r0 + 1}_cols']))
    assert sorted(rows) == list(range(20)), 'every test run needs Q queries'
    maps = T.run_maps({r: np.array([attrs[k][1] for k in rows[r]]) for r in range(20)}, v['shift_null']['n_shift'], v['shift_null']['min_shift_s'], v['shift_null']['seed'])
    tg = lambda K: sp['subject_folds'][K] + sp['extra_targets'].get(K, []); jobs = [(K, F) for K in range(5) for F in range(5)]
    main_c = load_cells(v, sd, 'test', jobs, tg, rows, cols, sets, attrs, maps, rule)
    noise_c = load_cells(v, sd, 'test_noise', jobs, tg, rows, cols, sets, attrs, maps, rule)
    pr_c = load_cells(v, sd, 'phaserand', [(0, F) for F in range(5)], lambda K: sp['subject_folds'][0], rows, cols, sets, attrs, maps, rule)
    pool = v['pool']; subs19 = sorted({s for (s, _) in main_c}); w = np.array([len(rows[r]) for r in range(20)], float); bs = v['bootstrap']
    TT, NU, RE = mat(main_c, pool, 'tts'), mat(main_c, pool, 'null'), mat(main_c, pool, 'rec')
    E = TT - NU; Dm = RE - TT; Eb = T.pigeonhole(E, w, bs['n_boot'], bs['seed']); Db = T.pigeonhole(Dm, w, bs['n_boot'], bs['seed'])
    E_pt, D_pt = T.pooled(E, w), T.pooled(Dm, w); iv = CS.interval(Eb, alphas=(vd['alpha'],)); lo = iv[f"lower_{vd['alpha']}"]
    cat, note = CS.category(E_pt, lo, iv['p95'], vd['sesoi'])
    # prerequisites (section 8)
    pre = {}
    NE = mat(noise_c, pool, 'tts') - mat(noise_c, pool, 'null'); nb = T.pigeonhole(NE, w, bs['n_boot'], bs['seed'])
    pre['1_noise'] = dict(excess=T.pooled(NE, w), upper90=float(np.percentile(nb, 95)), bound=0.01, ok=bool(np.percentile(nb, 95) < 0.01))
    prs = sp['subject_folds'][0]; PT = mat(pr_c, prs, 'tts'); pr_pt = T.pooled(PT, w)
    nd = np.stack([[pr_c[(s, r)]['null_draws'] for r in range(20)] for s in prs])            # [S, R, draws]
    pr_null = (nd.mean(0) * w[:, None]).sum(0) / w.sum(); pre['2_phaserand'] = dict(mrp=pr_pt, null_p975=float(np.percentile(pr_null, 97.5)), ok=bool(pr_pt <= np.percentile(pr_null, 97.5)))
    de = v['de_baseline']; dep = path(de['file']); de_ok = os.path.exists(dep) and sha(dep) == de['sha256']
    if de_ok and not a.sim:
        t_de = datetime.datetime.fromisoformat(de['committed_at']).timestamp()
        de_ok = all(os.path.getmtime(os.path.join(sd, f'test_k{K}_f{F}.npz')) > t_de for K, F in jobs)
    pre['3_de_first'] = dict(file=de['file'], commit=de.get('commit'), committed_at=de.get('committed_at'), ok=bool(de_ok))
    DE = json.load(open(dep)); DT = np.array([[DE['subjects'][str(s)]['cells'][str(r + 1)]['mrp_tts'] for r in range(20)] for s in pool])
    assert all(DE['subjects'][str(pool[0])]['cells'][str(r + 1)]['n'] == len(rows[r]) for r in range(20)), 'DE query counts differ'
    gb = T.pigeonhole(TT - DT, w, bs['n_boot'], bs['seed']); pre['4_decoder_gt_de'] = dict(diff=T.pooled(TT - DT, w), lower90=float(np.percentile(gb, 5)), ok=bool(np.percentile(gb, 5) > 0))
    devs = {m: x['device'] for m, x in summ.items()} | dict(dev=dev['inputs']['device']); chk = json.load(open(path(v['reinfer_check']['out'])))
    devs['reinfer_check'] = chk['device']; pre['5_device'] = dict(devices=devs, ok=all('RTX 4070' in d for d in devs.values()))
    pre['6_reinfer_check'] = dict(passed=chk['passed'], complete=chk['complete'], max_abs_mrp_diff=chk['max_abs_mrp_diff'], ok=bool(chk['passed'] and chk['complete']))
    md = json.load(open(path(v['mde']['file']))); md_ok = a.sim or all(os.path.getmtime(path(v['mde']['file'])) < os.path.getmtime(os.path.join(sd, f'test_k{K}_f{F}.npz')) for K, F in jobs)
    pre['7_mde'] = dict(MDE=md['MDE'], threshold=v['mde']['threshold'], recorded_before_test=bool(md_ok), hold=bool(md['MDE'] > v['mde']['threshold']), ok=bool(md_ok))
    pre_ok = all(x['ok'] for x in pre.values())
    a_res = dict(E_T=E_pt, lower95=lo, ci90=[iv['p5'], iv['p95']], category=cat, note=note)
    if not pre_ok: verdict_a, verdict_b = 'hold (prerequisite)', 'not judged'
    else:
        verdict_a = cat
        if cat != 'adopt': verdict_b = 'not judged (H_T2a not adopted)'
        elif pre['7_mde']['hold']: verdict_b = 'hold (MDE)'
        else: verdict_b = 'non-inferior' if np.percentile(Db, 95) < vd['ni_margin'] else 'not non-inferior'
    b_res = dict(D=D_pt, upper95=float(np.percentile(Db, 95)), margin=vd['ni_margin'], noninferior_002=bool(np.percentile(Db, 95) < vd['ni_report']), verdict=verdict_b)
    # reports (section 14, from the same files)
    REN = mat(main_c, pool, 'rec_null'); E_rec = T.pooled(RE - REN, w)
    rep = dict(MRP_tts=T.pooled(TT, w), MRP_rec=T.pooled(RE, w), E_rec=E_rec, retention=(E_pt / E_rec if E_rec else None),
               per_subject={str(s): dict(E_T=float(np.average([main_c[(s, r)]['tts'] - main_c[(s, r)]['null'] for r in range(20)], weights=w)),
                                         D=float(np.average([main_c[(s, r)]['rec'] - main_c[(s, r)]['tts'] for r in range(20)], weights=w))) for s in subs19},
               E_T_19=T.pooled(mat(main_c, subs19, 'tts') - mat(main_c, subs19, 'null'), w),
               null_kept_pairs=float(np.mean([c['kept_pairs'] for c in main_c.values()])), null_kept_candidates=float(np.mean([c['kept_cands'] for c in main_c.values()])),
               rec_same_data_as_density_audit='MRP_rec (fixed rule) is the same data as the density audit matched recorded MRP (D-109 sections 9, 13)' if rule == 'fixed' else 'lse0.5: MRP_rec from the re-inference path only',
               labels='within-subject calibration 14 runs (+2 early-stopping validation runs), ds004408 test runs re-used (opened in B1, D-090, B2, D-100, D-101; sixth and last confirmatory use), B1 test matrices re-opened, density-matched TTS text candidates, device RTX 4070')
    res = dict(opened=True, rule=rule, n_queries=int(w.sum()), queries_per_run=w.astype(int).tolist(), H_T2a=dict(a_res, verdict=verdict_a), H_T2b=b_res,
               prerequisites=pre, prerequisites_ok=pre_ok, report=rep, alpha=vd['alpha'], sesoi=vd['sesoi'],
               inputs=dict(matched_sets=sha(setp), match_attrs=sha(attp), score_files={m: {j: x['sha256'] for j, x in s_['jobs'].items()} for m, s_ in summ.items()}),
               written_at=datetime.datetime.now().isoformat(timespec='seconds'))
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(dict(H_T2a=res['H_T2a'], H_T2b=b_res, prerequisites={k: x['ok'] for k, x in pre.items()})), flush=True)


if __name__ == '__main__':
    main()
