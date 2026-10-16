"""D-113 arm c scoring (server CPU, gpuq). Part `pipeline`: the 30-channel pipeline control of section 6.4. Part `gao`: 6.3-6.5
(see gao(); wording: perception control fails -> undecidable (perception control); imagery excess lower <= 0 -> not detected;
arm c minus the larger meter (eye, muscle; arm b split, same test trials) lower <= 0 -> undecidable (meters); else observed).

Reads the infer run folder of scripts/fv_gao_c_infer.py (fetched to results/runs/) and the staged inputs (true columns).
Per (candidate, mode, target, input X63 / X30): the predicted embeddings of the seeds are L2-normalised, averaged and normalised again
(D-113 6.1, as the F-V verdict), scored by cosine against the 200 development concept names (names_text of configs/fv_train.yaml,
rows of fv_train.name_rows), and each evaluated concept gives the rank percentile 1 - rank / (K - 1) of its own name. MRP = mean
over the targets' concepts (each target weighted by its concept count, as gate D). Excess = MRP - mean of the concept-permutation
null (per target the true columns are re-assigned among the evaluated concepts, n_perm, perm_seed; gate D (i) definition).
Pass when excess(X30) >= ratio_min x excess(X63). The 63-channel embeddings are compared with the model's own
dev_scores.npz (A_ses4_E_<layer>) when the source run folder is on disk (consistency check, reported).
Output: results/gao_imagery/armc_pipeline[_<tag>].json, armc_gao[_<tag>].json
usage (gpuq): env/bin/python -I scripts/fv_gao_c_score.py configs/fv_gao_c.yaml --part pipeline --infer results/runs/<infer run> [--tag NAME]
              env/bin/python -I scripts/fv_gao_c_score.py configs/fv_gao_c.yaml --part gao --infer results/runs/<infer run> --names results/runs/<names run>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import argparse, json, sys, time
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def unit(A): return A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)


def rank_pct(S, true):
    rk = (S > S[np.arange(len(S)), true][:, None]).sum(1); return 1 - rk / (S.shape[1] - 1)


def pipeline(c, tc, a):
    from ese.things.data import Meta
    from fv_train import name_rows
    meta = Meta(P(tc['meta'])); dev_c = meta.dev_concepts(tc['dev']['salt'], tc['dev']['n'])
    cand = unit(np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['names_text']))[name_rows(tc, meta, dev_c)])
    inf = json.load(open(os.path.join(P(a.infer), 'metrics.json'))); sd = P(c['stage_dir']); pcfg = c['pipeline_control']
    groups = {}
    for m, v in inf['models'].items():
        if v['shuffle'] or not os.path.exists(os.path.join(sd, f"things_t{v['target']:02d}.npz")): continue   # pipeline-control targets only
        groups.setdefault((f"{v['candidate']}_{v['mode']}", v['target']), []).append(m)
    res = dict(infer=os.path.relpath(P(a.infer), ROOT), infer_device=inf['device'], layer=inf['layer'], ratio_min=pcfg['ratio_min'], candidates={})
    for cn in sorted({k[0] for k in groups}):
        targets = sorted(t for (x, t) in groups if x == cn); per = {}; cells = {'X63': [], 'X30': []}; Ss = {'X63': [], 'X30': []}; trues = []
        for t in targets:
            z = np.load(os.path.join(sd, f'things_t{t:02d}.npz')); true = z['true']; trues.append(true); ms = groups[(cn, t)]; per[str(t)] = dict(models=ms, n_concepts=int(len(true)))
            for key in ('X63', 'X30'):
                Es = [np.load(os.path.join(P(a.infer), f'emb_{m}.npz'))[f'things_t{t:02d}__{key}'] for m in ms]
                S = unit(np.mean([unit(E) for E in Es], 0)) @ cand.T; Ss[key].append(S); v = rank_pct(S, true); cells[key].append(v); per[str(t)][f'mrp_{key}'] = float(v.mean())
            chk = []
            for m in ms:                                                         # consistency with the training job's own development embeddings
                f = os.path.join(P('results/runs'), m, 'dev_scores.npz')
                if os.path.exists(f):
                    d = np.load(f); E0 = d[f"A_ses4_E_{inf['layer']}"]; E1 = np.load(os.path.join(P(a.infer), f'emb_{m}.npz'))[f'things_t{t:02d}__X63']
                    assert np.array_equal(d['A_ses4_true'], true), f'{m}: evaluated concepts differ from dev_scores.npz'
                    chk.append(dict(model=m, min_cosine=float((unit(E0) * unit(E1)).sum(1).min())))
            per[str(t)]['dev_scores_check'] = chk
        rng = np.random.default_rng(pcfg['perm_seed']); r = dict(targets=targets, per_target=per)
        for key in ('X63', 'X30'):
            obs = float(np.concatenate(cells[key]).mean()); null = np.empty(pcfg['n_perm'])
            for b in range(pcfg['n_perm']): null[b] = np.mean(np.concatenate([rank_pct(S, tr[rng.permutation(len(tr))]) for S, tr in zip(Ss[key], trues)]))
            r[key] = dict(mrp=obs, null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)), excess=float(obs - null.mean()))
        r['ratio'] = r['X30']['excess'] / r['X63']['excess'] if r['X63']['excess'] > 0 else None
        r['passed'] = bool(r['ratio'] is not None and r['ratio'] >= pcfg['ratio_min']); res['candidates'][cn] = r
    return res


def ens(Es):
    """L2-normalise each model's embeddings, average, normalise again (D-113 6.1)"""
    return unit(np.mean([unit(E) for E in Es], 0))


def gao(c, tc, a):
    """D-113 6.3-6.5 on the Gao inputs. Per subject x task, every non-bad trial with an item label from both sessions:
    perception tau = m - 6 s (content control) and imagery (mean of the m + 1, 2, 3 s embeddings) scored against the task's
    K candidate names (cosine; animals / objects: names_text rows, figures: the fv_gao_c_names rows). Balanced accuracy and MRP,
    null = labels permuted within (subject, task) (n_perm, perm_seed), excess = observed - null mean. Combined = per-subject mean
    of the `combined` tasks (animal, object), then subjects resampled; figures alone. Meters (eye, muscle) on the arm b split."""
    from ese.things.data import Meta
    from fv_train import name_rows
    from fv_gao_decode import features, Fit, metrics
    from ese.gao.design import load_trials, splits
    import csv
    sc = c['score']; g = yaml.safe_load(open(P(c['gao']['config']))); sd = P(c['stage_dir'])
    inf = json.load(open(os.path.join(P(a.infer), 'metrics.json')))
    ms = sorted(m for m, v in inf['models'].items() if v['mode'] == sc['mode'] and not v['shuffle']); cands = {inf['models'][m]['candidate'] for m in ms}
    assert ms and len(cands) == 1, f'models of one candidate expected ({cands})'
    # candidates
    pv = json.load(open(P(sc['prompt_index'])))['g0_pass_prompt']; tab = list(csv.DictReader(open(P(c['names']['table'])), delimiter='\t'))
    meta = Meta(P(tc['meta'])); NT = np.load(os.path.join(P(tc['targets']['dir']), f'names_text_{pv}.npy'))
    NG = np.load(os.path.join(P(a.names), f'clip_txt_{pv.lower()}.npy')); assert len(NG) == len(tab)
    th = [i for i, r in enumerate(tab) if r['things_folder']]; rows = name_rows(tc, meta, [tab[i]['things_folder'] for i in th])
    name_check = float((unit(NT[rows]) * unit(NG[th])).sum(1).min())
    emb = {r['gao_item']: unit(NT[rows[th.index(i)]] if r['things_folder'] else NG[i]) for i, r in enumerate(tab)}
    C = {t: np.stack([emb[x] for x in v['items']]) for t, v in g['tasks'].items()}
    # per-trial ensemble embeddings
    Z = {m: np.load(os.path.join(P(a.infer), f'emb_{m}.npz')) for m in ms}; trials = {}
    for f in sorted(x for x in os.listdir(sd) if x.startswith('gao_') and x.endswith('.npz')):
        z = np.load(os.path.join(sd, f)); tag = f[:-4]; sub, ses = z['subject'].item(), z['session'].item()
        E = {w: ens([Z[m][f'{tag}__X30_{w}'] for m in ms]) for w in ('perc', 'im1', 'im2', 'im3')}; E['imag'] = ens([E['im1'], E['im2'], E['im3']])
        Ei = {w: ens([Z[m][f'{tag}__X30_{w}_itemavg'] for m in ms]) for w in ('perc', 'im1', 'im2', 'im3')}; Ei['imag'] = ens([Ei['im1'], Ei['im2'], Ei['im3']])
        keep = (z['bad'] == 0) & (z['label'] >= 0)
        for j in np.flatnonzero(keep): trials[(sub, ses, str(z['task'][j]), int(z['pos'][j]))] = dict(label=int(z['label'][j]), **{w: E[w][j] for w in E})
        for j in range(len(z['itemavg_n'])): trials[(sub, ses, str(z['itemavg_task'][j]), -1 - j)] = dict(label=int(z['itemavg_label'][j]), itemavg=True, **{w: Ei[w][j] for w in Ei})
    rng = np.random.default_rng(sc['perm_seed']); subs = sorted({k[0] for k in trials}); res_rows = []
    for s in subs:
        for t in g['tasks']:
            ks = [k for k, v in trials.items() if k[0] == s and k[2] == t and not v.get('itemavg')]
            if not ks: continue
            y = np.array([trials[k]['label'] for k in ks]); K = len(C[t]); r = dict(subject=s, task=t, K=K, n=len(ks), chance=1 / K)
            for w in ('perc', 'imag', 'im1', 'im2', 'im3'):
                S = np.stack([trials[k][w] for k in ks]) @ C[t].T; r[w] = metrics(S, y, K)
                if w in ('perc', 'imag'): r[f'null_{w}_mean'] = np.mean([metrics(S, y[rng.permutation(len(y))], K) for _ in range(sc['n_perm'])], 0).tolist()
            ki = [k for k, v in trials.items() if k[0] == s and k[2] == t and v.get('itemavg')]; yi = np.array([trials[k]['label'] for k in ki])
            r['itemavg'] = {w: metrics(np.stack([trials[k][w] for k in ki]) @ C[t].T, yi, K) for w in ('perc', 'imag')}; r['itemavg']['n'] = len(ki)
            res_rows.append(r)
    # meters on the arm b split, paired with arm c on the same test trials
    T, runs = load_trials(P(g['bids']), g); Sb = splits(T, g, exclude=True); shift = float(json.load(open(P(os.path.join(g['out'], 'erp_check.json'))))['shift_s'])
    Fd, wn, bn, ch = features(g, shift); key = lambda i: (T['sub'][i], T['ses'][i], T['task'][i], int(T['pos'][i])); name = lambda i: '%s_%s_%s' % runs[T['run'][i]]['key']
    FT = np.stack([Fd[(name(i), int(T['pos'][i]))] for i in range(len(T['label']))])
    def fsel(spec, window):
        bi = [bn.index(b) for b in spec['bands']]; ci = [ch.index(x) for x in spec['channels']]; return FT[:, wn.index(window)][:, bi][:, :, ci].reshape(len(FT), -1)
    met = []
    for k in sorted(x for x in Sb if x[0] == 'b' and x[1] in subs):
        _, s, t = k; tr, te = Sb[k]; K = len(C[t]); yte = T['label'][te]; r = dict(subject=s, task=t, n_test=len(te))
        Sc = np.stack([trials[key(i)]['imag'] for i in te]) @ C[t].T; r['armc'] = metrics(Sc, yte, K)
        for mn, spec in sc['meters'].items():
            r[mn] = metrics(Fit(g, fsel(spec, 'perception')[tr], T['label'][tr]).scores(fsel(spec, 'imagery')[te]), yte, K)
        met.append(r)
    # aggregation
    br = np.random.default_rng(sc['boot']['seed']); out = dict(models=ms, candidate=cands.pop(), prompt=pv, name_check_min_cosine=name_check, rows=res_rows, meter_rows=met)
    def agg(R, tasks, fn):
        ss = sorted({r['subject'] for r in R if r['task'] in tasks}); v = np.array([np.mean([fn(r) for r in R if r['subject'] == x and r['task'] in tasks]) for x in ss]); return ss, v
    B = None; summ = {}
    for grp, tasks in (('combined', sc['combined']), ('figure', ['FVI'])):
        q = {}
        for nm, fn in (('perc_excess', lambda r: r['perc'][0] - r['null_perc_mean'][0]), ('imag_excess', lambda r: r['imag'][0] - r['null_imag_mean'][0]),
                       ('perc_mrp_excess', lambda r: r['perc'][1] - r['null_perc_mean'][1]), ('imag_mrp_excess', lambda r: r['imag'][1] - r['null_imag_mean'][1]),
                       ('imag_acc', lambda r: r['imag'][0]), ('perc_acc', lambda r: r['perc'][0]), ('chance', lambda r: r['chance']),
                       ('im1', lambda r: r['im1'][0]), ('im2', lambda r: r['im2'][0]), ('im3', lambda r: r['im3'][0]),
                       ('itemavg_perc', lambda r: r['itemavg']['perc'][0]), ('itemavg_imag', lambda r: r['itemavg']['imag'][0])):
            ss, q[nm] = agg(res_rows, tasks, fn)
        Bg = [br.integers(0, len(ss), len(ss)) for _ in range(sc['boot']['n'])]
        summ[grp] = dict(subjects=ss, **{k: dict(mean=float(v.mean()), p05=float(np.percentile([v[b].mean() for b in Bg], 5)), p95=float(np.percentile([v[b].mean() for b in Bg], 95))) for k, v in q.items()})
    ss, ac = agg(met, sc['combined'], lambda r: r['armc'][0]); mv = {mn: agg(met, sc['combined'], lambda r, mn=mn: r[mn][0])[1] for mn in sc['meters']}
    Bm = [br.integers(0, len(ss), len(ss)) for _ in range(sc['boot']['n'])]; d = np.array([ac[b].mean() - max(mv[mn][b].mean() for mn in mv) for b in Bm])
    summ['meters'] = dict(subjects=ss, armc_acc=float(ac.mean()), **{mn: float(v.mean()) for mn, v in mv.items()}, armc_minus_max_meter_p05=float(np.percentile(d, 5)))
    cs = summ['combined']; cond = dict(control_perc_excess_p05=cs['perc_excess']['p05'], control=cs['perc_excess']['p05'] > 0,
                                       c1_imag_excess_p05=cs['imag_excess']['p05'], c1=cs['imag_excess']['p05'] > 0,
                                       c2_armc_minus_max_meter_p05=summ['meters']['armc_minus_max_meter_p05'], c2=summ['meters']['armc_minus_max_meter_p05'] > 0)
    if not cond['control']: word = 'undecidable data (perception control failed)'
    elif not cond['c1']: word = 'not detected'
    elif not cond['c2']: word = 'undecidable data (meters)'
    else: word = 'perceptual representation transfer of the 7 trained concepts observed (exploratory, no alpha)'
    out.update(summary=summ, conditions=cond, wording=word); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--part', choices=['pipeline', 'gao'], required=True)
    ap.add_argument('--infer', required=True); ap.add_argument('--names'); ap.add_argument('--tag'); a = ap.parse_args(); t0 = time.time()
    c = yaml.safe_load(open(P(a.config))); tc = yaml.safe_load(open(P(c['train_config'])))
    res = pipeline(c, tc, a) if a.part == 'pipeline' else gao(c, tc, a); res['seconds'] = time.time() - t0
    out = P(os.path.join(c['out'], f"armc_{a.part}{'_' + a.tag if a.tag else ''}.json")); json.dump(res, open(out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    if a.part == 'pipeline': print(json.dumps({k: dict(ratio=v['ratio'], passed=v['passed']) for k, v in res['candidates'].items()}))
    elif not a.tag or a.tag != 'smoke': print(json.dumps(dict(conditions=res['conditions'], wording=res['wording']), default=str))


if __name__ == '__main__':
    main()
