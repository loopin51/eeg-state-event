"""F-VIII evaluation (D-117 amendments 1, 2, 5), server CPU (gpuq, env_nlp).

  --stage a     method A (retrieval): for every condition of lang_rows (main, controls, report, simulations) the pool sentence with
                the largest cos(CLIP text, z); confidence = margin to the second sentence. -> results/fv_lang/a_outputs.jsonl and
                a_summary.json (M1 / M2 / mention / precision / M3 per condition, F-V 1,854-way name top-1 of the same z beside A's M1).
  --stage sim   information-zero simulation of method A (amendment 5): 200 replicates of gauss / const / planted vectors in place
                of z (and an independent gauss draw in place of the label-shuffle z); pass rates of G1 (with A) and G2 (with A),
                power on planted vectors. -> results/fv_lang/sim_a.json. No EEG-derived value is read.
Cells are subject x test row; truth of row k = the names-table row of test concept k (test-code order).

  --stage verdict  once (amendments 1, 2, 5): G1 -> G2 -> G3 with their premises (rule N) and the report-only items from the
                method B outputs (fv_lang_b.py, fetched run folders) and the method C outputs (fv_lang_c.py, fetched run folders).
                -> results/fv_lang/verdict.json (refuses to overwrite).

usage: env_nlp/bin/python scripts/fv_lang_eval.py configs/fv_lang.yaml --stage {a,sim,verdict}
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, csv, glob, json, time, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
from ese.things import lang_rows as R, lang_eval as L
from ese.things.data import Meta

A_CONDS = ['dec|ses123|avg20', 'shuf|ses123|avg20', 'dec|ses123|noise', 'dec|ses123|phase', 'noeeg', 'dec|k0|avg20', 'dec|ses1|avg20',
           'dec|ses12|avg20', 'dec|ses123|single1', 'dec|k0|single1', 'sim_gauss', 'sim_const', 'sim_planted']


def setup(c):
    import spacy; from ese.things.lang import Vocab
    assert spacy.__version__ == c['parser']['spacy']; nlp = spacy.load('en_core_web_lg'); V = Vocab(nlp, P(c['names_table']), P(c['things_meta']))
    names = list(csv.DictReader(open(P(c['names_table']), encoding='utf-8'), delimiter='\t')); tc = yaml.safe_load(open(P(c['c']['train_config'])))
    meta = Meta(P(tc['meta'])); test_row = {x['folder']: i for i, x in enumerate(names) if x['split'] == 'test'}
    tt = np.array([test_row[f] for f in meta.test_img_concept]); split = [x['split'] for x in names]
    return V, names, tt, split


def a_outputs(Z, PE, pool):
    S = Z @ PE.T; o = np.argsort(-S, 1)[:, :2]; return [(pool[i[0]], float(S[r, i[0]] - S[r, i[1]]), int(i[0])) for r, i in enumerate(o)]


def summarize(J, outs, subjects, Etext_rows, test_img, split):
    T = L.table(J, outs, subjects); m3 = None
    if Etext_rows is not None: m3 = float(np.mean([L.m3_cells(Etext_rows[s], test_img, np.arange(200)).mean() for s in subjects]))
    grp = {}
    for (s, k), (sent, _) in outs.items(): g = J.group(sent, split); grp[g] = grp.get(g, 0) + 1
    return dict(m1=float(T['hit'].mean()), m2=float(T['err'].mean()), mention=float((T['hit'] | T['err']).mean()), precision=L.prec(T['hit'], T['err']),
                m3=m3, central_group={k: v / len(outs) for k, v in grp.items()})


def load_b(c):
    """method B outputs of every part -> {cond: {(s, k): (sentence, margin)}}, {cond: {(s, k): CLIP embedding}}"""
    outs, embs = {}, {}
    for part in c['b']['parts']:
        d = sorted(glob.glob(P(f'results/runs/*_fv_lang_b_{part}')))
        if not d: continue
        rows = [json.loads(l) for l in open(os.path.join(d[-1], f'b_{part}.jsonl'))]; E = np.load(os.path.join(d[-1], f'b_{part}_out.npy')).astype(np.float64)
        for r, e in zip(rows, E): outs.setdefault(r['cond'], {})[(r['subject'], r['row'])] = (r['output'], r['margin']); embs.setdefault(r['cond'], {})[(r['subject'], r['row'])] = e
    return outs, embs


def load_c(subjects):
    outs = {}
    for s in subjects:
        for suf in ('', '_shuffle'):
            d = [x for x in sorted(glob.glob(P(f'results/runs/*_fv_lang_c_t{s:02d}{suf}'))) if os.path.exists(os.path.join(x, 'c_out.jsonl'))]
            if not d: continue
            for l in open(os.path.join(d[-1], 'c_out.jsonl')):
                r = json.loads(l); outs.setdefault(r['cond'], {})[(r['subject'], r['row'])] = (r['output'], r['confidence'])
    return outs


def m3_of(embs, subjects, test_img):
    """[ns, 200] M3 cells; a row -1 output (no EEG) is scored against every truth"""
    M = np.zeros((len(subjects), 200))
    for i, s in enumerate(subjects):
        E = np.stack([embs[(s, k)] if (s, k) in embs else embs[(s, -1)] for k in range(200)]); M[i] = L.m3_cells(E, test_img, np.arange(200))
    return M


def verdict(c, J, V, tt, split, subjects, test_img, out, t0):
    from ese.things import verdict as VV
    vf = os.path.join(out, 'verdict.json'); assert not os.path.exists(vf), 'the verdict is computed once'
    B, BE = load_b(c); C = load_c(subjects); rng = np.random.default_rng(20261871); nb = 1000; res = dict(stage='verdict', order='G1 -> G2 -> G3')
    main, shuf = 'dec|ses123|avg20', 'shuf|ses123|avg20'
    # G1: method B, M3(EEG) - M3(label-shuffle)
    M_dec, M_sh = m3_of(BE[main], subjects, test_img), m3_of(BE[shuf], subjects, test_img); Dg1 = M_dec - M_sh
    g1 = VV.rule(VV.pigeonhole(Dg1, nb, rng), Dg1.mean(), 5, 0.05); e_dec = float(M_dec.mean() - 0.5); prem1 = {}
    for cond in ('dec|ses123|noise', 'dec|ses123|phase', 'noeeg'):
        Ec = m3_of(BE[cond], subjects, test_img) - 0.5; prem1[cond] = VV.control_rule(Ec, VV.pigeonhole(Ec, nb, rng), e_dec, 0.1, 5)
    g1['premises'] = prem1; g1['m3'] = dict(eeg=float(M_dec.mean()), shuffle=float(M_sh.mean())); g1['adopted_with_premises'] = bool(g1['adopted'] and all(p['passed'] for p in prem1.values()))
    res['G1'] = g1; go = g1['adopted_with_premises']

    def premise_noeeg(outs_noeeg, outs_dec):
        """rule N on M1 excess and M2 reduction over the exact concept-permutation means, no-EEG vs the decoder condition"""
        def ex(outs):
            T = L.table(J, outs, subjects); H = np.zeros((len(subjects), 200)); Eg = np.zeros_like(H)
            for i, s in enumerate(subjects):
                for k in range(200): ph, pe = J.perm_rates((outs[(s, k)] if (s, k) in outs else outs[(s, -1)])[0]); H[i, k], Eg[i, k] = ph, pe
            return T['hit'] - H, Eg - T['err']
        h0, e0 = ex(outs_noeeg); h1, e1 = ex(outs_dec)
        return dict(m1=VV.control_rule(h0, VV.pigeonhole(h0, nb, rng), h1.mean(), 0.1, 5), m2=VV.control_rule(e0, VV.pigeonhole(e0, nb, rng), e1.mean(), 0.1, 5))
    # G2: method B, central-object error rate, mismatched pairs - EEG, matched mention rates
    TE = L.table(J, B[main], subjects); TM = L.table(J, B[main], subjects, shift=1)
    pt, dr = L.boot_matched(TM, TE, lambda hA, eA, hB, eB: eA.mean() - eB.mean(), nb, rng)
    g2 = dict(point=pt, lower=float(np.percentile(dr, 5)), adopted=bool(np.percentile(dr, 5) > 0), err_eeg=float(TE['err'].mean()), err_mismatch=float(TM['err'].mean()),
              mention_eeg=float((TE['hit'] | TE['err']).mean()), mention_mismatch=float((TM['hit'] | TM['err']).mean()))
    g2['premise_noeeg'] = premise_noeeg(B['noeeg'], B[main]); g2['adopted_with_premises'] = bool(g2['adopted'] and all(v['passed'] for v in g2['premise_noeeg'].values()))
    res['G2'] = dict(g2, tested=go); go = go and g2['adopted_with_premises']
    # G3: precision B - C at matched mention rate
    if main in C:
        TC = L.table(J, C[main], subjects); pt, dr = L.boot_matched(TE, TC, lambda hA, eA, hB, eB: L.prec(hA, eA) - L.prec(hB, eB), nb, rng)
        g3 = dict(point=pt, lower=float(np.percentile(dr, 5)), adopted=bool(np.percentile(dr, 5) > 0), precision_B=L.prec(TE['hit'], TE['err']), precision_C=L.prec(TC['hit'], TC['err']),
                  mention_B=float((TE['hit'] | TE['err']).mean()), mention_C=float((TC['hit'] | TC['err']).mean()))
        if 'noeeg' in C: g3['premise_noeeg_C'] = premise_noeeg(C['noeeg'], C[main])
        g3['premise_noeeg_B'] = g2['premise_noeeg']; g3['adopted_with_premises'] = bool(g3['adopted'] and all(v['passed'] for k in ('premise_noeeg_B', 'premise_noeeg_C') if k in g3 for v in g3[k].values()))
        res['G3'] = dict(g3, tested=go)
    # report only: every method x condition summary
    rep = {}
    for name, D in (('B', B), ('C', C)):
        for cond, outs in D.items():
            T = L.table(J, outs, subjects); grp = {}
            for (s, k), (sent, _) in outs.items(): g = J.group(sent, split); grp[g] = grp.get(g, 0) + 1
            rep[f'{name}|{cond}'] = dict(m1=float(T['hit'].mean()), m2=float(T['err'].mean()), mention=float((T['hit'] | T['err']).mean()), precision=L.prec(T['hit'], T['err']),
                                       central_group={k: v / len(outs) for k, v in grp.items()}, m3=float(m3_of(BE[cond], subjects, test_img).mean()) if name == 'B' and cond in BE else None)
    res['report'] = rep; res['seconds'] = time.time() - t0
    json.dump(res, open(vf, 'w'), indent=1); print(json.dumps({k: res[k] for k in ('G1', 'G2', 'G3') if k in res}, indent=1, default=str)[:4000])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', required=True, choices=['a', 'sim', 'verdict'])
    a = ap.parse_args(); c = yaml.safe_load(open(P(a.config))); t0 = time.time(); out = P('results/fv_lang'); os.makedirs(out, exist_ok=True)
    V, names, tt, split = setup(c); J = L.Judged(V, tt); subjects = c['subjects']
    pool = [r['sentence'] for r in csv.DictReader(open(os.path.join(P(c['pool']['out_dir']), 'pool.tsv'), encoding='utf-8'), delimiter='\t')]
    pd = R.first_dir(c['pool_text']['dirs'], ROOT); PE = np.load(os.path.join(pd, 'pool_text.npy')).astype(np.float64); assert len(PE) == len(pool)
    test_img = np.load(P(c['test_image'])).astype(np.float64); NT = np.load(P(c['names_text'])).astype(np.float64)
    names_test = NT[tt]                                                     # 200 test names in test-code order
    if a.stage == 'verdict': return verdict(c, J, V, tt, split, subjects, test_img, out, t0)
    if a.stage == 'a':
        res, lines = {}, []
        for cond in A_CONDS:
            rows = R.rows(cond, c, ROOT, test_img=test_img, names_test=names_test); Z = np.stack([z for _, _, z in rows]); ao = a_outputs(Z, PE, pool)
            outs = {(s, k): (o[0], o[1]) for (s, k, _), o in zip(rows, ao)}; Er = {}
            for s in subjects:
                idx = [i for i, (ss, k, _) in enumerate(rows) if ss == s]; Er[s] = PE[[ao[i][2] for i in idx]] if len(idx) == 200 else np.repeat(PE[[ao[idx[0]][2]]], 200, 0)
            res[cond] = summarize(J, outs, subjects, Er, test_img, split)
            if cond.startswith(('dec', 'shuf', 'sim_planted')) and len(rows) == 2000:
                res[cond]['fv_name_top1_1854'] = float(np.mean([np.argmax(z @ NT.T) == tt[k] for _, k, z in rows]))
            lines += [dict(cond=cond, subject=s, row=k, output=o[0], margin=o[1], pool_row=o[2]) for (s, k, _), o in zip(rows, ao)]
            print(cond, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[cond].items() if k != 'central_group'}), f'({time.time() - t0:.0f}s)', flush=True)
        with open(os.path.join(out, 'a_outputs.jsonl'), 'w') as f:
            for x in lines: f.write(json.dumps(x, ensure_ascii=False) + '\n')
        json.dump(dict(stage='a', conditions=res, pool_text_dir=pd, seconds=time.time() - t0), open(os.path.join(out, 'a_summary.json'), 'w'), indent=1)
        return
    # information-zero / planted simulation (method A only; no EEG-derived input)
    rng = np.random.default_rng(c['sim']['seed'] + 1); n = 200; nb = 1000; brng = np.random.default_rng(20261871); D = PE.shape[1]
    _, Zp = R.planted(test_img, names_test, c['sim']['planted_target_mrp'], [c['sim']['seed'], 0])
    def gauss(): return R.unit(rng.standard_normal((len(subjects) * 200, D)))
    def const(): return np.repeat(R.unit(rng.standard_normal((len(subjects), D))), 200, 0)
    def plant(): g = rng.standard_normal((len(subjects), 200, D)) / np.sqrt(D); s_ = R.planted(test_img, names_test, c['sim']['planted_target_mrp'], [c['sim']['seed'], 0])[0]; return R.unit((test_img[None] + s_ * g).reshape(-1, D))
    def cells(Z):
        ao = a_outputs(Z, PE, pool); outs = {(s, k): (ao[i * 200 + k][0], ao[i * 200 + k][1]) for i, s in enumerate(subjects) for k in range(200)}
        T = L.table(J, outs, subjects); Tm = L.table(J, outs, subjects, shift=1)
        m3 = np.stack([L.m3_cells(PE[[ao[i * 200 + k][2] for k in range(200)]], test_img, np.arange(200)) for i in range(len(subjects))]); return T, Tm, m3
    res = {}
    for kind, gen in (('gauss', gauss), ('const', const), ('planted', plant)):
        g1 = g2 = 0; d1, d2 = [], []
        for rep in range(n):
            T, Tm, m3 = cells(gen()); _, _, m3s = cells(gauss())
            D1 = m3 - m3s; bd = np.array([D1[np.ix_(brng.integers(0, 10, 10), brng.integers(0, 200, 200))].mean() for _ in range(200)])
            g1 += bool(np.percentile(bd, 5) > 0 and D1.mean() >= 0.05); d1.append(float(D1.mean()))
            pt, dr = L.boot_matched(Tm, T, lambda hA, eA, hB, eB: eA.mean() - eB.mean(), 200, brng); g2 += bool(np.percentile(dr, 5) > 0); d2.append(pt)
            if rep % 20 == 0: print(kind, rep, f'G1 {g1} G2 {g2} ({time.time() - t0:.0f}s)', flush=True)
        res[kind] = dict(replicates=n, g1_pass_rate=g1 / n, g2_pass_rate=g2 / n, g1_diff_mean=float(np.mean(d1)), g2_diff_mean=float(np.mean(d2)))
        print(kind, json.dumps(res[kind]), flush=True)
    json.dump(dict(stage='sim', note='method A; G1 with A in place of B; 200 pigeonhole draws per replicate', results=res, seconds=time.time() - t0),
              open(os.path.join(out, 'sim_a.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
