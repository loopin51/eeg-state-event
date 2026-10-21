"""F-IX (D-119) post-hoc description after the single verdict (not a registered test, report only): paired differences between the
decoder and the linear references (de, time ridge -> the same read-out, their own calibration), on the same subject x concept cells and
the same pigeonhole index set (seed 20261993) as the verdict. Q1 (argmax and probability versions) and Q3 (precision, excess).
Output: results/fv_s/posthoc.json. usage: python scripts/fv_s_posthoc.py configs/fv_s.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import fv_s as S
P = S.P


def main():
    c = yaml.safe_load(open(P(sys.argv[1]))); st = S.Setup(c); st.smoke = False; out = P(c['out_dir'])
    vd = json.load(open(os.path.join(out, 'verdict.json'))); assert vd['dev_sha256'] == S.fsha(os.path.join(out, 'dev.json'))
    R, dv, nm, mcal = S.load_readout(c, st); thr = vd['thresholds_applied']; heads = dv['choice']; T = S.test_truth(st); sub = st.subjects
    catmask = st.has_cat[st.test_name]; B = S.Boot(len(sub), 200, c['n_boot'], c['seeds']['boot'])
    rows = S.LR.rows('dec|ses123|avg20', st.lc, ROOT, sub); Zm = []
    for s in sub:
        r = [x for x in rows if x[0] == s]; assert [x[1] for x in r] == list(range(200)); Zm.append(S.unit(np.stack([x[2] for x in r])))
    cm = S.test_cells(T, S.probs_of(R, heads, Zm), thr)
    rm = json.load(open(os.path.join(out, 'ridge_manifest.json'))); Rr = {}
    for s in sub:
        f_ = P(rm['subjects'][str(s)]['file']); assert S.fsha(f_) == rm['subjects'][str(s)]['sha256']; Rr[s] = np.load(f_)
    res = dict(stage='posthoc', note='descriptive, after the verdict; not part of the registered family', verdict_sha256=S.fsha(os.path.join(out, 'verdict.json')), paired={})
    for mk in ('de', 'time'):
        cl = S.test_cells(T, S.probs_of(R, heads, [S.unit(Rr[s][f'test_{mk}']) for s in sub], {s_: mcal[(mk, s_)] for s_ in S.STAGES}), thr); r = {}
        for q, vk, rk in (('q1', 'value', 'ref'), ('q1p', 'pvalue', 'pref')):
            D = (cm['cat'][vk] - cm['cat'][rk]) - (cl['cat'][vk] - cl['cat'][rk]); bt = B.mean(D, catmask)
            r[q] = dict(diff=float(D[:, catmask].mean()), lo95=float(np.percentile(bt, 2.5)), hi95=float(np.percentile(bt, 97.5)), share_le0=float(np.mean(bt <= 0)))
        def q3(cl_):
            n = cl_['cat']['said'].astype(float) * catmask[None]; return cl_['cat']['hit'] * n, cl_['cat']['hitref'] * n, n
        h1, p1, n1 = q3(cm); h2, p2, n2 = q3(cl)
        pr = B.ratio(h1, n1) - B.ratio(h2, n2); ex = (B.ratio(h1, n1) - B.ratio(p1, n1)) - (B.ratio(h2, n2) - B.ratio(p2, n2))
        r['q3_precision'] = dict(diff=float(h1.sum() / n1.sum() - h2.sum() / n2.sum()), lo95=float(np.percentile(pr, 2.5)), hi95=float(np.percentile(pr, 97.5)))
        r['q3_excess'] = dict(diff=float((h1.sum() - p1.sum()) / n1.sum() - (h2.sum() - p2.sum()) / n2.sum()), lo95=float(np.percentile(ex, 2.5)), hi95=float(np.percentile(ex, 97.5)))
        r['q3_cells_said'] = dict(decoder=float((n1 > 0).mean()), reference=float((n2 > 0).mean()))
        res['paired'][mk] = r
    json.dump(res, open(os.path.join(out, 'posthoc.json'), 'w'), indent=1); print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
