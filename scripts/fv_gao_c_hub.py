"""D-113 arm c, report-only diagnostic after the result (300b3f6, wording 1 unchanged; coordination session 2026-10-17).
Distribution of the top-1 predicted name of the arm c ensemble on the Gao trials, per subject x task x window (perception
tau = m - 6 s, imagery = ensemble of m + 1, 2, 3 s), on the same trials and embeddings as fv_gao_c_score.py --part gao:
  closed: top-1 among the task's K candidate names (the scored set): most frequent item and its share, counts per item,
  open  : top-1 among all 1,854 THINGS names (names_text of the G0 prompt): most frequent name and its share, and the share of
          the hub name seen on THINGS inputs (--hub, juicer: fv_decode_runbook_kr.md 3, D-112 result).
Item-average rows (report-only in the score) are counted the same way. No accuracy, no null, no verdict.
Output: results/gao_imagery/armc_hub.json
usage (gpuq): env/bin/python -I scripts/fv_gao_c_hub.py configs/fv_gao_c.yaml --infer results/runs/<infer run> --names results/runs/<names run> [--hub juicer]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import argparse, collections, csv, json, sys, time
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
from fv_gao_c_score import unit, ens


def dist(names):
    n = collections.Counter(names); top, k = n.most_common(1)[0]
    return dict(n=len(names), top=top, top_share=k / len(names), n_distinct=len(n), counts=dict(n.most_common()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--infer', required=True); ap.add_argument('--names', required=True)
    ap.add_argument('--hub', default='juicer'); a = ap.parse_args(); t0 = time.time()
    from ese.things.data import Meta
    from fv_train import name_rows
    c = yaml.safe_load(open(P(a.config))); tc = yaml.safe_load(open(P(c['train_config']))); sc = c['score']; g = yaml.safe_load(open(P(c['gao']['config']))); sd = P(c['stage_dir'])
    inf = json.load(open(os.path.join(P(a.infer), 'metrics.json')))
    ms = sorted(m for m, v in inf['models'].items() if v['mode'] == sc['mode'] and not v['shuffle']); assert ms and len({inf['models'][m]['candidate'] for m in ms}) == 1
    pv = json.load(open(P(sc['prompt_index'])))['g0_pass_prompt']; tab = list(csv.DictReader(open(P(c['names']['table'])), delimiter='\t'))
    meta = Meta(P(tc['meta'])); NT = np.load(os.path.join(P(tc['targets']['dir']), f'names_text_{pv}.npy')); NG = np.load(os.path.join(P(a.names), f'clip_txt_{pv.lower()}.npy'))
    th = [i for i, r in enumerate(tab) if r['things_folder']]; rows = name_rows(tc, meta, [tab[i]['things_folder'] for i in th])
    emb = {r['gao_item']: unit(NT[rows[th.index(i)]] if r['things_folder'] else NG[i]) for i, r in enumerate(tab)}
    items = {t: list(v['items']) for t, v in g['tasks'].items()}; C = {t: np.stack([emb[x] for x in v]) for t, v in items.items()}
    allnames = [r['display_name'] for r in csv.DictReader(open(P(tc['targets']['names_table'])), delimiter='\t')]; NTu = unit(NT.astype(np.float64)); assert len(allnames) == len(NT)
    hub_rows = [i for i, x in enumerate(allnames) if x == a.hub]; assert hub_rows, f'hub name {a.hub} not in the names table'
    Z = {m: np.load(os.path.join(P(a.infer), f'emb_{m}.npz')) for m in ms}; cells = collections.defaultdict(lambda: dict(closed=[], open=[]))
    for f in sorted(x for x in os.listdir(sd) if x.startswith('gao_') and x.endswith('.npz')):
        z = np.load(os.path.join(sd, f)); tag = f[:-4]; sub = z['subject'].item()
        for suf, sel in (('', (z['bad'] == 0) & (z['label'] >= 0)), ('_itemavg', None)):
            E = {w: ens([Z[m][f'{tag}__X30_{w}{suf}'] for m in ms]) for w in ('perc', 'im1', 'im2', 'im3')}; E['imag'] = ens([E['im1'], E['im2'], E['im3']])
            task = z['task'] if suf == '' else z['itemavg_task']; idx = np.flatnonzero(sel) if suf == '' else np.arange(len(z['itemavg_n']))
            for w in ('perc', 'imag'):
                Ow = (E[w][idx] @ NTu.T).argmax(1)
                for j, o in zip(idx, Ow):
                    t = str(task[j]); k = (sub, t, w + suf)
                    cells[k]['closed'].append(items[t][int(np.argmax(C[t] @ E[w][j]))]); cells[k]['open'].append(allnames[o])
    out = dict(models=len(ms), prompt=pv, hub=a.hub, cells=[]); opens = collections.defaultdict(list)
    for k in sorted(cells):
        s, t, w = k; cl = dist(cells[k]['closed']); op = dist(cells[k]['open']); op['hub_share'] = cells[k]['open'].count(a.hub) / len(cells[k]['open'])
        out['cells'].append(dict(subject=s, task=t, window=w, closed=cl, open=op)); opens[(t, w)] += cells[k]['open']
    summ = {}
    for t in items:
        for w in ('perc', 'imag', 'perc_itemavg', 'imag_itemavg'):
            R = [x for x in out['cells'] if x['task'] == t and x['window'] == w]
            if not R: continue
            ov = collections.Counter(opens[(t, w)]); nn = len(opens[(t, w)])
            summ[f'{t}|{w}'] = dict(subjects=len(R), K=len(items[t]), closed_top_share_mean=float(np.mean([x['closed']['top_share'] for x in R])),
                                    closed_top_share_min=float(min(x['closed']['top_share'] for x in R)),
                                    closed_cells_single_item=int(sum(x['closed']['n_distinct'] == 1 for x in R)),
                                    closed_top_item_over_subjects=dict(collections.Counter(x['closed']['top'] for x in R).most_common()),
                                    open_top_share_mean=float(np.mean([x['open']['top_share'] for x in R])),
                                    open_top10_pooled={n: v / nn for n, v in ov.most_common(10)}, open_hub_share_pooled=ov[a.hub] / nn,
                                    open_top_name_over_subjects=dict(collections.Counter(x['open']['top'] for x in R).most_common()))
    out.update(summary=summ, seconds=time.time() - t0)
    f = P(os.path.join(c['out'], 'armc_hub.json')); json.dump(out, open(f, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(json.dumps({k: dict(closed_mean=round(v['closed_top_share_mean'], 3), single=v['closed_cells_single_item'], hub=round(v['open_hub_share_pooled'], 3),
                              open_top=list(v['open_top10_pooled'])[:3]) for k, v in summ.items()}, indent=1), flush=True)


if __name__ == '__main__':
    main()
