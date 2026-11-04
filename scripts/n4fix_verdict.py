"""D-129 verdict (docs/09_vera_n4_plan.md §5; rules in configs/n4fix_verdict.yaml, registered before any D-129 run).

usage: env/bin/python scripts/n4fix_verdict.py configs/n4fix_verdict.yaml --stage 1|2 [--check]

--stage 1  N4 sub-step factorial on the SB clean conditions (seed 0) and the stage-2 selection; writes
           results/n4fix/stage1.json and selection.json (read by configs/n4fix_grid.yaml to list the s2 runs). Once.
--stage 2  H1 retest: relative shortcut reliance of VERA with the selected head vs EEGNet, REVE-Official (D-128 runs) and
           the same head on REVE without N1-N3; writes results/n4fix/stage2.json. Once.
--check    lists the runs the stage needs and which are missing; reads no score.

Metrics come from preds.npz of the run folders (probabilities over the K task classes; the null column is ignored), with
the D-128 subject bootstrap (scripts/nc_verdict.py: one draw shared by every run, seed means on the same draw).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, glob, argparse
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from nc_grid import folder
from nc_verdict import load, Boot, lower, interval, p_le, holm

T = 'physionet_mi'


class Runs:
    def __init__(self, v):
        self.rd, self.v, self.cache, self.boot, self.needed = os.path.join(ROOT, v['runs_dir']), v, {}, None, []

    def get(self, arm, tag, sb=None, shuffle=False, seed=0):
        r = dict(task=T, sb=sb, arm=arm, shuffle=shuffle, seed=seed, tag=tag); k = folder(r)
        if k not in self.cache: self.needed.append(k[1:]); self.cache[k] = None if CHECK else load(self.rd, r)
        return self.cache[k]

    def stat(self, arm, tag, sb=None, shuffle=False, seeds=(0,)):
        rs = [self.get(arm, tag, sb, shuffle, s) for s in seeds]
        if CHECK: return None
        assert all(r is not None for r in rs), f'missing run: {arm}{tag} {sb} shuffle={shuffle} seeds={seeds}'
        if self.boot is None: self.boot = Boot(len(rs[0]['C']), self.v['n_boot'], [self.v['boot_seed'], sum(map(ord, T))])
        ks, bs = zip(*[self.boot(r) for r in rs])
        return dict(kappa=np.mean(ks, 0), bal_acc=np.mean(bs, 0), runs=[r['run'] for r in rs])

    def g0(self, arm, tag, sb):
        m = self.stat(arm, tag, sb, True)
        if m is None: return None
        lo, hi = interval(m['kappa'], self.v['g0']['level'])
        return dict(kappa=float(m['kappa'][0]), ci99=[lo, hi], run=m['runs'][0], passed=bool(lo <= 0 <= hi or hi < self.v['g0']['upper']))


def summ(x, al):
    return dict(est=float(x[0]), lower95=lower(x, al), ci95=interval(x, 0.95))


def grid_tag(v): return yaml.safe_load(open(os.path.join(ROOT, v['grid'])))['tag']


def stage1(v, R):
    s, al, sig = v['stage1'], v['alpha'], v['signals']; tg, tn = v['d128_tag'], grid_tag(v)
    arms = {'base': (s['base'], tg), **{k: (a, tn) for k, a in s['variants'].items()}}
    kap = {k: {x: R.stat(a, t, f'{x}_clean') for x in sig} for k, (a, t) in arms.items()}
    g0 = {k: {x: R.g0(a, t, f'{x}_clean') for x in sig} for k, (a, t) in arms.items()}
    if CHECK: return None
    kap = {k: {x: m['kappa'] for x, m in d.items()} for k, d in kap.items()}
    out = dict(arms={k: f'{a}{t}' for k, (a, t) in arms.items()}, kappa_clean={}, G0=g0, learned={}, tests={})
    for k in arms:
        out['kappa_clean'][k] = {x: summ(kap[k][x], al) for x in sig}
        out['learned'][k] = bool(all(lower(kap[k][x], al) >= v['learned_kappa'] for x in sig) and all(g['passed'] for g in g0[k].values()))
    cell = lambda c: s['cells'].get(c, c)
    ps, names = [], []
    for name, t in s['tests'].items():
        d = np.mean([kap[cell(c)][t['signal']] for c in t['plus']], 0) - np.mean([kap[cell(c)][t['signal']] for c in t['minus']], 0)
        out['tests'][name] = dict(**summ(d, al), p=p_le(d, 0.0)); ps.append(out['tests'][name]['p']); names.append(name)
    for n, ok in zip(names, holm(ps, al)): out['tests'][n]['holm_reject'] = bool(ok)
    ok = [k for k in s['variants'] if out['learned'][k]]
    score = {k: round(float(np.mean([kap[k][x][0] for x in sig])), 3) for k in s['variants']}
    sel = sorted(ok, key=lambda k: (-score[k], s['tie_order'].index(k)))[0] if ok else None
    out['selection'] = dict(selected=sel, mean_kappa_clean=score, eligible=ok,
                            rule='learned on both signals and G0 passed; max mean kappa_clean (3 decimals); ties by tie_order')
    out['statements'] = dict(
        channel_aggregation_blocks_induced=out['tests']['T1_agg_ind']['holm_reject'],
        patch_pooling_blocks_evoked=out['tests']['T2_pool_ev']['holm_reject'],
        stage2='run with ' + sel if sel else 'not run: no variant learns both signals')
    return out


def stage2(v, R, sel):
    s, al, sig = v['stage2'], v['alpha'], v['signals']; tg, tn = v['d128_tag'], grid_tag(v)
    models = {'vera_sel': (s['vera'].format(sel=sel), tn), 'reve_head_sel': (s['reve_head'].format(sel=sel), tn),
              **{o: (o, tg) for o in s['others']}}
    seeds = s['seeds']; st = {}
    for k, (a, t) in models.items():
        st[k] = {x: (R.stat(a, t, f'{x}_clean', seeds=seeds), R.stat(a, t, f"{x}_{s['shortcut']}", seeds=seeds)) for x in sig}
    g0 = {k: {x: R.g0(a, t, f'{x}_clean') for x in sig} for k, (a, t) in models.items()}
    rep = {k: (R.stat(a, t), R.g0(a, t, None)) for k, (a, t) in
           {'vera_sel': models['vera_sel'], 'vera': ('reve_nc', tg), 'reve_official': ('reve_official', tg)}.items()}
    if CHECK: return None
    out = dict(selected=sel, models={k: f'{a}{t}' for k, (a, t) in models.items()}, seeds=seeds, by_model={}, G0=g0, tests={})
    Rrel, Rabs, kc = {}, {}, {}
    for k in models:
        rel = [(c['kappa'] - sc['kappa']) / c['kappa'] for c, sc in st[k].values()]
        ab = [c['kappa'] - sc['kappa'] for c, sc in st[k].values()]
        Rrel[k], Rabs[k], kc[k] = np.mean(rel, 0), np.mean(ab, 0), np.mean([c['kappa'] for c, _ in st[k].values()], 0)
        out['by_model'][k] = dict(R_rel=summ(Rrel[k], al), R_abs=summ(Rabs[k], al), kappa_clean=summ(kc[k], al),
                                  by_signal={x: dict(kappa_clean=summ(c['kappa'], al), kappa_shortcut=summ(sc['kappa'], al),
                                                     R_rel=summ((c['kappa'] - sc['kappa']) / c['kappa'], al))
                                             for x, (c, sc) in st[k].items()})
    learned = {f'{k}|{x}': bool(lower(c['kappa'], al) >= v['learned_kappa']) for k in models for x, (c, _) in st[k].items()}
    g_ok = all(g['passed'] for d in g0.values() for g in d.values())
    comp = ['eegnet', 'reve_official', 'reve_head_sel']; ps = []
    for o in comp:
        d = Rrel[o] - Rrel['vera_sel']
        out['tests'][o] = dict(**summ(d, al), ci90=interval(d, 0.90), p=p_le(d, 0.0)); ps.append(out['tests'][o]['p'])
    for o, ok in zip(comp, holm(ps, al)): out['tests'][o]['holm_reject'] = bool(ok)
    valid = all(learned.values()) and g_ok
    out['validity'] = dict(learned=learned, G0_all_passed=g_ok, valid=valid)
    lo90, hi90 = out['tests']['reve_head_sel']['ci90']; e = s['sesoi_rel']
    out['statements'] = dict(
        vera_relies_less_than_comparisons=('판정 불가' if not valid else
                                           '통과' if out['tests']['eegnet']['holm_reject'] and out['tests']['reve_official']['holm_reject'] else '불통과'),
        n1_n3_effect=('판정 불가' if not valid else '효과 있음' if out['tests']['reve_head_sel']['holm_reject'] else
                      f'효과 없음({e} 이상 배제)' if -e < lo90 and hi90 < e else '판정 보류(구간이 넓음)'))
    # D-128 absolute R with validity (b), reported next to the primary result
    out['absolute_R_D128_rule'] = {o: dict(diff=summ(Rabs[o] - Rabs['vera_sel'], al), p=p_le(Rabs[o] - Rabs['vera_sel'], 0.0),
                                           b_ok=bool(kc['vera_sel'][0] >= kc[o][0] - s['validity_margin_abs'])) for o in comp}
    # report only: real PhysioNet-MI labels, seed 0
    out['report_real_labels'] = {k: dict(bal_acc=summ(m['bal_acc'], al), kappa=summ(m['kappa'], al), run=m['runs'][0], G0=g)
                                 for k, (m, g) in rep.items()}
    out['report_real_labels']['diff_bal_acc'] = {
        'vera_sel - vera': summ(rep['vera_sel'][0]['bal_acc'] - rep['vera'][0]['bal_acc'], al),
        'vera_sel - reve_official': summ(rep['vera_sel'][0]['bal_acc'] - rep['reve_official'][0]['bal_acc'], al)}
    return out


def main():
    global CHECK
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', type=int, required=True, choices=[1, 2])
    ap.add_argument('--check', action='store_true'); a = ap.parse_args(); CHECK = a.check
    v = yaml.safe_load(open(os.path.join(ROOT, a.config))); od = os.path.join(ROOT, v['out_dir']); R = Runs(v)
    out_path = os.path.join(od, f'stage{a.stage}.json')
    if not CHECK: assert not os.path.exists(out_path), f'{out_path} exists: each stage is run once'
    if a.stage == 1: res = stage1(v, R)
    else:
        sp = os.path.join(od, 'selection.json'); assert os.path.exists(sp), 'stage 2 needs the stage-1 selection'
        sel = json.load(open(sp))['selected']; assert sel, 'stage 1 selected no variant: stage 2 is not run'
        res = stage2(v, R, sel)
    if CHECK:
        miss = [k for k in R.needed if not glob.glob(os.path.join(R.rd, '*_' + k, 'preds.npz'))]
        print(json.dumps(dict(needed=len(R.needed), missing=len(miss), runs_missing=miss), indent=1)); return
    os.makedirs(od, exist_ok=True); res['n_boot'] = v['n_boot']
    json.dump(res, open(out_path, 'w'), indent=1, ensure_ascii=False, default=float)
    if a.stage == 1: json.dump(res['selection'], open(os.path.join(od, 'selection.json'), 'w'), indent=1)
    print(json.dumps(res.get('statements'), ensure_ascii=False, indent=1))


CHECK = False
if __name__ == '__main__':
    main()
