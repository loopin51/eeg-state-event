"""Early-stopping cause check of the nine P2 pretraining runs (D-104 decision 7 (b); descriptive, no alpha).

Reads results/runs/*_c1_P2_pretrain_s*_{hours,sqrt,mixed}/history.json only (no training, no GPU). The decision rule
(r = post-best mean rise / pre-best 600-update improvement, f_pre, post-best slope, train-loss change; thresholds 0.25 and
0.10) was committed before the trajectories were read (reports/05_fii_stage/notes/pretrain_history_kr.md section 1, 6a854ef).
Output: results/fii_stage/pretrain_history.json and reports/05_fii_stage/figures/fig6_pretrain_history.{pdf,png}.

usage: python scripts/fii_pretrain_history.py
"""
import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R_SAT, F_PAT, WIN = 0.25, 0.10, 600
CANDS = ('hours', 'sqrt', 'mixed')
DATASETS = ('alice', 'sparrkulee', 'kul_aad')


def stats(u, v, t, b):
    """Rule quantities for one loss trajectory v (with train loss t, may be None) and best update b."""
    i = {x: k for k, x in enumerate(u)}; ib = i[b]
    pre = i.get(b - WIN, 0); post = [k for k in range(ib + 1, len(u)) if u[k] <= b + WIN]
    I_pre = v[pre] - v[ib]; tot = v[0] - v[ib]
    R_post = float(np.mean([v[k] for k in post])) - v[ib] if post else float('nan')
    A_post = float(np.ptp([v[k] for k in post])) if post else float('nan')
    slope = float(np.polyfit([u[k] for k in post], [v[k] for k in post], 1)[0]) if len(post) >= 2 else float('nan')
    out = dict(best_update=b, v0=v[0], v_best=v[ib], pre_update=u[pre], I_pre=I_pre, f_pre=I_pre / tot if tot > 0 else float('nan'),
               R_post=R_post, A_post=A_post, r=R_post / I_pre if I_pre > 0 else float('inf'), slope_post_per_1k=slope * 1000,
               n_post=len(post))
    if t is not None and post:
        out.update(dT_post=t[post[-1]] - t[ib], gap_best=v[ib] - t[ib], gap_end=v[post[-1]] - t[post[-1]])
    return out


def label(s):
    if s['r'] >= R_SAT and s['slope_post_per_1k'] > 0 and s.get('dT_post', -1) < 0: return 'H-sat'
    if s['r'] < R_SAT and s['f_pre'] >= F_PAT: return 'H-pat'
    if s['r'] < R_SAT and s['f_pre'] < F_PAT: return 'flat'
    return 'unclear'


def overall(labels, need):
    n = {k: labels.count(k) for k in ('H-sat', 'H-pat', 'flat', 'unclear')}
    for k, c in n.items():
        if c >= need: return k, n
    if n['H-sat'] + n['flat'] >= need and n['H-pat'] <= 2: return 'not patience (saturation or convergence)', n
    return 'undecided', n


def main():
    runs = []
    for d in sorted(glob.glob(os.path.join(ROOT, 'results/runs/*_c1_P2_pretrain_s20*_*'))):
        c = d.rsplit('_', 1)[1]
        if c not in CANDS: continue
        h = json.load(open(os.path.join(d, 'history.json'))); m = json.load(open(os.path.join(d, 'metrics.json')))
        u = [e['update'] for e in h]; v = [e['valid'] for e in h]; t = [e['train'] for e in h]
        b = u[int(np.argmin(v))]; assert b == m['best_update'], (d, b, m['best_update'])
        assert u[-1] == m['updates'] == b + WIN, (d, u[-1], m['updates'])
        s = stats(u, v, t, b); s['label'] = label(s)
        per = {}
        for ds in DATASETS:
            vd = [e['valid_per_dataset'][ds] for e in h]; bd = u[int(np.argmin(vd))]
            per[ds] = stats(u, vd, None, bd); per[ds]['rise_at_end'] = vd[-1] - min(vd)
        runs.append(dict(folder=os.path.basename(d), candidate=c, seed=m['seed'], combined=s, per_dataset=per,
                         traj=dict(update=u, valid=v, train=t, per={ds: [e['valid_per_dataset'][ds] for e in h] for ds in DATASETS})))
    assert len(runs) == 9, len(runs)
    labs = [r['combined']['label'] for r in runs]; hl = [r['combined']['label'] for r in runs if r['candidate'] == 'hours']
    res = dict(rule=dict(r_sat=R_SAT, f_pat=F_PAT, window=WIN, committed='6a854ef'),
               overall_all=overall(labs, 6), overall_hours=overall(hl, 4),
               runs=[{k: v for k, v in r.items() if k != 'traj'} for r in runs])
    os.makedirs(os.path.join(ROOT, 'results/fii_stage'), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, 'results/fii_stage/pretrain_history.json'), 'w'), indent=1)
    for r in runs:
        s = r['combined']
        print(f"{r['candidate']:5s} {r['seed']} b={s['best_update']:5d} I_pre={s['I_pre']:.4f} f_pre={s['f_pre']:.3f} R_post={s['R_post']:+.4f} "
              f"A_post={s['A_post']:.4f} r={s['r']:+.3f} slope/1k={s['slope_post_per_1k']:+.4f} dT={s['dT_post']:+.4f} -> {s['label']}  "
              + ' '.join(f"{ds}:b{p['best_update']} r{p['r']:+.2f}" for ds, p in r['per_dataset'].items()))
    print('overall all', res['overall_all'], 'hours', res['overall_hours'])
    if '--no-fig' not in sys.argv: figure(runs)


def figure(runs):
    sys.path.insert(0, os.path.join(ROOT, 'reports/05_fii_stage'))
    import style
    import matplotlib.pyplot as plt
    style.apply()
    col = dict(hours=style.TERTIARY, sqrt=style.PRIMARY, mixed=style.SECONDARY)
    fig, ax = plt.subplots(1, 4, figsize=(style.FULL, 4.6 * style.CM), sharex=True)
    names = [('combined', '결합 검증 손실'), ('alice', 'Alice'), ('sparrkulee', 'SparrKULee'), ('kul_aad', 'KU Leuven AAD')]
    for j, (k, title) in enumerate(names):
        a = ax[j]
        for r in runs:
            u = np.array(r['traj']['update']); y = np.array(r['traj']['valid'] if k == 'combined' else r['traj']['per'][k])
            b = r['combined']['best_update'] if k == 'combined' else r['per_dataset'][k]['best_update']
            a.plot(u, y, color=col[r['candidate']], lw=0.8, alpha=0.9)
            a.plot([b], [y[list(u).index(b)]], 'o', ms=2.4, color=col[r['candidate']])
        a.axhline(np.log(256), color=style.CHANCE, lw=0.6, ls='--', zorder=0)  # chance = ln(batch 256)
        a.set_title(title); a.set_xlabel('갱신 수'); a.grid(True, axis='y')
    ax[0].set_ylabel('검증 CLIP 손실')
    for c in CANDS: ax[0].plot([], [], color=col[c], lw=1.0, label=c)
    ax[0].plot([], [], color=style.CHANCE, lw=0.6, ls='--', label='우연 ln 256')
    ax[0].legend(loc='lower left')
    style.save(fig, os.path.join(ROOT, 'reports/05_fii_stage/figures/fig6_pretrain_history.pdf'))


if __name__ == '__main__':
    main()
