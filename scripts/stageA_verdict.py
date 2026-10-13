"""Stage A verdicts from finished run folders (D-072 pass/border/fail, D-077 SA1 excess over the phase-randomised
baseline and SA2 within-run report, D-074 row). Subject bootstrap resamples subjects of the seed-mean values.

usage: python scripts/stageA_verdict.py configs/stageA_verdict.yaml
"""
import os, sys, json, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ('top1', 'top5', 'top10', 'rank_pct', 'run_top1', 'run_top10', 'run_rank_pct')


def load(c, name):
    m = json.load(open(os.path.join(ROOT, c['runs_root'], name, 'metrics.json')))
    return {int(s): v for s, v in m['subjects'].items()}


def main(cfg):
    c = yaml.safe_load(open(cfg)); rng = np.random.default_rng(c['seed'])
    per = {v: [load(c, r) for r in rs] for v, rs in c['runs'].items()}; L = load(c, c['phaserand'])
    subs = sorted(per['clip_w2v'][0]); pools = {'19': subs, '17': [s for s in subs if s not in c['pool_exclude']]}
    out = {'n_boot': c['n_boot'], 'pools': {}}
    for pname, ps in pools.items():
        idx = np.array([subs.index(s) for s in ps]); Bp = rng.integers(0, len(ps), (c['n_boot'], len(ps)))
        sm = {v: {k: np.array([np.mean([r[s][k] for r in runs]) for s in subs]) for k in KEYS} for v, runs in per.items()}   # seed mean per subject
        seeds = {v: [float(np.mean([r[s]['top10'] for s in ps])) for r in runs] for v, runs in per.items()}
        lr = {k: np.array([L[s][k] for s in subs]) for k in KEYS}
        def ci(x): x = x[idx]; b = x[Bp].mean(1); return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        res = {v: {k: ci(sm[v][k]) for k in KEYS} for v in per}; res['phaserand'] = {k: ci(lr[k]) for k in KEYS}
        m = {v: res[v]['top10'][0] for v in per}
        order = {'base_mel<clip_mel': m['base_mel'] < m['clip_mel'], 'clip_mel<clip_w2v': m['clip_mel'] < m['clip_w2v'],
                 'nosubj<clip_w2v': m['clip_w2v_nosubj'] < m['clip_w2v']}
        x = sm['clip_w2v']['top10'][idx]; bx = x[Bp]
        boot_order = {'base_mel<clip_mel': float(np.mean(sm['base_mel']['top10'][idx][Bp].mean(1) < sm['clip_mel']['top10'][idx][Bp].mean(1))),
                      'clip_mel<clip_w2v': float(np.mean(sm['clip_mel']['top10'][idx][Bp].mean(1) < bx.mean(1))),
                      'nosubj<clip_w2v': float(np.mean(sm['clip_w2v_nosubj']['top10'][idx][Bp].mean(1) < bx.mean(1)))}
        ok = all(order.values()); hi = res['clip_w2v']['top10'][2]
        d072 = 'pass' if m['clip_w2v'] >= c['pass_top10'] and ok else 'fail' if hi < c['pass_top10'] or not ok else 'border'
        E = (sm['clip_w2v']['top10'] - lr['top10'])[idx]; Eb = E[Bp].mean(1); Eci = [float(E.mean()), float(np.percentile(Eb, 2.5)), float(np.percentile(Eb, 97.5))]
        sa1 = 'pass' if Eci[0] >= c['sa1_excess'] and ok else 'fail' if Eci[2] < c['sa1_excess'] or not ok else 'border'
        row = 'present' if m['clip_w2v'] > res['base_mel']['top10'][2] else 'absent'
        out['pools'][pname] = {'n_subjects': len(ps), 'seed_mean_ci': res, 'per_seed_top10': seeds, 'order': order, 'boot_order_freq': boot_order,
                               'D072': d072, 'SA1_excess_ci': Eci, 'SA1': sa1, 'D074_row': row}
        print(pname, json.dumps({k: out['pools'][pname][k] for k in ('D072', 'SA1', 'SA1_excess_ci', 'D074_row', 'order', 'boot_order_freq')}), flush=True)
    p = os.path.join(ROOT, c['out']); os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(out, open(p, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
