"""D-084 exploratory analysis (chosen after the A4 results; never used for a verdict): does block-boundary adjacency
explain part of the A4 stage A score? Re-evaluates finished models from best.pt (no training) with the same ranking as
stageA_train.evaluate, keeps per-segment ranks, and recomputes top-10 / within-run top-10 on test segments whose EEG
window stays at least `margin` seconds from every boundary between a test span and a non-test block of the same run.
Candidates are always all test segments (only the query set shrinks). Intervals are conditional subject bootstraps
(models and test segments fixed), as in stageA_verdict.py.

usage: python scripts/stageA_boundary_eval.py configs/stageA_boundary_eval.yaml
"""
import os, sys
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(v, os.environ.get('ESE_NUM_THREADS', '8'))
import json, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageA_train import Data, VARIANTS, spans, BrainModule, clip_scores


def ranks(net, D, dev, c):
    """per-segment rank among all test candidates and among the query's run (same code path as stageA_train.evaluate)."""
    net.eval(); items = D.items['test']; n = len(items)
    Y = torch.from_numpy(np.stack([D.feat(r, t) for r, t in items])).to(dev, torch.float16)
    run = torch.tensor([r for r, _ in items], device=dev); RK, RR = [], []
    with torch.no_grad():
        for si in range(len(D.subjects)):
            rk, rr = [], []
            for a in range(0, n, c['eval']['score_chunk']):
                idx = list(range(a, min(n, a + c['eval']['score_chunk']))); X, P, S, _ = D.batch('test', idx, [si] * len(idx))
                Z = net(X.to(dev), P.to(dev), S.to(dev)).float()
                sc = torch.cat([clip_scores(Z, Y[b:b + 512].float()) for b in range(0, n, 512)], 1)
                above = sc > sc[torch.arange(len(idx)), torch.tensor(idx)][:, None]
                rk.append(above.sum(1).cpu()); rr.append((above & (run[idx][:, None] == run[None])).sum(1).cpu())
            RK.append(torch.cat(rk).numpy()); RR.append(torch.cat(rr).numpy())
    return np.stack(RK), np.stack(RR)


def boundary_distance(D):
    """seconds from each test segment's EEG window to the nearest test / non-test block boundary of its run (inf if none)."""
    w, off = D.c['window'], D.c['eeg']['offset_s']; out = []
    for r, t in D.items['test']:
        b = D.B[D.B.run == r]; r0, r1 = b.start.min(), b.stop.max(); ws, we = t + w['tmin'] + off, t + w['tmax'] + off
        sp = [s for s in spans(D.B, 'test', r) if s[0] - 1e-6 <= t + w['tmin'] and t + w['tmax'] <= s[1] + 1e-6][0]
        d = [ws - sp[0]] if abs(sp[0] - r0) > 1e-6 else []
        d += [sp[1] - we] if abs(sp[1] - r1) > 1e-6 else []
        out.append(min(d) if d else np.inf)
    return np.array(out)


def main(cfg):
    c = yaml.safe_load(open(cfg)); od = os.path.join(ROOT, c['out_dir']); os.makedirs(od, exist_ok=True); dev = 'cuda'; data = {}
    for group, rs in c['runs'].items():
        for r in rs:
            p = os.path.join(od, f'{r}.npz')
            if os.path.exists(p): continue
            rc = yaml.safe_load(open(os.path.join(ROOT, c['runs_root'], r, 'config.yaml'))); fk, _, sl = VARIANTS[rc['variant']]
            key = (rc['eeg'].get('source', 'eeg'), fk)   # registered runs predate the key; Data uses the same default
            if key not in data:
                D = net = None; data.clear(); torch.cuda.empty_cache(); data[key] = Data(rc, fk); data[key].to_device(dev)
            D = data[key]; D.c = rc; m = rc['model']
            net = BrainModule(out_dim=D.F.shape[0], n_subjects=len(D.subjects), merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                              merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'],
                              depth=m['depth'], subject_layers=sl).to(dev)
            net.load_state_dict(torch.load(os.path.join(ROOT, c['runs_root'], r, 'best.pt'), map_location=dev))
            rk, rr = ranks(net, D, dev, rc); run = np.array([x for x, _ in D.items['test']])
            n_run = (run[:, None] == run[None]).sum(1)
            met = json.load(open(os.path.join(ROOT, c['runs_root'], r, 'metrics.json')))['subjects']
            gap = max(abs((rk[i] < 10).mean() - met[str(s)]['top10']) for i, s in enumerate(D.subjects))
            np.savez(p, rk=rk, rr=rr, run=run, t=np.array([t for _, t in D.items['test']]), n_run=n_run,
                     dist=boundary_distance(D), subjects=np.array(D.subjects))
            print(f'{r}: segments {rk.shape[1]}, max |top10 - metrics.json| = {gap:.2e}', flush=True)
    rng = np.random.default_rng(c['seed']); res = {'margins_s': c['margins_s'], 'primary_margin_s': c['primary_margin_s'], 'pools': {}}
    Z = {g: [np.load(os.path.join(od, f'{r}.npz')) for r in rs] for g, rs in c['runs'].items()}
    subs = list(Z['A4_clip_w2v'][0]['subjects']); dist = Z['A4_clip_w2v'][0]['dist']
    for g in Z: assert all(np.array_equal(z['dist'], dist) for z in Z[g])
    for pname, ps in {'19': subs, '17': [s for s in subs if s not in c['pool_exclude']]}.items():
        idx = np.array([subs.index(s) for s in ps]); Bp = rng.integers(0, len(ps), (c['n_boot'], len(ps))); pr = {}
        for mg in c['margins_s']:
            keep = dist >= mg - 1e-9 if mg > 0 else np.ones(len(dist), bool); row = {'n_segments': int(keep.sum())}
            per = {g: np.mean([(z['rk'][:, keep] < 10).mean(1) for z in zs], 0)[idx] for g, zs in Z.items()}   # seed mean per subject
            prun = {g: np.mean([(z['rr'][:, keep] < 10).mean(1) for z in zs], 0)[idx] for g, zs in Z.items()}
            ci = lambda x: [float(x.mean()), float(np.percentile(x[Bp].mean(1), 2.5)), float(np.percentile(x[Bp].mean(1), 97.5))]
            row['top10'] = {g: ci(v) for g, v in per.items()}; row['run_top10'] = {g: ci(v) for g, v in prun.items()}
            nr = Z['A4_clip_w2v'][0]['n_run'][keep]; row['chance_top10'] = 10 / len(dist); row['run_chance_top10'] = float(np.mean(np.minimum(10, nr) / nr))
            row['E_A4'] = ci(per['A4_clip_w2v'] - per['A4_phaserand']); row['E_reg'] = ci(per['reg_clip_w2v'] - per['reg_phaserand'])
            row['A4_minus_reg'] = ci(per['A4_clip_w2v'] - per['reg_clip_w2v'])
            pr[str(mg)] = row
        for mg in c['margins_s']:   # change from all segments to the boundary-far subset, per group, and the A4 - reg difference of that change
            if mg == 0: continue
            k0, k1 = np.ones(len(dist), bool), dist >= mg - 1e-9
            dd = {g: np.mean([(z['rk'][:, k1] < 10).mean(1) - (z['rk'][:, k0] < 10).mean(1) for z in zs], 0)[idx] for g, zs in Z.items()}
            pr[str(mg)]['far_minus_all'] = {g: [float(v.mean()), float(np.percentile(v[Bp].mean(1), 2.5)), float(np.percentile(v[Bp].mean(1), 97.5))] for g, v in dd.items()}
            x = dd['A4_clip_w2v'] - dd['reg_clip_w2v']
            pr[str(mg)]['far_minus_all_A4_minus_reg'] = [float(x.mean()), float(np.percentile(x[Bp].mean(1), 2.5)), float(np.percentile(x[Bp].mean(1), 97.5))]
        res['pools'][pname] = pr
        print(pname, json.dumps({mg: {'n': v['n_segments'], 'top10': {g: round(x[0] * 100, 2) for g, x in v['top10'].items()}} for mg, v in pr.items()}), flush=True)
    json.dump(res, open(os.path.join(od, 'boundary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
