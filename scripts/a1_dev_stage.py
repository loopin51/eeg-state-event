"""F-III A1 development stage (D-105 sections 3, 9 and 10): LS freeze and seed count from the S-diagnosis, then the gate.

  --stage ls   : from the S-diagnosis of the 5 existing P1 k0_f0 models (a1_diag metrics.json, pseudo-target k2 layer): the
                 26-cell query-weighted J250 MRP per seed for every LS rule; the rule with the highest 5-seed mean is frozen
                 (ties: earlier in configs/a1.yaml lag_search.rules); s_J = sample sd (ddof 1) of the 5 seed values of that
                 rule; n = min(8, max(5, ceil(2 s_J^2 / 0.004^2))). Also the diagnosis reproduction checks. This script does
                 not edit configs/a1.yaml: the rule and n are written there by an amendment.
  --stage gate : development Delta_J of every N candidate (n seeds of --dev-only runs, pseudo-target k2, J250, frozen rule)
                 minus the same P1 mean; the largest is chosen (ties: jitter, lse, lse_jit); gate (i) chosen Delta_J >= +0.01,
                 (ii) Alice: chosen candidate's n-seed mean J250 MRP minus P1's > 0, (iii) delta 0 development drop
                 (P1 mean - chosen mean of d0_fixed) <= 0.03. All three -> freeze the candidate by an amendment; otherwise
                 'A1 development-stage futility'. Gate values are biased towards the chosen candidate and are not effect estimates.
Every stage writes its block into <out> (earlier blocks kept) with the folders and sha256 of the files it read.

usage: python scripts/a1_dev_stage.py configs/a1_dev_stage.yaml --stage ls|gate
"""
import os, sys, glob, json, math, hashlib, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(p): return p if os.path.isabs(p) else os.path.join(ROOT, p)
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def one(pat, n_net=None):
    """the single finished folder matching pat; with n_net, only runs whose network had n_net subject rows (D-105 amendment 3:
    a1_train runs before the fix built a smaller network in --dev-only and are not development runs)"""
    hits = [h for h in sorted(glob.glob(pat)) if os.path.exists(os.path.join(h, 'metrics.json'))]
    if n_net is not None: hits = [h for h in hits if json.load(open(os.path.join(h, 'metrics.json')))['summary'].get('n_net_subjects') == n_net]
    assert len(hits) == 1, f'{pat}: {len(hits)} finished folders'; return hits[0]


def pooled(cells, key):
    num = sum(c[key] * c['n'] for c in cells); den = sum(c['n'] for c in cells); return num / den, den


def diag_cells(d, layer):
    M = json.load(open(os.path.join(d, 'metrics.json'))); cl = [dict(v, n=v['n_main']) for k, v in M['cells'].items() if k.startswith(layer + '_')]
    return M, cl


def dev_cells(d):
    M = json.load(open(os.path.join(d, 'metrics.json'))); assert M['summary']['eval_split'] == 'dev_only', d
    return M, [c for rr in M['targets']['_dev_a1_pseudo_k2'].values() for c in rr.values()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', required=True, choices=['ls', 'gate']); a = ap.parse_args()
    v = yaml.safe_load(open(path(a.config))); A = yaml.safe_load(open(path(v['a1_config']))); rules = A['lag_search']['rules']; root = path(v['runs_root'])
    out = path(v['out']); res = json.load(open(out)) if os.path.exists(out) else {}
    if a.stage == 'ls':
        per = {}; sig = None; files = {}
        for s in v['diag_seeds']:
            d = one(os.path.join(root, v['diag_pattern'].format(seed=s))); M, cl = diag_cells(d, 'pseudo')
            assert M['summary']['arm'] == 'P1' and v['device'] in M['summary']['device'], d
            sg = sorted((k, c['n_main']) for k, c in M['cells'].items() if k.startswith('pseudo_')); sig = sig or sg; assert sg == sig, 'diagnosis cells differ'
            assert len(cl) == v['n_cells'], (d, len(cl))
            per[str(s)] = {r: pooled(cl, f'j250_{r}')[0] for r in rules}
            files[str(s)] = dict(folder=os.path.basename(d), sha256=sha(os.path.join(d, 'metrics.json')), checks=M['checks'], n_queries=pooled(cl, f'j250_{rules[0]}')[1])
        mean = {r: float(np.mean([per[s][r] for s in per])) for r in rules}; best = max(rules, key=lambda r: (mean[r], -rules.index(r)))
        sJ = float(np.std([per[s][best] for s in per], ddof=1)); n = int(min(v['n_max'], max(v['n_min'], math.ceil(2 * sJ ** 2 / v['se_target'] ** 2))))
        res['ls'] = dict(per_seed=per, mean=mean, rule=best, s_J=sJ, n_seeds=n, files=files, note='rule chosen on P1 (conservative for N); the rule and n go into configs/a1.yaml by an amendment')
    else:
        assert 'ls' in res, 'run --stage ls first'; rule = A['lag_search']['rule']; assert rule == res['ls']['rule'], 'configs/a1.yaml lag_search.rule must equal the frozen rule'
        seeds = v['dev_seeds'][:res['ls']['n_seeds']]; key = f'j250_{rule}'; tab = {}; sig = None

        def arm(pat, kind):
            vals, d0, files = {}, {}, {}
            for s in seeds:
                d = one(os.path.join(root, pat.format(seed=s)), v['dev_n_net'] if kind == 'dev' else None)
                if kind == 'dev':
                    M, cl = dev_cells(d); sg = sorted(c['n'] for c in cl); nonlocal_sig.append(sg)
                    vals[str(s)] = pooled(cl, key)[0]; d0[str(s)] = pooled(cl, 'd0_fixed')[0]
                else:
                    M = json.load(open(os.path.join(d, 'metrics.json'))); cl = list(M['targets'].values()); vals[str(s)] = pooled(cl, key)[0]
                assert v['device'] in M['summary']['device'], d; files[str(s)] = dict(folder=os.path.basename(d), sha256=sha(os.path.join(d, 'metrics.json')))
            return dict(values=vals, mean=float(np.mean(list(vals.values()))), d0=d0, d0_mean=float(np.mean(list(d0.values()))) if d0 else None, files=files)
        nonlocal_sig = []
        P1 = arm(v['dev_pattern']['P1'], 'dev'); P1a = arm(v['alice_pattern']['P1'], 'alice')
        for c in v['candidates']:
            tab[c] = dict(dev=arm(v['dev_pattern']['N'].format(cand=c, seed='{seed}'), 'dev'), alice=arm(v['alice_pattern']['N'].format(cand=c, seed='{seed}'), 'alice'))
            tab[c]['dev_delta_J'] = tab[c]['dev']['mean'] - P1['mean']; tab[c]['alice_delta_J'] = tab[c]['alice']['mean'] - P1a['mean']
            tab[c]['d0_drop'] = P1['d0_mean'] - tab[c]['dev']['d0_mean']
        assert all(s_ == nonlocal_sig[0] for s_ in nonlocal_sig), 'development cells differ between runs'
        chosen = max(v['candidates'], key=lambda c: (tab[c]['dev_delta_J'], -v['candidates'].index(c))); t = tab[chosen]
        g = dict(i=bool(t['dev_delta_J'] >= v['threshold']), ii=bool(t['alice_delta_J'] > 0), iii=bool(t['d0_drop'] <= v['max_d0_drop']))
        res['gate'] = dict(seeds=seeds, rule=rule, P1=dict(dev=P1, alice=P1a), candidates=tab, dev_delta_J={c: tab[c]['dev_delta_J'] for c in tab}, chosen=chosen,
                           gate=g, passed=all(g.values()), decision=f'freeze candidate {chosen} (amendment)' if all(g.values()) else 'A1 development-stage futility',
                           note='gate values are biased towards the chosen candidate; not effect estimates')
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    b = res[a.stage]; print(json.dumps({a.stage: {k: x for k, x in b.items() if k not in ('files', 'candidates', 'P1')}}), flush=True)


if __name__ == '__main__':
    main()
