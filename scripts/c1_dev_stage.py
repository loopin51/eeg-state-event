"""C1 development stage (D-097 section 11 as changed by amendment 6): 4 GPU-hour rule, candidate selection D_sel and futility D_dev.

Development metric (section 11): query-weighted mean over the development cells (training subjects of subject fold 0 x
validation runs 5 and 13) of the within-run MRP of the selected shared network with each subject's own training layer, i.e.
metrics.json targets._dev_trainers_valid. Sources (amendment 6): P1 = the P1 main job s{seed}_k0_f0; P2 = the development run
c1_train.py configs/c1_p2.yaml --dev-only --candidate NAME --seed S --subject-fold 0 --run-fold 0 (folder suffix _dev-NAME).
Only metrics.json is read; test_cells.json is never opened.

  --stage hours    : sqrt seed-2036 pre-training metrics.json seconds / 3600 > 4.0 (device RTX 5080) -> hours dropped; reports the
                     value, the job folder and whether the 60,000-update cap was reached. Run right after that job, before any
                     hours job is submitted.
  --stage select   : for every candidate in force, D_sel = mean over seeds 2036-2037 of the candidate's metric - the same mean of
                     P1; the largest D_sel is chosen (ties: the earlier candidate in the registered order); every D_sel is
                     reported; the one-time shared-validation-loss check of section 11 is flagged when the chosen candidate's
                     mean (seeds 2036-2037) best shared validation CLIP loss is higher than P1's.
  --stage futility : D_dev = mean over the 5 seeds of the chosen candidate's metric - the same mean of P1; D_dev < +0.01 ->
                     'development-stage futility' (the C1 test is not opened); otherwise the chosen candidate is to be frozen
                     in configs/c1_p2.yaml pretrain.candidate by an amendment (this script does not edit the config).
D_sel and D_dev lean towards the chosen candidate; they are gates only and are not reported as effect estimates.
Every stage writes its block into <out> (earlier blocks kept) with the folders and sha256 of the files it read.

usage: python scripts/c1_dev_stage.py configs/c1_dev_stage.yaml --stage hours|select|futility
"""
import os, sys, glob, json, hashlib, argparse, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(p): return p if os.path.isabs(p) else os.path.join(ROOT, p)


def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def one(pat):
    hits = [h for h in sorted(glob.glob(pat)) if os.path.exists(os.path.join(h, 'metrics.json'))]
    assert len(hits) == 1, f'{pat}: {len(hits)} finished folders'; return hits[0]


def dev_metric(d, v, eval_split):
    p = os.path.join(d, 'metrics.json'); M = json.load(open(p)); s = M['summary']
    assert s['eval_split'] == eval_split, (d, s['eval_split']); assert v['device'] in str(s['device']), (d, s['device'])
    cells = M['targets']['_dev_trainers_valid']; sig = {f'{a}_{r}': x['n'] for a, rr in cells.items() for r, x in rr.items()}
    num = sum(x['mrp'] * x['n'] for rr in cells.values() for x in rr.values()); den = sum(x['n'] for rr in cells.values() for x in rr.values())
    return dict(metric=num / den, n_queries=den, n_cells=len(sig), shared_best_valid=s['shared_best_valid'], shared_best_update=s['shared_best_update'],
                device=s['device'], init=s.get('init'), folder=os.path.basename(d), sha256=sha(p)), sig


class Stage:
    def __init__(self, v):
        self.v = v; self.c1 = yaml.safe_load(open(path(v['arms']['P1']))); self.c2 = yaml.safe_load(open(path(v['arms']['P2'])))
        self.setting = yaml.safe_load(open(path(self.c1['b1_config'])))['decoder_setting']; self.root = path(v['runs_root']); self.sig = None

    def check_sig(self, sig):
        if self.sig is None: self.sig = sig
        assert sig == self.sig, 'development cells differ between runs'

    def cells_record(self):   # every run read so far passed check_sig: same cell list and the same query count (= weight) per cell
        return dict(identical_across_runs=True, n_cells=len(self.sig), n_queries=sum(self.sig.values()), queries_per_cell=dict(sorted(self.sig.items())),
                    weight='query count n per cell (query-weighted MRP)')

    def p1(self, seed):
        r, sig = dev_metric(one(os.path.join(self.root, f'*_c1_{self.c1["arm"]}_{self.setting}_s{seed}_k0_f0')), self.v, 'test'); self.check_sig(sig); return r

    def p2(self, seed, cand):
        r, sig = dev_metric(one(os.path.join(self.root, f'*_c1_{self.c2["arm"]}_{self.setting}_s{seed}_k0_f0_dev-{cand}')), self.v, 'dev_only')
        assert r['init'] and f'_c1_P2_pretrain_s{seed}_{cand}' in r['init'], (r['folder'], r['init']); self.check_sig(sig); return r

    def hours(self):
        h = self.v['hours_rule']; ck = one(os.path.join(self.root, os.path.basename(os.path.dirname(self.c2['pretrain']['ckpt_glob'].format(seed=h['seed'], candidate=h['candidate'])))))
        p = os.path.join(ck, 'metrics.json'); M = json.load(open(p)); assert self.v['device'] in str(M['device']), (ck, M['device'])
        gh = M['seconds'] / 3600; drop = bool(gh > h['max_gpu_hours']); cands = [c for c in self.c2['pretrain']['candidates'] if not (drop and c == h['drop'])]
        return dict(job_folder=os.path.basename(ck), gpu_hours=gh, max_gpu_hours=h['max_gpu_hours'], hours_dropped=drop, candidates_in_force=cands,
                    updates=M['updates'], update_cap=h['update_cap'], reached_cap=bool(M['updates'] >= h['update_cap']), best_update=M['best_update'],
                    device=M['device'], sha256=sha(p))

    def select(self, cands):
        S = self.v['select_seeds']; P1 = {str(s): self.p1(s) for s in S}; m1 = sum(P1[str(s)]['metric'] for s in S) / len(S)
        tab = {}
        for c in cands:
            R = {str(s): self.p2(s, c) for s in S}; m = sum(R[str(s)]['metric'] for s in S) / len(S)
            tab[c] = dict(runs=R, mean=m, D_sel=m - m1, per_seed_diff={str(s): R[str(s)]['metric'] - P1[str(s)]['metric'] for s in S})
        chosen = max(cands, key=lambda c: (tab[c]['D_sel'], -cands.index(c)))
        tied = [c for c in cands if c != chosen and tab[c]['D_sel'] == tab[chosen]['D_sel']]   # reported only; the rule above already breaks ties
        l1 = sum(P1[str(s)]['shared_best_valid'] for s in S) / len(S); l2 = sum(tab[chosen]['runs'][str(s)]['shared_best_valid'] for s in S) / len(S)
        return dict(seeds=S, P1=dict(runs=P1, mean=m1), candidates=tab, D_sel={c: tab[c]['D_sel'] for c in cands},
                    per_seed_diff={c: tab[c]['per_seed_diff'] for c in cands}, chosen=chosen, tie=dict(tied_with=tied, broken_by_order=bool(tied)),
                    shared_loss=dict(P1_mean=l1, chosen_mean=l2, check_needed=bool(l2 > l1)), dev_cells=self.cells_record(),
                    note='D_sel is a gate, not an effect estimate')

    def futility(self, chosen):
        S = self.v['all_seeds']; P1 = {str(s): self.p1(s) for s in S}; P2 = {str(s): self.p2(s, chosen) for s in S}
        m1 = sum(x['metric'] for x in P1.values()) / len(S); m2 = sum(x['metric'] for x in P2.values()) / len(S); d = m2 - m1
        fut = bool(d < self.v['futility_threshold'])
        return dict(seeds=S, chosen=chosen, P1=dict(runs=P1, mean=m1), P2=dict(runs=P2, mean=m2), D_dev=d, threshold=self.v['futility_threshold'],
                    futile=fut, decision='development-stage futility: the C1 test is not opened' if fut else f'freeze candidate {chosen} in configs/c1_p2.yaml (amendment)',
                    dev_cells=self.cells_record(), note='D_dev is a gate, not an effect estimate')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', required=True, choices=['hours', 'select', 'futility'])
    a = ap.parse_args(); v = yaml.safe_load(open(path(a.config))); out = path(v['out']); st = Stage(v)
    res = json.load(open(out)) if os.path.exists(out) else {}
    if a.stage == 'hours': res['hours'] = st.hours()
    elif a.stage == 'select':
        assert 'hours' in res, 'run --stage hours first'; res['select'] = st.select(res['hours']['candidates_in_force'])
    else:
        assert 'select' in res, 'run --stage select first'; res['futility'] = st.futility(res['select']['chosen'])
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    b = res[a.stage]; show = {k: x for k, x in b.items() if k not in ('runs', 'P1', 'P2', 'candidates')}
    print(json.dumps({a.stage: show}), flush=True)


if __name__ == '__main__':
    main()
