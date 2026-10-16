"""D-113 5.5: MDE after the shared-block exclusion (EEG-free; recorded before any Gao EEG value).

Counts the test trials of arms a and b per category after the D-113 4.3 exclusion and the label-anomaly exclusion, and
gives the D-100 R-d binomial approximation p_MDE - p0 = z * sqrt(p0 (1 - p0) / n) per category and for the equal-weight
category mean, plus the same with a between-subject sd added (subject-level variance p0(1-p0)/n_s + sd^2, averaged over
the subjects of the arm). Also the training-trial counts of arm a after the exclusion.
usage: python -I scripts/fv_gao_mde.py configs/fv_gao.yaml
"""
import json, math, os, sys
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.gao.design import load_trials, splits

c = yaml.safe_load(open(os.path.join(ROOT, sys.argv[1]))); P = lambda p: os.path.join(ROOT, p)
T, runs = load_trials(P(c['bids']), c)
z, sd = c['mde']['z'], c['mde']['sd_between']; res = dict(n_runs=len(runs), n_trials=int(len(T['label'])))
for exclude in (True, False):
    S = splits(T, c, exclude=exclude); tag = 'excluded' if exclude else 'no_exclusion'; res[tag] = {}
    for arm in ('b', 'a'):
        r = {}; comb_var, comb_var_sd = [], []
        for t, tc in c['tasks'].items():
            K = len(tc['items']); p0 = 1 / K
            keys = [k for k in S if k[0] == arm and k[2] == t]
            n_te = [len(S[k][1]) for k in keys]; n_tr = [len(S[k][0]) for k in keys]; n = sum(n_te)
            v = p0 * (1 - p0) / n; vs = np.mean([p0 * (1 - p0) / m + sd ** 2 for m in n_te]) / len(n_te)
            comb_var.append(v); comb_var_sd.append(vs)
            r[tc['category']] = dict(n_subjects=len(keys), n_test=n, test_per_subject=[min(n_te), int(np.median(n_te)), max(n_te)],
                                    train_per_subject=[min(n_tr), int(np.median(n_tr)), max(n_tr)], p0=round(p0, 4),
                                    mde_excess=round(z * math.sqrt(v), 4), mde_excess_sd=round(z * math.sqrt(vs), 4))
        r['combined'] = dict(mde_excess=round(z * math.sqrt(sum(comb_var) / 9), 4), mde_excess_sd=round(z * math.sqrt(sum(comb_var_sd) / 9), 4))
        res[tag][arm] = r
out = P(os.path.join(c['out'], 'mde_after_exclusion.json')); json.dump(res, open(out, 'w'), indent=1)
print(json.dumps(res, indent=1))
