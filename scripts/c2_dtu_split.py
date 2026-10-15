"""C2 split of the DTU two-talker trials (D-097 amendments 2 and 4, configs/c2_dtu_folds.yaml).

Unit = fixed male/female 50-s pair (62 pairs, a pair is named by its male segment). Trial fold f:
  test   : the subject's trials whose pair is in test_pairs[f] (attended and unattended streams both out of every other set),
  valid  : trials whose pair is in valid_pairs[f] (early stopping of the shared network and of the calibration),
  calib  : all other trials (calibration candidates of the target; training trials of the training subjects).
Subject folds: the 18 subjects permuted with default_rng(subject_fold_seed), cut into sizes 4, 4, 4, 3, 3 (amendment 2).
Calibration draw: the first n_k trials of default_rng([20261014, subject, fold]).permutation(sorted calibration trial numbers)
(nested, as stageB_calib_curve.draw_runs / c1_train.draw_runs); n_k from k_trials (k1 4, k2 7, k4 14, kmax all).
Trial records come from the DTU cache (data/cache/dtu/eeg_meta.json, recording layout of scripts/ext_prepare.py: rec, trial,
stim = attended segment, unattended = unattended segment) or, before the cache exists, from the expinfo table
(results/d099_ext/dtu/dtu_two_talker_trials.tsv).

usage: python scripts/c2_dtu_split.py configs/c2_dtu_folds.yaml      (checks the split against amendment 4 and writes the table)
"""
import os, sys, json, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = (4, 4, 4, 3, 3)


def path(p): return p if os.path.isabs(p) else os.path.join(ROOT, p)


def trials_from_table(tsv):
    import pandas as pd
    T = pd.read_csv(path(tsv), sep='\t'); out = {}
    for r in T.itertuples():
        out.setdefault(int(r.subject), []).append(dict(trial=int(r.trial), stim=r.attended, unattended=r.unattended))
    return out


def trials_from_cache(cache):
    meta = json.load(open(os.path.join(path(cache), 'eeg_meta.json')))
    return {int(s): [dict(r, index=i) for i, r in enumerate(R) if 'excluded' not in r and r.get('unattended')] for s, R in meta['subjects'].items()}


class DTUSplit:
    def __init__(self, cfg, trials):
        self.cfg = cfg; self.partner = cfg['partner']; self.female = {v: k for k, v in self.partner.items()}
        self.test_pairs = [set(x) for x in cfg['test_pairs']]; self.valid_pairs = [set(x) for x in cfg['valid_pairs']]
        allp = set().union(*self.test_pairs); assert allp == set(self.partner) and sum(map(len, self.test_pairs)) == len(allp), 'test folds must partition the pairs'
        for f, v in enumerate(self.valid_pairs): assert not v & self.test_pairs[f], f'fold {f}: validation pair inside the test fold'
        self.trials = trials; self.subjects = sorted(trials)
        for s, R in trials.items():
            for r in R: r['pair'] = self.pair(r['stim'], r['unattended'])
            assert len({r['pair'] for r in R}) == len(R), f'subject {s}: a pair occurs twice'
        perm = [int(x) for x in np.random.default_rng(cfg['subject_fold_seed']).permutation(self.subjects)]
        assert len(perm) == sum(SIZES), f'{len(perm)} subjects, folds sized {SIZES}'
        cut = np.cumsum((0,) + SIZES); self.subject_folds = [sorted(perm[a:b]) for a, b in zip(cut[:-1], cut[1:])]

    def pair(self, att, unatt):
        m = att if att in self.partner else unatt; f = unatt if m == att else att
        assert self.partner.get(m) == f, f'{att} / {unatt} is not a fixed pair'; return m

    def sets(self, s, f):
        R = self.trials[s]
        test = [r for r in R if r['pair'] in self.test_pairs[f]]; valid = [r for r in R if r['pair'] in self.valid_pairs[f]]
        calib = [r for r in R if r['pair'] not in self.test_pairs[f] | self.valid_pairs[f]]
        return dict(test=test, valid=valid, calib=calib)

    def n_k(self, k, n_calib):
        v = self.cfg['k_trials'].get(k, self.cfg['k_trials'].get(str(k)))
        return n_calib if v == 'all' else int(v)

    def draw(self, s, f, k, draw_seed=20261014):
        """nested calibration draw: trial records of the first n_k of the fixed permutation of the calibration trial numbers"""
        cal = self.sets(s, f)['calib']; nums = sorted(r['trial'] for r in cal); n = self.n_k(k, len(nums))
        keep = {int(x) for x in np.random.default_rng([draw_seed, s, f]).permutation(nums)[:n]}
        return sorted((r for r in cal if r['trial'] in keep), key=lambda r: r['trial'])

    def fold_job(self, K, f):
        targets = self.subject_folds[K]; trainers = sorted(s for j, fs in enumerate(self.subject_folds) if j != K for s in fs)
        return dict(targets=targets, trainers=trainers)

    def table(self):
        out = []
        for f in range(5):
            c = {k: [len(self.sets(s, f)[k]) for s in self.subjects] for k in ('test', 'valid', 'calib')}
            out.append({k: dict(min=min(v), max=max(v)) for k, v in c.items()})
        return out


def main():
    cfg = yaml.safe_load(open(path(sys.argv[1])))
    cache = cfg.get('cache_dir'); src = 'cache' if cache and os.path.exists(os.path.join(path(cache), 'eeg_meta.json')) else 'table'
    trials = trials_from_cache(cache) if src == 'cache' else trials_from_table(cfg['trials_table'])
    S = DTUSplit(cfg, trials); tab = S.table()
    want = cfg.get('expected')                                                     # amendment 4 counts (per-subject min, max)
    if want:
        for k in ('test', 'valid'):
            got = [[t[k]['min'], t[k]['max']] for t in tab]; assert got == want[k], f'{k}: {got} != {want[k]}'
        lo, hi = min(t['calib']['min'] for t in tab), max(t['calib']['max'] for t in tab); assert [lo, hi] == want['calib_range'], f'calib {lo}-{hi}'
    for f in range(5):
        for s in S.subjects:
            st = S.sets(s, f); pairs = {k: {r['pair'] for r in v} for k, v in st.items()}
            assert not (pairs['test'] & pairs['valid'] or pairs['test'] & pairs['calib'] or pairs['valid'] & pairs['calib'])
            assert sum(map(len, st.values())) == len(S.trials[s]), 'a trial is dropped'
            d7 = S.draw(s, f, 2); d14 = S.draw(s, f, 4); assert {r['trial'] for r in d7} <= {r['trial'] for r in d14}, 'draws not nested'
            assert len(d7) == S.n_k(2, len(st['calib'])), 'k2 draw size'
    res = dict(source=src, subject_folds=S.subject_folds, folds=tab, n_trials={str(s): len(S.trials[s]) for s in S.subjects},
               k_trials={str(k): S.n_k(k, 40) for k in (1, 2, 4)},
               draws_k2={f'{s}_{f}': [r['trial'] for r in S.draw(s, f, 2)] for s in S.subjects for f in range(5)})
    out = path(cfg.get('split_out', 'results/c2/dtu_split.json')); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(dict(source=src, subject_folds=S.subject_folds, folds=tab)), flush=True)


if __name__ == '__main__':
    main()
