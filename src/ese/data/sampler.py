"""Stage-A batch composer v2 (docs/06 §4, D-018).

v1 let both contrastive losses be solved by SUBJECT IDENTITY: coherence positives were
same-subject/same-trial while negatives were mostly other subjects, so z_state clustered by
subject and the cross-subject align positives were pushed apart (align loss rose above chance).

v2 makes subject identity useless for both losses:
  coherence block : S subjects x K trials x 2 windows (<= coh_dt apart). Negatives = OTHER TRIALS OF THE
                    SAME SUBJECT (enforced in the loss).
  align block     : pick a dataset, pick P subjects, pick G (clip, t_onset) keys that ALL P subjects have;
                    take every (subject, key) cell -> P x G items. For an anchor the positives are the other
                    subjects at the same key, negatives include THE SAME SUBJECT at other keys.
  random          : uniform filler (keeps non-stim-locked datasets in the JEPA/adversary losses).
Dataset mixing ~ sqrt(#subjects); align mixing additionally scaled by `align_weights` (D-017).
Yields lists of (row, pair_id).
"""
import numpy as np


class StageABatchSampler:
    def __init__(self, meta, coh_subjects=8, coh_trials=4, align_blocks=2, align_subjects=12, align_keys=4,
                 n_random=16, coh_dt=20.0, seed=0, align_weights=None):
        self.m = meta.reset_index(drop=True); self.rng = np.random.default_rng(seed)
        self.cs, self.ct, self.ab, self.asub, self.ak, self.n_random, self.coh_dt = coh_subjects, coh_trials, align_blocks, align_subjects, align_keys, n_random, coh_dt
        ds = self.m.groupby('dataset')['subject_global'].nunique()
        w = np.sqrt(ds.values); self.ds_names, self.ds_p = list(ds.index), w / w.sum()
        # dataset -> subject -> list of (rows, onsets) per trial
        self.trials = {nm: {} for nm in self.ds_names}
        for (nm, sg, tk), g in self.m.groupby(['dataset', 'subject_global', 'trial_key']):
            g = g.sort_values('t_onset')
            if len(g) >= 2: self.trials[nm].setdefault(sg, []).append((g.index.to_numpy(), g['t_onset'].to_numpy()))
        self.trials = {nm: {s: t for s, t in d.items() if len(t) >= 2} for nm, d in self.trials.items()}
        # dataset -> key (clip_key, t_onset) -> {subject: row}; and key -> clip
        self.cells, self.key_clip = {}, {}
        s = self.m[self.m['clip_key'] >= 0]
        for (nm, ck, t), g in s.groupby(['dataset', 'clip_key', 't_onset']):
            if g['subject_global'].nunique() >= 2:
                self.cells.setdefault(nm, {})[(ck, t)] = dict(zip(g['subject_global'], g.index))
                self.key_clip[(ck, t)] = ck
        self.align_names = [nm for nm in self.ds_names if nm in self.cells]
        aw = np.array([np.sqrt(ds[nm]) * float((align_weights or {}).get(nm, 1.0)) for nm in self.align_names])
        self.align_p = aw / aw.sum() if len(aw) else None
        self.keys = {nm: list(self.cells[nm].keys()) for nm in self.align_names}
        self.batch_size = 2 * coh_subjects * coh_trials + align_blocks * align_subjects * align_keys + n_random

    def __iter__(self):
        while True: yield self.sample()

    def _coherence(self, rows, pair, pid):
        for _ in range(self.cs):
            nm = self.rng.choice(self.ds_names, p=self.ds_p)
            if not self.trials[nm]: continue
            subj = list(self.trials[nm].keys()); tl = self.trials[nm][subj[self.rng.integers(len(subj))]]
            for ti in self.rng.choice(len(tl), size=min(self.ct, len(tl)), replace=False):
                idx, on = tl[ti]; i = self.rng.integers(len(idx))
                cand = np.where((np.abs(on - on[i]) <= self.coh_dt) & (np.arange(len(idx)) != i))[0]
                if not len(cand): continue
                j = self.rng.choice(cand); rows += [idx[i], idx[j]]; pair += [pid, pid]; pid += 1
        return pid

    def _align(self, rows, pair):
        nm = self.rng.choice(self.align_names, p=self.align_p); cells, keys = self.cells[nm], self.keys[nm]
        k0 = keys[self.rng.integers(len(keys))]; avail = list(cells[k0].keys())
        subs = self.rng.choice(avail, size=min(self.asub, len(avail)), replace=False)
        chosen, clips = [k0], {self.key_clip[k0]}
        for ki in self.rng.permutation(len(keys)):
            if len(chosen) >= self.ak: break
            k = keys[ki]
            if self.key_clip[k] in clips: continue
            if all(s in cells[k] for s in subs): chosen.append(k); clips.add(self.key_clip[k])
        for k in chosen:
            rows += [cells[k][s] for s in subs]; pair += [-1] * len(subs)

    def sample(self):
        rows, pair = [], []
        self._coherence(rows, pair, 0)
        if self.align_names:
            for _ in range(self.ab): self._align(rows, pair)
        n_fill = max(self.n_random, self.batch_size - len(rows))
        r = self.rng.choice(len(self.m), size=min(n_fill, len(self.m)), replace=False)
        rows += r.tolist(); pair += [-1] * len(r)
        return list(zip(rows, pair))
