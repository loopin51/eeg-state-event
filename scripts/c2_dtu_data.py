"""C2 window loader for the DTU two-talker trials (D-097 amendments 2 and 4), the DTU counterpart of stageB_b1_train.RunData.

Cache (eeg-a4, scripts/ext_prepare.py --dataset dtu, recording layout of D-099): <cache>/eeg_hp05_sub-XXX.npy [C, N] (robust
scaled, 120 Hz, 0.5 Hz high-pass, no re-reference), eeg_meta.json (per subject the recordings {rec, trial, offset, length, stim =
attended segment, unattended, stim_start_s, bad, flat}), positions.npy [C, 2] (ds004408 frame), stimuli.json {stim: {offset,
length}} and w2v_stim.npy [D, S] (wav2vec2-large-xlsr-53 layers 14-18 at 120 Hz) for both streams.
Items are (subject, recording index, t), t = window anchor in seconds from the trial start (the window is [t + tmin, t + tmax]
on the stimulus, the EEG window is shifted by eeg.offset_s, as in RunData). Training and validation items use the
window.train_step_s grid; queries use query_step_s (0.5 s) and the candidates of a query are all query windows of the same
trial (within-trial MRP). EEG handling = RunData.batch: CAR without bad and flat channels, baseline, clamp, INVALID positions.
batch() returns the CLAUDE.md loader tuple (x, positions, subject_id, dataset_id, trial_id, clip_id, t_onset) and the target y
of the requested stream ('stim' = attended, 'unattended' for the UNATTENDED meter and the report).

usage (check on a cache): python scripts/c2_dtu_data.py configs/c2_dtu_folds.yaml configs/stageB_b1.yaml [subject_fold run_fold]
"""
import os, sys, json, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from c2_dtu_split import DTUSplit, trials_from_cache, path
INVALID = -0.1                                                             # ese.model.brain_clip.INVALID
DATASET_ID = 0


class DTUData:
    def __init__(self, c, fc, K, f, source='eeg_hp05', split=None, subjects=None):
        """c: B1-style config (window, eeg); fc: configs/c2_dtu_folds.yaml; K, f: subject fold and trial fold."""
        self.c, self.fc = c, fc; cache = path(fc['cache_dir']); meta = json.load(open(os.path.join(cache, 'eeg_meta.json')))
        st = json.load(open(os.path.join(cache, 'stimuli.json'))); st = st.get('stimuli', st)
        self.fs = meta['sample_rate']; self.pos = torch.from_numpy(np.load(os.path.join(cache, 'positions.npy')).astype(np.float32))
        self.F = np.load(os.path.join(cache, 'w2v_stim.npy'), mmap_mode='r'); self.stim_off = {k: v['offset'] for k, v in st.items()}
        self.clip_index = {k: i for i, k in enumerate(sorted(st))}
        self.split = split or DTUSplit(fc, trials_from_cache(fc['cache_dir'])); job = self.split.fold_job(K, f)
        self.trainers, self.targets = job['trainers'], job['targets']; self.subjects = subjects or self.trainers + self.targets
        self.recs = {s: meta['subjects'][str(s)] for s in self.subjects}
        self.eeg = {s: np.load(os.path.join(cache, f'{source}_sub-{s:03d}.npy'), mmap_mode='r') for s in self.subjects}
        self.invalid = {s: [None if 'excluded' in r else np.array(r['bad'], bool) | np.array(r['flat'], bool) for r in self.recs[s]] for s in self.subjects}
        w, e = c['window'], c['eeg']; self.T = int(round((w['tmax'] - w['tmin']) * self.fs)); self.lo = -w['tmin']
        self.sets = {s: self.split.sets(s, f) for s in self.subjects}; self.f = f
        self.items = {'train': [it for s in self.trainers for it in self.grid(s, self.sets[s]['calib'], w['train_step_s'])],
                      'valid': [it for s in self.trainers for it in self.grid(s, self.sets[s]['valid'], w['train_step_s'])]}
        idx = sorted({self.stim_off[self.recs[s][i]['stim']] + int(round((t + w['tmin']) * self.fs)) for s, i, t in self.items['train'][::max(1, len(self.items['train']) // 4000)]})
        Fs = np.asarray(self.F[:, idx], np.float32); self.fmean, self.fstd = Fs.mean(1, keepdims=True), Fs.std(1, keepdims=True) + 1e-6

    def hi(self, r): return r['length'] / self.fs - self.c['window']['tmax'] - max(self.c['eeg']['offset_s'], 0)

    def grid(self, s, recs, step):
        """items (s, recording index, t) of the given trial records on a grid of `step` seconds"""
        return [(s, r['index'], float(t)) for r in recs for t in np.arange(self.lo, self.hi(self.recs[s][r['index']]) + 1e-9, step)]

    def calib_items(self, s, k, step=None):
        return self.grid(s, self.split.draw(s, self.f, k), step or self.c['window']['train_step_s'])

    def valid_items(self, s, step=None): return self.grid(s, self.sets[s]['valid'], step or self.c['window']['train_step_s'])

    def queries(self, s, which='test', pre_shift=0.0):
        """within-trial queries on the query_step_s grid; PRE (pre_shift < 0): only windows whose earlier EEG window is in the trial"""
        q = self.grid(s, self.sets[s][which], self.fc['query_step_s'])
        if pre_shift: q = [x for x in q if x[2] + self.c['window']['tmin'] + self.c['eeg']['offset_s'] + pre_shift >= 0]
        return q

    def to_device(self, dev):
        self.dev = dev; self.posg = self.pos.to(dev); return self

    def feat_idx(self, s, i, t, stream='stim'):
        r = self.recs[s][i]; return self.stim_off[r[stream]] + int(round((r.get('stim_start_s', 0.0) + t + self.c['window']['tmin']) * self.fs))

    def batch(self, items, subj_idx, eeg_shift=0.0, stream='stim'):
        """items: list of (subject, recording index, t); subj_idx: subject-layer index per item; returns ((x, positions, subject_id,
        dataset_id, trial_id, clip_id, t_onset), y)"""
        e, w = self.c['eeg'], self.c['window']; dev = self.dev; X, Pm, Y, trial, clip, onset = [], [], [], [], [], []
        for s, i, t in items:
            r = self.recs[s][i]; a = r['offset'] + int(round((t + w['tmin'] + e['offset_s'] + eeg_shift) * self.fs))
            assert r['offset'] <= a and a + self.T <= r['offset'] + r['length'], 'EEG window outside the trial'
            X.append(np.asarray(self.eeg[s][:, a:a + self.T], np.float32)); Pm.append(self.invalid[s][i])
            fi = self.feat_idx(s, i, t, stream); Y.append((np.asarray(self.F[:, fi:fi + self.T], np.float32) - self.fmean) / self.fstd)
            trial.append(r['trial']); clip.append(self.clip_index[r[stream]]); onset.append(r.get('stim_start_s', 0.0) + t + w['tmin'])
        X = torch.from_numpy(np.stack(X)).to(dev); bad = torch.from_numpy(np.stack(Pm)).to(dev)
        if e.get('reference', 'car') == 'car':
            ok = (~bad).float()[..., None]; X = X - (X * ok).sum(1, keepdim=True) / ok.sum(1, keepdim=True)
        X = X - X[..., :int(round(e['baseline_s'] * self.fs))].mean(-1, keepdim=True); X = X.clamp(-e['clamp'], e['clamp'])
        P = self.posg[None].expand(len(items), -1, -1).masked_fill(bad[..., None], INVALID)
        tt = lambda v, dt=torch.long: torch.as_tensor(np.asarray(v), dtype=dt, device=dev)
        return (X, P, tt(subj_idx), tt([DATASET_ID] * len(items)), tt(trial), tt(clip), tt(onset, torch.float32)), torch.from_numpy(np.stack(Y)).to(dev)


def main():
    fc = yaml.safe_load(open(path(sys.argv[1]))); c = yaml.safe_load(open(path(sys.argv[2]))); K, f = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 0)
    c['eeg'].setdefault('reference', 'car'); D = DTUData(c, fc, K, f).to_device('cpu')
    s = D.targets[0]; q = D.queries(s); n_tr = {s_: len(D.sets[s_]['test']) for s_ in D.targets}
    (x, p, sid, did, tid, cid, ton), y = D.batch(q[:8], [0] * 8)
    (_, _, _, _, _, cu, _), yu = D.batch(q[:8], [0] * 8, stream='unattended')
    pre = D.queries(s, pre_shift=-fc['pre_shift_s'])
    print(json.dumps(dict(fold=[K, f], trainers=D.trainers, targets=D.targets, train_windows=len(D.items['train']), valid_windows=len(D.items['valid']),
                          target_test_trials=n_tr, queries_first_target=len(q), pre_queries_first_target=len(pre),
                          k2_trials_first_target=[r['trial'] for r in D.split.draw(s, f, 2)], x=list(x.shape), y=list(y.shape),
                          clip_att=cid.tolist()[:2], clip_unatt=cu.tolist()[:2], same_target=bool(torch.allclose(y, yu)))), flush=True)


if __name__ == '__main__':
    main()
