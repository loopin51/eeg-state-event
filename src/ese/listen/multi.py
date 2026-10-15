"""Multi-dataset window loader for the C1 P2 external pretraining (D-099).

Datasets (configs/c1_p2_pretrain.yaml `datasets`):
  kind 'ext'   : caches written by scripts/ext_prepare.py (SparrKULee, KU Leuven AAD, ChineseEEG-2); recording layout.
  kind 'alice' : the D-092 Alice cache (data/cache/alice_eeg), 12 segments at shared offsets, positions_common.npy.
Every item is (dataset, subject, recording, t) with t the window anchor in seconds from the recording start (= stimulus time
for the ext caches). A batch returns the CLAUDE.md loader tuple
  (x, positions, subject_id, dataset_id, trial_id, clip_id, t_onset)
plus the target y (normalised w2v window): x [B, Cmax, T] after bad/flat-excluded CAR, baseline and clamp (as RunData.batch),
zero-padded to Cmax channels with INVALID positions; subject_id is the global subject-layer index; trial_id the recording
index inside the subject; clip_id the global stimulus index; t_onset the window start in stimulus time.
Validation windows: whole stimulus groups held out per dataset by a hash rule fixed in the config (valid_fraction,
valid_seed). Recordings whose attended OR unattended stimulus is in a held-out group are not used for training.
Optional extra filter (`eeg.extra_filter`, e.g. a low-pass for D-101): applied per recording to a derived memmap in
<cache>/derived/, so the broadband caches never change.
"""
import os, re, json, hashlib, numpy as np, torch
from ese.model.brain_clip import INVALID


def stim_group(name, rules):
    for pat, rep in rules: name = re.sub(pat, rep, name)
    return name


def held_out(groups, fraction, seed):
    """deterministic: groups sorted by sha256(group + seed); the first ceil(fraction * n) (at least 1) are validation."""
    g = sorted(set(groups), key=lambda x: hashlib.sha256(f'{x}|{seed}'.encode()).hexdigest())
    return set(g[:max(1, int(np.ceil(fraction * len(g))))])


class Source:
    """one dataset: per-subject EEG memmaps, recording table, stimulus feature memmap, positions."""
    def __init__(self, root, d, eeg_cfg, win):
        self.name, self.kind = d['name'], d['kind']; cache = os.path.join(root, d['cache_dir']); self.cache = cache
        if self.kind == 'ext':
            meta = json.load(open(os.path.join(cache, 'eeg_meta.json'))); st = json.load(open(os.path.join(cache, 'stimuli.json')))['stimuli']
            self.fs = meta['sample_rate']; self.pos = np.load(os.path.join(cache, 'positions.npy'))
            self.F = np.load(os.path.join(cache, 'w2v_stim.npy'), mmap_mode='r'); subs = sorted(int(s) for s in meta['subjects'])
            self.recs = {s: [r for r in meta['subjects'][str(s)] if 'excluded' not in r] for s in subs}
            self.stim_off = {k: v['offset'] for k, v in st.items()}
            files = {s: f'{d.get("source", "eeg_hp05")}_sub-{s:03d}.npy' for s in subs}
        else:   # alice (D-092 cache)
            meta = json.load(open(os.path.join(cache, 'eeg_meta.json'))); src = meta['sources'][d.get('source', 'eeg_hp05')]
            self.fs = src['sample_rate']; offs, lens = src['run_offsets'], src['run_lengths']; self.pos = np.load(os.path.join(cache, 'positions_common.npy'))
            self.F = np.load(os.path.join(cache, d.get('features', 'w2v_seg') + '.npy'), mmap_mode='r'); subs = sorted(int(s) for s in meta['subjects'])
            self.recs, self.stim_off = {}, {}
            for s in subs:
                m = meta['subjects'][str(s)]; flat = np.array(m['sources'][d.get('source', 'eeg_hp05')]['flat'])
                self.recs[s] = [dict(rec=f'S{s:02d}_seg{k + 1:02d}', offset=offs[k], length=lens[k], stim=f'seg{k + 1:02d}', stim_start_s=0.0,
                                     bad=list(m['bad_mask']), flat=flat.tolist()) for k in range(len(lens)) if k + 1 not in m['missing_segments']]
            self.stim_off = {f'seg{k + 1:02d}': offs[k] for k in range(len(lens))}
            files = {s: f'{d.get("source", "eeg_hp05")}_sub-{s:03d}.npy' for s in subs}
        if d.get('subjects'): subs = [s for s in subs if s in d['subjects']]
        self.subjects = subs; self.eeg = {}
        xf = eeg_cfg.get('extra_filter')
        for s in subs:
            p = os.path.join(cache, files[s])
            if xf: p = derived(p, self.recs[s], self.fs, xf)
            self.eeg[s] = np.load(p, mmap_mode='r')
        self.C = self.pos.shape[0]; self.T = int(round((win['tmax'] - win['tmin']) * self.fs))
        rules = [tuple(x) for x in d.get('stim_group_rules', [])]
        self.group = {k: stim_group(k, rules) for k in self.stim_off}
        self.valid_groups = held_out(self.group.values(), d['valid_fraction'], d['valid_seed'])


def derived(path, recs, fs, xf):
    """per-recording zero-phase filter of a cached source into <cache>/derived/<tag>_<file> (computed once)."""
    import mne
    tag = xf['tag']; out = os.path.join(os.path.dirname(path), 'derived', f'{tag}_{os.path.basename(path)}')
    if os.path.exists(out): return out
    os.makedirs(os.path.dirname(out), exist_ok=True); X = np.load(path, mmap_mode='r'); Y = np.lib.format.open_memmap(out + '.tmp.npy', 'w+', np.float32, X.shape)
    for r in recs:
        a, b = r['offset'], r['offset'] + r['length']
        Y[:, a:b] = mne.filter.filter_data(np.asarray(X[:, a:b], np.float64), fs, l_freq=xf.get('l_freq'), h_freq=xf.get('h_freq'), n_jobs=1, verbose=False)
    Y.flush(); del Y; os.replace(out + '.tmp.npy', out); return out


class MultiData:
    def __init__(self, root, cfg, smoke=False):
        self.cfg = cfg; w, e = cfg['window'], cfg['eeg']; self.w, self.e = w, e
        for d in cfg['datasets']:   # ds004408 (C1 evaluation) and DTU (C2 evaluation) never enter the pretraining (D-096, D-099)
            assert not any(k in (d['name'] + d['cache_dir']).lower() for k in ('ds004408', 'stagea', 'stageb', 'dtu')), d
        self.src = [Source(root, d, e, w) for d in cfg['datasets'] if d.get('enabled', True)]
        fs = {s.fs for s in self.src}; assert len(fs) == 1, f'sample rates differ: {fs}'; self.fs = fs.pop()
        self.T = int(round((w['tmax'] - w['tmin']) * self.fs)); self.Cmax = max(s.C for s in self.src)
        self.subj_index, self.items, self.clip_index = {}, {'train': [], 'valid': []}, {}
        self.hours = {'train': [0.0] * len(self.src), 'valid': [0.0] * len(self.src)}
        for di, s in enumerate(self.src):
            subs = s.subjects[:2] if smoke else s.subjects
            for sub in subs:
                self.subj_index[(di, sub)] = len(self.subj_index); seen = set()
                for ri, r in enumerate(s.recs[sub]):
                    gs = {s.group[r['stim']]} | ({s.group[r['unattended']]} if r.get('unattended') in s.group else set())
                    split = 'valid' if s.group[r['stim']] in s.valid_groups else 'train' if not gs & s.valid_groups else None
                    if split is None or (smoke and split in seen): continue          # smoke: one training and one validation recording
                    seen.add(split)
                    self.clip_index.setdefault((di, r['stim']), len(self.clip_index)); self.hours[split][di] += r['length'] / s.fs / 3600
                    hi = r['length'] / self.fs - w['tmax'] - max(e['offset_s'], 0)
                    for t in np.arange(-w['tmin'], hi + 1e-9, w['train_step_s']): self.items[split].append((di, sub, ri, float(t)))
        self.n_subjects = len(self.subj_index)
        self.by_ds = {k: [np.array([i for i, it in enumerate(v) if it[0] == di]) for di in range(len(self.src))] for k, v in self.items.items()}
        # per-dataset target normalisation from training-recording features (RunData: mean / SD over a subsample)
        self.fnorm = []
        for di, s in enumerate(self.src):
            idx = sorted({s.stim_off[r['stim']] + j for (dd, sub, ri, t) in self.items['train'][::max(1, len(self.items['train']) // 4000)]
                          if dd == di for r in [s.recs[sub][ri]] for j in range(0, r['length'], 600)})
            Fs = np.asarray(s.F[:, idx], np.float32) if idx else np.zeros((s.F.shape[0], 2), np.float32)
            self.fnorm.append((Fs.mean(1, keepdims=True), Fs.std(1, keepdims=True) + 1e-6))

    def summary(self):
        out = {}
        for di, s in enumerate(self.src):
            out[s.name] = dict(subjects=len([1 for k in self.subj_index if k[0] == di]), train_windows=int(len(self.by_ds['train'][di])),
                               valid_windows=int(len(self.by_ds['valid'][di])), valid_groups=sorted(s.valid_groups), channels=s.C,
                               train_hours=self.hours['train'][di], valid_hours=self.hours['valid'][di])
        return out

    def sample(self, rng, split, weights, n):
        """dataset per example from the mixing weights, then a uniform window of that dataset."""
        p = np.asarray(weights, float); p = p / p.sum(); ds = rng.choice(len(p), n, p=p); out = []
        for di in ds:
            pool = self.by_ds[split][di]; assert len(pool), f'{self.src[di].name}: no {split} windows'
            out.append(self.items[split][pool[rng.integers(len(pool))]])
        return out

    def batch(self, items, dev):
        w, e = self.w, self.e; B = len(items); X = np.zeros((B, self.Cmax, self.T), np.float32); P = np.full((B, self.Cmax, 2), INVALID, np.float32)
        Y = np.zeros((B, self.src[0].F.shape[0], self.T), np.float32); sid, dsid, trial, clip, onset = [], [], [], [], []
        for b, (di, sub, ri, t) in enumerate(items):
            s = self.src[di]; r = s.recs[sub][ri]; i = r['offset'] + int(round((t + w['tmin'] + e['offset_s']) * self.fs))
            x = np.asarray(s.eeg[sub][:, i:i + self.T], np.float32); ok = ~(np.array(r['bad'], bool) | np.array(r['flat'], bool))
            if e['reference'] == 'car': x = x - x[ok].mean(0, keepdims=True)
            x = x - x[:, :int(round(e['baseline_s'] * self.fs))].mean(-1, keepdims=True); x = np.clip(x, -e['clamp'], e['clamp'])
            X[b, :s.C] = x; P[b, :s.C] = np.where(ok[:, None], s.pos, INVALID)
            f = s.stim_off[r['stim']] + int(round((r.get('stim_start_s', 0.0) + t + w['tmin']) * self.fs)); m, sd = self.fnorm[di]
            Y[b] = (np.asarray(s.F[:, f:f + self.T], np.float32) - m) / sd
            sid.append(self.subj_index[(di, sub)]); dsid.append(di); trial.append(ri); clip.append(self.clip_index[(di, r['stim'])])
            onset.append(r.get('stim_start_s', 0.0) + t + w['tmin'])
        tt = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        return (torch.from_numpy(X).to(dev, non_blocking=True), torch.from_numpy(P).to(dev), tt(sid), tt(dsid), tt(trial), tt(clip),
                tt(onset, torch.float32)), torch.from_numpy(Y).to(dev, non_blocking=True)
