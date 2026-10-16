"""F-V (D-111, D-112 sections 3-4) data access: image metadata, concept splits, repetition-mean inputs and the batch tuple.

Inputs are the per-session image repetition means written by scripts/fv_prepare.py (sub-XX/ses-YY_training_imgavg.npy,
float16 [n_images, 63, 250], 0..1 s). A subject's input over a set of sessions is the repetition-count-weighted mean of those
session means (= the mean over every repetition of the image in those sessions). Only training-run files exist in the cache
before the verdict job; `session_mean` refuses any other partition.

Concept splits (D-112 section 3): development concepts = the first n of the 1,654 training concepts sorted by
sha256("{folder}|{salt}"); validation concepts for early stopping = the first `fraction` of the remaining (non-development)
concepts sorted by sha256("{folder}|{valid salt}").

The batch tuple follows CLAUDE.md: (x, positions, subject_id, dataset_id, trial_id, clip_id, t_onset) with clip_id = image
event code (training: 1..16540) and t_onset = 0 (stimulus-locked epochs).
"""
import os, json, hashlib, numpy as np, torch


def sha(s): return hashlib.sha256(s.encode()).hexdigest()


class Meta:
    """image_meta.json (scripts/fv_meta.py, from image_metadata.npy via the restricted unpickler). Concept names: the training
    folder name ('00001_aardvark', numbered inside the training split) is the split key; the THINGS id ('00001_aardvark' in
    *_THINGS, numbered over the 1,854 THINGS concepts) links a concept to the name table."""
    def __init__(self, path):
        m = json.load(open(path)); self.train_files = m['train_img_files']; self.train_img_concept = m['train_img_concepts']
        self.test_files = m['test_img_files']; self.test_img_concept = m['test_img_concepts']
        self.train_concepts = list(dict.fromkeys(self.train_img_concept)); ci = {c: i for i, c in enumerate(self.train_concepts)}
        self.img_concept = np.array([ci[c] for c in self.train_img_concept])          # concept index of training code k at row k-1
        assert len(self.train_concepts) == 1654 and len(self.train_files) == 16540, (len(self.train_concepts), len(self.train_files))
        self.train_things = list(dict.fromkeys(m['train_img_concepts_THINGS'])); self.test_things = list(dict.fromkeys(m['test_img_concepts_THINGS']))
        assert len(self.train_things) == 1654 and len(self.test_things) == 200, (len(self.train_things), len(self.test_things))
        assert not set(self.train_things) & set(self.test_things), 'training and test concepts overlap (THINGS ids)'

    def dev_concepts(self, salt, n): return sorted(self.train_concepts, key=lambda c: sha(f'{c}|{salt}'))[:n]

    def valid_concepts(self, exclude, salt, fraction):
        rest = [c for c in self.train_concepts if c not in set(exclude)]
        return sorted(rest, key=lambda c: sha(f'{c}|{salt}'))[:int(np.ceil(fraction * len(rest)))]

    def codes_of(self, concepts):
        s = {self.train_concepts.index(c) for c in concepts}; return np.array([k + 1 for k in range(len(self.img_concept)) if self.img_concept[k] in s])


def session_mean(cache, subject, session, part='training'):
    assert part == 'training', f'only training-run caches exist before the verdict job (asked: {part})'
    d = os.path.join(cache, f'sub-{subject:02d}'); e = np.load(os.path.join(d, f'ses-{session:02d}_training_events.npz'))
    return e['imgavg_code'], e['imgavg_nrep'], np.load(os.path.join(d, f'ses-{session:02d}_training_imgavg.npy'), mmap_mode='r')


def subject_mean(cache, subject, sessions, codes=None):
    """repetition-weighted mean over `sessions` for every image code present (optionally restricted to `codes`) ->
    (codes [n], X float16 [n, C, T], nrep [n])"""
    parts = [session_mean(cache, subject, s) for s in sessions]; allc = np.unique(np.concatenate([p[0] for p in parts]))
    if codes is not None: allc = allc[np.isin(allc, codes)]
    pos = {c: i for i, c in enumerate(allc)}; C, T = parts[0][2].shape[1:]
    S = np.zeros((len(allc), C, T), np.float32); n = np.zeros(len(allc), np.int64)
    for cc, nr, X in parts:
        keep = np.isin(cc, allc); rows = np.where(keep)[0]; dst = np.array([pos[c] for c in cc[keep]], np.int64)
        for a in range(0, len(rows), 2048):
            r = rows[a:a + 2048]; S[dst[a:a + 2048]] += np.asarray(X[r], np.float32) * nr[r, None, None]
        n[dst] += nr[keep]
    return allc, (S / n[:, None, None]).astype(np.float16), n


def concept_mean(codes, X, nrep, img_concept, concepts_idx):
    """per concept in concepts_idx: repetition-weighted mean over all its images present -> (present concept idx, X [m, C, T])"""
    ci = img_concept[codes - 1]; out, keep = [], []
    for j in concepts_idx:
        m = ci == j
        if m.any(): keep.append(j); out.append((X[m].astype(np.float32) * nrep[m, None, None]).sum(0) / nrep[m].sum())
    return np.array(keep, np.int64), (np.stack(out) if out else np.zeros((0,) + X.shape[1:], np.float32))


class GPUSet:
    """inputs of several subjects on the GPU (float16) with their subject-layer index, code and target row"""
    def __init__(self, blocks, positions, dev):
        """blocks: list of (subject_layer_index, codes, X float16 [n, C, T])"""
        N = sum(len(c) for _, c, _ in blocks); self.X = torch.empty((N,) + blocks[0][2].shape[1:], dtype=torch.float16, device=dev); a = 0
        for _, c, X in blocks: self.X[a:a + len(c)] = torch.from_numpy(np.ascontiguousarray(X, np.float16)).to(dev); a += len(c)   # no host-side concatenated copy
        self.subj = torch.cat([torch.full((len(c),), i, dtype=torch.long) for i, c, _ in blocks]).to(dev)
        self.code = torch.cat([torch.from_numpy(np.asarray(c, np.int64)) for _, c, _ in blocks]).to(dev)
        self.pos = torch.as_tensor(positions, dtype=torch.float32, device=dev); self.dev = dev

    def __len__(self): return len(self.code)

    def batch(self, idx):
        """CLAUDE.md tuple for rows idx (LongTensor on the device)"""
        x = self.X[idx].float(); B = len(idx)
        return (x, self.pos[None].expand(B, -1, -1), self.subj[idx], torch.zeros(B, dtype=torch.long, device=self.dev), idx,
                self.code[idx], torch.zeros(B, device=self.dev))
