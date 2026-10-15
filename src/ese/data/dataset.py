"""Unified multi-dataset window access over caches (docs/06 §4).

Every item is padded to N_CH=62 channels; padded channels are zero with INVALID
positions so the ChannelMerger ignores them. Global ids: subject_global =
offset[dataset] + subject, trial_key / clip_key are unique across datasets.
"""
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from .cache import Cache
from .montage import INVALID

N_CH = 62
DATASET_IDS = {'seediv': 0, 'deap': 1, 'faced': 2, 'eav': 3, 'bci2020': 4, 'nieuwland': 5, 'seedvii': 6}
STIM_LOCKED = {'seediv', 'deap', 'faced', 'seedvii'}


class MultiCache(Dataset):
    def __init__(self, root, names, subjects=None):
        """subjects: optional dict name -> list of local subject ids to keep."""
        self.caches, metas, off = {}, [], 0
        for nm in names:
            c = Cache(root, nm); self.caches[nm] = c
            m = c.meta.copy(); m['dataset'] = nm; m['dataset_id'] = DATASET_IDS[nm]
            m['subject_local'] = m['subject']; m['subject_global'] = m['subject'] + off
            m['row_local'] = np.arange(len(m))
            m['trial_key'] = (m['dataset_id'] * 10_000_000 + m['subject'] * 100_000 + m['session'] * 1000 + m['trial'])
            m['clip_key'] = np.where(nm in STIM_LOCKED, m['dataset_id'] * 100_000 + m['clip_id'], -1)
            m.loc[m['clip_id'] < 0, 'clip_key'] = -1
            if subjects is not None and nm in subjects: m = m[m['subject'].isin(subjects[nm])]
            metas.append(m); off += int(c.meta.subject.max()) + 1
        self.meta = pd.concat(metas, ignore_index=True)
        self.n_subjects_global = off
        self._pos = {nm: self._pad_pos(c.positions) for nm, c in self.caches.items()}

    @staticmethod
    def _pad_pos(p):
        out = np.full((N_CH, 2), INVALID, np.float32); out[:len(p)] = p; return out

    def __len__(self): return len(self.meta)

    def __getitem__(self, i):
        r = self.meta.iloc[i]; c = self.caches[r['dataset']]
        w = np.asarray(c.windows[int(r['row_local'])], dtype=np.float32)
        x = np.zeros((N_CH, w.shape[1]), np.float32); x[:w.shape[0]] = w
        return dict(x=torch.from_numpy(x), positions=torch.from_numpy(self._pos[r['dataset']]),
                    dataset_id=int(r['dataset_id']), subject=int(r['subject_global']),
                    trial_key=int(r['trial_key']), clip_key=int(r['clip_key']), t_onset=float(r['t_onset']),
                    valence=int(r['valence']), row=int(i))


def collate(items):
    out = {k: torch.stack([it[k] for it in items]) for k in ('x', 'positions')}
    for k in ('dataset_id', 'subject', 'trial_key', 'clip_key', 'valence', 'row', 'pair_id'):
        out[k] = torch.tensor([it.get(k, -1) for it in items], dtype=torch.long)
    out['t_onset'] = torch.tensor([it['t_onset'] for it in items], dtype=torch.float32)
    return out
