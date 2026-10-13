"""DEAP data wrapper + multi-dataset combiner (SEED-IV 62ch + DEAP 32ch).

DEAP is padded from 32 -> 62 channels: first 32 carry real signal + DEAP
positions; the remaining 30 are zeros marked with the INVALID position (-0.1),
which ChannelMerger masks to weight 0. Both datasets thus present a unified
[62, T] tensor through the same in_channels=62 backbone path.

Subject-id scheme: SEED-IV 0-14, DEAP 15-46 (n_subjects=47).
"""
import os, numpy as np, torch
from torch.utils.data import Dataset, ConcatDataset
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms as T
from .preprocess import preprocess_window, deap32_positions, seed62_positions

N_UNIFIED_CH = 62
DEAP_SUBJ_OFFSET = 15          # DEAP subject ids -> 15..46

def _pad_to_62(x32, dpos32):
    """x32 [32,Tt] -> [62,Tt] zero-padded; positions [32,2] -> [62,2] invalid-padded."""
    Tt = x32.shape[1]
    x = np.zeros((N_UNIFIED_CH, Tt), dtype=np.float32); x[:32] = x32
    pos = np.full((N_UNIFIED_CH, 2), -0.1, dtype=np.float32); pos[:32] = dpos32
    return x, pos

class DeapAffect(Dataset):
    def __init__(self, root_path, io_path, chunk_size=384, sfreq_in=128,
                 preprocess=True, subjects=None, valence_median=None):
        self.base = DEAPDataset(
            root_path=root_path, io_path=io_path,
            chunk_size=chunk_size, overlap=0, num_channel=32, num_baseline=3,
            online_transform=T.ToTensor(), label_transform=T.Select('valence'),
            num_worker=2, verbose=False)
        info = self.base.info.reset_index(drop=True)
        info['subj_int'] = info['subject_id'].astype(str).str.extract(r'(\d+)').astype(int) - 1
        # binary valence at MEDIAN (this mirror's ratings skew low)
        self.vmed = float(info['valence'].median()) if valence_median is None else valence_median
        if subjects is not None:
            self.idx = info.index[info['subj_int'].isin(subjects)].to_numpy()
        else:
            self.idx = info.index.to_numpy()
        self.info = info; self.sfreq_in = sfreq_in; self.preprocess = preprocess
        dpos, _ = deap32_positions(); self.dpos = dpos     # [32,2]

    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = int(self.idx[i]); x,_ = self.base[j]
        x = x.numpy().astype(np.float32)                  # [32,384]
        if self.preprocess: x = preprocess_window(x, sfreq_in=self.sfreq_in)  # [32,150]
        xp, pos = _pad_to_62(x, self.dpos)                # [62,150],[62,2]
        row = self.info.iloc[j]
        val = 1 if row['valence'] > self.vmed else -1
        return {'x':torch.from_numpy(np.ascontiguousarray(xp)),
                'subject_id':int(row['subj_int'])+DEAP_SUBJ_OFFSET,
                'emotion':-1, 'valence':val,
                'trial_id':int(row['trial_id']),'session_id':0,
                'dataset':'deap',
                'positions':torch.from_numpy(pos)}

def multi_collate(batch):
    x = torch.stack([b['x'] for b in batch]); pos = torch.stack([b['positions'] for b in batch])
    subj = torch.tensor([b['subject_id'] for b in batch]); val = torch.tensor([b['valence'] for b in batch])
    trial = torch.tensor([b['trial_id'] for b in batch]); sess = torch.tensor([b['session_id'] for b in batch])
    dset = [b.get('dataset','seediv') for b in batch]
    return {'x':x,'positions':pos,'subject_id':subj,'valence':val,'trial_id':trial,
            'session_id':sess,'dataset':dset}
