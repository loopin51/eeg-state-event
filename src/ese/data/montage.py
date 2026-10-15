"""Channel-name -> normalized 2D scalp coordinates shared by every dataset.

All montages are projected onto ONE common frame: the MNE standard_1020 x/y
positions, min-max normalized over the union of all 10-20/10-10 electrodes so
that a given electrode (e.g. Cz) has the same coordinate whichever dataset it
comes from. Channels without a standard position get the INVALID sentinel
(-0.1, -0.1), which the ChannelMerger masks to attention weight 0.

Ported from the 1st study's preprocess.py (seed62_positions / deap32_positions)
with one fix: the 1st study normalized each montage by its OWN bounding box,
so DEAP's Fp1 and SEED's Fp1 did not coincide exactly. Here the frame is global.
"""
from functools import lru_cache
import numpy as np

INVALID = -0.1

SEED62 = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
          'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ',
          'C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
          'P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4',
          'PO6','PO8','CB1','O1','OZ','O2','CB2']
DEAP32 = ['FP1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7',
          'PO3','O1','OZ','PZ','FP2','AF4','FZ','F4','F8','FC6','FC2','CZ',
          'C4','T8','CP6','CP2','P4','P8','PO4','O2']
# FACED Processed_data order = second-cohort order (Electrode_Location.xlsx note); HEOR/HEOL were
# attached to the right/left mastoids, i.e. A2/A1.
FACED32 = ['FP1','FP2','FZ','F3','F4','F7','F8','FC1','FC2','FC5','FC6','CZ','C3','C4','T7','T8',
           'CP1','CP2','CP5','CP6','PZ','P3','P4','P7','P8','PO3','PO4','OZ','O1','O2','HEOR','HEOL']
# SEED-specific cerebellar leads approximated by the nearest occipital electrode; FACED mastoids
ALIAS = {'CB1': 'O1', 'CB2': 'O2', 'HEOR': 'A2', 'HEOL': 'A1'}

MONTAGES = {'seediv': SEED62, 'deap': DEAP32, 'faced': FACED32, 'seedvii': SEED62}


@lru_cache(maxsize=1)
def _standard_xy():
    import mne
    try:
        mont = mne.channels.make_standard_montage('colin27_1020')      # MNE >= 1.13 name
    except ValueError:
        mont = mne.channels.make_standard_montage('standard_1020')
    pos = mont.get_positions()['ch_pos']
    names = list(pos.keys())
    xy = np.array([[pos[k][0], pos[k][1]] for k in names])
    mn, mx = xy.min(0), xy.max(0)                     # GLOBAL frame over all 1020 electrodes
    norm = (xy - mn) / (mx - mn)
    return {k.upper(): norm[i] for i, k in enumerate(names)}


def positions(channels):
    """List of channel names -> float32 [C, 2] in [0,1]^2, INVALID where unknown."""
    table = _standard_xy()
    out = np.full((len(channels), 2), INVALID, dtype=np.float32)
    for i, ch in enumerate(channels):
        key = ch.upper(); key = ALIAS.get(key, key)
        if key in table:
            out[i] = table[key]
    return out


def positions_for(dataset):
    return positions(MONTAGES[dataset]), MONTAGES[dataset]
