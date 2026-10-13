"""Extensible dataset registry for the affect encoder.

Every entry in DATASET_REGISTRY describes one raw-EEG source that has already
been staged + wrapped as a torch Dataset returning a unified 62-channel sample:
    {'x': [62,150] float32 (zero-padded + INVALID(-0.1) positions if the source
           has fewer than 62 real channels), 'positions': [62,2], 'subject_id':
           int (offset so ids never collide across datasets), 'valence': {-1,0,+1},
           'trial_id': int, 'session_id': int, 'dataset': str}

Currently registered: SEED-IV (15 subjects, native 62ch) and DEAP (32 subjects,
32ch padded to 62). Both raw sources are fully staged on the training server.

--- Adding a new dataset later (e.g. GAMEEMO / EAV / VUI / SEED-V once
    downloaded) ---
1. Write a wrapper Dataset class (model it on affect_enc/data.py's SeedIVAffect
   or affect_enc/deap_data.py's DeapAffect) whose __getitem__ returns the dict
   shape above. If the source has C < 62 real channels, pad with zeros and mark
   the extra channel positions as INVALID (-0.1, -0.1) -- ChannelMerger already
   masks those to attention weight 0, so no other code needs to change.
2. Pick a `subject_offset` that does not overlap any existing entry's
   [subject_offset, subject_offset + len(all_subjects)) range (append after the
   current maximum -- see `total_subjects()` below, which computes the required
   SubjectLayers embedding size automatically from whichever datasets are active).
3. Add ONE entry to DATASET_REGISTRY, e.g.:
       'gameemo': {
           'build': lambda subjects: GameemoAffect(GAMEEMO_ROOT, GAMEEMO_IO, subjects=subjects),
           'all_subjects': list(range(28)),
           'subject_offset': 47,   # 15 (SEED-IV) + 32 (DEAP)
           'label': 'GAMEEMO',
       }
4. Re-run full_train.py with --datasets seediv,deap,gameemo (or any subset) --
   no other code changes are needed; n_subjects_emb, the train/val split, the
   combined DataLoader, and the per-dataset Stage-2 readout all pick up the new
   entry automatically from the registry.
"""
import os
from .data import SeedIVAffect
from .deap_data import DeapAffect

# Reuse the LMDB caches already built during dataset validation (they were
# built over ALL subjects of each source, so no rebuild is needed for full
# training -- only the Python-side `subjects` filter changes per call).
SEED_ROOT = os.path.expanduser('~/affect_data/seed_iv/eeg_raw_data')
SEED_IO   = os.path.expanduser('~/affect_data/.torcheeg/seediv_smoke')
DEAP_ROOT = os.path.expanduser('~/affect_data/deap/deap-dataset/data_preprocessed_python')
DEAP_IO   = os.path.expanduser('~/affect_data/.torcheeg/deap_smoke')

DATASET_REGISTRY = {
    'seediv': {
        'build': lambda subjects: SeedIVAffect(SEED_ROOT, SEED_IO, subjects=subjects, preprocess=True),
        'all_subjects': list(range(15)),   # SEED-IV: 15 subjects total
        'subject_offset': 0,
        'label': 'SEED-IV',
    },
    'deap': {
        'build': lambda subjects: DeapAffect(DEAP_ROOT, DEAP_IO, subjects=subjects, preprocess=True),
        'all_subjects': list(range(32)),   # DEAP: 32 subjects total
        'subject_offset': 15,
        'label': 'DEAP',
    },
    # Future entries go here -- see module docstring for the exact recipe.
}


def total_subjects(active_names):
    """Required SubjectLayers embedding size (n_subjects) given which registry
    entries are active -- the max over (subject_offset + len(all_subjects))."""
    max_id = 0
    for nm in active_names:
        e = DATASET_REGISTRY[nm]
        max_id = max(max_id, e['subject_offset'] + len(e['all_subjects']))
    return max_id
