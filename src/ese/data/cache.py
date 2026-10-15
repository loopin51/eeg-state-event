"""Streaming memory-mapped window cache (docs/06 §3).

Layout of data/cache/<dataset>/:
  windows.f16      raw float16 bytes, shape [N, C, L] (row-major)
  header.json      {"n": N, "channels": C, "length": L, "sfreq": 100, "dtype": "float16", ...}
  meta.parquet     one row per window: subject, session, trial, clip_id, t_onset,
                   valence (-1/0/+1), label (dataset-specific int), extra columns
  positions.npy    float32 [C, 2] normalized scalp coordinates (INVALID = -0.1)

Writer appends window blocks to the open file, so peak memory is one
(subject, session) worth of trials — never the whole dataset (1st-study EAV OOM).
"""
import os, json, resource, time
import numpy as np, pandas as pd


class CacheWriter:
    def __init__(self, root, dataset, channels, length, positions, sfreq=100, extra=None):
        self.dir = os.path.join(root, dataset); os.makedirs(self.dir, exist_ok=True)
        self.f = open(os.path.join(self.dir, 'windows.f16'), 'wb')
        self.C, self.L, self.n = channels, length, 0
        self.rows = []; self.t0 = time.time()
        self.header = dict(dataset=dataset, channels=channels, length=length, sfreq=sfreq, dtype='float16', **(extra or {}))
        np.save(os.path.join(self.dir, 'positions.npy'), positions.astype(np.float32))

    def add(self, windows, rows):
        """windows: float32 [n, C, L]; rows: list of dicts (len n)."""
        assert windows.shape[1:] == (self.C, self.L), windows.shape
        assert len(rows) == windows.shape[0]
        self.f.write(np.ascontiguousarray(windows, dtype=np.float16).tobytes())
        self.rows.extend(rows); self.n += windows.shape[0]

    def close(self):
        self.f.close()
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if os.uname().sysname == 'Darwin' else 1024)
        self.header.update(n=self.n, peak_rss_mb=round(peak_mb, 1), seconds=round(time.time() - self.t0, 1))
        json.dump(self.header, open(os.path.join(self.dir, 'header.json'), 'w'), indent=1)
        pd.DataFrame(self.rows).to_parquet(os.path.join(self.dir, 'meta.parquet'), index=False)
        return self.header


class Cache:
    """Read side. `windows[i]` -> float16 [C, L]; `meta` -> DataFrame; `positions` -> [C,2]."""
    def __init__(self, root, dataset):
        self.dir = os.path.join(root, dataset)
        self.header = json.load(open(os.path.join(self.dir, 'header.json')))
        n, C, L = self.header['n'], self.header['channels'], self.header['length']
        self.windows = np.memmap(os.path.join(self.dir, 'windows.f16'), dtype=np.float16, mode='r', shape=(n, C, L))
        self.meta = pd.read_parquet(os.path.join(self.dir, 'meta.parquet'))
        self.positions = np.load(os.path.join(self.dir, 'positions.npy'))

    def __len__(self): return self.header['n']
