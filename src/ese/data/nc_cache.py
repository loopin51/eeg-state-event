"""D-128 epoch cache for REVE-NC (docs/08 §3.1, scripts/nc_prep.py).

Layout of data/cache/nc/<dataset>/:
  x.f32         float32 [N, C, L] EEG epochs in microvolts at 200 Hz (row-major memmap)
  ref.f32       float32 [N, R, L] measured noise-reference channels (EOG/EMG), only if the dataset has them
  meta.parquet  one row per epoch: subject, session, run, trial, kind ('task'|'null'), label (-1 for null),
                clip_id, t_onset (s, epoch start in its recording), t_order (0-1 position in the session), flags
  header.json   n, channels, length, sfreq, ch_names, ref_names, epoch_t0_s (epoch start relative to the event),
                windows (task / null slices in samples), lowpass_hz, raw sha256 per file, preprocessing parameters
  aux/          per-recording non-task data (rest, artifact calibration), npz, same preprocessing

float32 (not float16 as data/cache/<dataset>/windows.f16): the 45-70 Hz band of N1 is ~0.1-1 uV next to 100 uV
slow components, so float16 steps (0.06 uV at 100 uV) would be a visible fraction of it.
The writer appends per recording, so peak memory is one recording (1st-study EAV OOM).
"""
import os, json, time, resource
import numpy as np, pandas as pd


class NcCacheWriter:
    def __init__(self, root, dataset, ch_names, length, ref_names=(), extra=None):
        self.dir = os.path.join(root, dataset); os.makedirs(os.path.join(self.dir, 'aux'), exist_ok=True)
        self.C, self.R, self.L, self.n, self.rows, self.t0 = len(ch_names), len(ref_names), length, 0, [], time.time()
        self.fx = open(os.path.join(self.dir, 'x.f32'), 'wb')
        self.fr = open(os.path.join(self.dir, 'ref.f32'), 'wb') if self.R else None
        self.header = dict(dataset=dataset, channels=self.C, length=length, sfreq=200, dtype='float32',
                           ch_names=list(ch_names), ref_names=list(ref_names), **(extra or {}))

    def add(self, x, rows, ref=None):
        assert x.shape[1:] == (self.C, self.L) and len(rows) == len(x), (x.shape, len(rows))
        assert np.isfinite(x).all()
        self.fx.write(np.ascontiguousarray(x, dtype=np.float32).tobytes())
        if self.R:
            assert ref is not None and ref.shape == (len(x), self.R, self.L), None if ref is None else ref.shape
            self.fr.write(np.ascontiguousarray(ref, dtype=np.float32).tobytes())
        self.rows.extend(rows); self.n += len(x)

    def add_aux(self, name, **arrays):
        np.savez(os.path.join(self.dir, 'aux', f'{name}.npz'), **arrays)

    def close(self, **more):
        self.fx.close()
        if self.fr: self.fr.close()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if os.uname().sysname == 'Darwin' else 1024)
        self.header.update(n=self.n, peak_rss_mb=round(peak, 1), seconds=round(time.time() - self.t0, 1), **more)
        json.dump(self.header, open(os.path.join(self.dir, 'header.json'), 'w'), indent=1)
        pd.DataFrame(self.rows).to_parquet(os.path.join(self.dir, 'meta.parquet'), index=False)
        return {k: v for k, v in self.header.items() if k not in ('raw_sha256',)}


class NcCache:
    """Read side: x [N, C, L] memmap, ref [N, R, L] memmap or None, meta DataFrame, header dict."""
    def __init__(self, root, dataset):
        self.dir = os.path.join(root, dataset)
        self.header = h = json.load(open(os.path.join(self.dir, 'header.json')))
        self.x = np.memmap(os.path.join(self.dir, 'x.f32'), dtype=np.float32, mode='r', shape=(h['n'], h['channels'], h['length']))
        R = len(h['ref_names'])
        self.ref = np.memmap(os.path.join(self.dir, 'ref.f32'), dtype=np.float32, mode='r', shape=(h['n'], R, h['length'])) if R else None
        self.meta = pd.read_parquet(os.path.join(self.dir, 'meta.parquet'))
        assert len(self.meta) == h['n']

    def __len__(self): return self.header['n']

    def window(self, kind):
        """(start, stop) sample slice of the task or null window inside the stored epoch, or None."""
        w = self.header['windows'].get(kind); return None if w is None else tuple(w['slice'])
