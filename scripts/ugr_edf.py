"""D-110 section 1: sealed EDF reader for UGR-MINDVOICE. Only samples inside the registered read ranges R are decoded.

The EDF header (labels, record length, samples per record, scaling) is read first. For each R block [s, e) (1-kHz sample
index, = EDF sample index) the records lying wholly inside the block are read in one piece; the partial records at the block
edges are read channel by channel, only the samples inside [s, e). No sample outside R enters memory, so the decoded range of a
block is exactly [s, e). Before anything is decoded, ugr_seal.check() verifies the three registered sha256 values of
results/ugr/seal.json and that every planned range lies inside R and away from the sealed ranges +- margin.
filter_input_check() is called by every filter on its input array (interpretation record of D-110: block-internal filtering with
reflect padding depends on R samples only).

usage (library): blocks = read_session(cfg, 'sub-04/ses-01', channels) -> list of dict(start, end, data [n_ch, e - s] uV)
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ugr_seal import check
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def header(path):
    with open(path, 'rb') as f:
        h = f.read(256); ns = int(h[252:256]); hb = int(h[184:192]); e = f.read(ns * 256)
    def fld(start, w): return [e[start * ns + k * w:start * ns + (k + 1) * w].decode('latin-1').strip() for k in range(ns)]
    # per-signal fields: label 16, transducer 80, dimension 8, pmin 8, pmax 8, dmin 8, dmax 8, prefilter 80, samples 8, reserved 32
    lab, dim = fld(0, 16), fld(96, 8); pmin, pmax, dmin, dmax = (np.array(fld(o, 8), float) for o in (104, 112, 120, 128))
    spr = np.array(fld(216, 8), int); off = np.r_[0, np.cumsum(spr)][:-1]
    return dict(labels=lab, dim=dim, n_records=int(h[236:244]), rec_dur=float(h[244:252]), header_bytes=hb, spr=spr, offsets=off,
                rec_bytes=int(2 * spr.sum()), gain=(pmax - pmin) / (dmax - dmin), pmin=pmin, dmin=dmin)


def edf_path(cfg, session):
    sub, ses = session.split('/'); return os.path.join(cfg['data_root'], 'eeg', sub, ses, 'eeg', f'{sub}_{ses}_task-VCV_run-01_eeg.edf')


def read_range(path, h, s, e, ch):
    """decode samples [s, e) of channels ch (indices) in physical units; partial edge records are read per channel"""
    n = int(h['spr'][ch[0]]); assert all(h['spr'][c] == n for c in ch) and n == round(1000 * h['rec_dur']), 'channels must be 1 kHz'
    hb, rb, off = h['header_bytes'], h['rec_bytes'], h['offsets']; out = np.empty((len(ch), e - s), np.float32)
    rs, re = -(-s // n), e // n; assert rs <= re, 'block shorter than one record'
    with open(path, 'rb') as f:
        if rs < re:
            f.seek(hb + rs * rb); buf = np.frombuffer(f.read((re - rs) * rb), '<i2').reshape(re - rs, rb // 2)
            for i, c in enumerate(ch): out[i, rs * n - s:re * n - s] = buf[:, off[c]:off[c] + n].reshape(-1)
        edges = ([(rs - 1, s - (rs - 1) * n, n)] if s % n else []) + ([(re, 0, e - re * n)] if e % n else [])
        for r, j0, j1 in edges:
            for i, c in enumerate(ch):
                f.seek(hb + r * rb + 2 * (off[c] + j0)); seg = np.frombuffer(f.read(2 * (j1 - j0)), '<i2')
                out[i, r * n + j0 - s:r * n + j1 - s] = seg
    g, pmin, dmin = h['gain'][ch][:, None], h['pmin'][ch][:, None], h['dmin'][ch][:, None]
    return ((out - dmin) * g + pmin).astype(np.float32)


def read_session(cfg, session, channels):
    """all R blocks of a session for the named channels (uV). Seal sha256 and planned ranges are checked before decoding."""
    seal = json.load(open(os.path.join(ROOT, cfg['seal']['out']))); ranges = [tuple(r) for r in seal['sessions'][session]['real']]
    check(seal, cfg['seal_registered'], session, ranges)
    p = edf_path(cfg, session); h = header(p); idx = {l: i for i, l in enumerate(h['labels'])}; ch = [idx[c] for c in channels]
    assert all(h['dim'][c] in ('uV', 'µV') for c in ch), 'EEG channels must be in uV'
    assert ranges[-1][1] <= h['n_records'] * int(h['spr'][ch[0]]), 'read range beyond the recording'
    return [dict(start=s, end=e, data=read_range(p, h, s, e, ch)) for s, e in ranges], ranges, h


def filter_input_check(x, block, decoded):
    """the array entering a filter covers exactly one decoded R block"""
    assert (block['start'], block['end']) in decoded, 'filter input is not a decoded R block'
    assert x.shape[-1] == block['end'] - block['start'], 'filter input length differs from the decoded block'
