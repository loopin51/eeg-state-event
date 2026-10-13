"""D-092 stage 0: Alice EEG caches for the REVE comparison (DRUM derivative, 33 subjects, one continuous recording each).

Segments: the 12 audio segments start at the annotations '1'..'12' of each recording. Every processed signal is computed on
the whole continuous recording, then segment k is cut from its onset for floor(duration_k * fs) samples and the 12 pieces
are stored back to back (segment = "run", shared offsets for all subjects). A segment without an onset annotation (S26,
S34, S35, S36: segment 1) or not fully recorded is stored as zeros and marked missing.
Sources (configs/alice_prepare.yaml): eeg (A′, no extra filter), eeg_hp05 (Défossez arm), reve (REVE arm: bad channels
interpolated, band-passed, 200 Hz, robust z-scored and clipped). Scaling statistics use the stored (present) segments.
Positions: 2D layout of the file montage scaled to [0, 1] (Défossez arm); 3D file positions mapped to the REVE position bank
frame by a similarity fit on standard-name sites (REVE arm), stopping if the mean anchor error exceeds the configured bound.
Audio at 120 Hz: wav2vec2-large-xlsr-53 hidden-state mean over the configured layers on each whole segment (w2v_seg) and on
each A′ block (w2v_block, brainmagick split_wav_as_block).
Split: A′ sentence blocks (blocks.tsv); main split (subject folds, segment folds, validation segments) and query counts with
chance levels (split.json).

usage: python scripts/alice_prepare.py configs/alice_prepare.yaml [--stage split|eeg|audio|all] [--smoke]
       --smoke: subjects S01 and S26 only, EEG and split stages, written to <out_dir>_smoke
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, re, json, time, random, hashlib, argparse, yaml, numpy as np, pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.stats import binom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def subjects(c):
    d = os.path.join(ROOT, c['raw_root'], 'eeg'); return sorted(int(m.group(1)) for x in os.listdir(d) if (m := re.fullmatch(r'S(\d\d)', x)))


def fif(c, s): return os.path.join(ROOT, c['raw_root'], 'eeg', f'S{s:02d}', f'S{s:02d}_alice-raw.fif')


def wav(c, k):
    fs, w = wavfile.read(os.path.join(ROOT, c['raw_root'], 'stimuli', f'{k}.wav')); w = w.astype(np.float64)
    return (w.mean(1) if w.ndim > 1 else w), int(fs)


def seg_durs(c):
    out = []
    for k in range(1, c['n_segments'] + 1):
        fs, w = wavfile.read(os.path.join(ROOT, c['raw_root'], 'stimuli', f'{k}.wav'), mmap=True); out.append(len(w) / fs)
    return out


def layout(c, durs, fs):
    lens = [int(np.floor(d * fs)) for d in durs]; return lens, np.r_[0, np.cumsum(lens)].astype(int)


def words(c):
    W = pd.read_csv(os.path.join(ROOT, c['raw_root'], 'stimuli', 'AliceChapterOne-EEG.csv'))
    return W.assign(seg=W.Segment - 1).sort_values(['seg', 'onset'], kind='stable').reset_index(drop=True)


# ---------------------------------------------------------------- split
def aprime_blocks(c, W, durs):
    """sentence blocks per segment (stage A make_blocks with segments as runs), hash-assigned as brainmagick."""
    s = c['split_aprime']; rows = []
    for r in range(c['n_segments']):
        w = W[W.seg == r]; on = w.onset.values; sent = w.Sentence.values; wd = w.Word.astype(str).values
        starts = [0] + [i for i in range(1, len(on)) if sent[i] != sent[i - 1]]; sents = []
        for j, a in enumerate(starts):
            b = starts[j + 1] if j + 1 < len(starts) else len(on)
            t0 = 0.0 if j == 0 else float(on[a]); t1 = float(on[b]) if b < len(on) else float(durs[r])
            sents.append(dict(start=t0, stop=t1, uid=f'seg{r + 1:02d}:' + ' '.join(wd[a:b])))
        cur, t_start = [], 0.0
        for j, se in enumerate(sents):
            cur.append(se['uid']); last = j == len(sents) - 1
            if last or se['stop'] > t_start + s['min_block_s']:
                rows.append(dict(run=r, start=t_start, stop=se['stop'], uid=','.join(cur))); cur, t_start = [], se['stop']
    B = pd.DataFrame(rows); ratios = [s['test_ratio'], s['valid_ratio']]; ratios.append(1 - sum(ratios)); cdf = np.cumsum(ratios)
    B['split'] = [('test', 'valid', 'train')[int(np.argmax(random.Random(int(hashlib.sha256(u.encode()).hexdigest(), 16) + s['seed']).random() < cdf))]
                  for u in B.uid]
    return B


def aprime_test_queries(c, B, W, durs):
    """stage A windows(): word onsets whose -0.5..2.5 s window lies in one contiguous test span and whose EEG is in the segment."""
    w = c['window']; q = []
    for r in range(c['n_segments']):
        b = B[B.run == r].sort_values('start'); sp = []
        for _, x in b.iterrows():
            if x.split != 'test': continue
            if sp and abs(sp[-1][1] - x.start) < 1e-6: sp[-1][1] = x.stop
            else: sp.append([x.start, x.stop])
        on = np.unique(W.onset[W.seg == r].values)
        for a, bb in sp:
            q += [(r, float(t)) for t in on[(on >= a - w['tmin']) & (on <= bb - w['tmax'])]
                  if t + w['tmin'] + w['offset_s'] >= 0 and t + w['tmax'] + w['offset_s'] <= durs[r]]
    return q


def main_queries(c, W, durs):
    """B1 rule: word onsets t with t - 0.5 >= 0 and t + 2.5 + 0.15 <= segment duration, per segment."""
    w = c['window']; out = {}
    for r in range(c['n_segments']):
        on = np.unique(W.onset[W.seg == r].values); hi = durs[r] - w['tmax'] - max(w['offset_s'], 0)
        out[r] = on[(on >= -w['tmin']) & (on <= hi)]
    return out


def split_stage(c, durs, subj, missing):
    out = os.path.join(ROOT, c['out_dir']); W = words(c); B = aprime_blocks(c, W, durs); B.to_csv(os.path.join(out, 'blocks.tsv'), sep='\t', index=False)
    np.savez(os.path.join(out, 'words.npz'), seg=W.seg.values, onset=W.onset.values, offset=W.offset.values, word=W.Word.astype(str).values,
             sentence=W.Sentence.values)
    dur = B.stop - B.start; q = aprime_test_queries(c, B, W, durs); qseg = np.array([r for r, _ in q])
    nseg = np.bincount(qseg, minlength=c['n_segments'])
    def chance(N, nr): return dict(N=int(N), chance_top10=10 / N, binom95_top10=float(binom.ppf(0.95, N, 10 / N) / N),
                                   seg_chance_top10=float(np.mean(np.minimum(10, nr) / nr)))
    ap = {'blocks': int(len(B)), 'block_median_s': float(dur.median()),
          'split_seconds': {k: float(dur[B.split == k].sum()) for k in ('train', 'valid', 'test')},
          'test_queries_per_segment': nseg.tolist(), 'all_segments': chance(len(q), nseg[qseg])}
    if any(missing.values()):   # subjects without segment 1: their queries and candidates exclude segment 1
        keep = qseg != 0; ap['without_segment1'] = chance(keep.sum(), nseg[qseg[keep]])
    sm = c['split_main']; perm = np.random.default_rng(sm['subject_seed']).permutation(subj); cut = np.cumsum([0] + sm['subject_fold_sizes'])
    assert cut[-1] == len(subj), (cut, len(subj))
    folds = [sorted(int(x) for x in perm[cut[i]:cut[i + 1]]) for i in range(len(sm['subject_fold_sizes']))]
    mq = main_queries(c, W, durs); per = [int(len(mq[r])) for r in range(c['n_segments'])]
    main = {'subject_folds': folds, 'segment_folds': sm['segment_folds'], 'valid_segment': sm['valid_segment'],
            'queries_per_segment': per, 'queries_total': int(sum(per)),
            'seg_chance_top10': float(np.sum([min(10, n) for n in per]) / sum(per)), 'seg_chance_mrp': 0.5,
            'fold_queries': [int(sum(per[k - 1] for k in f)) for f in sm['segment_folds']]}
    res = dict(subjects=[int(s) for s in subj], missing_segments={str(s): [k + 1 for k in v] for s, v in missing.items() if v},
               segment_durations=durs, aprime=ap, main=main)
    json.dump(res, open(os.path.join(out, 'split.json'), 'w'), indent=1)
    print('A′:', json.dumps(ap), '\nmain:', json.dumps(main), flush=True)


# ---------------------------------------------------------------- EEG
def similarity_fit(A, B):
    """Umeyama: s, R, t minimising |s R a + t - b|^2."""
    ma, mb = A.mean(0), B.mean(0); A0, B0 = A - ma, B - mb; U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / (A0 ** 2).sum(1).mean()
    return s, R, mb - s * R @ ma


def positions(c, raw, names):
    import mne
    cp = raw.get_montage().get_positions()['ch_pos']; P = np.array([cp[n] for n in names])
    info = mne.create_info(names, raw.info['sfreq'], 'eeg'); info.set_montage(raw.get_montage(), on_missing='raise')
    lay = mne.find_layout(info); xy = lay.pos[[lay.names.index(n) for n in names], :2]; xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    ra = c['reve_alignment']; tmpl = mne.channels.make_standard_montage('easycap-M10').get_positions()['ch_pos']
    bank = json.load(open(os.path.join(ROOT, c['reve_positions']))); src, dst, used = [], [], {}
    for site, name in ra['anchors'].items():
        d = np.linalg.norm(P - tmpl[site], axis=1); j = int(np.argmin(d))
        if d[j] * 1000 <= ra['site_match_mm']: src.append(P[j]); dst.append(bank[name]); used[names[j]] = name
    src, dst = np.array(src), np.array(dst); s, R, t = similarity_fit(src, dst); err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    fit = dict(anchors=used, scale=float(s), mean_error_m=float(err.mean()), max_error_m=float(err.max()))
    assert len(used) >= 6 and fit['mean_error_m'] <= ra['max_mean_error_m'], f'REVE position fit failed (D-092 stop rule): {fit}'
    return xy.astype(np.float32), ((s * (R @ P.T)).T + t).astype(np.float32), P, fit


def eeg_stage(c, durs, subj):
    import mne
    out = os.path.join(ROOT, c['out_dir']); e = c['eeg']; names = [str(i) for i in range(1, e['n_channels'] + 1)]; q = e['robust_quantiles']
    meta = dict(ch_names=names, segment_durations=durs, sources={}, subjects={}); pos_done = None
    for name, sc in e['sources'].items():
        lens, offs = layout(c, durs, sc['sample_rate']); meta['sources'][name] = dict(sc, run_offsets=offs.tolist(), run_lengths=lens)
    for s in subj:
        t0 = time.time(); raw = mne.io.read_raw_fif(fif(c, s), preload=True, verbose=False)
        ev, ids = mne.events_from_annotations(raw, verbose=False); inv = {v: k for k, v in ids.items()}; fs = raw.info['sfreq']
        onset = {int(inv[i]) - 1: (smp - raw.first_samp) / fs for smp, _, i in ev if inv[i].isdigit()}
        raw.pick(names); raw.reorder_channels(names); bads = [b for b in raw.info['bads'] if b in names]
        missing = [k for k in range(c['n_segments']) if k not in onset or (onset[k] + durs[k]) * fs > raw.n_times]
        P2, P3, Pfile, fit = positions(c, raw, names)
        if pos_done is None: pos_done = (P2, P3, Pfile, fit)
        else: assert np.allclose(Pfile, pos_done[2]), f'S{s:02d}: positions differ from the first subject'
        info = dict(bads=bads, bad_mask=[n in bads for n in names], missing_segments=[k + 1 for k in missing],
                    onsets_s={str(k + 1): float(v) for k, v in sorted(onset.items())}, recording_s=raw.n_times / fs, sources={})
        for name, sc in e['sources'].items():
            r = raw.copy()
            if sc.get('interpolate_bads') and bads:
                r.info['bads'] = list(bads); r.interpolate_bads(reset_bads=True, verbose=False)
            if sc.get('highpass_hz') or sc.get('lowpass_hz'):
                r.filter(l_freq=sc.get('highpass_hz') or None, h_freq=sc.get('lowpass_hz'), picks='all', n_jobs=1, verbose=False)   # zero-phase FIR
            x = r.get_data(); fo = sc['sample_rate']; g = np.gcd(int(fs), fo); y = resample_poly(x, fo // g, int(fs) // g, axis=-1)
            lens, offs = meta['sources'][name]['run_lengths'], meta['sources'][name]['run_offsets']
            Z = np.zeros((len(names), offs[-1]), np.float32); have = np.zeros(offs[-1], bool)
            for k in range(c['n_segments']):
                if k in missing: continue
                i0 = int(round(onset[k] * fo)); Z[:, offs[k]:offs[k + 1]] = y[:, i0:i0 + lens[k]]; have[offs[k]:offs[k + 1]] = True
            if sc['scaling'] == 'robust':
                med = np.median(Z[:, have], 1, keepdims=True); iqr = np.subtract(*np.percentile(Z[:, have], [q[1], q[0]], axis=1))[:, None]
                flat = iqr[:, 0] <= 1e-12; Z = (Z - med) / np.where(flat[:, None], 1.0, iqr)
            else:   # zscore: mean / SD (REVE specification); robust_z: median / (IQR / 1.349), equal to it for Gaussian data (D-092 supplement 3)
                if sc['scaling'] == 'robust_z':
                    mu = np.median(Z[:, have], 1, keepdims=True); sd = np.subtract(*np.percentile(Z[:, have], [75, 25], axis=1))[:, None] / 1.349
                else: mu, sd = Z[:, have].mean(1, keepdims=True), Z[:, have].std(1, keepdims=True)
                flat = sd[:, 0] <= 1e-12; sd_ratio = float(np.median(Z[:, have].std(1) / np.where(flat, 1.0, sd[:, 0])))   # plain SD / used SD
                Z = np.clip((Z - mu) / np.where(flat[:, None], 1.0, sd), -sc['clip_sd'], sc['clip_sd'])
            Z[:, ~have] = 0.0
            np.save(os.path.join(out, f'{name}_sub-{s:03d}.npy'), Z.astype(np.float32))
            info['sources'][name] = dict(flat=flat.tolist(), clipped_frac=float(np.mean(np.abs(Z[:, have]) >= sc.get('clip_sd', np.inf))))
            if sc['scaling'] != 'robust': info['sources'][name]['median_sd_ratio'] = sd_ratio
        meta['subjects'][str(s)] = info
        print(f'S{s:02d}: bads {len(bads)}, missing segments {[k + 1 for k in missing]}, recording {raw.n_times / fs:.1f} s ({time.time() - t0:.0f}s)', flush=True)
    P2, P3, Pfile, fit = pos_done; meta['reve_fit'] = fit
    np.save(os.path.join(out, 'positions.npy'), P2); np.save(os.path.join(out, 'positions_reve.npy'), P3); np.save(os.path.join(out, 'positions_file.npy'), Pfile)
    json.dump(meta, open(os.path.join(out, 'eeg_meta.json'), 'w'), indent=1); print('REVE position fit', json.dumps(fit), flush=True)


# ---------------------------------------------------------------- audio
def audio_stage(c, durs):
    import torch
    from transformers import Wav2Vec2Model
    out = os.path.join(ROOT, c['out_dir']); a = c['audio']; fo = a['feature_rate']; dt = getattr(np, a['feature_dtype'])
    lens, offs = layout(c, durs, fo); B = pd.read_csv(os.path.join(out, 'blocks.tsv'), sep='\t')
    net = Wav2Vec2Model.from_pretrained(a['wav2vec_model']).cuda().eval(); layers = a['wav2vec_layers']

    def w2v(x, n_out):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            x = torch.from_numpy(x).cuda()[None]; x = (x - x.mean()) / (x.std() + 1e-7)
            h = torch.stack(net(x, output_hidden_states=True).hidden_states)[layers].float().mean(0)[0].T
        return torch.nn.functional.interpolate(h[None], n_out)[0].cpu().numpy()

    feats = {k: np.lib.format.open_memmap(os.path.join(out, f'{k}.npy'), 'w+', dt, (1024, int(offs[-1]))) for k in ('w2v_seg', 'w2v_block')}
    t0 = time.time()
    for r in range(c['n_segments']):
        w, fs = wav(c, r + 1); w /= np.abs(w).max() + 1e-12; g = np.gcd(fs, a['sample_rate'])
        w = resample_poly(w, a['sample_rate'] // g, fs // g).astype(np.float32); L = lens[r]; o = offs[r]; w = w[:int(L / fo * a['sample_rate'])]
        feats['w2v_seg'][:, o:o + L] = w2v(w, L).astype(dt)
        for _, b in B[B.run == r].iterrows():
            i0, i1 = int(round(b.start * fo)), min(int(round(b.stop * fo)), L); seg = w[int(i0 / fo * a['sample_rate']):int(i1 / fo * a['sample_rate'])]
            if i1 - i0 < 2 or len(seg) < 400: continue
            feats['w2v_block'][:, o + i0:o + i1] = w2v(seg, i1 - i0).astype(dt)
        print(f'audio segment {r + 1}: {L / fo:.1f} s, {int((B.run == r).sum())} blocks ({time.time() - t0:.0f}s)', flush=True)
    for m in feats.values(): m.flush()
    json.dump(dict(run_offsets=offs.tolist(), run_lengths=lens, sample_rate=fo), open(os.path.join(out, 'audio_meta.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', default='all', choices=['split', 'eeg', 'audio', 'all'])
    ap.add_argument('--smoke', action='store_true'); args = ap.parse_args(); c = yaml.safe_load(open(args.config))
    if args.smoke: c['out_dir'] += '_smoke'; c['split_main']['subject_fold_sizes'] = [1, 1, 0, 0, 0]; args.stage = 'eeg'
    out = os.path.join(ROOT, c['out_dir']); os.makedirs(out, exist_ok=True)
    durs = seg_durs(c); subj = [1, 26] if args.smoke else subjects(c); json.dump(dict(config=c, durations=durs), open(os.path.join(out, 'prepare_config.json'), 'w'), indent=1)
    if args.stage in ('eeg', 'all'): eeg_stage(c, durs, subj)
    meta = json.load(open(os.path.join(out, 'eeg_meta.json')))
    missing = {int(s): [k - 1 for k in v['missing_segments']] for s, v in meta['subjects'].items()}
    split_stage(c, durs, subj, missing)
    if args.stage in ('audio', 'all'): audio_stage(c, durs)


if __name__ == '__main__':
    main()
