"""Stage A cache for the Défossez et al. (2023) reproduction on ds004408 (D-065, faculty direction 1).

EEG (per subject): 128 channels of each run, cut to the audio length (recording start = audio start, README),
resampled 512 -> 120 Hz (polyphase), no filtering and no re-referencing (the released brainmagick loader uses the
Broderick .mat data as is), robust-scaled per run and channel ((x - median) / IQR). Stored unreferenced so that a common
average can be applied at load time. Bad channels from channels.tsv are stored as a mask but kept (the released code
has no bad-channel handling). Flat channels (IQR = 0) are marked invalid.
Split (shared by all subjects, as every subject heard the same audio): sentences from the restored sentence ends,
spanning from their first word onset to the next sentence's first word onset (first block starts at 0, last ends at the
audio end), merged until longer than min_block_s, then assigned to test / valid / train by sha256(uid) + seed exactly
as brainmagick.events.assign_blocks.
Audio features at 120 Hz: wav2vec2-large-xlsr-53 hidden-state mean over the configured layers, computed separately on
the audio of each merged block (brainmagick split_wav_as_block, so no feature sees audio of another split) and also on
each whole run (for stage B run-level splits); 120-band log-mel spectrogram (n_fft 512, hop 128, normalised STFT).

usage: python scripts/stageA_prepare.py configs/stageA_prepare.yaml [--stage eeg|audio|all]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, random, hashlib, argparse, yaml, numpy as np, pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def positions_biosemi128(ch_names):
    import mne
    info = mne.create_info(list(ch_names), 120.0, 'eeg'); info.set_montage('biosemi128')
    lay = mne.find_layout(info); idx = [lay.names.index(n) for n in ch_names]; xy = lay.pos[idx, :2]
    return ((xy - xy.min(0)) / (xy.max(0) - xy.min(0))).astype(np.float32)


def audio_16k(c, k):
    fs, w = wavfile.read(os.path.join(ROOT, c['raw_root'], 'stimuli', f'audio{k:02d}.wav')); w = w.astype(np.float64)
    if w.ndim > 1: w = w.mean(1)
    w /= np.abs(w).max() + 1e-12          # scale is irrelevant: wav2vec input is standardised, log-mel is standardised later
    g = np.gcd(int(fs), c['audio']['sample_rate']); return resample_poly(w, c['audio']['sample_rate'] // g, int(fs) // g).astype(np.float32), len(w) / fs


def audio_dur(c, k):
    fs, w = wavfile.read(os.path.join(ROOT, c['raw_root'], 'stimuli', f'audio{k:02d}.wav'), mmap=True); return len(w) / fs


def make_blocks(c, durs):
    W = np.load(os.path.join(ROOT, c['word_features']), allow_pickle=True); s = c['split']; rows = []
    for r in range(c['n_runs']):
        m = W['run'] == r; on = W['onset'][m]; words = W['word'][m]; end = W['sent_end_punct'][m]
        starts = [0] + [i + 1 for i in np.where(end)[0] if i + 1 < len(on)]
        sents = []
        for j, a in enumerate(starts):
            b = starts[j + 1] if j + 1 < len(starts) else len(on)
            t0 = 0.0 if j == 0 else float(on[a]); t1 = float(on[b]) if b < len(on) else float(durs[r])
            sents.append(dict(start=t0, stop=t1, uid=f'audio{r + 1:02d}:' + ' '.join(words[a:b])))
        cur, t_start = [], 0.0                        # brainmagick events._merge_blocks
        for j, se in enumerate(sents):
            cur.append(se['uid']); last = j == len(sents) - 1
            if last or se['stop'] > t_start + s['min_block_s']:
                rows.append(dict(run=r, start=t_start, stop=se['stop'], uid=','.join(cur))); cur, t_start = [], se['stop']
    B = pd.DataFrame(rows); ratios = [s['test_ratio'], s['valid_ratio']]; ratios.append(1 - sum(ratios)); cdf = np.cumsum(ratios)
    split = []
    for uid in B.uid:                                 # brainmagick events.assign_blocks
        score = random.Random(int(hashlib.sha256(uid.encode()).hexdigest(), 16) + s['seed']).random()
        split.append(int(np.argmax(score < cdf)))
    B['split'] = pd.Series(split).map({0: 'test', 1: 'valid', 2: 'train'}); return B


def eeg_stage(c, durs):
    out = os.path.join(ROOT, c['out_dir']); fs_out = c['eeg']['sample_rate']; q = c['eeg']['robust_quantiles']; import mne
    name = c['eeg'].get('name', 'eeg'); hp = c['eeg'].get('highpass_hz', 0)   # variants (D-078): e.g. eeg_hp05 = 0.5 Hz highpass
    lens = [int(np.floor(d * fs_out)) for d in durs]; offs = np.r_[0, np.cumsum(lens)]; meta = {}
    for sub in c['subjects']:
        t0 = time.time(); mm = np.lib.format.open_memmap(os.path.join(out, f'{name}_sub-{sub:03d}.npy'), 'w+', np.float32, (128, int(offs[-1])))
        bad = np.zeros((c['n_runs'], 128), bool); flat = np.zeros((c['n_runs'], 128), bool); names = None
        for r in range(c['n_runs']):
            base = os.path.join(ROOT, c['raw_root'], f'sub-{sub:03d}', 'eeg', f'sub-{sub:03d}_task-listening_run-{r + 1:02d}')
            raw = mne.io.read_raw_brainvision(base + '_eeg.vhdr', preload=True, verbose=False); fs = int(raw.info['sfreq'])
            if hp: raw.filter(l_freq=hp, h_freq=None, picks='all', n_jobs=1, verbose=False)   # zero-phase FIR at the native rate
            assert names is None or raw.ch_names == names; names = raw.ch_names
            ch = pd.read_csv(base + '_channels.tsv', sep='\t')
            if 'status' in ch: bad[r] = ch.set_index('name').loc[names, 'status'].values == 'bad'
            x = raw.get_data()[:, :int(np.ceil(durs[r] * fs)) + fs]
            g = np.gcd(fs, fs_out); y = resample_poly(x, fs_out // g, fs // g, axis=-1)[:, :lens[r]]
            assert y.shape[1] == lens[r], f'sub-{sub:03d} run {r + 1}: EEG shorter than audio'
            med = np.median(y, 1, keepdims=True); iqr = np.subtract(*np.percentile(y, [q[1], q[0]], axis=1))[:, None]
            flat[r] = iqr[:, 0] <= 1e-12; y = (y - med) / np.where(flat[r][:, None], 1.0, iqr)
            mm[:, offs[r]:offs[r + 1]] = y.astype(np.float32)
        mm.flush(); del mm
        meta[sub] = dict(bad=bad.tolist(), flat=flat.tolist())
        print(f'sub-{sub:03d}: bad per run {bad.sum(1).tolist()}, flat {int(flat.sum())} ({time.time() - t0:.0f}s)', flush=True)
    if name != 'eeg':                                   # variants share offsets, positions and masks with the base cache
        json.dump(meta, open(os.path.join(out, f'{name}_meta.json'), 'w')); return
    np.save(os.path.join(out, 'positions.npy'), positions_biosemi128(names))
    json.dump(dict(ch_names=list(names), run_offsets=offs.tolist(), run_lengths=lens, sample_rate=fs_out, subjects=meta),
              open(os.path.join(out, 'eeg_meta.json'), 'w'))


def audio_stage(c, B, durs):
    import torch, torchaudio
    from transformers import Wav2Vec2Model
    out = os.path.join(ROOT, c['out_dir']); a = c['audio']; fs_out = c['eeg']['sample_rate']; dt = getattr(np, a['feature_dtype'])
    lens = [int(np.floor(d * fs_out)) for d in durs]; offs = np.r_[0, np.cumsum(lens)]
    net = Wav2Vec2Model.from_pretrained(a['wav2vec_model']).cuda().eval(); layers = a['wav2vec_layers']
    melspec = torchaudio.transforms.MelSpectrogram(sample_rate=a['sample_rate'], n_fft=a['mel']['n_fft'], hop_length=a['mel']['hop'],
                                                   n_mels=a['mel']['n_mels'], normalized=True).cuda()

    def w2v(wav, n_out):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            x = torch.from_numpy(wav).cuda()[None]; x = (x - x.mean()) / (x.std() + 1e-7)   # Wav2Vec2FeatureExtractor normalisation
            h = torch.stack(net(x, output_hidden_states=True).hidden_states)[layers].float().mean(0)[0].T   # [1024, frames]
        return torch.nn.functional.interpolate(h[None], n_out)[0].cpu().numpy()          # nearest, as brainmagick

    def mel(wav, n_out):
        with torch.no_grad():
            m = torch.log(a['mel']['log_eps'] + melspec(torch.from_numpy(wav).cuda()))
        return torch.nn.functional.interpolate(m[None], n_out)[0].cpu().numpy()

    feats = {k: np.lib.format.open_memmap(os.path.join(out, f'{k}.npy'), 'w+', dt, (d, int(offs[-1])))
             for k, d in (('w2v_block', 1024), ('w2v_run', 1024), ('mel_block', a['mel']['n_mels']), ('mel_run', a['mel']['n_mels']))}
    t0 = time.time()
    for r in range(c['n_runs']):
        wav, _ = audio_16k(c, r + 1); o = offs[r]; L = lens[r]; wav = wav[:int(L / fs_out * a['sample_rate'])]
        feats['w2v_run'][:, o:o + L] = w2v(wav, L).astype(dt); feats['mel_run'][:, o:o + L] = mel(wav, L).astype(dt)
        for _, b in B[B.run == r].iterrows():
            i0, i1 = int(round(b.start * fs_out)), min(int(round(b.stop * fs_out)), L)
            seg = wav[int(i0 / fs_out * a['sample_rate']):int(i1 / fs_out * a['sample_rate'])]
            if i1 - i0 < 2 or len(seg) < 400: continue
            feats['w2v_block'][:, o + i0:o + i1] = w2v(seg, i1 - i0).astype(dt); feats['mel_block'][:, o + i0:o + i1] = mel(seg, i1 - i0).astype(dt)
        print(f'audio run {r + 1}: {len(B[B.run == r])} blocks ({time.time() - t0:.0f}s)', flush=True)
    for m in feats.values(): m.flush()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', default='all', choices=['split', 'eeg', 'audio', 'all']); args = ap.parse_args()
    c = yaml.safe_load(open(args.config)); out = os.path.join(ROOT, c['out_dir']); os.makedirs(out, exist_ok=True)
    durs = [audio_dur(c, k) for k in range(1, c['n_runs'] + 1)]
    B = make_blocks(c, durs); B.to_csv(os.path.join(out, 'blocks.tsv'), sep='\t', index=False)
    dur = B.stop - B.start
    print(f"blocks: {len(B)} (median {dur.median():.1f} s); split seconds " +
          ', '.join(f'{k} {dur[B.split == k].sum():.0f}' for k in ('train', 'valid', 'test')), flush=True)
    json.dump(dict(config=c, durations=durs), open(os.path.join(out, 'prepare_config.json'), 'w'), indent=1)
    if args.stage in ('eeg', 'all'): eeg_stage(c, durs)
    if args.stage in ('audio', 'all'): audio_stage(c, B, durs)


if __name__ == '__main__':
    main()
