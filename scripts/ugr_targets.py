"""D-110 2.4: wav2vec2 targets of the UGR-MINDVOICE cue wavs (stimulus only, no EEG). RTX 4070 (gpuq).

Every Materials wav is placed at cue_at_s inside a clip_s silent clip (cue onset = wav start, truncated at the clip end), then
processed exactly as w2v_run (scripts/stageA_prepare.py, scripts/t1_tts_features.py): peak scaling, 16 kHz polyphase resampling,
waveform standardisation, mean of the configured hidden layers, nearest interpolation to clip_s x sample_rate frames.
Output: <cache>/cue_targets.npy [n_wav, frames, 1024] (float16) and cue_targets.json (wav stems in row order, wav sha256, meta).

usage: python scripts/ugr_targets.py configs/ugr_u.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import json, time, hashlib, argparse, yaml, numpy as np, torch, soundfile as sf
from scipy.signal import resample_poly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); t0 = time.time()
    c = yaml.safe_load(open(os.path.join(ROOT, a.config))); tg = c['target']; fs_out = c['window']['sample_rate']
    M = os.path.join(c['data_root'], c['materials']); stems = sorted(f[:-4] for f in os.listdir(M) if f.lower().endswith('.wav'))
    n_out = int(round(tg['clip_s'] * fs_out)); out = os.path.join(ROOT, c['cache']); os.makedirs(out, exist_ok=True)
    from transformers import Wav2Vec2Model
    net = Wav2Vec2Model.from_pretrained(tg['model']).cuda().eval(); layers = tg['layers']
    F = np.lib.format.open_memmap(os.path.join(out, 'cue_targets.npy'), mode='w+', dtype=np.float16, shape=(len(stems), n_out, 1024))
    sha, wav_s = {}, {}
    for i, s in enumerate(stems):
        p = os.path.join(M, s + '.wav'); sha[s] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        w, fs = sf.read(p, dtype='float64'); w = w.mean(1) if w.ndim == 2 else w; wav_s[s] = len(w) / fs
        clip = np.zeros(int(round(tg['clip_s'] * fs))); i0 = int(round(tg['cue_at_s'] * fs)); seg = w[:len(clip) - i0]; clip[i0:i0 + len(seg)] = seg
        clip /= np.abs(clip).max() + 1e-12; g = np.gcd(int(fs), 16000); x16 = resample_poly(clip, 16000 // g, int(fs) // g).astype(np.float32)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            x = torch.from_numpy(x16).cuda()[None]; x = (x - x.mean()) / (x.std() + 1e-7)
            h = torch.stack(net(x, output_hidden_states=True).hidden_states)[layers].float().mean(0)[0].T
        F[i] = torch.nn.functional.interpolate(h[None], n_out)[0].T.cpu().numpy().astype(np.float16)
    F.flush()
    json.dump(dict(stems=stems, wav_sha256=sha, wav_s=wav_s, fs=fs_out, frames=n_out, clip_s=tg['clip_s'], cue_at_s=tg['cue_at_s'], layers=layers,
                   model=tg['model'], max_memory_gb=torch.cuda.max_memory_allocated() / 1e9, device=torch.cuda.get_device_name(), seconds=time.time() - t0),
              open(os.path.join(out, 'cue_targets.json'), 'w'), indent=1)
    print(f'{len(stems)} wavs, {n_out} frames, longest wav {max(wav_s.values()):.3f} s, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
