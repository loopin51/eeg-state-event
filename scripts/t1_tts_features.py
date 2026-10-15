"""D-104 T1: wav2vec2-large-xlsr-53 features of the TTS run renderings, exactly as w2v_run (scripts/stageA_prepare.py):
16 kHz polyphase resampling, waveform standardisation, mean of the configured hidden layers, nearest interpolation to
floor(duration x 120) frames, one pass per run. Output: data/cache/t1_tts/<voice>/w2v_run.npy [1024, sum frames] (float16)
and meta.json (run offsets, lengths, durations). RTX 4070 (gpuq).

usage: python scripts/t1_tts_features.py configs/t1_tts.yaml [--voice af_heart]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import json, time, argparse, yaml, numpy as np, torch, soundfile as sf
from scipy.signal import resample_poly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--voice'); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); voice = a.voice or c['tts']['voice']; f = c['features']; t0 = time.time()
    d = os.path.join(ROOT, c['cache'], voice); fs_out = 120
    from transformers import Wav2Vec2Model
    net = Wav2Vec2Model.from_pretrained(f['wav2vec_model']).cuda().eval(); layers = f['wav2vec_layers']
    wavs, durs = [], []
    for r in range(20):
        w, fs = sf.read(os.path.join(d, f'run{r:02d}.wav'), dtype='float64')
        w /= np.abs(w).max() + 1e-12; g = np.gcd(int(fs), 16000); wavs.append(resample_poly(w, 16000 // g, int(fs) // g).astype(np.float32)); durs.append(len(w) / fs)
    lens = [int(np.floor(x * fs_out)) for x in durs]; offs = np.r_[0, np.cumsum(lens)].astype(int)
    F = np.lib.format.open_memmap(os.path.join(d, 'w2v_run.npy'), mode='w+', dtype=np.float16, shape=(1024, int(offs[-1])))
    for r, wav in enumerate(wavs):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            x = torch.from_numpy(wav).cuda()[None]; x = (x - x.mean()) / (x.std() + 1e-7)
            h = torch.stack(net(x, output_hidden_states=True).hidden_states)[layers].float().mean(0)[0].T
        F[:, offs[r]:offs[r] + lens[r]] = torch.nn.functional.interpolate(h[None], lens[r])[0].cpu().numpy().astype(np.float16)
        print(f'{voice} run {r}: {durs[r]:.1f} s, {lens[r]} frames ({time.time() - t0:.0f}s)', flush=True)
    F.flush(); json.dump(dict(voice=voice, run_offsets=offs[:-1].tolist(), run_lengths=lens, durations_s=durs, sample_rate=fs_out, layers=layers,
                              model=f['wav2vec_model'], max_memory_gb=torch.cuda.max_memory_allocated() / 1e9, device=torch.cuda.get_device_name()),
                         open(os.path.join(d, 'meta.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
