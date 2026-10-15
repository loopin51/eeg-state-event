"""D-106 G0-T meter W2V_LOW: wav2vec2-large-xlsr-53 hidden layers 1-4 (mean) of the ds004408 run audio, computed exactly as
w2v_run of scripts/stageA_prepare.py (16 kHz, standardised waveform, one pass per run, nearest interpolation to the 120-Hz
run length of the stage-A cache). Output: data/cache/d106/w2v_low.npy [1024, sum run lengths] (float16). RTX 4070 (gpuq).

usage: python scripts/d106_w2v_low.py configs/d106_gates.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageA_prepare import audio_16k


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); c = yaml.safe_load(open(a.config)); t0 = time.time()
    pc = yaml.safe_load(open(os.path.join(ROOT, c['stageA_prepare_config']))); meta = json.load(open(os.path.join(ROOT, pc['out_dir'], 'eeg_meta.json')))
    offs, lens = meta['run_offsets'], meta['run_lengths']; layers = c['g0t']['w2v_low_layers']
    from transformers import Wav2Vec2Model
    net = Wav2Vec2Model.from_pretrained(pc['audio']['wav2vec_model']).cuda().eval()
    out = os.path.join(ROOT, c['cache']); os.makedirs(out, exist_ok=True)
    F = np.lib.format.open_memmap(os.path.join(out, 'w2v_low.npy'), mode='w+', dtype=np.float16, shape=(1024, int(offs[-1] + lens[-1])))
    for r in range(20):
        wav, _ = audio_16k(pc, r + 1)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            x = torch.from_numpy(wav).cuda()[None]; x = (x - x.mean()) / (x.std() + 1e-7)
            h = torch.stack(net(x, output_hidden_states=True).hidden_states)[layers].float().mean(0)[0].T
        F[:, offs[r]:offs[r] + lens[r]] = torch.nn.functional.interpolate(h[None], lens[r])[0].cpu().numpy().astype(np.float16)
        print(f'run {r}: {lens[r]} frames ({time.time() - t0:.0f}s)', flush=True)
    F.flush(); json.dump(dict(layers=layers, device=torch.cuda.get_device_name(), max_memory_gb=torch.cuda.max_memory_allocated() / 1e9), open(os.path.join(out, 'w2v_low_meta.json'), 'w'))


if __name__ == '__main__':
    main()
