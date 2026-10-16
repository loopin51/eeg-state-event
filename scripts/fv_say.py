"""F-V display output (D-111 decision 1): render one fixed sentence with Kokoro-82M 0.9.4 (same model, revision, sha256 check and
pipeline as scripts/t1_tts_render.py, configs/t1_tts.yaml). Display-only post-processing; never part of a verdict.
Runs in env_tts on CPU. Prints one JSON line (wav path, sha256, duration, voice, model sha256).

usage: env_tts/bin/python scripts/fv_say.py configs/t1_tts.yaml --text 'I see a dog.' --out out.wav [--voice af_heart]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import argparse, hashlib, json
import numpy as np, yaml

SR = 24000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--text', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--voice'); a = ap.parse_args()
    t = yaml.safe_load(open(a.config))['tts']; voice = a.voice or t['voice']
    import torch; torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    from huggingface_hub import snapshot_download
    d = snapshot_download(t['model_repo'], revision=t['model_revision'], allow_patterns=['config.json', t['model_file'], f'voices/{voice}.pt'])
    sha = hashlib.sha256(open(os.path.join(d, t['model_file']), 'rb').read()).hexdigest(); assert sha == t['model_sha256'], sha
    import kokoro, misaki; assert kokoro.__version__ == '0.9.4' and misaki.__version__ == '0.9.4', (kokoro.__version__, misaki.__version__)
    from kokoro import KModel, KPipeline
    import soundfile as sf
    model = KModel(repo_id=t['model_repo'], config=os.path.join(d, 'config.json'), model=os.path.join(d, t['model_file'])).eval()
    pipe = KPipeline(lang_code=t['lang_code'], repo_id=t['model_repo'], model=model)
    audio = []
    with torch.no_grad():
        for res in pipe(a.text, voice=os.path.join(d, 'voices', f'{voice}.pt'), speed=t['speed'], split_pattern=t['split_pattern']):
            audio.append(res.audio.numpy() if hasattr(res.audio, 'numpy') else np.asarray(res.audio))
    audio = np.concatenate(audio); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sf.write(a.out, audio, SR, subtype='PCM_16')
    print(json.dumps(dict(wav=a.out, sha256=hashlib.sha256(open(a.out, 'rb').read()).hexdigest(), duration_s=round(len(audio) / SR, 3),
                          voice=voice, text=a.text, model_sha256=sha, sample_rate=SR)))


if __name__ == '__main__':
    main()
