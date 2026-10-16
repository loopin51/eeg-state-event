"""Install a fetched fv_embed run folder into the server caches (D-111; no new computation).

1. Copies the arrays of results/runs/<run>/ into <embed_out> (data/cache/fv_clip/) after checking every file against the
   sha256 list in the run's metrics.json; metrics.json is kept as metrics_<stage>.json.
2. Writes the decoder-target copy for the training code (eeg-b1 format) in data/cache/things_eeg2/clip_vith14/:
   train_image.npy [16540, 1024] and test_image.npy [200, 1024] in image_metadata.npy order (train_img_files / test_img_files,
   event code k -> row k-1), names_text_P1.npy / names_text_P2.npy [1854, 1024] in configs/fv_names.tsv row order, index.json.
   The image order is matched path by path against image_metadata.npy (read with the restricted loader); any mismatch stops.

usage: python scripts/fv_install_embed.py configs/fv_eegfree.yaml results/runs/<date>_fv_embed_{image,text}
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
import hashlib, json, shutil, sys
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
META = 'data/raw/things_eeg2/image_metadata.npy'
TARGET = 'data/cache/things_eeg2/clip_vith14'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def main():
    c = yaml.safe_load(open(os.path.join(ROOT, sys.argv[1]))); run = os.path.join(ROOT, sys.argv[2]).rstrip('/')
    m = json.load(open(os.path.join(run, 'metrics.json'))); stage = m['stage']; assert stage in ('image', 'text'), stage
    E = os.path.join(ROOT, c['embed_out']); os.makedirs(E, exist_ok=True)
    for f, h in m['files'].items():
        assert sha256(os.path.join(run, f)) == h, f'{f}: sha256 differs from the run metrics'
        dst = os.path.join(E, f)
        if os.path.exists(dst): assert sha256(dst) == h, f'{dst} exists with other content'
        else: shutil.copy2(os.path.join(run, f), dst)
    shutil.copy2(os.path.join(run, 'metrics.json'), os.path.join(E, f'metrics_{stage}.json'))
    T = os.path.join(ROOT, TARGET); os.makedirs(T, exist_ok=True)
    ip = os.path.join(T, 'index.json'); index = json.load(open(ip)) if os.path.exists(ip) else {}
    common = dict(model='ViT-H-14', pretrained_tag=c['clip']['pretrained_tag'], weights_sha256=m['clip_weights_sha256'],
                  open_clip=m['open_clip'], torch=m['torch'], device=m['device'])
    if stage == 'image':
        from ese.things import rawio
        meta = rawio.load_dict(os.path.join(ROOT, META), 'meta'); idx = json.load(open(os.path.join(E, 'index.json')))
        pos = {p: i for i, p in enumerate(idx['paths'])}; I = np.load(os.path.join(E, 'clip_img.npy'))
        rows = {}
        for sp, key in (('train', 'training_images'), ('test', 'test_images')):
            want = [f'{key}/{cc}/{ff}' for cc, ff in zip(meta[f'{sp}_img_concepts'], meta[f'{sp}_img_files'])]
            miss = [w for w in want if w not in pos]; assert not miss, f'{sp}: {len(miss)} metadata paths not embedded, e.g. {miss[:3]}'
            r = np.array([pos[w] for w in want]); rows[sp] = r
            out = os.path.join(T, f'{sp}_image.npy'); np.save(out, I[r])
            index[f'{sp}_image'] = dict(file=f'{sp}_image.npy', sha256=sha256(out), shape=list(I[r].shape),
                                        order=f'image_metadata {sp}_img_files (event code k -> row k-1)', paths=want,
                                        image_sha256=[idx['sha256'][i] for i in r], **common, preprocess=m['clip_preprocess'])
        assert len(set(rows['train'])) == 16540 and len(set(rows['test'])) == 200
        index['image_metadata_sha256'] = sha256(os.path.join(ROOT, META))
    else:
        for p in ('p1', 'p2'):
            out = os.path.join(T, f'names_text_{p.upper()}.npy'); shutil.copy2(os.path.join(E, f'clip_txt_{p}.npy'), out)
            index[f'names_text_{p.upper()}'] = dict(file=os.path.basename(out), sha256=sha256(out), order='configs/fv_names.tsv row order',
                                                     prompt=c['prompts'][p], names_sha256=m['names_sha256'], **common)
        index.setdefault('g0_pass_prompt', None)    # filled after the G0 job (scripts/fv_gate.py)
    json.dump(index, open(ip, 'w'))
    print(json.dumps({k: (v['sha256'] if isinstance(v, dict) else v) for k, v in index.items()}), flush=True)


if __name__ == '__main__':
    main()
