"""F-V (D-111): image_metadata.npy (pickled dict) -> data/cache/things_eeg2/image_meta.json through the restricted unpickler.

The JSON keeps every key of the file as plain lists of strings, plus the sha256 of the source; data loaders and the k3s jobs
read only this JSON (no pickle on the cluster).

usage: python scripts/fv_meta.py configs/fv_preprocess.yaml
"""
import os, sys, json, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.things import rawio


def main():
    c = yaml.safe_load(open(os.path.join(ROOT, sys.argv[1]))); src = os.path.join(ROOT, c['meta_source'])
    d = rawio.load_dict(src, 'meta'); out = {k: [str(s) for s in list(v)] for k, v in d.items()}
    out['source'] = dict(file=c['meta_source'], sha256=rawio.sha256(src))
    p = os.path.join(ROOT, c['cache_dir'], 'image_meta.json'); os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(out, open(p, 'w'), indent=0)
    print(json.dumps({k: (len(v), v[:2]) for k, v in out.items() if k != 'source'}), out['source'], flush=True)


if __name__ == '__main__':
    main()
