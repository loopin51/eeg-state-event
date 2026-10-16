"""D-113 arm c candidates (6.3): name text embeddings of the 10 Gao items (k3s RTX 5080, data group fv_weights).

The text stage of scripts/fv_embed.py (run_text: the same CLIP weights with their sha256 check, prompt p1 and the 80-template
p2, L2-normalised float32) on the name table configs/fv_gao_c_names.tsv instead of configs/fv_names.tsv. Gao's scissor uses
the THINGS name scissors. The 7 THINGS rows are recomputed only to check them against names_text_P1 / P2 in the scoring job
(the candidates of those 7 items are the names_text rows, D-113 6.3); the 3 figure rows (circle, pentagram, square) are new.
Output: results/runs/<date>_fv_gao_c_names/ (clip_txt_p1.npy, clip_txt_p2.npy [10, 1024], names.json, metrics.json, config.yaml).
usage (k3s): python scripts/fv_gao_c_names.py configs/fv_gao_c.yaml
"""
import os, sys, json, time, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import fv_embed                                                          # sets the thread and HF_HUB_OFFLINE defaults


def main():
    c = yaml.safe_load(open(os.path.join(ROOT, sys.argv[1]))); e = yaml.safe_load(open(os.path.join(ROOT, c['names']['embed_config'])))
    e = dict(e, names=c['names']['table'])
    import torch, open_clip, numpy as np
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; assert torch.cuda.is_available()
    out = os.path.join(ROOT, 'results', 'runs', time.strftime('%Y-%m-%d') + '_fv_gao_c_names'); os.makedirs(out, exist_ok=True)
    info = fv_embed.run_text(e, out, torch, open_clip)
    info.update(device=torch.cuda.get_device_name(0), torch=torch.__version__, open_clip=open_clip.__version__, numpy=np.__version__,
                config=sys.argv[1], embed_config=c['names']['embed_config'],
                files={f: fv_embed.sha256(os.path.join(out, f)) for f in sorted(os.listdir(out)) if f.endswith(('.npy', '.json'))})
    json.dump(info, open(os.path.join(out, 'metrics.json'), 'w'), indent=1); yaml.safe_dump(e, open(os.path.join(out, 'config.yaml'), 'w'), sort_keys=False)
    print(json.dumps({k: v for k, v in info.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    main()
