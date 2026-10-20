"""F-IX (D-119 amendment 1 D(ii)): OpenCLIP ViT-H-14 text embeddings of the method S prototype prompts (18 category prompts, 11
colour prompts), L2-normalised float32. k3s RTX 5080 (data group fv_langemb, same encoder loader as fv_lang_embed.py).
Output: results/runs/<date>_fv_s_text/{prompt_text.npy [29, 1024], prompts.json}.
usage: python scripts/fv_s_text.py configs/fv_s.yaml
"""
import os, sys, json, time, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def prompts(c):
    cat = sorted(c['categories']['phrase']); col = c['colour']['names']
    return ([('category', k, c['categories']['prompt'].format(phrase=c['categories']['phrase'][k])) for k in cat] +
            [('colour', k, c['colour_prompt'].format(colour='gray' if k == 'grey' else k)) for k in col])


def main():
    import torch, open_clip
    from fv_lang_embed import load_text_encoder, encode, fsha
    c = yaml.safe_load(open(P(sys.argv[1]))); lc = yaml.safe_load(open(P(c['lang_config']))); t0 = time.time()
    out = os.path.join(ROOT, 'results', 'runs', time.strftime('%Y-%m-%d') + '_fv_s_text'); os.makedirs(out, exist_ok=True)
    pr = prompts(c); model, tok = load_text_encoder(lc, torch, open_clip); E = encode(model, tok, [p[2] for p in pr], torch)
    np.save(os.path.join(out, 'prompt_text.npy'), E)
    info = dict(prompts=pr, sha256=fsha(os.path.join(out, 'prompt_text.npy')), seconds=round(time.time() - t0, 1), torch=torch.__version__, open_clip=open_clip.__version__)
    json.dump(info, open(os.path.join(out, 'prompts.json'), 'w'), indent=1); print(json.dumps(info))


if __name__ == '__main__':
    main()
