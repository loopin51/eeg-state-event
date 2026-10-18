"""F-VIII (D-117 amendment 5): OpenCLIP ViT-H-14 text embeddings of the sentence pool (results/fv_lang/pool/pool.tsv), sentences
as written (no template), L2-normalised, float32 [n_sentences, 1024] in pool.tsv row order. Also the abstention sentence.
k3s RTX 5080 (data group fv_langemb). Output: results/runs/<date>_fv_lang_embed/{pool_text.npy, abstain_text.npy, info.json}.

usage: python scripts/fv_lang_embed.py configs/fv_lang.yaml
"""
import os, sys, csv, json, time, hashlib, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def load_text_encoder(c, torch, open_clip):
    w = P(c['clip']['weights']); assert fsha(w) == c['clip']['weights_sha256'], 'CLIP weights sha256'
    model, _, _ = open_clip.create_model_and_transforms(c['clip']['arch'], pretrained=w); model.visual = None
    return model.eval().half().to('cuda'), open_clip.get_tokenizer(c['clip']['arch'])


def encode(model, tok, texts, torch, batch=256):
    out = []
    with torch.no_grad():
        for b in range(0, len(texts), batch):
            out.append(torch.nn.functional.normalize(model.encode_text(tok(texts[b:b + batch]).to('cuda')).float(), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 1024), np.float32)


def main():
    import torch, open_clip
    from ese.things.lang import ABSTAIN
    c = yaml.safe_load(open(P(sys.argv[1]))); t0 = time.time()
    pf = os.path.join(P(c['pool']['out_dir']), 'pool.tsv'); pool = list(csv.DictReader(open(pf, encoding='utf-8'), delimiter='\t'))
    out = os.path.join(ROOT, 'results', 'runs', time.strftime('%Y-%m-%d') + '_fv_lang_embed'); os.makedirs(out, exist_ok=True)
    model, tok = load_text_encoder(c, torch, open_clip)
    E = encode(model, tok, [r['sentence'] for r in pool], torch, c['clip']['batch']); np.save(os.path.join(out, 'pool_text.npy'), E)
    np.save(os.path.join(out, 'abstain_text.npy'), encode(model, tok, [ABSTAIN], torch))
    info = dict(n=len(pool), pool_sha256=fsha(pf), pool_text_sha256=fsha(os.path.join(out, 'pool_text.npy')), seconds=round(time.time() - t0, 1),
                device=torch.cuda.get_device_name(0), torch=torch.__version__, open_clip=open_clip.__version__, dtype='float16 encoder, float32 output')
    json.dump(info, open(os.path.join(out, 'info.json'), 'w'), indent=1); print(json.dumps(info))


if __name__ == '__main__':
    main()
