"""Sentence embeddings for the ds007808 listening segments (D-057 target TXT).

Model intfloat/multilingual-e5-base (public weights), input "query: <text>", mean pooling over tokens, L2 normalised.
Writes data/cache/ds007808_listening/sub-XX_txt.npy aligned with sub-XX.npz.

usage: python scripts/embed_ds007808_text.py [--subjects 1 2]
"""
import os, argparse, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CACHE = os.path.join(ROOT, 'data/cache/ds007808_listening')
MODEL = 'intfloat/multilingual-e5-base'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--subjects', type=int, nargs='+', default=[1, 2]); a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL); net = AutoModel.from_pretrained(MODEL).cuda().eval()
    for sub in a.subjects:
        texts = np.load(os.path.join(CACHE, f'sub-{sub:02d}.npz'), allow_pickle=True)['text']; out = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                b = tok([f'query: {t}' for t in texts[i:i + 256]], padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
                h = net(**b).last_hidden_state; m = b['attention_mask'].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1); out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        E = np.concatenate(out).astype(np.float32); np.save(os.path.join(CACHE, f'sub-{sub:02d}_txt.npy'), E)
        print(f'sub-{sub:02d}: {E.shape}, unique texts {len(set(texts.tolist()))}', flush=True)


if __name__ == '__main__':
    main()
