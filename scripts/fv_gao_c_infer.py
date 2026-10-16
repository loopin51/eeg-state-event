"""D-113 arm c inference (k3s RTX 5080, D-095): frozen F-V models -> predicted embeddings only (no score, no cosine, no MRP).

For every model folder under <stage_dir>/models/ (copies made by fv_gao_c_prep.py --part models): the network of shared.pt,
the population layer `layer` (layers.pt, k0 = mean of the trainers' layers) written into the target row (index = number of
trainers, as fv_train.py), eval mode, no Gao calibration. Inputs: every <stage_dir>/things_tXX.npz of the model's own target
(pipeline control, D-113 6.4) and every <stage_dir>/gao_*.npz (all models). Every array key starting with X is an input:
X63* with pos63, any other with pos30 (INVALID outside the 30 channels); a [n, 30, T] array is put into its 63-channel slots
(ch30_index), the other slots 0.
A 30-channel slot input (INVALID positions) is refused for a model whose spatial layer is not position-masked (nice).
Before any forward pass, the sha256 of every staged file is checked against <stage_dir>/manifest.json.
Output: results/runs/<date>_fv_gao_c_infer[_<tag>]/ (RUN_DIR on k3s) with emb_<model>.npz (float32 embeddings per input key,
the input file's sha256), config.yaml and metrics.json (device, counts, seconds, torch peak memory).
--inputs things: the pipeline control only (no Gao input is read); --inputs gao: the Gao inputs only, run after the control passed
(D-113 6.4: a model that fails the control is not applied to Gao).
usage (k3s): python scripts/fv_gao_c_infer.py configs/fv_gao_c.yaml --inputs {things,gao} [--tag NAME] [--models RUN ...]
"""
import os, sys, glob, json, time, hashlib, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
from ese.things.models import build


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--tag'); ap.add_argument('--models', nargs='+'); ap.add_argument('--inputs', choices=['things', 'gao'], required=True); a = ap.parse_args()
    c = yaml.safe_load(open(P(a.config))); sd = P(c['stage_dir']); t0 = time.time(); dev = 'cuda'
    man = json.load(open(os.path.join(sd, 'manifest.json')))['files']
    for f, v in man.items(): assert sha256(os.path.join(sd, f)) == v['sha256'], f'staged file differs from the manifest: {f}'
    name = time.strftime('%Y-%m-%d') + '_fv_gao_c_infer' + (f'_{a.tag}' if a.tag else ''); out = os.path.join(P('results/runs'), name); os.makedirs(out, exist_ok=True)
    mdirs = sorted(glob.glob(os.path.join(sd, 'models', '*'))); mdirs = [d for d in mdirs if not a.models or os.path.basename(d) in a.models]
    assert mdirs, 'no model folder'
    gao = sorted(glob.glob(os.path.join(sd, 'gao_*.npz'))) if a.inputs == 'gao' else []; summ = dict(models={}, device=torch.cuda.get_device_name(0), layer=c['layer'], manifest_files=len(man))
    yaml.safe_dump(dict(c, run=dict(inputs=a.inputs, models=[os.path.basename(d) for d in mdirs], gao_inputs=[os.path.basename(g) for g in gao], device=summ['device'])), open(os.path.join(out, 'config.yaml'), 'w'))
    for d in mdirs:
        mc = yaml.safe_load(open(os.path.join(d, 'config.yaml'))); run = mc['run']; ti = len(run['trainers'])
        net = build(run['candidate'], ti + 1, mc['model']).to(dev); net.load_state_dict(torch.load(os.path.join(d, 'shared.pt'), map_location=dev)['net']); net.eval()
        layers = torch.load(os.path.join(d, 'layers.pt'), map_location=dev)
        with torch.no_grad(): net.subject.weights[ti] = layers[c['layer']].to(dev)
        inputs = (sorted(glob.glob(os.path.join(sd, f"things_t{run['target']:02d}.npz"))) if a.inputs == 'things' else gao); arrays, info = {}, {}
        for f in inputs:
            z = np.load(f, allow_pickle=False); tag = os.path.basename(f)[:-4]; info[tag] = dict(sha256=man[os.path.basename(f)]['sha256'])
            for key in sorted(k for k in z.files if k.startswith('X')):
                pkey = 'pos63' if key.startswith('X63') else 'pos30'; X = z[key]
                if pkey == 'pos30' and run['candidate'] != 'bm': info[tag][key] = 'refused (spatial layer not position-masked)'; continue
                ci = torch.as_tensor(z['ch30_index'], device=dev) if X.shape[1] == 30 else None   # compact 30-channel array -> its 63-channel slots
                pos = torch.as_tensor(z[pkey], dtype=torch.float32, device=dev); E = np.empty((len(X), mc['model']['out_dim']), np.float32)
                with torch.no_grad():
                    for b in range(0, len(X), c['infer']['batch']):
                        xb = torch.from_numpy(np.asarray(X[b:b + c['infer']['batch']], np.float32)).to(dev); B = len(xb)
                        if ci is not None: xs = torch.zeros((B, 63, xb.shape[2]), device=dev); xs[:, ci] = xb; xb = xs
                        E[b:b + B] = net(xb, pos[None].expand(B, -1, -1), torch.full((B,), ti, dtype=torch.long, device=dev)).float().cpu().numpy()
                arrays[f'{tag}__{key}'] = E; info[tag][key] = int(len(X))
        np.savez(os.path.join(out, f'emb_{os.path.basename(d)}.npz'), **arrays)
        summ['models'][os.path.basename(d)] = dict(target=run['target'], seed=run['seed'], candidate=run['candidate'], mode=run['mode'], shuffle=run['shuffle'],
                                                  smoke=run.get('smoke', False), shared_sha256=man[f'models/{os.path.basename(d)}/shared.pt']['sha256'], inputs=info)
        print(os.path.basename(d), json.dumps(info), f'{time.time() - t0:.0f}s', flush=True); del net
    summ.update(seconds=time.time() - t0, max_memory_gb=torch.cuda.max_memory_allocated() / 1e9)
    json.dump(summ, open(os.path.join(out, 'metrics.json'), 'w'), indent=1); print('SUMMARY', json.dumps({k: v for k, v in summ.items() if k != 'models'}), flush=True)


if __name__ == '__main__':
    main()
