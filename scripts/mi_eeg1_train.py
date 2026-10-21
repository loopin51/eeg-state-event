"""D-120 section 2: THINGS-EEG1 group-average decoder (k3s admin, rtx5080-16g).

The fv_train bm encoder and optimisation (D-112 frozen values: model.bm, optim.bm, max_epochs, patience, logit scale init)
with ONE subject layer (the group) and 64 input channels. Inputs: whitened group means of the training and validation
concepts (scripts/mi_eeg1_prep.py; development concepts are not in the file). Target: ViT-H-14 image embedding of the photo
(row of the EEG2 training photos). InfoNCE with a learned logit scale, early stopping on the validation-concept loss.
--shuffle: label-shuffle model, concept permutation configs/mi_eeg1.yaml seeds.shuffle_perm (fv_train shuffled_rows rule).
After training: outputs for the validation photos (valid_out.npz) and the 146-way validation concept-row name MRP (the
signal-to-noise reference of the gate). Output: results/runs/<date>_mi_eeg1_bm_s<seed>[_shuffle][_smoke]/ with config.yaml,
history.json, metrics.json, shared.pt, valid_out.npz.
usage: python scripts/mi_eeg1_train.py configs/mi_eeg1.yaml --seed N [--shuffle] [--smoke]
"""
import os, sys, copy, json, time, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.things.data import Meta, GPUSet
from ese.things.models import build, LogitScale, info_nce
from ese.things import verdict as V
from fv_train import name_rows, shuffled_rows
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def model_cfg(c, tc):
    m = copy.deepcopy(tc['model']); m['bm'] = dict(m['bm'], n_in=c['model']['n_in']); return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--shuffle', action='store_true')
    ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    c = yaml.safe_load(open(P(a.config))); tc = yaml.safe_load(open(P(c['train_config']))); t0 = time.time(); dev = 'cuda'
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed)
    cache = P(c['cache_dir']); meta = Meta(P(tc['meta']))                     # the smoke run reads the same training / validation file (no development photos in it)
    dev_c = meta.dev_concepts(tc['dev']['salt'], tc['dev']['n']); valid_c = meta.valid_concepts(dev_c, tc['valid']['salt'], tc['valid']['fraction'])
    ic = np.array(meta.train_img_concept); is_va = np.isin(ic, valid_c); is_dev = np.isin(ic, dev_c)
    rows = np.load(os.path.join(cache, 'group_trainvalid_rows.npy')); X = np.load(os.path.join(cache, 'group_trainvalid.npy'), mmap_mode='r')
    assert not is_dev[rows].any(), 'development photos in the training file'; mv = is_va[rows]
    o = dict(dict(max_epochs=tc['optim']['max_epochs'], patience=tc['optim']['patience']), **tc['optim']['bm']); max_updates = None
    if a.smoke: o['max_epochs'] = 2; max_updates = 20
    name = time.strftime('%Y-%m-%d') + f"_mi_eeg1_bm_s{a.seed}{'_shuffle' if a.shuffle else ''}{'_smoke' if a.smoke else ''}"
    out = os.path.join(P(tc['out_root']), name); os.makedirs(out, exist_ok=True)
    Timg = np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['train_image'])); assert Timg.shape == (16540, 1024)
    trow = np.arange(16540)
    if a.shuffle: trow, _ = shuffled_rows(meta, c['seeds']['shuffle_perm'])
    Tg = torch.from_numpy(Timg[trow]).to(dev)                                   # code k (= row + 1) -> target
    mcfg = model_cfg(c, tc)
    yaml.safe_dump(dict(c, run=dict(seed=a.seed, shuffle=a.shuffle, smoke=a.smoke, n_train=int((~mv).sum()), n_valid=int(mv.sum()), model=mcfg['bm'], optim=o,
                                    device=torch.cuda.get_device_name(0))), open(os.path.join(out, 'config.yaml'), 'w'))
    pos = np.load(os.path.join(cache, 'positions.npy')); Xa = np.asarray(X, np.float16)
    TR = GPUSet([(0, rows[~mv] + 1, Xa[~mv])], pos, dev); VA = GPUSet([(0, rows[mv] + 1, Xa[mv])], pos, dev); del Xa
    print(f'train {len(TR)} valid {len(VA)} ({time.time() - t0:.0f}s)', flush=True)
    net = build('bm', 1, mcfg).to(dev); scale = LogitScale(tc['optim']['logit_scale_init']).to(dev); params = list(net.parameters()) + list(scale.parameters())
    opt = torch.optim.Adam(params, lr=o['lr'], betas=tuple(o['betas'])); hist = []

    def loss_of(S, idx):
        x, p, subj, _, _, code, _ = S.batch(idx); u, inv = torch.unique(code, return_inverse=True); return info_nce(net(x, p, subj), Tg[u - 1], inv, scale())
    vb = [torch.arange(b, min(b + o['batch'], len(VA)), device=dev) for b in range(0, len(VA), o['batch'])]

    def vloss():
        net.eval()
        with torch.no_grad(): return float(np.mean([loss_of(VA, b).item() for b in vb]))
    best, bad, state = float('inf'), 0, None
    for ep in range(o['max_epochs']):
        net.train(); perm = torch.from_numpy(rng.permutation(len(TR))).to(dev); tl = []
        for k, b in enumerate(range(0, len(TR) - o['batch'] + 1, o['batch'])):
            if max_updates and k >= max_updates: break
            loss = loss_of(TR, perm[b:b + o['batch']]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl.append(loss.item())
        vl = vloss(); hist.append(dict(epoch=ep, train=float(np.mean(tl)), valid=vl, scale=float(scale().item()), seconds=time.time() - t0))
        print(f'epoch {ep}: train {np.mean(tl):.4f} valid {vl:.4f} scale {scale().item():.1f} ({time.time() - t0:.0f}s)', flush=True)
        if vl < best: best, bad, state = vl, 0, (copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()}), float(scale.v.item()), ep)
        else:
            bad += 1
            if bad >= o['patience']: break
    net.load_state_dict(state[0]); torch.save(dict(net=state[0], logit_v=state[1]), os.path.join(out, 'shared.pt')); net.eval()
    with torch.no_grad():
        Z = np.concatenate([net(*[t for t in VA.batch(b)[:3]]).float().cpu().numpy() for b in vb])
    vr = rows[mv]; np.savez_compressed(os.path.join(out, 'valid_out.npz'), rows=vr, z=Z.astype(np.float32))
    vci = ic[vr]; T = V.unit(np.load(os.path.join(P(tc['targets']['dir']), tc['targets']['names_text']))[name_rows(tc, meta, valid_c)])
    R = V.unit(np.stack([V.unit(Z[vci == k]).mean(0) for k in valid_c])); mrp = float(V.cells(R @ T.T, np.arange(len(valid_c))).mean())
    json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
    res = dict(seed=a.seed, shuffle=a.shuffle, smoke=a.smoke, best_epoch=state[2], best_valid_loss=best, epochs_run=len(hist), valid_name_mrp=mrp, n_valid_concepts=len(valid_c),
               seconds=time.time() - t0, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
    json.dump(res, open(os.path.join(out, 'metrics.json'), 'w'), indent=1); print('SUMMARY', json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
