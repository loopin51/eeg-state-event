"""D-092 collapse check on the first finished job of each batch, validation data only (never the test segments / blocks).

Main runs: rebuild the readout from shared.pt and target_layers.pt, score 512 windows of the validation segment for every
target and record the std across windows of the readout output (mean over output units). A′: best.pt on the validation
blocks for every subject. Pass if every value exceeds check.min_output_std (B1 rule). Reads no metrics.json or scores.npz.

usage: python scripts/alice_check.py configs/alice_train.yaml <run folder name>
"""
import os, sys, json, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from alice_train import Alice, build_net


def out_std(net, D, items, si, amp, chunk=128):
    Z = []
    with torch.no_grad():
        for b in range(0, len(items), chunk):
            X, P, _ = D.batch(items[b:b + chunk], np.full(len(items[b:b + chunk]), si))
            with torch.autocast(**amp): Z.append(net(X, P, torch.full((len(X),), si, device=D.dev)).float().flatten(1))
    return float(torch.cat(Z).std(0).mean())


def main():
    c0 = yaml.safe_load(open(sys.argv[1])); d = os.path.join(ROOT, c0['out_root'], sys.argv[2]); c = yaml.safe_load(open(os.path.join(d, 'config.yaml'))); r = c['run']
    dev = 'cuda'; thr = c0['check_min_output_std']; rng = np.random.default_rng(0); res = {}
    if r['mode'] == 'aprime':
        import pandas as pd
        o = c['aprime']; pre = dict(reference=o['reference'], invalid=o['invalid'], clamp=o['clamp'], baseline_s=o['baseline_s'])
        subjects = sorted(int(s) for s in json.load(open(os.path.join(ROOT, c['cache_dir'], 'eeg_meta.json')))['subjects'])
        D = Alice(c, o['source'], c['features']['aprime'], subjects, 'defossez', pre); B = pd.read_csv(os.path.join(ROOT, c['cache_dir'], 'blocks.tsv'), sep='\t')
        items = [it for _, x in B[B.split == 'valid'].iterrows() for it in D.grid(int(x.run), x.start, x.stop)]
        B_tr = B[B.split == 'train']; D.feat_norm([(int(x.run), x.start, x.stop) for _, x in B_tr.iterrows()]); D.to_device(dev)
        net = build_net(c, 'defossez', len(subjects), D, dev); net.load_state_dict(torch.load(os.path.join(d, 'best.pt'))); net.eval()
        amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=o['precision'] == 'bf16')
        for si, s in enumerate(subjects):
            its = [x for x in items if x[0] not in D.missing[s]]; res[str(s)] = out_std(net, D, its, si, amp)
    else:
        arm = r['arm']; ac = c['arms'][arm]; subjects = r['trainers'] + r['targets']; nT = len(r['trainers'])
        D = Alice(c, r['source'], c['features']['main'], subjects, arm, ac); train = [x - 1 for x in r['train_segments']]
        D.feat_norm([(x, 0.0, D.durs[x]) for x in train]); D.to_device(dev)
        net = build_net(c, arm, len(subjects), D, dev); sd = torch.load(os.path.join(d, 'shared.pt'))
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}); net.eval()
        with torch.no_grad(): net.subject.weights[nT:] = torch.load(os.path.join(d, 'target_layers.pt')).to(dev)
        amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=c['main']['precision'] == 'bf16'); v = r['valid_segment'] - 1
        for j, s in enumerate(r['targets']):
            if v in D.missing[s]: res[str(s)] = None; continue
            its = D.grid(v); its = [its[i] for i in rng.choice(len(its), min(512, len(its)), replace=False)]; res[str(s)] = out_std(net, D, its, nT + j, amp)
    vals = [x for x in res.values() if x is not None]; ok = min(vals) > thr
    out = dict(run=os.path.basename(d), split='valid', min_output_std=thr, output_std=res, passed=bool(ok))
    p = os.path.join(d, 'check.json'); json.dump(out, open(p, 'w'), indent=1); print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
