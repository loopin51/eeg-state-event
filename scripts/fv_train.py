"""F-V (D-111, D-112 sections 3-4): one LOSO fold (target subject) x candidate x seed of the THINGS-EEG2 concept decoder.

1. Shared encoder on the nine other subjects: per image, the mean over every repetition of sessions 1-4 (scripts/fv_prepare.py
   image means), concepts outside the development set (mode dev) or all training concepts (mode final), minus the early-
   stopping concepts (valid.salt). InfoNCE from EEG to the batch's unique image embeddings (ViT-H-14), learned logit scale,
   early stopping on the validation concepts' loss. The target's data never enter this stage.
2. D-090 calibration of the target's subject layer only (shared network frozen, eval mode), from the mean of the trainers'
   layers, on the target's own image means over sessions {1}, {1,2}, {1,2,3} (mode dev: non-development concepts only),
   early stopping on the target's validation concepts in the same sessions; the initial layer is kept when fitting never
   improves the validation loss. k0 = the mean layer.
3. Mode dev: development evaluation on the target's session-4 training runs (set A, D-112: gate D and candidate choice). Per
   development concept present in session 4, the mean over all its images x repetitions there (concepts without an image are
   left out and counted; a concept's 10 images share one session of each pair {1,2}, {3,4}); set B (report only) takes every
   development concept from its own session 3 or 4 (session 3 is a calibration session). 200 development
   concept names as candidates (name text embeddings), cosine scores. Stored per condition: score matrix, MRP, top-1/top-5,
   and the same for Gaussian input with the evaluation inputs' channel variances (gate D iii, condition ses123).
   Mode final: no evaluation here (the sealed test runs are scored only in the verdict job); checkpoints are stored.
--shuffle: label-shuffle model (concept permutation shuffle_seed; image j of concept A -> embedding of image j of pi(A)).
--smoke  : 2 trainers, 2 epochs of at most 20 updates, 1 calibration epoch; still development data only (never test data).
--fake-targets: random unit vectors instead of the ViT-H-14 files (code smoke before the embeddings exist; never reported).

Output: results/runs/<date>_fv_<mode>_<candidate>_s<seed>_t<target>[_shuffle][_smoke]/ with config.yaml, history.json,
metrics.json, shared.pt, layers.pt, dev_scores.npz (mode dev).

usage: python scripts/fv_train.py configs/fv_train.yaml --target S --seed N --candidate {bm,nice} --mode {dev,final}
                                  [--shuffle] [--smoke] [--fake-targets]
"""
import os, sys, csv, json, time, copy, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.things.data import Meta, subject_mean, concept_mean, GPUSet
from ese.things.models import build, LogitScale, info_nce, scores

P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def name_rows(c, meta, folders):
    """rows of the name table / names_text for training folders ('00001_aardvark'); asserts every folder has exactly one row"""
    rows = list(csv.DictReader(open(P(c['targets']['names_table'])), delimiter='\t'))
    idx = {}
    for i, r in enumerate(rows):
        if r['split'].startswith('train'): assert r['folder'] not in idx, f'duplicate training folder {r["folder"]}'; idx[r['folder']] = i
    miss = [f for f in folders if f not in idx]; assert not miss, f'name table rows missing for {miss[:5]}'
    return np.array([idx[f] for f in folders])


def shuffled_rows(meta, seed):
    """target row of every training code under the concept permutation (image j of concept A -> image j of pi(A))"""
    imgs = [np.where(meta.img_concept == a)[0] for a in range(len(meta.train_concepts))]
    assert all(len(x) == len(imgs[0]) for x in imgs), 'concepts with different image counts'
    pi = np.random.default_rng(seed).permutation(len(imgs)); row = np.arange(len(meta.img_concept))
    for a, b in enumerate(pi): row[imgs[a]] = imgs[b]
    return row, pi


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--target', type=int, required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--candidate', choices=['bm', 'nice'], required=True); ap.add_argument('--mode', choices=['dev', 'final'], required=True)
    ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--fake-targets', action='store_true')
    a = ap.parse_args(); assert not a.fake_targets or a.smoke, '--fake-targets is for code smoke tests only'
    c = yaml.safe_load(open(P(a.config))); pc = yaml.safe_load(open(P(c['prep_config']))); cache = P(pc['cache_dir']); t0 = time.time(); dev = 'cuda'
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed)
    meta = Meta(P(c['meta'])); dev_c = meta.dev_concepts(c['dev']['salt'], c['dev']['n']); valid_c = meta.valid_concepts(dev_c, c['valid']['salt'], c['valid']['fraction'])
    excl = set(dev_c) if a.mode == 'dev' else set(); train_c = [x for x in meta.train_concepts if x not in excl and x not in set(valid_c)]
    train_codes, valid_codes, dev_codes = meta.codes_of(train_c), meta.codes_of(valid_c), meta.codes_of(dev_c)
    trainers = [s for s in c['subjects'] if s != a.target]; ti = len(trainers)
    o = dict(c['optim'][a.candidate], max_epochs=c['optim']['max_epochs'], patience=c['optim']['patience']); cal = dict(c['calibration'])
    max_updates = None
    if a.smoke:
        have = [s for s in trainers if all(os.path.exists(os.path.join(cache, f'sub-{s:02d}', f'ses-{e:02d}_training_imgavg.npy')) for e in c['sessions'])]
        trainers = have[:c['smoke']['trainers']]; ti = len(trainers); o['max_epochs'] = c['smoke']['max_epochs']; max_updates = c['smoke']['max_updates']; cal['max_epochs'] = c['smoke']['cal_epochs']
    name = time.strftime('%Y-%m-%d') + f"_fv_{a.mode}_{a.candidate}_s{a.seed}_t{a.target:02d}{'_shuffle' if a.shuffle else ''}{'_smoke' if a.smoke else ''}"
    out = os.path.join(P(c['out_root']), name); os.makedirs(out, exist_ok=True)
    # targets
    if a.fake_targets: Timg = np.random.default_rng(0).standard_normal((len(meta.img_concept), c['model']['out_dim'])).astype(np.float32); Tname = np.random.default_rng(1).standard_normal((len(meta.train_concepts) + 200, c['model']['out_dim'])).astype(np.float32); drow = np.arange(len(dev_c))
    else:
        td = P(c['targets']['dir']); Timg = np.load(os.path.join(td, c['targets']['train_image'])); Tname = np.load(os.path.join(td, c['targets']['names_text'])); drow = name_rows(c, meta, dev_c)
    assert Timg.shape == (len(meta.img_concept), c['model']['out_dim']), Timg.shape
    trow = np.arange(len(meta.img_concept)); pi = None
    if a.shuffle: trow, pi = shuffled_rows(meta, c['shuffle_seed'])
    Tg = torch.from_numpy(Timg[trow]).to(dev)                                   # row k-1 = training target of code k
    yaml.safe_dump(dict(c, run=dict(target=a.target, seed=a.seed, candidate=a.candidate, mode=a.mode, shuffle=a.shuffle, smoke=a.smoke, fake_targets=a.fake_targets,
                                    trainers=trainers, n_train_concepts=len(train_c), n_valid_concepts=len(valid_c), n_dev_concepts=len(dev_c),
                                    device=torch.cuda.get_device_name(0))), open(os.path.join(out, 'config.yaml'), 'w'))
    # 1. shared encoder
    pos = np.load(P(c['positions'])); tr_blocks, va_blocks = [], []
    for i, s in enumerate(trainers):
        codes, X, _ = subject_mean(cache, s, c['sessions'], codes=np.r_[train_codes, valid_codes]); m = np.isin(codes, valid_codes)
        tr_blocks.append((i, codes[~m], X[~m])); va_blocks.append((i, codes[m], X[m])); del X
    TR, VA = GPUSet(tr_blocks, pos, dev), GPUSet(va_blocks, pos, dev); del tr_blocks, va_blocks
    print(f'target {a.target} trainers {trainers}: train items {len(TR)} valid items {len(VA)} ({time.time() - t0:.0f}s)', flush=True)
    net = build(a.candidate, ti + 1, c['model']).to(dev); scale = LogitScale(c['optim']['logit_scale_init']).to(dev)
    opt = torch.optim.Adam(list(net.parameters()) + list(scale.parameters()), lr=o['lr'], betas=tuple(o['betas']))
    nparam = sum(p.numel() for p in net.parameters()); hist = []

    def loss_of(S, idx):
        x, p, subj, _, _, code, _ = S.batch(idx); u, inv = torch.unique(code, return_inverse=True)
        return info_nce(net(x, p, subj), Tg[u - 1], inv, scale())

    vb = [torch.arange(b, min(b + o['batch'], len(VA)), device=dev) for b in range(0, len(VA), o['batch'])]

    def vloss():
        net.eval()
        with torch.no_grad(): return float(np.mean([loss_of(VA, b).item() for b in vb]))

    best, bad, state = float('inf'), 0, None; t_shared = time.time()
    for ep in range(o['max_epochs']):
        net.train(); perm = torch.from_numpy(rng.permutation(len(TR))).to(dev); tl = []
        for k, b in enumerate(range(0, len(TR) - o['batch'] + 1, o['batch'])):
            if max_updates and k >= max_updates: break
            loss = loss_of(TR, perm[b:b + o['batch']]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl.append(loss.item())
        vl = vloss(); hist.append(dict(stage='shared', epoch=ep, train=float(np.mean(tl)), valid=vl, scale=float(scale().item()), seconds=time.time() - t_shared))
        print(f'shared epoch {ep}: train {np.mean(tl):.4f} valid {vl:.4f} scale {scale().item():.1f} ({time.time() - t0:.0f}s)', flush=True)
        if vl < best: best, bad, state = vl, 0, (copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()}), float(scale.v.item()))
        else:
            bad += 1
            if bad >= o['patience']: break
    net.load_state_dict(state[0])
    with torch.no_grad(): scale.v.fill_(state[1])
    torch.save(dict(net=state[0], logit_v=state[1]), os.path.join(out, 'shared.pt')); t_shared = time.time() - t_shared
    del TR, VA; torch.cuda.empty_cache()
    # 2. calibration of the target's layer
    for p in net.parameters(): p.requires_grad_(False)
    net.eval(); W = net.subject.weights; mean_layer = W[:ti].mean(0).clone(); layers = {'k0': mean_layer.cpu()}; cal_meta = {}; W0 = W[:ti].detach().clone()
    cal_codes = np.r_[train_codes, valid_codes]                                 # mode dev: development concepts excluded
    for kname, sess in cal['sessions'].items():
        t1 = time.time(); codes, X, _ = subject_mean(cache, a.target, sess, codes=cal_codes); m = np.isin(codes, valid_codes)
        CT, CV = GPUSet([(ti, codes[~m], X[~m])], pos, dev), GPUSet([(ti, codes[m], X[m])], pos, dev); del X
        cvb = [torch.arange(b, min(b + cal['batch'], len(CV)), device=dev) for b in range(0, len(CV), cal['batch'])]

        def cv():
            with torch.no_grad(): return float(np.mean([loss_of(CV, b).item() for b in cvb]))
        with torch.no_grad(): W[ti] = mean_layer
        W.requires_grad_(True); copt = torch.optim.Adam([W], lr=cal['lr']); init = cv(); b_v, row, bad = init, mean_layer.clone(), 0
        hist.append(dict(stage=f'cal_{kname}', epoch=-1, train=None, valid=init))
        for ep in range(cal['max_epochs']):
            perm = torch.from_numpy(rng.permutation(len(CT))).to(dev); tl = []
            for k, b in enumerate(range(0, max(len(CT) - cal['batch'] + 1, 1), cal['batch'])):
                if max_updates and k >= max_updates: break
                loss = loss_of(CT, perm[b:b + cal['batch']]); copt.zero_grad(set_to_none=True); loss.backward()
                W.grad[:ti] = 0; copt.step(); tl.append(loss.item())                # only the target's row moves (zero gradient -> zero Adam step)
            vl = cv(); hist.append(dict(stage=f'cal_{kname}', epoch=ep, train=float(np.mean(tl)), valid=vl))
            if vl < b_v: b_v, row, bad = vl, W[ti].detach().clone(), 0
            else:
                bad += 1
                if bad >= cal['patience']: break
        W.requires_grad_(False)
        with torch.no_grad(): W[ti] = row
        layers[kname] = row.cpu(); cal_meta[kname] = dict(sessions=sess, n_train=len(CT), n_valid=len(CV), init_valid=init, best_valid=b_v,
                                                         kept_initial=bool(b_v >= init), epochs=sum(h['stage'] == f'cal_{kname}' and h['epoch'] >= 0 for h in hist), seconds=time.time() - t1)
        print(f'calibration {kname}: valid {init:.4f} -> {b_v:.4f} ({time.time() - t1:.0f}s)', flush=True); del CT, CV
    assert torch.equal(W[:ti], W0), 'a trainer layer moved during calibration'
    torch.save(layers, os.path.join(out, 'layers.pt')); json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
    res = dict(summary=dict(target=a.target, trainers=trainers, candidate=a.candidate, seed=a.seed, mode=a.mode, shuffle=a.shuffle, smoke=a.smoke,
                            n_params=nparam, shared_best_valid=best, shared_epochs=sum(h['stage'] == 'shared' for h in hist), shared_seconds=t_shared,
                            calibration=cal_meta, device=torch.cuda.get_device_name(0), max_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
                            shuffle_perm_sha=None if pi is None else __import__('hashlib').sha256(pi.tobytes()).hexdigest()))
    # 3. development evaluation (mode dev only; development concepts of the target's session 4 training runs)
    if a.mode == 'dev':
        dci = np.array([meta.train_concepts.index(x) for x in dev_c]); cand = torch.from_numpy(Tname[drow]).to(dev); arrays, res['dev'] = {}, {}
        for setname, sess in (('A_ses4', [c['dev']['eval_session']]), ('B_ses34', c['dev']['report_sessions'])):   # A: D-112 development set; B: report only
            codes, X, nrep = subject_mean(cache, a.target, sess, codes=dev_codes); keep, Xc = concept_mean(codes, X, nrep, meta.img_concept, dci)
            true = np.array([list(dci).index(j) for j in keep]); xg = torch.from_numpy(Xc.astype(np.float32)).to(dev)
            pg = torch.as_tensor(pos, dtype=torch.float32, device=dev)[None].expand(len(xg), -1, -1); sd = Xc.std(axis=(0, 2))
            xn = torch.from_numpy((np.random.default_rng(c['eval']['noise_seed']).standard_normal(Xc.shape) * sd[None, :, None]).astype(np.float32)).to(dev)
            arrays[f'{setname}_concepts'] = keep; arrays[f'{setname}_true'] = true; ev = {}
            for kname, layer in layers.items():
                with torch.no_grad():
                    W[ti] = layer.to(dev); sj = torch.full((len(xg),), ti, device=dev)
                    for tag, xx in (('', xg),) + ((('_noise', xn),) if kname == 'ses123' else ()):
                        Ev = net(xx, pg, sj); S = scores(Ev, cand).cpu().numpy(); rk = (S > S[np.arange(len(S)), true][:, None]).sum(1)
                        arrays[f'{setname}_S_{kname}{tag}'] = S; arrays[f'{setname}_E_{kname}{tag}'] = Ev.float().cpu().numpy()   # predicted embeddings (seed means)
                        ev[f'{kname}{tag}'] = dict(mrp=float(np.mean(1 - rk / (S.shape[1] - 1))), **{f'top{k}': float(np.mean(rk < k)) for k in c['eval']['topk']})
            res['dev'][setname] = dict(sessions=sess, n_concepts_present=int(len(keep)), n_concepts_missing=int(len(dci) - len(keep)), n_candidates=int(len(dci)), conditions=ev)
            print('DEV', setname, json.dumps({k: round(v['mrp'], 4) for k, v in ev.items()}), flush=True)
        np.savez_compressed(os.path.join(out, 'dev_scores.npz'), **arrays)
    res['summary']['seconds'] = time.time() - t0
    json.dump(res, open(os.path.join(out, 'metrics.json'), 'w'), indent=1); print('SUMMARY', json.dumps({k: v for k, v in res['summary'].items() if k != 'calibration'}), flush=True)


if __name__ == '__main__':
    main()
