"""D-128 post-hoc diagnosis (outside the verdict family): linear probes of VERA features on the SB clean conditions.

usage: env/bin/python scripts/nc_diag_sb.py configs/nc_diag_sb.yaml --mode init|trained [--cond ind_clean] [--arm reve_nc]
                                            [--folds 0] [--smoke]

Data, labels, folds and the model are built by scripts/nc_train.py code (TaskData, splits, build) from the frozen training
config, so the windows and random SB labels are exactly those of the grid runs (seed `seed`).

--mode init     VERA (reve_nc) with REVE weights, no training, eval mode; one network (torch seed of fold 0). Its encoder
                and query context equal those of -N4 at initialisation (LoRA B = 0, same N1 embedding).
--mode trained  per fold, the scored model saved by `nc_train.py ... --save-ckpt --tag _diag`; features of every window from
                that fold's model, probe fitted on that fold's training subjects and scored on its test subjects.

Features (per window, task window only; t = encoder tokens [C, P, E], C = 64 channels, P = 4 patches, E = 512):
  ctx            query-token context (E)                              - first part of the VERA head input
  ctx_resid      ctx after the N2 residualisation (-N4: the ctx part of its head input)
  tangent        the model's own TangentHead output (136)             - second part of the VERA head input
  h              the model's head input (VERA: [ctx, tangent] after N2, 648)
  pool_proj      per-channel attention-pooled tokens projected by TangentHead.proj, flattened (C x 16) - before covariance
  tangent_pca<d> pooled tokens (VERA: TangentHead attention pooling; -N4: patch mean) projected on the top-d PCs of the fold's
                 training windows (centred), then spd_log_vec (d(d+1)/2) - the N4 covariance step with an informed projection
  tok_chmean     tokens averaged over patches per channel, flattened (C x E = 32,768)
  tok_flat_rp    [ctx, all tokens] flattened (the -N4 head input, 131,584) x fixed Gaussian random projection (rp_dim)
  logits         the model's own output (K or K+1), probed and also argmax-scored (trained mode: must equal metrics.json)
  raw_cov        input reference: logm of the trace-normalised channel covariance in raw_cov_band (2,080)
  raw_erp        input reference: input in raw_erp_band, raw_erp_bin-sample bin means (C x 40)
Probe: standardised features, multinomial L2 logistic regression by full-batch L-BFGS (GPU), C from C_grid by inner-validation
balanced accuracy, refit on train+val, test subjects scored once. Label-shuffle: labels permuted within each train/val subject,
whole pipeline (C selection included) repeated shuffle_seeds times. Writes results/nc/diag_sb/<mode>_<cond>[_<arm>].json.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, glob, time, argparse, hashlib, collections
import numpy as np, yaml, torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.reve_nc import band_split, spd_log_vec
from nc_train import TaskData, splits, scores, build


# ------------------------------------------------------------------ features
def raw_features(d, fc, dev):
    out = {}
    cov, erp = [], []
    for i in range(0, len(d.y), 256):
        x = torch.from_numpy(d.x[i:i + 256]).to(dev).double()
        xb = band_split(x, 200, [tuple(fc['raw_cov_band'])])[:, :, 0]; xb = xb - xb.mean(-1, keepdim=True)
        S = xb @ xb.transpose(1, 2) / xb.shape[-1]; C = S.shape[1]
        S = S / (S.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None] / C)
        S = S + 1e-3 * torch.eye(C, device=dev, dtype=S.dtype)
        w, V = torch.linalg.eigh(S); L = V @ torch.diag_embed(w.clamp_min(1e-10).log()) @ V.transpose(1, 2)
        iu = torch.triu_indices(C, C, device=dev); sc = torch.where(iu[0] == iu[1], 1.0, 2 ** 0.5).to(L.dtype)
        cov.append((L[:, iu[0], iu[1]] * sc).float().cpu())
        xe = band_split(x, 200, [tuple(fc['raw_erp_band'])])[:, :, 0]; k = fc['raw_erp_bin']; T = xe.shape[-1] // k * k
        erp.append(xe[..., :T].reshape(*xe.shape[:2], -1, k).mean(-1).flatten(1).float().cpu())
    out['raw_cov'] = torch.cat(cov).numpy(); out['raw_erp'] = torch.cat(erp).numpy()
    return out


@torch.no_grad()
def extract(net, d, dev, bs, R, idx=None):
    """Features of windows idx (default: all) from one network (eval mode, bf16 autocast as in nc_train.predict)."""
    net.eval(); pos = d.pos.to(dev); E = net.E; n4, n2 = net.cfg['n4'], net.cfg['n2']
    idx = np.arange(len(d.y)) if idx is None else idx; out = collections.defaultdict(list); checked = False
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]; B = len(j)
        x = torch.from_numpy(d.x[j]).to(dev); ref = torch.from_numpy(d.ref[j]).to(dev)
        scale = torch.from_numpy(d.scale[d.rec[j]]).to(dev); P_ = pos[None].expand(B, -1, -1)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            t = net.encode_signal(x, P_, scale); flat = t.reshape(B, -1, E)
            q = net.enc.cls_query_token.expand(B, -1, -1)
            ctx = (torch.softmax(q @ flat.transpose(1, 2) / E ** 0.5, -1) @ flat).squeeze(1)
            if n4:                                   # TangentHead.forward, step by step
                a = torch.softmax((t @ net.tangent.q) / t.shape[-1] ** 0.5, dim=-1)
                pool = (a[..., None] * t).sum(2); z = net.tangent.proj(pool); tan = net.tangent.norm(spd_log_vec(z))
                h = torch.cat([ctx, tan], -1)
            else:
                pool = t.mean(2)
            a_ = h if n4 else ctx
            if n2: a_ = net.resid(a_, net.encode_noise(ref))
            hin = a_ if n4 else torch.cat([a_[:, None], flat], 1).flatten(1)
            logits = net.head(hin)
            if not checked:                          # the step-by-step path must reproduce the model's forward
                ref_lg = net(x, P_, ref, scale)['logits']
                err = float((ref_lg.float() - logits.float()).abs().max()); assert err < 1e-2 * (1 + float(ref_lg.float().abs().max())), err
                out['_check_err'] = [np.array([err])]; checked = True
        out['ctx'].append(ctx.float().cpu().numpy())
        out['ctx_resid'].append((a_[:, :E] if n4 else a_).float().cpu().numpy())
        out['logits'].append(logits.float().cpu().numpy())
        out['tok_chmean'].append(t.float().mean(2).flatten(1).half().cpu().numpy())
        out['_pool'].append(pool.half().cpu().numpy())
        fl = torch.cat([ctx[:, None], flat], 1).flatten(1).half()
        out['tok_flat_rp'].append((fl @ R).float().cpu().numpy())
        if n4:
            out['tangent'].append(tan.float().cpu().numpy()); out['h'].append(a_.float().cpu().numpy())
            out['pool_proj'].append(z.float().flatten(1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def tangent_pca(pool, fit_idx, dims, n_fit, seed, dev):
    """pool [N, C, E] (float16) -> {f'tangent_pca{d}': [N, d(d+1)/2]} with PCs of the channel tokens of windows fit_idx."""
    rng = np.random.default_rng(seed); sub = np.sort(rng.choice(fit_idx, min(n_fit, len(fit_idx)), replace=False))
    Z = torch.from_numpy(pool[sub].astype(np.float32)).to(dev).reshape(-1, pool.shape[-1]); mu = Z.mean(0)
    Cv = (Z - mu).T @ (Z - mu) / len(Z); w, V = torch.linalg.eigh(Cv); V = V.flip(-1); out = {}
    for dd in dims:
        W = V[:, :dd]; parts = []
        for i in range(0, len(pool), 512):
            z = (torch.from_numpy(pool[i:i + 512].astype(np.float32)).to(dev) - mu) @ W
            parts.append(spd_log_vec(z).cpu())
        out[f'tangent_pca{dd}'] = torch.cat(parts).numpy()
    return out


# ------------------------------------------------------------------ probe
def lr_fit(X, y, K, C, max_iter):
    """sklearn-style multinomial L2 logistic regression: min C * sum CE + 0.5 ||W||^2 (bias not penalised)."""
    n, D = X.shape; W = torch.zeros(D, K, device=X.device, requires_grad=True); b = torch.zeros(K, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, history_size=20, tolerance_grad=1e-6, tolerance_change=1e-10,
                            line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(X @ W + b, y) + (W ** 2).sum() / (2 * C * n); loss.backward(); return loss
    opt.step(closure); return W.detach(), b.detach()


def probe_split(Fe, y, tr, va, te, K, pc, dev):
    """C selection on va (balanced accuracy), refit on tr+va, predictions on te."""
    from sklearn.metrics import balanced_accuracy_score
    Xall = torch.from_numpy(np.ascontiguousarray(Fe, dtype=np.float32)).to(dev); yt = torch.from_numpy(y).to(dev)
    def std(fit): mu = Xall[fit].mean(0); sd = Xall[fit].std(0) + 1e-6; return lambda ix: (Xall[ix] - mu) / sd
    s = std(tr); best = (-1, None)
    for C in pc['C_grid']:
        W, b = lr_fit(s(tr), yt[tr], K, C, pc['max_iter'])
        ba = balanced_accuracy_score(y[va], (s(va) @ W + b).argmax(1).cpu().numpy())
        if ba > best[0]: best = (ba, C)
    tv = np.concatenate([tr, va]); s = std(tv); W, b = lr_fit(s(tv), yt[tv], K, best[1], pc['max_iter'])
    p = (s(te) @ W + b).argmax(1).cpu().numpy(); del Xall
    return p, best[1], float(best[0])


def shuffled(y, subj, tv, seed, f, s):
    rng = np.random.default_rng([seed, 77, f, s]); ys = y.copy()
    for g in np.unique(subj[tv]): j = tv[subj[tv] == g]; ys[j] = ys[rng.permutation(j)]
    return ys


def run_probes(feats, d, fold_list, pc, seed, dev, log):
    """feats: {name: [N, D]} or {name: {fold: [N, D]}} -> per-feature fold and pooled scores (real + label-shuffle)."""
    res = {}
    for name in feats:
        t0 = time.time(); P = {'real': {}, **{f'shuf{s}': {} for s in range(pc['shuffle_seeds'])}}; Cs = {}
        for f, tr, va, te in fold_list:
            Fe = feats[name][f] if isinstance(feats[name], dict) else feats[name]
            p, C, vba = probe_split(Fe, d.y, tr, va, te, d.K, pc, dev); P['real'][f] = (te, p); Cs[f] = dict(C=C, val_bal_acc=vba)
            tv = np.concatenate([tr, va])
            for s in range(pc['shuffle_seeds']):
                ys = shuffled(d.y, d.subj, tv, seed, f, s); ps, _, _ = probe_split(Fe, ys, tr, va, te, d.K, pc, dev)
                P[f'shuf{s}'][f] = (te, ps)
        row = dict(dim=int((feats[name][fold_list[0][0]] if isinstance(feats[name], dict) else feats[name]).shape[1]), C=Cs)
        for k, v in P.items():
            te_all = np.concatenate([v[f][0] for f in v]); p_all = np.concatenate([v[f][1] for f in v])
            row[k] = dict(pooled=scores(d.y[te_all], p_all), folds={f: scores(d.y[v[f][0]], v[f][1]) for f in v})
        row['shuffle_kappa'] = [row[f'shuf{s}']['pooled']['kappa'] for s in range(pc['shuffle_seeds'])]
        row['seconds'] = round(time.time() - t0, 1); res[name] = row
        log(dict(feature=name, dim=row['dim'], kappa=round(row['real']['pooled']['kappa'], 4),
                 shuffle=[round(v, 4) for v in row['shuffle_kappa']], C=[v['C'] for v in Cs.values()], s=row['seconds']))
    return res


# ------------------------------------------------------------------ main
def md5(p): return hashlib.md5(open(os.path.join(ROOT, p), 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--mode', choices=['init', 'trained'], required=True)
    ap.add_argument('--cond'); ap.add_argument('--arm'); ap.add_argument('--folds'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); g = yaml.safe_load(open(os.path.join(ROOT, a.config))); fc, pc = g['features'], g['probe']
    c = yaml.safe_load(open(os.path.join(ROOT, g['train_config']))); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conds = [a.cond] if a.cond else g['conditions']; arms = [a.arm] if a.arm else g['trained_arms']
    want = [int(v) for v in a.folds.split(',')] if a.folds else None
    if a.smoke: pc = {**pc, 'C_grid': pc['C_grid'][:2], 'shuffle_seeds': 1, 'max_iter': 50}
    outd = os.path.join(ROOT, g['out_dir']); os.makedirs(outd, exist_ok=True)
    for cond in conds:
        sb = c['sb'][cond]; assert not sb['shortcuts'], 'diagnosis is defined on the clean SB conditions only'
        t0 = time.time(); d = TaskData(c, g['task'], dev, sb=sb); load_s = round(time.time() - t0, 1)
        fold_list = [(f, tr, va, te) for f, tr, va, te in splits(d, c, None) if want is None or f in want]
        if a.smoke:                                  # tiny subsets of every split (pipeline check only)
            r = np.random.default_rng(0); fold_list = [(f, np.sort(r.choice(tr, 600, replace=False)), va[:200], te[:300]) for f, tr, va, te in fold_list[:1]]
        Rg = torch.Generator(device=dev).manual_seed(fc['rp_seed']); Dflat = (d.x.shape[1] * 4 + 1) * 512   # 4 patches of 1 s in 4 s
        R = torch.empty(Dflat, fc['rp_dim'], device=dev, dtype=torch.float16)
        for i in range(0, Dflat, 8192):                  # generated in row blocks (no float32 copy of the whole matrix)
            R[i:i + 8192] = (torch.randn(min(8192, Dflat - i), fc['rp_dim'], device=dev, generator=Rg) / fc['rp_dim'] ** 0.5).half()
        meta = dict(mode=a.mode, cond=cond, task=g['task'], seed=g['seed'], n_windows=int(len(d.y)), n_subjects=len(d.subjects),
                    K=d.K, sb_info=d.sb_info, load_seconds=load_s, folds=[f for f, *_ in fold_list], smoke=a.smoke,
                    device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', torch=torch.__version__,
                    code_md5={p: md5(p) for p in ['scripts/nc_diag_sb.py', 'scripts/nc_train.py', 'src/ese/model/reve_nc.py',
                                                   'src/ese/sb/generate.py', g['train_config'], a.config]},
                    probe=pc, features_cfg=fc)
        log = lambda o: print(json.dumps(dict(cond=cond, **o)), flush=True)
        if a.mode == 'init':
            arm = c['arms'][g['init_arm']]; torch.manual_seed(1000 * g['seed'] + 0)
            n_subj = len(np.unique(d.subj[fold_list[0][1]]))
            net, _ = build(c, arm, d, n_subj, dev); t1 = time.time()
            ix = np.unique(np.concatenate([np.concatenate([tr, va, te]) for _, tr, va, te in fold_list])) if a.smoke else None
            Fx = extract(net, d, dev, g['batch'], R, ix); del net, R; torch.cuda.empty_cache()
            if ix is not None:                       # smoke: scatter into full-size arrays
                full = {}
                for k, v in Fx.items():
                    if k.startswith('_check'): full[k] = v; continue
                    z = np.zeros((len(d.y), *v.shape[1:]), v.dtype); z[ix] = v; full[k] = z
                Fx = full
            meta['extract_seconds'] = round(time.time() - t1, 1); meta['forward_check_max_abs_err'] = float(Fx.pop('_check_err')[0])
            pool = Fx.pop('_pool'); feats = dict(Fx)
            per_fold = collections.defaultdict(dict)       # PCA projections are fitted on each fold's training windows
            for f, tr, va, te in fold_list:
                for k, v in tangent_pca(pool, np.concatenate([tr, va]), fc['tangent_pca_dims'], fc['pca_fit_windows'], g['seed'] + f, dev).items():
                    per_fold[k][f] = v
            del pool; feats.update(per_fold); feats.update(raw_features(d, fc, dev))
            res = run_probes(feats, d, fold_list, pc, g['seed'], dev, log)
            json.dump(dict(meta=meta, features=res), open(os.path.join(outd, f"init_{cond}{'_smoke' if a.smoke else ''}.json"), 'w'), indent=1, default=float)
        else:
            for armn in arms:
                pat = g['trained_run_glob'].format(task=g['task'], cond=cond, arm=armn, seed=g['seed'])
                runs = sorted(r for r in glob.glob(os.path.join(ROOT, c['runs_dir'], pat + ('_smoke' if a.smoke else '')))
                              if glob.glob(os.path.join(r, 'ckpt_fold*.pt')))
                assert runs, f'no run with checkpoints: {pat}'; run = runs[-1]; m = json.load(open(os.path.join(run, 'metrics.json')))
                fl = [(f, tr, va, te) for f, tr, va, te in fold_list if os.path.exists(os.path.join(run, f'ckpt_fold{f}.pt'))]
                assert fl, 'no checkpoint for the requested folds'
                feats = collections.defaultdict(dict); chk = {}; t1 = time.time()
                for f, tr, va, te in fl:
                    ck = torch.load(os.path.join(run, f'ckpt_fold{f}.pt'), map_location='cpu', weights_only=False)
                    if not a.smoke:
                        assert all(np.array_equal(np.asarray(ck[k]), v) for k, v in (('tr', tr), ('va', va), ('te', te))), 'fold split differs'
                    torch.manual_seed(1000 * g['seed'] + f); net, _ = build(c, c['arms'][armn], d, ck['n_subjects'], dev)
                    net.load_state_dict(ck['state'], strict=True); Fx = extract(net, d, dev, g['batch'], R); del net, ck; torch.cuda.empty_cache()
                    ref_k = m['folds'][str(f)]['test']['kappa']; my = scores(d.y[te], Fx['logits'][te][:, :d.K].argmax(1))
                    chk[f] = dict(rescored_test=my, metrics_json_test=m['folds'][str(f)]['test'], forward_check_max_abs_err=float(Fx['_check_err'][0]),
                                  best_epoch=m['folds'][str(f)]['best_epoch'], epochs_run=m['folds'][str(f)]['epochs_run'])
                    log(dict(arm=armn, fold=f, rescored_kappa=round(my['kappa'], 4), metrics_kappa=round(ref_k, 4)))
                    pool = Fx.pop('_pool'); Fx.pop('_check_err')
                    for k, v in tangent_pca(pool, np.concatenate([tr, va]), fc['tangent_pca_dims'], fc['pca_fit_windows'], g['seed'] + f, dev).items():
                        feats[k][f] = v
                    for k, v in Fx.items(): feats[k][f] = v
                    del pool, Fx
                meta.update(arm=armn, run=os.path.relpath(run, ROOT), extract_seconds=round(time.time() - t1, 1), model_check=chk,
                            run_pooled=m['info']['pooled'], run_device=m['info']['device'])
                res = run_probes(dict(feats), d, fl, pc, g['seed'], dev, log)
                json.dump(dict(meta=meta, features=res), open(os.path.join(outd, f"trained_{cond}_{armn}{'_smoke' if a.smoke else ''}.json"), 'w'), indent=1, default=float)
        del d
        if dev.type == 'cuda': torch.cuda.empty_cache()
    print(json.dumps(dict(done=True)), flush=True)


if __name__ == '__main__':
    main()
