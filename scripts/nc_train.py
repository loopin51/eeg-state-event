"""D-128 stage 1: task training for every arm of docs/08 §5 (EEGNet, EEGNet-NC, REVE-FT, REVE-Official, REVE-NC and
its ablations) on the label-free caches of scripts/nc_prep.py (k3s admin queue).

usage: python scripts/nc_train.py configs/nc_train.yaml --arm <arm> --task <task> --seed <k> [--shuffle] [--folds 0,1]
                                  [--sb <sb config name>] [--smoke] [--hp key=value ...]

Protocol (docs/08 §4.2, §6): subject 5-fold (fold assignment seed fixed in the config) or within-subject session T -> E.
Early stopping and model selection use only an inner validation split of the training subjects (D-125 lesson); the
weights of the 5 epochs ending at the best inner-validation epoch are averaged and that model is scored once on the
test fold. --shuffle permutes labels within each training subject (label-shuffle control). Writes
results/runs/<date>_nc_<task>_<arm>[_shuf]_s<seed>/ with config.yaml, metrics.json, history.json, preds.npz.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, copy, argparse, datetime, collections
import numpy as np, yaml, torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.nc_cache import NcCache
from ese.model.reve_nc import ReveNC, band_split, add_lora, lora_parameters, frozen


# ------------------------------------------------------------------ data
class TaskData:
    """Windows of one task in RAM: x [n, C, L], xn (paired null window) or None, ref / refn [n, R, L], labels and groups."""
    def __init__(self, c, task, dev, sb=None):
        t = c['tasks'][task]; self.t = t; cache = NcCache(os.path.join(ROOT, c['cache_dir']), t['data']); h = cache.header
        self.lowpass = h['lowpass_hz']
        m = cache.meta.reset_index(drop=True); self.K = t['n_classes']; self.names = h['ch_names']; L0 = h['length']
        tw = h['windows']['task']['slice']; tws = tw if isinstance(tw[0], list) else [tw]
        rows = np.where((m['kind'] == 'task').values & ~np.isin(m['subject'].values, t.get('exclude_subjects', [])))[0]
        pairs = [(r, a, b) for r in rows for a, b in tws]
        nw = h['windows'].get('null'); self.has_null = nw is not None and t.get('null', True)
        if self.has_null:
            na, nb_ = nw['slice']
            if nw['rows'] == 'task': nrow = {r: r for r in rows}
            else:                               # separate null rows (PhysioNet T0): the null event right before the trial
                prev = {}
                for r in rows:
                    s, run, tr = m.at[r, 'subject'], m.at[r, 'run'], m.at[r, 'trial']
                    q = np.where((m['subject'].values == s) & (m['run'].values == run) & (m['trial'].values == tr - 1) & (m['kind'].values == 'null'))[0]
                    prev[r] = int(q[0]) if len(q) else -1
                nrow = prev
            keep = [p for p in pairs if nrow[p[0]] >= 0]; dropped = len(pairs) - len(keep); pairs = keep
        else: dropped = 0
        self.info = dict(n_windows=len(pairs), null_unpaired_dropped=dropped)
        r_idx = np.array([p[0] for p in pairs])
        self.x = np.stack([np.asarray(cache.x[r, :, a:b]) for r, a, b in pairs]).astype(np.float32)
        self.xn = np.stack([np.asarray(cache.x[nrow[r], :, na:nb_]) for r, _, _ in pairs]).astype(np.float32) if self.has_null else None
        refm = (lambda rr, a, b: np.asarray(cache.ref[rr, :, a:b])) if cache.ref is not None else None
        if t.get('unit_factor'):                 # Cho2017: unknown stored unit, matched to PhysioNet-MI (configs/nc_train.yaml)
            self.x *= t['unit_factor']
            if self.has_null: self.xn *= t['unit_factor']
        self.y = m['label'].values[r_idx].astype(np.int64)
        self.subject_names = m['subject'].values[r_idx].astype(str)
        subs = sorted(set(self.subject_names)); self.subjects = subs; sid = {s: i for i, s in enumerate(subs)}
        self.subj = np.array([sid[s] for s in self.subject_names]); self.session = m['session'].values[r_idx].astype(int)
        rec = [f'{s}|{e}' for s, e in zip(self.subject_names, self.session)]; ru = sorted(set(rec)); self.rec = np.array([ru.index(v) for v in rec])
        self.t_order = m['t_order'].values[r_idx].astype(np.float32)
        self.clip = m['clip_id'].values[r_idx]; self.trial = m['trial'].values[r_idx]
        # clipping at +-clip_sd x the recording's median channel SD (REVE pretraining practice), label-free
        k = c['train']['clip_sd']; self.clip_info = {}
        for r in np.unique(self.rec):
            j = np.where(self.rec == r)[0]; med = float(np.median(self.x[j].std(-1)))
            for X in [self.x] + ([self.xn] if self.has_null else []):
                X[j] = np.clip(X[j], -k * med, k * med)
            self.clip_info[int(r)] = med
        if sb is not None:                       # semi-synthetic shortcut benchmark (docs/08 §7 H1), applied to the windows
            from ese.sb.generate import apply_sb
            self.y, self.sb_info = apply_sb(self, sb, c, task); self.sb_cfg = sb
        # noise reference: measured channels + proxies (frontal-pole < 4 Hz, temporal high band), z-scored per recording
        self.hi = (45.0, 70.0) if self.lowpass >= 70 else (30.0, 45.0); ch = {n: i for i, n in enumerate(self.names)}
        self.eog = [ch[n] for n in t['proxies']['eog'] if n in ch]; self.emg = [ch[n] for n in t['proxies']['emg'] if n in ch]
        self.refm_x = np.stack([refm(r, a, b) for r, a, b in pairs]).astype(np.float32) if refm else None
        self.refm_xn = np.stack([refm(r, na, nb_) for r, _, _ in pairs]).astype(np.float32) if refm and self.has_null else None
        self.ref_names = list(h['ref_names']) + ['eog_proxy'] + [f'{self.names[i]}_hi' for i in self.emg]
        self.pos = torch.tensor([json.load(open(os.path.join(ROOT, c['positions'])))[n] for n in self.names], dtype=torch.float32)
        bands = [tuple(b) for b in c['nc_defaults']['bands']]
        self.bands = [b for b in bands if b[1] <= self.lowpass]       # FACED (47 Hz low-pass) drops the 45-70 Hz band
        self.derive(dev)

    def _refs(self, X, meas):
        xt = torch.from_numpy(X); parts = [] if meas is None else [meas]
        parts.append(band_split(xt[:, self.eog].mean(1, keepdim=True), 200, [(0.5, 4.0)])[:, :, 0].numpy())
        parts.append(band_split(xt[:, self.emg], 200, [self.hi])[:, :, 0].numpy())
        return np.concatenate(parts, 1).astype(np.float32)

    def derive(self, dev):
        """Noise references, per-recording band standard deviations (N1), reference z-scoring and the noise-adversary
        target, all from the current windows (label-free). Re-run whenever the windows change (SB shortcuts per fold)."""
        self.ref = self._refs(self.x, self.refm_x); self.refn = self._refs(self.xn, self.refm_xn) if self.has_null else None
        R = len(np.unique(self.rec)); self.scale = np.zeros((R, self.x.shape[1], len(self.bands)), np.float32)
        for r in range(R):
            j = np.where(self.rec == r)[0]; xs = [self.x[j]] + ([self.xn[j]] if self.has_null else [])
            acc = 0.0; n = 0
            for X in xs:
                for i in range(0, len(X), 256):
                    xb = band_split(torch.from_numpy(X[i:i + 256]).to(dev), 200, self.bands)
                    acc = acc + (xb ** 2).sum((0, 3)).double().cpu(); n += xb.shape[0] * xb.shape[-1]
            self.scale[r] = np.sqrt((acc / n).numpy())
            refs = [self.ref[j]] + ([self.refn[j]] if self.has_null else [])
            sd = np.sqrt(np.mean([np.mean(z ** 2, (0, 2)) for z in refs], 0)) + 1e-6
            self.ref[j] /= sd[None, :, None]
            if self.has_null: self.refn[j] /= sd[None, :, None]
        hb = [(30.0, min(70.0, float(self.lowpass)))]                  # noise-adversary target: high-band power of the reference
        out = []
        for i in range(0, len(self.ref), 512):
            z = band_split(torch.from_numpy(self.ref[i:i + 512]), 200, hb)[:, :, 0]; out.append(torch.log((z ** 2).mean((1, 2)) + 1e-6))
        self.noise_pow = torch.cat(out).numpy()


def subject_folds(subjects, k, seed):
    rng = np.random.default_rng(seed); perm = rng.permutation(len(subjects)); return [sorted(perm[f::k].tolist()) for f in range(k)]


def splits(d, c, a):
    """yield (fold, train idx, val idx, test idx) of window indices."""
    t = d.t; seed = c['seed']
    if t['protocol'] == 'subject5fold':
        folds = subject_folds(d.subjects, 5, seed)
        for f, te_s in enumerate(folds):
            tr_s = [s for s in range(len(d.subjects)) if s not in te_s]
            rng = np.random.default_rng(seed + 1000 + f); nv = max(2, int(round(c['train']['val_fraction'] * len(tr_s))))
            va_s = sorted(rng.choice(tr_s, nv, replace=False).tolist()); tr_s = [s for s in tr_s if s not in va_s]
            yield f, np.where(np.isin(d.subj, tr_s))[0], np.where(np.isin(d.subj, va_s))[0], np.where(np.isin(d.subj, te_s))[0]
    elif t['protocol'] == 'within_te':          # per subject: session 0 (T) -> session 1 (E), validation from T
        for f, s in enumerate(range(len(d.subjects))):
            tr = np.where((d.subj == s) & (d.session == 0))[0]; te = np.where((d.subj == s) & (d.session == 1))[0]
            rng = np.random.default_rng(seed + 1000 + f); va = np.sort(rng.choice(tr, int(round(c['train']['val_fraction'] * len(tr))), replace=False))
            yield f, np.setdiff1d(tr, va), va, te
    else: raise ValueError(t['protocol'])


# ------------------------------------------------------------------ models
class EEGNetNC(nn.Module):
    """braindecode EEGNet; with n3 the classifier becomes a linear K+1 head on the flattened features plus adversaries."""
    def __init__(self, n_chans, n_times, K, nc, n_subjects):
        super().__init__()
        import braindecode.models as M
        net = M.EEGNet(n_chans=n_chans, n_outputs=K, n_times=n_times); self.n3 = nc['n3']
        if not self.n3: self.net = net; return
        self.feat = nn.Sequential(*[m for n, m in net.named_children() if n != 'final_layer'], nn.Flatten())
        with torch.no_grad(): D = self.feat(torch.zeros(2, n_chans, n_times)).shape[1]
        self.head = nn.Linear(D, K + 1); h, nbin = nc['adv_hidden'], nc['adv_bins']
        def mlp(o): return nn.Sequential(nn.LayerNorm(D), nn.Linear(D, h), nn.GELU(), nn.Linear(h, o))
        self.adv_time, self.adv_noise, self.adv_null = mlp(nbin), mlp(nbin), mlp(K)
        self.adv_subj = mlp(n_subjects) if n_subjects else None

    def forward(self, x, pos=None, ref=None, scale=None, adv_lambda=0.0):
        mu, sd = x.mean(-1, keepdim=True), x.std(-1, keepdim=True); x = (x - mu) / sd.clamp_min(1e-6)   # per-epoch z-score (D-125)
        if not self.n3: return dict(logits=self.net(x))
        h = self.feat(x); g = h.detach(); out = dict(logits=self.head(h))
        for k in ('time', 'noise', 'null', 'subj'):
            hd = getattr(self, f'adv_{k}')
            if hd is not None: out[f'adv_{k}'], out[f'conf_{k}'] = hd(g), frozen(hd, h)
        return out


def build(c, arm, d, n_subjects, dev):
    nc = {**c['nc_defaults'], **arm.get('nc', {})}; nc['bands'] = d.bands
    if arm['model'] == 'eegnet':
        return EEGNetNC(d.x.shape[1], d.x.shape[2], d.K, nc, n_subjects).to(dev), nc
    from safetensors.torch import load_file
    r = c['reve'][arm.get('size', 'base')]; init = arm.get('init', 'reve')
    w = load_file(os.path.join(ROOT, r['weights'])) if init == 'reve' else None
    if init not in ('reve', 'scratch'):                     # stage 0 continued-pretraining checkpoint
        w = {k[4:]: v for k, v in torch.load(os.path.join(ROOT, init), map_location='cpu').items() if k.startswith('enc.')}
    net = ReveNC(dict(nc=nc, encoder=r['encoder']), d.K, n_subjects, weights=w)
    if arm.get('lora'): add_lora(net.enc.transformer, arm['lora']['r'], arm['lora']['alpha'], arm['lora'].get('dropout', 0.0))
    net = net.to(dev)
    if not nc['n4']:                                         # build the lazy flatten head before the optimizer exists
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            net.eval(); net(torch.zeros(2, *d.x.shape[1:], device=dev), d.pos[None].expand(2, -1, -1).to(dev),
                            torch.zeros(2, d.ref.shape[1], d.x.shape[2], device=dev), torch.ones(2, d.x.shape[1], len(d.bands), device=dev))
    return net, nc


def param_groups(net, arm, phase, T):
    """phase 'probe': new modules only; 'tune': + LoRA (lora arms) or + every encoder weight (full fine-tuning).
    The N1 band-wise patch embedding replaces REVE's patch embedding, so it is treated as an encoder weight: frozen in
    the probe phase and trained with lr_encoder afterwards (training it at lr_head from the start feeds a frozen REVE
    transformer with a moving input embedding; development smoke: no learning, D-128 C10)."""
    if arm['model'] == 'eegnet' or arm.get('init') == 'scratch' and not arm.get('lora'):
        return [dict(params=list(net.parameters()), lr=T['lr'])]
    emb = list(net.band_patch.parameters()) if hasattr(net, 'band_patch') else []
    enc = set(id(p) for p in net.enc.parameters()) | set(id(p) for p in emb)
    lora = set(id(p) for p in lora_parameters(net))
    new = [p for p in net.parameters() if id(p) not in enc]
    groups = [dict(params=new, lr=T['lr_head'])]
    if phase == 'tune':
        tune = [p for p in net.parameters() if id(p) in lora] if arm.get('lora') else list(net.enc.parameters())
        groups.append(dict(params=tune + emb, lr=T['lr_encoder']))
    for p in net.parameters(): p.requires_grad_(False)
    for g in groups:
        for p in g['params']: p.requires_grad_(True)
    return groups


# ------------------------------------------------------------------ training
def scores(y, p):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
    return dict(acc=float(accuracy_score(y, p)), bal_acc=float(balanced_accuracy_score(y, p)), kappa=float(cohen_kappa_score(y, p)))


def batch_to(d, idx, dev, null):
    g = lambda a: torch.from_numpy(a[idx]).to(dev, non_blocking=True)
    out = dict(x=g(d.x), ref=g(d.ref), scale=torch.from_numpy(d.scale[d.rec[idx]]).to(dev), y=g(d.y),
               t=torch.from_numpy(d.t_bin[idx]).to(dev), npow=torch.from_numpy(d.noise_bin[idx]).to(dev),
               subj=torch.from_numpy(d.subj_train[idx]).to(dev))
    if null: out.update(xn=g(d.xn), refn=g(d.refn))
    return out


@torch.no_grad()
def predict(net, d, idx, dev, K, bs=128):
    net.eval(); P = []
    pos = d.pos.to(dev)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]; b = batch_to(d, j, dev, False)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            o = net(b['x'], pos[None].expand(len(j), -1, -1), b['ref'], b['scale'])
        P.append(torch.softmax(o['logits'].float(), -1).cpu())
    P = torch.cat(P).numpy(); return P, P[:, :K].argmax(1)


def snapshot(net):
    """Trainable parameters and buffers only (frozen REVE weights are never copied)."""
    keep = {n for n, p in net.named_parameters() if p.requires_grad} | {n for n, _ in net.named_buffers()}
    return {k: v.detach().to('cpu', copy=True) for k, v in net.state_dict().items() if k in keep}


def average(snaps):
    out = {}
    for k, v in snaps[-1].items():
        vs = [s[k] for s in snaps if k in s]
        out[k] = torch.stack([u.float() for u in vs]).mean(0).to(v.dtype) if v.is_floating_point() else v
    return out


def run_fold(c, arm, d, f, tr, va, te, dev, seed, shuffle, smoke):
    T = {**c['train'], **arm.get('train', {})}; K = d.K; torch.manual_seed(seed); rng = np.random.default_rng(seed)
    y_orig = d.y.copy()
    sc = getattr(d, 'sc_x', None) is not None
    if sc:                                       # SB: shortcuts only in the training and inner-validation subjects of this fold
        tv = np.concatenate([tr, va]); d.x[tv] += d.sc_x[tv]  # removed again at the end of the fold (no clean copy: memory)
        if d.has_null and d.sc_xn is not None: d.xn[tv] += d.sc_xn[tv]
        d.derive(dev)
    if shuffle:                                  # within each training/validation subject; test labels untouched
        for s in np.unique(d.subj[np.concatenate([tr, va])]):
            j = np.concatenate([tr, va])[d.subj[np.concatenate([tr, va])] == s]; d.y[j] = d.y[rng.permutation(j)]
    if smoke: tr = rng.choice(tr, min(len(tr), T['smoke_n']), replace=False); va = va[:min(len(va), 256)]; te = te[:min(len(te), 256)]
    # adversary targets are standardised on the training windows; subject ids re-indexed over training subjects
    nbin = {**c['nc_defaults'], **arm.get('nc', {})}['adv_bins']        # adversary targets: quantile bins of the training windows
    qs = np.linspace(0, 1, nbin + 1)[1:-1]
    d.t_bin = np.digitize(d.t_order, np.quantile(d.t_order[tr], qs)).astype(np.int64)
    d.noise_bin = np.digitize(d.noise_pow, np.quantile(d.noise_pow[tr], qs)).astype(np.int64)
    ts = np.unique(d.subj[tr]); d.subj_train = np.full(len(d.subj), -1); d.subj_train[np.isin(d.subj, ts)] = np.searchsorted(ts, d.subj[np.isin(d.subj, ts)])
    n_subj = len(ts) if d.t['protocol'] == 'subject5fold' else 0
    net, nc = build(c, arm, d, n_subj, dev); null = d.has_null and nc['n3']
    pos = d.pos.to(dev); B = T['batch']; hist, snaps = [], collections.deque(maxlen=T['avg_last'])
    best, best_ep, best_state, bad = -1.0, -1, None, 0
    probe = arm.get('probe_epochs', 0) if arm['model'] == 'reve' and arm.get('init', 'reve') != 'scratch' else 0
    E = T['smoke_epochs'] if smoke else T['max_epochs']; total_steps = max(1, E * (len(tr) // B)); step = 0
    opt = None; phase = None; t_fold = time.time()
    for ep in range(E):
        ph = 'probe' if ep < probe else 'tune'
        if ph != phase:
            phase = ph; groups = param_groups(net, arm, ph, T) if arm['model'] == 'reve' else [dict(params=list(net.parameters()), lr=T['lr'])]
            opt = torch.optim.AdamW(groups, weight_decay=T['weight_decay']); ph0 = ep
            for g in opt.param_groups: g['base_lr'] = g['lr']
        # same schedule for every arm: per phase, linear warm-up over warmup_epochs, cosine decay to 0 at the last epoch
        spe = max(1, len(tr) // B); wu = T['warmup_epochs'] * spe; span = max(1, (E - ph0) * spe)
        net.train(); perm = rng.permutation(tr); t0 = time.time(); tl = collections.defaultdict(float); nb_ = 0; gmax = 0.0
        for i in range(0, len(perm) - B + 1, B):
            j = perm[i:i + B]; b = batch_to(d, j, dev, null); k_ = (ep - ph0) * spe + i // B
            f_ = (k_ + 1) / wu if k_ < wu else 0.5 * (1 + np.cos(np.pi * min(1.0, (k_ - wu) / max(1, span - wu))))
            for g in opt.param_groups: g['lr'] = g['base_lr'] * f_
            lam = nc['adv_lambda'] * (2 / (1 + np.exp(-10 * step / total_steps)) - 1)          # DANN ramp
            yk = F.one_hot(b['y'], K + int(nc['n3'])).float(); x, ref, scale = b['x'], b['ref'], b['scale']
            tt, npw, sj = F.one_hot(b['t'], nc['adv_bins']).float(), F.one_hot(b['npow'], nc['adv_bins']).float(), b['subj']
            if T['mixup_alpha'] > 0:
                l = float(rng.beta(T['mixup_alpha'], T['mixup_alpha'])); q = torch.randperm(len(j), device=dev)
                mix = lambda a: l * a + (1 - l) * a[q]
                x, ref, scale, yk, tt, npw = mix(x), mix(ref), mix(scale), mix(yk), mix(tt), mix(npw)
                sjk = (l, sj, sj[q])
            else: sjk = (1.0, sj, sj)
            if null:
                x = torch.cat([x, b['xn']]); ref = torch.cat([ref, b['refn']]); scale = torch.cat([scale, b['scale']])
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                o = net(x, pos[None].expand(len(x), -1, -1), ref, scale)
            lg = o['logits'].float(); n = len(j)
            loss = -(yk * F.log_softmax(lg[:n], -1)).sum(-1).mean(); tl['ce'] += loss.item()
            if null:
                ln = F.cross_entropy(lg[n:], torch.full((n,), K, device=dev)); loss = loss + nc['w_null'] * ln; tl['null'] += ln.item()
            if 'bias_logits' in o:
                lb = -(yk[:, :K] * F.log_softmax(o['bias_logits'][:n].float(), -1)).sum(-1).mean(); loss = loss + lb; tl['bias'] += lb.item()
            if nc['n3']:                         # heads learn the targets (detached features); encoder: confusion towards uniform
                sce = lambda lg_, tg: -(tg * F.log_softmax(lg_.float(), -1)).sum(-1).mean()
                unif = lambda lg_: -F.log_softmax(lg_.float(), -1).mean(-1).mean()
                la = sce(o['adv_time'][:n], tt) + sce(o['adv_noise'][:n], npw)
                lc = unif(o['conf_time'][:n]) + unif(o['conf_noise'][:n])
                if 'adv_subj' in o:
                    ls = o['adv_subj'][:n].float(); la = la + sjk[0] * F.cross_entropy(ls, sjk[1]) + (1 - sjk[0]) * F.cross_entropy(ls, sjk[2])
                    lc = lc + unif(o['conf_subj'][:n])
                if null: la = la + F.cross_entropy(o['adv_null'][n:].float(), b['y']); lc = lc + unif(o['conf_null'][n:])
                loss = loss + la + lam * lc; tl['adv'] += la.item(); tl['conf'] += lc.item()
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], T['clip']))
            tl['gnorm'] += gn; gmax = max(gmax, gn)
            tl['hnorm'] += float(o['h'].float().norm(dim=-1).mean())
            opt.step(); step += 1; nb_ += 1
        _, pv = predict(net, d, va, dev, K); s = scores(d.y[va], pv)
        snaps.append(snapshot(net))
        hist.append(dict(epoch=ep, phase=phase, seconds=round(time.time() - t0, 1), samples_per_s=round(nb_ * B * (2 if null else 1) / max(time.time() - t0, 1e-6), 1),
                         **{k: v / max(nb_, 1) for k, v in tl.items() if k != 'gmax'}, grad_max=gmax,
                         **({'resid_W': float(torch.linalg.norm(net.resid.S_ns))} if hasattr(net, 'resid') else {}),
                         **{f'val_{k}': v for k, v in s.items()}))
        print(json.dumps(dict(fold=f, **hist[-1])), flush=True)
        if s['bal_acc'] > best: best, best_ep, best_state, bad = s['bal_acc'], ep, average(list(snaps)), 0
        else:
            bad += 1
            if bad >= T['patience'] and ep >= probe: break
    miss = net.load_state_dict(best_state, strict=False); assert not miss.unexpected_keys; Pt, pt = predict(net, d, te, dev, K); _, pv = predict(net, d, va, dev, K)
    res = dict(test=scores(d.y[te], pt), val=scores(d.y[va], pv), best_epoch=best_ep, epochs_run=len(hist),
               per_subject={d.subjects[s]: scores(d.y[te][d.subj[te] == s], pt[d.subj[te] == s]) for s in np.unique(d.subj[te])},
               n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)), seconds=round(time.time() - t_fold, 1),
               params=int(sum(p.numel() for p in net.parameters())), trainable_last=int(sum(p.numel() for p in net.parameters() if p.requires_grad)),
               max_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == 'cuda' else None)
    pred = dict(idx=te, y=d.y[te], subject=d.subj[te], probs=Pt.astype(np.float32))
    if sc and d.sb_cfg.get('test_reversed', True):      # report-only: shortcuts with the label map k -> k+1 on the test fold
        from ese.sb.generate import reversed_shortcuts
        keep = d.x[te].copy(); d.x[te] += reversed_shortcuts(d, d.sb_cfg, te); d.derive(dev)
        Pr, prr = predict(net, d, te, dev, K); res['test_reversed'] = scores(d.y[te], prr); pred['probs_reversed'] = Pr.astype(np.float32)
        d.x[te] = keep
    d.y[:] = y_orig
    if sc:
        d.x[tv] -= d.sc_x[tv]
        if d.has_null and d.sc_xn is not None: d.xn[tv] -= d.sc_xn[tv]
    del net, opt
    if dev.type == 'cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return res, hist, pred


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--arm', required=True); ap.add_argument('--task', required=True)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--folds')
    ap.add_argument('--sb'); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--hp', nargs='*', default=[])
    ap.add_argument('--tag', default=''); a = ap.parse_args()
    c = yaml.safe_load(open(os.path.join(ROOT, a.config))); arm = copy.deepcopy(c['arms'][a.arm])
    for kv in a.hp:                              # dotted overrides, e.g. train.lr_head=3e-4 nc.adv_lambda=0.3
        k, v = kv.split('='); node = arm; ks = k.split('.')
        for q in ks[:-1]: node = node.setdefault(q, {})
        val = yaml.safe_load(v)
        if isinstance(val, str):                 # YAML 1.1 reads 3e-5 (no dot) as a string
            try: val = float(val)
            except ValueError: pass
        node[ks[-1]] = val
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sb = c['sb'][a.sb] if a.sb else None
    t0 = time.time(); d = TaskData(c, a.task, dev, sb=sb); load_s = round(time.time() - t0, 1)
    tag = f"{datetime.date.today().isoformat()}_nc_{a.task}{'_' + a.sb if a.sb else ''}_{a.arm}{'_shuf' if a.shuffle else ''}_s{a.seed}{a.tag}{'_smoke' if a.smoke else ''}"
    out = os.path.join(ROOT, c['runs_dir'], tag); os.makedirs(out, exist_ok=True)
    yaml.safe_dump(dict(config=c, arm_name=a.arm, arm=arm, task=a.task, seed=a.seed, shuffle=a.shuffle, sb=a.sb, hp=a.hp, smoke=a.smoke),
                   open(os.path.join(out, 'config.yaml'), 'w'), sort_keys=False)
    want = [int(v) for v in a.folds.split(',')] if a.folds else None; metrics, history, preds = {}, {}, collections.defaultdict(list)
    for f, tr, va, te in splits(d, c, a):
        if want is not None and f not in want: continue
        res, hist, p = run_fold(c, arm, d, f, tr, va, te, dev, 1000 * a.seed + f, a.shuffle, a.smoke)
        metrics[f], history[f] = res, hist
        for k, v in p.items(): preds[k].append(v)
        print(json.dumps({f'fold{f}': res['test'], 'best_epoch': res['best_epoch'], 'max_mem_gb': res['max_mem_gb']}), flush=True)
        json.dump(history, open(os.path.join(out, 'history.json'), 'w'))
    P = {k: np.concatenate(v) for k, v in preds.items()}
    info = dict(arm=a.arm, task=a.task, seed=a.seed, shuffle=a.shuffle, sb=a.sb, n_classes=d.K, load_seconds=load_s, data=d.info,
                bands=d.bands, ref_names=d.ref_names, seconds=round(time.time() - t0, 1), torch=torch.__version__,
                device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', pooled=scores(P['y'], P['probs'][:, :d.K].argmax(1)) if P else None,
                sb_info=getattr(d, 'sb_info', None))
    json.dump(dict(info=info, folds=metrics), open(os.path.join(out, 'metrics.json'), 'w'), indent=1, default=float)
    np.savez(os.path.join(out, 'preds.npz'), **P); print(json.dumps(info, default=float), flush=True)


if __name__ == '__main__':
    main()
