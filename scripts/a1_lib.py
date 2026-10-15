"""F-III A1 (D-105) shared scoring: J250 query offsets, per-item shifted EEG batches, lag search (LS) score aggregation,
30-s-excluded within-run MRP, pseudo-target k2 refit and the one-pass extended-window forward of N(b)/(c).
Definitions are read from configs/a1.yaml; nothing here touches test_cells.json or scores.npz.
"""
import os, json, numpy as np, torch
from ese.model.brain_clip import clip_scores, clip_loss, INVALID
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lags(a):
    g = a['lag_search']['lags_s']; return [float(round(x, 2)) + 0.0 for x in np.arange(g['lo'], g['hi'] + 1e-9, g['step'])]


def run_onsets(run):
    """sorted unique word onsets of a 0-based run (the J250 draw is over all of them, before any query filter)"""
    W = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True); return np.unique(W['onset'][W['run'] == run])


def j_offsets(a, subject, items, width=None):
    """delta per item (run, t) for one subject: default_rng([seed, subject, run]).uniform(-w, w) over the run's sorted onsets"""
    w = a['j250']['width_s'] if width is None else width; cache, out = {}, []
    for r, t in items:
        if r not in cache:
            on = run_onsets(r); d = np.random.default_rng([a['j250']['seed'], int(subject), int(r)]).uniform(-w, w, len(on))
            cache[r] = dict(zip(np.round(on, 6).tolist(), d.tolist()))
        out.append(cache[r][round(float(t), 6)])
    return np.array(out)


def inside(D, items, margin):
    """indices of items whose EEG windows moved by up to +-margin (and whose target windows) stay inside the run"""
    w, e = D.c['window'], D.c['eeg']; keep = []
    for k, (r, t) in enumerate(items):
        dur = D.lens[r] / D.fs
        if t + w['tmin'] + e['offset_s'] - margin >= 0 and t + w['tmin'] >= 0 and t + w['tmax'] + e['offset_s'] + margin <= dur - 1.0 / D.fs:
            keep.append(k)
    return np.array(keep, int)


def batch_shift(D, items, si, shifts, ext=0):
    """RunData.batch with a per-item EEG shift (seconds) and `ext` extra samples on both sides (extended window).
    With ext = 0 and a constant shift this is RunData.batch exactly (same rounding, CAR, baseline, clamp, mask).
    With ext > 0 the baseline is the first baseline_s of the central (unextended) window, common to every crop.
    A window moved outside its run is clamped to the run edge (training jitter only; scored queries are inside by `inside`)."""
    e, w = D.c['eeg'], D.c['window']; T = D.T + 2 * ext; X, runs = [], []
    for (r, t), s_i, sh in zip(items, si, shifts):
        j = int(round((t + w['tmin'] + e['offset_s'] + sh) * D.fs)) - ext
        j = min(max(j, 0), D.lens[r] - T)                                  # a training jitter never reads outside the run (clamped)
        i = D.offs[r] + j
        X.append(D.eeg[D.subjects[s_i]][:, i:i + T]); runs.append(r)
    X = torch.from_numpy(np.stack(X).astype(np.float32)).to(D.dev, non_blocking=True)
    S = torch.as_tensor(np.asarray(si), device=D.dev); bad = D.badg[S, torch.as_tensor(runs, device=D.dev)]
    if e['reference'] == 'car':
        ok = (~bad).float()[..., None]; X = X - (X * ok).sum(1, keepdim=True) / ok.sum(1, keepdim=True)
    nb = int(round(e['baseline_s'] * D.fs)); X = X - X[..., ext:ext + nb].mean(-1, keepdim=True); X = X.clamp(-e['clamp'], e['clamp'])
    P = D.posg[None].expand(len(S), -1, -1).masked_fill(bad[..., None], INVALID)
    return X, P


def targets(D, items):
    """true w2v target windows [n, F, T] (standardised as in training)"""
    return torch.stack([D.Fg[:, D.feat_idx(r, t) + D.ar] for r, t in items]).float()


def targets_shift(D, items, shift):
    """target windows moved by `shift` s ([n, F, T]) and a mask of the items whose moved window stays inside the run"""
    w = D.c['window']; ok, fi = [], []
    for r, t in items:
        good = t + shift + w['tmin'] >= 0 and D.feat_idx(r, t + shift) + D.T <= D.offs[r] + D.lens[r]
        ok.append(good); fi.append(D.feat_idx(r, t + shift) if good else D.feat_idx(r, t))
    fi = torch.as_tensor(fi, device=D.dev)
    return D.Fg[:, fi[:, None] + D.ar].permute(1, 0, 2).float(), torch.as_tensor(ok, device=D.dev)


def forward(net, D, items, si, shifts, amp, chunk=128):
    out = []
    with torch.no_grad():
        for b0 in range(0, len(items), chunk):
            it = items[b0:b0 + chunk]; X, P = batch_shift(D, it, si[b0:b0 + chunk], shifts[b0:b0 + chunk])
            with torch.autocast(**amp): out.append(net(X, P, torch.as_tensor(np.asarray(si[b0:b0 + chunk]), device=D.dev)).float())
    return torch.cat(out)


def lag_stack(net, D, items, si, base, lag_list, cand, amp):
    """[L, n_query, n_cand] CLIP scores of the EEG windows moved by base_i + l (one forward per lag) against `cand`"""
    return torch.stack([clip_scores(forward(net, D, items, si, base + l, amp), cand) for l in lag_list])


def aggregate(S, rule):
    """S [L, n, m] -> [n, m]; max, or lseC with tau_i = C x sd of row i over (lag, candidate)"""
    if rule == 'max': return S.max(0).values
    assert rule.startswith('lse'), rule; c = float(rule[3:])
    tau = (c * S.permute(1, 0, 2).reshape(S.shape[1], -1).std(1)).clamp(min=1e-6)[None, :, None]
    return tau[0] * torch.logsumexp(S / tau, 0)


def mrp_excl(S, t, excl):
    """within-run MRP of the diagonal truth among the candidates >= excl s from the query (excl 0: every other query)"""
    tt = torch.as_tensor(np.asarray(t), device=S.device, dtype=torch.float64); n = S.shape[0]
    far = (tt[:, None] - tt[None]).abs() >= excl if excl > 0 else ~torch.eye(n, dtype=torch.bool, device=S.device)
    above = (S > S.diagonal()[:, None]) & far
    return float((1 - above.sum(1).double() / far.sum(1).clamp(min=1).double()).mean().item())


def refit_layer(net, D, j, mean_others, runs_k, items_train, cfg, rng, amp, loss_fn=None):
    """pseudo-target layer: subject index j restarts from `mean_others` and is refitted on its own runs_k for cfg updates
    (Adam lr, batch from the B1 calibration settings), no validation; the shared network stays frozen. Returns the layer."""
    W = net.subject.weights; items_k = [x for x in items_train if x[0] in runs_k]; cal = D.c['calibration']
    with torch.no_grad(): W[j] = mean_others
    W.requires_grad_(True); opt = torch.optim.Adam([W], lr=cal['lr']); loss_fn = loss_fn or (lambda est, Y: clip_loss(est, Y))
    for _ in range(cfg['updates']):
        idx = rng.choice(len(items_k), cal['batch_size'], replace=False); it = [items_k[i] for i in idx]; si = np.full(len(it), j)
        X, P, Y = D.batch(it, si)
        with torch.autocast(**amp): est = net(X, P, torch.as_tensor(si, device=D.dev)).float()
        loss = loss_fn(est, Y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    W.requires_grad_(False); return W[j].detach().clone()


def ext_forward(net, D, items, si, base, lag_list, amp):
    """one forward of the extended window (+-max|l|), cropped into one estimate per lag: [L, n, F, T]"""
    ext = int(round(max(abs(l) for l in lag_list) * D.fs)); X, P = batch_shift(D, items, si, base, ext=ext)
    with torch.no_grad(), torch.autocast(**amp): E = net(X, P, torch.as_tensor(np.asarray(si), device=D.dev)).float()
    return torch.stack([E[..., ext + int(round(l * D.fs)): ext + int(round(l * D.fs)) + D.T] for l in lag_list])
