"""Shared helpers for F-IV T2 (D-109): rebuild a stored B1 job (configs/stageB_b1.yaml, scripts/stageB_b1_train.py) for
inference only and compute decoder estimates with the B1 input path. Nothing here trains or calibrates.

load_job(K, F, phaserand) rebuilds RunData with the job's own subject list (trainers + targets, the B1 order), loads
shared.pt and writes target_layers.pt into the target rows, so that the network is the B1 job's network at evaluation time.
"""
import os, sys, json, glob, numpy as np, torch, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ese.model.brain_clip import BrainModule, clip_scores, INVALID
from stageB_b1_train import RunData


def path(p): return p if os.path.isabs(p) else os.path.join(ROOT, p)


def b1_folder(c, K, F, phaserand=False):
    pat = os.path.join(ROOT, c['out_root'], f"*_stageB_b1_{c['decoder_setting']}_k{K}_f{F}{'_phaserand' if phaserand else ''}")
    hits = [h for h in sorted(glob.glob(pat)) if os.path.exists(os.path.join(h, 'scores.npz'))]
    assert len(hits) == 1, f'{pat}: {len(hits)} folders'; return hits[0]


class Job:
    """one stored B1 job at evaluation time (eval mode, frozen); est() follows stageB_b1_train.main's fwd"""

    def __init__(self, b1_config, K, F, phaserand=False, dev='cuda'):
        c = yaml.safe_load(open(path(b1_config))); sp = c['split']
        st = c['settings'][c['decoder_setting']]; c['eeg']['reference'] = st['reference']; source = st['phaserand_source' if phaserand else 'source']
        self.folder = b1_folder(c, K, F, phaserand); run = yaml.safe_load(open(os.path.join(self.folder, 'config.yaml')))['run']
        test_runs = [r - 1 for r in sp['run_folds'][F]]; valid_runs = [r - 1 for r in sp['valid_runs'][F]]
        train_runs = [r for r in range(20) if r not in test_runs + valid_runs]
        targets = sp['subject_folds'][K] + sp.get('extra_targets', {}).get(K, sp.get('extra_targets', {}).get(str(K), []))
        trainers = sorted(s for j, fs in enumerate(sp['subject_folds']) if j != K for s in fs)
        assert run['trainers'] == trainers and run['targets'] == targets and run['source'] == source, 'stored job differs from the split'
        assert [r for r in run['eval_runs']] == test_runs and run['train_runs'] == train_runs and run['valid_runs'] == valid_runs
        self.c, self.K, self.F, self.source = c, K, F, source
        self.trainers, self.targets, self.nT = trainers, targets, len(trainers)
        self.test_runs, self.valid_runs, self.train_runs = test_runs, valid_runs, train_runs
        self.D = RunData(c, trainers + targets, train_runs, valid_runs, test_runs, source); self.D.to_device(dev)
        m = c['model']
        self.net = BrainModule(out_dim=self.D.F.shape[0], n_subjects=len(trainers) + len(targets), merger_channels=m['merger_channels'],
                               pos_dim=m['pos_dim'], merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'],
                               hidden=m['hidden'], depth=m['depth'], subject_layers=True).to(dev)
        state = torch.load(os.path.join(self.folder, 'shared.pt'), map_location=dev); self.net.load_state_dict(state)
        tl = torch.load(os.path.join(self.folder, 'target_layers.pt'), map_location=dev)
        assert tl.shape[0] == len(targets)
        with torch.no_grad(): self.net.subject.weights[self.nT:] = tl
        self.net.eval()
        for p in self.net.parameters(): p.requires_grad_(False)
        self.amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=c['optim']['precision'] == 'bf16'); self.dev = dev

    def subject_index(self, s): return (self.trainers + self.targets).index(s)

    def est(self, items, s, shift=0.0, chunk=None):
        """decoder estimates [n, F, T] (float32, on the device) of subject s for items [(run0, t)], EEG window moved by shift"""
        chunk = chunk or self.c['eval']['score_chunk']; si = self.subject_index(s); out = []
        with torch.no_grad():
            for b0 in range(0, len(items), chunk):
                it = items[b0:b0 + chunk]; X, P, _ = self.D.batch(it, [si] * len(it), shift)
                with torch.autocast(**self.amp): out.append(self.net(X, P, torch.full((len(it),), si, device=self.dev)).float())
        return torch.cat(out)

    def recorded_windows(self, items):
        """recorded w2v candidate windows [n, F, T] (float16, normalised with the run fold's training-run statistics)"""
        D = self.D; return torch.stack([D.Fg[:, D.feat_idx(r, t) + D.ar] for r, t in items])


def scores(est, cand, block=512):
    """clip_scores of est [n, F, T] against candidate windows [m, F, T] in blocks -> [n, m] float32 numpy"""
    return torch.cat([clip_scores(est, cand[b:b + block].float()) for b in range(0, len(cand), block)], 1).cpu().numpy()


def b1_mrp(S):
    """B1 within-run MRP of a square score matrix (diagonal truth, all other queries of the run, exclusion 0)"""
    ok = ~np.isnan(np.diag(S)); rk = (S[ok] > np.diag(S)[ok][:, None]).sum(1); return float(np.mean(1 - rk / (len(S) - 1))), int(ok.sum())


# ---------- matched sets, query set Q, candidate windows (D-109 sections 3-5) ----------

def parse_key(k):
    """'r{run 1-based}_w{word index within the run, 0-based}' -> (run0, widx)"""
    r, w = k.split('_'); return int(r[1:]) - 1, int(w[1:])


def load_sets(v, dev=False):
    """matched sets {query key: dict(cands=[19 keys], ...)} and attributes {key: [run, t, tau, n_w, n_L, n_S, D_T, seq]}"""
    from t2_io import read_sets, read_attrs, full
    return full(read_sets(path(v['matched_sets_dev' if dev else 'matched_sets']))[1]), read_attrs(path(v['match_attrs']))


def eeg_ok(job, t, run0, margin):
    """the EEG window moved by +-margin stays inside the run (D-109 section 4 (b))"""
    D, w, e = job.D, job.c['window'], job.c['eeg']
    lo = int(round((t + w['tmin'] + e['offset_s'] - margin) * D.fs)); hi = int(round((t + w['tmin'] + e['offset_s'] + margin) * D.fs)) + D.T
    return lo >= 0 and hi <= D.lens[run0]


class TTS:
    def __init__(self, d):
        d = path(d); m = json.load(open(os.path.join(d, 'meta.json')))
        self.F = np.load(os.path.join(d, 'w2v_run.npy'), mmap_mode='r'); self.offs, self.lens = m['run_offsets'], m['run_lengths']

    def windows(self, job, taus_runs):
        """TTS candidate windows [m, F, T] float16 on the job device, normalised with the job's run-fold training statistics"""
        D = job.D; T = D.T; out = []
        for tau, r in taus_runs:
            i0 = int(round((tau + job.c['window']['tmin']) * D.fs)); assert 0 <= i0 and i0 + T <= self.lens[r]
            out.append(np.asarray(self.F[:, self.offs[r] + i0:self.offs[r] + i0 + T], np.float32))
        X = torch.from_numpy(np.stack(out)).to(job.dev); m, sd = torch.from_numpy(D.fmean).to(job.dev), torch.from_numpy(D.fstd).to(job.dev)
        return ((X - m) / sd).half()


def lse_rows(S_lag, same_cols, c):
    """S_lag [L, n, m] float64; same_cols: boolean [m] of the same-run Q columns used for tau (D-109 section 5)"""
    tau = c * S_lag[:, :, same_cols].transpose(1, 0, 2).reshape(S_lag.shape[1], -1).std(1)
    z = S_lag / tau[None, :, None]; mx = z.max(0); return tau[:, None] * (mx + np.log(np.exp(z - mx[None]).sum(0)))


def noise_scale(job, s, n_windows):
    """channel sd of subject s over preprocessed training-run windows (CAR, baseline, clamp; evenly spaced items)"""
    it = job.D.items['train']; it = it[:: max(1, len(it) // n_windows)][:n_windows]; si = job.subject_index(s); acc = []
    with torch.no_grad():
        for b0 in range(0, len(it), 256):
            X, _, _ = job.D.batch(it[b0:b0 + 256], [si] * len(it[b0:b0 + 256]))
            acc.append(X.double().permute(1, 0, 2).reshape(X.shape[1], -1).cpu())
    return torch.cat(acc, 1).std(1).numpy().astype(np.float32)               # [C] sd over all samples of the windows


def est_noise(job, items, s, shift, sd, seed, run0):
    """decoder estimates for white-noise EEG: one noise run per (subject, run) from default_rng([seed, s, run 1-based]) with the
    channel sd of noise_scale, cut at the same window positions as the real EEG; fed as preprocessed input (no CAR/baseline)"""
    D, w, e = job.D, job.c['window'], job.c['eeg']; rng = np.random.default_rng([seed, s, run0 + 1])
    Xr = (rng.standard_normal((len(sd), D.lens[run0]), dtype=np.float32) * sd[:, None]); si = job.subject_index(s); out = []
    bad = job.D.badg[si, run0]; P0 = job.D.posg.masked_fill(bad[:, None], INVALID)
    with torch.no_grad():
        for b0 in range(0, len(items), job.c['eval']['score_chunk']):
            it = items[b0:b0 + job.c['eval']['score_chunk']]
            X = np.stack([Xr[:, int(round((t + w['tmin'] + e['offset_s'] + shift) * D.fs)):int(round((t + w['tmin'] + e['offset_s'] + shift) * D.fs)) + D.T] for r, t in it])
            X = torch.from_numpy(X).to(job.dev); P = P0[None].expand(len(it), -1, -1)
            with torch.autocast(**job.amp): out.append(job.net(X, P, torch.full((len(it),), si, device=job.dev)).float())
    return torch.cat(out)


# ---------- MRP, 30-s-excluded shift null, pigeonhole (D-109 sections 5, 9) ----------

def cand_index(rows, cols, sets, extra=None):
    """[n_rows, 20] column indices: truth first, then its 19 candidates (or extra[key] instead of sets[key]['cands'])"""
    ci = {k: j for j, k in enumerate(cols)}
    return np.array([[ci[k]] + [ci[c] for c in (extra[k] if extra is not None else sets[k]['cands'])] for k in rows])


def cell_obs(S, C):
    """observed MRP per row: (below + 0.5 ties) / 19 over the row's candidates"""
    ar = np.arange(len(C)); v = S[ar[:, None], C]; t = v[:, :1]
    return ((v[:, 1:] < t).sum(1) + 0.5 * (v[:, 1:] == t).sum(1)) / (C.shape[1] - 1)


def cell_null(S, C, rows_t, col_t, col_same, maps, excl, chunk=50):
    """null MRP per draw (mean over kept pairs): the estimate of row i becomes that of its shift partner j = maps[d, i]; the
    truth and candidates of i stay; same-run candidates within excl of t_j and pairs with |t_j - t_i| < excl are left out.
    rows_t [n]; col_t, col_same: time and same-run flag of every column. Returns (null [n_draw], kept pair fraction, kept candidate fraction)."""
    n = len(C); ar = np.arange(n); ct = col_t[C[:, 1:]]; cs = col_same[C[:, 1:]]; out, kp, kc = [], [], []
    for b0 in range(0, len(maps), chunk):
        J = maps[b0:b0 + chunk]                                                    # [B, n]
        v = S[J[..., None], C[None]]                                               # [B, n, 20]
        tj = rows_t[J]; keep_c = ~(cs[None] & (np.abs(ct[None] - tj[..., None]) < excl))
        ok = (np.abs(tj - rows_t[None]) >= excl) & (keep_c.sum(-1) > 0)
        t = v[..., :1]; w = v[..., 1:]; m = ((w < t) & keep_c).sum(-1) + 0.5 * ((w == t) & keep_c).sum(-1)
        mrp = np.where(ok, m / np.maximum(keep_c.sum(-1), 1), 0.0); out.append(mrp.sum(1) / np.maximum(ok.sum(1), 1))
        kp.append(ok.mean(1)); kc.append((keep_c.sum(-1) * ok).sum(1) / np.maximum(19 * ok.sum(1), 1))
    return np.concatenate(out), float(np.mean(np.concatenate(kp))), float(np.mean(np.concatenate(kc)))


def run_maps(rows_t_by_run, n_shift, min_shift, seed):
    """B1 segment-order circular shift (stageB_b1_verdict.shift_maps), one map per run shared by all subjects; runs in sorted order"""
    from stageB_b1_verdict import shift_maps
    return shift_maps({r: {'onset': np.asarray(rows_t_by_run[r])} for r in sorted(rows_t_by_run)}, n_shift, min_shift, np.random.default_rng(seed))


def pooled(M, w):
    """query-weighted mean of subject x run cell values M [S, R] with run weights w [R] (nan cells are not allowed)"""
    return float((M.mean(0) * w).sum() / w.sum())


def pigeonhole(M, w, n_boot, seed):
    rng = np.random.default_rng(seed); ns, nr = M.shape; out = np.empty(n_boot)
    for b in range(n_boot):
        si = rng.integers(0, ns, ns); ri = rng.integers(0, nr, nr); out[b] = pooled(M[np.ix_(si, ri)], w[ri])
    return out


def unmatched(rows, attrs, t, seed, run1):
    """19 random same-run Q queries >= 30 s from the row with a different word sequence (default_rng([seed, run, i]))"""
    out = {}
    for i, k in enumerate(rows):
        pool = [rows[j] for j in range(len(rows)) if abs(t[j] - t[i]) >= 30.0 and attrs[rows[j]][7] != attrs[k][7]]
        out[k] = list(np.random.default_rng([seed, run1, i]).choice(pool, 19, replace=False)) if len(pool) >= 19 else None
    return out
