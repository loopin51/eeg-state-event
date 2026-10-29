"""Semi-synthetic shortcut benchmark SB (D-128, docs/08 §7 H1) on PhysioNet-MI windows.

The real labels are replaced by random balanced labels per subject, so the recorded imagery information becomes noise.
Two kinds of true signal s (present in every subject, task window only) and four label-correlated shortcuts, which the
training script adds only to the training and inner-validation subjects of each fold (the test fold stays clean):

  induced  8-30 Hz random-phase source (new every trial), class pattern a_k = S^1/2 g_k with g on motor channels
  evoked   class template b_k(t) (1-8 Hz), class pattern a'_k = S^1/2 g'_k (whole scalp)
  c1       EMG-like 45-70 Hz bursts on temporal channels, class-specific amplitude profile, task window only
  c2       slow class state: class template v_k(t) (0.5-2 Hz) x scalp pattern x a smooth positive amplitude trajectory
           over session time drawn per (subject, class); identical in the task window and its paired null window
  c3       EOG-like frontal deflection (Gaussian bump, class-specific latency, trial jitter), task window only
  c4       (not targeted by any REVE-NC part) occipital 16-18 Hz oscillation, class-specific amplitude, task window only

S is the pooled channel covariance of the clean background task windows; patterns, templates and class profiles are
shared by all subjects (cross-subject decoding must be possible), per-trial draws come from the trial's own stream.
Every component is scaled so that its RMS over channels and time equals amp x the background RMS (amp in configs).
"""
import numpy as np

BG_REGIONS = dict(
    motor=['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4'],
    temporal=['FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8'],
    frontal=['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8'],
    occipital=['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'])
SIGNALS, SHORTCUTS = ('induced', 'evoked'), ('c1', 'c2', 'c3', 'c4')


def _band_noise(rng, shape, lo, hi, fs):
    """Gaussian noise band-limited by an FFT mask, unit RMS along the last axis."""
    T = shape[-1]; X = np.fft.rfft(rng.standard_normal(shape), axis=-1); f = np.fft.rfftfreq(T, 1 / fs)
    X[..., (f < lo) | (f > hi)] = 0; x = np.fft.irfft(X, n=T, axis=-1)
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True))


def _unit(a): return a / np.sqrt((a ** 2).mean())


class SB:
    def __init__(self, x_task, names, subj, t_order, K, fs, seed):
        self.n, self.C, self.L = x_task.shape; self.K, self.fs, self.subj, self.t_order = K, fs, subj, t_order
        self.ch = {n: i for i, n in enumerate(names)}; rng = np.random.default_rng([seed, 0])
        sub = rng.choice(self.n, min(self.n, 2000), replace=False)
        Xc = x_task[np.sort(sub)].astype(np.float64); Xc -= Xc.mean(2, keepdims=True)
        S = np.einsum('nct,ndt->cd', Xc, Xc, optimize=True) / (Xc.shape[0] * self.L); e, V = np.linalg.eigh(S)
        self.Sh = (V * np.sqrt(np.maximum(e, 0))) @ V.T; self.bg_rms = float(np.sqrt((Xc ** 2).mean()))
        self.idx = {k: np.array([self.ch[n] for n in v if n in self.ch]) for k, v in BG_REGIONS.items()}
        rid = {None: 0, **{k: i + 1 for i, k in enumerate(BG_REGIONS)}}       # deterministic (str hash is salted per process)
        def pattern(region, k):
            g = np.zeros(self.C); r = self.idx[region] if region else np.arange(self.C)
            g[r] = np.random.default_rng([seed, 1, rid[region], k]).standard_normal(len(r)); return _unit(self.Sh @ g)
        t = np.arange(self.L) / fs; self.t = t
        self.pat = {'induced': np.stack([pattern('motor', k) for k in range(K)]),
                    'evoked': np.stack([pattern(None, 100 + k) for k in range(K)]),
                    'c2': np.stack([pattern(None, 200 + k) for k in range(K)]),
                    'c4': np.stack([pattern('occipital', k) for k in range(K)])}
        r2 = np.random.default_rng([seed, 2])
        self.b_ev = _band_noise(r2, (K, self.L), 1, 8, fs); self.v_c2 = _band_noise(r2, (K, self.L), 0.5, 2, fs)
        nt = len(self.idx['temporal']); self.c1_prof = np.stack([_unit(r2.uniform(0.1, 1.0, nt)) for _ in range(K)])
        self.c3_lat = np.linspace(0.6, 2.4, K); self.c4_amp = np.linspace(0.5, 1.5, K)
        fr = [n for n in BG_REGIONS['frontal'] if n in self.ch]; self.c3_prof = np.array([1.0 if n.startswith('Fp') else 0.5 for n in fr])
        # smooth positive amplitude trajectory per (subject, class) over session time (c2)
        self.c2_traj = {}
        for s in np.unique(subj):
            for k in range(K):
                q = np.random.default_rng([seed, 3, int(s), k]); r, ph = q.standard_normal(3), q.uniform(0, 2 * np.pi, 3)
                self.c2_traj[(int(s), k)] = (r, ph)
        self.seed = seed

    def labels(self, seed):
        """Random balanced labels per subject (the recorded labels are discarded)."""
        y = np.zeros(self.n, np.int64)
        for s in np.unique(self.subj):
            j = np.where(self.subj == s)[0]; q = np.random.default_rng([seed, 4, int(s)])
            y[j] = q.permutation(np.resize(np.arange(self.K), len(j)))
        return y

    def _trial_rng(self, kind, i): return np.random.default_rng([self.seed, 5, SHORTCUTS.index(kind) if kind in SHORTCUTS else 9, int(i)])

    def render(self, kind, y, idx=None):
        """Component `kind` for windows idx with labels y[idx] -> (task [m, C, L], null [m, C, L] or None), unit-free:
        RMS over the full set equals the background RMS (multiply by amp)."""
        idx = np.arange(self.n) if idx is None else np.asarray(idx); m = len(idx); out = np.zeros((m, self.C, self.L), np.float32)
        nul = None; fs, L = self.fs, self.L
        if kind == 'induced':
            for a, i in enumerate(idx):
                src = _band_noise(self._trial_rng(kind, i), (L,), 8, 30, fs); out[a] = self.pat['induced'][y[i]][:, None] * src[None]
        elif kind == 'evoked':
            out[:] = self.pat['evoked'][y[idx]][:, :, None] * self.b_ev[y[idx]][:, None, :]
        elif kind == 'c1':
            ti = self.idx['temporal']
            for a, i in enumerate(idx):
                q = self._trial_rng(kind, i); dur = q.uniform(0.5, 1.0); on = q.uniform(0.2, L / fs - dur - 0.2)
                env = np.zeros(L); k0, k1 = int(on * fs), int((on + dur) * fs); env[k0:k1] = np.hanning(k1 - k0)
                car = _band_noise(q, (len(ti), L), 45, 70, fs); out[a][ti] = self.c1_prof[y[i]][:, None] * car * env[None]
        elif kind == 'c2':
            for a, i in enumerate(idx):
                r, ph = self.c2_traj[(int(self.subj[i]), int(y[i]))]; tau = float(self.t_order[i])
                A = np.exp(0.5 * (r * np.sin(2 * np.pi * np.arange(1, 4) * tau + ph)).sum() / np.sqrt(3))
                out[a] = A * self.pat['c2'][y[i]][:, None] * self.v_c2[y[i]][None]
            nul = out.copy()
        elif kind == 'c3':
            fi = np.array([self.ch[n] for n in BG_REGIONS['frontal'] if n in self.ch])
            for a, i in enumerate(idx):
                q = self._trial_rng(kind, i); lat = self.c3_lat[y[i]] + q.normal(0, 0.1)
                bump = np.exp(-0.5 * ((self.t - lat) / 0.15) ** 2); out[a][fi] = self.c3_prof[:, None] * bump[None]
        elif kind == 'c4':
            for a, i in enumerate(idx):
                q = self._trial_rng(kind, i); osc = _band_noise(q, (L,), 16, 18, fs)
                out[a] = self.c4_amp[y[i]] * q.uniform(0.8, 1.2) * self.pat['c4'][y[i]][:, None] * osc[None]
        else: raise ValueError(kind)
        return out, nul

    def norm(self, kind, y):
        """Scale factor that brings the component's RMS (over all windows) to the background RMS."""
        sub = np.random.default_rng([self.seed, 6]).choice(self.n, min(self.n, 1000), replace=False)
        o, _ = self.render(kind, y, np.sort(sub)); return self.bg_rms / float(np.sqrt((o.astype(np.float64) ** 2).mean()))


def apply_sb(d, sb, c, task):
    """Called by scripts/nc_train.py TaskData: replaces the labels, adds the true signal to every task window and stores
    the shortcut components (d.sc_x, d.sc_xn) that run_fold adds to the training/validation subjects only."""
    p = c['sb_params']; K = d.K
    g = SB(d.x, d.names, d.subj, d.t_order, K, 200, p['seed']); y = g.labels(p['seed'])
    sig, _ = g.render(sb['signal'], y); d.x += (p['amp'][sb['signal']] * g.norm(sb['signal'], y)) * sig; del sig
    d.sc_x = np.zeros_like(d.x) if sb['shortcuts'] else None; d.sc_xn = np.zeros_like(d.xn) if sb['shortcuts'] and d.has_null else None
    d.sb_gen, d.sb_scale = g, {}
    for k in sb['shortcuts']:
        a, nl = g.render(k, y); s = p['amp'][k] * g.norm(k, y); d.sb_scale[k] = s; d.sc_x += s * a
        if nl is not None and d.sc_xn is not None: d.sc_xn += s * nl
        del a, nl
    return y, dict(signal=sb['signal'], shortcuts=list(sb['shortcuts']), amp={k: p['amp'][k] for k in [sb['signal'], *sb['shortcuts']]},
                   bg_rms=g.bg_rms, label_seed=p['seed'])


def reversed_shortcuts(d, sb, idx):
    """Shortcuts rendered with the label map k -> k+1 mod K for windows idx (report-only test, docs/08 §7 H1)."""
    y2 = (d.y + 1) % d.K; sx = np.zeros((len(idx), *d.x.shape[1:]), np.float32)
    for k in sb['shortcuts']:
        a, _ = d.sb_gen.render(k, y2, idx); sx += d.sb_scale[k] * a
    return sx
