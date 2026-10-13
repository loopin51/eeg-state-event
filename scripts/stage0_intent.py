"""Stage 0-a / 0-b (D-039): BCI2020 Track 3 pipeline validation and per-user calibration curves.

0a  pipeline validation. Train on the official Training set, test on the official Validation set.
      pooled   : one model on all 15 subjects' Training sets (10% stratified hold-out for early stopping)
      per_subj : one model per subject (10% of its Training set for early stopping)
    Methods: EEGNet (braindecode defaults, 5-class CE, Adam 1e-3, batch 64, <=150 epochs, patience 20 on hold-out
    loss) under P0 / P1 / P2, and DE + shrinkage LDA. Control: pooled EEGNet on the pre-cue window (-0.5..0 s), P1.
0b  calibration curves. For target subject s, k in {25, 50, 100, 250} class-balanced calibration trials from its
    Training set (5 draws), tested on its Validation set (50).
      M1 DE(P1, 6 bands) + LDA   M2 frozen front end (190-subject model) + LDA   M3 EEGNet from scratch (P1)
      M4 EEGNet pretrained on the other 14 subjects (P1), fine-tuned on the k trials; k = 0 is no fine-tuning.

usage: python scripts/stage0_intent.py 0a OUT_JSON [--seeds 3]
       python scripts/stage0_intent.py 0b OUT_JSON
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.eval.features import de_features, BANDS
from ese.eval.readout import subject_center, bootstrap_ci
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
K = 5; KS = [25, 50, 100, 250]; N_DRAWS = 5
DEV = torch.device(os.environ.get('ESE_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
BANDS6 = dict(BANDS, gamma_high=(45, 95))


def load(p):
    """-> X float16 [N, C, T], meta DataFrame (subject, session, label), sfreq, onset sample."""
    if p == 'p0':
        c = Cache(os.path.join(ROOT, 'data/cache'), 'bci2020')
        return np.asarray(c.windows), c.meta[['subject', 'session', 'label']].reset_index(drop=True), 100, 50
    d = os.path.join(ROOT, 'data/cache', f'bci2020_{p}'); h = json.load(open(os.path.join(d, 'header.json')))
    return np.load(os.path.join(d, 'x.npy')), pd.read_parquet(os.path.join(d, 'meta.parquet')), h['sfreq'], h['onset_sample']


def lda_pred(Xtr, ytr, Xte): return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xtr, ytr).predict(Xte)


def summ(d):
    a = np.array(list(d.values())); return dict(mean=float(a.mean()), ci95=bootstrap_ci(a), per_subject={int(k): v for k, v in d.items()})


def paired(a, b):
    ks = sorted(set(a) & set(b)); d = np.array([a[k] - b[k] for k in ks])
    return dict(diff=float(d.mean()), ci95=bootstrap_ci(d), wins=int((d > 0).sum()), n=len(d))


# ------------------------------------------------------------------ EEGNet
def make_net(C, T, fs, seed):
    from braindecode.models import EEGNet
    torch.manual_seed(seed); return EEGNet(n_chans=C, n_outputs=K, n_times=T, sfreq=fs).to(DEV)


def batches(idx, bs, rng):
    o = rng.permutation(idx); return [o[i:i + bs] for i in range(0, len(o), bs)]


def logits(net, X, idx, bs=512):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs): out.append(net(torch.from_numpy(X[idx[i:i + bs]].astype(np.float32)).to(DEV)).cpu())
    return torch.cat(out)


def train(net, X, y, tr, rng, lr, epochs, es=None, patience=20):
    """Adam, batch 64. With es (hold-out indices): keep the lowest hold-out loss state, stop after `patience`."""
    opt = torch.optim.Adam(net.parameters(), lr=lr); best, state, bad, ep = np.inf, None, 0, 0
    for ep in range(epochs):
        net.train()
        for b in batches(tr, 64, rng):
            if len(b) < 2: continue
            loss = F.cross_entropy(net(torch.from_numpy(X[b].astype(np.float32)).to(DEV)), torch.from_numpy(y[b]).to(DEV))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if es is not None:
            v = float(F.cross_entropy(logits(net, X, es), torch.from_numpy(y[es])))
            if v < best - 1e-4: best, bad, state = v, 0, {k: t.detach().clone() for k, t in net.state_dict().items()}
            else: bad += 1
            if bad >= patience: break
    if state is not None: net.load_state_dict(state)
    return dict(epochs=ep + 1, best_holdout_loss=None if es is None else best)


def strat_holdout(idx, y, frac, rng):
    """Stratified (by label) hold-out of `frac` of idx."""
    ho = np.concatenate([rng.choice(idx[y[idx] == c], max(1, int(round(frac * (y[idx] == c).sum()))), replace=False) for c in range(K)])
    return np.setdiff1d(idx, ho), np.sort(ho)


def acc_by_subject(pred, y, subj, te):
    return {int(s): float((pred[subj[te] == s] == y[te][subj[te] == s]).mean()) for s in np.unique(subj[te])}


# ------------------------------------------------------------------ 0-a
def run_0a(out_path, seeds):
    res = dict(chance=1 / K, seeds=seeds); t0 = time.time()
    for p in ['p1', 'p0', 'p2']:
        X, meta, fs, onset = load(p); y = meta['label'].to_numpy(); subj = meta['subject'].to_numpy(); sess = meta['session'].to_numpy()
        C, T = X.shape[1:]; tr_all = np.where(sess == 0)[0]; te = np.where(sess == 1)[0]; r = {}
        # DE + LDA
        De = de_features(X, sfreq=fs, bands=BANDS6 if p == 'p1' else BANDS, nperseg=int(fs))
        within = {int(s): float((lda_pred(De[(subj == s) & (sess == 0)], y[(subj == s) & (sess == 0)], De[(subj == s) & (sess == 1)])
                                 == y[(subj == s) & (sess == 1)]).mean()) for s in np.unique(subj)}
        Dc = subject_center(De, subj); pooled = acc_by_subject(lda_pred(Dc[tr_all], y[tr_all], Dc[te]), y, subj, te)
        r['DE_per_subj'], r['DE_pooled'] = summ(within), summ(pooled)
        print(f"[0a {p}] DE per-subject {r['DE_per_subj']['mean']:.3f} {np.round(r['DE_per_subj']['ci95'], 3)} | "
              f"pooled {r['DE_pooled']['mean']:.3f} {np.round(r['DE_pooled']['ci95'], 3)} ({time.time() - t0:.0f}s)", flush=True)
        # pooled EEGNet (+ pre-cue control under P1)
        windows = [('full', slice(None))] + ([('precue', slice(0, onset))] if p == 'p1' else [])
        for wname, sl in windows:
            Xw = np.ascontiguousarray(X[:, :, sl]); per_seed = []
            for seed in range(seeds):
                rng = np.random.default_rng(seed)
                tr, ho = strat_holdout(tr_all, y, 0.1, rng)
                net = make_net(C, Xw.shape[2], fs, seed); info = train(net, Xw, y, tr, rng, 1e-3, 150, es=ho)
                pred = logits(net, Xw, te).argmax(1).numpy(); per_seed.append(acc_by_subject(pred, y, subj, te))
                print(f"[0a {p}] EEGNet pooled {wname} seed {seed}: {np.mean(list(per_seed[-1].values())):.3f} "
                      f"(epochs {info['epochs']}, {time.time() - t0:.0f}s)", flush=True)
            mean_seed = {s: float(np.mean([d[s] for d in per_seed])) for s in per_seed[0]}
            r[f'EEGNet_pooled_{wname}'] = dict(summ(mean_seed), seed_means=[float(np.mean(list(d.values()))) for d in per_seed])
        # per-subject EEGNet
        ps = {}
        for s in np.unique(subj):
            rng = np.random.default_rng(1000 + int(s)); idx = np.where((subj == s) & (sess == 0))[0]
            tr, ho = strat_holdout(idx, y, 0.1, rng); tes = np.where((subj == s) & (sess == 1))[0]
            net = make_net(C, T, fs, 1000 + int(s)); train(net, X, y, tr, rng, 1e-3, 150, es=ho)
            ps[int(s)] = float((logits(net, X, tes).argmax(1).numpy() == y[tes]).mean())
        r['EEGNet_per_subj'] = summ(ps)
        print(f"[0a {p}] EEGNet pooled {r['EEGNet_pooled_full']['mean']:.3f} {np.round(r['EEGNet_pooled_full']['ci95'], 3)} | "
              f"per-subject {r['EEGNet_per_subj']['mean']:.3f} {np.round(r['EEGNet_per_subj']['ci95'], 3)}"
              + (f" | pre-cue {r['EEGNet_pooled_precue']['mean']:.3f} {np.round(r['EEGNet_pooled_precue']['ci95'], 3)}" if p == 'p1' else '')
              + f" ({time.time() - t0:.0f}s)", flush=True)
        res[p] = r; json.dump(res, open(out_path, 'w'), indent=1)
    res['P1_minus_P0_EEGNet_pooled'] = paired(res['p1']['EEGNet_pooled_full']['per_subject'], res['p0']['EEGNet_pooled_full']['per_subject'])
    json.dump(res, open(out_path, 'w'), indent=1)


# ------------------------------------------------------------------ 0-b
def calib_draw(idx, y, k, rng):
    return np.sort(np.concatenate([rng.choice(idx[y[idx] == c], k // K, replace=False) for c in range(K)]))


def run_0b(out_path):
    X, meta, fs, _ = load('p1'); y = meta['label'].to_numpy(); subj = meta['subject'].to_numpy(); sess = meta['session'].to_numpy()
    m0 = Cache(os.path.join(ROOT, 'data/cache'), 'bci2020').meta
    assert (m0['label'].to_numpy() == y).all() and (m0['subject'].to_numpy() == subj).all(), 'P0/P1 trial order differs'
    De = de_features(X, sfreq=fs, bands=BANDS6, nperseg=int(fs))
    Fz = np.load(os.path.join(ROOT, 'results/features/bci2020_frozen.npy')); assert len(Fz) == len(y)
    C, T = X.shape[1:]; res = dict(ks=KS, n_draws=N_DRAWS, methods={}); acc = {mth: {k: {} for k in KS} for mth in ['M1', 'M2', 'M3', 'M4']}
    acc['M4'][0] = {}; t0 = time.time()
    for s in np.unique(subj):
        s = int(s); cal_pool = np.where((subj == s) & (sess == 0))[0]; te = np.where((subj == s) & (sess == 1))[0]
        # M4 pretraining on the other 14 subjects (both official sets), 10% hold-out for early stopping
        rng = np.random.default_rng(2000 + s); other = np.where(subj != s)[0]; tr, ho = strat_holdout(other, y, 0.1, rng)
        base = make_net(C, T, fs, 2000 + s); info = train(base, X, y, tr, rng, 1e-3, 150, es=ho)
        base_state = {k_: v.clone() for k_, v in base.state_dict().items()}
        acc['M4'][0][s] = float((logits(base, X, te).argmax(1).numpy() == y[te]).mean())
        print(f"[0b] subject {s}: pretrained on 14 ({info['epochs']} epochs) zero-shot {acc['M4'][0][s]:.3f} ({time.time() - t0:.0f}s)", flush=True)
        for k in KS:
            vals = {mth: [] for mth in acc}
            for d in range(N_DRAWS):
                rng = np.random.default_rng(10000 * s + 100 * k + d); cal = calib_draw(cal_pool, y, k, rng)
                vals['M1'].append(float((lda_pred(De[cal], y[cal], De[te]) == y[te]).mean()))
                vals['M2'].append(float((lda_pred(Fz[cal], y[cal], Fz[te]) == y[te]).mean()))
                net = make_net(C, T, fs, 3000 + d); train(net, X, y, cal, rng, 1e-3, 100)
                vals['M3'].append(float((logits(net, X, te).argmax(1).numpy() == y[te]).mean()))
                net = make_net(C, T, fs, 0); net.load_state_dict(base_state); train(net, X, y, cal, rng, 3e-4, 40)
                vals['M4'].append(float((logits(net, X, te).argmax(1).numpy() == y[te]).mean()))
            for mth in vals: acc[mth][k][s] = float(np.mean(vals[mth]))
            print(f"[0b] subject {s} k={k}: " + ' '.join(f"{mth} {acc[mth][k][s]:.3f}" for mth in ['M1', 'M2', 'M3', 'M4'])
                  + f" ({time.time() - t0:.0f}s)", flush=True)
        res['methods'] = {mth: {str(k): summ(v) for k, v in acc[mth].items() if v} for mth in acc}
        json.dump(res, open(out_path, 'w'), indent=1)
    res['primary_M4_k250'] = res['methods']['M4']['250']
    res['M4_minus_M3_k50'] = paired(acc['M4'][50], acc['M3'][50]); res['M2_minus_M1_k50'] = paired(acc['M2'][50], acc['M1'][50])
    res['M4_minus_M1_k250'] = paired(acc['M4'][250], acc['M1'][250])
    json.dump(res, open(out_path, 'w'), indent=1)
    for k in [0] + KS:
        print(f"[0b] k={k:3d} " + ' '.join(f"{mth} {np.mean(list(acc[mth][k].values())):.3f}" for mth in acc if acc[mth].get(k)), flush=True)
    for key in ['M4_minus_M3_k50', 'M2_minus_M1_k50', 'M4_minus_M1_k250']:
        print(f"[0b] {key} {res[key]['diff']:+.3f} {np.round(res[key]['ci95'], 3)} ({res[key]['wins']}/{res[key]['n']})", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['0a', '0b']); ap.add_argument('out_json')
    ap.add_argument('--seeds', type=int, default=3); a = ap.parse_args()
    run_0a(a.out_json, a.seeds) if a.stage == '0a' else run_0b(a.out_json)
