"""Phase 3.0 (D-035): is imagined-speech phrase decoding possible ACROSS subjects on BCI2020 Track 3?

5-class accuracy (chance 0.2), subject bootstrap CI.
  cross-subject : leave-one-subject-out over 15 subjects (primary)
  within-subject: per subject, fit on the Training split (session 0), test on the Validation split (session 1)
Methods:
  DE       : 5-band differential entropy x 64 ch, subject-centred, shrinkage LDA
  frozen   : time-mean of E(t) from the adopted 190-subject front end (never saw BCI2020), subject-centred, LDA
  EEGNet   : braindecode EEGNet, supervised; cross-subject folds hold out 2 training subjects for early stopping,
             within-subject folds hold out the last 50 Training-split trials for early stopping
Usage: python scripts/phase3_feasibility.py OUT_JSON [--ckpt PATH] [--skip-eegnet] [--skip-frozen] [--eegnet-max-steps N]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.eval.features import de_features
from ese.eval.readout import subject_center, bootstrap_ci
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

K = 5


def lda_pred(Xtr, ytr, Xte):
    return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xtr, ytr).predict(Xte)


def eval_features(X, y, subj, sess, center=True):
    Xc = subject_center(X, subj) if center else X
    cross, within = {}, {}
    for s in np.unique(subj):
        te, tr = subj == s, subj != s
        cross[int(s)] = float((lda_pred(Xc[tr], y[tr], Xc[te]) == y[te]).mean())
        a, b = (subj == s) & (sess == 0), (subj == s) & (sess == 1)
        within[int(s)] = float((lda_pred(X[a], y[a], X[b]) == y[b]).mean())      # within: raw features, one subject
    return cross, within


@torch.no_grad()
def frozen_features(ckpt, W, pos, dev, bs=128):
    from ese.model.ese_model import ESEModel
    ck = torch.load(ckpt, map_location=dev); cfg = ck['cfg']
    m = ESEModel(n_subjects=ck['n_subjects'], n_datasets=8, **cfg['model']).to(dev); m.load_state_dict(ck['model']); m.eval()
    P = torch.from_numpy(pos).float().to(dev); out = []
    for i in range(0, len(W), bs):
        x = torch.from_numpy(np.asarray(W[i:i + bs], dtype=np.float32)).to(dev)
        ds = torch.zeros(len(x), dtype=torch.long, device=dev)                     # SEED-IV row (D-035)
        E = m.front(x, P.expand(len(x), -1, -1), ds)                               # [B, N_p, 512]
        out.append(E.mean(1).float().cpu().numpy())
    return np.concatenate(out)


def eegnet_fit_predict(W, y, tr, va, te, n_chans, n_times, dev, seed, max_steps=None, epochs=40):
    from braindecode.models import EEGNet
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = EEGNet(n_chans=n_chans, n_outputs=K, n_times=n_times, sfreq=100).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    def pred(idx):
        net.eval(); o = []
        with torch.no_grad():
            for i in range(0, len(idx), 512):
                o.append(net(torch.from_numpy(np.asarray(W[np.sort(idx[i:i + 512])], dtype=np.float32)).to(dev)).argmax(1).cpu().numpy())
        return np.concatenate(o)
    best, state, bad, step = -1, None, 0, 0
    for ep in range(epochs):
        net.train(); order = rng.permutation(tr)
        for i in range(0, len(order), 64):
            b = np.sort(order[i:i + 64])
            x = torch.from_numpy(np.asarray(W[b], dtype=np.float32)).to(dev); t = torch.from_numpy(y[b]).to(dev)
            loss = F.cross_entropy(net(x), t); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            if max_steps and step >= max_steps: break
        acc = float((pred(va) == y[np.sort(va)]).mean())
        if acc > best + 1e-4: best, bad, state = acc, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else: bad += 1
        if bad >= 6 or (max_steps and step >= max_steps): break
    if state is not None: net.load_state_dict(state)
    te = np.sort(te); return pred(te), y[te], dict(val_acc=best, epochs=ep + 1)


def summarize(name, d):
    a = np.array(list(d.values())); return dict(mean=float(a.mean()), ci95=bootstrap_ci(a), per_subject=d, n=len(a))


def paired(a, b):
    ks = sorted(set(a) & set(b)); d = np.array([a[k] - b[k] for k in ks])
    return dict(diff=float(d.mean()), ci95=bootstrap_ci(d), wins=int((d > 0).sum()), n=len(d))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out_json')
    ap.add_argument('--ckpt', default=os.path.join(ROOT, 'results/runs/20260922_2135_clisa_align_only/best.pt'))
    ap.add_argument('--skip-eegnet', action='store_true'); ap.add_argument('--skip-frozen', action='store_true')
    ap.add_argument('--eegnet-max-steps', type=int, default=None); ap.add_argument('--device', default=None)
    a = ap.parse_args(); dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    c = Cache(os.path.join(ROOT, 'data/cache'), 'bci2020'); meta = c.meta
    y = meta['label'].to_numpy(); subj = meta['subject'].to_numpy(); sess = meta['session'].to_numpy()
    res = {'chance': 1 / K}
    t0 = time.time(); X = de_features(c.windows, sfreq=c.header['sfreq'], nperseg=100)
    cr, wi = eval_features(X, y, subj, sess); res['DE'] = dict(cross=summarize('DE', cr), within=summarize('DE', wi))
    print(f"[DE] cross {res['DE']['cross']['mean']:.3f} {np.round(res['DE']['cross']['ci95'], 3)} | within {res['DE']['within']['mean']:.3f} {np.round(res['DE']['within']['ci95'], 3)}  ({time.time()-t0:.0f}s)", flush=True)
    if not a.skip_frozen:
        t0 = time.time(); Z = frozen_features(a.ckpt, c.windows, c.positions, dev)
        cr, wi = eval_features(Z, y, subj, sess); res['frozen'] = dict(cross=summarize('frozen', cr), within=summarize('frozen', wi), ckpt=a.ckpt)
        res['frozen_minus_DE'] = dict(cross=paired(cr, res['DE']['cross']['per_subject']), within=paired(wi, res['DE']['within']['per_subject']))
        print(f"[frozen] cross {res['frozen']['cross']['mean']:.3f} {np.round(res['frozen']['cross']['ci95'], 3)} | within {res['frozen']['within']['mean']:.3f} {np.round(res['frozen']['within']['ci95'], 3)}  ({time.time()-t0:.0f}s)", flush=True)
        p = res['frozen_minus_DE']['cross']; print(f"[frozen-DE] cross {p['diff']:+.3f} {np.round(p['ci95'], 3)} ({p['wins']}/{p['n']})", flush=True)
    if not a.skip_eegnet:
        t0 = time.time(); cr, wi, logs = {}, {}, []
        subs = np.unique(subj)
        for s in subs:
            rest = subs[subs != s]; vs = np.random.default_rng(int(s)).choice(rest, 2, replace=False)
            tr = np.where(np.isin(subj, rest) & ~np.isin(subj, vs))[0]; va = np.where(np.isin(subj, vs))[0]; te = np.where(subj == s)[0]
            p, t, info = eegnet_fit_predict(c.windows, y, tr, va, te, c.header['channels'], c.header['length'], dev, int(s), a.eegnet_max_steps)
            cr[int(s)] = float((p == t).mean()); logs.append(dict(mode='cross', subject=int(s), **info))
            idx = np.where((subj == s) & (sess == 0))[0]; tr, va = idx[:-50], idx[-50:]; te = np.where((subj == s) & (sess == 1))[0]
            p, t, info = eegnet_fit_predict(c.windows, y, tr, va, te, c.header['channels'], c.header['length'], dev, 100 + int(s), a.eegnet_max_steps)
            wi[int(s)] = float((p == t).mean()); logs.append(dict(mode='within', subject=int(s), **info))
            print(f"[EEGNet] subject {s}: cross {cr[int(s)]:.3f} within {wi[int(s)]:.3f}", flush=True)
        res['EEGNet'] = dict(cross=summarize('EEGNet', cr), within=summarize('EEGNet', wi), logs=logs)
        res['EEGNet_minus_DE'] = dict(cross=paired(cr, res['DE']['cross']['per_subject']), within=paired(wi, res['DE']['within']['per_subject']))
        print(f"[EEGNet] cross {res['EEGNet']['cross']['mean']:.3f} {np.round(res['EEGNet']['cross']['ci95'], 3)} | within {res['EEGNet']['within']['mean']:.3f} {np.round(res['EEGNet']['within']['ci95'], 3)}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(a.out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
