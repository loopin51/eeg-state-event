"""EEGNet baseline under the SAME subject cross-fit splits as the model (roadmap 2.8, docs/05 §2).

EEGNet is trained supervised (binary valence, neutral windows excluded) separately per dataset and per fold,
on that fold's Stage-A TRAIN subjects; early stopping uses the fold's inner VAL subjects (validation AUC);
the fold's TEST subjects are scored once. Splits are read from the split.json of the given fold runs, so every
subject is evaluated exactly once and never seen in training - the same guarantee as eval_crossfit.py.

Reported per dataset: unseen-subject AUC (subject bootstrap CI), ISC of the EEGNet score, and paired per-subject
differences against the model and DE taken from the cross-fit result JSON.
Not done here (cost): subject x clip disjoint retraining.

Usage: python scripts/eegnet_crossfit.py CROSSFIT_JSON OUT_JSON FOLD_RUN_DIR [FOLD_RUN_DIR ...]
       [--datasets a,b] [--epochs 30] [--max-steps N]   (--max-steps: smoke tests)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.eval.readout import bootstrap_ci, isc_test
from sklearn.metrics import roc_auc_score


def batches(idx, bs, rng=None):
    idx = np.array(idx)
    if rng is not None: idx = rng.permutation(idx)
    for i in range(0, len(idx), bs): yield np.sort(idx[i:i + bs])       # sorted -> contiguous memmap reads


@torch.no_grad()
def predict(net, W, idx, dev, bs=512):
    net.eval(); out = []
    for b in batches(idx, bs):
        out.append(net(torch.from_numpy(np.asarray(W[b], dtype=np.float32)).to(dev)).float().squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def train_one(W, y, tr, va, n_chans, n_times, dev, epochs, max_steps, seed):
    from braindecode.models import EEGNet
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = EEGNet(n_chans=n_chans, n_outputs=1, n_times=n_times, sfreq=100).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pos_w = torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())], device=dev, dtype=torch.float32)
    best, best_state, bad, step = -1.0, None, 0, 0
    for ep in range(epochs):
        net.train()
        for b in batches(tr, 128, rng):
            x = torch.from_numpy(np.asarray(W[b], dtype=np.float32)).to(dev); t = torch.from_numpy(y[b].astype(np.float32)).to(dev)
            loss = F.binary_cross_entropy_with_logits(net(x).squeeze(-1), t, pos_weight=pos_w)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            if max_steps and step >= max_steps: break
        v = roc_auc_score(y[va], predict(net, W, va, dev)) if len(np.unique(y[va])) == 2 else float('nan')
        if v > best + 1e-4: best, bad, best_state = v, 0, {k: t.detach().clone() for k, t in net.state_dict().items()}
        else: bad += 1
        if bad >= 5 or (max_steps and step >= max_steps): break
    if best_state is not None: net.load_state_dict(best_state)
    return net, dict(val_auc=float(best), epochs=ep + 1, steps=step)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('crossfit_json'); ap.add_argument('out_json'); ap.add_argument('runs', nargs='+')
    ap.add_argument('--datasets', default='seediv,deap,faced,seedvii'); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--max-steps', type=int, default=None); ap.add_argument('--device', default=None)
    a = ap.parse_args(); dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cf = json.load(open(a.crossfit_json)) if os.path.exists(a.crossfit_json) else {}
    out = {}
    for nm in a.datasets.split(','):
        c = Cache(os.path.join(ROOT, 'data/cache'), nm); meta = c.meta.reset_index(drop=True)
        subj = meta['subject'].to_numpy(); val = meta['valence'].to_numpy(); m = val != 0; y = (val > 0).astype(int)
        scores = np.full(len(meta), np.nan); subj_auc = {}; logs = []
        for k, r in enumerate(a.runs):
            sp = json.load(open(os.path.join(r, 'split.json')))
            tr = np.where(np.isin(subj, sp['train'][nm]) & m)[0]; va = np.where(np.isin(subj, sp['val'][nm]) & m)[0]
            te = np.where(np.isin(subj, sp['test'][nm]))[0]
            t0 = time.time()
            net, info = train_one(c.windows, y, tr, va, c.header['channels'], c.header['length'], dev, a.epochs, a.max_steps, seed=k)
            scores[te] = predict(net, c.windows, te, dev)
            for s_ in sp['test'][nm]:
                kk = te[(subj[te] == s_) & m[te]]
                if len(np.unique(y[kk])) == 2: subj_auc[int(s_)] = float(roc_auc_score(y[kk], scores[kk]))
            info.update(fold=k, seconds=round(time.time() - t0, 1), n_train=len(tr), n_val=len(va), n_test=len(te)); logs.append(info)
            print(f"[{nm}] fold {k}: val AUC {info['val_auc']:.3f}, epochs {info['epochs']}, {info['seconds']}s", flush=True)
        arr = np.array(list(subj_auc.values()))
        traj = {}
        for (cl, s_), g in meta.assign(p=scores).dropna(subset=['p']).groupby(['clip_id', 'subject']):
            if cl >= 0: traj.setdefault(int(cl), {})[int(s_)] = g.sort_values('t_onset')['p'].to_numpy()
        isc = isc_test(traj, n_null=500) if traj else None
        res = dict(unseen_subject_auc=float(arr.mean()), ci95=bootstrap_ci(arr), n_subjects=len(arr), per_subject=subj_auc,
                   isc=isc['isc'] if isc else None, isc_p=isc['p_value'] if isc else None, folds=logs)
        print(f"[{nm}] EEGNet unseen-subject AUC {res['unseen_subject_auc']:.3f} [{res['ci95'][0]:.3f}, {res['ci95'][1]:.3f}] (n={len(arr)}) | ISC {res['isc']:.4f} (p={res['isc_p']:.3f})", flush=True)
        ps = cf.get(nm, {}).get('paired', {}).get('per_subject', {})
        for j, name in ((0, 'model'), (1, 'DE')):
            ks = sorted(set(int(k) for k in ps) & set(subj_auc))
            if not ks: continue
            d = np.array([ps[str(k)][j] - subj_auc[k] for k in ks])
            res[f'paired_{name}_minus_eegnet'] = dict(diff=float(d.mean()), ci95=bootstrap_ci(d), wins=int((d > 0).sum()), n=len(d))
            print(f"[{nm}] PAIRED {name}-EEGNet {d.mean():+.3f} [{bootstrap_ci(d)[0]:+.3f}, {bootstrap_ci(d)[1]:+.3f}] ({name} better in {(d > 0).sum()}/{len(d)})", flush=True)
        out[nm] = res
        json.dump(out, open(a.out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
