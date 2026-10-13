"""Chisco neural-network readouts on the day-held-out split (D-054).

Reads data/cache/chisco_epochs/sub-XX/ (scripts/prepare_chisco_epochs.py). Inputs are z-scored per subject, day and
channel over all of that day's trials and samples (label-free, the waveform analogue of the day centring in D-047).
Splits
  day : each recording day is the test set once; the calibration day just before it (cyclic) is the early-stopping
        validation set; the remaining days train.
  mix : official-style split (position % 5 == 1 test, == 3 validation, rest train), diagnostic only (D-053).
Models
  eegnet       braindecode EEGNet (F1 8, D 2, kernel 125 = 0.5 s at 250 Hz), trained from scratch per subject
  conformer_pt braindecode EEGConformer pretrained on the other four subjects (all their days; validation = each one's
               last day), then fine-tuned on the target subject's calibration days
Metric: balanced accuracy over the classes present on the test day (39 classes, chance about 0.026).
--perm N repeats each fold N times with calibration (train + validation) labels permuted.

usage: python scripts/chisco_nn.py OUT_JSON --model eegnet --input imag --split day [--k all|20] [--perm 0] [--subjects ...]
"""
import os, sys, json, time, argparse, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPO = os.path.join(ROOT, 'data/cache/chisco_epochs'); CKPT = os.path.join(ROOT, 'data/cache/chisco_nn_ckpt')
K = 39; SFREQ = 250
HP = {'eegnet': dict(lr=1e-3, epochs=40, patience=6), 'conformer_pt': dict(lr=2e-4, epochs=40, patience=6),
      'pretrain': dict(lr=2e-4, epochs=30, patience=4), 'batch': 64}


def load(sub, inp):
    d = os.path.join(EPO, f'sub-{sub:02d}'); M = np.load(os.path.join(d, 'meta.npz'))
    X = np.load(os.path.join(d, f'{inp}.npy')).astype(np.float32)
    assert np.isfinite(X).all(), f'non-finite values in {inp}'
    for g in np.unique(M['day']):                      # per-day, per-channel z-score (label-free)
        m = M['day'] == g; mu = X[m].mean((0, 2), keepdims=True); sd = X[m].std((0, 2), keepdims=True) + 1e-6
        X[m] = (X[m] - mu) / sd
    return X, M['label'].astype(np.int64), M['day'], M['run'], M


def make(model, n_chans, n_times):
    from braindecode.models import EEGNet, EEGConformer
    if model == 'eegnet': return EEGNet(n_chans=n_chans, n_outputs=K, n_times=n_times, kernel_length=125)
    return EEGConformer(n_outputs=K, n_chans=n_chans, n_times=n_times)


def bacc_of(net, X, y, dev):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256): out.append(net(torch.from_numpy(X[i:i + 256]).to(dev)).argmax(1).cpu().numpy())
    return balanced_accuracy_score(y, np.concatenate(out)), np.concatenate(out)


def fit(net, Xtr, ytr, Xva, yva, hp, dev, seed, Xte=None, yte=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    w = torch.tensor(len(ytr) / (K * np.maximum(np.bincount(ytr, minlength=K), 1)), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=hp['lr'], weight_decay=1e-2); Xg = torch.from_numpy(Xtr).to(dev, dtype=torch.float16); yg = torch.from_numpy(ytr).to(dev)   # fp16 storage, fp32 compute
    best, state, bad, ep = -1.0, None, 0, 0; curve = []
    for ep in range(hp['epochs']):
        net.train()
        for b in np.array_split(rng.permutation(len(ytr)), max(1, len(ytr) // HP['batch'])):
            bi = torch.from_numpy(b).to(dev); loss = F.cross_entropy(net(Xg[bi].float()), yg[bi], weight=w)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        v, _ = bacc_of(net, Xva, yva, dev)
        if Xte is not None: curve.append((float(v), float(bacc_of(net, Xte, yte, dev)[0]))); continue   # --no-early-stop: record, never stop
        if v > best + 1e-4: best, bad, state = v, 0, {k: t.detach().clone() for k, t in net.state_dict().items()}
        else: bad += 1
        if bad >= hp['patience']: break
    if Xte is not None: return dict(curve=curve, epochs=ep + 1, val_bacc=max(c[0] for c in curve))
    net.load_state_dict(state); return dict(val_bacc=float(best), epochs=ep + 1)


def pretrained(target, inp, subjects, dev, n_chans, n_times):
    os.makedirs(CKPT, exist_ok=True); path = os.path.join(CKPT, f'conformer_{inp}_without_sub-{target:02d}.pt')
    net = make('conformer_pt', n_chans, n_times).to(dev)
    if os.path.exists(path): net.load_state_dict(torch.load(path, map_location=dev)); return net, None
    Xs, ys, Xv, yv = [], [], [], []
    for s in subjects:
        if s == target: continue
        X, y, day, _, _ = load(s, inp); last = day == day.max()
        Xs.append(X[~last]); ys.append(y[~last]); Xv.append(X[last]); yv.append(y[last])
    info = fit(net, np.concatenate(Xs), np.concatenate(ys), np.concatenate(Xv), np.concatenate(yv), HP['pretrain'], dev, 1000 + target)
    torch.save(net.state_dict(), path); return net, info


def folds(split, y, day, run):
    if split == 'mix':
        t = np.arange(len(y)); pos = np.empty(len(y), int); pos[np.lexsort((t, run))] = np.arange(len(y))
        yield 'mix', np.where((pos % 5 != 1) & (pos % 5 != 3))[0], np.where(pos % 5 == 3)[0], np.where(pos % 5 == 1)[0]; return
    days = np.unique(day)
    for i, td in enumerate(days):
        vd = days[i - 1]
        yield int(td), np.where((day != td) & (day != vd))[0], np.where(day == vd)[0], np.where(day == td)[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out_json'); ap.add_argument('--model', required=True, choices=['eegnet', 'conformer_pt'])
    ap.add_argument('--input', required=True, choices=['imag', 'pre', 'exg']); ap.add_argument('--split', default='day', choices=['day', 'mix'])
    ap.add_argument('--k', default='all'); ap.add_argument('--perm', type=int, default=0); ap.add_argument('--subjects', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--no-early-stop', action='store_true', help='exploratory: train the full epoch budget and record the test curve per epoch')
    ap.add_argument('--timing', action='store_true', help='labels shuffled, first fold only, nothing saved (cost estimate without seeing results)')
    a = ap.parse_args(); dev = 'cuda'; t0 = time.time()
    res = dict(args=vars(a), hp=HP, subjects={}) if not os.path.exists(a.out_json) else json.load(open(a.out_json))
    for sub in a.subjects:
        if str(sub) in res['subjects'] and res['subjects'][str(sub)].get('done'): continue
        X, y, day, run, _ = load(sub, a.input); n_chans, n_times = X.shape[1], X.shape[2]; r = dict(folds={}, perm=[])
        if a.timing: y = np.random.default_rng(0).permutation(y)
        if a.model == 'conformer_pt': _, r['pretrain'] = pretrained(sub, a.input, a.subjects if len(a.subjects) == 5 else [1, 2, 3, 4, 5], dev, n_chans, n_times)
        for name, tr, va, te in folds(a.split, y, day, run):
            if a.k != 'all':
                rng = np.random.default_rng(100 * sub + (name if isinstance(name, int) else 0))
                tr = np.concatenate([rng.choice(tr[y[tr] == q], min(int(a.k), (y[tr] == q).sum()), replace=False) for q in np.unique(y[tr])])
            for p in range(a.perm + 1):
                ytr, yva = y[tr].copy(), y[va].copy()
                if p > 0:
                    rng = np.random.default_rng(10_000 * p + sub); yy = rng.permutation(np.r_[ytr, yva]); ytr, yva = yy[:len(tr)], yy[len(tr):]
                if a.model == 'conformer_pt': net, _ = pretrained(sub, a.input, [1, 2, 3, 4, 5], dev, n_chans, n_times)
                else: net = make(a.model, n_chans, n_times).to(dev)
                info = fit(net, X[tr], ytr, X[va], yva, HP[a.model], dev, 10 * sub + p, *((X[te], y[te]) if a.no_early_stop else ()))
                acc, _ = bacc_of(net, X[te], y[te], dev); info.update(test_bacc=float(acc), n_train=int(len(tr)), n_test=int(len(te)))
                if p == 0: r['folds'][str(name)] = info
                else: r['perm'].append(dict(fold=str(name), perm=p, **info))
                print(f'sub-{sub:02d} {a.model} {a.input} {a.split} k={a.k} fold {name} perm {p}: test {acc:.4f} val {info["val_bacc"]:.4f} '
                      f'epochs {info["epochs"]} ({time.time() - t0:.0f}s)', flush=True)
                del net; torch.cuda.empty_cache()
                if a.timing: print(f'TIMING fold {name}: {time.time() - t0:.0f}s, epochs {info["epochs"]}', flush=True); return
        r['mean'] = float(np.mean([f['test_bacc'] for f in r['folds'].values()]))
        if a.perm: r['perm_means'] = [float(np.mean([q['test_bacc'] for q in r['perm'] if q['perm'] == p])) for p in range(1, a.perm + 1)]
        r['done'] = True; res['subjects'][str(sub)] = r; json.dump(res, open(a.out_json, 'w'), indent=1)
        print(f'sub-{sub:02d} mean {r["mean"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
