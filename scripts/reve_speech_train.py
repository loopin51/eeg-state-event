"""D-125 neural arms on the Khanday et al. (2026) speech benchmark protocol (k3s admin queue, RTX 5080).

usage: python scripts/reve_speech_train.py configs/reve_speech.yaml --arm eegnet|labram|reve|reve_shuffle|reve_lr1e4
                                           --task mode|cat|word|bci_within|bci_loso [--smoke]

One job runs every fold of one arm x task. Protocol (paper 2.4, D-125 2): LOSO (UGR 14 subjects, BCI2020 15) or, for
bci_within, Training -> Validation per subject; Adam (weight decay 1e-4), batch 128, cross-entropy, up to 100 epochs,
early stopping with patience 10 on the held-out subject / validation split (accuracy for UGR, balanced accuracy for
BCI2020) as in the paper, and the best epoch's predictions are scored. EEGNet gets per-epoch per-channel z-scored input,
the foundation models microvolts / 100. REVE: official classifier head (pooling "no": attention context from the
pretrained query token concatenated with all tokens, RMSNorm, dropout 0.3, linear); inputs padded with zeros to whole
seconds. Writes results/runs/<date>_rs_<arm>_<task>/ (config.yaml, metrics.json, history.json, preds.npz).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, glob, argparse, datetime
import numpy as np, yaml, torch, torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))


# ------------------------------------------------------------------ data
def load_task(c, task):
    """Per-subject dicts {sub, x [n, C, T] float32 (microvolts), y, split?, ch_names}."""
    t = c['tasks'][task]; d = os.path.join(ROOT, c['cache_dir'], t['data']); out = []
    for k, f in enumerate(sorted(glob.glob(os.path.join(d, 'sub-*.npz')))):
        z = np.load(f); sub = os.path.basename(f)[:-4]; names = [str(n) for n in z['ch_names']]
        if t['data'] == 'bci2020':
            out.append(dict(sub=sub, x=z['x'], y=z['y'].astype(np.int64), split=z['split'], ch_names=names)); continue
        if task == 'mode':                      # overt 0 / covert 1 from the speech epochs, rest 2 subsampled to match
            sp, mode, rest = z['speech'], z['mode'].astype(np.int64), z['rest']
            rng = np.random.default_rng(c['seed'] + k); idx = np.sort(rng.choice(len(rest), int(round(len(sp) / 2)), replace=False))
            x = np.concatenate([sp, rest[idx]]); y = np.concatenate([mode, np.full(len(idx), 2, np.int64)])
        else:
            m = z['word'] >= 0; x = z['speech'][m]; y = (z['cat'] if task == 'cat' else z['word'])[m].astype(np.int64)
        out.append(dict(sub=sub, x=x, y=y, ch_names=names))
    assert out and all(s['ch_names'] == out[0]['ch_names'] for s in out)
    return out


def folds(subs, protocol):
    if protocol == 'within':
        for s in subs:
            tr, te = s['split'] == 0, s['split'] == 1
            yield s['sub'], (s['x'][tr], s['y'][tr], np.zeros(tr.sum(), int)), (s['x'][te], s['y'][te])
    else:
        for i, s in enumerate(subs):
            tr = [o for j, o in enumerate(subs) if j != i]
            yield s['sub'], (np.concatenate([o['x'] for o in tr]), np.concatenate([o['y'] for o in tr]),
                             np.concatenate([np.full(len(o['y']), j) for j, o in enumerate(tr)])), (s['x'], s['y'])


def prepare(x, arm_model, c):
    if arm_model == 'eegnet':
        mu, sd = x.mean(-1, keepdims=True), x.std(-1, keepdims=True); return ((x - mu) / np.maximum(sd, 1e-6)).astype(np.float32)
    x = (x / c['train']['fm_scale']).astype(np.float32)
    if arm_model == 'reve':
        m = c['reve']['pad_to_multiple']; T = x.shape[-1]; P = -T % m
        if P: x = np.concatenate([x, np.zeros(x.shape[:-1] + (P,), np.float32)], -1)
    return x


# ------------------------------------------------------------------ models
class ReveClassifier(nn.Module):
    """REVE encoder + official head, pooling "no" (reve_eeg src/models/classifier.py)."""
    def __init__(self, c, n_classes, n_chans, n_times):
        super().__init__()
        from safetensors.torch import load_file
        from ese.third_party.reve import Reve, ReveConfig
        from ese.third_party.reve.modeling_reve import RMSNorm
        r = c['reve']; self.enc = Reve(ReveConfig(**r['encoder'])); self.enc.load_state_dict(load_file(os.path.join(ROOT, r['weights'])), strict=True)
        from ese.model.reve_readout import checkpointed
        checkpointed(self.enc.transformer)               # per-layer activation checkpointing: memory only, same result (D-092)
        E = r['encoder']['embed_dim']; size, step = r['encoder']['patch_size'], r['encoder']['patch_size'] - r['encoder']['patch_overlap']
        P = (n_times - size) // step + 1; D = (n_chans * P + 1) * E; self.E = E
        self.head = nn.Sequential(RMSNorm(D), nn.Dropout(r['dropout']), nn.Linear(D, n_classes))

    def forward(self, x, pos):
        t = self.enc(x, pos.expand(x.shape[0], -1, -1)).flatten(1, 2)                  # [B, C*P, E]
        q = self.enc.cls_query_token.expand(t.shape[0], -1, -1)
        ctx = torch.softmax(q @ t.transpose(1, 2) / self.E ** 0.5, -1) @ t              # [B, 1, E]
        return self.head(torch.cat([ctx, t], 1).flatten(1))


def build(arm_model, c, n_classes, names, n_chans, n_times, dev):
    if arm_model == 'reve':
        bank = json.load(open(os.path.join(ROOT, c['reve']['positions']))); miss = [n for n in names if n not in bank]
        assert not miss, f'channels missing from the REVE position bank (D-125 stop rule): {miss}'
        m = ReveClassifier(c, n_classes, n_chans, n_times); pos = torch.tensor([bank[n] for n in names], dtype=torch.float32)[None]
        return m.to(dev), (lambda net, xb: net(xb, pos.to(dev)))
    import braindecode.models as M
    if arm_model == 'eegnet':
        m = M.EEGNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times); return m.to(dev), (lambda net, xb: net(xb))
    # LaBraM: pretrained weights at their native length, temporal embedding cut to our patches (as scripts/extract_fm_features.py)
    snap = glob.glob(os.path.join(ROOT, c['arms']['labram']['weights'], 'hub', 'models--braindecode--labram-pretrained', 'snapshots', '*'))
    assert len(snap) == 1, snap                          # staged HF snapshot folder (local files only, no hub request)
    chs = [{'ch_name': n} for n in names]
    full = M.Labram.from_pretrained(snap[0], n_outputs=0, sfreq=200, n_chans=n_chans, chs_info=chs)
    m = M.Labram(n_outputs=n_classes, n_times=n_times, sfreq=200, n_chans=n_chans, chs_info=chs)
    sd, tgt = full.state_dict(), m.state_dict()
    for k, v in list(sd.items()):
        if k in tgt and tgt[k].shape != v.shape and k.endswith('temporal_embedding'): sd[k] = v[:, :tgt[k].shape[1]].clone()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith(('final_layer', 'head', 'fc_norm')) for k in missing), (missing, unexpected)
    return m.to(dev), (lambda net, xb: net(xb, ch_names=names))


# ------------------------------------------------------------------ training
def scores(y, p):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score
    return dict(acc=float(accuracy_score(y, p)), bal_acc=float(balanced_accuracy_score(y, p)),
                wf1=float(f1_score(y, p, average='weighted', zero_division=0)), kappa=float(cohen_kappa_score(y, p)))


@torch.no_grad()
def predict(net, fwd, x, dev, batch=256):
    net.eval(); out = []
    for i in range(0, len(x), batch): out.append(fwd(net, torch.from_numpy(x[i:i + batch]).to(dev)).argmax(1).cpu())
    return torch.cat(out).numpy()


def run_fold(c, a, arm, task, fold_i, tr, te, names, n_classes, dev, smoke):
    T = c['train']; data = c['tasks'][task]['data']; am = arm['model']
    lr = arm.get('lr', T['lr'][data][am]); seed = c['seed'] + fold_i; torch.manual_seed(seed); rng = np.random.default_rng(seed)
    xtr, ytr, gtr = tr; xte, yte = te
    if arm.get('shuffle_labels'):
        ytr = ytr.copy()
        for g in np.unique(gtr): j = np.where(gtr == g)[0]; ytr[j] = ytr[rng.permutation(j)]
    if smoke: k = rng.choice(len(ytr), min(512, len(ytr)), replace=False); xtr, ytr = xtr[k], ytr[k]
    xtr, xte = prepare(xtr, am, c), prepare(xte, am, c)
    net, fwd = build(am, c, n_classes, names, xtr.shape[1], xtr.shape[2], dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=T['weight_decay']); ce = nn.CrossEntropyLoss()
    key = 'bal_acc' if data == 'bci2020' else 'acc'; best, best_ep, best_pred, bad, hist = -1.0, -1, None, 0, []
    xtr_t, ytr_t = torch.from_numpy(xtr), torch.from_numpy(ytr); n = len(ytr); B = T['batch']
    for ep in range(2 if smoke else T['max_epochs']):
        net.train(); perm = rng.permutation(n); tl, t0 = 0.0, time.time()
        for i in range(0, n, B):
            j = perm[i:i + B]
            if len(j) < 2: continue                          # BatchNorm needs two samples
            loss = ce(fwd(net, xtr_t[j].to(dev, non_blocking=True)), ytr_t[j].to(dev)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tl += loss.item() * len(j)
        p = predict(net, fwd, xte, dev); s = scores(yte, p)
        hist.append(dict(epoch=ep, train_loss=tl / n, seconds=round(time.time() - t0, 1), **s))
        if s[key] > best: best, best_ep, best_pred, bad = s[key], ep, p, 0
        else:
            bad += 1
            if bad >= T['patience']: break
    res = dict(scores(yte, best_pred), best_epoch=best_ep, epochs_run=len(hist), n_train=int(n), n_test=int(len(yte)), lr=lr,
               params=int(sum(q.numel() for q in net.parameters())), max_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == 'cuda' else None)
    del net, opt
    if dev.type == 'cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return res, hist, best_pred


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--arm', required=True); ap.add_argument('--task', required=True)
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--max-folds', type=int); a = ap.parse_args()
    c = yaml.safe_load(open(os.path.join(ROOT, a.config))); arm = c['arms'][a.arm]
    assert a.task in arm.get('tasks', c['tasks']), (a.arm, a.task)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.backends.cudnn.benchmark = False
    subs = load_task(c, a.task); names = subs[0]['ch_names']; K = c['tasks'][a.task]['n_classes']
    tag = f"{datetime.date.today().isoformat()}_rs_{a.arm}_{a.task}{'_smoke' if a.smoke else ''}"
    out = os.path.join(ROOT, c['runs_dir'], tag); os.makedirs(out, exist_ok=True)
    yaml.safe_dump(dict(config=c, arm=a.arm, task=a.task, smoke=a.smoke), open(os.path.join(out, 'config.yaml'), 'w'), sort_keys=False)
    metrics, history, preds, t0 = {}, {}, {}, time.time()
    for i, (sub, tr, te) in enumerate(folds(subs, c['tasks'][a.task]['protocol'])):
        if a.max_folds is not None and i >= a.max_folds: break
        res, hist, p = run_fold(c, a, arm, a.task, i, tr, te, names, K, dev, a.smoke)
        metrics[sub], history[sub], preds[f'{sub}_pred'], preds[f'{sub}_true'] = res, hist, p, te[1]
        print(json.dumps({sub: res}), flush=True)
        json.dump(history, open(os.path.join(out, 'history.json'), 'w'))
    info = dict(arm=a.arm, task=a.task, n_classes=K, n_subjects=len(metrics), seconds=round(time.time() - t0, 1),
                device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', torch=torch.__version__,
                n_channels=len(names), n_times=int(subs[0]['x'].shape[-1]))
    json.dump(dict(info=info, subjects=metrics), open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    np.savez(os.path.join(out, 'preds.npz'), **preds); print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
