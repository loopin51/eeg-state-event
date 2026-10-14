"""B2 target-structure diagnosis (D-091 amendment 7 diagnosis, CPU, no EEG, no test runs).
For the B2 language target (R_f{f}, residualised and re-standardised as in stageB_b2_train.LangData) and the B1 w2v target
(standardised feature window as in stageB_b1_train.RunData), on the 20 fixed validation batches (same draws as the training
scripts) and the first 20 training batches (same draws as the shared training), report:
  - CLIP loss of an oracle that outputs the true target (clip_loss as in training, i.e. estimates not normalised), and the
    best loss over a temperature grid on the cosine matrix
  - in-batch cosine of target pairs (mean, share >= 0.9 and >= 0.95), rows with any other target at cosine >= 0.95
  - isotropy: norm of the mean unit target, variance share of the top 1 / 5 / 10 principal components (training windows)
usage: python scripts/stageB_b2_diag_target.py configs/stageB_b2.yaml --subject-fold 0 --run-fold 0"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'src'))
from stageB_b1_train import RunData
from stageB_b2_train import LangData
from ese.model.brain_clip import clip_loss

TAUS = np.logspace(-1, 4, 51)


def batch_stats(Y):
    """Y [B, D] flattened targets (as fed to clip_loss after expansion over time)"""
    B = len(Y); Yt = torch.from_numpy(Y)
    nat = clip_loss(Yt[:, :, None], Yt[:, :, None]).item()
    U = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8); C = U @ U.T; off = C[~np.eye(B, dtype=bool)]
    ce = [torch.nn.functional.cross_entropy(torch.from_numpy(t * C), torch.arange(B)).item() for t in TAUS]
    near = ((C >= 0.95) & ~np.eye(B, dtype=bool)).any(1).mean()
    return dict(oracle_loss=nat, oracle_loss_best_tau=min(ce), best_tau=float(TAUS[int(np.argmin(ce))]), cos_mean=float(off.mean()),
                cos_ge_0_9=float((off >= 0.9).mean()), cos_ge_0_95=float((off >= 0.95).mean()),
                pairs_ge_0_95=float((off >= 0.95).sum() / 2), rows_with_twin_0_95=float(near))


def summarise(rows):
    return {k: [float(np.mean([r[k] for r in rows])), float(np.min([r[k] for r in rows])), float(np.max([r[k] for r in rows]))] for k in rows[0]}


def isotropy(Y):
    U = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8); mu = np.linalg.norm(U.mean(0))
    Yc = Y - Y.mean(0, keepdims=True); s = np.linalg.svd(Yc, compute_uv=False) ** 2; r = s / s.sum()
    return dict(mean_unit_norm=float(mu), pc1=float(r[:1].sum()), pc5=float(r[:5].sum()), pc10=float(r[:10].sum()), n=int(len(Y)), dim=int(Y.shape[1]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--subject-fold', type=int, required=True); ap.add_argument('--run-fold', type=int, required=True)
    a = ap.parse_args(); c2 = yaml.safe_load(open(a.config)); c = yaml.safe_load(open(os.path.join(ROOT, c2['b1_config']))); sp = c['split']; k, f = a.subject_fold, a.run_fold
    st = c['settings'][c['decoder_setting']]; c['eeg']['reference'] = st['reference']; source = st['source']
    test_runs = [r - 1 for r in sp['run_folds'][f]]; valid_runs = [r - 1 for r in sp['valid_runs'][f]]
    train_runs = [r for r in range(20) if r not in test_runs + valid_runs]
    targets = sp['subject_folds'][k] + sp.get('extra_targets', {}).get(k, [])
    trainers = sorted(s for j, fs in enumerate(sp['subject_folds']) if j != k for s in fs); nT = len(trainers); subjects = trainers + targets
    bs = c['optim']['batch_size']; out = dict(subject_fold=k, run_fold=f, batch_size=bs, chance=float(np.log(bs)))
    D2 = LangData(c, c2, subjects, train_runs, valid_runs, valid_runs, source, f)       # eval runs = valid: test runs never touched
    R = ((D2.R - D2.rmean) / D2.rstd).astype(np.float32)
    D1 = RunData(c, subjects, train_runs, valid_runs, valid_runs, source)

    def y2(items): return R[D2.idx(items, False)]

    def y1(items):
        fi = [D1.feat_idx(r, t) for r, t in items]
        return np.stack([((np.asarray(D1.F[:, i:i + D1.T], np.float32) - D1.fmean) / D1.fstd).reshape(-1) for i in fi])

    for name, D, yf, seed in (('B2_language', D2, y2, c2['seed']), ('B1_w2v', D1, y1, c['seed'] if 'seed' in c else None)):
        va, tr = D.items['valid'], D.items['train']; vr = np.random.default_rng(0); vb = []
        for _ in range(20): vb.append(vr.choice(len(va), bs, replace=False)); vr.integers(0, nT, bs)
        rng = np.random.default_rng(seed); tb = []
        for _ in range(20): tb.append(rng.choice(len(tr), bs, replace=False)); rng.integers(0, nT, bs)
        res = out[name] = dict(n_valid_windows=len(va), n_train_windows=len(tr), train_seed=seed)
        res['valid_batches'] = summarise([batch_stats(yf([va[i] for i in b])) for b in vb])
        res['train_batches'] = summarise([batch_stats(yf([tr[i] for i in b])) for b in tb])
        sub = np.random.default_rng(1).choice(len(tr), min(len(tr), 4000), replace=False)
        Ytr = yf([tr[i] for i in sub])
        if name == 'B1_w2v': Ytr = Ytr.reshape(len(Ytr), D1.F.shape[0], -1).mean(-1)      # w2v: time-mean per window for isotropy
        res['isotropy_train'] = isotropy(Ytr)
        print(name, json.dumps(res), flush=True)
    p = os.path.join(ROOT, 'results/stageB_b2/diag_target_k%d_f%d.json' % (k, f)); os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w'), indent=1); print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
