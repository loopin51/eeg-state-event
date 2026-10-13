"""Stage 0-c (D-039/D-041): does the stimulus-aligned representation transfer to EAV (conversation, no shared stimulus)?

extract : encode the EAV cache with each run's best.pt (FACED dataset-embedding row, D-041) -> <run>/z_eav.npz,
          and compute DE (5 bands) -> results/features/eav_de.npy
eval    : subject LOSO (42 folds), subject-centred shrinkage LDA, window scores, per-subject valence AUC,
          separately for speaking (primary) and listening. Controls: T1 trial-order-only readout, T2 features with each
          subject's linear+quadratic trend over trial order regressed out, E1 DE without the gamma band (speaking).

usage: python scripts/stage0_eav.py extract RUN [RUN ...]
       python scripts/stage0_eav.py eval OUT_JSON PRIMARY_RUN [FOLD_RUN ...]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.data.dataset import DATASET_IDS
from ese.eval.features import de_features, BANDS
from ese.eval.readout import subject_center, bootstrap_ci
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

EMB_ROW = DATASET_IDS['faced']


def extract(runs):
    import torch
    from ese.model.ese_model import ESEModel
    dev = os.environ.get('ESE_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(int(os.environ['OMP_NUM_THREADS'])); c = Cache(os.path.join(ROOT, 'data/cache'), 'eav')
    P = torch.from_numpy(c.positions).float().to(dev)
    fp = os.path.join(ROOT, 'results/features/eav_de.npy')
    if not os.path.exists(fp): np.save(fp, de_features(c.windows, sfreq=100)); print('DE saved', flush=True)
    for r in runs:
        ck = torch.load(os.path.join(r, 'best.pt'), map_location=dev); cfg = ck['cfg']
        m = ESEModel(n_subjects=ck['n_subjects'], n_datasets=8, **cfg['model']).to(dev); m.load_state_dict(ck['model']); m.eval(); zs = []
        with torch.no_grad():
            for i in range(0, len(c.meta), 128):
                x = torch.from_numpy(np.asarray(c.windows[i:i + 128], dtype=np.float32)).to(dev)
                ds = torch.full((len(x),), EMB_ROW, dtype=torch.long, device=dev)
                _, _, z = m.represent(x, P.expand(len(x), -1, -1), ds); zs.append(z.float().cpu().numpy())
        Z = np.concatenate(zs); np.savez(os.path.join(r, 'z_eav.npz'), z_state=Z, emb_row=EMB_ROW); print(r, Z.shape, flush=True)


def detrend(X, subj, order):
    """Per subject, regress features on [1, t, t^2] of trial order and keep residuals."""
    X = np.asarray(X, np.float64).copy()
    for s in np.unique(subj):
        k = subj == s; t = order[k].astype(float); t = (t - t.mean()) / (t.std() + 1e-9)
        A = np.c_[np.ones_like(t), t, t ** 2]; X[k] -= A @ np.linalg.lstsq(A, X[k], rcond=None)[0]
    return X


def loso(X, y, subj):
    """Subject-centred LDA, leave-one-subject-out; returns per-subject AUC dict."""
    Xc = subject_center(X, subj); out = {}
    for s in np.unique(subj):
        tr, te = subj != s, subj == s
        sc = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xc[tr], y[tr]).decision_function(Xc[te])
        out[int(s)] = float(roc_auc_score(y[te], sc))
    return out


def summ(d): a = np.array(list(d.values())); return dict(mean=float(a.mean()), ci95=bootstrap_ci(a), per_subject=d)


def paired(a, b):
    ks = sorted(set(a) & set(b)); d = np.array([a[k] - b[k] for k in ks])
    return dict(diff=float(d.mean()), ci95=bootstrap_ci(d), wins=int((d > 0).sum()), n=len(d))


def evaluate(out_path, primary, folds):
    t0 = time.time(); meta = Cache(os.path.join(ROOT, 'data/cache'), 'eav').meta.reset_index(drop=True)
    subj = meta['subject'].to_numpy(); val = meta['valence'].to_numpy(); cond = meta['cond'].to_numpy(); order = meta['order'].to_numpy()
    DE = np.load(os.path.join(ROOT, 'results/features/eav_de.npy')); nb = len(BANDS)
    DE4 = DE.reshape(len(DE), -1, nb)[:, :, :4].reshape(len(DE), -1)
    Zp = np.load(os.path.join(primary, 'z_eav.npz'))['z_state'].mean(1)
    res = dict(primary_run=primary, fold_runs=folds, emb_row=EMB_ROW)
    for cname, cv in [('speaking', 1), ('listening', 0)]:
        m = (cond == cv) & (val != 0); y = (val[m] > 0).astype(int); s_ = subj[m]; r = {}
        feats = dict(model=Zp[m], DE=DE[m])
        if cv == 1: feats['DE_no_gamma'] = DE4[m]
        for nm, X in feats.items():
            r[nm] = summ(loso(X, y, s_)); r[nm + '_detrended'] = summ(loso(detrend(X, s_, order[m]), y, s_))
        r['T1_order_only'] = summ(loso(np.c_[order[m], order[m] ** 2].astype(float), y, s_))
        r['model_minus_DE'] = paired(r['model']['per_subject'], r['DE']['per_subject'])
        r['model_minus_DE_detrended'] = paired(r['model_detrended']['per_subject'], r['DE_detrended']['per_subject'])
        fr = []
        for f in folds:
            Zf = np.load(os.path.join(f, 'z_eav.npz'))['z_state'].mean(1)[m]; fr.append(summ(loso(Zf, y, s_))['mean'])
        r['fold_encoder_means'] = fr
        res[cname] = r
        print(f"[{cname}] model {r['model']['mean']:.3f} {np.round(r['model']['ci95'], 3)} | DE {r['DE']['mean']:.3f} {np.round(r['DE']['ci95'], 3)} | "
              f"model-DE {r['model_minus_DE']['diff']:+.3f} {np.round(r['model_minus_DE']['ci95'], 3)} ({r['model_minus_DE']['wins']}/{r['model_minus_DE']['n']}) | "
              f"T1 order-only {r['T1_order_only']['mean']:.3f} {np.round(r['T1_order_only']['ci95'], 3)} ({time.time() - t0:.0f}s)", flush=True)
        print(f"[{cname}] detrended: model {r['model_detrended']['mean']:.3f} {np.round(r['model_detrended']['ci95'], 3)} | DE {r['DE_detrended']['mean']:.3f} "
              f"{np.round(r['DE_detrended']['ci95'], 3)} | model-DE {r['model_minus_DE_detrended']['diff']:+.3f} {np.round(r['model_minus_DE_detrended']['ci95'], 3)}"
              + (f" | DE no-gamma {r['DE_no_gamma']['mean']:.3f} {np.round(r['DE_no_gamma']['ci95'], 3)}" if cv == 1 else '')
              + f" | fold encoders {np.round(fr, 3)}", flush=True)
        json.dump(res, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'extract': extract(sys.argv[2:])
    else: evaluate(sys.argv[2], sys.argv[3], sys.argv[4:])
