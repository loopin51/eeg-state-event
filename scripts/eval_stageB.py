"""Stage B (docs/04 §8, docs/05 §3): extract z_state / z_event for every window of each dataset
with a Stage-A checkpoint, then run the read-out battery:
  LOSO shrinkage-LDA valence AUC (+ trial-level label shuffle, subject bootstrap CI)
  ISC of the LDA score across subjects per clip vs circular-shift null
  circular-shift label read-out (Δ = 0/30/60/90/180 s)
  cross read-out: valence from z̄_event (should be ~chance if factorized)
  subject probe on z_state (linear, trial-held-out; should be near chance)
Usage: python scripts/eval_stageB.py RUN_DIR [--datasets seediv,deap]
"""
import os
# Cap BLAS/OpenMP thread pools BEFORE numpy/torch import. The read-out battery runs thousands of small
# LDA / logistic fits; unconstrained OpenBLAS (MAX_THREADS=64) grabs every core and spends most of it
# spinning in thread-pool synchronisation. 8 threads is as fast and leaves the shared server usable.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import os, sys, json, argparse, numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from torch.utils.data import DataLoader
from ese.data.dataset import MultiCache, collate
from ese.model.ese_model import ESEModel
from ese.eval.readout import loso_auc, subject_center, isc_test, circular_shift_labels, bootstrap_ci, continuous_label_test
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


@torch.no_grad()
def extract(model, ds, dev, bs=64, nw=0):
    zs, ze = [], []
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate)
    for b in dl:
        E, z_event, z_state = model.represent(b['x'].to(dev), b['positions'].to(dev), b['dataset_id'].to(dev))
        zs.append(z_state.float().cpu().numpy()); ze.append(z_event.mean(1).float().cpu().numpy())
    return np.concatenate(zs), np.concatenate(ze)          # [N,N_s,ds], [N,de]


def subject_probe(Z, subject, trial_key, seed=0, max_n=20000, max_iter=200, folds=3):
    """Auxiliary diagnostic ("is subject identity still linearly present?"), so keep it cheap:
    subsample windows, few lbfgs iterations, 3 folds. The full 54k x 256 x 20-class fit with
    max_iter=2000 took hours and dominated the whole battery."""
    rng = np.random.default_rng(seed); Zc = np.asarray(Z, dtype=np.float64)
    if len(Zc) > max_n:
        idx = rng.choice(len(Zc), max_n, replace=False); Zc, subject, trial_key = Zc[idx], subject[idx], trial_key[idx]
    Zc = (Zc - Zc.mean(0)) / (Zc.std(0) + 1e-8)                      # scaling helps lbfgs converge fast
    accs = []
    for tr, te in GroupKFold(folds).split(Zc, subject, groups=trial_key):
        clf = LogisticRegression(max_iter=max_iter, C=1.0).fit(Zc[tr], subject[tr])
        accs.append(float((clf.predict(Zc[te]) == subject[te]).mean()))
    return dict(acc=float(np.mean(accs)), chance=float(1 / len(np.unique(subject))), n=int(len(Zc)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--datasets', default=None); ap.add_argument('--ckpt', default='best.pt'); ap.add_argument('--device', default=None); ap.add_argument('--reuse', action='store_true', help='reuse saved z_*.npz instead of re-encoding'); ap.add_argument('--extract-only', action='store_true', help='only encode and save z_*.npz (cross-fitting)')
    a = ap.parse_args(); dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(os.path.join(a.run, a.ckpt), map_location=dev); cfg = ck['cfg']
    names = a.datasets.split(',') if a.datasets else cfg['datasets']
    model = ESEModel(n_subjects=ck['n_subjects'], n_datasets=8, **cfg['model']).to(dev); model.load_state_dict(ck['model']); model.eval()
    report = dict(run=a.run, ckpt=a.ckpt, step=ck.get('step'))
    for nm in names:
        ds = MultiCache(os.path.join(ROOT, cfg['cache_root']), [nm], subjects={nm: cfg['subjects'][nm]} if cfg.get('subjects') else None)
        zp = os.path.join(a.run, f'z_{nm}.npz')
        if a.reuse and os.path.exists(zp):
            d = np.load(zp); Zs, Ze = d['z_state'], d['z_event_mean']; print(f'[{nm}] reusing {zp}', flush=True)
        else:
            Zs, Ze = extract(model, ds, dev); np.savez(zp, z_state=Zs, z_event_mean=Ze)
        meta = ds.meta.reset_index(drop=True)
        if a.extract_only: print(f'[{nm}] extracted {Zs.shape}', flush=True); continue
        feat = Zs.mean(1)                                          # [N, d_state] context-level state
        subj = meta['subject_local'].to_numpy(); tkey = meta['trial_key'].to_numpy()
        m = meta['valence'].to_numpy() != 0; y = (meta['valence'].to_numpy()[m] > 0).astype(int)
        r = loso_auc(feat[m], y, subj[m], trial=tkey[m], n_shuffle=5)
        res = dict(n_windows=int(len(meta)), n_subjects=int(len(np.unique(subj))),
                   valence_lda=dict(mean=r['mean'], ci95=r['ci95'], pooled=r['pooled'], label_shuffle=r['label_shuffle_mean'], per_subject=r['per_subject']))
        print(f"[{nm}] z_state LOSO valence AUC {r['mean']:.3f} {np.round(r['ci95'],3)} shuffle {r['label_shuffle_mean']:.3f}", flush=True)
        # ISC on LDA score (direction from the LOSO folds' mean) per clip
        score = subject_center(feat, subj) @ r['direction_mean']
        traj = {}
        for (cl, s), g in meta.assign(score=score).groupby(['clip_id', 'subject_local']):
            if cl < 0: continue
            traj.setdefault(int(cl), {})[int(s)] = g.sort_values('t_onset')['score'].to_numpy()
        if traj:
            isc = isc_test(traj, n_null=1000); res['isc'] = {k: v for k, v in isc.items() if k != 'null'}
            print(f"[{nm}] ISC {isc['isc']:.4f} null p99 {isc['null_p99']:.4f} p={isc['p_value']:.3f}", flush=True)
        shifts = {}
        for dlt in [0, 30, 60, 90, 180]:
            ys = circular_shift_labels(meta, (meta['valence'].to_numpy() > 0).astype(int), dlt)
            rr = loso_auc(feat[m], ys[m], subj[m], trial=tkey[m], n_shuffle=0); shifts[str(dlt)] = dict(mean=rr['mean'], ci95=rr['ci95'])
        res['circular_shift_auc'] = shifts; print(f"[{nm}] shift: " + ', '.join(f"{k}s={v['mean']:.3f}" for k, v in shifts.items()), flush=True)
        rx = loso_auc(Ze[m], y, subj[m], trial=tkey[m], n_shuffle=3)
        res['cross_valence_from_z_event'] = dict(mean=rx['mean'], ci95=rx['ci95'], label_shuffle=rx['label_shuffle_mean'])
        print(f"[{nm}] cross read-out valence from z_event: {rx['mean']:.3f} (shuffle {rx['label_shuffle_mean']:.3f})", flush=True)
        if 'cont_label' in meta.columns and meta['cont_label'].notna().any():
            ct = continuous_label_test(feat, meta.assign(subject=subj)); res['continuous_label'] = {k: v for k, v in ct.items() if k != 'per_subject'}
            print(f"[{nm}] continuous-label tracking r {ct['mean_r']:.4f} (null p99 {ct['null_p99']:.4f}, p={ct['p_value']:.3f}, {ct['n_trials']} trials)", flush=True)
        res['subject_probe_z_state'] = subject_probe(feat, subj, tkey); res['subject_probe_z_event'] = subject_probe(Ze, subj, tkey)
        print(f"[{nm}] subject probe acc z_state {res['subject_probe_z_state']['acc']:.3f} z_event {res['subject_probe_z_event']['acc']:.3f} (chance {res['subject_probe_z_state']['chance']:.3f})", flush=True)
        report[nm] = res
    json.dump(report, open(os.path.join(a.run, 'stageB.json'), 'w'), indent=1, default=float); print('STAGE_B_DONE', flush=True)

if __name__ == '__main__':
    main()
