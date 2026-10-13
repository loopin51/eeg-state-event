"""Why does the cross read-out fail? Measures, from saved z_*.npz of one run:
  - top canonical correlation between z_state (32d) and z̄_event (256d)   [what xcov should suppress]
  - correlation between the two valence projections specifically
  - mean squared cross-correlation (= the xcov loss value) and the share the top direction contributes
  - valence AUC from z_event's WITHIN-WINDOW DETAIL (z_event(t) - window mean), which is the part the
    slow head cannot represent — the factorization claim concerns this, not the window mean.
Usage: python scripts/diagnose_factorization.py RUN_DIR [--datasets a,b]
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
import sys, json, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.data.cache import Cache
from ese.eval.readout import loso_auc, subject_center, isc_test
from sklearn.cross_decomposition import CCA

ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--datasets', default='seediv,deap,faced,seedvii')
a = ap.parse_args(); out = {}
for nm in a.datasets.split(','):
    p = os.path.join(a.run, f'z_{nm}.npz')
    if not os.path.exists(p): continue
    d = np.load(p); Zs, Ze = d['z_state'], d['z_event_mean']
    meta = Cache(os.path.join(ROOT, 'data/cache'), nm).meta
    subj = meta['subject'].to_numpy(); m = meta['valence'].to_numpy() != 0
    y = (meta['valence'].to_numpy()[m] > 0).astype(int); tkey = meta['trial'].to_numpy() + 1000 * meta['session'].to_numpy()
    S = subject_center(Zs.mean(1), subj); E = subject_center(Ze, subj)
    Ss = (S - S.mean(0)) / (S.std(0) + 1e-8); Es = (E - E.mean(0)) / (E.std(0) + 1e-8)
    C = Es.T @ Ss / (len(Ss) - 1)                                        # [256,32]
    sv = np.linalg.svd(C, compute_uv=False)
    idx = np.random.default_rng(0).choice(len(Ss), min(8000, len(Ss)), replace=False)
    cca = CCA(n_components=1, max_iter=1000).fit(Es[idx], Ss[idx]); u, v = cca.transform(Es[idx], Ss[idx])
    top_cca = float(abs(np.corrcoef(u[:, 0], v[:, 0])[0, 1]))
    rs = loso_auc(S[m], y, subj[m], trial=tkey[m], n_shuffle=0); re = loso_auc(E[m], y, subj[m], trial=tkey[m], n_shuffle=0)
    ps = S @ rs['direction_mean']; pe = E @ re['direction_mean']
    r_val = float(np.corrcoef(ps, pe)[0, 1])
    # The property that actually distinguishes the branches: is the valence score STIMULUS-ALIGNED
    # across subjects? Decoding valence within a subject is not the same as tracking the stimulus.
    iscs = {}
    for tag, proj in (('state', ps), ('event', pe)):
        traj = {}
        for (cl, sb), g in meta.assign(p=proj).groupby(['clip_id', 'subject']):
            if cl < 0: continue
            traj.setdefault(int(cl), {})[int(sb)] = g.sort_values('t_onset')['p'].to_numpy()
        iscs[tag] = isc_test(traj, n_null=500) if traj else None
    out[nm] = dict(xcov_loss=float((C ** 2).mean()), top_singular=float(sv[0]), top_share=float(sv[0] ** 2 / (sv ** 2).sum()),
                   top_cca=top_cca, valence_proj_corr=r_val, auc_state=rs['mean'], auc_event_mean=re['mean'],
                   isc_state=iscs['state']['isc'] if iscs['state'] else None, isc_state_p=iscs['state']['p_value'] if iscs['state'] else None,
                   isc_event=iscs['event']['isc'] if iscs['event'] else None, isc_event_p=iscs['event']['p_value'] if iscs['event'] else None)
    o = out[nm]
    print(f"[{nm}] top_CCA={top_cca:.3f} valence-proj r={r_val:.3f} | AUC state {rs['mean']:.3f} event {re['mean']:.3f} "
          f"| ISC state {o['isc_state']:.4f} (p={o['isc_state_p']:.3f}) event {o['isc_event']:.4f} (p={o['isc_event_p']:.3f})", flush=True)
json.dump(out, open(os.path.join(a.run, 'factorization_diag.json'), 'w'), indent=1)
