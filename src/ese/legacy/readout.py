"""Stage-2: parameter-free mean-difference direction read-out on the FROZEN
Stage-1 backbone (Doc II 2.4, "Option B fixed direction probe").

Given frozen H(t) [B, D, T]:
  - per-window pooled feature h = mean_t H(t)            [B, D]
  - valence direction d = mean(h|val=+1) - mean(h|val=-1) on the TRAIN split only
    (parameter-free; no gradient, no learned probe)
  - z_affect(t) = <H(:,t) - mu, d/|d|>  per timestep     [B, T]   (time-resolved)
  - window score s = mean_t z_affect(t)  -> binary valence AUC vs chance 0.5

Split is SUBJECT/TRIAL-independent: whole (subject,session,trial) groups go to
train xor test, so no window leaks across the split (Doc III 1.2).

Falsification hooks (Doc III Stage-2 battery):
  - shuffle_labels=True : permute valence labels -> AUC should collapse to ~0.5
  - shuffle_time=True   : shuffle H(t) along time before scoring -> trajectory /
                          timing structure destroyed (tests time-resolution claim)
"""
import numpy as np, torch
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def extract_H(model, loader, device):
    """Returns H_all [N,D,T], and arrays: valence, subject, trial, session, group."""
    model.eval()
    Hs, val, subj, trial, sess = [], [], [], [], []
    for batch in loader:
        x = batch['x'].to(device); s = batch['subject_id'].to(device)
        pos = batch['positions'].to(device)
        H = model.encode(x, s, pos)                 # [B,D,T]
        Hs.append(H.cpu()); val.append(batch['valence'])
        subj.append(batch['subject_id']); trial.append(batch['trial_id'])
        sess.append(batch['session_id'])
    H_all = torch.cat(Hs).numpy()
    val = torch.cat(val).numpy(); subj=torch.cat(subj).numpy()
    trial=torch.cat(trial).numpy(); sess=torch.cat(sess).numpy()
    # group id per (subject,session,trial)
    group = np.array([f"{a}_{b}_{c}" for a,b,c in zip(subj,sess,trial)])
    return H_all, val, subj, trial, sess, group


def valence_readout(H_all, valence, group, seed=0, test_frac=0.4,
                    shuffle_labels=False, shuffle_time=False):
    """Mean-difference direction + AUC under a group-disjoint split.
    Only binary valence windows (+1 vs -1); neutral (0) excluded from direction/AUC."""
    rng = np.random.default_rng(seed)
    binm = valence != 0
    H = H_all[binm]; y = (valence[binm] > 0).astype(int); g = group[binm]  # 1=positive,0=negative

    if shuffle_labels:
        y = rng.permutation(y)
    if shuffle_time:
        # destroy temporal order of H(t) independently per window
        H = H.copy()
        for i in range(H.shape[0]):
            H[i] = H[i][:, rng.permutation(H.shape[2])]

    # group-disjoint split
    ug = np.unique(g); rng.shuffle(ug)
    n_test = max(1, int(len(ug)*test_frac))
    test_groups = set(ug[:n_test].tolist())
    te = np.array([gi in test_groups for gi in g]); tr = ~te
    if tr.sum()==0 or te.sum()==0 or len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2:
        return None  # degenerate split

    # pooled per-window feature
    h = H.mean(axis=2)                                   # [N,D]
    mu = h[tr].mean(axis=0, keepdims=True)
    # mean-difference direction on TRAIN only (parameter-free)
    d = h[tr][y[tr]==1].mean(0) - h[tr][y[tr]==0].mean(0)   # [D]
    dn = d / (np.linalg.norm(d)+1e-8)

    # z_affect(t) time-resolved, test windows: <H(:,t)-mu, dn>
    z_t = np.einsum('ndt,d->nt', H[te]-mu[...,None], dn)  # [N_te,T]
    score = z_t.mean(axis=1)                              # window score
    auc = roc_auc_score(y[te], score)
    return {
        'auc': float(auc),
        'n_train': int(tr.sum()), 'n_test': int(te.sum()),
        'n_train_groups': int(len(ug)-n_test), 'n_test_groups': int(n_test),
        'direction_norm': float(np.linalg.norm(d)),
        'direction_unit': dn,        # [D] unit valence direction (train-only, parameter-free)
        'z_t_test': z_t,             # [N_te, T] trajectories
        'y_test': y[te], 'score': score,
    }
