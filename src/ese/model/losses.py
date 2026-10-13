"""Stage A losses (docs/04 §1-§5). All operate on per-context vectors unless noted."""
import torch, torch.nn.functional as F


def predict_loss(pred, target, mask):
    """pred/target [B,N_p,d_event], mask [B,N_p] bool -> Smooth-L1 on masked positions."""
    t = F.layer_norm(target, target.shape[-1:])
    return F.smooth_l1_loss(pred[mask], t[mask])


def _nce(sim, pos_mask, neg_mask):
    """Multi-positive InfoNCE. sim [B,B] (already /T). For each anchor with >=1 positive:
    -mean_p log( exp(s_ap) / (exp(s_ap) + sum_n exp(s_an)) ). Returns mean over anchors."""
    neg_exp = (sim.exp() * neg_mask).sum(1, keepdim=True)                        # [B,1]
    log_prob = sim - torch.log(sim.exp() + neg_exp)                               # [B,B]
    has = pos_mask.sum(1) > 0
    if has.sum() == 0: return sim.new_zeros(())
    per = -(log_prob * pos_mask).sum(1)[has] / pos_mask.sum(1)[has]
    return per.mean()


def coherence_loss(z, trial_key, pair_id, subject, T=0.1):
    """z [B,d] context-level state; positives = same pair_id (>=0); negatives = OTHER TRIALS OF THE SAME
    SUBJECT (D-018). With other-subject negatives the loss is solved by subject identity alone, which
    clusters z_state by subject and fights the cross-subject align loss."""
    zn = F.normalize(z, dim=-1); sim = zn @ zn.t() / T
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    pos = (pair_id[:, None] == pair_id[None, :]) & (pair_id[:, None] >= 0) & ~eye
    neg = (trial_key[:, None] != trial_key[None, :]) & (subject[:, None] == subject[None, :])
    return _nce(sim, pos.float(), neg.float())


def stim_align_loss(z, subject, clip_key, t_onset, dataset_id, T=0.1, pos_dt=2.0, neg_dt=10.0):
    """positives: other subject, same clip, |Δt|<=pos_dt; negatives: SAME dataset and (different clip
    or |Δt|>neg_dt). Same-dataset restriction for the same reason as in coherence_loss."""
    zn = F.normalize(z, dim=-1); sim = zn @ zn.t() / T
    same_clip = (clip_key[:, None] == clip_key[None, :]) & (clip_key[:, None] >= 0)
    dt = (t_onset[:, None] - t_onset[None, :]).abs()
    other = subject[:, None] != subject[None, :]
    same_ds = dataset_id[:, None] == dataset_id[None, :]
    pos = same_clip & other & (dt <= pos_dt)
    neg = ((~same_clip) | (dt > neg_dt)) & same_ds
    neg = neg & ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    return _nce(sim, pos.float(), neg.float())


def xcov_loss(ze, zs, mode='spectral'):
    """ze [B,de], zs [B,ds] -> cross-correlation penalty.
    'mean'     : mean squared entry of C (the original). Averaged over de*ds entries, a single shared
                 direction is invisible, so raising its weight 10x left the top canonical correlation at
                 0.99 (see reports/phase1_pilot_kr.md and D-023).
    'spectral' : squared LARGEST singular value of C, normalised by batch size — targets exactly the one
                 shared direction that the mean-squared version cannot see."""
    ze = ze.float(); zs = zs.float()          # linalg.matrix_norm has no bf16 kernel (autocast is on)
    ze = (ze - ze.mean(0)) / (ze.std(0) + 1e-6); zs = (zs - zs.mean(0)) / (zs.std(0) + 1e-6)
    C = (ze.t() @ zs).float() / (len(ze) - 1)      # matmul re-casts to bf16 under autocast; SVD needs fp32
    if mode == 'mean': return (C ** 2).mean()
    # Under independence the largest singular value of an empirical cross-correlation is already
    # ~(sqrt(de)+sqrt(ds))/sqrt(B) (Marchenko-Pastur scale), so penalise only the EXCESS over that floor.
    B, de, ds = len(ze), ze.shape[1], zs.shape[1]
    floor = (de ** 0.5 + ds ** 0.5) / B ** 0.5
    return F.relu(torch.linalg.matrix_norm(C, ord=2) / floor - 1.0) ** 2


def make_block_mask(B, N, ratio=0.5, min_len=15, max_len=40, device='cpu'):
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    target = int(N * ratio)
    for b in range(B):
        covered = 0; tries = 0
        while covered < target and tries < 50:
            L = int(torch.randint(min_len, max_len + 1, (1,)))
            s = int(torch.randint(0, max(1, N - L), (1,)))
            mask[b, s:s + L] = True; covered = int(mask[b].sum()); tries += 1
    return mask


def variance_loss(z, gamma=1.0):
    """VICReg-style anti-collapse hinge on the per-dimension std of z (flattened over all but last dim)."""
    z = z.reshape(-1, z.shape[-1]).float()
    std = torch.sqrt(z.var(0) + 1e-4)
    return F.relu(gamma - std).mean()
