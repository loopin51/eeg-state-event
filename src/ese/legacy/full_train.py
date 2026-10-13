"""FULL TRAINING run for the affect encoder Stage-1 (self-supervised masked-time
reconstruction) + Stage-2 (frozen mean-difference valence read-out), over ALL
subjects of every dataset registered in affect_enc/registry.py.

Extensibility: to add a new dataset once it finishes downloading, register it
in affect_enc/registry.py (see the docstring there) and add its name to
--datasets on the command line. No other code in this file needs to change --
the combined DataLoader, the held-out validation split, the SubjectLayers
embedding size, and the per-dataset Stage-2 readout all derive from the
registry automatically.

Held-out validation: for each active dataset, a subject-disjoint slice (whole
subjects, not windows) is held out at --val_frac and used ONLY to track
val_loss for the cosine LR schedule's convergence check and early stopping --
never for gradient updates. This is separate from Stage-2's own group-disjoint
train/test split, which is computed independently after Stage-1 finishes.
"""
import os, json, time, math, argparse, numpy as np, torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader, ConcatDataset, Subset
from affect_enc.registry import DATASET_REGISTRY, total_subjects
from affect_enc.deap_data import multi_collate
from affect_enc.model import AffectPretrainer, masked_recon_loss
from affect_enc.readout import extract_H, valence_readout


def group_disjoint_subject_split(all_subjects, val_frac, seed):
    rng = np.random.default_rng(seed)
    subs = list(all_subjects)
    rng.shuffle(subs)
    n_val = max(1, round(len(subs) * val_frac))
    return sorted(subs[n_val:]), sorted(subs[:n_val])  # train_subs, val_subs


def build_all(active_names, val_frac=0.1, seed=0):
    """Builds ONE dataset object per registry entry (subjects=None -> all), then
    slices train/val via torch Subset on subject id. Building one object per
    source (rather than separate train/val constructions of the same io_path)
    avoids torcheeg's LMDB 'environment already open in this process' error."""
    train_parts, val_parts, named = [], [], {}
    for nm in active_names:
        e = DATASET_REGISTRY[nm]
        full = e['build'](None)
        train_subs, val_subs = group_disjoint_subject_split(e['all_subjects'], val_frac, seed)
        subj_of_idx = full.info.loc[full.idx, 'subj_int'].to_numpy()
        train_local = np.where(np.isin(subj_of_idx, train_subs))[0]
        val_local = np.where(np.isin(subj_of_idx, val_subs))[0]
        train_parts.append(Subset(full, train_local))
        val_parts.append(Subset(full, val_local))
        named[nm] = dict(full=full, train_subs=train_subs, val_subs=val_subs,
                          n_train=int(len(train_local)), n_val=int(len(val_local)))
    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    return train_ds, val_ds, named


@torch.no_grad()
def eval_val_loss(model, loader, device, mask_ratio, span, max_batches=None):
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        x = batch['x'].to(device); s = batch['subject_id'].to(device); pos = batch['positions'].to(device)
        loss, _ = masked_recon_loss(model, x, s, pos, mask_ratio=mask_ratio, span=span)
        losses.append(loss.item())
        if max_batches and i + 1 >= max_batches:
            break
    model.train()
    return float(np.mean(losses)) if losses else float('nan')


def cosine_lr(step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = min(1.0, (step - warmup_steps) / max(1, (total_steps - warmup_steps)))
    return 0.5 * base_lr * (1 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='seediv,deap', help='comma-separated registry keys')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup_frac', type=float, default=0.05)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--patience', type=int, default=8, help='#val checks w/o improvement before early stop')
    ap.add_argument('--val_every_steps', type=int, default=200)
    ap.add_argument('--ckpt_every_steps', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.expanduser('~/affect_data/full_run'))
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    active = args.datasets.split(',')
    n_sub = total_subjects(active)

    train_ds, val_ds, named = build_all(active, val_frac=args.val_frac, seed=args.seed)
    print(f"=== datasets={active} n_subjects_emb={n_sub} n_train={len(train_ds)} n_val={len(val_ds)} ===", flush=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=6, collate_fn=multi_collate, pin_memory=True,
                           persistent_workers=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=0, collate_fn=multi_collate)

    steps_per_epoch = len(train_dl)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_frac))
    print(f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} warmup_steps={warmup_steps}", flush=True)

    torch.manual_seed(args.seed)
    model = AffectPretrainer(n_channels=62, n_subjects=n_sub, out_dim=1024).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    best_val = float('inf'); best_step = 0; bad_evals = 0
    ckpt_path = os.path.join(args.out, 'stage1_backbone.pt')
    best_path = os.path.join(args.out, 'stage1_backbone_best.pt')
    step = 0; t0 = time.time(); stop = False; ep = 0
    for ep in range(args.epochs):
        for batch in train_dl:
            lr_now = cosine_lr(step, total_steps, args.lr, warmup_steps)
            for g in opt.param_groups:
                g['lr'] = lr_now
            x = batch['x'].to(dev); s = batch['subject_id'].to(dev); pos = batch['positions'].to(dev)
            loss, _ = masked_recon_loss(model, x, s, pos, mask_ratio=0.5, span=10)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            if step % args.val_every_steps == 0:
                vloss = eval_val_loss(model, val_dl, dev, 0.5, 10, max_batches=40)
                history.append({'step': step, 'epoch': ep, 'train_loss': float(loss.item()),
                                 'val_loss': vloss, 'lr': lr_now})
                print(f"[full] ep{ep} step{step} lr{lr_now:.2e} train {loss.item():.5f} val {vloss:.5f}", flush=True)
                if vloss < best_val - 1e-5:
                    best_val = vloss; best_step = step; bad_evals = 0
                    torch.save({'model_state': model.state_dict(), 'step': step, 'epoch': ep,
                                'val_loss': vloss}, best_path)
                else:
                    bad_evals += 1
                    if bad_evals >= args.patience:
                        print(f"[full] EARLY STOP at step{step} (no val improvement for "
                              f"{bad_evals} checks; best={best_val:.5f} @ step{best_step})", flush=True)
                        stop = True

            if args.ckpt_every_steps and step > 0 and step % args.ckpt_every_steps == 0:
                torch.save({'model_state': model.state_dict(), 'optim_state': opt.state_dict(),
                            'step': step, 'epoch': ep, 'history': history}, ckpt_path)
            step += 1
            if stop:
                break
        if stop:
            break
    torch.save({'model_state': model.state_dict(), 'optim_state': opt.state_dict(),
                'step': step, 'epoch': ep, 'history': history}, ckpt_path)
    train_seconds = time.time() - t0
    json.dump(history, open(os.path.join(args.out, 'full_history.json'), 'w'), indent=2)

    # ---- Stage-2: per-dataset frozen readout, using the BEST checkpoint ----
    m2 = AffectPretrainer(n_channels=62, n_subjects=n_sub, out_dim=1024).to(dev)
    best_ck_path = best_path if os.path.exists(best_path) else ckpt_path
    best_ck = torch.load(best_ck_path, map_location=dev)
    m2.load_state_dict(best_ck['model_state']); m2.eval()

    def eval_readout(dataset, name):
        dl = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=multi_collate, num_workers=0)
        H, val, subj, trial, sess, group = extract_H(m2, dl, dev)
        real = valence_readout(H, val, group, seed=0, test_frac=0.4)
        if real is None:
            return {'status': 'degenerate'}
        labs = [valence_readout(H, val, group, seed=k, shuffle_labels=True) for k in range(5)]
        tims = [valence_readout(H, val, group, seed=k, shuffle_time=True) for k in range(5)]
        out = {'auc': real['auc'], 'n_test': real['n_test'], 'n_train': real['n_train'],
               'z_shape': list(real['z_t_test'].shape),
               'label_shuffle_auc': float(np.mean([r['auc'] for r in labs if r])),
               'timestep_shuffle_auc': float(np.mean([r['auc'] for r in tims if r])),
               'direction_unit': real['direction_unit'].tolist()}
        np.savez(os.path.join(args.out, f'z_{name}.npz'), z_t=real['z_t_test'], y=real['y_test'], score=real['score'])
        return out

    stage2 = {}
    for nm in active:
        o = eval_readout(named[nm]['full'], nm)
        stage2[nm] = o
        print(f"[{nm}] AUC {o.get('auc','NA')} label-shuf {o.get('label_shuffle_auc','NA')} "
              f"timestep-shuf {o.get('timestep_shuffle_auc','NA')}", flush=True)

    cross_sim = None
    names_with_dir = [nm for nm in active if stage2[nm].get('direction_unit')]
    if len(names_with_dir) >= 2:
        d0 = np.array(stage2[names_with_dir[0]]['direction_unit'])
        d1 = np.array(stage2[names_with_dir[1]]['direction_unit'])
        cross_sim = float(np.dot(d0, d1) / (np.linalg.norm(d0) * np.linalg.norm(d1) + 1e-8))
        print(f"cross-dataset direction cosine similarity "
              f"({names_with_dir[0]} vs {names_with_dir[1]}) = {cross_sim:.4f}", flush=True)

    report = dict(
        datasets=active, n_subjects_emb=n_sub, n_train=len(train_ds), n_val=len(val_ds),
        epochs_run=ep + 1, total_steps_planned=total_steps, step_stopped=step,
        early_stopped=bool(stop), best_val_loss=best_val, best_step=best_step,
        train_seconds=round(train_seconds, 1), stage2=stage2,
        cross_dataset_direction_cosine=cross_sim,
        per_dataset_subjects={nm: {'train_subs': named[nm]['train_subs'], 'val_subs': named[nm]['val_subs'],
                                    'n_train_windows': named[nm]['n_train'], 'n_val_windows': named[nm]['n_val']}
                              for nm in active},
    )
    json.dump(report, open(os.path.join(args.out, 'full_report.json'), 'w'), indent=2, default=str)
    print("=== FULL REPORT ==="); print(json.dumps(report, indent=2, default=str)); print("FULL_DONE")


if __name__ == '__main__':
    main()
