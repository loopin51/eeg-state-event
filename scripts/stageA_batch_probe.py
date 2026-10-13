"""Stage A week-1 batch measurement: peak GPU memory and step time of the Défossez et al. (2023) brain module + CLIP loss
on random ds004408-shaped inputs (128 ch x 3 s at 120 Hz, 1024-d targets), per batch size and precision.

usage: python scripts/stageA_batch_probe.py configs/stageA_batch_probe.yaml
"""
import os, sys, json, time, yaml, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.model.brain_clip import BrainModule, clip_loss


def main(cfg_path):
    c = yaml.safe_load(open(cfg_path)); out = os.path.join(ROOT, c['out_dir']); os.makedirs(out, exist_ok=True)
    T = int(c['window_s'] * c['sample_rate']); dev = 'cuda'; res = []
    print(torch.cuda.get_device_name(0), 'free/total GB', [round(v / 2 ** 30, 2) for v in torch.cuda.mem_get_info()], flush=True)
    for prec in c['precisions']:
        for bs in c['batch_sizes']:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); r = dict(precision=prec, batch=bs)
            try:
                torch.manual_seed(0); net = BrainModule(out_dim=c['out_dim'], n_subjects=c['n_subjects'], n_in=c['n_channels']).to(dev)
                opt = torch.optim.Adam(net.parameters(), lr=c['lr'])
                x = torch.randn(bs, c['n_channels'], T, device=dev).clamp(-20, 20); pos = torch.rand(bs, c['n_channels'], 2, device=dev)
                y = torch.randn(bs, c['out_dim'], T, device=dev); s = torch.randint(0, c['n_subjects'], (bs,), device=dev)
                for i in range(3 + c['steps']):
                    if i == 3: torch.cuda.synchronize(); t0 = time.time()
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=prec == 'bf16'):
                        loss = clip_loss(net(x, pos, s).float(), y)
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                torch.cuda.synchronize()
                r.update(ok=True, step_s=(time.time() - t0) / c['steps'], peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                         n_params_m=sum(p.numel() for p in net.parameters()) / 1e6)
            except torch.cuda.OutOfMemoryError:
                r.update(ok=False, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            print(r, flush=True); res.append(r)
            for v in ('net', 'opt', 'x', 'y', 'pos', 's', 'loss'): locals().pop(v, None)
    # epoch cost at the paper setting (1,200 updates per epoch)
    for r in res:
        if r.get('ok'): r['epoch_min_at_1200_updates'] = r['step_s'] * 1200 / 60
    json.dump({'config': c, 'results': res}, open(os.path.join(out, 'metrics.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
