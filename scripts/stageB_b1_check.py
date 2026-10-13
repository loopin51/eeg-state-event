"""Stage B B1 (D-085 amendment): pipeline check on the first finished job, validation runs only (never the test runs).

Pass if, for every target subject, (a) the decoder output varies across validation windows (mean over output units of the
std across windows > check.min_output_std) and (b) the calibration history holds a finite validation CLIP loss.
Reads shared.pt, target_layers.pt, history.json and config.yaml of the run folder; does not open metrics.json or scores.npz.

usage: python scripts/stageB_b1_check.py configs/stageB_b1.yaml --subject-fold K --run-fold F [--smoke]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, glob, json, math, argparse, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageB_b1_train import RunData
from ese.model.brain_clip import BrainModule


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--subject-fold', type=int, required=True)
    ap.add_argument('--run-fold', type=int, required=True); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args(); c0 = yaml.safe_load(open(a.config))
    d = sorted(glob.glob(os.path.join(ROOT, c0['out_root'], f"*_stageB_b1_{c0['decoder_setting']}_k{a.subject_fold}_f{a.run_fold}{'_smoke' if a.smoke else ''}")))[-1]
    c = yaml.safe_load(open(os.path.join(d, 'config.yaml'))); r = c['run']; subj = r['trainers'] + r['targets']; nT = len(r['trainers'])
    D = RunData(c, subj, r['train_runs'], r['valid_runs'], r['valid_runs'], r['source']); D.to_device('cuda'); m = c['model']
    net = BrainModule(out_dim=D.F.shape[0], n_subjects=len(subj), merger_channels=m['merger_channels'], pos_dim=m['pos_dim'],
                      merger_dropout=m['merger_dropout'], initial_linear=m['initial_linear'], hidden=m['hidden'], depth=m['depth'],
                      subject_layers=True).cuda()
    net.load_state_dict(torch.load(os.path.join(d, 'shared.pt'))); net.eval()
    with torch.no_grad(): net.subject.weights[nT:] = torch.load(os.path.join(d, 'target_layers.pt')).cuda()
    hist = json.load(open(os.path.join(d, 'history.json'))); items = D.items['valid']
    sel = [items[i] for i in np.random.default_rng(0).choice(len(items), 512, replace=False)]
    res = {}
    for j, s in enumerate(r['targets']):
        si = nT + j; X, P, _ = D.batch(sel, np.full(len(sel), si))
        with torch.no_grad(): Z = net(X, P, torch.full((len(sel),), si, device='cuda')).float().flatten(1)
        cal = [h['valid'] for h in hist if h['stage'] == f'cal_{s}']
        res[str(s)] = dict(output_std=float(Z.std(0).mean()), cal_valid_finite=bool(cal) and all(math.isfinite(v) for v in cal), n_cal_entries=len(cal))
    ok = all(v['output_std'] > c0['check']['min_output_std'] and v['cal_valid_finite'] for v in res.values())
    out = dict(run=os.path.basename(d), split='valid', targets=res, passed=ok)
    p = os.path.join(ROOT, c0['out'], f"check_k{a.subject_fold}_f{a.run_fold}{'_smoke' if a.smoke else ''}.json"); os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w'), indent=1); print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
