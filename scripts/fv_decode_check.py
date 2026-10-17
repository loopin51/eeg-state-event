"""Check of scripts/fv_decode.py with the frozen F-V final runs (D-111 decision 1, completion item "frozen checkpoint and
inference CLI", coordination session 2026-10-17). No sealed test input: the inputs are training-run epochs of session 4.

Inputs (target 1, session 4 training runs, development concepts of configs/fv_train.yaml dev):
  concept means  : the first --n-concepts rows of X63 in data/cache/gao_armc/things_t01.npz (fv_gao_c_prep.py --part things,
                   development concepts present in session 4, [63, 250] each),
  one raw stack  : every session 4 training epoch of the first of those concepts, [r, 63, 300] float16 from the cache
                   (tests the stored -0.2..1.0 s grid and the repetition average of the CLI).
For each input the CLI runs as a subprocess (wall time, peak RSS of the child). An independent path then recomputes the scores:
fv_train.py networks with layers.pt[layer] in the target row, the seed mean of ese.things.verdict.seed_mean (the verdict
function: mean of L2-normalised seed outputs, normalised again, float64) and the cosine to the L2-normalised name embeddings.
The CLI top-5 (test200 and all1854) must equal the independent top-5 and the scores must agree within --tol.
The outputs are names of training concepts seen by the final models (development concepts are in the final shared training,
D-112 amendment 1), so no accuracy is computed or reported here.

usage: env/bin/python -I scripts/fv_decode_check.py --runs RUN RUN RUN --subject 1 [--layer ses123] [--n-concepts 5] [--device cpu]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import argparse, csv, json, resource, subprocess, sys, time
import numpy as np, torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
from ese.things.data import Meta
from ese.things.models import build
from ese.things import verdict as V


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--subject', type=int, required=True)
    ap.add_argument('--layer', default='ses123'); ap.add_argument('--n-concepts', type=int, default=5); ap.add_argument('--device', default='cpu')
    ap.add_argument('--things', default='data/cache/gao_armc/things_t{t:02d}.npz'); ap.add_argument('--session', type=int, default=4)
    ap.add_argument('--tol', type=float, default=1e-4); a = ap.parse_args()
    t0 = time.time(); runs = [P(r) for r in a.runs]; cfgs = [yaml.safe_load(open(os.path.join(r, 'config.yaml'))) for r in runs]; c = cfgs[0]
    assert all(x['run']['target'] == a.subject and x['run']['mode'] == 'final' and not x['run']['shuffle'] and not x['run'].get('smoke') for x in cfgs)
    name = time.strftime('%Y-%m-%d') + f'_fv_decode_check_t{a.subject:02d}_{a.layer}_{a.device}'; out = os.path.join(P('results/runs'), name)
    inp = os.path.join(P('data/cache/fv_decode_check'), name); os.makedirs(out, exist_ok=True); os.makedirs(inp, exist_ok=True)
    meta = Meta(P(c['meta'])); pc = yaml.safe_load(open(P('configs/fv_preprocess.yaml'))); cache = P(pc['cache_dir'])

    # inputs (no sealed file: session 4 training runs only)
    z = np.load(P(a.things.format(t=a.subject)), allow_pickle=False); assert int(z['session']) == a.session and int(z['target']) == a.subject
    items = []
    for i in range(min(a.n_concepts, len(z['concepts']))):
        f = os.path.join(inp, f'concept{i}.npy'); np.save(f, z['X63'][i]); items.append(dict(kind='concept_mean', file=f, concept=meta.train_concepts[int(z['concepts'][i])]))
    c0 = meta.train_concepts[int(z['concepts'][0])]; ev = np.load(os.path.join(cache, f'sub-{a.subject:02d}', f'ses-{a.session:02d}_training_events.npz'))
    rows_ep = np.flatnonzero(np.isin(ev['code'], meta.codes_of([c0]))); Ep = np.load(os.path.join(cache, f'sub-{a.subject:02d}', f'ses-{a.session:02d}_training.npy'), mmap_mode='r')
    f = os.path.join(inp, 'stack0.npy'); np.save(f, np.asarray(Ep[rows_ep])); items.append(dict(kind='epoch_stack', file=f, concept=c0, n_epochs=int(len(rows_ep))))

    # CLI runs
    py = sys.executable; res = []
    for it in items:
        o = os.path.join(out, 'cli_' + os.path.basename(it['file'])[:-4] + '.json'); cmd = [py, '-I', P('scripts/fv_decode.py'), '--runs', *runs, '--subject', str(a.subject),
                                                                                         '--layer', a.layer, '--eeg', it['file'], '--out', o, '--device', a.device]
        s = time.time(); p = subprocess.run(cmd, capture_output=True, text=True); it.update(returncode=p.returncode, seconds=round(time.time() - s, 2))
        if p.returncode: it['stderr'] = p.stderr[-2000:]; res.append(it); continue
        it['cli'] = json.load(open(o)); res.append(it)
    peak_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6

    # independent path (verdict seed mean)
    trainers = list(c['run']['trainers']); ti = len(trainers); dev = torch.device(a.device); pos = torch.from_numpy(np.load(P(c['positions'])).astype(np.float32)).to(dev)
    nets = []
    for r, x in zip(runs, cfgs):
        net = build(x['run']['candidate'], ti + 1, x['model']); net.load_state_dict(torch.load(os.path.join(r, 'shared.pt'), map_location='cpu', weights_only=True)['net'])
        L = torch.load(os.path.join(r, 'layers.pt'), map_location='cpu', weights_only=True)
        with torch.no_grad(): net.subject.weights[ti] = L[a.layer]
        nets.append(net.to(dev).eval())
    names = list(csv.DictReader(open(P(c['targets']['names_table'])), delimiter='\t')); test = [i for i, r in enumerate(names) if r['split'] == 'test']
    T = V.unit(np.load(os.path.join(P(c['targets']['dir']), c['targets']['names_text'])).astype(np.float64))
    checks = []
    for it in res:
        if 'cli' not in it: checks.append(dict(file=os.path.basename(it['file']), ok=False, reason='cli failed')); continue
        x = np.load(it['file']).astype(np.float32); x = x[None] if x.ndim == 2 else x; x = x[..., 50:] if x.shape[2] == 300 else x; x = x.mean(0)
        with torch.no_grad():
            Es = [net(torch.from_numpy(x)[None].to(dev), pos[None], torch.full((1,), ti, dtype=torch.long, device=dev)).float().cpu().numpy() for net in nets]
        S = (V.seed_mean(Es) @ T.T)[0]; cli = it['cli']; row = dict(file=os.path.basename(it['file']), kind=it['kind'], concept=it['concept'])
        for key, idx in (('all1854', np.arange(len(names))), ('test200', np.array(test))):
            o = idx[np.argsort(-S[idx], kind='stable')[:5]]; ref = [names[j]['folder'] for j in o]; got = [r['folder'] for r in cli[key]]
            row[key] = dict(same_top5=ref == got, max_abs_score_diff=float(np.max(np.abs(S[o] - np.array([r['score'] for r in cli[key]])))))
        row['names_file_matches'] = cli['names_file'].endswith(c['targets']['names_text']); row['n_seeds'] = cli['n_seeds']; row['n_repetitions_averaged'] = cli['n_repetitions_averaged']
        row['ok'] = all(row[k]['same_top5'] and row[k]['max_abs_score_diff'] < a.tol for k in ('all1854', 'test200')) and row['names_file_matches'] and cli['n_seeds'] == len(runs)
        checks.append(row)
    m = dict(subject=a.subject, layer=a.layer, device=a.device, runs=[os.path.relpath(r, ROOT) for r in runs], session=a.session,
             inputs=[{k: v for k, v in it.items() if k not in ('cli', 'file')} | dict(file=os.path.relpath(it['file'], ROOT)) for it in res],
             checks=checks, all_ok=all(x['ok'] for x in checks), cli_seconds=[it['seconds'] for it in res], cli_peak_rss_gb=round(peak_child, 3),
             self_peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3), seconds=round(time.time() - t0, 1),
             note='training concepts of the final shared training; output names are not an accuracy measure')
    yaml.safe_dump(dict(args=vars(a)), open(os.path.join(out, 'config.yaml'), 'w')); json.dump(m, open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in m.items() if k != 'inputs'}, indent=1), flush=True)
    sys.exit(0 if m['all_ok'] else 1)


if __name__ == '__main__':
    main()
