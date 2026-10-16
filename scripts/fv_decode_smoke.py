"""Smoke test of scripts/fv_decode.py with fake checkpoints, fake name embeddings and Gaussian input (no EEG data, no sealed file).

Fake runs follow the fv_train.py output format (eeg-b1, 2026-10-17): config.yaml = configs/fv_train.yaml + run {candidate,
trainers, target}, shared.pt = {'net': state_dict, 'logit_v'}, layers.pt = {'k0', 'ses1', 'ses12', 'ses123'}.
Checks: output structure, seed averaging against an independent forward pass, fold / cross-fitting rules, [63, 300] input,
seal refusals (path and sha256), and optionally the Kokoro rendering (--tts).

usage: env/bin/python scripts/fv_decode_smoke.py OUT_DIR [--tts]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import argparse, copy, csv, hashlib, json, subprocess, sys
import numpy as np, torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.things.models import build

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--tts', action='store_true'); a = ap.parse_args()
out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
base = yaml.safe_load(open(os.path.join(ROOT, 'configs/fv_train.yaml')))
trainers = list(range(2, 11)); ti = len(trainers)


def fake_run(name, cand, seed):
    torch.manual_seed(seed); d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
    c = copy.deepcopy(base); c['run'] = dict(candidate=cand, trainers=trainers, target=1, seed=seed, mode='final', fake=True)
    yaml.safe_dump(c, open(os.path.join(d, 'config.yaml'), 'w'))
    net = build(cand, ti + 1, c['model']); W = net.subject.weights.detach()
    torch.save(dict(net=net.state_dict(), logit_v=2.66), os.path.join(d, 'shared.pt'))
    k0 = W[:ti].mean(0); g = torch.Generator().manual_seed(seed)
    torch.save({'k0': k0, **{k: k0 + 0.01 * torch.randn(k0.shape, generator=g) for k in ('ses1', 'ses12', 'ses123')}}, os.path.join(d, 'layers.pt'))
    return d


runs_bm = [fake_run('run_bm_s1', 'bm', 1), fake_run('run_bm_s2', 'bm', 2)]
run_nice = fake_run('run_nice_s1', 'nice', 3)
rows = list(csv.DictReader(open(os.path.join(ROOT, base['targets']['names_table'])), delimiter='\t'))
E = np.random.default_rng(0).standard_normal((len(rows), base['model']['out_dim'])).astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True); nf = os.path.join(out, 'names_fake.npy'); np.save(nf, E)
X = np.random.default_rng(1).standard_normal((20, 63, 250)).astype(np.float32); xf = os.path.join(out, 'eeg_fake_r20.npy'); np.save(xf, X)
X300 = np.random.default_rng(2).standard_normal((63, 300)).astype(np.float32); x3 = os.path.join(out, 'eeg_fake_300.npy'); np.save(x3, X300)
seal = os.path.join(out, 'seal_fake.tsv')
open(seal, 'w').write('path\tbytes\tsha256\nfake\t0\t%s\n' % hashlib.sha256(open(x3, 'rb').read()).hexdigest())
bad_name = os.path.join(out, 'raw_eeg_test_copy.npy'); np.save(bad_name, X[0])
CLI = [sys.executable, os.path.join(ROOT, 'scripts', 'fv_decode.py')]


def run(args, ok=True):
    p = subprocess.run(CLI + args, capture_output=True, text=True)
    if ok:
        assert p.returncode == 0, p.stderr[-2000:]
        return json.loads(p.stdout)
    assert p.returncode != 0, 'expected a refusal: ' + p.stdout[-500:]
    return (p.stderr.strip().splitlines() or [''])[-1]


report = {}
sys.path.insert(0, os.path.join(ROOT, 'scripts')); from fv_decode import sentence
report['article_rule'] = {n: sentence(n) for n in ('ostrich', 'dog', 'Umbrella', 'egg', 'hour', 'unicorn')}
assert report['article_rule']['ostrich'] == 'I see an ostrich.' and report['article_rule']['dog'] == 'I see a dog.'
# 1. two bm seeds, calibrated layer, stack of 20 repetitions
r1 = run(['--runs', *runs_bm, '--subject', '1', '--eeg', xf, '--layer', 'ses123', '--names-file', nf, '--out', os.path.join(out, 'decode_bm.json')])
assert len(r1['test200']) == 5 and all(t['split'] == 'test' for t in r1['test200']) and len(r1['all1854']) == 5
assert r1['n_seeds'] == 2 and r1['n_repetitions_averaged'] == 20 and r1['cross_fitted'] is True
nm = r1['test200'][0]['display_name']; assert r1['sentence'] == 'I see %s %s.' % ('an' if nm[0].lower() in 'aeiou' else 'a', nm)
sc = [t['score'] for t in r1['test200']]; assert sc == sorted(sc, reverse=True)
assert r1['all1854_top1_is_training_concept'] == (r1['all1854'][0]['split'] == 'train')
# independent forward pass: same seed averaging and cosine
pos = np.load(os.path.join(ROOT, base['positions'])).astype(np.float32); est = []
for d in runs_bm:
    c = yaml.safe_load(open(os.path.join(d, 'config.yaml'))); net = build('bm', ti + 1, c['model'])
    net.load_state_dict(torch.load(os.path.join(d, 'shared.pt'), weights_only=True)['net']); L = torch.load(os.path.join(d, 'layers.pt'), weights_only=True)
    with torch.no_grad():
        net.subject.weights[ti] = L['ses123']; net.eval()
        e = net(torch.from_numpy(X.mean(0))[None], torch.from_numpy(pos)[None], torch.full((1,), ti)); est.append(torch.nn.functional.normalize(e, dim=-1))
e = torch.nn.functional.normalize(torch.stack(est).mean(0), dim=-1)[0].numpy(); S = E @ e
test = [i for i, r in enumerate(rows) if r['split'] == 'test']; top = [test[j] for j in np.argsort(-S[test])[:5]]
assert [rows[j]['folder'] for j in top] == [t['folder'] for t in r1['test200']], 'CLI ranking differs from the independent forward pass'
assert np.allclose([S[j] for j in top], sc, atol=1e-5)
report['bm_two_seeds'] = dict(top1=r1['test200'][0], sentence=r1['sentence'], independent_check='same top-5 and scores')
# 2. nice candidate, [63, 300] input (stored grid), k0 for the fold target
r2 = run(['--runs', run_nice, '--subject', '1', '--eeg', x3, '--layer', 'k0', '--names-file', nf, '--seal-table', os.path.join(out, 'no_seal.tsv')])
assert r2['candidate'] == 'nice' and r2['n_repetitions_averaged'] == 1
report['nice_300_k0'] = dict(top1=r2['test200'][0]['display_name'])
# 3. fold and cross-fitting rules
report['refuse_calibrated_other_subject'] = run(['--runs', *runs_bm, '--subject', '2', '--eeg', xf, '--layer', 'ses123', '--names-file', nf], ok=False)
report['refuse_k0_seen_subject'] = run(['--runs', *runs_bm, '--subject', '2', '--eeg', xf, '--layer', 'k0', '--names-file', nf], ok=False)
r3 = run(['--runs', *runs_bm, '--subject', '2', '--eeg', xf, '--layer', 'k0', '--names-file', nf, '--allow-seen'])
assert r3['cross_fitted'] is False; report['allow_seen_flag'] = r3['cross_fitted']
report['k0_unseen_subject_11'] = run(['--runs', *runs_bm, '--subject', '11', '--eeg', xf, '--layer', 'k0', '--names-file', nf])['cross_fitted']
report['refuse_mixed_folds'] = run(['--runs', runs_bm[0], run_nice, '--subject', '1', '--eeg', xf, '--names-file', nf], ok=False)
# 4. seal refusals
report['refuse_sealed_sha'] = run(['--runs', *runs_bm, '--subject', '1', '--eeg', x3, '--names-file', nf, '--seal-table', seal], ok=False)
report['refuse_raw_test_name'] = run(['--runs', *runs_bm, '--subject', '1', '--eeg', bad_name, '--names-file', nf], ok=False)
report['refuse_bad_shape'] = run(['--runs', *runs_bm, '--subject', '1', '--eeg', nf, '--names-file', nf], ok=False)
# 5. display audio
if a.tts:
    r5 = run(['--runs', *runs_bm, '--subject', '1', '--eeg', xf, '--names-file', nf, '--tts', os.path.join(out, 'sentence.wav')])
    assert r5['tts']['duration_s'] > 0.3; report['tts'] = r5['tts']
json.dump(report, open(os.path.join(out, 'smoke_report.json'), 'w'), indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False)); print('SMOKE OK')
