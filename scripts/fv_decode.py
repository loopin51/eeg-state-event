"""F-V (D-111 decision 1, D-112) inference CLI: one preprocessed EEG epoch (or a stack of repetitions) -> concept names.

Input  : --eeg  .npy (np.load with allow_pickle=False), whitened epochs in the F-V cache format (63 channels in
         data/cache/things_eeg2/channels.json order, 250 Hz): [63, 250] or [r, 63, 250] (the 0..1.0 s model input), or the
         stored -0.2..1.0 s grid [.., 63, 300] (the first 50 samples are dropped). A stack is averaged over r first.
         --subject S and one or more fv_train.py run folders (--runs; several seeds of one condition). Layer --layer
         (k0 = training-subject mean, ses1 / ses12 / ses123 = D-090 calibration). A calibrated layer needs run target == S.
         k0 needs S outside the run's trainers (cross-fitted), unless --allow-seen (then the output says so).
Output : JSON (stdout or --out) with
         (a) test200: top-5 of the 200 test concept names (configs/fv_names.tsv split == test) with cosine scores,
         (b) all1854: top-5 over all 1,854 THINGS names, and whether that top-1 is a training concept,
         (c) sentence: 'I see a/an {display_name}.' for the test200 top-1 (fixed template with an a/an spelling rule, display only),
         (d) with --tts out.wav: Kokoro-82M 0.9.4 rendering of the sentence (scripts/fv_say.py in env_tts, configs/t1_tts.yaml).
Scores : estimates of every seed L2-normalised, averaged, normalised again; cosine to L2-normalised name embeddings; no
         temperature (same as the F-V verdict, ese.things.models.scores).
Seal   : refuses any input under a raw-eeg folder, any file named raw_eeg_test*, and any file whose sha256 is in the seal
         table (results/things_eeg2/seal_test_sha256.tsv). The sealed test runs are decoded only inside the verdict job.

usage: python scripts/fv_decode.py --runs RUN [RUN ...] --subject S --eeg x.npy [--layer ses123] [--names-file F]
                                   [--out out.json] [--tts out.wav --voice af_heart] [--device cpu|cuda]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import argparse, csv, hashlib, json, subprocess, sys
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def sentence(name):
    """display sentence 'I see a/an {display_name}.' (D-111 decision 1, display only, never part of a verdict).
    Article rule (coordination session 2026-10-17): 'an' when the display name starts with a, e, i, o or u (case-insensitive),
    otherwise 'a'. Spelling only: exceptions by sound (hour, honest, unicorn, one, ...) and uncountable or plural names are left
    as the rule gives them."""
    return 'I see %s %s.' % ('an' if name[:1].lower() in 'aeiou' else 'a', name)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def seal_guard(path, seal_table):
    real = os.path.realpath(path); parts = real.split(os.sep)
    if 'raw-eeg' in parts or os.path.basename(real).startswith('raw_eeg_test'):
        raise SystemExit(f'refused: {path} is a raw THINGS-EEG2 file (the CLI takes preprocessed epochs only)')
    if os.path.exists(seal_table):
        sealed = {r['sha256'] for r in csv.DictReader(open(seal_table), delimiter='\t')}
        if sha256(real) in sealed: raise SystemExit(f'refused: {path} matches a sealed test file (seal table {seal_table})')


def load_eeg(path, n_ch):
    x = np.load(path, allow_pickle=False).astype(np.float32)
    if x.ndim == 2: x = x[None]
    assert x.ndim == 3 and x.shape[1] == n_ch, f'expected [r, {n_ch}, 250|300] or [{n_ch}, 250|300], got {x.shape}'
    if x.shape[2] == 300: x = x[..., 50:]                              # stored grid -0.2..1.0 s -> model input 0..1.0 s
    assert x.shape[2] == 250, f'expected 250 time samples (0..1.0 s at 250 Hz), got {x.shape[2]}'
    assert np.isfinite(x).all(), 'non-finite values in the input'
    return x.mean(0), x.shape[0]


def load_runs(runs, subject, layer, allow_seen):
    import torch
    from ese.things.models import build
    cfgs = [yaml.safe_load(open(os.path.join(r, 'config.yaml'))) for r in runs]
    c0 = cfgs[0]
    for c in cfgs[1:]:
        assert c['model'] == c0['model'] and c['run']['candidate'] == c0['run']['candidate'], 'runs differ in model or candidate'
        assert c['run']['target'] == c0['run']['target'] and c['run']['trainers'] == c0['run']['trainers'], 'runs differ in fold'
    run = c0['run']; trainers = list(run['trainers']); ti = len(trainers); seen = subject in trainers
    if layer != 'k0': assert run['target'] == subject, f"layer {layer} is the calibrated layer of subject {run['target']}, not {subject}"
    elif seen and not allow_seen: raise SystemExit(f'subject {subject} is a trainer of these runs (not cross-fitted); use the fold with target {subject} or --allow-seen')
    nets = []
    for r, c in zip(runs, cfgs):
        net = build(run['candidate'], ti + 1, c['model'])
        net.load_state_dict(torch.load(os.path.join(r, 'shared.pt'), map_location='cpu', weights_only=True)['net'])
        layers = torch.load(os.path.join(r, 'layers.pt'), map_location='cpu', weights_only=True)
        with torch.no_grad(): net.subject.weights[ti] = layers[layer]
        nets.append(net.eval())
    return c0, nets, ti, seen


def names_file(c, explicit):
    if explicit: return P(explicit)
    td = P(c['targets']['dir']); ip = os.path.join(td, 'index.json')
    idx = json.load(open(ip)) if os.path.exists(ip) else {}
    g = idx.get('g0_pass_prompt')                                        # 'names_text_P1' or 'P1' once G0 has fixed the prompt version
    if not g: raise SystemExit('the G0 prompt version is not fixed yet (index.json g0_pass_prompt is null); pass --names-file')
    key = g if g in idx else f'names_text_{g}'
    return os.path.join(td, idx[key]['file'])


def topk(S, rows, k=5):
    o = np.argsort(-S, kind='stable')[:k]
    return [dict(rank=i + 1, display_name=rows[j]['display_name'], folder=rows[j]['folder'], concept=rows[j]['concept'],
                 split=rows[j]['split'], score=round(float(S[j]), 6)) for i, j in enumerate(o)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--subject', type=int, required=True)
    ap.add_argument('--eeg', required=True); ap.add_argument('--layer', default='ses123', choices=['k0', 'ses1', 'ses12', 'ses123'])
    ap.add_argument('--names-file'); ap.add_argument('--names-table'); ap.add_argument('--positions'); ap.add_argument('--seal-table')
    ap.add_argument('--allow-seen', action='store_true'); ap.add_argument('--out'); ap.add_argument('--device', default='cpu')
    ap.add_argument('--tts'); ap.add_argument('--voice'); ap.add_argument('--tts-python', default=os.path.join(ROOT, 'env_tts', 'bin', 'python'))
    ap.add_argument('--tts-config', default='configs/t1_tts.yaml'); a = ap.parse_args()
    prep = yaml.safe_load(open(P('configs/fv_preprocess.yaml')))
    seal_guard(a.eeg, P(a.seal_table or prep['seal_table']))
    import torch
    from ese.things.models import scores
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    c, nets, ti, seen = load_runs([P(r) for r in a.runs], a.subject, a.layer, a.allow_seen)
    pos = np.load(P(a.positions or c['positions'])).astype(np.float32); n_ch = pos.shape[0]
    x, n_rep = load_eeg(a.eeg, n_ch)
    rows = list(csv.DictReader(open(P(a.names_table or c['targets']['names_table'])), delimiter='\t'))
    nf = names_file(c, a.names_file); E = np.load(nf, allow_pickle=False).astype(np.float32)
    assert E.shape == (len(rows), c['model']['out_dim']), f'name embeddings {E.shape} do not match {len(rows)} names x {c["model"]["out_dim"]}'
    dev = torch.device(a.device)
    xb = torch.from_numpy(x)[None].to(dev); pb = torch.from_numpy(pos)[None].to(dev); sb = torch.full((1,), ti, dtype=torch.long, device=dev)
    est = []
    with torch.no_grad():
        for net in nets:
            e = net.to(dev)(xb, pb, sb).float(); est.append(torch.nn.functional.normalize(e, dim=-1))
        e = torch.stack(est).mean(0)                                           # seed mean of normalised estimates (scores renormalises)
        S = scores(e, torch.from_numpy(E).to(dev))[0].cpu().numpy()
    test = [i for i, r in enumerate(rows) if r['split'] == 'test']
    assert len(test) == 200, len(test)
    t200 = topk(S[test], [rows[i] for i in test]); a1854 = topk(S, rows)
    res = dict(subject=a.subject, layer=a.layer, cross_fitted=not seen, runs=[os.path.relpath(P(r), ROOT) for r in a.runs],
               candidate=c['run']['candidate'], fold_target=c['run']['target'], n_seeds=len(nets), n_repetitions_averaged=n_rep,
               eeg=os.path.abspath(a.eeg), eeg_sha256=sha256(a.eeg), names_file=os.path.relpath(nf, ROOT), names_sha256=sha256(nf),
               test200=t200, all1854=a1854, all1854_top1_is_training_concept=a1854[0]['split'] == 'train',
               sentence=sentence(t200[0]['display_name']),
               note='closed 200-name candidate ranking; the sentence and audio are display-only post-processing (D-111 decision 1)')
    if a.tts:
        cmd = [a.tts_python, os.path.join(ROOT, 'scripts', 'fv_say.py'), P(a.tts_config), '--text', res['sentence'], '--out', os.path.abspath(a.tts)]
        if a.voice: cmd += ['--voice', a.voice]
        res['tts'] = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1])
    s = json.dumps(res, indent=1, ensure_ascii=False)
    if a.out: open(a.out, 'w').write(s + '\n')
    print(s)


if __name__ == '__main__':
    main()
