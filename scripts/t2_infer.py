"""F-IV T2 score matrices (D-109 sections 4, 5, 7, 8): the stored B1 decoders (no training) scored against the matched TTS and
recorded candidate windows. Writes score matrices only; no MRP is computed or printed here (section 9 blinding).

Per (subject, run) cell: rows = the run's Q queries (matched set with 19 candidates, EEG window inside the run at +-0.25 s);
columns = rows plus every candidate of a row (other runs of the fold at L1 or above). Matrices: tts_fixed, rec_fixed
(lag 0) and, with lse0.5 among the rules, tts_lse0.5 and rec_lse0.5 (EEG windows moved by each lag and combined by
t2_lib.lse_rows, tau_i from the row's same-run Q columns). float32.

modes
  dev       : B1 k0_f0, the 13 training subjects' own layers, validation runs 5 and 13, matched_sets_dev (section 7)
  test      : every main job's targets (19 subjects) on its test runs (sections 9, 11)
  phaserand : the 5 phase-randomised jobs, their targets on their test runs (section 8, 2)
  --noise   : the same decoder on white-noise EEG (section 8, 1; section 7 gate (ii) in dev)
  --recorded-only : test mode, recorded candidates only (density audit at L1 or above, after the re-inference check; section 13)

usage: python scripts/t2_infer.py configs/t2.yaml --mode dev|test|phaserand [--noise] [--rules fixed lse0.5] [--jobs K_F ...]
       [--subjects N] [--sets PATH --attrs PATH --out DIR]   (subset and path options are for the code-path smoke only)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, hashlib, yaml, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from t2_lib import Job, TTS, scores, path, parse_key, load_sets, eeg_ok, lse_rows, noise_scale, est_noise


def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--mode', required=True, choices=['dev', 'test', 'phaserand'])
    ap.add_argument('--noise', action='store_true'); ap.add_argument('--recorded-only', action='store_true'); ap.add_argument('--rules', nargs='+'); ap.add_argument('--jobs', nargs='*')
    ap.add_argument('--subjects', type=int); ap.add_argument('--sets'); ap.add_argument('--attrs'); ap.add_argument('--out')
    a = ap.parse_args(); v = yaml.safe_load(open(path(a.config))); t0 = time.time(); al = v['align']
    if a.sets: v['matched_sets' + ('_dev' if a.mode == 'dev' else '')] = a.sets
    if a.attrs: v['match_attrs'] = a.attrs
    rules = a.rules or (['fixed', 'lse0.5'] if a.mode == 'dev' else ['fixed'] if a.recorded_only else sorted({'fixed', al['rule']}))
    assert all(r in ('fixed', 'lse0.5') for r in rules), f'alignment rule not frozen: {rules}'
    sets, attrs = load_sets(v, dev=a.mode == 'dev'); tts = TTS(v['tts_features']); lags = al['lags'] if 'lse0.5' in rules else [0.0]
    jobs = [tuple(map(int, j.split('_'))) for j in a.jobs] if a.jobs else ([(0, 0)] if a.mode == 'dev' else [(0, F) for F in range(5)] if a.mode == 'phaserand'
                                                                             else [(K, F) for K in range(5) for F in range(5)])
    assert not (a.recorded_only and (a.noise or a.mode != 'test')), '--recorded-only is the test-mode recorded path of the density audit (section 13)'
    tag = a.mode + ('_noise' if a.noise else '') + ('_rec' if a.recorded_only else ''); od = path(a.out or v['infer_out']); os.makedirs(od, exist_ok=True); summ = {}
    for K, F in jobs:
        J = Job(v['b1_config'], K, F, phaserand=a.mode == 'phaserand'); runs = J.valid_runs if a.mode == 'dev' else J.test_runs
        subs = J.trainers if a.mode == 'dev' else J.targets
        subs = subs[:a.subjects] if a.subjects else subs; arrays = {}; info = dict(folder=os.path.basename(J.folder), subjects=subs, runs=[r + 1 for r in runs], cells={})
        for r in runs:
            rows = sorted((k for k in sets if parse_key(k)[0] == r and eeg_ok(J, attrs[k][1], r, v['eeg_margin_s'])), key=lambda k: attrs[k][1])
            if not rows: continue
            cols = sorted(set(rows) | {c for k in rows for c in sets[k]['cands']}, key=lambda k: (attrs[k][0], attrs[k][1]))
            rs = set(rows); same = np.array([k in rs for k in cols])
            assert a.mode != 'dev' or all(parse_key(k)[0] in runs for k in cols), 'development candidates outside the validation runs'
            Yt = None if a.recorded_only else tts.windows(J, [(attrs[k][2], parse_key(k)[0]) for k in cols]); Yr = J.recorded_windows([(parse_key(k)[0], attrs[k][1]) for k in cols])
            items = [(r, attrs[k][1]) for k in rows]; arrays[f'r{r + 1}_rows'] = np.array(rows); arrays[f'r{r + 1}_cols'] = np.array(cols)
            for s in subs:
                sd = noise_scale(J, s, v['noise']['n_scale_windows']) if a.noise else None; St, Sr = [], []
                for lag in lags:
                    E = est_noise(J, items, s, lag, sd, v['noise']['seed'], r) if a.noise else J.est(items, s, lag)
                    St.append(None if Yt is None else scores(E, Yt)); Sr.append(scores(E, Yr)); del E
                Sr = np.stack(Sr).astype(np.float64); i0 = lags.index(0.0); arrays[f'rec_fixed_s{s}_r{r + 1}'] = Sr[i0].astype(np.float32)
                if Yt is not None: St = np.stack(St).astype(np.float64); arrays[f'tts_fixed_s{s}_r{r + 1}'] = St[i0].astype(np.float32)
                if 'lse0.5' in rules:
                    if Yt is not None: arrays[f'tts_lse0.5_s{s}_r{r + 1}'] = lse_rows(St, same, al['lse_c']).astype(np.float32)
                    arrays[f'rec_lse0.5_s{s}_r{r + 1}'] = lse_rows(Sr, same, al['lse_c']).astype(np.float32)
                info['cells'][f's{s}_r{r + 1}'] = dict(n_rows=len(rows), n_cols=len(cols))
                torch.cuda.empty_cache()
            del Yt, Yr; torch.cuda.empty_cache()
        out = os.path.join(od, f'{tag}_k{K}_f{F}.npz'); np.savez_compressed(out, **arrays)
        info.update(file=os.path.basename(out), sha256=sha(out)); summ[f'k{K}_f{F}'] = info
        print(f'k{K}_f{F}: {len(info["cells"])} cells written ({time.time() - t0:.0f}s)', flush=True)
        del J; torch.cuda.empty_cache()
    res = dict(mode=a.mode, noise=a.noise, recorded_only=a.recorded_only, rules=rules, lags=lags, device=torch.cuda.get_device_name(), torch=torch.__version__,
               matched_sets=dict(file=v['matched_sets' + ('_dev' if a.mode == 'dev' else '')], sha256=sha(path(v['matched_sets' + ('_dev' if a.mode == 'dev' else '')]))),
               match_attrs=dict(file=v['match_attrs'], sha256=sha(path(v['match_attrs']))), jobs=summ, smoke=bool(a.jobs or a.subjects or a.sets),
               max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2 ** 30, seconds=time.time() - t0,
               note='score matrices only (D-109 section 9); no MRP computed here')
    json.dump(res, open(os.path.join(od, f'{tag}_summary.json'), 'w'), indent=1)
    print('SUMMARY', json.dumps({k: res[k] for k in ('mode', 'noise', 'rules', 'device', 'smoke', 'max_memory_allocated_gb', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
