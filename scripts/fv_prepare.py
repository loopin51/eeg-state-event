"""F-V (D-111): preprocess the TRAINING file of one THINGS-EEG2 subject x session into data/cache/things_eeg2/.

Reads data_root/sub-XX/ses-YY/raw_eeg_training.npy with the restricted unpickler (src/ese/things/rawio.py), then
src/ese/things/preprocess.py: FIR low-pass, 250-Hz epochs -0.2..1.0 s, baseline, catch removal, MVNN fitted on these training
epochs. The test file is never opened here (rawio.guard stops it); it is preprocessed with the stored W inside the verdict job.
Writes into cache_dir/sub-XX/:
  ses-YY_training.npy         float16 [n, 63, 300] whitened epochs (open_memmap)
  ses-YY_training_events.npz  code (image condition, 1-based), sample (1000-Hz event sample), rep (repetition index of the code
                              inside this session, in time order), order (position in the session)
  ses-YY_mvnn.npz             sigma, W (= sigma^-1/2), ch_names, times
  ses-YY_prep.json            input sha256, counts, timings, peak memory (ru_maxrss) after each stage
and cache_dir/positions.npy + channels.json (2D positions in the common frame of src/ese/listen/positions.py; checked equal for
every session).

usage: python scripts/fv_prepare.py configs/fv_preprocess.yaml --subject S --session N [--no-sha]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, resource, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.things import rawio, preprocess as P


def peak_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # Linux: kB


def positions(names):
    from ese.listen.positions import template_3d, common_positions
    ref = template_3d('standard_1005'); low = {k.lower(): k for k in ref}; b64 = set(template_3d('biosemi64'))
    miss = [n for n in names if n.lower() not in low]; assert not miss, f'channels without a standard_1005 position: {miss}'
    pos3d = np.array([ref[low[n.lower()]] for n in names]); anchors = {n: low_b for n in names for low_b in b64 if low_b.lower() == n.lower()}
    return common_positions(names, pos3d, anchors)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--subject', type=int, required=True)
    ap.add_argument('--session', type=int, required=True); ap.add_argument('--no-sha', action='store_true'); a = ap.parse_args()
    c = yaml.safe_load(open(os.path.join(ROOT, a.config))); t0 = time.time(); mem = {}
    src = os.path.join(ROOT, c['data_root'], f'sub-{a.subject:02d}', f'ses-{a.session:02d}', 'raw_eeg_training.npy')
    out = os.path.join(ROOT, c['cache_dir'], f'sub-{a.subject:02d}'); os.makedirs(out, exist_ok=True); tag = f'ses-{a.session:02d}'
    sha = None if a.no_sha else rawio.sha256(src); mem['sha'] = peak_gb()
    d = rawio.load_dict(src, 'raw'); mem['load'] = peak_gb(); x = d['raw_eeg_data']
    print(f'loaded {src}: {x.shape} {x.dtype} sfreq {d["sfreq"]} ({time.time() - t0:.0f}s, peak {mem["load"]:.2f} GB)', flush=True)
    E = P.epochs_from_raw(d, c, log=lambda s: print(s, flush=True)); mem['epochs'] = peak_gb()
    info = dict(dtype=str(x.dtype), shape=list(x.shape), sfreq=float(d['sfreq']), amplitude_sd=float(np.std(x[[i for i, t in enumerate(d['ch_types']) if t == 'eeg'][0], ::1000])))
    del d, x
    code = E['code']; order = np.arange(len(code)); rep = np.zeros(len(code), np.int64)
    for v in np.unique(code): ii = np.where(code == v)[0]; rep[ii] = np.arange(len(ii))
    S, W = P.mvnn_fit(E['X'], code, E['times'], c); mem['mvnn'] = peak_gb(); t_mvnn = time.time() - t0
    P.whiten(E['X'], W)
    Y = np.lib.format.open_memmap(os.path.join(out, f'{tag}_training.npy'), mode='w+', dtype=c['store_dtype'], shape=E['X'].shape)
    Y[:] = E['X']; Y.flush(); del Y; mem['write'] = peak_gb()
    np.savez(os.path.join(out, f'{tag}_training_events.npz'), code=code, sample=E['sample'], rep=rep, order=order)
    np.savez(os.path.join(out, f'{tag}_mvnn.npz'), sigma=S, W=W, ch_names=np.array(E['ch_names']), times=E['times'])
    xy, fit = positions(E['ch_names']); pp = os.path.join(ROOT, c['cache_dir'], 'positions.npy'); cj = os.path.join(ROOT, c['cache_dir'], 'channels.json')
    if os.path.exists(pp):
        assert json.load(open(cj))['ch_names'] == E['ch_names'] and np.allclose(np.load(pp), xy), 'channel set or positions differ from the cache'
    else:
        np.save(pp, xy); json.dump(dict(ch_names=E['ch_names'], fit=fit, frame='src/ese/listen/positions.py (biosemi64 Umeyama, biosemi128 topomap frame)'), open(cj, 'w'), indent=1)
    reps = np.bincount(np.bincount(code)[np.unique(code)])
    res = dict(subject=a.subject, session=a.session, source=os.path.relpath(src, ROOT), sha256=sha, raw=info, counts=E['counts'], fir_taps=E['fir_taps'],
               n_epochs=int(len(code)), n_conditions=int(len(np.unique(code))), code_range=[int(code.min()), int(code.max())],
               reps_per_condition_hist={str(k): int(v) for k, v in enumerate(reps) if v}, times=[float(E['times'][0]), float(E['times'][-1]), len(E['times'])],
               mvnn=dict(min_eig=float(np.linalg.eigvalsh(S).min()), max_eig=float(np.linalg.eigvalsh(S).max()), seconds=t_mvnn),
               whitened_sd=float(E['X'][:, :, 50:].std()), peak_rss_gb=mem, seconds=time.time() - t0, config=c)
    json.dump(res, open(os.path.join(out, f'{tag}_prep.json'), 'w'), indent=1)
    print('SUMMARY', json.dumps({k: res[k] for k in ('n_epochs', 'n_conditions', 'code_range', 'reps_per_condition_hist', 'counts', 'peak_rss_gb', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
