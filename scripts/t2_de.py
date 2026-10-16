"""F-IV T2 DE+ridge baseline on the matched candidates (D-109 section 11), committed before any decoder TTS score.

Fit exactly as the B1 DE (scripts/stageB_b1_meters.py, D-085 baseline, A4 setting): log band power (5 bands) x 128 channels of
the EEG window after the decoder's reference, baseline and clamp; ridge per subject and run fold on the subject's own 14
training runs (0.5-s grid windows), penalty by grouped CV (7 groups of 2 runs); target = time-pooled recorded w2v standardised
with the fold's training-run statistics. Scoring: cosine of the prediction for each Q query (results/t2/matched_sets.json,
EEG margin of section 4) with the time-pooled candidate windows, truth + the 19 matched candidates, MRP (below + 0.5 ties)/19,
TTS candidates (primary) and recorded candidates; fixed alignment only. All 19 subjects. No MRP level is printed (the values
are read by the verdict for prerequisites 3 and 4).

usage: python scripts/t2_de.py configs/t2.yaml [--subjects S ...] [--sets PATH --attrs PATH --out PATH]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, argparse, hashlib, yaml, numpy as np
from scipy.signal import welch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stageB_b1_meters import BANDS, fit_predict, unit
from t2_io import read_sets, read_attrs, full


def path(p): return p if os.path.isabs(p) else os.path.join(ROOT, p)
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
def parse_key(k): r, w = k.split('_'); return int(r[1:]) - 1, int(w[1:])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--subjects', type=int, nargs='*'); ap.add_argument('--sets'); ap.add_argument('--attrs')
    ap.add_argument('--out'); a = ap.parse_args(); v = yaml.safe_load(open(path(a.config))); t0 = time.time()
    c = yaml.safe_load(open(path(v['b1_config']))); sp = c['split']; st = c['settings'][c['decoder_setting']]; d = path(c['cache_dir'])
    setp, attp = path(a.sets or v['matched_sets']), path(a.attrs or v['match_attrs'])
    sets = full(read_sets(setp)[1]); attrs = read_attrs(attp)
    meta = json.load(open(os.path.join(d, 'eeg_meta.json'))); fs, offs, lens = meta['sample_rate'], meta['run_offsets'], meta['run_lengths']
    w, e = c['window'], c['eeg']; T = int(round((w['tmax'] - w['tmin']) * fs)); lo = -w['tmin']
    def hi(r): return lens[r] / fs - w['tmax'] - max(e['offset_s'], 0)
    def eeg_ok(t, r0):
        m = v['eeg_margin_s']; a0 = int(round((t + w['tmin'] + e['offset_s'] - m) * fs)); return a0 >= 0 and int(round((t + w['tmin'] + e['offset_s'] + m) * fs)) + T <= lens[r0]
    rows = {r: sorted((k for k in sets if parse_key(k)[0] == r and eeg_ok(attrs[k][1], r)), key=lambda k: attrs[k][1]) for r in range(20)}
    grid = [(r, float(t)) for r in range(20) for t in np.arange(lo, hi(r) + 1e-9, w['train_step_s'])]; ng = len(grid)
    qkeys = [k for r in range(20) for k in rows[r]]; qpos = {k: i for i, k in enumerate(qkeys)}; items = grid + [(parse_key(k)[0], attrs[k][1]) for k in qkeys]
    run_all = np.array([r for r, _ in items])
    F = np.asarray(np.load(os.path.join(d, c['features']['w2v'] + '.npy'), mmap_mode='r'), np.float32)
    fi = np.array([offs[r] + int(round((t + w['tmin']) * fs)) for r, t in items]); pooled_rec = np.stack([F[:, i:i + T].mean(1) for i in fi])
    allk = sorted(set(qkeys) | {cc for k in qkeys for cc in sets[k]['cands']}); kidx = {k: i for i, k in enumerate(allk)}
    rec_c = np.stack([F[:, offs[parse_key(k)[0]] + int(round((attrs[k][1] + w['tmin']) * fs)):][:, :T].mean(1) for k in allk])
    tm = json.load(open(os.path.join(path(v['tts_features']), 'meta.json'))); TF = np.load(os.path.join(path(v['tts_features']), 'w2v_run.npy'), mmap_mode='r')
    tts_c = np.stack([np.asarray(TF[:, tm['run_offsets'][parse_key(k)[0]] + int(round((attrs[k][2] + w['tmin']) * fs)):][:, :T], np.float32).mean(1) for k in allk])
    fold_of = {r - 1: f for f in range(5) for r in sp['run_folds'][f]}; stats = {}
    for f in range(5):
        test = [r - 1 for r in sp['run_folds'][f]]; valid = [r - 1 for r in sp['valid_runs'][f]]; train = [r for r in range(20) if r not in test + valid]
        samp = np.concatenate([np.arange(offs[r], offs[r] + lens[r]) for r in train]); samp = np.sort(samp[:: max(1, len(samp) // 200000)])
        stats[f] = (test, train, F[:, samp].mean(1), F[:, samp].std(1) + 1e-6)
    del F
    subjects = a.subjects or sorted(int(s) for s in meta['subjects']); res = dict(subjects={}, setting=c['decoder_setting'], car_reference=st['reference'])
    for s in subjects:
        eeg = np.load(os.path.join(d, f"{st['source']}_sub-{s:03d}.npy"), mmap_mode='r'); flat = np.array(meta['subjects'][str(s)]['flat'])
        X = np.empty((len(items), 128 * len(BANDS)), np.float32)
        for b0 in range(0, len(items), 1000):
            xs = []
            for (r, t) in items[b0:b0 + 1000]:
                i = offs[r] + int(round((t + w['tmin'] + e['offset_s']) * fs)); x = np.asarray(eeg[:, i:i + T], np.float32).copy()
                if st['reference'] == 'car': x -= x[~flat[r]].mean(0, keepdims=True)
                x -= x[:, : int(round(e['baseline_s'] * fs))].mean(1, keepdims=True); xs.append(np.clip(x, -e['clamp'], e['clamp']))
            fr, p = welch(np.stack(xs), fs=fs, nperseg=min(fs, T), axis=-1)
            X[b0:b0 + len(xs)] = np.log(np.stack([p[..., (fr >= l) & (fr < h)].mean(-1) for l, h in BANDS], -1) + 1e-8).reshape(len(xs), -1)
        cells, alphas = {}, {}
        for f in range(5):
            test, train, fm, fsd = stats[f]; tr_i = np.where(np.isin(run_all[:ng], train))[0]
            te_k = [k for r in test for k in rows[r]]; te_i = ng + np.array([qpos[k] for k in te_k]); tpos = {k: i for i, k in enumerate(te_k)}
            grp = np.array([train.index(r) // 2 for r in run_all[tr_i]])
            P, al = fit_predict(X[tr_i], (pooled_rec[tr_i] - fm) / fsd, X[te_i], grp); alphas[str(f)] = al; P = unit(P)
            for kind, Cw in (('tts', tts_c), ('rec', rec_c)):
                Cn = unit((Cw - fm) / fsd)
                for r in test:
                    if not rows[r]: continue
                    idx = [tpos[k] for k in rows[r]]; Ci = np.array([[kidx[k]] + [kidx[x] for x in sets[k]['cands']] for k in rows[r]])
                    sc = np.einsum('nf,nkf->nk', P[idx], Cn[Ci]); tr_ = sc[:, :1]
                    m = ((sc[:, 1:] < tr_).sum(1) + 0.5 * (sc[:, 1:] == tr_).sum(1)) / 19.0
                    cells.setdefault(str(r + 1), dict(n=len(idx)))[f'mrp_{kind}'] = float(m.mean())
        res['subjects'][str(s)] = dict(cells=cells, alpha=alphas)
        print(f'DE sub-{s:03d} done ({time.time() - t0:.0f}s)', flush=True)
    res.update(eval_split='test', matched_sets=dict(file=os.path.relpath(setp, ROOT), sha256=sha(setp)), match_attrs=dict(file=os.path.relpath(attp, ROOT), sha256=sha(attp)),
               queries_per_run={str(r + 1): len(rows[r]) for r in range(20)}, eeg_margin_s=v['eeg_margin_s'], alignment='fixed', seconds=time.time() - t0,
               note='D-109 section 11; values are read by the verdict (prerequisites 3, 4) and not printed here')
    out = path(a.out or 'results/baselines/t2_de.json'); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    print('wrote', out, f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
