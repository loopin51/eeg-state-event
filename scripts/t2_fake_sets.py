"""Code-path smoke helper for F-IV T2 (not a registered computation): fake matched sets and attributes in the formats of
D-109 amendment 2 (match_attrs.json: key r{run}_w{word index} -> [run, t, tau, n_w, n_L, n_S, D_T, seq]; matched_sets.json:
key -> {cands: [19 keys], level}). Candidates are 19 random queries of the same run >= 30 s away (no attribute matching), so
nothing computed from these files means anything. Uses only word onsets and TTS onsets (no EEG, no decoder output).

usage: python scripts/t2_fake_sets.py configs/t2.yaml OUT_DIR [--runs 1 5 13] [--dev]
"""
import os, sys, json, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('out'); ap.add_argument('--runs', type=int, nargs='*'); ap.add_argument('--level', default='FAKE'); a = ap.parse_args()
    v = yaml.safe_load(open(os.path.join(ROOT, a.config))); b1 = yaml.safe_load(open(os.path.join(ROOT, v['b1_config'])))
    meta = json.load(open(os.path.join(ROOT, b1['cache_dir'], 'eeg_meta.json'))); fs, lens = meta['sample_rate'], meta['run_lengths']
    w, e = b1['window'], b1['eeg']; T = int(round((w['tmax'] - w['tmin']) * fs))
    Wd = np.load(os.path.join(ROOT, 'data/cache/ds004408/word_features.npz'), allow_pickle=True); td = os.path.join(ROOT, v['tts_features'])
    tm = json.load(open(os.path.join(td, 'meta.json'))); attrs, sets = {}, {}; rng = np.random.default_rng(0)
    for r in (np.array(a.runs) - 1 if a.runs else range(20)):
        m = np.where(Wd['run'] == r)[0]; on = Wd['onset'][m]; uq, first = np.unique(on, return_index=True)
        tau = np.asarray(json.load(open(os.path.join(td, f'run{r:02d}.json')))['onsets_s'])
        keep = (uq >= -w['tmin']) & (uq <= lens[r] / fs - w['tmax'] - e['offset_s'])
        for t, wi in zip(uq[keep], first[keep]):
            i0 = int(round((tau[wi] + w['tmin']) * fs))
            if i0 >= 0 and i0 + T <= tm['run_lengths'][r]: attrs[f'r{r + 1}_w{wi}'] = [int(r + 1), float(t), float(tau[wi]), 1, 1, 1, 1.0, f'fake{wi}']
        ks = [k for k in attrs if attrs[k][0] == r + 1]; ts = np.array([attrs[k][1] for k in ks])
        for i, k in enumerate(ks):
            far = [ks[j] for j in np.where(np.abs(ts - ts[i]) >= 30.0)[0]]
            if len(far) >= 19: sets[k] = dict(cands=[str(x) for x in rng.choice(far, 19, replace=False)], level=a.level)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(fields='run,t,tau,n_w,n_L,n_S,D_T,seq', queries=attrs, fake=True), open(os.path.join(a.out, 'match_attrs.json'), 'w'))
    json.dump(dict(level=a.level, sets=[dict(q=k, cands=x['cands'], cands_same_run=[]) for k, x in sets.items()], fake=True), open(os.path.join(a.out, 'matched_sets.json'), 'w'))
    print(f'fake: {len(attrs)} attrs, {len(sets)} sets', flush=True)


if __name__ == '__main__':
    main()
