"""D-110 section 4.5 diagnosis after the failed U1 (no new test statistic). Per session, cue-type average ERPs from the U1 cache:
audio aligned to speech onset (u1), audio / text / pictures aligned to StartStimulus (cue). ROI = U1 frontocentral ROI and the
P100 occipital ROI (flat channels left out); GFP = spatial sd over the non-flat channels of the average ERP. For each curve the
most negative ROI time and the largest GFP time in [50, 300] ms are listed. Writes results/ugr/u1_diag.json and a figure.

usage: python scripts/ugr_u1_diag.py configs/ugr_u.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, yaml, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ugr_u1 import sessions, load
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    c = yaml.safe_load(open(os.path.join(ROOT, sys.argv[1]))); u = c['u1']; fs = u['sample_rate']
    times = np.arange(int(round((u['epoch_s'][1] - u['epoch_s'][0]) * fs))) / fs + u['epoch_s'][0]; w = (times >= 0.05) & (times <= 0.30)
    res = dict(times_s=times.tolist(), sessions={}); curves = {}
    for s in sessions(c):
        S = load(c, s); ch = S['channels']; cue = np.array([m['cue'] for m in S['meta']])
        out = {}
        for name, arr, k in (('audio_speech', 'u1', 'Audio'), ('audio_stim', 'cue', 'Audio'), ('text_stim', 'cue', 'Text'), ('pictures_stim', 'cue', 'Pictures')):
            E = np.nanmean(S[arr][cue == k], 0)                                     # [ch, T]
            good = ~np.isnan(E).any(1); gfp = E[good].std(0)
            fc = np.nanmean(E[[ch.index(r) for r in u['roi']]], 0); oc = np.nanmean(E[[ch.index(r) for r in u['p100']['roi']]], 0)
            out[name] = dict(n=int((cue == k).sum()), roi_fc_min_ms=float(1000 * times[w][np.argmin(fc[w])]), roi_fc_min_uv=float(fc[w].min()),
                             gfp_max_ms=float(1000 * times[w][np.argmax(gfp[w])]), gfp_max_uv=float(gfp[w].max()),
                             roi_oc_max_ms=float(1000 * times[w][np.argmax(oc[w])]), roi_oc_max_uv=float(oc[w].max()),
                             roi_fc=fc.tolist(), roi_oc=oc.tolist(), gfp=gfp.tolist())
            curves.setdefault(name, {})[s] = (fc, oc, gfp)
        res['sessions'][s] = out
        print(s, {k: (v['roi_fc_min_ms'], round(v['roi_fc_min_uv'], 2), v['gfp_max_ms'], round(v['gfp_max_uv'], 2)) for k, v in out.items()}, flush=True)
    p = os.path.join(ROOT, c['out'], 'u1_diag.json'); json.dump(res, open(p, 'w'))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    col = dict(audio_speech='#B5652D', audio_stim='#5B4A86', text_stim='#2E6E6B', pictures_stim='#182129')
    ss = sessions(c); fig, ax = plt.subplots(len(ss), 2, figsize=(9, 1.6 * len(ss)), sharex=True)
    for i, s in enumerate(ss):
        for name in col:
            fc, oc, gfp = curves[name][s]
            ax[i, 0].plot(1000 * times, fc, color=col[name], lw=1, label=name); ax[i, 1].plot(1000 * times, gfp, color=col[name], lw=1)
        ax[i, 0].set_ylabel(s, fontsize=7); ax[i, 0].axvline(0, color='0.6', lw=0.5); ax[i, 1].axvline(0, color='0.6', lw=0.5)
    ax[0, 0].set_title('frontocentral ROI (uV)', fontsize=8); ax[0, 1].set_title('GFP (uV)', fontsize=8); ax[0, 0].legend(fontsize=6)
    ax[-1, 0].set_xlabel('ms'); ax[-1, 1].set_xlabel('ms'); fig.tight_layout(); fig.savefig(os.path.join(ROOT, c['out'], 'u1_diag.png'), dpi=110)


if __name__ == '__main__':
    main()
