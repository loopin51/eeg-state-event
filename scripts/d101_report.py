"""D-101 report: band-input retention (a), coherence / lag flags A_w, A_e, A_f (b), onset latencies (c), and the registered
decision table (rules 1-4, docs/DECISIONS.md D-101).

usage: python scripts/d101_report.py configs/d101_artifact.yaml
"""
import os, sys, glob, json, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pooled(M, w):
    ok = ~np.isnan(M).all(0); return float((np.nanmean(M[:, ok], 0) * w[ok]).sum() / w[ok].sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args(); c = yaml.safe_load(open(a.config))
    b1 = yaml.safe_load(open(os.path.join(ROOT, c['b1_config']))); sp = b1['split']; cache = os.path.join(ROOT, c['cache']); bi = c['band_input']
    pool = sorted(s for f in sp['subject_folds'] for s in f); bands = list(bi['bands']); res = {}
    # (a)
    V = {'main': {}, 'phaserand': {}}
    for p in glob.glob(os.path.join(cache, 'band_*.npz')):
        tag = 'phaserand' if 'phaserand' in p else 'main'; Z = np.load(p); V[tag].update({k: Z[k] for k in Z.files})
    def mats(tag, subjects, kind, band):
        obs = np.full((len(subjects), 20), np.nan); nm = obs.copy(); nul = np.full((len(subjects), 20, bi['shift_null']['n_shift']), np.nan); w = np.zeros(20)
        for i, s in enumerate(subjects):
            for r in range(20):
                v = V[tag].get(f'{kind}_s{s}_r{r}_{band}')
                if v is not None: obs[i, r], w[r], nul[i, r], nm[i, r] = v[0], v[1], v[2:], v[2:].mean()
        return obs, nm, nul, w
    rng = np.random.default_rng(bi['bootstrap']['seed']); NBt = bi['bootstrap']['n_boot']; SI = rng.integers(0, len(pool), (NBt, len(pool))); RI = rng.integers(0, 20, (NBt, 20))
    ex, boot = {}, {}; band_res = {}
    for band in bands:
        obs, nm, nul, w = mats('main', pool, 'b1', band)
        if np.isnan(obs).all(): continue
        E = obs - nm; ex[band] = pooled(E, w); boot[band] = np.array([pooled(E[np.ix_(SI[b], RI[b])], w[RI[b]]) for b in range(NBt)])
        gnull = np.array([pooled(nul[..., b], w) for b in range(nul.shape[-1])])
        o2, nm2, _, _ = mats('main', pool, 'ex', band)
        po, pn, _, pw = mats('phaserand', sp['subject_folds'][0], 'b1', band)
        band_res[band] = dict(mrp=pooled(obs, w), null_mean=pooled(nm, w), excess=ex[band], obs_above_null_p975=bool(pooled(obs, w) > np.percentile(gnull, 97.5)),
                              excl30_mrp=pooled(o2, w), excl30_excess=pooled(o2 - nm2, w), phaserand_excess=pooled(po - pn, pw) if not np.isnan(po).all() else None)
    for band in band_res:
        rb = boot[band] / boot['full']; band_res[band]['retention'] = ex[band] / ex['full']; band_res[band]['retention_ci95'] = [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]
    res['band_input'] = band_res
    # (b)
    Z = np.load(os.path.join(cache, 'coherence.npz'), allow_pickle=True); M = json.load(open(os.path.join(cache, 'coherence_meta.json'))); co = c['coherence']
    bnames = list(Z['band_names']); freqs = Z['freqs']; ch = list(Z['ch_names']); coh = {}
    for k in ('wave', 'env'):
        G = Z[f'coh_{k}'].mean(0)                                                       # [C, F]
        Nb = Z[f'coh_null_band_{k}'].mean(0)                                            # [B, C, nb]
        out = {}
        for j, b in enumerate(bnames):
            lo, hi = co['bands_hz'][b]; m = (freqs >= lo) & (freqs <= hi); ob = G[:, m].mean(1)
            exc = ob - Nb[:, :, j].mean(0); top = np.argsort(-exc)[:10]
            out[b] = dict(global_obs=float(ob.mean()), global_null_p95=float(np.percentile(Nb[:, :, j].mean(1), 95)), global_null_mean=float(Nb[:, :, j].mean()),
                          chmax_obs=float(ob.max()), chmax_null_p95=float(np.percentile(Nb[:, :, j].max(1), 95)),
                          top10=[dict(ch=ch[i], excess=float(exc[i])) for i in top])
            out[b]['global_above_null'] = bool(out[b]['global_obs'] > out[b]['global_null_p95'])
        coh[k] = out
    try:
        import mne; pos = mne.channels.make_standard_montage('biosemi128').get_positions()['ch_pos']
        for k in coh:
            for b in coh[k]:
                for t in coh[k][b]['top10']: t['xyz_m'] = [round(float(x), 4) for x in pos[t['ch']]] if t['ch'] in pos else None
    except Exception as e: res['montage_error'] = str(e)
    res['coherence'] = coh
    lag = {k: np.array(v) for k, v in M['lag_ms'].items()}; xcr = {}; inst = co['instantaneous_ms']
    for k in lag:
        g = np.sqrt((Z[f'xc_{k}'].mean(0) ** 2).mean(0)); gn = np.sqrt((Z[f'xc_null_{k}'].mean(0) ** 2).mean(1))      # [L], [B, L]
        pk = int(np.argmax(g)); d = dict(peak_lag_ms=float(lag[k][pk]), peak=float(g[pk]), null_peak_p95=float(np.percentile(gn.max(1), 95)), curve=g.tolist())
        if k.startswith('env'):
            z0 = np.abs(lag[k]) <= inst['env']; late = lag[k] >= 50
            d.update(v0=float(g[z0].max()), v0_null_p95=float(np.percentile(gn[:, z0].max(1), 95)), late_max=float(g[late].max()), late_peak_lag_ms=float(lag[k][late][np.argmax(g[late])]))
        xcr[k] = d
    res['xcorr'] = {k: {kk: vv for kk, vv in v.items() if kk != 'curve'} for k, v in xcr.items()}
    A_w = bool(coh['wave']['b20_60']['global_above_null'] and abs(xcr['wave_b20_60']['peak_lag_ms']) <= inst['wave'] and xcr['wave_b20_60']['peak'] > xcr['wave_b20_60']['null_peak_p95'])
    A_f = bool(coh['wave']['b70_250']['global_above_null'] and abs(xcr['wave_b70_250']['peak_lag_ms']) <= inst['wave'])
    e = xcr['env_b1_60']; A_e = bool(e['v0'] > e['v0_null_p95'] and e['v0'] > e['late_max'])
    # (c)
    t = Z['erp_t_ms']; base = t <= 0; lat = {}
    for k, X in (('erp', Z['erp']), ('high', Z['high']), ('audio_env', Z['aenv'])):
        if k == 'audio_env': g = X.mean(0)
        else:
            Xm = X.mean(0); Xm = Xm - Xm[:, base].mean(1, keepdims=True); g = np.sqrt((Xm ** 2).mean(0)) if k == 'erp' else Xm.mean(0)
        thr = g[base].mean() + c['onset_erp']['threshold_sd'] * g[base].std(); need = int(np.ceil(c['onset_erp']['min_dur_ms'] * 512 / 1000)); above = (g > thr) & (t >= 0); L = None
        for i in range(len(t) - need):
            if above[i:i + need].all(): L = float(t[i]); break
        lat[k] = L
    res['onset_latency_ms'] = lat; res['n_onsets'] = M['n_events']
    # decision (registered order)
    rho20, rho8 = band_res['lp20']['retention'], band_res['lp8']['retention']; th = c['rules']['retention']
    if A_w and rho20 < th: dec = '인공물 의심(파형): C1 전처리에 20 Hz 저역통과를 넣는 결과 전 보완을 제안'
    elif A_e: dec = '인공물 의심(포락선 즉시 결합): 대응은 별도 결정'
    elif not A_w and not A_e and rho20 >= th:
        dec = '신경 추적 해석 유지; ' + ('8 Hz 이상 성분이 식별 초과분의 절반 이상을 담는다(50 ms 민감도와 맞음)' if rho8 < th else '8 Hz 이하 성분이 절반 이상을 담는다')
    else: dec = '결정 불가' + (' (20 Hz 이상 성분에 기대지만 원인은 정하지 못함)' if rho20 < th and not A_w and not A_e else '')
    res['flags'] = dict(A_w=A_w, A_e=A_e, A_f=A_f); res['decision'] = dec
    if A_f: res['note_A_f'] = '외부 자료(C1 사전학습, C2) 전처리에서 반에일리어싱 저역통과를 확인할 것'
    sp_path = os.path.join(ROOT, c['out'], 'sparrkulee_artefact.json')      # D-101 amendment 1: report-only reference, not in rules 1-4
    if os.path.exists(sp_path): res['sparrkulee_reference'] = json.load(open(sp_path))
    out = os.path.join(ROOT, c['out']); os.makedirs(out, exist_ok=True); json.dump(res, open(os.path.join(out, 'd101.json'), 'w'), indent=1, ensure_ascii=False)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), facecolor='#EEF0EC')
    for k, col in (('wave', '#B5652D'), ('env', '#2E6E6B')):
        ax[0].semilogy(freqs, Z[f'coh_{k}'].mean(0).mean(0), color=col, label=f'{k} obs'); ax[0].semilogy(freqs, Z[f'coh_null_spec_{k}'].mean(0).mean(0), color=col, ls='--', label=f'{k} null mean')
    ax[0].set_xlabel('Hz'); ax[0].set_ylabel('coherence (channel mean)'); ax[0].legend(fontsize=7)
    for k, col in (('wave_b20_60', '#B5652D'), ('wave_b70_250', '#5B4A86'), ('env_b1_60', '#2E6E6B')): ax[1].plot(lag[k], xcr[k]['curve'], color=col, label=k)
    ax[1].axvline(0, color='#182129', lw=0.5); ax[1].set_xlabel('lag (ms, EEG after audio > 0)'); ax[1].legend(fontsize=7)
    bb = [b for b in bands if b in band_res]; ax[2].bar(bb, [band_res[b]['retention'] for b in bb], color='#B5652D'); ax[2].axhline(0.5, color='#182129', ls='--', lw=0.5); ax[2].set_ylabel('retention of B1 excess')
    for x in ax: x.set_facecolor('#EEF0EC')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'd101.png'), dpi=150)
    print(json.dumps(dict(decision=dec, flags=res['flags'], band={b: (round(v['retention'], 3), round(v['excess'], 4), v['phaserand_excess']) for b, v in band_res.items()},
                          coh={k: {b: (round(v['global_obs'], 5), round(v['global_null_p95'], 5)) for b, v in coh[k].items()} for k in coh},
                          xcorr=res['xcorr'], latency=lat), ensure_ascii=False, indent=1, default=str), flush=True)


if __name__ == '__main__':
    main()
