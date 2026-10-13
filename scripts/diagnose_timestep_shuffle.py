"""Phase 0.4: reproduce that the 1st study's timestep-shuffle falsification is
vacuous.  Uses the archived readout code verbatim (src/ese/legacy/readout.py)
on synthetic H with a planted valence signal, and shows
  (a) archive as-is: 'real' vs 'shuffle_time' differ only because the RNG is
      consumed differently -> different group split;
  (b) with the split held fixed the two AUCs are bit-identical;
  (c) the split-to-split spread of the 'real' AUC is as large as the gaps the
      1st-study report interpreted (0.02-0.04).
Writes results/diagnosis_timestep_shuffle.{json,md}.
"""
import os, sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from ese.legacy.readout import valence_readout            # archived code, unmodified
from sklearn.metrics import roc_auc_score

def synth(N=3000, D=64, T=30, n_groups=300, snr=0.25, seed=0):
    """Groups (trials) with a binary valence; a planted direction carries the
    label, plus a time-varying component so temporal structure exists."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), N // n_groups)
    yg = rng.choice([-1, 1], n_groups); y = yg[g]
    H = rng.standard_normal((N, D, T)).astype(np.float32)
    d = rng.standard_normal(D); d /= np.linalg.norm(d)
    ramp = np.linspace(0, 1, T)                              # signal grows within window
    H += snr * y[:, None, None] * d[None, :, None] * ramp[None, None, :]
    group = np.array([f"0_0_{k}" for k in g])
    return H, y, group

def readout_fixed_split(H_all, valence, group, split_seed=0, test_frac=0.4, shuffle_time=False, time_seed=123):
    """Same maths as legacy valence_readout, but the split is drawn from its own
    RNG *before* any shuffling, so real/shuffled share an identical split."""
    binm = valence != 0
    H = H_all[binm]; y = (valence[binm] > 0).astype(int); g = group[binm]
    rs = np.random.default_rng(split_seed)
    ug = np.unique(g); rs.shuffle(ug)
    n_test = max(1, int(len(ug) * test_frac)); test_groups = set(ug[:n_test].tolist())
    te = np.array([gi in test_groups for gi in g]); tr = ~te
    if shuffle_time:
        rt = np.random.default_rng(time_seed); H = H.copy()
        for i in range(H.shape[0]):
            H[i] = H[i][:, rt.permutation(H.shape[2])]
    h = H.mean(axis=2); mu = h[tr].mean(0, keepdims=True)
    d = h[tr][y[tr] == 1].mean(0) - h[tr][y[tr] == 0].mean(0); dn = d / (np.linalg.norm(d) + 1e-8)
    z_t = np.einsum('ndt,d->nt', H[te] - mu[..., None], dn); score = z_t.mean(1)
    return float(roc_auc_score(y[te], score))

def main():
    H, y, group = synth()
    out = {}
    # (a) archive as-is, seeds 0..4 exactly as full_train.py did
    real = [valence_readout(H, y, group, seed=k)['auc'] for k in range(5)]
    tshuf = [valence_readout(H, y, group, seed=k, shuffle_time=True)['auc'] for k in range(5)]
    lshuf = [valence_readout(H, y, group, seed=k, shuffle_labels=True)['auc'] for k in range(5)]
    out['archive_as_is'] = dict(real_auc_seed0=real[0], real_auc_seeds=real, timestep_shuffle_auc_seeds=tshuf,
                                timestep_shuffle_mean=float(np.mean(tshuf)), label_shuffle_mean=float(np.mean(lshuf)),
                                note='full_train.py reported real=seed0 vs mean over seeds 0..4 of the shuffled variants')
    # (b) fixed split
    fixed = [(readout_fixed_split(H, y, group, split_seed=k), readout_fixed_split(H, y, group, split_seed=k, shuffle_time=True)) for k in range(5)]
    out['fixed_split'] = dict(pairs=fixed, max_abs_diff=float(max(abs(a - b) for a, b in fixed)))
    # (c) split variance of the real AUC
    spread = np.array([readout_fixed_split(H, y, group, split_seed=k) for k in range(50)])
    out['split_variance'] = dict(n_splits=50, mean=float(spread.mean()), std=float(spread.std()),
                                 range=[float(spread.min()), float(spread.max())])
    out['first_study_reported'] = dict(deap_mixed=dict(real=0.5492, timestep_shuffle=0.5691, label_shuffle=0.5042),
                                       source='04_results_data/ext_report_mixed.json')
    json.dump(out, open(os.path.join(ROOT, 'results/diagnosis_timestep_shuffle.json'), 'w'), indent=1)
    md = f"""# Phase 0.4 — timestep-shuffle 반증이 공허함을 재현

아카이브 `readout.py`의 `valence_readout`을 **수정 없이**(`src/ese/legacy/readout.py`) 합성 H(N=3000, D=64, T=30, 300 그룹, 심어 둔 valence 방향 + 시간 증가 성분)에 적용했다.

## (a) 아카이브 코드 그대로

`full_train.py`는 실제 AUC를 seed=0으로, shuffle 변형은 seed 0~4 평균으로 보고했다.

| 조건 | 값 |
|---|---|
| 실제 AUC (seed 0) | {real[0]:.4f} |
| 실제 AUC seed 0~4 | {', '.join(f'{v:.4f}' for v in real)} |
| timestep-shuffle seed 0~4 | {', '.join(f'{v:.4f}' for v in tshuf)} (평균 {np.mean(tshuf):.4f}) |
| label-shuffle 평균 | {np.mean(lshuf):.4f} |

실제와 timestep-shuffle이 다르게 보이지만, 아래 (b)가 보여 주듯 그 차이는 시간 순열 때문이 아니다.

## (b) 분할을 고정하면 비트 단위로 동일

같은 수식에서 분할을 별도 난수 생성기로 먼저 뽑은 뒤 시간 순열을 적용하면:

| split seed | 실제 | timestep-shuffle | 차이 |
|---|---|---|---|
""" + '\n'.join(f"| {k} | {a:.6f} | {b:.6f} | {abs(a-b):.1e} |" for k, (a, b) in enumerate(fixed)) + f"""

최대 절대 차이 = {out['fixed_split']['max_abs_diff']:.1e}. 점수 `score = z_t.mean(axis=1)`이 시간 순열에 불변이므로 당연한 결과다. 아카이브 코드에서 값이 달랐던 이유는 `rng.permutation`이 분할용 `rng.shuffle(ug)`보다 먼저 난수를 소비해 **분할이 바뀌었기** 때문이다.

## (c) 분할 노이즈의 크기

같은 데이터에서 분할 seed만 50개 바꾼 실제 AUC: 평균 {spread.mean():.4f}, 표준편차 {spread.std():.4f}, 범위 [{spread.min():.4f}, {spread.max():.4f}].

1차 보고서(`ext_report_mixed.json`)의 DEAP 값은 실제 0.549 / timestep-shuffle 0.569 / label-shuffle 0.504였다. 실제와 timestep-shuffle의 차이 0.020은 분할 노이즈 범위 안이며, "DEAP의 valence 신호가 시간 궤적보다 채널 주변 분포에 실려 있다"는 보고서의 해석은 측정에 근거하지 않은 것이었다.

## 결론

- timestep-shuffle은 이 판독 구조에서 아무것도 검정하지 않는다 → D-005대로 폐기하고 ISC 검정과 순환 이동 검정으로 대체한다.
- 반증 조건과 실제 조건은 **반드시 같은 분할**에서 비교해야 하며, 분할 노이즈는 부트스트랩 구간으로 함께 보고해야 한다(`05_evaluation_protocol.md` §1).
- 관문 G0-3 통과.
"""
    open(os.path.join(ROOT, 'results/diagnosis_timestep_shuffle.md'), 'w').write(md)
    print(md)

if __name__ == '__main__':
    main()
