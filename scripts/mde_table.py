"""Analytic minimum detectable effect (MDE) of the existing verdicts (direction 2, minimal version; D-065).

For each primary verdict the decision rule was "subject-bootstrap 95 % lower bound > chance threshold". With the
bootstrap standard error SE of the subject mean, a true mean mu is detected with 80 % power when
mu = threshold + (z_0.975 + z_0.80) * SE. A t-based variant (t_{0.975,n-1} + t_{0.80,n-1}) * sd / sqrt(n) is given as a
small-sample sensitivity column. Effects are reported as accuracy, as accuracy above theoretical chance (1/K) and as
Wolpaw bits/selection and bits/min (selection time per dataset, see SEL_TIME; the TOL value is the analysis epoch only,
so its bits/min is an upper bound). Reads committed result files only.

usage: python scripts/mde_table.py OUT_JSON
"""
import json, sys, os, numpy as np
from scipy.stats import norm, t as tdist
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = lambda p: json.load(open(os.path.join(ROOT, 'results', p)))
SEL_TIME = {'TOL': 4.5, 'Chisco': 10.08, 'ds004408': 3.0}   # seconds per selection (TOL: epoch length, lower bound of trial time)


def wolpaw(P, N):
    if P <= 1.0 / N: return 0.0
    if P >= 1.0: return float(np.log2(N))
    return float(np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1)))


def row(name, decision, data, K, scores, thr):
    x = np.asarray(scores, float); n = len(x); rng = np.random.default_rng(0)
    se = float(np.std([rng.choice(x, n).mean() for _ in range(10000)], ddof=1))
    mu = thr + (norm.ppf(0.975) + norm.ppf(0.80)) * se
    mu_t = thr + (tdist.ppf(0.975, n - 1) + tdist.ppf(0.80, n - 1)) * x.std(ddof=1) / np.sqrt(n)
    T = SEL_TIME[data]
    return dict(name=name, decision=decision, dataset=data, K=K, n_subjects=n, observed=float(x.mean()), threshold=float(thr), chance=1.0 / K,
                se_boot=se, mde=float(mu), mde_above_chance=float(mu - 1.0 / K), mde_t=float(mu_t), mde_t_above_chance=float(mu_t - 1.0 / K),
                bits_sel_at_mde=wolpaw(mu, K), bits_min_at_mde=wolpaw(mu, K) * 60 / T, bits_min_at_mde_t=wolpaw(mu_t, K) * 60 / T,
                bits_min_observed=wolpaw(float(x.mean()), K) * 60 / T, selection_time_s=T)


def main():
    out = sys.argv[1]; rows = []
    t = R('tol_calib.json')['summary']['inner']['DE']
    rows.append(row('Thinking Out Loud, 내적 발화 4단어, DE+LDA', 'D-046', 'TOL', 4, t['curve']['all']['per_subject'], t['chance_p95']))
    c = R('chisco_calib.json')['summary']['DE_IMAG']
    rows.append(row('Chisco 39범주, DE+LDA', 'D-048', 'Chisco', 39, c['curve']['all']['per_subject'], c['chance_p95']))
    for f, lab in (('eegnet_imag_day', 'EEGNet'), ('conformer_imag_day', 'Conformer-pt')):
        d = R(f'chisco_nn/{f}.json'); s = [d['subjects'][k]['mean'] for k in sorted(d['subjects'], key=int)]
        rows.append(row(f'Chisco 39범주, {lab}', 'D-056', 'Chisco', 39, s, 0.0263))   # D-054 threshold: max of 5 permuted EEGNet rounds
    i = R('ds004408/identify.json')['summary']
    rows.append(row('ds004408 3 s 조각 20후보, EEG→문장 임베딩, 피험자별', 'D-060', 'ds004408', 20, i['EEG->TXT']['per_subject'], i['EEG->TXT']['chance_threshold']))
    p = R('ds004408/identify_pool.json')['summary']['pool17']
    rows.append(row('ds004408 3 s 조각 20후보, EEG→문장 임베딩, 통합 17명', 'D-063', 'ds004408', 20, p['EEG->TXT']['per_subject'], p['EEG->TXT']['chance_threshold']))
    rows.append(row('ds004408 3 s 조각 20후보, EEG→포락선, 통합 17명 (검출됨, 참고)', 'D-063', 'ds004408', 20, p['EEG->ENV']['per_subject'], p['EEG->ENV']['chance_threshold']))
    json.dump(dict(method=__doc__, selection_time_s=SEL_TIME, rows=rows), open(out, 'w'), indent=1, ensure_ascii=False)
    for r in rows:
        print(f"{r['decision']} {r['name']}: obs {r['observed']:.4f} thr {r['threshold']:.4f} SE {r['se_boot']:.4f} | MDE {r['mde']:.4f} "
              f"(+{r['mde_above_chance']:.4f} over chance; t {r['mde_t']:.4f}) | {r['bits_min_at_mde']:.3f} bits/min (t {r['bits_min_at_mde_t']:.3f})")


if __name__ == '__main__':
    main()
