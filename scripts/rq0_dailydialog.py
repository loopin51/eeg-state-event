"""RQ0: how much does affect narrow the speech-act distribution?

Computes mutual information I(emotion; act) on DailyDialog utterances (and
several variants) to bound the gain a perfect affect channel could give the
rerank term  lambda_2 * log p(act | a)  in L3.  Outputs
results/rq0_dailydialog.json and results/rq0_dailydialog.md.
"""
import json, os, numpy as np, pandas as pd
from itertools import chain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PQ = os.path.join(ROOT, 'data/raw/dailydialog/hf_parquet/default')
OUT = os.path.join(ROOT, 'results')
os.makedirs(OUT, exist_ok=True)

EMO = {0: 'no_emotion', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'happiness', 5: 'sadness', 6: 'surprise'}
ACT = {1: 'inform', 2: 'question', 3: 'directive', 4: 'commissive'}
# circumplex-style collapse used for p(act | a) initialisation
VALENCE = {1: -1, 2: -1, 3: -1, 5: -1, 4: +1, 6: 0, 0: 0}   # surprise/no_emotion -> neutral

def load():
    dfs = [pd.read_parquet(os.path.join(PQ, s, '0000.parquet')) for s in ['train', 'validation', 'test']]
    df = pd.concat(dfs, ignore_index=True)
    rows = []
    for di, r in df.iterrows():
        emo, act = list(r['emotion']), list(r['act'])
        for k in range(len(act)):
            rows.append(dict(dialog=di, turn=k, emo=int(emo[k]), act=int(act[k]),
                             prev_emo=int(emo[k-1]) if k > 0 else -1,
                             next_act=int(act[k+1]) if k+1 < len(act) else -1,
                             own_next_act=int(act[k+2]) if k+2 < len(act) else -1))
    return pd.DataFrame(rows)

def entropy(p):
    p = p[p > 0]; return float(-(p * np.log2(p)).sum())

def mi(x, y):
    ct = pd.crosstab(x, y).to_numpy().astype(float)
    p = ct / ct.sum(); px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())

def analyse(x, y, name, n_perm=200, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    I = mi(x, y); Hy = entropy(pd.Series(y).value_counts(normalize=True).to_numpy())
    Hx = entropy(pd.Series(x).value_counts(normalize=True).to_numpy())
    perm = np.array([mi(rng.permutation(x), y) for _ in range(n_perm)])
    idx = np.arange(len(x))
    boot = np.array([mi(x[b], y[b]) for b in (rng.choice(idx, len(idx)) for _ in range(n_boot))])
    # Bayes-optimal act accuracy given x vs marginal majority
    ct = pd.crosstab(x, y)
    acc_cond = float(ct.max(axis=1).sum() / ct.to_numpy().sum())
    acc_marg = float(pd.Series(y).value_counts(normalize=True).max())
    return dict(name=name, n=int(len(x)), I_bits=I, H_act_bits=Hy, H_x_bits=Hx,
                I_over_H_act=I / Hy, I_perm_mean=float(perm.mean()), I_perm_p99=float(np.quantile(perm, .99)),
                I_boot_ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                bayes_acc_given_x=acc_cond, majority_acc=acc_marg)

def main():
    df = load()
    res = {}
    res['all_same_turn'] = analyse(df.emo, df.act, 'emotion(t) -> act(t), all utterances')
    e = df[df.emo != 0]
    res['emotional_same_turn'] = analyse(e.emo, e.act, 'emotion(t) -> act(t), emotional utterances only')
    v = df.assign(val=df.emo.map(VALENCE)); v = v[v.val != 0]
    res['valence_same_turn'] = analyse(v.val, v.act, 'valence(t) -> act(t), pos/neg only')
    p = df[df.prev_emo > 0]
    res['emotional_prev_turn'] = analyse(p.prev_emo, p.act, 'emotion(t-1) -> act(t)  [partner-affect prior]')
    n = df[(df.emo != 0) & (df.own_next_act > 0)]
    res['emotional_own_next_turn'] = analyse(n.emo, n.own_next_act, 'emotion(t) -> act(t+2)  [same speaker next act]')
    a = df[df.emo != 0].assign(arousal=df.emo.map({1: 1, 3: 1, 6: 1, 4: 1, 2: 0, 5: 0}))  # high: anger/fear/surprise/happiness, low: disgust/sadness
    res['arousal_same_turn'] = analyse(a.arousal, a.act, 'arousal(t) -> act(t), emotional only')
    # conditional tables for p(act | emotion) initialisation
    tab = pd.crosstab(df.emo.map(EMO), df.act.map(ACT), normalize='index').round(4)
    tab_v = pd.crosstab(v.val.map({-1: 'negative', 1: 'positive'}), v.act.map(ACT), normalize='index').round(4)
    counts = pd.crosstab(df.emo.map(EMO), df.act.map(ACT))
    out = dict(source='li2017dailydialog/daily_dialog (HF parquet conversion), train+validation+test',
               n_dialogs=int(df.dialog.nunique()), n_utterances=int(len(df)),
               emotion_marginal=df.emo.map(EMO).value_counts(normalize=True).round(4).to_dict(),
               act_marginal=df.act.map(ACT).value_counts(normalize=True).round(4).to_dict(),
               analyses=res, p_act_given_emotion=tab.to_dict(orient='index'),
               p_act_given_valence=tab_v.to_dict(orient='index'), counts=counts.to_dict(orient='index'))
    json.dump(out, open(os.path.join(OUT, 'rq0_dailydialog.json'), 'w'), indent=1, ensure_ascii=False)
    lines = ['# RQ0 — DailyDialog에서 정서가 화행 분포를 얼마나 좁히는가', '',
             f"출처: {out['source']} · 대화 {out['n_dialogs']:,}개 · 발화 {out['n_utterances']:,}개", '',
             '| 분석 | n | I(bits) | H(act) | I/H(act) | 순열 99% | 부트스트랩 95% | Bayes acc \\| x | 다수결 acc |', '|---|---|---|---|---|---|---|---|---|']
    for r in res.values():
        lines.append(f"| {r['name']} | {r['n']:,} | {r['I_bits']:.4f} | {r['H_act_bits']:.3f} | {r['I_over_H_act']:.3%} | {r['I_perm_p99']:.4f} | [{r['I_boot_ci95'][0]:.4f}, {r['I_boot_ci95'][1]:.4f}] | {r['bayes_acc_given_x']:.3f} | {r['majority_acc']:.3f} |")
    lines += ['', '## p(act | emotion)', '', tab.to_markdown(), '', '## p(act | valence)', '', tab_v.to_markdown(), '', '## 발화 수', '', counts.to_markdown()]
    open(os.path.join(OUT, 'rq0_dailydialog.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
