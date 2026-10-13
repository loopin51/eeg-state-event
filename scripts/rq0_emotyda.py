"""RQ0 re-check on EMOTyDA (MELD + IEMOCAP with 12 dialogue-act tags and emotion).
Same estimator as rq0_dailydialog.py; finer act taxonomy than DailyDialog's 4.
Writes results/rq0_emotyda.{json,md}.
"""
import json, os, glob, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'data/raw/emotyda/EMOTyDA-master/Dataset')
OUT = os.path.join(ROOT, 'results')

DA_NAMES = {'g': 'greeting', 'q': 'question', 'ans': 'answer', 'o': 'statement-opinion', 's': 'statement-non-opinion',
            'ap': 'apology', 'c': 'command', 'ag': 'agreement', 'dag': 'disagreement', 'a': 'acknowledge',
            'b': 'backchannel', 'oth': 'others'}
VAL_MELD = {'joy': 1, 'anger': -1, 'disgust': -1, 'fear': -1, 'sadness': -1, 'surprise': 0, 'neutral': 0}
VAL_IEMO = {'hap': 1, 'exc': 1, 'ang': -1, 'fru': -1, 'sad': -1, 'fea': -1, 'dis': -1, 'sur': 0, 'neu': 0, 'oth': 0, 'xxx': 0}

def entropy(p): p = p[p > 0]; return float(-(p * np.log2(p)).sum())
def mi(x, y):
    ct = pd.crosstab(x, y).to_numpy().astype(float); p = ct / ct.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True); nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())
def analyse(x, y, name, n_perm=200, n_boot=200, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x); y = np.asarray(y)
    I = mi(x, y); Hy = entropy(pd.Series(y).value_counts(normalize=True).to_numpy())
    perm = np.array([mi(rng.permutation(x), y) for _ in range(n_perm)])
    idx = np.arange(len(x)); boot = np.array([mi(x[b], y[b]) for b in (rng.choice(idx, len(idx)) for _ in range(n_boot))])
    ct = pd.crosstab(x, y)
    return dict(name=name, n=int(len(x)), I_bits=I, H_act_bits=Hy, I_over_H_act=I / Hy,
                I_perm_p99=float(np.quantile(perm, .99)), I_boot_ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                bayes_acc_given_x=float(ct.max(axis=1).sum() / ct.to_numpy().sum()),
                majority_acc=float(pd.Series(y).value_counts(normalize=True).max()))

def load():
    m = pd.read_csv(os.path.join(D, 'MELD.csv'), encoding='latin1')
    meld = pd.DataFrame(dict(corpus='meld', emo=m['Emotion'].str.strip().str.lower(), act=m['Dialogue Act'].str.strip().str.lower()))
    meld['val'] = meld.emo.map(VAL_MELD)
    ie = pd.concat([pd.read_csv(f, encoding='latin1') for f in sorted(glob.glob(os.path.join(D, 'iemocap_session*.csv')))])
    iemo = pd.DataFrame(dict(corpus='iemocap', emo=ie['EMOTION'].str.strip().str.lower(), act=ie['DA'].str.strip().str.lower()))
    iemo['val'] = iemo.emo.map(VAL_IEMO)
    df = pd.concat([meld, iemo], ignore_index=True).dropna(subset=['emo', 'act'])
    df = df[df.act.isin(DA_NAMES)]
    return df

def main():
    df = load(); res = {}
    for corp in ['meld', 'iemocap', 'both']:
        d = df if corp == 'both' else df[df.corpus == corp]
        neutral = {'neutral', 'neu'}
        res[f'{corp}_emotion_all'] = analyse(d.emo, d.act, f'[{corp}] emotion -> act(12), all')
        e = d[~d.emo.isin(neutral | {'xxx', 'oth'})]
        res[f'{corp}_emotion_emotional'] = analyse(e.emo, e.act, f'[{corp}] emotion -> act(12), emotional only')
        v = d[d.val.isin([-1, 1])]
        res[f'{corp}_valence'] = analyse(v.val, v.act, f'[{corp}] valence -> act(12), pos/neg only')
    tab = pd.crosstab(df.emo, df.act.map(DA_NAMES), normalize='index').round(3)
    counts = pd.crosstab(df.emo, df.act.map(DA_NAMES))
    out = dict(source='EMOTyDA (Saha et al. 2020) MELD + IEMOCAP, GitHub sahatulika15/EMOTyDA', n_utterances=int(len(df)),
               n_meld=int((df.corpus == 'meld').sum()), n_iemocap=int((df.corpus == 'iemocap').sum()),
               act_marginal=df.act.map(DA_NAMES).value_counts(normalize=True).round(4).to_dict(),
               analyses=res, p_act_given_emotion=tab.to_dict(orient='index'), counts=counts.to_dict(orient='index'))
    json.dump(out, open(os.path.join(OUT, 'rq0_emotyda.json'), 'w'), indent=1, ensure_ascii=False)
    lines = ['# RQ0 재확인 — EMOTyDA (MELD + IEMOCAP, 대화행위 12종)', '', f"출처: {out['source']} · 발화 {out['n_utterances']:,}개 (MELD {out['n_meld']:,}, IEMOCAP {out['n_iemocap']:,})", '',
             '| 분석 | n | I(bits) | H(act) | I/H(act) | 순열 99% | 부트스트랩 95% | Bayes acc \\| x | 다수결 acc |', '|---|---|---|---|---|---|---|---|---|']
    for r in res.values():
        lines.append(f"| {r['name']} | {r['n']:,} | {r['I_bits']:.4f} | {r['H_act_bits']:.3f} | {r['I_over_H_act']:.3%} | {r['I_perm_p99']:.4f} | [{r['I_boot_ci95'][0]:.4f}, {r['I_boot_ci95'][1]:.4f}] | {r['bayes_acc_given_x']:.3f} | {r['majority_acc']:.3f} |")
    lines += ['', '## p(act | emotion)', '', tab.to_markdown(), '', '## 발화 수', '', counts.to_markdown()]
    open(os.path.join(OUT, 'rq0_emotyda.md'), 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines[:16]))

if __name__ == '__main__':
    main()
