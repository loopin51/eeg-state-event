"""Direction 3, step 2: context-conditional candidate sets for the faithfulness device (D-086).

Utterances are normalised to standard spacing first (detok; DailyDialog spaces punctuation and apostrophes, the LM does not).
One trial per DailyDialog dialogue (test split; validation split for the calibration pilot): a random eligible turn t is the
target, the (at most max_context_turns) turns before it are the context. A fixed public LM samples continuations of the
context for the target speaker; alternatives are accepted in sampling order if their word count matches the target
(|w - w_t| <= max(a, b * w_t)) and at most max_one_side of the 19 are strictly shorter (longer) than the target, their e5 residual cosine (D-076 metric, D-075 length regression) to the target is below
max_resid_cos_target and to every accepted alternative below max_resid_cos_pair. 19 alternatives + the target, shuffled
with a per-trial seed, form the candidate set. Residual cosine distances to the target are stratified into pooled
quartiles. The candidate file is hashed in a manifest before any device output is generated (D-086).

usage: python scripts/d3_candidates.py configs/d3_candidates.yaml --split test|validation [--n 30] [--tag pilot]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, time, hashlib, argparse, re, yaml, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d3_metric_gate import Embedder, nwords, len_feats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def detok(s):
    """DailyDialog spacing -> standard text (the LM alternatives have none of it): curly apostrophes, "I ' ll", " ." etc."""
    s = s.replace('\u2019', "'").replace('\u2018', "'").replace('\u201c', '"').replace('\u201d', '"')
    s = re.sub(r"(\w) ' (\w)", r"\1'\2", s); s = re.sub(r"\s+([.,!?;:%)\]])", r"\1", s); s = re.sub(r"([(\[$])\s+", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def trials(c, split, n, rng):
    d = pd.read_parquet(os.path.join(ROOT, c['data']['dailydialog_dir'], split, '0000.parquet')); t = c['trials']; out = []
    for di, dl in enumerate(d['dialog']):
        dl = [detok(u) for u in dl]
        ok = [i for i in range(t['min_turn'], len(dl)) if dl[i] and t['target_words'][0] <= nwords(dl[i]) <= t['target_words'][1]]
        if not ok: continue
        i = int(rng.choice(ok)); ctx = dl[max(0, i - t['max_context_turns']):i]
        spk = ['A' if (j % 2 == i % 2) else 'B' for j in range(i - len(ctx), i)]
        out.append(dict(dialogue=di, turn=i, context=ctx, speakers=spk, target=dl[i], target_speaker='A'))
    if n: out = [out[k] for k in sorted(rng.choice(len(out), n, replace=False))]
    return out


def prompt(tok, tr):
    lines = '\n'.join(f'{s}: {u}' for s, u in zip(tr['speakers'], tr['context']))
    msg = (f'Here is a conversation between two speakers, A and B.\n\n{lines}\n\n'
           f'Write the next thing speaker A says. Reply with that one utterance only, about {nwords(tr["target"])} words, '
           'without the speaker label or quotation marks.')
    return tok.apply_chat_template([{'role': 'user', 'content': msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def clean(s):
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.S).strip().split('\n')[0].strip()
    s = re.sub(r'^(speaker\s*)?A\s*:\s*', '', s, flags=re.I).strip().strip('"').strip()
    return detok(s)


def load_lm(spec):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(spec['model'], padding_side='left')
    m = AutoModelForCausalLM.from_pretrained(spec['model'], dtype=getattr(torch, spec['dtype'])).cuda().eval()
    return tok, m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--split', required=True, choices=['test', 'validation'])
    ap.add_argument('--n', type=int, default=0); ap.add_argument('--tag', default=''); a = ap.parse_args()
    c = yaml.safe_load(open(a.config)); rng = np.random.default_rng(c['seed']); t0 = time.time(); f = c['filter']; L = c['lm']
    out = os.path.join(ROOT, c['out_dir'] + (f'_{a.tag}' if a.tag else '')); os.makedirs(out, exist_ok=True)
    T = trials(c, a.split, a.n, rng); print(f'{len(T)} trials ({a.split})', flush=True)
    fit = list(pd.read_csv(os.path.join(ROOT, c['data']['resid_fit']), sep='\t', keep_default_na=False).text)
    E = Embedder(c['embedder'], 128, fit); E.m.to('cpu'); torch.cuda.empty_cache()   # embedder on CPU: GPU memory goes to the LM
    def resid(texts):
        R = E.raw(texts) - len_feats(texts) @ E.W; return R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    tok, m = load_lm(L); torch.manual_seed(c['seed']); print(f'LM loaded ({time.time() - t0:.0f}s), cuda max {torch.cuda.max_memory_allocated() / 2**30:.2f} GB', flush=True)
    acc = [[] for _ in T]; stats = [dict(sampled=0, empty=0, length=0, balance=0, para=0, dup=0) for _ in T]; Rt = resid([tr['target'] for tr in T])
    for rnd in range(L['max_rounds']):
        todo = [i for i in range(len(T)) if len(acc[i]) < f['n_alternatives']]
        if not todo: break
        for b0 in range(0, len(todo), L['batch_contexts']):
            idx = todo[b0:b0 + L['batch_contexts']]; enc = tok([prompt(tok, T[i]) for i in idx], return_tensors='pt', padding=True).to('cuda')
            with torch.no_grad():
                g = m.generate(**enc, do_sample=True, temperature=L['temperature'], top_p=L['top_p'], max_new_tokens=L['max_new_tokens'],
                               num_return_sequences=L['samples_per_round'], pad_token_id=tok.pad_token_id)
            txt = tok.batch_decode(g[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
            for k, i in enumerate(idx):
                wt = nwords(T[i]['target']); tol = max(f['len_tol_words'][0], f['len_tol_words'][1] * wt)
                cands = [clean(s) for s in txt[k * L['samples_per_round']:(k + 1) * L['samples_per_round']]]; st = stats[i]; st['sampled'] += len(cands)
                st['empty'] += sum(not s for s in cands); cands = [s for s in cands if s]
                ok = [s for s in cands if abs(nwords(s) - wt) <= tol]; st['length'] += len(cands) - len(ok)
                if not ok: continue
                R = resid(ok); seen = {x.lower() for x in acc[i]} | {T[i]['target'].lower()}
                Ra = resid(acc[i]) if acc[i] else np.zeros((0, R.shape[1]))
                for s, r in zip(ok, R):
                    if len(acc[i]) >= f['n_alternatives']: break
                    if s.lower() in seen or (len(Ra) and (Ra @ r).max() >= f['max_resid_cos_pair']): st['dup'] += 1; continue
                    if r @ Rt[i] >= f['max_resid_cos_target']: st['para'] += 1; continue
                    side = np.sign(nwords(s) - wt)          # length balance: at most max_one_side alternatives strictly shorter / longer
                    if side and sum(np.sign(nwords(x) - wt) == side for x in acc[i]) >= f['max_one_side']: st['balance'] += 1; continue
                    acc[i].append(s); seen.add(s.lower()); Ra = np.vstack([Ra, r])
        print(f'round {rnd}: {sum(len(x) >= f["n_alternatives"] for x in acc)}/{len(T)} complete ({time.time() - t0:.0f}s), '
              f'cuda max {torch.cuda.max_memory_allocated() / 2**30:.2f} GB', flush=True)
    rows, dists, lc_rr, cen_pct = [], [], [], []
    for i, tr in enumerate(T):
        if len(acc[i]) < f['n_alternatives']: continue
        r = np.random.default_rng([c['seed'], tr['dialogue']]); cand = [tr['target']] + acc[i]; perm = r.permutation(len(cand))
        cand = [cand[j] for j in perm]; ti = int(np.where(perm == 0)[0][0]); Rc = resid(cand); dist = (1 - Rc @ Rc[ti]).tolist()
        C = Rc @ Rc.T; np.fill_diagonal(C, np.nan); cen = np.nanmean(C, 1); o = np.delete(cen, ti)   # reporting baseline (not a verdict):
        lc_rr.append(1 / (1 + (o < cen[ti]).sum() + 0.5 * (o == cen[ti]).sum()))             # pick the least central candidate
        cen_pct.append(((o < cen[ti]).sum() + 0.5 * (o == cen[ti]).sum()) / len(o))
        rows.append(dict(trial=len(rows), **{k: tr[k] for k in ('dialogue', 'turn', 'context', 'speakers', 'target')}, candidates=cand, target_index=ti,
                         resid_dist=dist, gen_stats=stats[i])); dists += [d for j, d in enumerate(dist) if j != ti]
    edges = np.quantile(dists, [0, .25, .5, .75, 1]).tolist() if dists else []
    for rw in rows: rw['stratum'] = [None if j == rw['target_index'] else int(np.clip(np.searchsorted(edges, d, 'right') - 1, 0, 3)) for j, d in enumerate(rw['resid_dist'])]
    path = os.path.join(out, f'candidates_{a.split}.jsonl')
    with open(path, 'w') as fh:
        for rw in rows: fh.write(json.dumps(rw, ensure_ascii=False) + '\n')
    tot = {k: int(sum(s[k] for s in stats)) for k in stats[0]}
    man = dict(split=a.split, n_trials_eligible=len(T), n_trials=len(rows), dropped=len(T) - len(rows), quartile_edges=edges, gen_stats=tot,
               baseline_least_central_rr=float(np.mean(lc_rr)) if rows else None, chance_rr=float(np.mean(1 / np.arange(1, f['n_alternatives'] + 2))),
               target_centrality_pct=float(np.mean(cen_pct)) if rows else None,
               target_words_mean=float(np.mean([nwords(r['target']) for r in rows])) if rows else None,
               alt_words_mean=float(np.mean([nwords(s) for r in rows for j, s in enumerate(r['candidates']) if j != r['target_index']])) if rows else None,
               lm=L, filter=f, seed=c['seed'], cuda_max_gb=torch.cuda.max_memory_allocated() / 2**30, seconds=time.time() - t0,
               sha256=hashlib.sha256(open(path, 'rb').read()).hexdigest())
    json.dump(man, open(os.path.join(out, f'manifest_{a.split}.json'), 'w'), indent=1); print(json.dumps(man, indent=1), flush=True)


if __name__ == '__main__':
    main()
