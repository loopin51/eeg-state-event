"""F-VIII method B (D-117 amendments 1, 2, 5): generate-then-verify. The LLM (Qwen3-4B-Instruct-2507, vLLM) sees only candidate
sentences, never the EEG, z or the truth; every selection is cos(CLIP text(sentence), z).

Per row (condition, subject, test row) with predicted embedding z:
  start  = the n_start pool sentences with the largest cosine to z
  round r = 1..rounds: the LLM reads the current candidates (round 1: n_start, later the n_keep survivors) and writes n_new
           sentences (rewrite one / combine two, one main object). Variants that fail the truth-free filter
           (ese.things.lang.Vocab.single_object, CLIP context) or repeat a known sentence are dropped; survivors = the n_keep
           highest-cosine sentences of (current candidates + valid variants)
  output = round-`rounds` top-1; margin = cos(top-1) - cos(top-2)
Per-row LLM seed: sha256(condition | subject | row | seed number | round) (vLLM SamplingParams.seed).
k3s RTX 5080 (data group fv_langb). Output: results/runs/<date>_fv_lang_b_<part>/ b_<part>.jsonl (one line per row: output,
margin, final top-5 with cosines, per-round survivors and whether each survivor is an original pool sentence),
b_<part>_out.npy (CLIP text embedding of each output, row order of the jsonl), info.json.

usage: python scripts/fv_lang_b.py configs/fv_lang.yaml --part {main,shuffle,noise,phase,noeeg,sim,seeds12,report} [--smoke]
"""
import os
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')   # in-process engine (the spawned engine core fails to start on the k3s image)
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')      # no nvcc in the image for the FlashInfer JIT; PyTorch sampler, same settings
import sys, csv, json, time, re, hashlib, argparse, yaml, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def parse(txt):
    try:
        s = json.loads(txt[txt.find('{'): txt.rfind('}') + 1])['sentences']; return [x.strip() for x in s if isinstance(x, str) and x.strip()]
    except Exception:
        return [re.sub(r'^\s*(\d+[.)]|[-*])\s*', '', l).strip().strip('"') for l in txt.splitlines() if re.match(r'^\s*(\d+[.)]|[-*])\s+\S', l)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--part', required=True); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); c = yaml.safe_load(open(P(a.config))); b = c['b']; t0 = time.time()
    import torch, open_clip, spacy
    from vllm import LLM, SamplingParams
    from ese.things.lang import Vocab
    from ese.things import lang_rows as R
    from fv_lang_embed import load_text_encoder, encode, fsha
    for f, h in b['llm_sha256'].items(): assert fsha(os.path.join(P(b['llm']), f)) == h, f'LLM weights sha256 {f}'
    assert spacy.__version__ == c['parser']['spacy']; nlp = spacy.load('en_core_web_lg'); assert f"en_core_web_lg-{nlp.meta['version']}" == c['parser']['model']
    V = Vocab(nlp, P(c['names_table']), P(c['things_meta']))
    pf = os.path.join(P(c['pool']['out_dir']), 'pool.tsv')
    if a.smoke and not os.path.exists(pf):                         # code smoke before the pool exists: template sentences, never reported
        pool = [dict(sentence=t.format(r['display_name'])) for r in csv.DictReader(open(P(c['names_table']), encoding='utf-8'), delimiter='\t')
                for t in ('A photo of a {}.', 'A {} on a plain background.', 'A close-up photo of a {}.', 'A {} seen from above.', 'A small {} outdoors.')]
    else: pool = list(csv.DictReader(open(pf, encoding='utf-8'), delimiter='\t'))
    pool_txt = [r['sentence'] for r in pool]; pool_set = set(pool_txt)
    name = time.strftime('%Y-%m-%d') + f'_fv_lang_b_{a.part}' + ('_smoke' if a.smoke else ''); out = os.path.join(ROOT, 'results', 'runs', name); os.makedirs(out, exist_ok=True)
    clip, tok = load_text_encoder(c, torch, open_clip); ctx = c['clip']['context']
    PE = encode(clip, tok, pool_txt, torch, c['clip']['batch']).astype(np.float64)
    test_img = np.load(P(c['test_image'])); rows_tab = list(csv.DictReader(open(P(c['names_table']), encoding='utf-8'), delimiter='\t'))
    names_test = np.load(P(c['names_text']))[[i for i, r in enumerate(rows_tab) if r['split'] == 'test']]
    # test-name rows in test-code order are not needed here: sim_planted only needs the 200 test names as a set of candidates
    jobs = []
    for cond in b['parts'][a.part]:
        for s, k, z in R.rows(cond, c, ROOT, test_img=test_img, names_test=names_test): jobs.append((cond, s, k, z))
    if a.smoke: jobs = jobs[:16]
    Z = np.stack([j[3] for j in jobs]); print(f'{len(jobs)} rows, conditions {b["parts"][a.part]} ({time.time() - t0:.0f}s)', flush=True)
    llm = LLM(model=P(b['llm']), dtype='bfloat16', seed=0, **b['vllm']); tz = llm.get_tokenizer()
    ok_cache = {}

    def valid(s):
        if s not in ok_cache:
            n = len(tok.tokenizer.encode(s)) + 2 if hasattr(tok, 'tokenizer') else len(tok.encode(s)) + 2
            ok_cache[s] = n <= ctx and V.single_object(s)
        return ok_cache[s]

    S0 = Z @ PE.T; cand = [list(np.argsort(-S0[i])[:b['n_start']]) for i in range(len(jobs))]
    cur = [[(pool_txt[j], float(S0[i, j])) for j in cand[i]] for i in range(len(jobs))]
    hist = [[] for _ in jobs]; sp = b['sampling']; n_bad = n_new = 0
    for r in range(1, b['rounds'] + 1):
        t1 = time.time()
        msgs = [[{'role': 'user', 'content': b['prompt'].format(candidates='\n'.join(f'{q + 1}. {x}' for q, (x, _) in enumerate(cur[i])), n=b['n_new'])}] for i in range(len(jobs))]
        params = [SamplingParams(temperature=sp['temperature'], top_p=sp['top_p'], top_k=sp['top_k'], max_tokens=sp['max_tokens'],
                                 seed=R.row_seed(cond, s, k, r)) for cond, s, k, _ in jobs]
        res = llm.chat(msgs, params, use_tqdm=False)
        newv = []
        for i, o in enumerate(res):
            known = {x for x, _ in cur[i]}; vs = []
            for x in parse(o.outputs[0].text)[:b['n_new']]:
                if x in known or x in vs: continue
                if valid(x): vs.append(x)
                else: n_bad += 1
            newv.append(vs); n_new += len(vs)
        flat = sorted({x for vs in newv for x in vs}); E = {x: e for x, e in zip(flat, encode(clip, tok, flat, torch, c['clip']['batch']).astype(np.float64))}
        for i, vs in enumerate(newv):
            allc = cur[i] + [(x, float(E[x] @ Z[i])) for x in vs]; allc.sort(key=lambda t: -t[1]); cur[i] = allc[:b['n_keep']]
            hist[i].append([[x, round(sc, 5), x in pool_set] for x, sc in cur[i]])
        print(f'round {r}: {time.time() - t1:.0f}s, valid variants {n_new}, filtered {n_bad}', flush=True)
    with open(os.path.join(out, f'b_{a.part}.jsonl'), 'w') as f:
        for (cond, s, k, _), cu, h in zip(jobs, cur, hist):
            f.write(json.dumps(dict(cond=cond, subject=s, row=k, output=cu[0][0], margin=cu[0][1] - cu[1][1] if len(cu) > 1 else None,
                                    top=[[x, sc] for x, sc in cu], rounds=h), ensure_ascii=False) + '\n')
    np.save(os.path.join(out, f'b_{a.part}_out.npy'), encode(clip, tok, [cu[0][0] for cu in cur], torch, c['clip']['batch']))   # M3 input
    info = dict(part=a.part, n_rows=len(jobs), valid_variants=n_new, filtered_variants=n_bad, seconds=round(time.time() - t0, 1),
                device=torch.cuda.get_device_name(0), torch=torch.__version__, vllm=__import__('vllm').__version__, spacy=spacy.__version__,
                pool_sha256=fsha(pf) if os.path.exists(pf) else 'smoke-template', vocab_sha256=V.sha, max_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
                output_sha256=fsha(os.path.join(out, f'b_{a.part}.jsonl')))
    json.dump(info, open(os.path.join(out, 'info.json'), 'w'), indent=1); print('SUMMARY', json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
