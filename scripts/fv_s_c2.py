"""F-IX (D-119 section 3, amendment 1 I) method C2: frozen decoder output z~ (1,024) -> one linear projection (1,024 -> n_tok x d_llm)
-> frozen LLM (Qwen3-4B-Instruct-2507, bf16) -> one sentence. Only the projection is trained, on text only: input of pool sentence j
= unit(a * x_j + r), x_j = unit(M t_j) (ridge map from the sentence's CLIP text embedding to the centred image space, fitted on
training concepts, fv_s.py --stage c2prep), r from the development residual model (concept level), a new draw per step; target =
the sentence. k3s RTX 5080 (Kueue team-b-lq).

  --stage train : max(steps) steps, batch 32, AdamW lr 1e-3; at every candidate step the projection is saved and the development
                  concept rows (10 subjects x 200, inputs file) are decoded (greedy, max_new_tokens) -> c2_dev.jsonl
  --stage test  : the chosen step's projection (--proj, from the train run) decodes every test condition of the inputs file
                  -> c2_test.jsonl. Run only after the step is chosen on the development rows (fv_s_c2_judge.py --stage dev).
Output: results/runs/<date>_fv_s_c2_{train,test}[_smoke]/ config.yaml, history.json, metrics.json, *.jsonl, proj_s<step>.pt
usage: python scripts/fv_s_c2.py configs/fv_s.yaml --stage {train,test} [--proj PATH --step N] [--smoke]
"""
import os, sys, json, time, argparse, yaml, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--stage', choices=['train', 'test'], required=True)
    ap.add_argument('--proj'); ap.add_argument('--step', type=int); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    c = yaml.safe_load(open(P(a.config))); cc = c['c2']; lc = yaml.safe_load(open(P(c['lang_config']))); t0 = time.time(); dev = 'cuda'
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fv_lang_embed import fsha
    llm_dir = P(lc['c']['llm'])
    for f, h in lc['c']['llm_sha256'].items(): assert fsha(os.path.join(llm_dir, f)) == h, f'LLM sha256 {f}'
    inf = P(cc['inputs']); Z = np.load(inf); assert fsha(inf) == open(inf + '.sha256').read().split()[0], 'inputs sha256'
    tokz = AutoTokenizer.from_pretrained(llm_dir); llm = AutoModelForCausalLM.from_pretrained(llm_dir, dtype=torch.bfloat16).to(dev).eval()
    for p in llm.parameters(): p.requires_grad_(False)
    emb = llm.get_input_embeddings(); d_llm = emb.weight.shape[1]; nt = cc['n_tok']
    txt = tokz.apply_chat_template([{'role': 'user', 'content': '<EEG>\n' + lc['c']['instruction']}], tokenize=False, add_generation_prompt=True)
    pre, post = txt.split('<EEG>'); pre_ids = tokz(pre, add_special_tokens=False).input_ids; post_ids = tokz(post, add_special_tokens=False).input_ids
    end_ids = tokz('<|im_end|>', add_special_tokens=False).input_ids; eos = tokz.convert_tokens_to_ids('<|im_end|>')
    torch.manual_seed(cc['seed']); rng = np.random.default_rng(cc['seed'])
    proj = nn.Linear(Z['sent_x'].shape[1], nt * d_llm).to(dev)
    name = time.strftime('%Y-%m-%d') + f'_fv_s_c2_{a.stage}' + ('_smoke' if a.smoke else ''); out = os.path.join(ROOT, 'results', 'runs', name); os.makedirs(out, exist_ok=True)
    yaml.safe_dump(dict(c, run=dict(stage=a.stage, smoke=a.smoke, proj=a.proj, step=a.step, d_llm=d_llm, device=torch.cuda.get_device_name(0), inputs_sha256=fsha(inf))), open(os.path.join(out, 'config.yaml'), 'w'))

    def tokens(x):
        return proj(x.float()).view(len(x), nt, d_llm).to(torch.bfloat16)

    def generate(X, bs):
        res = []
        for b in range(0, len(X), bs):
            xb = torch.from_numpy(np.asarray(X[b:b + bs], np.float32)).to(dev); B = len(xb)
            with torch.no_grad():
                E = tokens(xb); pe = emb(torch.tensor([pre_ids], device=dev)).expand(B, -1, -1); po = emb(torch.tensor([post_ids], device=dev)).expand(B, -1, -1)
                g = llm.generate(inputs_embeds=torch.cat([pe, E, po], 1), attention_mask=torch.ones((B, len(pre_ids) + nt + len(post_ids)), dtype=torch.long, device=dev),
                                 max_new_tokens=cc['max_new_tokens'], do_sample=False, pad_token_id=tokz.pad_token_id)
                for s_ in g.tolist():                                           # inputs_embeds only: every returned token is new
                    n = s_.index(eos) if eos in s_ else len(s_); res.append(tokz.decode(s_[:n], skip_special_tokens=True).strip())
        return res

    if a.stage == 'test':
        assert a.proj and a.step is not None, '--proj and --step (chosen on the development rows)'
        proj.load_state_dict(torch.load(P(a.proj), map_location=dev)); conds = [k[len('test_'):] for k in Z.files if k.startswith('test_')]; rows = []
        llm.config.use_cache = True
        for cn in conds:
            X = Z[f'test_{cn}']; ns, nk = X.shape[:2]; gen = generate(X.reshape(ns * nk, -1)[:64] if a.smoke else X.reshape(ns * nk, -1), cc['gen_batch'])
            for i, s_ in enumerate(gen): rows.append(dict(cond=cn, subject=int(Z['subjects'][i // nk]), row=int(i % nk) if nk > 1 else -1, output=s_))
            print(f'test {cn}: {len(gen)} rows, e.g. {gen[0]!r} ({time.time() - t0:.0f}s)', flush=True)
        with open(os.path.join(out, 'c2_test.jsonl'), 'w') as f:
            for x in rows: f.write(json.dumps(x, ensure_ascii=False) + '\n')
        json.dump(dict(stage='test', step=a.step, proj=a.proj, proj_sha256=fsha(P(a.proj)), c2_test_sha256=fsha(os.path.join(out, 'c2_test.jsonl')), seconds=time.time() - t0), open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
        return
    # train
    if cc.get('grad_checkpointing'): llm.gradient_checkpointing_enable(); llm.config.use_cache = False; llm.train()
    sx = torch.from_numpy(Z['sent_x'].astype(np.float32)).to(dev); sents = [str(s) for s in Z['sentences']]; a_ = float(Z['a']); m_ = torch.from_numpy(Z['m'].astype(np.float32)).to(dev)
    Lc = torch.from_numpy(Z['L'].astype(np.float32)).to(dev); ids = [tokz(s, add_special_tokens=False).input_ids + end_ids for s in sents]
    opt = torch.optim.AdamW(proj.parameters(), lr=cc['lr'], weight_decay=cc['weight_decay']); steps = sorted(cc['steps']); smax = 20 if a.smoke else steps[-1]; hist = []

    def nll(E, ans):
        B = len(ans); L = len(pre_ids) + nt + len(post_ids) + max(len(q) for q in ans); a0 = len(pre_ids) + nt + len(post_ids)
        inp = torch.full((B, L), tokz.pad_token_id or 0, dtype=torch.long, device=dev); lab = torch.full((B, L), -100, dtype=torch.long, device=dev); att = torch.zeros((B, L), dtype=torch.long, device=dev)
        for i, q in enumerate(ans):
            inp[i, :len(pre_ids)] = torch.tensor(pre_ids); inp[i, len(pre_ids) + nt:a0] = torch.tensor(post_ids); inp[i, a0:a0 + len(q)] = torch.tensor(q); lab[i, a0:a0 + len(q)] = torch.tensor(q); att[i, :a0 + len(q)] = 1
        e = emb(inp).clone(); e[:, len(pre_ids):len(pre_ids) + nt] = E
        h = llm.model(inputs_embeds=e, attention_mask=att).last_hidden_state[:, :-1]; y = lab[:, 1:]; mk = y != -100
        return F.cross_entropy(llm.lm_head(h[mk]).float(), y[mk], reduction='sum'), int(mk.sum())
    dev_rows = []; t1 = time.time(); torch.cuda.reset_peak_memory_stats()
    for step in range(1, smax + 1):
        j = rng.integers(0, len(sents), cc['batch']); jt = torch.from_numpy(j).to(dev)
        x = F.normalize(a_ * sx[jt] + m_ + torch.randn(len(j), sx.shape[1], device=dev) @ Lc.T, dim=-1); ans = [ids[k] for k in j]; n = sum(len(q) for q in ans)
        opt.zero_grad(set_to_none=True); E = tokens(x); Ed = E.detach().requires_grad_(True); tot = 0.0
        for ch in np.array_split(np.arange(len(j)), cc['micro_batches']):
            t, _ = nll(Ed[torch.as_tensor(ch, device=dev)], [ans[i] for i in ch]); (t / n).backward(); tot += t.item()
        E.backward(Ed.grad); torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0); opt.step()
        if step % 50 == 0 or step == 1: hist.append(dict(step=step, loss=tot / n, seconds=time.time() - t1)); print(f'step {step} loss {tot / n:.4f} ({time.time() - t1:.0f}s, max {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)', flush=True)
        if step in steps or (a.smoke and step == smax):
            pf = os.path.join(out, f'proj_s{step}.pt'); torch.save(proj.state_dict(), pf); llm.eval(); llm.config.use_cache = True
            X = Z['dev_z']; ns, nk = X.shape[:2]; flat = X.reshape(ns * nk, -1); gen = generate(flat[:64] if a.smoke else flat, cc['gen_batch'])
            for i, s_ in enumerate(gen): dev_rows.append(dict(step=step, subject=int(Z['subjects'][i // nk]), concept=int(i % nk), output=s_))
            print(f'dev step {step}: {len(gen)} rows, e.g. {gen[0]!r}', flush=True)
            if cc.get('grad_checkpointing'): llm.config.use_cache = False; llm.train()
    with open(os.path.join(out, 'c2_dev.jsonl'), 'w') as f:
        for x in dev_rows: f.write(json.dumps(x, ensure_ascii=False) + '\n')
    json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
    json.dump(dict(stage='train', steps=steps, seconds=time.time() - t0, max_memory_gb=torch.cuda.max_memory_allocated() / 1e9, device=torch.cuda.get_device_name(0),
                   c2_dev_sha256=fsha(os.path.join(out, 'c2_dev.jsonl')), proj_sha256={s: fsha(os.path.join(out, f'proj_s{s}.pt')) for s in steps if os.path.exists(os.path.join(out, f'proj_s{s}.pt'))}),
              open(os.path.join(out, 'metrics.json'), 'w'), indent=1); print('done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
