"""F-VIII method C (D-117 corrections 1-2, amendments 1, 5, 6): EEG encoder tokens -> one linear projection -> frozen LLM input.

One LOSO fold (target subject), as the F-V final runs:
  encoder   F-V BrainModule from the same fold's final run (seed 20261811; --shuffle: that seed's label-shuffle run), body output
            [B, 256, T] -> adaptive average pooling into 10 time bins -> 10 tokens -> Linear(256, LLM hidden) -> the user turn of the
            chat template "<10 EEG tokens>\nDescribe the photo in one sentence." -> assistant turn = one pool sentence of the
            image's concept (--shuffle: of pi(concept), fv_train's concept permutation). Loss on the answer tokens only.
  stage 1   encoder (with the trainers' subject layers) + projection on the nine trainers' image means (all sessions, all training
            concepts minus the validation concepts); one pool sentence drawn per item and step. Validation (valid_c, the
            trainers' items, the five sentences of each concept in turn) every `eval_every` steps, early stopping by patience.
  stage 2   the target's subject layer only (D-090), from the trainers' mean layer, on the target's sessions {1}, {1,2}, {1,2,3}
            image means (validation concepts for early stopping); k0 = the mean layer.
  generate  greedy decoding (max_new_tokens) on the test inputs (verdict inputs, session 4): per calibration condition and input
            kind; plus the no-EEG input (the subject's mean avg20 input, one row). Confidence = mean log-probability of the
            generated tokens.
--stage measure: memory and step time on a few batches (no checkpoint, no generation; decides the LLM by amendment 5).
Output: results/runs/<date>_fv_lang_c_t<TT>[_shuffle][_smoke]/ config.yaml, history.json, metrics.json, c_out.jsonl (cond,
subject, row, output, confidence), proj.pt + encoder.pt (git-ignored).

usage: python scripts/fv_lang_c.py configs/fv_lang.yaml --target S [--shuffle] [--stage {measure,train}] [--smoke]
"""
import os, sys, csv, glob, json, time, copy, argparse, yaml, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
from ese.things.data import Meta, subject_mean, GPUSet
from ese.things.models import build


class Tokens(nn.Module):
    def __init__(self, enc, d_llm, n_bins):
        super().__init__(); self.enc = enc; self.pool = nn.AdaptiveAvgPool1d(n_bins); self.proj = nn.Linear(enc.proj.in_features // n_bins, d_llm)

    def forward(self, x, pos, subj):
        return self.proj(self.pool(self.enc.body(x, pos, subj)).transpose(1, 2))       # [B, n_bins, d_llm]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--target', type=int, required=True); ap.add_argument('--shuffle', action='store_true')
    ap.add_argument('--stage', choices=['measure', 'train'], default='train'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--grad-ckpt', action='store_true', help='measurement of the amendment-5 fallback (LLM gradient checkpointing)')
    a = ap.parse_args(); c = yaml.safe_load(open(P(a.config))); cc = c['c']
    if a.grad_ckpt: cc['grad_checkpointing'] = True
    t0 = time.time(); dev = 'cuda'
    tc = yaml.safe_load(open(P(cc['train_config']))); pc = yaml.safe_load(open(P(tc['prep_config']))); cache = P(pc['cache_dir'])
    torch.manual_seed(cc['seed']); rng = np.random.default_rng(cc['seed'])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fv_lang_embed import fsha
    llm_dir = P(cc['llm'])
    for f, h in cc['llm_sha256'].items(): assert fsha(os.path.join(llm_dir, f)) == h, f'LLM sha256 {f}'
    tokz = AutoTokenizer.from_pretrained(llm_dir); llm = AutoModelForCausalLM.from_pretrained(llm_dir, dtype=torch.bfloat16).to(dev).eval()
    for p in llm.parameters(): p.requires_grad_(False)
    if cc.get('grad_checkpointing'): llm.gradient_checkpointing_enable(); llm.config.use_cache = False; llm.train()   # dropout-free model: train() only switches checkpointing on
    emb = llm.get_input_embeddings(); d_llm = emb.weight.shape[1]
    # chat template pieces around the EEG tokens
    txt = tokz.apply_chat_template([{'role': 'user', 'content': '<EEG>\n' + cc['instruction']}], tokenize=False, add_generation_prompt=True)
    pre, post = txt.split('<EEG>'); pre_ids = tokz(pre, add_special_tokens=False).input_ids; post_ids = tokz(post, add_special_tokens=False).input_ids
    end_ids = tokz('<|im_end|>', add_special_tokens=False).input_ids
    # F-V final run of this fold
    suf = '_shuffle' if a.shuffle else ''; runs = sorted(glob.glob(cc['final_glob'].format(seed=cc['final_seed'], tt=f'{a.target:02d}', suffix=suf)))
    assert len(runs) == 1, runs; rd = runs[0]; rc = yaml.safe_load(open(os.path.join(rd, 'config.yaml'))); r = rc['run']
    assert r['target'] == a.target and r['mode'] == 'final' and r['shuffle'] == a.shuffle and r['seed'] == cc['final_seed']
    trainers = r['trainers']; ti = len(trainers); enc = build('bm', ti + 1, rc['model']); enc.load_state_dict(torch.load(os.path.join(rd, 'shared.pt'))['net'])
    layers0 = torch.load(os.path.join(rd, 'layers.pt')); net = Tokens(enc, d_llm, rc['model']['bm']['n_bins']).to(dev)
    # concepts, sentences
    meta = Meta(P(tc['meta'])); dev_c = meta.dev_concepts(tc['dev']['salt'], tc['dev']['n']); valid_c = meta.valid_concepts(dev_c, tc['valid']['salt'], tc['valid']['fraction'])   # as fv_train (both modes)
    assert r['n_valid_concepts'] == len(valid_c), 'validation concepts differ from the final run'
    train_c = [x for x in meta.train_concepts if x not in set(valid_c)]; train_codes, valid_codes = meta.codes_of(train_c), meta.codes_of(valid_c)
    names = list(csv.DictReader(open(P(c['names_table']), encoding='utf-8'), delimiter='\t')); pf = os.path.join(P(c['pool']['out_dir']), 'pool.tsv')
    if (a.smoke or a.stage == 'measure') and not os.path.exists(pf):        # before the pool exists: template sentences, never reported
        pool = [dict(concept=x['concept'], sentence=t.format(x['display_name'])) for x in names
                for t in ('A photo of a {}.', 'A {} on a plain background.', 'A close-up photo of a {}.', 'A {} seen from above.', 'A small {} outdoors.')]
    else: pool = list(csv.DictReader(open(pf, encoding='utf-8'), delimiter='\t'))
    folder2concept = {x['folder']: x['concept'] for x in names if x['split'] == 'train'}
    sents = {}
    for p in pool: sents.setdefault(p['concept'], []).append(p['sentence'])
    concept_sents = [sents[folder2concept[f]] for f in meta.train_concepts]               # training-concept index -> its 5 sentences
    if a.shuffle:
        from fv_train import shuffled_rows
        _, pi = shuffled_rows(meta, tc['shuffle_seed']); concept_sents = [concept_sents[pi[j]] for j in range(len(concept_sents))]
    tok_cache = {}

    def ids(s):
        if s not in tok_cache: tok_cache[s] = tokz(s, add_special_tokens=False).input_ids + end_ids
        return tok_cache[s]

    def lm_loss(x, pos, subj, codes, pick):
        """mean answer-token NLL; pick(code) -> sentence"""
        E = net(x, pos, subj).to(torch.bfloat16); B, nb, _ = E.shape; ans = [ids(pick(int(k))) for k in codes.tolist()]
        L = len(pre_ids) + nb + len(post_ids) + max(len(q) for q in ans)
        inp = torch.full((B, L), tokz.pad_token_id or 0, dtype=torch.long, device=dev); lab = torch.full((B, L), -100, dtype=torch.long, device=dev)
        att = torch.zeros((B, L), dtype=torch.long, device=dev); a0 = len(pre_ids) + nb + len(post_ids)
        for i, q in enumerate(ans):
            inp[i, :len(pre_ids)] = torch.tensor(pre_ids); inp[i, len(pre_ids) + nb:a0] = torch.tensor(post_ids); inp[i, a0:a0 + len(q)] = torch.tensor(q)
            lab[i, a0:a0 + len(q)] = torch.tensor(q); att[i, :a0 + len(q)] = 1
        e = emb(inp).clone(); e[:, len(pre_ids):len(pre_ids) + nb] = E
        h = llm.model(inputs_embeds=e, attention_mask=att).last_hidden_state[:, :-1]; y = lab[:, 1:]; m = y != -100   # lm_head on answer positions only
        return F.cross_entropy(llm.lm_head(h[m]).float(), y[m])

    ci = meta.img_concept; samp = lambda k: concept_sents[ci[k - 1]][rng.integers(len(concept_sents[ci[k - 1]]))]
    name = time.strftime('%Y-%m-%d') + f'_fv_lang_c_t{a.target:02d}{suf}' + ('_measure' if a.stage == 'measure' else '') + ('_ckpt' if a.grad_ckpt else '') + ('_smoke' if a.smoke else '')
    out = os.path.join(ROOT, 'results', 'runs', name); os.makedirs(out, exist_ok=True)
    yaml.safe_dump(dict(c, run=dict(target=a.target, shuffle=a.shuffle, stage=a.stage, smoke=a.smoke, final_run=os.path.basename(rd), trainers=trainers,
                                    d_llm=d_llm, device=torch.cuda.get_device_name(0))), open(os.path.join(out, 'config.yaml'), 'w'))
    pos = np.load(P(tc['positions'])); blocks, vblocks = [], []
    use = trainers[:2] if (a.smoke or a.stage == 'measure') else trainers
    for i, s in enumerate(use):
        codes, X, _ = subject_mean(cache, s, tc['sessions'], codes=np.r_[train_codes, valid_codes]); m = np.isin(codes, valid_codes)
        blocks.append((trainers.index(s), codes[~m], X[~m])); vblocks.append((trainers.index(s), codes[m], X[m])); del X
    TR, VA = GPUSet(blocks, pos, dev), GPUSet(vblocks, pos, dev); del blocks, vblocks
    bs = cc['batch']; hist = []
    vidx = torch.arange(len(VA), device=dev); vpick = lambda k: concept_sents[ci[k - 1]][(k - 1) % len(concept_sents[ci[k - 1]])]   # fixed sentence per item

    def vloss(S, idx_all):
        net.eval(); tot = []
        with torch.no_grad():
            for b in range(0, len(idx_all), bs):
                x, p, sj, _, _, code, _ = S.batch(idx_all[b:b + bs]); tot.append(lm_loss(x, p, sj, code, vpick).item())
        net.train(); return float(np.mean(tot))
    groups = [{'params': [p for n, p in net.named_parameters() if n.startswith('proj')], 'lr': cc['lr_proj']},
              {'params': [p for n, p in net.named_parameters() if not n.startswith('proj')], 'lr': cc['lr_enc']}]
    opt = torch.optim.AdamW(groups, weight_decay=cc['weight_decay'])
    if a.stage == 'measure':
        torch.cuda.reset_peak_memory_stats(); net.train(); times = []
        for k in range(cc['measure_steps']):
            t1 = time.time(); idx = torch.from_numpy(rng.integers(0, len(TR), bs)).to(dev); x, p, sj, _, _, code, _ = TR.batch(idx)
            loss = lm_loss(x, p, sj, code, samp); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); torch.cuda.synchronize(); times.append(time.time() - t1)
        n_items = 9 * 14_900
        res = dict(stage='measure', batch=bs, step_seconds=float(np.median(times[2:])), max_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
                   steps_per_pass=n_items // bs, minutes_per_pass=float(np.median(times[2:]) * (n_items // bs) / 60), grad_checkpointing=bool(cc.get('grad_checkpointing')),
                   llm=cc['llm'], loss_first=None, d_llm=d_llm)
        json.dump(res, open(os.path.join(out, 'metrics.json'), 'w'), indent=1); print('MEASURE', json.dumps(res), flush=True); return
    # stage 1
    best, bad, state, step = float('inf'), 0, None, 0; t1 = time.time(); max_steps = cc['smoke_steps'] if a.smoke else cc['max_steps']
    v0 = vloss(VA, vidx); hist.append(dict(stage='shared', step=0, valid=v0)); print(f'valid at start {v0:.4f}', flush=True)
    while step < max_steps:
        perm = torch.from_numpy(rng.permutation(len(TR))).to(dev)
        for b in range(0, len(TR) - bs + 1, bs):
            x, p, sj, _, _, code, _ = TR.batch(perm[b:b + bs]); loss = lm_loss(x, p, sj, code, samp)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); step += 1
            if step % 100 == 0: print(f'step {step} loss {loss.item():.4f} ({time.time() - t1:.0f}s)', flush=True)
            if step % (cc['smoke_eval'] if a.smoke else cc['eval_every']) == 0 or step >= max_steps:
                vl = vloss(VA, vidx); hist.append(dict(stage='shared', step=step, valid=vl, train=loss.item(), seconds=time.time() - t1)); print(f'step {step}: valid {vl:.4f}', flush=True)
                if vl < best: best, bad, state = vl, 0, copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
                else: bad += 1
                if bad >= cc['patience'] or step >= max_steps: break
        if bad >= cc['patience'] or step >= max_steps: break
    net.load_state_dict(state); t_shared = time.time() - t1; del TR, VA; torch.cuda.empty_cache()
    torch.save({k: v for k, v in state.items() if k.startswith('proj')}, os.path.join(out, 'proj.pt')); torch.save({k: v for k, v in state.items() if not k.startswith('proj')}, os.path.join(out, 'encoder.pt'))
    # stage 2: target layer
    for p_ in net.parameters(): p_.requires_grad_(False)
    W = net.enc.subject.weights; mean_layer = W[:ti].mean(0).clone(); layers = {'k0': mean_layer.cpu()}; cal_meta = {}
    for kname, sess in (tc['calibration']['sessions'].items() if not a.smoke else [('ses123', [1, 2, 3])]):
        t2 = time.time(); codes, X, _ = subject_mean(cache, a.target, sess, codes=np.r_[train_codes, valid_codes]); m = np.isin(codes, valid_codes)
        CT, CV = GPUSet([(ti, codes[~m], X[~m])], pos, dev), GPUSet([(ti, codes[m], X[m])], pos, dev); del X
        with torch.no_grad(): W[ti] = mean_layer
        W.requires_grad_(True); copt = torch.optim.Adam([W], lr=cc['cal_lr']); cvi = torch.arange(len(CV), device=dev); init = vloss(CV, cvi); b_v, row, badc = init, mean_layer.clone(), 0
        for ep in range(1 if a.smoke else cc['cal_max_epochs']):
            perm = torch.from_numpy(rng.permutation(len(CT))).to(dev)
            for k, b in enumerate(range(0, max(len(CT) - bs + 1, 1), bs)):
                if a.smoke and k >= cc['smoke_steps']: break
                x, p, sj, _, _, code, _ = CT.batch(perm[b:b + bs]); loss = lm_loss(x, p, sj, code, samp); copt.zero_grad(set_to_none=True); loss.backward()
                W.grad[:ti] = 0; copt.step()
            vl = vloss(CV, cvi); hist.append(dict(stage=f'cal_{kname}', epoch=ep, valid=vl))
            if vl < b_v: b_v, row, badc = vl, W[ti].detach().clone(), 0
            else:
                badc += 1
                if badc >= cc['cal_patience']: break
        W.requires_grad_(False)
        with torch.no_grad(): W[ti] = row
        layers[kname] = row.cpu(); cal_meta[kname] = dict(init_valid=init, best_valid=b_v, kept_initial=bool(b_v >= init), seconds=time.time() - t2); del CT, CV
        print(f'calibration {kname}: {init:.4f} -> {b_v:.4f}', flush=True)
    torch.save(layers, os.path.join(out, 'layers.pt'))
    # generation on the test inputs
    pg = torch.as_tensor(pos, dtype=torch.float32, device=dev); rows_out = []
    if a.smoke:                                                               # code check: Gaussian inputs, the test inputs are never opened
        g = np.random.default_rng(0).standard_normal((200, pos.shape[0], rc['model']['n_times'])).astype(np.float32)
        inputs = {k: g for k in ('avg20', 'phase', 'noise', 'single1')}; inputs['noeeg'] = g.mean(0, keepdims=True)
    else:
        Zin = np.load(os.path.join(P(cc['inputs_dir']), f't{a.target:02d}.npz'))
        code = Zin['single_code']; first = np.array([int(np.where(code == k + 1)[0][0]) for k in range(200)])
        inputs = {'avg20': Zin['avg20'], 'phase': Zin['phase'], 'noise': Zin['noise'], 'single1': np.asarray(Zin['single'][first], np.float32),
                  'noeeg': Zin['avg20'].mean(0, keepdims=True)}
    plan = [('ses123', k) for k in ('avg20', 'phase', 'noise', 'single1', 'noeeg')] + [(k, 'avg20') for k in ('k0', 'ses1', 'ses12') if k in layers]
    if a.smoke: plan = plan[:2]

    def generate(X):
        outs = []
        for b in range(0, len(X), cc['gen_batch']):
            xb = torch.from_numpy(np.asarray(X[b:b + cc['gen_batch']], np.float32)).to(dev); B = len(xb)
            with torch.no_grad():
                E = net(xb, pg[None].expand(B, -1, -1), torch.full((B,), ti, device=dev)).to(torch.bfloat16)
                pe = emb(torch.tensor([pre_ids], device=dev)).expand(B, -1, -1); po = emb(torch.tensor([post_ids], device=dev)).expand(B, -1, -1)
                g = llm.generate(inputs_embeds=torch.cat([pe, E, po], 1), attention_mask=torch.ones((B, len(pre_ids) + E.shape[1] + len(post_ids)), dtype=torch.long, device=dev),
                                 max_new_tokens=cc['max_new_tokens'], do_sample=False, output_scores=True, return_dict_in_generate=True, pad_token_id=tokz.pad_token_id)
                seq = g.sequences[:, -len(g.scores):]; eos = tokz.convert_tokens_to_ids('<|im_end|>')
                tl = torch.stack([F.log_softmax(sc.float(), -1).gather(-1, seq[:, t, None])[:, 0] for t, sc in enumerate(g.scores)], 1)   # [B, T]
                for i in range(B):
                    s_ = seq[i].tolist(); n = s_.index(eos) if eos in s_ else len(s_)
                    outs.append((tokz.decode(s_[:n], skip_special_tokens=True).strip(), float(tl[i, :max(n, 1)].mean())))
        return outs
    llm.eval(); llm.config.use_cache = True
    for cal, kind in plan:
        with torch.no_grad(): W[ti] = layers[cal].to(dev)
        res = generate(inputs[kind]); arm = 'shuf' if a.shuffle else 'dec'; cond = 'noeeg' if kind == 'noeeg' else f'{arm}|{cal}|{kind}'
        if a.shuffle and kind == 'noeeg': cond = 'shuf|noeeg'
        for k, (txt_, conf) in enumerate(res): rows_out.append(dict(cond=cond, subject=a.target, row=-1 if kind == 'noeeg' else k, output=txt_, confidence=conf))
        print(f'generated {cond}: {len(res)} rows, e.g. {res[0][0]!r}', flush=True)
    with open(os.path.join(out, 'c_out.jsonl'), 'w') as f:
        for x in rows_out: f.write(json.dumps(x, ensure_ascii=False) + '\n')
    json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
    res = dict(target=a.target, shuffle=a.shuffle, final_run=os.path.basename(rd), shared_best_valid=best, shared_start_valid=v0, steps=step, shared_seconds=t_shared,
               calibration=cal_meta, seconds=time.time() - t0, max_memory_gb=torch.cuda.max_memory_allocated() / 1e9, device=torch.cuda.get_device_name(0),
               pool_sha256=fsha(pf) if os.path.exists(pf) else 'smoke-template', c_out_sha256=fsha(os.path.join(out, 'c_out.jsonl')))
    json.dump(res, open(os.path.join(out, 'metrics.json'), 'w'), indent=1); print('SUMMARY', json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
