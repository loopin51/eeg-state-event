"""F-VIII sentence pool (D-117 amendments 3-5): 1,854 THINGS concepts x 5 one-sentence photo descriptions from OpenRouter
Claude Haiku 5.5, written from the concept name and the THINGS definition only (no image, no EEG, no result values).

Per concept: one request for 5 sentences (JSON). Format check: 5 sentences parsed, and each sentence's central object
(src/ese/things/lang.py) is a hit for its own concept. When fewer than 5 sentences pass, the same request is sent again (at most
`retries` more times); passing sentences are collected in response order across attempts, exact duplicates dropped, the first 5
kept; any slot still empty is filled with the fixed template ("A photo of a {name}.") and flagged. No concept is dropped.
Saved (out_dir): requests_responses.jsonl (request body without headers, full response JSON, per attempt), pool.tsv,
pool_report.json (model ids and providers seen, sha256 of the raw file and the pool, refusals / fallbacks, test concepts among
them, other-THINGS-name rate per sentence = the M2 floor). The key is read from the environment or a key file and never written.

usage: env_nlp/bin/python scripts/fv_lang_pool.py configs/fv_lang.yaml --dry-run      # print one request, send nothing
       env_nlp/bin/python scripts/fv_lang_pool.py configs/fv_lang.yaml [--limit N]     # generate (resumes from the jsonl)
"""
import os, sys, csv, json, time, hashlib, argparse, threading, urllib.request, urllib.error, yaml
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
DEF = 'Definition (from WordNet, Google, or Wikipedia)'


def fsha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def key(c):
    k = os.environ.get(c['key_env'])
    if k: return k.strip()
    f = os.environ.get('OPENROUTER_KEY_FILE') or os.path.join(ROOT, c['key_file'])
    if os.path.exists(f): return open(f).read().strip()
    sys.exit(f'no API key: set {c["key_env"]} or write the key to {f}')


def body(c, name, definition, max_tokens=None):
    schema = {'type': 'object', 'properties': {'sentences': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['sentences'],
              'additionalProperties': False}
    return {'model': c['model'], 'provider': c['provider'], 'max_tokens': max_tokens or c['max_tokens'], 'usage': {'include': True},
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'captions', 'strict': True, 'schema': schema}},
            'messages': [{'role': 'system', 'content': c['system']},
                         {'role': 'user', 'content': c['user'].format(name=name, definition=definition, n=c['n_sentences'])}]}


def post(c, k, b):
    req = urllib.request.Request(c['endpoint'], data=json.dumps(b).encode(), method='POST',
                                 headers={'Authorization': f'Bearer {k}', 'Content-Type': 'application/json'})
    for t in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r: return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors='replace')[:500]
            if e.code in (429, 500, 502, 503, 529) and t < 3: time.sleep(5 * (t + 1)); continue
            return {'error': {'http': e.code, 'body': msg}}
        except Exception as e:
            if t < 3: time.sleep(5 * (t + 1)); continue
            return {'error': {'exception': repr(e)}}


def truncated(resp): return ((resp.get('choices') or [{}])[0].get('finish_reason')) == 'length'


def next_tokens(c, att, recs):
    """max_tokens of attempt `att` given the earlier records, or None when no further attempt is allowed (amendment 7: attempts
    after the first round only for a concept with a truncated response, with truncation_round.max_tokens)"""
    n1, tr = 1 + c['retries'], c.get('truncation_round') or {}
    if att < n1: return c['max_tokens']
    if tr and att < n1 + tr['retries'] and any(truncated(d['response']) for d in recs): return tr['max_tokens']
    return None


def parse(resp):
    try:
        txt = resp['choices'][0]['message']['content']; s = json.loads(txt[txt.find('{'): txt.rfind('}') + 1])['sentences']
        return [x.strip() for x in s if isinstance(x, str) and x.strip()]
    except Exception: return []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--limit', type=int)
    a = ap.parse_args(); C = yaml.safe_load(open(P(a.config))); c = C['pool']
    assert fsha(P(C['things_meta'])) == C['things_meta_sha256'], 'THINGS metadata sha256 differs from the registered copy'
    import spacy; from ese.things.lang import Vocab
    assert spacy.__version__ == C['parser']['spacy'], spacy.__version__
    nlp = spacy.load('en_core_web_lg'); assert f"en_core_web_lg-{nlp.meta['version']}" == C['parser']['model'], nlp.meta['version']
    V = Vocab(nlp, P(C['names_table']), P(C['things_meta'])); lock = threading.Lock()
    meta = {r['uniqueID'].strip().replace(' ', '_'): r for r in csv.DictReader(open(P(C['things_meta']), encoding='utf-8'), delimiter='\t')}
    rows = list(csv.DictReader(open(P(C['names_table']), encoding='utf-8'), delimiter='\t'))
    todo = [(i, r['concept'], r['display_name'], meta[r['concept']][DEF].strip()) for i, r in enumerate(rows)]
    if a.limit: todo = todo[:a.limit]
    if a.dry_run: print(json.dumps(body(c, todo[0][2], todo[0][3]), indent=1)); return
    out = P(c['out_dir']); os.makedirs(out, exist_ok=True); raw = os.path.join(out, 'requests_responses.jsonl')
    done = {}
    if os.path.exists(raw):
        for l in open(raw): d = json.loads(l); done.setdefault(d['concept'], []).append(d)
    k = key(c); spent = [sum(float(d['response'].get('usage', {}).get('cost') or 0) for v in done.values() for d in v)]

    def run(item):
        i, concept, name, definition = item
        recs = list(done.get(concept, [])); good = []
        for d in recs: good += [s for s in parse(d['response']) if V.judge(s, i)['cls'] == 'hit' and s not in good]
        att = len(recs)
        while len(good) < c['n_sentences']:
            mt = next_tokens(c, att, recs)
            if mt is None: break
            with lock:
                if spent[0] >= c['cost_cap_usd']: print('cost cap reached; stopping', flush=True); return
            b = body(c, name, definition, mt); r = post(c, k, b); new = parse(r)
            ok = [s for s in new if V.judge(s, i)['cls'] == 'hit']; good += [s for s in ok if s not in good]
            rec = dict(concept=concept, row=i, attempt=att, date=time.strftime('%Y-%m-%dT%H:%M:%S%z'), request=b, response=r,
                       n_parsed=len(new), n_pass=len(ok))
            rec['final'] = len(good) >= c['n_sentences'] or next_tokens(c, att + 1, recs + [rec]) is None
            with lock:
                spent[0] += float(r.get('usage', {}).get('cost') or 0)
                with open(raw, 'a') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            recs.append(rec); att += 1

    t0 = time.time()
    with ThreadPoolExecutor(c['threads']) as ex:
        for n, _ in enumerate(ex.map(run, todo)):
            if n % 100 == 0: print(f'{n}/{len(todo)} concepts, cost {spent[0]:.4f} USD, {time.time() - t0:.0f}s', flush=True)
    if spent[0] >= c['cost_cap_usd']: sys.exit('stopped at the cost cap; pool not written')
    # pool and report
    recs = {}
    for l in open(raw): d = json.loads(l); recs.setdefault(d['concept'], []).append(d)
    pool, rep = [], dict(models={}, providers={}, n_requests=0, cost_usd=0.0, n_fallback_sentences=0, fallback_concepts=[],
                         refusal_or_error_requests=0, truncated_requests=0, retried_concepts=0,
                         truncation_round=dict(requests=0, cost_usd=0.0, concepts=[]))
    for i, concept, name, _ in todo:
        ds = sorted(recs.get(concept, []), key=lambda d: d['attempt']); good = []
        rep['retried_concepts'] += len(ds) > 1
        for d in ds:
            r = d['response']; rep['n_requests'] += 1; rep['cost_usd'] += float(r.get('usage', {}).get('cost') or 0)
            rep['models'][r.get('model', '?')] = rep['models'].get(r.get('model', '?'), 0) + 1
            rep['providers'][r.get('provider', '?')] = rep['providers'].get(r.get('provider', '?'), 0) + 1
            if 'error' in r or not parse(r): rep['refusal_or_error_requests'] += 1
            rep['truncated_requests'] += truncated(r)
            if d['attempt'] > c['retries']:
                t = rep['truncation_round']; t['requests'] += 1; t['cost_usd'] += float(r.get('usage', {}).get('cost') or 0)
                if concept not in t['concepts']: t['concepts'].append(concept)
            for s in parse(r):
                if V.judge(s, i)['cls'] == 'hit' and s not in [g[0] for g in good]: good.append((s, d['attempt']))
        good = good[:c['n_sentences']]; nf = c['n_sentences'] - len(good)
        if nf: rep['n_fallback_sentences'] += nf; rep['fallback_concepts'].append([concept, rows[i]['split'], nf])
        good += [(c['fallback'].format(name=name), -1)] * nf
        for kk, (s, att) in enumerate(good): pool.append((rows[i]['things_id'], concept, rows[i]['split'], kk, s, 'fallback' if att < 0 else 'llm', att))
    pf = os.path.join(out, 'pool.tsv')
    with open(pf, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n'); w.writerow(['things_id', 'concept', 'split', 'k', 'sentence', 'source', 'attempt']); w.writerows(pool)
    other = [V.judge(p[4], V.index[p[1]]) for p in pool]
    rep.update(n_concepts=len(todo), n_sentences=len(pool), raw_sha256=fsha(raw), pool_sha256=fsha(pf), vocab_sha256=V.sha,
               spacy=spacy.__version__, parser_model=C['parser']['model'], expected_canonical=c['expected_canonical'],
               test_fallback_concepts=sum(x[1] == 'test' for x in rep['fallback_concepts']),
               other_name_rate=float(sum(any(k == 'error' for _, k in j['mentions']) for j in other) / len(other)),
               central_hit_rate=float(sum(j['cls'] == 'hit' for j in other) / len(other)))
    json.dump(rep, open(os.path.join(out, 'pool_report.json'), 'w'), indent=1); print(json.dumps({k: v for k, v in rep.items() if k != 'fallback_concepts'}))


if __name__ == '__main__':
    main()
