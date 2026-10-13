"""D-086 pre-registration check of a candidate file (pilot): is the target exchangeable with the 19 alternatives under the
D-076 residual metric, and does the D-075 length residualisation (fitted on un-normalised DailyDialog train text) still
remove length from normalised text?

(a) exchangeability, per metric (e5 primary, bge sensitivity; residual and raw cosine):
    centrality  - rank percentile of the target among the 20 candidates by mean cosine to the other 19 (0.5 if exchangeable)
    probe       - the targets of `probes` random other trials as zero-information outputs: target RR among the 20 candidates
                  (exchangeable: H_20 / 20 = 0.1799) and rank percentile (0.5)
    R zero info - D-083 R structure at g = inf (noise only, D-068 TXT embedding centred on the resid_fit mean): mean
                  probability that the evidence is the target (0.05 if exchangeable)
(b) length: share of residual-embedding variance explained by len_feats over all candidates (raw embedding for contrast);
    Spearman of word count with the mean residual cosine to candidates of other trials (D-075 T3 form); Spearman over
    alternatives of |word-count difference to the target| with the residual cosine to the target.
Also the target's word-count rank percentile among the candidates.

usage: python scripts/d3_candidates_check.py configs/d3_device.yaml OUT.json CANDIDATES.jsonl [CANDIDATES.jsonl ...]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, json, yaml, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d3_metric_gate import Embedder, len_feats, nwords
from d3_device import RStructure
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def unit(X): return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


def rank_pct(s, j):
    """rank percentile of entry j among s (higher score = higher percentile), ties 0.5"""
    o = np.delete(s, j); return float(((o < s[j]).sum() + 0.5 * (o == s[j]).sum()) / len(o))


def rr(s, j): return float(1 / (1 + (np.delete(s, j) > s[j]).sum() + 0.5 * (np.delete(s, j) == s[j]).sum()))


def check(T, E, emb_txt, sigma, probes, rng):
    n = len(T); K = len(T[0]['candidates']); ti = np.array([t['target_index'] for t in T]); flat = [s for t in T for s in t['candidates']]
    raw = E.raw(flat); res = raw - len_feats(flat) @ E.W
    out = {}
    for kind, X in (('resid', unit(res)), ('raw', unit(raw))):
        X = X.reshape(n, K, -1); cen, prr, ppct = [], [], []
        for i in range(n):
            S = X[i] @ X[i].T; np.fill_diagonal(S, np.nan); c = np.nanmean(S, 1); cen.append(rank_pct(c, ti[i]))
            for k in rng.choice(np.delete(np.arange(n), i), probes, replace=False):
                s = X[i] @ X[k, ti[k]]; prr.append(rr(s, ti[i])); ppct.append(rank_pct(s, ti[i]))
        se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v)))
        out[kind] = dict(centrality_pct=float(np.mean(cen)), centrality_pct_se=se(cen), probe_rr=float(np.mean(prr)), probe_rr_expected=float(np.mean(1 / np.arange(1, K + 1))),
                         probe_pct=float(np.mean(ppct)), probe_pct_se=float(np.std(np.array(ppct).reshape(n, probes).mean(1), ddof=1) / np.sqrt(n)))
    w = np.array([nwords(s) for s in flat], float); F = len_feats(flat)
    def r2(Y):
        Yc = Y - Y.mean(0); B = np.linalg.lstsq(F, Y, rcond=None)[0]; return float(1 - ((Y - F @ B) ** 2).sum() / (Yc ** 2).sum())
    Rn = unit(res).reshape(n, K, -1); other = []
    for i in range(n):
        o = Rn[np.delete(np.arange(n), i)].reshape(-1, Rn.shape[-1]); other.append((Rn[i] @ o.T).mean(1))
    other = np.concatenate(other); dl, cs = [], []
    for i in range(n):
        wi = w.reshape(n, K)[i]
        for j in range(K):
            if j != ti[i]: dl.append(abs(wi[j] - wi[ti[i]])); cs.append(float(Rn[i, j] @ Rn[i, ti[i]]))
    out['length'] = dict(r2_len_feats_resid=r2(res), r2_len_feats_raw=r2(raw), spearman_words_vs_mean_cos_other=float(spearmanr(w, other)[0]),
                         spearman_absdiff_vs_cos_target=float(spearmanr(dl, cs)[0]),
                         target_word_rank_pct=float(np.mean([rank_pct(w.reshape(n, K)[i], ti[i]) for i in range(n)])))
    if emb_txt is not None:
        R = RStructure(emb_txt, ti, sigma, 200, 0); q = R.q(np.inf); out['R_zero_info_target_prob'] = float(q[np.arange(n), ti].mean())
        out['R_zero_info_target_prob_se'] = float(q[np.arange(n), ti].std(ddof=1) / np.sqrt(n))
    return out


def main():
    cfg, out_path, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    c = yaml.safe_load(open(cfg)); fit = list(pd.read_csv(os.path.join(ROOT, c['resid_fit']), sep='\t', keep_default_na=False).text)
    sigma = np.load(os.path.join(ROOT, c['sigma']))['sigma']; rng = np.random.default_rng(c['seed']); res = {}
    from ds004408_identify import embed
    mu = embed(fit).mean(0)
    for f in files:
        T = [json.loads(l) for l in open(os.path.join(ROOT, f))]; flat = [s for t in T for s in t['candidates']]
        emb_txt = (embed(flat).reshape(len(T), len(T[0]['candidates']), -1) - mu).astype(np.float64)
        res[f] = {}
        for name, spec in c['metrics'].items():
            E = Embedder(spec, 128, fit); res[f][spec['model']] = check(T, E, emb_txt if name == 'primary' else None, sigma, 5, rng); del E
        print(f, json.dumps(res[f], indent=1), flush=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, out_path)), exist_ok=True); json.dump(res, open(os.path.join(ROOT, out_path), 'w'), indent=1)


if __name__ == '__main__':
    main()
