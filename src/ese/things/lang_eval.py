"""F-VIII (D-117 amendments 1, 2, 5) metrics on subject x test-concept cells.

Per cell (subject s, test row k, truth t = names-table row of test concept k) and one output sentence:
  hit   strict M1: the central span is a hit for t          err  M2: the central span is another THINGS concept's name
  ment  hit or err (a THINGS name is the central object)    conf the method's confidence (B: margin, C: mean log-probability)
  m3    rank percentile of the true test image among the 200 test images by cos(CLIP text(output), CLIP image) (when given)
Matched mention rate (amendment 5): inside each pigeonhole draw, the condition with more mention cells turns its lowest-confidence
mention cells into abstentions (no hit, no error) until both have the same number; ties in confidence are broken by cell order.
"""
import numpy as np


class Judged:
    """judgement of a set of outputs for the 200 test truths; cache by sentence"""
    def __init__(self, V, test_truth):
        self.V, self.tt, self.cache = V, np.asarray(test_truth), {}

    def central(self, s):
        if s not in self.cache: self.cache[s] = self.V.central(s)[0]
        return self.cache[s]

    def cell(self, s, k):
        """-> (hit, err) for output s at test row k"""
        term = self.central(s)
        if term is None: return False, False
        c = self.V.classify(term, int(self.tt[k])); return c == 'hit', c == 'error'

    def perm_rates(self, s):
        """exact concept-permutation means of (hit, err) for output s: the mean over the 200 test truths"""
        term = self.central(s)
        if term is None: return 0.0, 0.0
        cl = [self.V.classify(term, int(t)) for t in self.tt]; return float(np.mean([x == 'hit' for x in cl])), float(np.mean([x == 'error' for x in cl]))

    def group(self, s, split):
        """central object group: 'test' / 'train' (a THINGS name of that split) / 'other' / 'none'"""
        term = self.central(s)
        if term is None: return 'none'
        cs = self.V.name_of.get(term, set()) | {i for i, S in enumerate(self.V.S) if term in S}
        if not cs: return 'other'
        return 'test' if any(split[i] == 'test' for i in cs) else 'train'


def table(J, outs, subjects, shift=0):
    """outs: dict (subject, row) -> (sentence, conf); row -1 = one output for all rows of the subject.
    shift: truth of row k is row (k + shift) % 200 (mismatched pairs, G2). -> dict of [n_subjects, 200] arrays"""
    ns = len(subjects); H = np.zeros((ns, 200), bool); E = np.zeros((ns, 200), bool); C = np.zeros((ns, 200))
    for i, s in enumerate(subjects):
        for k in range(200):
            sent, conf = outs[(s, k)] if (s, k) in outs else outs[(s, -1)]
            H[i, k], E[i, k] = J.cell(sent, (k + shift) % 200); C[i, k] = conf if conf is not None else 0.0
    return dict(hit=H, err=E, conf=C)


def match(hA, eA, cA, hB, eB, cB):
    """flattened cells of two conditions -> the same arrays after mention-rate matching"""
    hA, eA, hB, eB = hA.copy(), eA.copy(), hB.copy(), eB.copy(); mA, mB = hA | eA, hB | eB; d = int(mA.sum() - mB.sum())
    if d:
        h, e, c, m = (hA, eA, cA, mA) if d > 0 else (hB, eB, cB, mB); idx = np.where(m)[0]; drop = idx[np.argsort(c[idx], kind='stable')[:abs(d)]]
        h[drop] = False; e[drop] = False
    return hA, eA, hB, eB


def prec(h, e):
    n = h.sum() + e.sum(); return float(h.sum() / n) if n else 0.0


def boot_matched(TA, TB, stat, n_boot, rng):
    """pigeonhole (subjects x concepts, the same draw for both conditions) of stat(matched arrays) -> (point, draws)"""
    ns, nc = TA['hit'].shape

    def one(si, ci):
        g = lambda T, k: T[k][np.ix_(si, ci)].ravel()
        return stat(*match(g(TA, 'hit'), g(TA, 'err'), g(TA, 'conf'), g(TB, 'hit'), g(TB, 'err'), g(TB, 'conf')))
    pt = one(np.arange(ns), np.arange(nc)); draws = np.empty(n_boot)
    for b in range(n_boot): draws[b] = one(rng.integers(0, ns, ns), rng.integers(0, nc, nc))
    return pt, draws


def m3_cells(Etext, test_img, truth_rows):
    """Etext [n, D] output embeddings, rows k -> rank percentile of test image truth_rows among the 200 test images"""
    S = Etext @ test_img.T; t = S[np.arange(len(S)), truth_rows]; return 1 - (S > t[:, None]).sum(1) / (S.shape[1] - 1)
