"""F-V (D-112 section 6) statistics on subject x test-concept cells.

A cell is the rank percentile of the true name among K candidates: 1 - (#candidates scoring strictly above the true one) /
(K - 1). The statistic is the equal-weight mean over subjects x concepts. Intervals come from the crossed pigeonhole bootstrap
(subjects and concepts drawn with replacement independently, the same concept draw for every subject).
"""
import numpy as np


def unit(A): return A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)


def seed_mean(E_list):
    """seed-mean predicted embedding: mean of the L2-normalised seed outputs, normalised again"""
    return unit(np.mean([unit(np.asarray(E, np.float64)) for E in E_list], 0))


def cells(S, true):
    """S [n, K] scores, true [n] column of the true candidate -> rank-percentile cells [n]"""
    S = np.asarray(S, np.float64); t = S[np.arange(len(S)), true]; return 1 - (S > t[:, None]).sum(1) / (S.shape[1] - 1)


def ranks(S, true):
    S = np.asarray(S, np.float64); t = S[np.arange(len(S)), true]; return (S > t[:, None]).sum(1)


def pigeonhole(M, n_boot, rng):
    """M [subjects, concepts] -> n_boot crossed-bootstrap means"""
    ns, nc = M.shape; out = np.empty(n_boot)
    for b in range(n_boot):
        si = rng.integers(0, ns, ns); ci = rng.integers(0, nc, nc); out[b] = M[np.ix_(si, ci)].mean()
    return out


def perm_null(S, true, n_perm, rng):
    """concept-permutation null of one subject's mean cell: the true columns re-assigned among the concepts (a bijection)"""
    return np.array([cells(S, true[rng.permutation(len(true))]).mean() for _ in range(n_perm)])


def wolpaw(p, n):
    p = min(max(p, 1e-12), 1 - 1e-12); return float(np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1)))


def null_mean_exact(S, true):
    """exact expectation of the mean cell over all bijections of the true columns among the evaluated concepts: the mean cell of
    every evaluated row at every evaluated true column (strict ties as in cells)"""
    S = np.asarray(S, np.float64); K = S.shape[1]; srt = np.sort(S, 1)
    return float(np.mean([1 - (K - np.searchsorted(srt[i], S[i, true], 'right')) / (K - 1) for i in range(len(S))]))


def control_rule(excess_cells, excess_boot, e_dec, ratio, lower_pct):
    """D-112 amendment 5, rule N: control excess E_c (cells minus the exact permutation-null mean) passes when (a) the pigeonhole
    lower_pct percentile <= 0 and (b) the point E_c <= ratio x E_dec (decoder excess on the same set and condition)"""
    pt = float(np.mean(excess_cells)); lo = float(np.percentile(excess_boot, lower_pct)); a = lo <= 0; b = pt <= ratio * e_dec
    return dict(excess=pt, lower=lo, lower_pct=lower_pct, e_dec=float(e_dec), ratio=ratio, bound_b=float(ratio * e_dec), passed_a=bool(a), passed_b=bool(b),
                passed=bool(a and b))


def rule(diff_boot, point, lower_pct, min_point):
    lo = float(np.percentile(diff_boot, lower_pct)); return dict(point=float(point), lower=lo, lower_pct=lower_pct, min_point=min_point,
                                                                 adopted=bool(lo > 0 and point >= min_point))
