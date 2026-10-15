"""Shared statistics of the F-II confirmatory comparisons (D-097): C1 (ds004408, cell = (target subject, test run)) and C2
(DTU, cell = (target subject, test pair)). Imported by scripts/c1_verdict.py, scripts/c1_mde.py and the C2 verdict.

Arrays: an arm is m[n_seeds, n_subjects, n_units] (cell MRP of each seed, nan where the cell does not exist) and the query
counts n[n_subjects, n_units] (0 where the cell does not exist; identical for every arm, seed and k, D-097 section 4).
Statistic (section 4): query-weighted mean over cells of the seed-mean cell values; Delta = same weights on v_P2 - v_P1.
Interval (section 5): seed-stratified pigeonhole bootstrap. In every draw, in this order: (1) for every arm separately n_a
seeds with replacement, (2) n_subjects subjects and (3) n_units units with replacement (the same draw for both arms); the cell
weight is query count x subject draw count x unit draw count.
Categories (section 6), family alpha and transitions (section 1, amendments 1 and 2), the collapse-model check (section 7.4)
and the phase-randomisation check (section 7.1) are defined here once.
"""
import numpy as np

CATEGORIES = ('adopt', 'superior_below_sesoi', 'harmful', 'futile', 'undecidable')
REJECTED = ('adopt', 'superior_below_sesoi')


def level(v, n):
    """query-weighted mean of cell values v [S, U] with weights n [S, U] (cells with n == 0 ignored)"""
    ok = n > 0; return float((np.where(ok, v, 0.0) * n).sum() / n[ok].sum())


def seed_mean(m): return m.mean(0)


def seed_pigeonhole(arms, n, n_boot, rng, contrast=None, seed_inflate=False):
    """arms: list of m arrays [n_a, S, U]; contrast: weights per arm (default: last minus first for two arms, the level for one).
    Returns the n_boot bootstrap values of sum_W (sum_a c_a v*_a) / sum_W.
    seed_inflate: seed deviations from the arm's seed mean are scaled by sqrt(n_a / (n_a - 1)) before resampling (the SE_0
    correction of section 10; candidate ② of amendment 5, chosen or not by the amendment-5 simulation)."""
    if contrast is None: contrast = [1.0] if len(arms) == 1 else [-1.0] + [0.0] * (len(arms) - 2) + [1.0]
    S, U = n.shape; out = np.empty(n_boot); arms = [np.where(n[None] > 0, a, 0.0) for a in arms]
    if seed_inflate: arms = [a.mean(0) + (a - a.mean(0)) * np.sqrt(len(a) / (len(a) - 1)) if len(a) > 1 else a for a in arms]
    for b in range(n_boot):
        vs = [a[rng.integers(0, len(a), len(a))].mean(0) for a in arms]                 # (1) seeds, arm by arm
        cs = np.bincount(rng.integers(0, S, S), minlength=S); cu = np.bincount(rng.integers(0, U, U), minlength=U)   # (2), (3)
        W = n * cs[:, None] * cu[None]; out[b] = (W * sum(c * v for c, v in zip(contrast, vs))).sum() / W.sum()
    return out


def delta(m1, m2, n):
    """point estimate Delta = level(v2 - v1) (section 4)"""
    return level(np.where(n > 0, seed_mean(m2) - seed_mean(m1), 0.0), n)


def welch(m1, m2, n):
    """one-sided Welch t test of the seed-level pooled values (H1: arm 2 > arm 1), Welch-Satterthwaite df (amendment 5, ④)"""
    from scipy.stats import t as tdist
    L1 = np.array([level(x, n) for x in m1]); L2 = np.array([level(x, n) for x in m2]); d = L2.mean() - L1.mean()
    a1, a2 = L1.var(ddof=1) / len(L1), L2.var(ddof=1) / len(L2); se2 = a1 + a2
    if se2 <= 0: return dict(t=None, df=None, p=0.0 if d > 0 else 1.0, diff=float(d))
    df = se2 ** 2 / (a1 ** 2 / (len(L1) - 1) + a2 ** 2 / (len(L2) - 1)); tt = d / np.sqrt(se2)
    return dict(t=float(tt), df=float(df), p=float(tdist.sf(tt, df)), diff=float(d))


def category_welch(point, lower, upper90, welch_p, alpha, sesoi=0.02):
    """amendment 5 ④: "lower > 0" reads "lower > 0 and Welch p <= alpha_A"; otherwise the null is not rejected"""
    return category(point, lower if welch_p <= alpha else min(lower, 0.0), upper90, sesoi)


def interval(boot, alphas=(0.04, 0.05)):
    """one-sided (1 - alpha) lower bounds and the 90% interval (5th, 95th percentile)"""
    out = {f'lower_{a}': float(np.percentile(boot, 100 * a)) for a in alphas}
    out.update(p5=float(np.percentile(boot, 5)), p95=float(np.percentile(boot, 95)), p2_5=float(np.percentile(boot, 2.5)),
               p97_5=float(np.percentile(boot, 97.5)), sd=float(boot.std(ddof=1)))
    return out


def category(point, lower, upper90, sesoi=0.02):
    """D-097 section 6, judged top to bottom (mutually exclusive)"""
    if lower > 0 and point >= sesoi: return 'adopt', ''
    if lower > 0 and upper90 < sesoi: return 'superior_below_sesoi', 'not counted as a new-model contribution'
    if upper90 < 0: return 'harmful', 'for the record'
    if lower <= 0 and upper90 < sesoi: return 'futile', ''
    note = 'effect present, whether it reaches the SESOI is unknown' if lower > 0 else ''
    return 'undecidable', note


def alpha_chain_A(l1):
    """alpha of chain A at the C1 verdict (section 1 with amendment 1).
    l1 = dict(gate0='pass'|'fail'|'pending', gate0_before_c1=bool, gateJ='present'|'hold'|'futile'|'undecidable'|'pending',
              l1_rejected=True|False|None, dev_futile=bool)
    Returns (alpha in force, reason, alpha_after_graph) where alpha_after_graph is the level of a later graph re-verdict
    (0.05 once L1's 0.01 has moved to chain A), or None."""
    if l1.get('dev_futile'): return None, 'development-stage futility: C1 test not opened, chain A alpha unused and not passed to L1', None
    g0, gJ, rej = l1.get('gate0', 'pending'), l1.get('gateJ', 'pending'), l1.get('l1_rejected')
    if g0 == 'fail':
        if l1.get('gate0_before_c1'): return 0.05, 'gate 0 failed before the C1 verdict: C1 judged at 0.05 from the start', None
        return 0.04, 'gate 0 failed after the C1 verdict: 0.01 moves to chain A, re-verdict at 0.05 (both reported)', 0.05
    if g0 == 'pass' and gJ not in ('present', 'pending'):
        return 0.04, f'gate 0 passed, gate J {gJ}: L1 closed, its 0.01 unused', None
    if rej is True: return 0.04, 'L1 rejected: its 0.01 moves to chain A, re-verdict at 0.05 (both reported)', 0.05
    return 0.04, 'chain A at its own 0.04 (L1 open or pending)', None


def l1_closed(l1):
    """amendment 1: L1 is open only when gate 0 passes and gate J is 'present'; once closed, chain A never passes its alpha to L1"""
    g0, gJ = l1.get('gate0', 'pending'), l1.get('gateJ', 'pending')
    return g0 == 'fail' or (g0 == 'pass' and gJ not in ('present', 'pending'))


def chain_A_all_rejected(c1_cat, c2_exists, c2_cat=None):
    """section 1 / amendment 2 condition 2: when chain A passes its 0.04 to L1"""
    if c1_cat not in REJECTED: return False
    if not c2_exists: return c1_cat == 'adopt'
    return c1_cat == 'adopt' and c2_cat in REJECTED                   # C2 is tested only when C1 is 'adopt'


# ------------------------------------------------------------------ circular-shift null on within-unit score matrices
def rank_matrix(S):
    """R[i, j] = number of candidates scoring strictly above candidate j in row i (stageB_b1_verdict.rank_matrix)"""
    Ssort = np.sort(S, axis=1); return np.stack([S.shape[1] - np.searchsorted(srt, row, side='right') for row, srt in zip(S, Ssort)])


def shift_null_cells(mats, maps):
    """mats[s][u] = square score matrix (queries = candidates, in the order the maps index); maps[u] = [n_shift, n_u] answer maps.
    Returns observed cell MRP [S, U] and null cell MRP [n_shift, S, U] (nan where the cell does not exist)."""
    subs = sorted(mats); units = sorted(maps); ns = next(iter(maps.values())).shape[0]
    obs = np.full((len(subs), len(units)), np.nan); null = np.full((ns, len(subs), len(units)), np.nan)
    for js, s in enumerate(subs):
        for ju, u in enumerate(units):
            if u not in mats[s]: continue
            R = rank_matrix(np.asarray(mats[s][u], np.float64)); n = R.shape[1]; i = np.arange(len(R))
            obs[js, ju] = np.mean(1 - R[i, i] / (n - 1)); null[:, js, ju] = (1 - R[i[None], maps[u]] / (n - 1)).mean(-1)
    return obs, null


def perm_p(T, null_stats): return float((1 + (np.asarray(null_stats) >= T).sum()) / (1 + len(null_stats)))


def collapse_matrices(mats):
    """section 7.4: keep only the column means (scores depend on the candidate only, information 0)"""
    return {s: {u: np.broadcast_to(np.asarray(S, np.float64).mean(0, keepdims=True), S.shape).copy() for u, S in d.items()} for s, d in mats.items()}


def collapse_check(mats, maps, n, p_min=0.01, ks_min=0.01, identity_passes=True):
    """pooled p and per-subject p of the information-0 matrices under the primary circular-shift null, KS of the subject p's.
    With square within-unit candidate sets and one-to-one shift maps the collapsed statistic equals every null draw exactly
    (every row is the same, so any permutation of the answers gives the same mean), and p = 1 everywhere; this identity is
    detected and reported (identity=True) rather than read as a KS failure."""
    from scipy.stats import kstest
    obs, null = shift_null_cells(collapse_matrices(mats), maps)
    T = level(obs, n); g = np.array([level(x, n) for x in null]); p = perm_p(T, g)
    ps = []
    for js in range(n.shape[0]):
        nj = n[js:js + 1]; ps.append(perm_p(level(obs[js:js + 1], nj), [level(x[js:js + 1], nj) for x in null]))
    ks = float(kstest(ps, 'uniform').pvalue); identity = bool(np.nanmax(np.abs(null - obs[None])) < 1e-12)          # amendment 5 (a)
    lit = bool(p >= p_min and ks >= ks_min)
    return dict(stat=T, null_mean=float(g.mean()), p=p, subject_p=ps, ks_p=ks, identity=identity, max_abs_null_minus_obs=float(np.nanmax(np.abs(null - obs[None]))),
                passed_literal=lit, passed=bool(lit or (identity and identity_passes)),
                note='identity: square within-unit candidates and one-to-one shifts, the column-bias path cannot move the statistic' if identity else '')
