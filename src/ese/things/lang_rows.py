"""F-VIII (D-117 amendment 5) input rows: one predicted embedding z per (condition, subject, test concept) and the shared helpers
of methods A, B and the verdict.

Condition keys `arm|calibration|kind[@seed]`:
  dec|<cal>|<kind>   decoder, seed mean of the three final seeds (mean of L2-normalised outputs, as the F-V verdict)
  shuf|<cal>|<kind>  label-shuffle model (one seed)
  kind avg20 / phase / noise as in V2; single1 = the first session-4 single trial of every test code (time order of V2's
  single_code), report only
  noeeg              one row per subject: the subject's mean of dec|ses123|avg20 over its 200 rows (language-model prior baseline)
  sim_gauss          isotropic Gaussian per row; sim_const: one Gaussian per subject for all its rows;
  sim_planted        unit(CLIP image(i) + s * Gaussian), s set by bisection so that the 200-test-name MRP is planted_target_mrp
  '@k' (k = 1, 2) is the LLM sampling seed number for report-only reruns of the same rows (default 0)
Rows of a subject follow test codes 1..200 (truth = row index). Mismatched-pair rows (G2) reuse dec|ses123|avg20 outputs with the
truth shifted by one place (circular, inside the subject).
"""
import os, hashlib, numpy as np


def unit(x): x = np.asarray(x, np.float64); return x / np.linalg.norm(x, axis=-1, keepdims=True)


def seed_mean(E): return unit(np.mean([unit(e) for e in E], 0))


def first_dir(dirs, root):
    for d in dirs:
        p = d if os.path.isabs(d) else os.path.join(root, d)
        if os.path.isdir(p): return p
    raise FileNotFoundError(f'none of {dirs}')


def mrp(S, true):
    rk = (S > S[np.arange(len(S)), true][:, None]).sum(1); return 1 - rk / (S.shape[1] - 1)


def planted(test_img, names_test, target, seed):
    """noise scale s (bisection) so that MRP(unit(img + s * g) vs the 200 test names) ~= target; returns (s, Z [200, D])"""
    g = np.random.default_rng(seed).standard_normal(test_img.shape) / np.sqrt(test_img.shape[1]); true = np.arange(len(test_img))
    lo, hi = 0.0, 50.0
    for _ in range(60):
        s = (lo + hi) / 2; m = mrp(unit(test_img + s * g) @ unit(names_test).T, true).mean()
        lo, hi = (s, hi) if m > target else (lo, s)
    s = (lo + hi) / 2; return s, unit(test_img + s * g)


def rows(cond, c, root, subjects=None, test_img=None, names_test=None):
    """-> list of (subject, row, z [D]); truth of a row = row (test code row) except noeeg / sim_const, which have row -1"""
    subjects = subjects or c['subjects']; base, _, seedno = cond.partition('@'); out = []
    if base.startswith('sim_'):
        D = test_img.shape[1]
        for s in subjects:
            rng = np.random.default_rng([c['sim']['seed'], s, ['sim_gauss', 'sim_const', 'sim_planted'].index(base)])
            if base == 'sim_gauss': out += [(s, k, unit(rng.standard_normal(D))) for k in range(200)]
            elif base == 'sim_const': out.append((s, -1, unit(rng.standard_normal(D))))
            else:
                _, Z = planted(test_img, names_test, c['sim']['planted_target_mrp'], [c['sim']['seed'], s]); out += [(s, k, Z[k]) for k in range(200)]
        return out
    zd = first_dir(c['z']['dirs'], root)
    for s in subjects:
        Z = np.load(os.path.join(zd, f'z_emb_t{s:02d}.npz'))
        if base == 'noeeg':
            E = seed_mean([Z[f'seed{i}|ses123|avg20'] for i in range(c['z']['n_seeds'])]); out.append((s, -1, unit(E.mean(0)))); continue
        arm, cal, kind = base.split('|'); src = 'single' if kind == 'single1' else kind
        get = (lambda: seed_mean([Z[f'seed{i}|{cal}|{src}'] for i in range(c['z']['n_seeds'])])) if arm == 'dec' else (lambda: unit(Z[f'shuffle|{cal}|{src}']))
        E = get()
        if kind == 'single1':
            code = Z['single_code']; first = [int(np.where(code == k + 1)[0][0]) for k in range(200)]; E = E[first]
        out += [(s, k, E[k]) for k in range(200)]
    return out


def row_seed(cond, s, k, r):
    """LLM sampling seed of a row and round (amendment 5: sha256 of condition | subject | concept | seed number)"""
    _, _, seedno = cond.partition('@')
    return int(hashlib.sha256(f'{cond.partition("@")[0]}|{s}|{k}|{seedno or 0}|{r}'.encode()).hexdigest()[:8], 16)
