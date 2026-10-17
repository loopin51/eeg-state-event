"""Format check of fetched F-V development run folders (k3s runbook section 6; D-112). Prints no development metric values.

Checks per folder: k3s attempts (1 expected; more = preemption restart, reported), commit = expected SHA, the six files,
summary.device contains 'RTX 5080', 9 trainers without the target, dev.A_ses4 n_candidates 200 and present + missing = 200,
every number in history.json finite, plus timing fields (summary.seconds, shared_seconds, shared_epochs, max_memory_gb).

--final: final-mode folders (D-112 4.6): five files (no dev_scores.npz), summary.mode 'final', calibration ses1 / ses12 / ses123
present with epochs <= 100, no 'dev' key in metrics.json (no scores computed).
--perm <8 hex>: expected first 8 characters of summary.shuffle_perm_sha for shuffle folders.

usage: python scripts/fv_check_dev.py <expected sha> [--final] [--perm c2348261] results/runs/<folder> [...]
"""
import json, math, os, sys

FILES = ['config.yaml', 'history.json', 'metrics.json', 'shared.pt', 'layers.pt', 'dev_scores.npz']


def finite(x):
    if isinstance(x, float): return math.isfinite(x)
    if isinstance(x, dict): return all(finite(v) for v in x.values())
    if isinstance(x, list): return all(finite(v) for v in x)
    return True


def main():
    a = sys.argv[1:]; sha = a.pop(0); final = '--final' in a; a = [x for x in a if x != '--final']; perm = None
    if '--perm' in a: i = a.index('--perm'); perm = a[i + 1]; del a[i:i + 2]
    folders = a; files = [f for f in FILES if not (final and f == 'dev_scores.npz')]
    for d in folders:
        p = lambda f: os.path.join(d, f); issues = []
        att = open(p('k3s_attempts.txt')).read().strip().splitlines() if os.path.exists(p('k3s_attempts.txt')) else []
        com = open(p('k3s_commit.txt')).read().strip() if os.path.exists(p('k3s_commit.txt')) else ''
        if len(att) != 1: issues.append(f'attempts={len(att)}')
        if com != sha: issues.append(f'commit={com[:12]}')
        miss = [f for f in files if not os.path.exists(p(f))]
        if miss: issues.append(f'missing={miss}')
        out = dict(folder=os.path.basename(d), attempts=len(att))
        if os.path.exists(p('metrics.json')):
            m = json.load(open(p('metrics.json'))); s = m['summary']; A = m.get('dev', {}).get('A_ses4', {})
            if 'RTX 5080' not in s.get('device', ''): issues.append(f'device={s.get("device")}')
            tr = s.get('trainers', [])
            if len(tr) != 9 or s.get('target') in tr: issues.append(f'trainers={tr}')
            if final:
                if s.get('mode') != 'final': issues.append(f'mode={s.get("mode")}')
                if 'dev' in m: issues.append('dev key present')
                cal = s.get('calibration') or {}
                for k in ('ses1', 'ses12', 'ses123'):
                    e = (cal.get(k) or {}).get('epochs')
                    if k not in cal or (e is not None and e > 100): issues.append(f'calibration {k}')
            elif A.get('n_candidates') != 200 or A.get('n_concepts_present', 0) + A.get('n_concepts_missing', 0) != 200:
                issues.append('A_ses4 counts')
            if s.get('shuffle') and perm and not (s.get('shuffle_perm_sha') or '').startswith(perm): issues.append('shuffle_perm_sha')
            out.update(target=s.get('target'), candidate=s.get('candidate'), seed=s.get('seed'), shuffle=s.get('shuffle'), shuffle_perm_sha=(s.get('shuffle_perm_sha') or '')[:8],
                       n_present=A.get('n_concepts_present'), seconds=round(s.get('seconds', 0)), shared_seconds=round(s.get('shared_seconds', 0)),
                       shared_epochs=s.get('shared_epochs'), max_memory_gb=round(s.get('max_memory_gb', 0), 2))
        if os.path.exists(p('history.json')) and not finite(json.load(open(p('history.json')))): issues.append('history non-finite')
        out['check'] = 'OK' if not issues else 'ISSUES ' + '; '.join(issues)
        print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
