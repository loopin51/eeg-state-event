"""Format check of fetched F-V development run folders (k3s runbook section 6; D-112). Prints no development metric values.

Checks per folder: k3s attempts (1 expected; more = preemption restart, reported), commit = expected SHA, the six files,
summary.device contains 'RTX 5080', 9 trainers without the target, dev.A_ses4 n_candidates 200 and present + missing = 200,
every number in history.json finite, plus timing fields (summary.seconds, shared_seconds, shared_epochs, max_memory_gb).

usage: python scripts/fv_check_dev.py <expected sha> results/runs/<folder> [...]
"""
import json, math, os, sys

FILES = ['config.yaml', 'history.json', 'metrics.json', 'shared.pt', 'layers.pt', 'dev_scores.npz']


def finite(x):
    if isinstance(x, float): return math.isfinite(x)
    if isinstance(x, dict): return all(finite(v) for v in x.values())
    if isinstance(x, list): return all(finite(v) for v in x)
    return True


def main():
    sha, folders = sys.argv[1], sys.argv[2:]
    for d in folders:
        p = lambda f: os.path.join(d, f); issues = []
        att = open(p('k3s_attempts.txt')).read().strip().splitlines() if os.path.exists(p('k3s_attempts.txt')) else []
        com = open(p('k3s_commit.txt')).read().strip() if os.path.exists(p('k3s_commit.txt')) else ''
        if len(att) != 1: issues.append(f'attempts={len(att)}')
        if com != sha: issues.append(f'commit={com[:12]}')
        miss = [f for f in FILES if not os.path.exists(p(f))]
        if miss: issues.append(f'missing={miss}')
        out = dict(folder=os.path.basename(d), attempts=len(att))
        if os.path.exists(p('metrics.json')):
            m = json.load(open(p('metrics.json'))); s = m['summary']; A = m.get('dev', {}).get('A_ses4', {})
            if 'RTX 5080' not in s.get('device', ''): issues.append(f'device={s.get("device")}')
            tr = s.get('trainers', [])
            if len(tr) != 9 or s.get('target') in tr: issues.append(f'trainers={tr}')
            if A.get('n_candidates') != 200 or A.get('n_concepts_present', 0) + A.get('n_concepts_missing', 0) != 200:
                issues.append('A_ses4 counts')
            out.update(target=s.get('target'), candidate=s.get('candidate'), seed=s.get('seed'), shuffle=s.get('shuffle'), shuffle_perm_sha=(s.get('shuffle_perm_sha') or '')[:8],
                       n_present=A.get('n_concepts_present'), seconds=round(s.get('seconds', 0)), shared_seconds=round(s.get('shared_seconds', 0)),
                       shared_epochs=s.get('shared_epochs'), max_memory_gb=round(s.get('max_memory_gb', 0), 2))
        if os.path.exists(p('history.json')) and not finite(json.load(open(p('history.json')))): issues.append('history non-finite')
        out['check'] = 'OK' if not issues else 'ISSUES ' + '; '.join(issues)
        print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
