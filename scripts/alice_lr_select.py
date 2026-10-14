"""D-092 REVE encoder learning-rate selection (subject fold 0 x segment fold 0, validation segment 4).

Per run: MRP of validation segment 4 averaged over the target subjects of subject fold 0, weighted by each target's number of
word-onset queries in that segment (= the query-weighted mean MRP over all target queries). The largest value wins; among the
learning rates whose value is within 0.002 of the largest, the smallest learning rate is chosen.
Checks: every run is eval_split valid, subject fold 0, segment fold 0, not frozen / phaserand / smoke / timing, scores only
segment 4, and all runs have the same target list and query counts.

usage: python scripts/alice_lr_select.py <run folder> <run folder> <run folder> [--out file.json]
"""
import os, sys, json, argparse, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIE = 0.002


def load(d):
    d = d if os.path.isabs(d) else os.path.join(ROOT, d)
    run = yaml.safe_load(open(os.path.join(d, 'config.yaml')))['run']; m = json.load(open(os.path.join(d, 'metrics.json')))
    assert run['arm'] == 'reve' and run['eval_split'] == 'valid' and run['subject_fold'] == 0 and run['segment_fold'] == 0, run
    assert not (run['frozen'] or run['phaserand'] or run['smoke'] or run['timing']) and run['encoder_lr'], run
    assert m['summary']['eval_segments'] == [4], m['summary']['eval_segments']
    cells = {s: t['segments']['4']['main'] for s, t in m['targets'].items()}
    assert all(list(t['segments']) == ['4'] for t in m['targets'].values())
    n = sum(c['n'] for c in cells.values())
    return dict(run=os.path.basename(d.rstrip('/')), encoder_lr=float(run['encoder_lr']), mrp=sum(c['mrp'] * c['n'] for c in cells.values()) / n,
                n_queries=n, targets={s: dict(mrp=c['mrp'], n=c['n']) for s, c in cells.items()},
                shared_best_epoch=m['summary']['shared_best_epoch'], shared_epochs=m['summary']['shared_epochs'], seconds=m['summary']['seconds'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs', nargs='+'); ap.add_argument('--out'); a = ap.parse_args()
    R = sorted((load(d) for d in a.runs), key=lambda r: r['encoder_lr'])
    assert len({r['encoder_lr'] for r in R}) == len(R)
    ref = {s: v['n'] for s, v in R[0]['targets'].items()}
    assert all({s: v['n'] for s, v in r['targets'].items()} == ref for r in R), 'target lists or query counts differ'
    best = max(r['mrp'] for r in R); within = [r for r in R if best - r['mrp'] <= TIE]; pick = within[0]   # R is sorted by lr
    for r in R: print(f"lr {r['encoder_lr']:g}: valid seg 4 MRP {r['mrp']:.4f} (n {r['n_queries']}, shared best epoch {r['shared_best_epoch']}/{r['shared_epochs']})", flush=True)
    out = dict(rule=f'query-weighted mean MRP over targets, validation segment 4; max, ties within {TIE} -> smallest lr', targets=sorted(ref, key=int),
               best_mrp=best, within_tie=[r['encoder_lr'] for r in within], chosen_encoder_lr=pick['encoder_lr'], runs=R)
    print('CHOSEN', json.dumps(dict(chosen_encoder_lr=pick['encoder_lr'], best_mrp=best, within_tie=out['within_tie'])), flush=True)
    if a.out: json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
