"""D-110 seal list for UGR-MINDVOICE: which EEG samples may be read (Real main-trial blocks) and which are sealed (Silent).

Built from the BIDS events of the eeg folder only (no EDF is opened). A trial runs from StartFixation to EndITI of the same item
(1-kHz 'sample' column); a block is a run of consecutive trials of one mode with gaps below block_gap_ms.
  real  = Real Experiment blocks, [first StartFixation - real_preroll_ms, last EndITI)       (read ranges)
  silent = Silent blocks of every part, [first StartFixation, last EndITI)                    (sealed ranges)
The script asserts that no read range meets a sealed range widened by silent_margin_ms on both sides, nor any practice
trial. It writes the canonical JSON (sorted keys, no spaces) and prints the sha256 of the real list, the silent list and the
whole file; D-110 registers the three values before any EDF is read. Loaders call check() with the decoded sample ranges.

usage: python scripts/ugr_seal.py configs/ugr_u.yaml
"""
import os, re, csv, glob, json, hashlib, argparse, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r'^(Silent|Real)(Words|Syllables)(Experiment|Practice)(Start|End)(Fixation|Stimulus|ISI|Speech|ITI):(Audio|Text|Pictures)_(.*)$')


def trials(path):
    out, cur = [], None
    for r in csv.DictReader(open(path, encoding='utf-8-sig'), delimiter='\t'):
        m = PAT.match(r['trial_type'])
        assert m, f'{path}: unparsed trial_type {r["trial_type"]!r}'
        mode, kind, part, se, phase, cue, item = m.groups(); s = int(r['sample'])
        assert abs(s - round(float(r['onset']) * 1000)) <= 1, f'{path}: sample/onset mismatch'
        if se == 'Start' and phase == 'Fixation': cur = dict(mode=mode, part=part, item=item, start=s)
        elif se == 'End' and phase == 'ITI' and cur is not None and cur['item'] == item: cur['end'] = s; out.append(cur); cur = None
    return out


def blocks(ts, gap):
    bl = []
    for t in ts:
        if bl and bl[-1]['mode'] == t['mode'] and t['start'] - bl[-1]['end'] < gap: bl[-1]['end'] = t['end']
        else: bl.append(dict(mode=t['mode'], start=t['start'], end=t['end']))
    return bl


def overlap(a, b): return a[0] < b[1] and b[0] < a[1]


def build(cfg):
    sc, root = cfg['seal'], cfg['data_root']; sess = {}
    for f in sorted(glob.glob(os.path.join(root, cfg['events_glob']))):
        key = '/'.join(f.split('/')[-4:-2]); ts = trials(f)
        read = [t for t in ts if t['mode'] == 'Real' and t['part'] in sc['read_parts']]
        sealed = [t for t in ts if t['mode'] == 'Silent' and t['part'] in sc['sealed_parts']]
        practice = [(t['start'], t['end']) for t in ts if t['part'] == 'Practice']
        real = [[b['start'] - sc['real_preroll_ms'], b['end']] for b in blocks(read, sc['block_gap_ms'])]
        silent = [[b['start'], b['end']] for b in blocks(sealed, sc['block_gap_ms'])]
        m = sc['silent_margin_ms']
        for r in real:
            assert r[0] >= 0, f'{key}: read range before recording start'
            assert not any(overlap(r, (s[0] - m, s[1] + m)) for s in silent), f'{key}: read range {r} meets a sealed range +-{m} ms'
            assert not any(overlap(r, p) for p in practice), f'{key}: read range {r} meets a practice trial'
        sess[key] = dict(real=real, silent=silent, n_real_trials=len(read), n_silent_trials=len(sealed), n_practice_trials=len(practice))
    return dict(rule=dict(sc, events_glob=cfg['events_glob']), sessions=sess)


def canon(o): return json.dumps(o, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def hashes(seal):
    s = seal['sessions']
    return dict(real=hashlib.sha256(canon({k: v['real'] for k, v in s.items()})).hexdigest(),
                silent=hashlib.sha256(canon({k: v['silent'] for k, v in s.items()})).hexdigest(),
                file=hashlib.sha256(canon(seal)).hexdigest())


def check(seal, registered, session, decoded):
    """Raise unless the seal file matches the registered sha256 values and every decoded [start, end) lies inside one read
    range and away from every sealed range +- margin. decoded: list of (start, end) sample ranges actually decoded."""
    h = hashes(seal); assert h == registered, f'seal sha256 {h} != registered {registered}'
    s, m = seal['sessions'][session], seal['rule']['silent_margin_ms']
    for d in decoded:
        assert any(r[0] <= d[0] and d[1] <= r[1] for r in s['real']), f'{session}: decoded {d} outside the read ranges'
        assert not any(overlap(d, (x[0] - m, x[1] + m)) for x in s['silent']), f'{session}: decoded {d} meets a sealed range'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); seal = build(cfg)
    out = os.path.join(ROOT, cfg['seal']['out']); os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'wb').write(canon(seal))
    for k, v in seal['sessions'].items():
        print(f"{k}: read blocks {len(v['real'])} ({sum(e - s for s, e in v['real']) / 60000:.1f} min, {v['n_real_trials']} trials), "
              f"sealed blocks {len(v['silent'])} ({sum(e - s for s, e in v['silent']) / 60000:.1f} min, {v['n_silent_trials']} trials), "
              f"practice {v['n_practice_trials']}")
    print(json.dumps(dict(sha256=hashes(seal), out=cfg['seal']['out'], sessions=len(seal['sessions']))))


if __name__ == '__main__':
    main()
