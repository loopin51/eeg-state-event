"""Fact-check reports/07_fiv_stage/notes/sparrkulee_audit_kr.md (eeg-a4) against cache metadata and the RDR file list.

Reads only metadata (no EEG values): data/cache/sparrkulee/eeg_meta.json, stimuli.json,
data/raw/sparrkulee/_meta.json and results/d099_ext/weights.json. Prints the recomputed values.
Run on the server from the project folder: env/bin/python -I scripts/fiv_verify_sparrkulee.py
"""
import collections, json, re, statistics as st

m = json.load(open('data/cache/sparrkulee/eeg_meta.json'))
stimj = json.load(open('data/cache/sparrkulee/stimuli.json'))['stimuli']
fs = m['sample_rate']
allrec = [(int(s), r) for s, rs in m['subjects'].items() for r in rs]
R = {int(s): [r for r in rs if 'excluded' not in r] for s, rs in m['subjects'].items()}
grp = lambda n: re.sub(r'^(audiobook_\d+).*$', r'\1', n)          # D-099 stim_group_rules
mins = lambda rs: sum(r['length'] for r in rs) / fs / 60
clean = lambda r: r['snr'] == 100.0 and 'shifted' not in r['stim']  # no SNR -4/2.5 noise, no pitch-shifted copy


def gname(s):
    ses = R[s][0]['session']
    return 'SS01' if 'shortstories' in ses else 'VS%02d' % int(ses[-2:])


gk = lambda g: (int(g.split('_')[1]) + (0 if g.startswith('a') else 100), g)
GS = collections.defaultdict(list)
for s in sorted(R):
    GS[gname(s)].append(s)

print('== records')
print('entries', len(allrec), 'excluded', [(s, r['rec'], r['excluded']) for s, r in allrec if 'excluded' in r])
print('kept', sum(map(len, R.values())), 'hours %.2f' % (sum(mins(v) for v in R.values()) / 60))
G = sorted({grp(r['stim']) for v in R.values() for r in v})
print('story groups', len(G), 'audiobook', sum(g.startswith('audiobook') for g in G), 'podcast', sum(g.startswith('podcast') for g in G))
used = sorted({r['stim'] for v in R.values() for r in v})
print('stimulus files used', len(used), 'hours %.2f' % (sum(stimj[u]['duration_s'] for u in used) / 3600))

print('== groups')
for g in sorted(GS):
    ss = GS[g]
    pm = [mins(R[s]) for s in ss]
    print(g, len(ss), ','.join(map(str, ss)), ','.join(sorted({grp(r['stim']) for s in ss for r in R[s]}, key=gk)),
          '%.1f~%.1f' % (min(pm), max(pm)), '%.2f h' % (sum(pm) / 60))
lis = collections.defaultdict(set)
for s in R:
    for r in R[s]:
        lis[grp(r['stim'])].add(s)
print('shared across groups', {g: len(v) for g, v in lis.items() if len({gname(s) for s in v}) > 1},
      'audiobook_1 missing', sorted(set(R) - lis['audiobook_1']))
print('noise / pitch / no_noise records', [(s, r['stim'], r['condition'], r['snr']) for s in R for r in R[s] if not clean(r) or r['condition'] == 'no_noise'])

print('== split candidates (selection VS06, gate VS05+VS07)')
sel, gate = GS['VS06'], GS['VS05'] + GS['VS07']
for name, tg in (('A', ['VS01', 'VS02', 'VS03', 'VS04']), ('B', ['VS08', 'VS09', 'VS10']), ('C', ['SS01'])):
    test = [s for g in tg for s in GS[g]]
    train = [s for s in R if s not in test + sel + gate]
    held = {grp(r['stim']) for s in test + sel + gate for r in R[s]} - {'audiobook_1'}
    ts = sorted({grp(r['stim']) for s in test for r in R[s]} - {'audiobook_1'}, key=gk)
    tb = sorted(mins([r for r in R[s] if grp(r['stim']) in ts and clean(r)]) for s in test)
    u_clean = {r['stim'] for s in test for r in R[s] if grp(r['stim']) in ts and clean(r)}
    u_all = {r['stim'] for s in test for r in R[s] if grp(r['stim']) in ts}
    cal = {s: round(mins([r for r in R[s] if grp(r['stim']) == 'audiobook_1' and clean(r)]), 1) for s in test}
    print(name, 'train/sel/gate/test %d/%d/%d/%d' % (len(train), len(sel), len(gate), len(test)),
          'stories %d (audiobook %d, podcast %d)' % (len(ts), sum(t.startswith('a') for t in ts), sum(t.startswith('p') for t in ts)),
          'unique audio clean %.1f min (%d files), incl. noise %.1f min (%d files)' % (
              sum(stimj[x]['duration_s'] for x in u_clean) / 60, len(u_clean), sum(stimj[x]['duration_s'] for x in u_all) / 60, len(u_all)),
          'test bundle min %.1f median %.1f (median_high %.1f) max %.1f, >=15 min: %d' % (tb[0], st.median(tb), st.median_high(tb), tb[-1], sum(x >= 15 for x in tb)),
          'train EEG %.2f h, dropped %.2f h' % (sum(mins(R[s]) for s in train) / 60, sum(mins([r for r in R[s] if grp(r['stim']) in held]) for s in train) / 60),
          'calibration != 15.6:', {s: v for s, v in cal.items() if v != 15.6})


def bundle(ss):
    stories = {grp(r['stim']) for s in ss for r in R[s]} - {'audiobook_1'}
    return [mins([r for r in R[s] if grp(r['stim']) in stories and clean(r)]) for s in ss]


print('selection bundle %.1f~%.1f, gate bundle %.1f~%.1f' % (min(bundle(sel)), max(bundle(sel)), min(bundle(gate)), max(bundle(gate))))

print('== P2 valid groups')
w = json.load(open('results/d099_ext/weights.json'))['data']['sparrkulee']
VG = w['valid_groups']
print('weights.json valid_groups', VG, 'valid_hours %.2f train_hours %.2f' % (w['valid_hours'], w['train_hours']))
tot, uall, subs = 0.0, set(), set()
for g in sorted(VG, key=gk):
    rs = [(s, r) for s in R for r in R[s] if grp(r['stim']) == g]
    u = {r['stim'] for _, r in rs}
    e = sum(r['length'] for _, r in rs) / fs / 60
    tot += e; uall |= u; subs |= {s for s, _ in rs}
    print(g, sorted({gname(s) for s, _ in rs}), len({s for s, _ in rs}), 'EEG %.1f min' % e, 'audio %.1f min' % (sum(stimj[x]['duration_s'] for x in u) / 60))
print('total EEG %.1f min = %.2f h, unique audio %.1f min, subjects %d' % (tot, tot / 60, sum(stimj[x]['duration_s'] for x in uall) / 60, len(subs)))

print('== appendix A (subject group n_rec EEG_min n_groups audiobook_1_clean_min P2_valid_min noise_or_shift)')
for s in sorted(R):
    rs = R[s]
    nz = sorted(r['stim'] for r in rs if not clean(r) or r['condition'] == 'no_noise')
    print('%03d %s %d %.1f %d %.1f %.1f %s' % (s, gname(s), len(rs), mins(rs), len({grp(r['stim']) for r in rs}),
          mins([r for r in rs if grp(r['stim']) == 'audiobook_1' and clean(r)]), mins([r for r in rs if grp(r['stim']) in VG]), '; '.join(nz)))

print('== RDR file list')
M = json.load(open('data/raw/sparrkulee/_meta.json'))['data']['latestVersion']
F = M['files']


def top(f):
    d = f.get('directoryLabel', '') or ''
    for p in ('stimuli/eeg', 'derivatives/preprocessed_eeg', 'derivatives/preprocessed_stimuli'):
        if d.startswith(p):
            return p
    return 'sub-*/ses-*' if d.startswith('sub-') else '(root)'


ext = lambda f: re.sub(r'^[^.]*', '', f['dataFile']['filename'])
print('files', len(F), 'restricted', sum(bool(f.get('restricted')) for f in F), 'license', M['license']['name'])
for t, n in sorted(collections.Counter(map(top, F)).items()):
    print(t, n, dict(collections.Counter(ext(f) for f in F if top(f) == t)), 'restricted', sum(bool(f.get('restricted')) for f in F if top(f) == t))
print('restricted subjects', sorted({tuple(f['directoryLabel'].split('/')[:2]) for f in F if f.get('restricted')}))
print('transcript/alignment-like files', [f['dataFile']['filename'] for f in F if re.search(r'(?i)transcr|align|textgrid|\.csv$', f['dataFile']['filename'])])
