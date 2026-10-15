"""D-100 (c): UGR-MINDVOICE (OSF 6sh5d) CPU entry check from metadata only (events / channels / sidecars). No EEG signal is read.

Per subject and session: main-experiment trials by condition (Real = overt, Silent = covert; Syllables / Words), cue modality,
distinct items and repetitions, the condition sequence (mini-block lengths and the position after an overt block), phase
durations (fixation, stimulus, ISI, speech, ITI), trials per minute, channel types, overt-speech audio events. Then the R-d MDE
(D-100 (b)) of a session-level split from trial counts only.

usage: python scripts/d100_ugr_entry.py META_DIR OUT_JSON
"""
import os, re, sys, csv, json, glob, math, collections


def wolpaw(p, K): return math.log2(K) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (K - 1))


def mde_bits_min(n, K, rate):
    p0 = 1 / K; p = p0 + 2.486 * math.sqrt(p0 * (1 - p0) / n); return p, wolpaw(min(p, 0.999), K) * rate


def main():
    meta, out = sys.argv[1], sys.argv[2]; D = os.path.join(meta, 'files', 'Data'); res = {'sessions': {}, 'listing_files': None}
    L = json.load(open(os.path.join(meta, 'listing.json'))); res['listing_files'] = dict(n=len(L), total_bytes=sum(x['size'] or 0 for x in L),
                                                                                    edf=sum(1 for x in L if x['path'].endswith('.edf')), wav=sum(1 for x in L if x['path'].endswith('.wav')))
    for f in sorted(glob.glob(os.path.join(D, 'sub-*', 'ses-*', 'eeg', '*_events.tsv'))):
        sub, ses = re.findall(r'(sub-\d+)', f)[0], re.findall(r'(ses-\d+)', f)[0]; rows = list(csv.DictReader(open(f, encoding='utf-8-sig'), delimiter='\t'))
        trials = []; cur = None
        for r in rows:
            a, _, b = r['trial_type'].partition(':'); m = re.match(r'(Real|Silent)(Syllables|Words)(Experiment|Practice)(Start|End)(\w+)$', a)
            if not m: continue
            mode, kind, part, se, phase = m.groups(); t = float(r['onset'])
            if se == 'Start' and phase == 'Fixation':
                cur = dict(mode=mode, kind=kind, part=part, cue=b.split('_')[0], item='_'.join(b.split('_')[1:]), t0=t, ph={}); trials.append(cur)
            if cur is not None: cur['ph'].setdefault(phase, {})[se] = t
        main = [x for x in trials if x['part'] == 'Experiment']
        seq = [x['mode'] + x['kind'] for x in main]; blocks = []
        for k in seq:
            if blocks and blocks[-1][0] == k: blocks[-1][1] += 1
            else: blocks.append([k, 1])
        dur = {p: [x['ph'][p]['End'] - x['ph'][p]['Start'] for x in main if p in x['ph'] and 'End' in x['ph'][p] and 'Start' in x['ph'][p]] for p in ('Fixation', 'Stimulus', 'ISI', 'Speech', 'ITI')}
        per = [b['t0'] - a['t0'] for a, b in zip(main[:-1], main[1:]) if 0 < b['t0'] - a['t0'] < 20]
        cnt = collections.Counter((x['mode'], x['kind']) for x in main); cue = collections.Counter((x['mode'], x['kind'], x['cue']) for x in main)
        items = {f'{m_}{k_}': collections.Counter(x['item'] for x in main if x['mode'] == m_ and x['kind'] == k_) for m_ in ('Real', 'Silent') for k_ in ('Syllables', 'Words')}
        after_overt = sum(1 for i in range(1, len(main)) if main[i]['mode'] == 'Silent' and main[i - 1]['mode'] == 'Real')
        ch = glob.glob(os.path.join(os.path.dirname(f), '*_channels.tsv')); chtypes = collections.Counter(r['type'] for r in csv.DictReader(open(ch[0], encoding='utf-8-sig'), delimiter='\t')) if ch else {}
        sj = glob.glob(os.path.join(os.path.dirname(f), '*_eeg.json')); side = json.load(open(sj[0], encoding='utf-8-sig')) if sj else {}
        aud = glob.glob(os.path.join(os.path.dirname(os.path.dirname(f)), 'audio', '*_events.tsv'))
        res['sessions'][f'{sub}_{ses}'] = dict(
            n_main_trials=len(main), counts={f'{a}{b}': v for (a, b), v in cnt.items()}, cue_counts={f'{a}{b}_{c}': v for (a, b, c), v in cue.items()},
            distinct_items={k: len(v) for k, v in items.items()}, reps_per_item={k: (min(v.values()), max(v.values())) if v else None for k, v in items.items()},
            n_blocks=len(blocks), block_lengths=sorted(collections.Counter(b[1] for b in blocks).items()), block_order_head=[b[0] for b in blocks[:12]],
            silent_trials_right_after_overt=after_overt,
            phase_mean_s={p: (round(sum(v) / len(v), 3) if v else None) for p, v in dur.items()}, phase_sd_s={p: (round((sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) ** 0.5, 3) if v else None) for p, v in dur.items()},
            trial_period_s=round(sum(per) / len(per), 3) if per else None, trials_per_min=round(60 / (sum(per) / len(per)), 2) if per else None,
            channel_types=dict(chtypes), sampling_hz=side.get('SamplingFrequency'), duration_s=side.get('RecordingDuration'), overt_audio_events=bool(aud))
    S = res['sessions']; subs = sorted({k.split('_')[0] for k in S}); res['subjects'] = len(subs)
    res['sessions_per_subject'] = {s: sorted(k.split('_')[1] for k in S if k.startswith(s + '_')) for s in subs}
    two = [s for s, v in res['sessions_per_subject'].items() if len(v) >= 2]; res['subjects_with_two_sessions'] = two
    # R-d MDE from counts: covert words, session-level split (test = one session of a two-session subject), K = distinct covert words
    ex = next(iter(S.values())); K = ex['distinct_items'].get('SilentWords') or 0; rate = ex['trials_per_min'] or 0
    n_test_per_subject = [S[f'{s}_ses-02']['counts'].get('SilentWords', 0) for s in two if f'{s}_ses-02' in S]
    if K > 1 and n_test_per_subject:
        p1, b1 = mde_bits_min(min(n_test_per_subject), K, rate); pp, bp = mde_bits_min(sum(n_test_per_subject), K, rate)
        res['R_d_example'] = dict(classes=K, trials_per_min=rate, per_subject_n=min(n_test_per_subject), per_subject_p_mde=p1, per_subject_bits_min=b1,
                                  pooled_n=sum(n_test_per_subject), pooled_p_mde=pp, pooled_bits_min=bp, threshold=0.3)
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != 'sessions'}, ensure_ascii=False, indent=1))
    print(json.dumps(ex, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
