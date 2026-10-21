"""Fact-check reports/07_fiv_stage/notes/t2_gate0T2_result_kr.md against results/t2 (no new computation; reads JSON only).

Usage: python scripts/fiv_verify_t2_note.py OUT.json   (prints one line per value: O/X, item, note value, file value)
"""
import json, sys
g=json.load(open('results/t2/gate0T2.json'))
m=json.load(open('results/t2/matched_sets.json'))
a=json.load(open('results/t2/match_attrs.json'))
rows=[]
def r3(x): return f"{x:+.3f}"
def chk(name, note, raw, fmt):
    val=fmt(raw)
    rows.append((name, note, raw, val, 'O' if val==note else 'X'))
f3=lambda x: f"{x:.3f}"; s3=lambda x: f"{x:+.3f}".replace('-','−')
ci=lambda c: f"[{s3(c[0])}, {s3(c[1])}]"
ci0=lambda c: f"[{f3(c[0])}, {f3(c[1])}]"
for L,n in zip(['L0','L1','L2','L3'],['0.047','0.638','0.821','0.921']):
    chk(f'정합 비율 {L}', n, g['level_coverage'][L], f3)
chk('선택 단계', 'L2', g['level'], str)
chk('유효 질의', '11,220', g['n_queries_valid'], lambda x:f"{x:,}")
chk('남은 질의', '9,208', g['n_queries_retained'], lambda x:f"{x:,}")
chk('남은 질의(matched_sets 집합 수)', '9,208', len(m['sets']), lambda x:f"{x:,}")
pr=g['coverage_per_run'].values()
chk('실행별 정합 비율 범위', '0.75~0.88', (min(pr),max(pr)), lambda t:f"{t[0]:.2f}~{t[1]:.2f}")
chk('다른 실행 후보 비율', '80.5%', g['candidates_other_run_share'], lambda x:f"{100*x:.1f}%")
# recount from matched_sets
oth=sum(c.split('_')[0]!=s['q'].split('_')[0] for s in m['sets'] for c in s['cands']); tot=sum(len(s['cands']) for s in m['sets'])
chk('다른 실행 후보 비율(matched_sets 재집계)', '80.5%', oth/tot, lambda x:f"{100*x:.1f}%")
chk('cands_same_run 찬 질의', '2,416', g['same_run_lists_full'], lambda x:f"{x:,}")
chk('cands_same_run 찬 질의(matched_sets 재집계)', '2,416', sum(len(s['cands_same_run'])==19 for s in m['sets']), lambda x:f"{x:,}")
chk('후보 수 19(모든 집합)', '19', set(len(s['cands']) for s in m['sets']), lambda x:','.join(map(str,sorted(x))))
R=g['residuals']
chk('글자 수 상대 차 중앙값·p90','0.065·0.129',(R['letters_rel']['median'],R['letters_rel']['p90']),lambda t:f"{t[0]:.3f}·{t[1]:.3f}")
chk('음절 수 차 중앙값·p90','1·1',(R['syllables_abs']['median'],R['syllables_abs']['p90']),lambda t:f"{t[0]:g}·{t[1]:g}")
chk('TTS 발화 시간 상대 차','0.068·0.131',(R['tts_dur_rel']['median'],R['tts_dur_rel']['p90']),lambda t:f"{t[0]:.3f}·{t[1]:.3f}")
chk('TTS 창 단어 수 차','1·3',(R['tts_window_words_abs']['median'],R['tts_window_words_abs']['p90']),lambda t:f"{t[0]:g}·{t[1]:g}")
chk('TTS 창 글자 수 상대 차','0.115·0.293',(R['tts_window_letters_rel']['median'],R['tts_window_letters_rel']['p90']),lambda t:f"{t[0]:.3f}·{t[1]:.3f}")
I=g['i_information']
chk('(i) MRP','0.985',I['mrp'],f3); chk('(i) MRP 90% 구간','[0.981, 0.989]',I['mrp_ci90'],ci0)
chk('(i) 초과분','0.483',I['excess'],f3); chk('(i) 초과분 구간','[0.479, 0.487]',I['excess_ci90'],ci0)
chk('(i) 귀무 p97.5','0.513',I['null_p975'],f3)
chk('귀무 제외 쌍','3.0%',I['null_dropped_pair_share'],lambda x:f"{100*x:.1f}%")
chk('귀무 제외 후보','7.7%',I['null_dropped_cand_share'],lambda x:f"{100*x:.1f}%")
pm=[(k,v['mrp']) for k,v in I['per_run'].items()]
lo=min(pm,key=lambda t:t[1]); hi=max(pm,key=lambda t:t[1])
chk('실행별 MRP 범위','0.942(실행 15)~0.995',(lo,hi),lambda t:f"{t[0][1]:.3f}(실행 {t[0][0]})~{t[1][1]:.3f}")
D=g['report']['delay_search']
chk('지연 탐색 MRP','0.991',D['mrp'],f3); chk('지연 탐색 구간','[0.986, 0.995]',D['mrp_ci90'],ci0); chk('지연 탐색 초과분','0.487',D['excess'],f3)
U=g['report']['unmatched_i']
chk('무정합 MRP','0.987',U['mrp'],f3); chk('무정합 구간','[0.983, 0.990]',U['mrp_ci90'],ci0); chk('무정합 초과분','0.486',U['excess'],f3)
note={'LOWONLY_PLUS':('+0.141','[+0.131, +0.152]','+0.043','[+0.033, +0.051]'),
'TTS_DUR':('+0.198','[+0.190, +0.206]','+0.122','[+0.110, +0.132]'),
'PAUSE_PUNCT':('+0.259','[+0.253, +0.265]','+0.221','[+0.210, +0.231]'),
'RATE':('+0.158','[+0.150, +0.165]','+0.147','[+0.135, +0.159]'),
'SENTPOS':('+0.237','[+0.225, +0.247]','+0.203','[+0.187, +0.219]'),
'ORDER':('+0.015','[+0.005, +0.025]','+0.002','[−0.007, +0.011]')}
fails=0
for k,(te,tc,ae,ac) in note.items():
    for dom,e,c in (('tts',te,tc),('audio',ae,ac)):
        M=g['meters'][k+'/'+dom]
        chk(f'{k}/{dom} 초과분',e,M['excess'],s3); chk(f'{k}/{dom} 90% 구간',c,M['excess_ci90'],ci)
        fails+= M['excess_ci90'][1]>=0.02
chk('불만족 칸 수(상한 ≥ 0.02)','11',fails,str)
mr=[(f"{k}/{d}",g['meters'][k+'/'+d]['mrp']) for k in note for d in ('tts','audio')]
chk('원시 MRP 최소','0.507(ORDER/audio)',min(mr,key=lambda t:t[1]),lambda t:f"{t[1]:.3f}({t[0]})")
chk('원시 MRP 최대','0.757(PAUSE_PUNCT/tts)',max(mr,key=lambda t:t[1]),lambda t:f"{t[1]:.3f}({t[0]})")
E=g['report']['ENV_T_lag0']; EL=g['report']['ENV_T_lagmax']
chk('ENV_T 지연 0 MRP','0.804',E['mrp'],f3); chk('ENV_T 지연 0 초과분','+0.305',E['excess'],s3); chk('ENV_T 지연 0 구간','[+0.296, +0.314]',E['excess_ci90'],ci)
chk('ENV_T 최댓값 MRP','0.789',EL['mrp'],f3); chk('ENV_T 최댓값 초과분','+0.290',EL['excess'],s3)
UL=g['report']['unmatched_LOWONLY_PLUS_tts']
chk('무정합 LOWONLY+ MRP','0.750',UL['mrp'],f3); chk('무정합 LOWONLY+ 초과분','+0.250',UL['excess'],s3); chk('무정합 LOWONLY+ 구간','[+0.241, +0.257]',UL['excess_ci90'],ci)
chk('판정','fail',g['verdict'],str); chk('조건 (i)(ii)(iii)','true,false,true',g['conditions'],lambda d:f"{str(d['i_information']).lower()},{str(d['ii_meters']).lower()},{str(d['iii_coverage']).lower()}")
chk('match_attrs 질의 수','11,220',len(a['queries']),lambda x:f"{x:,}")
chk('gate0T2.json 기록 match_attrs sha','9e7863e0',g['match_attrs_sha256'][:8],str)
chk('gate0T2.json 기록 matched_sets sha','5f153a93',g['matched_sets_sha256'][:8],str)
t1=json.load(open('results/t1_tts/gate0T.json'))['report']['duration_ratio']
chk('0-T 길이 비 평균(results/t1_tts/gate0T.json)','0.85',sum(t1.values())/len(t1),lambda x:f"{x:.2f}")
json.dump([list(map(lambda z: z if isinstance(z,(str,int,float)) else json.dumps(z,default=str),r)) for r in rows],open(sys.argv[1],'w'),ensure_ascii=False,indent=0)
for r in rows: print(r[4], r[0], '|', r[1], '|', r[3])
