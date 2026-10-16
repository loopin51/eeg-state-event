"""Fact-check the numbers of reports/fiv_stage/fiv_stage_report.tex (eeg-dir3 scope) against result files.

Scope (coordinator request): abstracts, sections 1, 2.1, 2.2, 2.4, 3.1, 3.2, 3.5, 3.6, 4, 5, appendices B and C, and the
figure captions 1-3. UGR sections 2.3, 3.3, 3.4 and figure 4 are checked by eeg-a4. No new computation: values are
read from committed JSON files and from a read-only dump of the server gpuq queue (jobs 446-477 of this project).

Each check is (label, literal as written in the tex, expected literal from the source, source). A check passes when the
two literals are equal; the tex lines that contain the written literal (inside the scope) are reported. Afterwards every
number token in the scope lines that no check covers is listed, so that nothing is skipped silently.

Usage: python scripts/fiv_verify_report.py GPUQ_DUMP.json OUT.json [TEX_COPY]
"""
import hashlib, json, re, sys

TEX = sys.argv[3] if len(sys.argv) > 3 else 'reports/fiv_stage/fiv_stage_report.tex'   # optional frozen copy of the draft
lines = open(TEX, encoding='utf-8').read().split('\n')
# scope by markers (the draft is still being edited, so line numbers are found at run time)
MARK = [('\\begin{koabstract}{국문 초록}', '초록', 1), ('\\section{서론}', '1절', 1), ('\\section{방법}', None, 0),
        ('\\subsection{T2와 자극 관문', '2.1', 1), ('\\subsection{재추론 점검과 B1 밀도 감사(', '2.2', 1),
        ('\\subsection{UGR-MINDVOICE U0', None, 0), ('\\subsection{두 감사와 장치}', '2.4', 1), ('\\section{결과}', None, 0),
        ('\\subsection{0-T2:', '3.1', 1), ('\\subsection{재추론 점검과 B1 밀도 감사}', '3.2', 1), ('\\subsection{UGR U0:', None, 0),
        ('\\subsection{두 감사}', '3.5', 1), ('\\subsection{실행 기록}', '3.6', 1), ('\\section{논의}', '4절', 1),
        ('\\section{결론}', '5절', 1), ('\\section*{부록 A.', None, 0), ('\\section*{부록 B.', '부록 B', 1),
        ('\\section*{부록 C.', '부록 C', 1), ('\\section*{부록 D.', '부록 D', 1),
        ('% ================================================================ 참고문헌', None, 0)]
pos = [(next(i for i, l in enumerate(lines) if l.startswith(m)) + 1, name) for m, name, _ in MARK]
SCOPE = [(a, b - 1, name) for (a, name), (b, _) in zip(pos, pos[1:]) if name]
inscope = {i: name for a, b, name in SCOPE for i in range(a, b + 1)}


def R(p):
    return json.load(open(p))


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


g, d, rc = R('results/t2/gate0T2.json'), R('results/t2/density_audit.json'), R('results/t2/reinfer_check.json')
u1, u0 = R('results/ugr/u1.json'), R('results/ugr/u0.json')
t1 = R('results/t1_tts/gate0T.json')
jobs = {j['id']: j for j in R(sys.argv[1])}
el = lambda i: jobs[i]['ended_at'] - jobs[i]['started_at']
f3, f4 = (lambda x: f'{x:.3f}'), (lambda x: f'{x:.4f}')
s3 = lambda x: f'{x:+.3f}'
M = lambda k: g['meters'][k]
checks = []


def C(label, written, expected, source):
    checks.append((label, written, expected, source))


# ---- 0-T2 (gate0T2.json)
for L, w in zip(['L0', 'L1', 'L2', 'L3'], ['0.047', '0.638', '0.821', '0.921']):
    C(f'정합 비율 {L}', w, f3(g['level_coverage'][L]), 'gate0T2.json level_coverage')
C('유효 질의', '11,220', f"{g['n_queries_valid']:,}", 'gate0T2.json n_queries_valid')
C('남은 질의', '9,208', f"{g['n_queries_retained']:,}", 'gate0T2.json n_queries_retained')
pr = g['coverage_per_run'].values()
C('실행별 정합 비율', '0.75--0.88', f'{min(pr):.2f}--{max(pr):.2f}', 'gate0T2.json coverage_per_run')
C('다른 실행 후보 비율', '80.5\\%', f"{100 * g['candidates_other_run_share']:.1f}\\%", 'gate0T2.json candidates_other_run_share')
Rz = g['residuals']
C('글자 수 잔차', '0.065(0.129)', f"{Rz['letters_rel']['median']:.3f}({Rz['letters_rel']['p90']:.3f})", 'gate0T2.json residuals')
C('음절 수 잔차', '차 1(1)', f"차 {Rz['syllables_abs']['median']:g}({Rz['syllables_abs']['p90']:g})", 'gate0T2.json residuals')
C('TTS 발화 시간 잔차', '0.068(0.131)', f"{Rz['tts_dur_rel']['median']:.3f}({Rz['tts_dur_rel']['p90']:.3f})", 'gate0T2.json residuals')
C('TTS 창 단어 수 차 p90', '백분위 3)', f"백분위 {Rz['tts_window_words_abs']['p90']:g})", 'gate0T2.json residuals')
I = g['i_information']
C('(i) MRP', '0.985', f3(I['mrp']), 'gate0T2.json i_information.mrp')
C('(i) 90% 구간', '0.981--0.989', f"{I['mrp_ci90'][0]:.3f}--{I['mrp_ci90'][1]:.3f}", 'gate0T2.json i_information.mrp_ci90')
pm = [v['mrp'] for v in I['per_run'].values()]
C('실행별 MRP', '0.942--0.995', f'{min(pm):.3f}--{max(pm):.3f}', 'gate0T2.json i_information.per_run')
C('지연 탐색 MRP', '0.991', f3(g['report']['delay_search']['mrp']), 'gate0T2.json report.delay_search')
C('무정합 MRP', '0.987', f3(g['report']['unmatched_i']['mrp']), 'gate0T2.json report.unmatched_i')
names = {'LOWONLY_PLUS': 'LOWONLY+', 'TTS_DUR': 'TTS\\_DUR', 'PAUSE_PUNCT': 'PAUSE·PUNCT', 'RATE': 'RATE', 'SENTPOS': 'SENTPOS', 'ORDER': 'ORDER'}
for k, nm in names.items():
    cells = []
    for dom in ('tts', 'audio'):
        e, (lo, hi) = M(f'{k}/{dom}')['excess'], M(f'{k}/{dom}')['excess_ci90']
        ex = f'${s3(e)}$' if not (k == 'PAUSE_PUNCT' and dom == 'tts') else f'$\\mathbf{{{s3(e)}}}$'
        los = f'${lo:.3f}$' if lo < 0 else f'{lo:.3f}'
        cells.append(f'{ex} [{los}, {hi:.3f}]')
    C(f'표 1 {nm}', next((l.strip() for l in lines[319:331] if l.startswith(nm + ' &')), '(행 없음)'),
      f'{nm} & {cells[0]} & {cells[1]} \\\\', 'gate0T2.json meters')
C('ORDER/들은 오디오 상한', '(상한 0.011)', f"(상한 {M('ORDER/audio')['excess_ci90'][1]:.3f})", 'gate0T2.json meters.ORDER/audio')
fails = sum(M(f'{k}/{dm}')['excess_ci90'][1] >= 0.02 for k in names for dm in ('tts', 'audio'))
C('불만족 칸 수', '11칸', f'{fails}칸', 'gate0T2.json meters(상한 ≥ 0.02)')
C('최대 초과분(PAUSE·PUNCT/TTS)', '+0.259', s3(M('PAUSE_PUNCT/tts')['excess']), 'gate0T2.json meters')
C('ENV_T 초과분', '+0.305', s3(g['report']['ENV_T_lag0']['excess']), 'gate0T2.json report.ENV_T_lag0')
C('무정합 LOWONLY+ 초과분', '+0.250에서 +0.141', f"{s3(g['report']['unmatched_LOWONLY_PLUS_tts']['excess'])}에서 {s3(M('LOWONLY_PLUS/tts')['excess'])}", 'gate0T2.json')
C('들은 오디오 휴지·리듬 값', '(PAUSE·PUNCT +0.221, SENTPOS +0.203, RATE +0.147, TTS\\_DUR +0.122)',
  f"(PAUSE·PUNCT {s3(M('PAUSE_PUNCT/audio')['excess'])}, SENTPOS {s3(M('SENTPOS/audio')['excess'])}, RATE {s3(M('RATE/audio')['excess'])}, TTS\\_DUR {s3(M('TTS_DUR/audio')['excess'])})", 'gate0T2.json meters')
four = ['PAUSE_PUNCT', 'SENTPOS', 'RATE']
tt = [M(f'{k}/tts')['excess'] for k in four]
au = [M(f'{k}/audio')['excess'] for k in four + ['TTS_DUR']]
C('4절 TTS 범위(휴지·구두점·문장 위치·리듬)', '+0.15--+0.26', f'+{min(tt):.2f}--+{max(tt):.2f}', 'gate0T2.json meters PAUSE·PUNCT, SENTPOS, RATE /tts')
C('4절 들은 오디오 범위', '+0.12--+0.22', f'+{min(au):.2f}--+{max(au):.2f}', 'gate0T2.json meters PAUSE·PUNCT, SENTPOS, RATE, TTS_DUR /audio')
# ---- 0-T (T1, section 1)
C('0-T LOWONLY 초과분', '0.188', f"{t1['meters']['LOWONLY']['excess']:.3f}", 'results/t1_tts/gate0T.json meters.LOWONLY')
C('0-T ORDER 90% 상한(1절에 없음)', '(1절에 ORDER 언급 없음)', f"ORDER 상한 {t1['meters']['ORDER']['excess_ci90'][1]:.3f} ≥ 0.02도 실패", 'results/t1_tts/gate0T.json meters.ORDER')
# ---- re-inference check and density audit
C('재추론 칸 수', '380칸', f"{rc['n_cells']}칸", 'reinfer_check.json n_cells')
C('칸 MRP 차 최대', '$2.86 \\times 10^{-6}$', f"${rc['max_abs_mrp_diff'] * 1e6:.2f} \\times 10^{{-6}}$", 'reinfer_check.json max_abs_mrp_diff')
C('차가 0인 칸', '377칸', f"{sum(c['mrp_diff'] == 0 for c in rc['cells'])}칸", 'reinfer_check.json cells')
C('점수 차 최대/칸 sd', '$8.37 \\times 10^{-6}$', f"${rc['max_rel_score_diff'] * 1e6:.2f} \\times 10^{{-6}}$", 'reinfer_check.json max_rel_score_diff')
C('밀도 감사 Q', '9,195', f"{d['n_queries']:,}", 'density_audit.json n_queries')
C('EEG 여유로 뺀 질의', '13개', f"{g['n_queries_retained'] - d['n_queries']}개", 'gate0T2.json − density_audit.json')
C('MRP_전체', '0.8776', f4(d['MRP_full']), 'density_audit.json MRP_full')
C('MRP_정합', '0.8770', f4(d['MRP_matched']), 'density_audit.json MRP_matched')
C('MRP_무정합', '0.8785', f4(d['MRP_unmatched']), 'density_audit.json MRP_unmatched')
C('R_dens', '0.998', f3(d['R_dens']), 'density_audit.json R_dens')
C('R_dens 90% 구간', '0.990--1.007', f"{d['R_dens_ci90'][0]:.3f}--{d['R_dens_ci90'][1]:.3f}", 'density_audit.json R_dens_ci90')
C('R_dens 19명', '0.999', f3(d['R_dens_19']), 'density_audit.json R_dens_19')
ps = [v for k, v in d['per_subject'].items() if k not in ('1', '15')]
C('피험자별(풀 17명)', '0.994--1.013', f'{min(ps):.3f}--{max(ps):.3f}', 'density_audit.json per_subject')
prr = list(d['per_run'].values())
C('실행별', '0.944--1.039', f'{min(prr):.3f}--{max(prr):.3f}', 'density_audit.json per_run')
C('sub-015', '$-0.778$', f"${d['per_subject']['15']:.3f}$", 'density_audit.json per_subject.15')
C('정합 − 무정합', '$-0.0016$', f"${d['MRP_matched'] - d['MRP_unmatched']:.4f}$", 'density_audit.json')
C('N_H 조건', '거짓', '거짓' if d['N_H_condition'] is False else '참', 'density_audit.json N_H_condition')
# ---- UGR numbers inside the scope (abstract, section 4, appendix B)
C('U0 단어 수', '단어 80개', f"단어 {u0['L0']['words_with_min_candidates']}개", 'u0.json L0.words_with_min_candidates')
C('U0 단어 수(영문)', 'with 80 words', f"with {u0['L0']['words_with_min_candidates']} words", 'u0.json')
C('U1 합친 p', '$p = 0.598$', f"$p = {u1['primary']['p']:.3f}$", 'u1.json primary.p')
oe = u1['report_single_session_odd_even']
C('홀짝 분할 양수', '9명 모두', f"{sum(v['D_uv'] > 0 for v in oe.values())}명 모두" if all(v['D_uv'] > 0 for v in oe.values()) else '일부', 'u1.json report_single_session_odd_even')
C('P100 p', '(P100\n$p = 0.001$', f"(P100\n$p = {u1['report_p100']['p']:.3f}$", 'u1.json report_p100.p')
FIG = open('reports/fiv_stage/make_figures.py', encoding='utf-8').read().split('\n')
FIGC = []


def F(label, written, expected, source):
    FIGC.append((label, written, expected, source))


F('그림 1 U1 6명 중 1명', '6명 중 1명', f"6명 중 {u1['primary']['k_of_6']}명", 'u1.json primary.k_of_6')
F('그림 1 P100 p', 'P100 p = 0.001', f"P100 p = {u1['report_p100']['p']:.3f}", 'u1.json report_p100.p')
F('그림 1 U0 단어 수', '단어 80개', f"단어 {u0['L0']['words_with_min_candidates']}개", 'u0.json L0.words_with_min_candidates')
F('그림 1 재추론 칸', '380칸, MRP 차 최대', f"{rc['n_cells']}칸, MRP 차 최대", 'reinfer_check.json n_cells')
F('그림 1 MRP 차 최대', '2.9e−6', f"{rc['max_abs_mrp_diff']:.1e}".replace('-0', '−'), 'reinfer_check.json max_abs_mrp_diff')
F('그림 1 정합 − 무정합', '−0.0016', f"{d['MRP_matched'] - d['MRP_unmatched']:.4f}".replace('-', '−'), 'density_audit.json')
F('그림 1 N_H', '조건 거짓', '조건 거짓' if d['N_H_condition'] is False else '조건 참', 'density_audit.json N_H_condition')
F('그림 2 선택 기준', '선택 기준 0.80', '선택 기준 %.2f' % float(re.search(r'level_coverage_target: ([\d.]+)', open('configs/t2_gate0T2.yaml').read()).group(1)), 'configs/t2_gate0T2.yaml')
F('그림 2 문턱', '문턱 0.02', '문턱 %.2f' % float(re.search(r'pass_excess_upper90: ([\d.]+)', open('configs/t2_gate0T2.yaml').read()).group(1)), 'configs/t2_gate0T2.yaml')
F('그림 3 정합 후보 수', 'L2, 19개', f"{g['level']}, {len(R('results/t2/matched_sets.json')['sets'][0]['cands'])}개", 'gate0T2.json level, matched_sets.json')
# ---- gpuq (read-only dump of ~/.gpuq/queue.db)
T2J = [i for i in jobs if jobs[i]['name'].startswith('t2') and i <= 459]
UGJ = [i for i in jobs if jobs[i]['name'].startswith('ugr')]
st = lambda ids: '완료 %d, 실패 %d, 취소 %d' % tuple(sum(jobs[i]['status'] == s for i in ids) for s in ('done', 'failed', 'cancelled'))
dur = lambda ids: sum(el(i) for i in ids if jobs[i]['started_at'])
C('T2 작업 수', '13개(완료 10, 실패 2, 취소 1)', f'{len(T2J)}개({st(T2J)})', 'gpuq queue.db jobs 446-459')
C('T2 경과 합', '883.3~s', f'{dur(T2J):.1f}~s', 'gpuq queue.db ended_at − started_at')
C('T2 GPU 작업 경과 합', '(GPU 686.4~s)', f"(GPU {sum(el(i) for i in T2J if jobs[i]['peak_mem'] > 0):.1f}~s)", 'gpuq queue.db peak_mem > 0')
C('UGR 작업 수', '16개(완료\n11, 실패 3, 취소 2)', f'{len(UGJ)}개(완료\n{st(UGJ)[3:]})', 'gpuq queue.db ugr_*')
C('UGR 경과 합', '1,068.2~s', f'{dur(UGJ):,.1f}~s', 'gpuq queue.db')
C('UGR GPU 작업', '(9.4~s)', f"({el(460):.1f}~s)", 'gpuq queue.db job 460 (peak_mem > 0만)')
for i, w in ((447, '59.3~s'), (452, '180.5~s'), (454, '195.4~s'), (456, '211.4~s'), (459, '14.7~s'), (460, '9.4~s')):
    C(f'job {i} 경과', w, f'{el(i):.1f}~s', 'gpuq queue.db')
C('job 473 경과', '2분 34초', '%d분 %d초' % divmod(round(el(473)), 60), 'gpuq queue.db')
C('job 474 경과', '474(판정 3~s)', f'474(판정 {el(474):.0f}~s)', 'gpuq queue.db')
C('job 475 경과', '475(진단 3~s)', f'475(진단 {el(475):.0f}~s)', 'gpuq queue.db')
C('job 459 장치', '459(CPU', '459(CPU' if jobs[459]['peak_mem'] == 0 else '459(GPU', 'gpuq queue.db peak_mem')
# ---- registered constants (configs read as text: no PyYAML needed) and other recorded facts
import math
yg = open('configs/t2_gate0T2.yaml').read()
yt = open('configs/t2.yaml').read()


def rx(pat, text):
    return re.search(pat, text).group(1)


lv = dict((m[0], m[1:]) for m in re.findall(r'name: (L\d), pool: (\w+),\s*letters_rel: ([\d.]+), syl_abs: (\d), dur_rel: ([\d.]+)', yg))
pct = lambda x: '%d' % round(float(x) * 100)
PM, PC = '$\\pm$', '\\%'
C('L0 허용 폭', '글자 수 $\\pm$10\\%,\n음절 수 $\\pm$1, TTS 발화 시간 $\\pm$10\\%',
  '글자 수 ' + PM + pct(lv['L0'][1]) + PC + ',\n음절 수 ' + PM + lv['L0'][2] + ', TTS 발화 시간 ' + PM + pct(lv['L0'][3]) + PC,
  'configs/t2_gate0T2.yaml matching.levels L0')
C('L2·L3 허용 폭', 'L2는 글자 수와 발화 시간을 $\\pm$15\\%로, L3은 $\\pm$20\\%와 음절 수 $\\pm$2로',
  'L2는 글자 수와 발화 시간을 ' + PM + pct(lv['L2'][1]) + PC + '로, L3은 ' + PM + pct(lv['L3'][1]) + PC + '와 음절 수 ' + PM + lv['L3'][2] + '로',
  'configs/t2_gate0T2.yaml matching.levels L2·L3')
C('같은 실행 최소 간격', '30~s 이상 떨어진', '%g~s 이상 떨어진' % float(rx(r'min_sep_s: ([\d.]+)', yg)), 'configs/t2_gate0T2.yaml matching.min_sep_s')
C('단계 선택 기준', '처음 0.80 이상', '처음 %.2f 이상' % float(rx(r'level_coverage_target: ([\d.]+)', yg)), 'configs/t2_gate0T2.yaml level_coverage_target')
C('(i) 문턱', '90\\% 하한이 0.90 이상', '90\\%% 하한이 %.2f 이상' % float(rx(r'pass_raw_mrp_lower90: ([\d.]+)', yg)), 'configs/t2_gate0T2.yaml gate.pass_raw_mrp_lower90')
C('(ii) 문턱', '90\\% 상한이 0.02 미만', '90\\%% 상한이 %.2f 미만' % float(rx(r'pass_excess_upper90: ([\d.]+)', yg)), 'configs/t2_gate0T2.yaml gate.pass_excess_upper90')
C('(iii) 문턱', '정합 비율 0.50 이상', '정합 비율 %.2f 이상' % float(rx(r'pass_coverage: ([\d.]+)', yg)), 'configs/t2_gate0T2.yaml gate.pass_coverage')
C('귀무·부트스트랩 횟수', '30~s, 1,000회)이고, 구간은 시험 실행 부트스트랩(1,000회)',
  '%g~s, %s회)이고, 구간은 시험 실행 부트스트랩(%s회)' % (float(rx(r'min_shift_s: ([\d.]+)', yg)), format(int(rx(r'n_shift: (\d+)', yg)), ','),
                                                format(int(rx(r'bootstrap: \{n: (\d+)', yg)), ',')), 'configs/t2_gate0T2.yaml gate.shift_null, gate.bootstrap')
C('개발 관문', '$\\ge +0.04$, 잡음 입력 초과분 90\\% 상한 $< 0.01$',
  '$\\ge +%.2f$, 잡음 입력 초과분 90\\%% 상한 $< %.2f$' % (float(rx(r'excess_min: ([\d.]+)', yt)), float(rx(r'noise_upper: ([\d.]+)', yt))), 'configs/t2.yaml dev_gate')
pool = rx(r'pool: \[([\d, ]+)\]', yt).split(',')
C('시험 칸', '풀 17명 $\\times$\n시험 실행 20개 = 340칸', '풀 %d명 $\\times$\n시험 실행 20개 = %d칸' % (len(pool), len(pool) * 20), 'configs/t2.yaml pool')
C('H_T2a SESOI', '점추정 $\\ge +0.02$', '점추정 $\\ge +%.2f$' % float(rx(r'sesoi: ([\d.]+)', yt)), 'configs/t2.yaml verdict.sesoi')
C('H_T2b 여유', '상한 $< 0.03$', '상한 $< %.2f$' % float(rx(r'ni_margin: ([\d.]+)', yt)), 'configs/t2.yaml verdict.ni_margin')
C('재추론 허용오차', '칸 MRP 차 $\\le 10^{-4}$, 점수 차 최대 $\\le$ 칸 점수 표준편차의 $10^{-3}$배',
  '칸 MRP 차 $\\le 10^{%d}$, 점수 차 최대 $\\le$ 칸 점수 표준편차의 $10^{%d}$배' % (round(math.log10(rc['tolerances']['mrp_tol'])), round(math.log10(rc['tolerances']['score_tol_rel']))),
  'reinfer_check.json tolerances')
C('재추론 장치', 'RTX 4070에서 다시 계산', 'RTX 4070에서 다시 계산' if '4070' in rc['device'] else rc['device'], 'reinfer_check.json device')
C('Q 여유', '$\\pm$0.25~s 여유로', '$\\pm$%.2f~s 여유로' % float(rx(r'eeg_margin_s: ([\d.]+)', yt)), 'configs/t2.yaml eeg_margin_s')
b1v = R('results/stageB_b1/verdict.json')
C('B1 MRP(풀 17명)', 'MRP) 0.877로', 'MRP) %.3f로' % b1v['mrp'], 'results/stageB_b1/verdict.json mrp (D-085)')
C('U2 계획 작업 수', '작업 72개', '작업 %d개' % (6 * 5 * 2 + 12), 'D-110 6절 4: 6명 × 5시드 × 2팔 = 60 + 가중치 순열 12')
C('U0 모의 횟수', '모의 100회', '모의 %d회' % R('results/ugr/u0_amend1_info0.json')['draws'], 'u0_amend1_info0.json draws')
# ---- section 3.2 last paragraph and appendix D (eeg-b1 9bd4ac2, added after the first request)
sr = R('results/t2/density_audit_same_run.json')
LV = {L: R(f'results/t2/levels/density_{L}.json') for L in ('L0', 'L1', 'L3')}
ci3 = lambda c: '[%.3f, %.3f]' % tuple(c)
C('3.2 같은 실행만 R_dens', '1.005(1.000--1.010)', '%.3f(%.3f--%.3f)' % (sr['R_dens'], *sr['R_dens_ci90']), 'density_audit_same_run.json')
C('3.2 단계별 R_dens', '1.002, 0.998, 0.998', ', '.join(f3(LV[L]['R_dens']) for L in ('L0', 'L1', 'L3')), 'levels/density_L{0,1,3}.json R_dens')
for name, dd in (('같은 실행만(L2)', sr), ('L0', LV['L0']), ('L1', LV['L1']), ('L2(본문)', d), ('L3', LV['L3'])):
    C(f'부록 D 표 {name}', next((l.strip() for l in lines if l.startswith(name + ' &')), '(행 없음)'),
      f"{name} & {dd['n_queries']:,} & {dd['MRP_full']:.4f} & {dd['MRP_matched']:.4f} & {f3(dd['R_dens'])} {ci3(dd['R_dens_ci90'])} \\\\",
      'density_audit_same_run.json, levels/density_L*.json, density_audit.json')
C('부록 D L0 실행당 질의', '실행당 질의가 2--50개', '실행당 질의가 %d--%d개' % (min(LV['L0']['queries_per_run']), max(LV['L0']['queries_per_run'])), 'levels/density_L0.json queries_per_run')
C('부록 D L0 무정합 불가 칸', '340칸 중 136칸', '340칸 중 %d칸' % LV['L0']['unmatched_missing_cells'], 'levels/density_L0.json unmatched_missing_cells')
C('부록 D N_H 나머지 거짓', '모두\n거짓', '모두\n거짓' if all(x['N_H_condition'] is False for x in (sr, LV['L1'], LV['L3'], d)) else '일부 참', 'N_H_condition (L0은 None)')
C('부록 D L2 재구성 바이트 일치', '바이트 단위로 같음', '바이트 단위로 같음' if R('results/t2/levels/levels_rebuild.json')['rebuild']['same_bytes'] else '다름', 'levels/levels_rebuild.json rebuild.same_bytes')
for f, w in (('results/t2/density_audit_same_run.json', 'dca933ae'), ('results/t2/levels/density_L0.json', 'a35a17ac'),
             ('results/t2/levels/density_L1.json', 'd814d77d'), ('results/t2/levels/density_L3.json', '9ab3ef5e'),
             ('results/t2/levels/levels_rebuild.json', 'fce88821')):
    C(f'부록 D sha256 {f}', w, sha8(f), f)
DJ = [i for i in range(476, 488) if i in jobs and jobs[i]['name'].startswith('t2')]
C('부록 B 부록 D 작업 경과 합', '경과 합 513.4~s(GPU 431.3~s)',
  '경과 합 %.1f~s(GPU %.1f~s)' % (sum(el(i) for i in DJ), sum(el(i) for i in DJ if jobs[i]['peak_mem'] > 0)), 'gpuq queue.db jobs 476-487 중 t2_* 11개(원값 합에서 반올림)')
C('부록 B 부록 D 작업 범위', 'job 476--487(실패 479', 'job 476--487(478 제외, 실패 %s' % ','.join(str(i) for i in DJ if jobs[i]['status'] == 'failed'),
  'gpuq queue.db: 478은 fiv_verify_cocktail(eeg-dir3 사실표 대조)')
# ---- appendix C hashes
for f, w in (('results/t2/gate0T2.json', '003cbcb2'), ('results/t2/matched_sets.json', '5f153a93'), ('results/t2/match_attrs.json', '9e7863e0'),
             ('results/t2/reinfer_check.json', 'f89f340f'), ('results/t2/scores/test_rec_summary.json', '9ece99ab'),
             ('results/t2/density_audit.json', '0195bc00'), ('results/ugr/seal.json', '7ec2a78f'), ('results/ugr/u0.json', 'e84d1c40'),
             ('results/ugr/u0_amend1_info0.json', 'd94d5817'), ('results/ugr/u0_amend1_plant.json', 'd8d859e9'),
             ('results/ugr/u1.json', '8c33f511'), ('results/ugr/u1_diag.json', 'faf676e9')):
    C(f'sha256 {f}', w, sha8(f), f)

figrows = []
for label, written, expected, source in FIGC:
    hits = [i + 1 for i, l in enumerate(FIG) if written in l]
    figrows.append(dict(label=label, tex=written, expected=expected, source=source, lines=hits, match=written == expected and bool(hits)))
rows, covered = [], set()
for label, written, expected, source in checks:
    hits = [i + 1 for i, l in enumerate(lines) if (i + 1) in inscope and written.split('\n')[0] in l]
    if '\n' in written:
        hits = [i + 1 for i in range(len(lines) - 1) if (i + 1) in inscope and written.split('\n')[0] in lines[i] and lines[i + 1].startswith(written.split('\n')[1])]
    rows.append(dict(label=label, tex=written.replace('\n', ' '), expected=expected.replace('\n', ' '), source=source,
                     lines=hits, match=written == expected and bool(hits)))
    for h in hits:
        covered.add((h, written.split('\n')[0]))

num = re.compile(r'(?<![A-Za-z{\\])[-+−]?\d[\d,]*(?:\.\d+)?')
uncovered = []
for i in sorted(inscope):
    l = lines[i - 1]
    if l.lstrip().startswith('%'):
        continue
    for mt in num.finditer(l):
        if not any(h == i and mt.group(0).lstrip('+-−') in w for h, w in covered):
            uncovered.append((i, inscope[i], mt.group(0)))
json.dump(dict(checks=rows, figure_checks=figrows, uncovered=uncovered), open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
for r in rows:
    print('O' if r['match'] else 'X', r['lines'], r['label'], '|', r['tex'], '|', r['expected'])
for r in figrows:
    print('O' if r['match'] else 'X', 'make_figures.py', r['lines'], r['label'], '|', r['tex'], '|', r['expected'])
print('uncovered number tokens', len(uncovered))
