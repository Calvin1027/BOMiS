# 1단계 합성 DGP와 채점기 명세서

## 0. 출처와 범위

- **원문(기준 문서)**
  - 설계: `/data1/calvinkim1027/BOMIS_new_수만이형/docs/superpowers/specs/2026-10-22-synthetic-dgp-design.md` (10/23 개정판)
  - 계획: `/data1/calvinkim1027/BOMIS_new_수만이형/docs/superpowers/plans/2026-10-22-synthetic-dgp.md` (인터페이스, 테스트 42개)
  - 배경: 수만이형 초안 `/data1/calvinkim1027/BOMIS_new_수만이형/refs/draft_certifying_structural_zeros.pdf` (식 (1)–(3), 정리 1, 5.3절, G.3절)
- **구현 범위**: 합성 모집단(18개 환경) 생성, 표본 추출, 정답 분류, 채점 (a)–(d), 1단계 점검 실행기
- **범위 밖**: 인증·$\hat S$ 같은 방법 구현(2단계), 디코더, baseline, real 데이터, 실험 본실행
- **기호표**

  | 기호 | 뜻 |
  |---|---|
  | $U$ | 셀 65,536개 |
  | $p_0$, $p_1$ | majority, minority 법칙. (65536,) |
  | $\pi_y$ | 조각 확률. (480,) |
  | $S_0$, $S_1$ | 지지집합 |
  | $R$, $Z$ | 영역 (설계 3.2절) |
  | $h$ | 섞는 법칙 |
  | $\omega$ | 섞기 비율 0.2 |
  | $\gamma$ | 겹침 수준 |
  | $e$, $e^T$ | 조건부 손실 비율 |
  | $\eta$ | Dirichlet 농도 |
  | $n_0$, $n_1$ | 표본 크기 |
  | $\hat x$ | 인증 기준값 |
  | $\lambda^*$ | 인증 경계 |
  | $s_\beta$ | 인증 신호 |

## 1. 구성 요소 (C)

| ID | 이름 | 원문 위치 | 입력 | 출력 | 정의 |
|---|---|---|---|---|---|
| C1 | 공간과 조각 | 설계 1절, 계획 T1 | 없음 | `CELLS`, `cell_index`, `pattern_id`, `CELL_PATTERNS`, `pattern_probs`, `pattern_counts`, `cells_containing` | 셀·조각 번호와 조각 확률·개수 |
| C2 | 트리 법칙 | 설계 2절, T2 | rng, $\eta$ | `TreeTables`, `tree_law` | 부모 사슬 + 잎 트리, Dirichlet 표 |
| C3 | 부모 쌍 위의 0 | 설계 2절, T2 | 트리 법칙, rng | 공유 0 6개, $Z$ 쌍 2개, $p_0$, $S_0$ | 0을 넣고 다시 정규화 |
| C4 | 금지 쌍 선택 | 설계 3.1절, T3 | $p_0$, 수준 | 수준별 Rule 4개 | 잎 간선마다 $(P=a', L=a)$ |
| C5 | 3변수 규칙 선택 | 설계 3.1절, T3 | $p_0$, 세 수준 금지 쌍 | Rule 2개 | $(P=c', L=c, P^+=b)$, 부모 0과 2 |
| C6 | 영역 | 설계 3.2절, T3 | 0 쌍, 규칙, $\gamma$ | $R$, $Z$, $S_1$ | |
| C7 | 질량 이동 | 설계 3.3절 1, T3 | $p_0$, $R$, 규칙 | $p_1^{\text{base}}$, 버린 질량 | 부모 좌표만 바꿔 $R$로 옮김 |
| C8 | 섞기 | 설계 3.3절 2, T3 | $p_1^{\text{base}}$, $h$, $R$, $Z$, $\gamma$ | $p_1$ | $(1-\omega)p_1^{\text{base}} + \omega[(1-\gamma)h(\cdot\mid R) + \gamma h(\cdot\mid Z)]$ |
| C9 | minority 점검 | 설계 8절 1–5, T3 | $p_0$, $p_1$, $S_1$, $Z$, 규칙, $\gamma$ | 점검 dict | |
| C10 | 모집단·환경 조립과 수락 | 설계 4절, T4 | pop_seed, $\eta$ | `Population` 또는 거절 사유 | 18개 환경, 짝지은 설계 |
| C11 | $\eta$ 예비 실행 | 설계 2절, T4 | 후보 $\eta$, seed 범위 | 수락률, 고른 $\eta$ | |
| C12 | 저장·불러오기 | 설계 9절, T4 | `Population` | npz | |
| C13 | 표본 추출 | 설계 5절, T5 | `Population`, 환경, seed, $n$ | 셀 번호 배열 | 역누적분포 |
| C14 | 정답 분류 | 설계 6절, T5 | `Population`, 환경, $D_0$, $D_1$ | `Truth` | |
| C15 | 채점 (a) 집합 | 설계 7절 a, T6 | `Truth`, $\hat S$ | dict | |
| C16 | 채점 (b) 법칙 | 설계 7절 b, T6 | `Truth`, $\hat q$ | dict | |
| C17 | 채점 (c) 생성 행 | 설계 7절 c, T6 | `Truth`, 셀 번호 | dict | |
| C18 | 채점 (d) 인증과 정확 계산 | 설계 7절 d, T7 | `Truth`, $\hat C$, $\hat x$, $n_0$, $n_1$, $r$ | dict, 곡선 | |
| C19 | 점검 표·판정·보고서 | 설계 8절, T8 | 수락된 모집단 목록 | csv 4개, verdict, md | |
| C20 | 진입 스크립트·sbatch | 계획 Global Constraints, T8 | 인자 | 파일 | `-I` 모드 경로 처리 |

**호출 순서**

- `run_pilot`: C11 → (seed마다 C10)
- `run_check`: 다음 순서로 진행한다.
  1. C10. 내부 순서는 C2 → C3 → C2($h$) → C4(3수준) → C5 → 환경마다 C6 → C7(재사용) → C8 → C9
  2. C12
  3. (seed, 환경, $n_1$, 표본 seed)마다 C13 → C14
  4. C15–C18(정답 점검, `oracle_table`). 입력과 기대값은 계획 T8의 `oracle_table` 목록을 따른다. C17은 `score_rows`, C18은 `score_cert`·`exact_cert_prob`·`exact_fwer_bound`·`clopper_pearson_upper`·`power_curve`를 모두 부른다
  5. C19

## 2. 세부 (D)

| ID | 소속 | 범주 | 내용 | 근거 |
|---|---|---|---|---|
| D1.1 | C1 | 인덱스 | 셀 번호 $=\sum_j c_j 4^{7-j}$. 열 0이 최상위 자리 | 계획 GC |
| D1.2 | C1 | 인덱스 | 단일 조각 `4j+v`. 쌍 조각 `32+16*rank(j,l)+4v+w`, $(j,l)$ 사전순, $j<l$ | 계획 GC |
| D1.3 | C1 | 형태 | `CELL_PATTERNS` (65536,36) = 단일 8 + 쌍 28 | T1 |
| D2.1 | C2 | 초기화 | 뿌리(열 0)와 7개 간선 표의 각 행을 `rng.dirichlet(np.full(4, eta))`로 뽑음. 간선 (0,1),(1,2),(2,3),(0,4),(1,5),(2,6),(3,7), 방향 부모→자식 | 설계 2절 |
| D2.2 | C2 | 난수 | $p_0$ 표 `[pop_seed,0]`, $h$ 표 `[pop_seed,1]` | 계획 GC |
| D3.1 | C3 | 절차 | 공유 0은 `PARENT_PAIRS` 순서로 하나씩. 값 쌍은 0으로 만들기 전 트리 법칙의 쌍 확률이 0.05 이하인 것 중 균등 | 설계 2절 |
| D3.2 | C3 | 절차 | $Z$ 쌍 2개: 서로 다른 부모 쌍 2개를 균등으로 고르고, 그 쌍의 공유 0과 다른 값 쌍 중 질량 0.05 이하인 것을 균등으로 고름 | 설계 2절 |
| D3.3 | C3 | 난수 | 0 쌍 선택 난수 `[pop_seed,2]`. 공유 0을 먼저, $Z$ 쌍을 나중에 같은 흐름에서 뽑음 | 계획 GC |
| D3.4 | C3 | 정의 | $p_0$ = 트리 법칙 × 1[공유 0도 $Z$ 쌍도 포함하지 않음], 다시 정규화. $S_0=\{p_0>0\}$ | 설계 2절 |
| D4.1 | C4 | 정의 | 잎 간선마다 하나, 부모 0→3 순서. Rule(kind="pair", cols=(i, 4+i), vals=(a', a), move_col=i) | 설계 3.1절 |
| D4.2 | C4 | 확률 정의 | $e=P_0(L=a\mid P=a')$와 $\pi_0$는 0을 넣은 뒤의 $p_0$로 계산. 조건 $e\le 0.2$, $\pi_0\in$[하한, 상한] (양 끝 포함) | 설계 3.1절, 4절 |
| D4.3 | C4 | 순위 | $\lvert\pi_0-\text{target}\rvert$ 최소. 동점이면 $(a',a)$ 사전순. 후보 없으면 None | 계획 GC |
| D4.4 | C4 | 값 | 수준: frequent (0.12, 0.09, 0.16), mid (0.03, 0.025, 0.04), rare (0.005, 0.003, 0.008) | 설계 4절 |
| D5.1 | C5 | 정의 | 부모 0, 2에 하나씩. cols=(p, p+4, p+1), vals=(c', c, b), move_col=p | 설계 3.1절 |
| D5.2 | C5 | 조건 | $e^T=P_0(L=c\mid P=c')\le 0.1$, $c'\notin$ {세 수준 금지 쌍의 $a'$}, $\pi_0(T)\in[0.01,0.04]$, 하위 쌍 $(P=c',P^+=b)$의 $\pi_0>0$ | 설계 3.1절 |
| D5.3 | C5 | 순위 | $\lvert\pi_0-0.02\rvert$ 최소. 동점이면 $(c',c,b)$ 사전순. 후보 없으면 None | 계획 GC |
| D6.1 | C6 | 정의 | $R=S_0\cap$(규칙을 하나도 포함하지 않음). $Z$=($Z$ 쌍을 하나 이상 포함)∩(공유 0 없음)∩(규칙 없음). $S_1=R$ ($\gamma=0$), $R\cup Z$ (그 외). $Z$는 $\gamma=0$이어도 계산해서 돌려줌 | 설계 3.2절 |
| D7.1 | C7 | 절차 | 규칙을 포함하는 $S_0$ 셀만 이동. 포함한 규칙들의 `move_col`을 각각 자기 값이 아닌 3개 값으로 바꾼 모든 조합 중 $R$ 안의 셀로 똑같이 나눔. 없으면 버림. 끝에 재정규화. 반환하는 버린 질량은 재정규화 전 값 | 설계 3.3절 |
| D7.2 | C7 | 재사용 | $p_1^{\text{base}}$는 (STZ 수준, 가정 수준)마다 한 번 계산하고 세 $\gamma$에 재사용 | 계획 T4 |
| D8.1 | C8 | 수식 | $h(\cdot\mid A)=h\,1_A/h(A)$. $\omega=0.2$, $\gamma\in\{0,0.5,1\}$. $\gamma=0$이면 $Z$ 항 없음. $p_1(Z)=\omega\gamma$ | 설계 3.3절 |
| D9.1 | C9 | 정의 | `sum_ok`: \|합−1\|<1e-12. `support_ok`: $p_1>0\iff S_1$ | 설계 8절 1 |
| D9.2 | C9 | 분기 | `local_support_ok`: triple이 없으면 `pairwise_feasible_cells(pi1)==S1`, 있으면 `F & ~S1`이 `F & (triple 포함 셀)`과 정확히 같고 비어 있지 않음(`F = pairwise_feasible_cells(pi1)`. triple과 금지 쌍을 함께 포함하는 셀은 F에 없음). `triples_minimal_ok`: triple의 하위 쌍 3개 모두 $\pi_1>0$ | 설계 8절 2 |
| D9.3 | C9 | 정의 | `floor`: $\pi_0>0$, $\pi_1>0$인 조각에서 $\min\pi_1/\pi_0\ge 0.5$. `infeasible_ok`: $\pi_0>0$, $\pi_1=0$인 조각 = pair 규칙 pid 4개. `z_mass_ok`: \|$p_1(Z)-\omega\gamma$\|<1e-12 | 설계 8절 3–5 |
| D10.1 | C10 | 순서 | `ENV_KEYS`: freq(frequent, mid, rare) × ov(fav, avg, unf) × asm(holds, viol), 바깥부터. 이름 `f"{freq}__{ov}__{asm}"` | 계획 GC |
| D10.2 | C10 | 조건 분기 | 18개 환경이 모두 통과해야 수락. 사유 `ok`/`no_zeros`/`no_pair:<freq>`(frequent, mid, rare 순서로 처음 실패한 것)/`no_triple`/`check_fail:<env>:<key>` | 계획 T4 |
| D10.3 | C10 | 종료 조건 | `generate_accepted`: start_seed부터 1씩 올림. 거절 목록 반환. max_tries(2000) 안에 못 채우면 `StopGeneration` | 설계 4절 |
| D11.1 | C11 | 값 | $\eta$ 후보 (1.0, 0.5, 0.3), seed 5000–5099, 수락률 ≥ 0.05인 것 중 가장 큰 $\eta$. 없으면 None | 계획 GC |
| D12.1 | C12 | 형식 | npz 하나. 환경 배열 `p1__<env>`, `S1__<env>`, `R__<env>`, `Z__<env>`, `pi1__<env>`. 나머지는 JSON `meta`. 불러온 Rule과 0 쌍이 원본과 같음(튜플 비교) | 계획 T4 |
| D13.1 | C13 | 수치 | `draw_cells`: cumsum을 마지막 값으로 나눔, `searchsorted(side="right")`, 확률 0인 셀(범위 밖 포함)은 앞쪽의 가장 가까운 양수 확률 셀로 | 계획 T5 |
| D13.2 | C13 | 난수 | $D_0$ `[pop_seed, sample_seed, 10]`, 길이 $n_0$. $D_1$ 균등 난수 `[pop_seed, sample_seed, 11]` 800개 중 앞 $n_1$개, 모든 환경 공통 | 설계 5절 |
| D14.1 | C14 | 정의 | `SZ`=$S_1\setminus O$, `SZ1`=SZ∩$O_0$, `SZ2`=SZ∖$O_0$, `SZ_nomaj`=SZ∩{$p_0=0$}, `STZ`=¬$S_1$, `STZ_maj`=STZ∩{$p_0>0$}, `STZ_common`=STZ∩{$p_0=0$}, `STZ_triple`=STZ_maj∩(triple 포함)∩(pair 규칙 미포함), `M0`=$p_1$(SZ) | 설계 6절 |
| D14.2 | C14 | 정의 | `N0`, `N1` = `pattern_counts`($D_0$), ($D_1$) | 설계 6절 |
| D15.1 | C15 | 지표 정의 | 질량 회수율 = $p_1(\hat S\cap T)/p_1(T)$ (T = SZ, SZ1, SZ2, SZ_nomaj). 개수 회수율·억제율은 셀 수. majority 질량 억제율 = $1-p_0(\hat S\cap\text{STZ\_maj})/p_0(\text{STZ\_maj})$. `rule_left_j`는 환경 rules 순서. 분모 0이면 nan | 설계 7절 a |
| D16.1 | C16 | 지표 정의 | $\Gamma=\hat q(\text{SZ})$, $\Lambda$=$\hat q$(STZ 종류별). TV $=\tfrac12\sum\lvert\hat q(\cdot\mid\text{SZ})-p_1(\cdot\mid\text{SZ})\rvert$, $\hat q(\text{SZ})=0$이면 nan | 설계 7절 b |
| D17.1 | C17 | 지표 정의 | 누출 = 해당 STZ에 떨어진 행 수 / 전체 행 수. 회수율(개수) = 맞힌 서로 다른 SZ 셀 수 / \|SZ\|. 회수율(질량) = 맞힌 서로 다른 SZ 셀의 $p_1$ 합 / $M_0$. `stz_maj_hit` = 맞힌 서로 다른 STZ_maj 셀 수 / \|STZ_maj\| | 설계 7절 c |
| D18.1 | C18 | 수식 | $\lambda^*=\tau/\{\kappa\log(1+1/\kappa)\}$, $\kappa=n_0/(n_1r)$, $\tau=\log(480/0.05)$. $s=n_1r\pi_0/\lambda^*$ | 설계 4절 |
| D18.2 | C18 | 수식 | 정확 인증 확률 $=(1-\pi_1)^{n_1}\cdot$`binom.sf(x_hat-1, n0, pi0)` (즉 $P\{\text{Bin}\ge\hat x\}$) | 설계 7절 d |
| D18.3 | C18 | 지표 정의 | FWER 정확 상한 = $\pi_0>0$, $\pi_1>0$인 조각의 위 값 합. `any_false`/`n_false` = $\hat C$ 중 $\pi_1>0$인 조각. 대상 금지 쌍은 pair 규칙 4개뿐 | 설계 7절 d |
| D18.4 | C18 | 통계량 | Clopper–Pearson 단측 상한 = `beta.ppf(level, k+1, n-k)`, k=n이면 1 | 설계 7절 d |
| D19.1 | C19 | 판정 | `checks_pass`, `oracle_pass`, `cross_frequent`/`cross_mid`(해당 수준 모든 쌍 `s_50<1<=s_800`), `overlap_ok`($n_1$마다 overlap별 `nomaj_share_mass` 평균이 fav=0, avg<unf), `triple_ok`(`n_stz_triple>0` ⟺ viol), `stage1_pass`(모두) | 설계 8절 |
| D20.1 | C20 | 경로 | 진입 스크립트 첫 줄에서 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`. pytest는 `pythonpath = .`. `--quick`은 9000번대 seed로 모집단 2개, 표본 seed 2개 | 계획 GC, T8 |

## 3. 모호점 (Q)

| ID | 관련 | 무엇이 불명확한가 | 결정 | 상태 |
|---|---|---|---|---|
| Q1 | D3.1, D3.2 | "균등"의 구현 | 조건을 만족하는 후보를 사전순 목록으로 만들고 `rng.choice`로 하나 | [임시 결정] |
| Q2 | D16.1 | TV 정의 | $\tfrac12 L_1$ | [임시 결정] |
| Q3 | D11.1 | 수락률 기준값 | 0.05 | [승인] 10/23 |
| Q4 | D17.1 | 생성 행 회수율을 행 기준이 아닌 서로 다른 셀 기준으로 셈 | 서로 다른 셀 | [임시 결정] |
| Q5 | D7.1 | 이동 대상이 $R$ 안이어야 하는지($Z$로는 안 보냄) | $R$ 안만 | [승인] 설계 3.3절 |

## 4. 대응표

구현자가 채운다. 모든 C와 D에 `파일:함수:줄`을 적는다.

| ID | 코드 위치 | 비고 |
|---|---|---|
| C1 | | |
| … | | |

## 5. 검증 사례 (V)

| ID | 종류 | 입력 | 기대 결과 | 허용 오차 | 잡는 항목 | 계획 테스트 |
|---|---|---|---|---|---|---|
| V1 | 손계산 | 셀 [0..0], [3..3], [0,…,0,1], [1,0,…,0] | 0, 65535, 1, 16384 | 정확 | D1.1 | T1 `test_cell_index_order` |
| V2 | 손계산 | `pattern_id` | (0,),(0,)→0. (7,),(3,)→31. (0,1),(0,0)→32. (6,7),(3,3)→479 | 정확 | D1.2 | T1 |
| V3 | 손계산 | 균등 $p$ | 단일 조각 0.25, 쌍 1/16 | 1e-12 | C1 | T1 |
| V4 | 이론 성질 | 0을 넣은 $p_0$ | $P(L=a\mid P=a',X=x)$ = 트리 표 값 | 1e-10 | D3.4, 설계 3.4절 전제 | T2 |
| V5 | 이론 성질 | 금지 쌍만 | 부모 하나 고정 조각 비율 $1-e$, 두 부모 $(1-e_0)(1-e_1)$ | 1e-10 | D7.1 | T3 |
| V6 | 이론 성질 | 18개 환경 조합 | 버린 질량이 0이면 $\min\pi_1/\pi_0\ge 0.512$ | 1e-12 | C7, C8 | T3 |
| V7 | 손계산 | $\gamma\in\{0,0.5,1\}$ | $p_1(Z)=0, 0.1, 0.2$ | 1e-12 | D8.1 | T3 |
| V8 | 이론 성질 | 위반 환경 | 쌍 단위로는 가능하지만 $S_1$ 밖인 셀 = 쌍 단위로 가능한 triple 포함 셀, 비어 있지 않음 | 정확 | D9.2, D5 | T3 |
| V9 | 극한 사례 | 마지막 셀 확률 0, $u=1.0$ | 확률 0인 셀이 안 나옴 | 정확 | D13.1 | T5 |
| V10 | 성질 | 같은 seed | $D_1(n_1=100)[:50]=D_1(50)$, 환경 간 같은 난수 | 정확 | D13.2 | T5 |
| V11 | 성질 | 정답 분류 | 분할 관계(설계 6절), fav에서 SZ_nomaj 없음, triple STZ는 viol에서만 | 정확 | D14.1 | T5 |
| V12 | 손계산 | $\hat S=S_1, O, U$ | 회수율·억제율 1/0 (설계 8절) | 정확 | C15 | T6 |
| V13 | 손계산 | 행 = STZ_maj 3개 + SZ 2개 × 2 | `leak_maj`=3/7 | 1e-12 | D17.1 | T6 |
| V14 | 극한 사례 | 분모 0 | nan | 정확 | D15.1, D16.1 | T6 |
| V15 | 원문 수치 | $n_0=8000$, $r=0.5$ | $\pi_0^*$ 0.367/0.184/0.092/0.046/0.023 | 1.5e-3 | D18.1 | T7 |
| V16 | 손계산 | $\pi_0=0.5$, $n_0=10$, $\hat x=8$ | 56/1024. $\pi_1=0.1$, $n_1=5$이면 $0.9^5\cdot56/1024$. $\hat x=11$이면 0 | 1e-15 | D18.2 | T7 |
| V17 | 손계산 | $k=0$, $n=3000$ | $1-0.05^{1/3000}$ | 1e-12 | D18.4 | T7 |
| V18 | 성질 | `-I` 모드 | `--help` 종료 코드 0 | 정확 | D20.1 | T8 |
| V19 | 끝에서 끝까지 | `--quick` | csv 4개와 verdict, seed마다 18행, min_ratio ≥ 0.5, `oracle.csv` 2×18×9+3=327행 모두 통과 | 정확 | C19, C20 | T8 |

## 6. 결과 민감 항목

검토에서 먼저 볼 것이다.

- **D7.1**: 바꾸는 좌표가 부모 좌표뿐이어야 floor 보장이 맞고, 이동 대상이 $R$ 안이어야 $p_1(Z) = \omega\gamma$가 정확하다.
- **D4.2, D5.2**: $e$와 $\pi_0$를 0을 넣은 뒤의 $p_0$로 계산해야 한다. 트리 법칙으로 계산하면 범위 판정이 어긋난다.
- **D13.2**: 환경 간에 같은 난수를 쓰고 $n_1$ 사이에 포함 관계가 있어야 한다. 그래야 짝지은 비교가 성립한다.
- **D14.1**: `STZ_triple`의 정의. pair 규칙을 포함하는 셀은 제외한다.
- **D18.2**: `binom.sf`의 `x_hat-1` 인자. 하나만 어긋나도 인증 확률이 틀린다.
