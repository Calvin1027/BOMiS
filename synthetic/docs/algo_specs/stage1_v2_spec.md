# 합성 실험 v2 구성 요소 명세서 (DGP, 채점기, 방법 연결, 쉬운 환경 실행)

## 0. 출처와 범위

- **원문(기준 문서)**
  - 설계: `/data1/calvinkim1027/BOMIS_new_수만이형/docs/superpowers/specs/2026-10-26-synthetic-study-ademp-design.md` (10/27판, 9환경)
  - 계획: `/data1/calvinkim1027/BOMIS_new_수만이형/docs/superpowers/plans/2026-10-27-synthetic-study-v2.md`
  - 방법: 박사형 코드 `refs/suman_repo_clone/bomis` 커밋 `fa088a9` (`bomis/model.py`, `bomis/metrics.py`, `bomis/populations.py`)
  - baseline: `/data1/calvinkim1027/과거 BOMIS/TDG_baseline_categorical` (`categorical_baselines.cli`, `configs/models.reference.json`)
  - 참조 구현(진단용, 기대값 대조에만 씀): `/mnt/data1/calvinkim1027/scratch_bomis/accept_v3.py`, `mix_ab3.py`
- **구현 범위**
  - 1단계: 9환경 합성 모집단, 표본, 정답, 채점기 (a)–(e), 박사형 코드 연결, 1단계 점검
  - 랩미팅 미리보기(A단계): 쉬운 환경 `frequent__fav`에서 A/B/C support, BOMiS 전체, 빠른 baseline 4개(ROS, SMOTE-N, SMOTE-ENC, CTGAN) 실행과 요약
- **범위 밖**
  - local support 위반 환경(3변수 규칙), $r$ 민감도, Chow-Liu, ARF, Oracle 실험(설계 10절)
  - B단계(랩미팅 뒤): TabSyn, TabDiff(같은 래퍼로 실행만 추가), TabDDPM·ForestDiffusion(범주형 어댑터 이식 필요), SMOTE-NC(Q7), 나머지 격자
  - 박사형 코드와 baseline 코드의 수정
- **기호표**

  | 기호 | 뜻 |
  |---|---|
  | $U$ | 셀 65,536개 ($d=8$, $K=4$) |
  | $p_0$, $p_1$ | majority, 환경별 minority 법칙. (65536,) |
  | $\pi_y$ | 1·2차 조각 확률. (480,) |
  | $S_0$, $S_1$ | 지지집합 |
  | $R$, $Z$ | 영역 (설계 4.4절) |
  | $h$ | 섞는 법칙 |
  | $\omega=0.2$, $\gamma\in\{0,0.5,1\}$ | 섞기 비율, 겹침 수준 |
  | $t_{lv}$ | 금지 쌍 목표 $\pi_0$: 0.06 / 0.03 / 0.005 |
  | $a'_i$, $e$ | 심는 부모 값, 손실 비율 $t_{lv}/P_0(a'_i)$ |
  | $O$, $O_0$ | $D_1$, $D_0$의 서로 다른 셀 |
  | SZ, STZ | $S_1\setminus O$, $U\setminus S_1$ |
  | $M_0$ | $p_1(\text{SZ})$ |
  | $x^*$ | `support.x_star` |
  | $Q$, $Q^{\text{u}}$ | BOMiS 디코더의 minority 법칙(마스크 있음, 없음) |
  | $L'$ | 생성 행 중 $O$ 밖 서로 다른 셀 수 |
  | $R_1, R_2, R_4$ | 회수율 분해 (설계 7.2절 (e)) |

## 1. 구성 요소 (C)

C 번호는 v1 코드 주석과 맞춘다. v1의 C5(3변수 규칙)와 C11($\eta$ 예비 실행)은 없앴다.

| ID | 이름 | 원문 위치 | 입력 | 출력 | 정의 |
|---|---|---|---|---|---|
| C1 | 공간과 조각 | 설계 4.1 | 없음 | `CELLS`, `cell_index`, `pattern_id`, `CELL_PATTERNS`, `pattern_probs`, `pattern_counts`, `cells_containing` | 기존 코드 그대로 |
| C2 | 트리 법칙 | 설계 4.2 | rng, $\eta$ | `TreeTables`, `tree_law` | 기존 코드 그대로 |
| C3 | 부모 쌍 위의 0 | 설계 4.2 | 트리 법칙, rng | 공유 0 6개, $Z$ 쌍 2개 | 기존 코드 그대로 |
| C4 | 금지 쌍 심기 | 설계 4.3 | 심기 전 표, 0 목록(공유 + $Z$ 쌍), rng | 심은 표, 수준별 Rule 4개 또는 거절 사유 | 잎 표 행 $a'_i$에 목표 확률을 넣음 |
| C6 | 영역 | 설계 4.4 | 0 쌍, 규칙, $\gamma$ | $R$, $Z$, $S_1$ | 기존 `regions` |
| C7 | 질량 이동 | 설계 4.4 | $p_0$, $R$, 규칙 | $p_1^{\text{base}}$, 버린 질량 | 기존 `move_mass` |
| C8 | 섞기 | 설계 4.4 | $p_1^{\text{base}}$, $h$, $R$, $Z$, $\gamma$ | $p_1$ | 기존 `mix` |
| C9 | minority 점검 | 설계 8절 1–6 | $p_0$, $p_1$, $S_1$, $Z$, 규칙, $\gamma$ | 점검 dict | 3변수 분기 삭제 |
| C10 | 모집단 조립과 수락 | 설계 4.7 | pop_seed | `Population` 또는 거절 사유 | 9환경, 짝지은 설계 |
| C12 | 저장·불러오기 | 설계 9 | `Population` | npz | |
| C13 | 표본 추출 | 설계 4.8 | `Population`, 환경, seed | 셀 번호 배열, 행 | 역누적분포 |
| C14 | 정답 | 설계 5 | `Population`, 환경, $D_0$, $D_1$ | `Truth` | STZ 하나 |
| C15 | 채점 (a) 집합 | 설계 7.2 (a) | `Truth`, $\hat S$ | dict | |
| C16 | 채점 (b) 법칙 | 설계 7.2 (b) | `Truth`, $\hat q$ | dict | |
| C17 | 채점 (c) 생성 행 | 설계 7.2 (c) | `Truth`, 행 셀 번호 | dict | |
| C18 | 채점 (d) 인증 | 설계 7.2 (d), 4.6 | `Truth`, $\hat C$, $x^*$, 규칙 | dict, 곡선 | |
| C19 | 1단계 점검·판정·보고 | 설계 8 | 모집단 목록 | csv, verdict, md | |
| C20 | 진입 스크립트·sbatch | 계획 Global Constraints | 인자 | 파일 | |
| C21 | 박사형 코드 고정 사본 | 설계 6.1 | 원본 clone | `vendor/bomis_fa088a9`, `SHA256SUMS` | |
| C22 | 모집단 연결부 | 설계 6.2 | `Population`, 환경, 참조 표본 | `PopAdapter`, `split` dict | `evaluate`가 쓰는 속성 |
| C23 | A/B/C support와 $\hat C$ | 설계 6.3, 6.4 | $D_0$, $D_1$ | $\hat S$, `Support`, $\hat C$ | `Support` 인자만 바꿈 |
| C24 | BOMiS 실행 | 설계 6.5 | $D_0$, $D_1$, seed | 학습된 모델, 생성 행 | `BOMiS(seed).fit().sample()` |
| C25 | BOMiS 단계별 진단 | 설계 7.2 (e) | 모델, `Truth`, 생성 행 | dict | 회수율 분해 등 |
| C26 | ROS | 설계 10 baseline | $D_1$, $n$, rng | 행 | minority 행 복원 추출 |
| C27 | 외부 baseline 래퍼 | 설계 10 baseline | 이름, $D_0$, $D_1$, $n$, seed | 행 또는 실패 | `categorical_baselines.cli`를 하위 프로세스로 |
| C28 | 쉬운 환경 실행과 요약 | 설계 11 진행 순서 1 | 모집단 3개, $n_1$, seed | 칸별 csv, 요약 md | |

**호출 순서**

- `scripts/build_pops.py`: (seed마다) C10[C2 → C3 → C4 → C2($h$) → 수준마다 C6($\gamma=0$) → C7 → $\gamma$마다 C6 → C8 → C9] → C12
- `scripts/run_check.py`: C21 해시 확인 → C12 불러오기 → (모집단, 환경, $n_1$, 표본 seed)마다 C13 → C14 → C23 → C15·C18 → 정답 점검(C15–C18 손계산 입력) → 연결 점검(C22 `evaluate` 대조, C24 `steps=250` 한 번 → C17·C25) → C19
- `scripts/run_easy.py`: 칸마다 C13 → C14 → C22 → C23 → C15·C18(A/B/C) → C24 → `evaluate`·C17·C25 → C26·C27(방법마다) → `evaluate`·C17 → 칸 csv. 끝에 `scripts/summarize_easy.py`(C28)

## 2. 세부 (D)

| ID | 소속 | 범주 | 내용 | 근거 |
|---|---|---|---|---|
| D1.1 | C1 | 인덱스 | 셀 번호 $=\sum_j c_j 4^{7-j}$, 열 0이 최상위 | 설계 4.1 |
| D1.2 | C1 | 인덱스 | 단일 조각 `4j+v`, 쌍 조각 `32+16*rank(j,l)+4v+w`, $(j,l)$ 사전순 | 기존 코드 |
| D2.1 | C2 | 초기화 | 뿌리와 7개 간선 표의 각 행을 Dirichlet($\eta=1$) | 설계 4.2 |
| D2.2 | C2 | 난수 | $p_0$ 표 `default_rng([pop_seed, 0])`, $h$ 표 `default_rng([pop_seed, 1])` | 설계 4.3 |
| D3.1 | C3 | 절차 | 0 선택은 **심기 전** 트리 법칙으로, `choose_parent_zeros(tree_law(t), default_rng([pop_seed, 2]))`. None이면 `no_zeros` | 설계 4.3 순서 |
| D4.1 | C4 | 값 | `T_LEVELS = {"frequent": 0.06, "mid": 0.03, "rare": 0.005}`, `P_MIN = 0.3` | 설계 4.3 |
| D4.2 | C4 | 확률 정의 | 부모 확률은 `pattern_probs(apply_zeros(tree_law(t), shared + major))`의 단일 조각. $a'_i$ = `np.argmax`(동점이면 작은 값) | 설계 4.3 |
| D4.3 | C4 | 난수 순서 | `rng = default_rng([pop_seed, 3])`. $i=0,1,2,3$ 순서로 `perm_i = rng.permutation(4)[:3]`를 **모두 먼저** 뽑는다. `perm_i`는 (frequent, mid, rare) 순서의 잎 값 | 설계 4.3, 참조 구현 |
| D4.4 | C4 | 분기 | 네 순열을 뽑은 뒤 $i$ 오름차순으로 $P_0(a'_i) < 0.3$인 첫 $i$가 있으면 `"no_plant:<i>"` 반환 | 설계 4.3 |
| D4.5 | C4 | 수식 | 잎 표 `cond[(i, 4+i)]`의 행 $a'_i$를 새 행으로 바꿈: `perm_i`의 값에 $t_{lv}/P_0(a'_i)$, 나머지 한 값에 $1-\sum_{lv} t_{lv}/P_0(a'_i)$. 다른 행은 그대로. 입력 표는 바꾸지 않고 복사본을 돌려줌 | 설계 4.3 |
| D4.6 | C4 | 정의 | `Rule(kind="pair", cols=(i, 4+i), vals=(a'_i, a_lv), move_col=i, e=t_lv/P_0(a'_i), pi0=t_lv)`. 수준별 목록은 $i$ 오름차순 | 기존 `Rule` |
| D4.7 | C4 | 점검 | $p_0$ = `apply_zeros(tree_law(심은 표), shared + major)`. 각 Rule의 $\pi_0$가 $t_{lv}$와 $10^{-12}$ 이내, $e \le 0.2$ | 설계 8절 6 |
| D6.1 | C6 | 정의 | 기존 `regions` 그대로. rules = 그 수준의 금지 쌍 4개 | 설계 4.4 |
| D7.1 | C7 | 절차 | 기존 `move_mass` 그대로. 수준마다 한 번 계산해 세 $\gamma$에 재사용 | 설계 4.4 |
| D8.1 | C8 | 수식 | 기존 `mix` 그대로. $\omega=0.2$ | 설계 4.4 |
| D9.1 | C9 | 정의 | `sum_ok`, `support_ok`, `infeasible_ok`, `z_mass_ok`, `floor_ok`(min ratio ≥ 0.5)는 기존과 같다 | 설계 8절 1, 3–5 |
| D9.2 | C9 | 정의 | `local_support_ok` = `np.array_equal(pairwise_feasible_cells(pi1), S1)`. `triples_minimal_ok` 키와 triple 분기는 삭제 | 설계 8절 2 |
| D10.1 | C10 | 순서 | `ENV_KEYS = [f"{lv}__{ov}" for lv in ("frequent","mid","rare") for ov in ("fav","avg","unf")]` | 설계 4.6 |
| D10.2 | C10 | 분기 | 거절 사유 `no_zeros`, `no_plant:<i>`, `check_fail:<env>:<key>`(ENV_KEYS 순서로 첫 환경, dict 순서로 첫 False 키). 모두 통과하면 수락 | 설계 4.7 |
| D10.3 | C10 | 종료 조건 | `generate_accepted(start, n, max_tries=2000)`: start부터 1씩 올려 n개 수락. 거절 목록도 돌려줌. 못 채우면 `StopGeneration` | 설계 4.7 |
| D12.1 | C12 | 형식 | npz 하나: `p0`, `h`, 환경마다 `p1__<env>`, `S1__<env>`, `R__<env>`, `Z__<env>`. `meta`(JSON 문자열): seed, shared, major, 수준별 rules, 수준별 dropped, 환경별 min_ratio. 불러온 값이 원본과 같아야 함 | 설계 9 |
| D13.1 | C13 | 수치 | `draw_cells(p, u)`: `c = np.cumsum(p); c /= c[-1]; idx = np.searchsorted(c, u, side="right")`, `np.minimum(idx, N_CELLS-1)` | 설계 4.8 |
| D13.2 | C13 | 난수 | $D_0$: `default_rng([pop, s, 10]).random(8000)` → $p_0$. $D_1$: `default_rng([pop, s, 11]).random(800)`의 앞 $n_1$개 → 환경의 $p_1$. 시험: `[pop, s, 12]`의 2000개 → $p_0$, `[pop, s, 13]`의 2000개 → $p_1$. 참조: `[pop, 14]`의 50,000개 → $p_0$, `[pop, 15]`의 50,000개 → $p_1$ | 설계 4.8 |
| D13.3 | C13 | 형식 | 행 = `CELLS[idx].astype(np.int64)`. `test_X = vstack([시험 minority, 시험 majority])`, `test_y = r_[ones(2000), zeros(2000)].astype(int)` | `populations.known_split` |
| D14.1 | C14 | 정의 | `O` = $D_1$ 셀이 있는 곳의 bool(65536,). `O0` 같은 방식. `SZ = S1 & ~O`, `SZ1 = SZ & O0`, `SZ2 = SZ & ~O0`, `SZ_nomaj = SZ & (p0 == 0)`, `STZ = ~S1`, `M0 = p1[SZ].sum()`. STZ는 나누지 않는다 | 설계 5 |
| D14.2 | C14 | 정의 | `N0 = pattern_counts(D0_idx)`, `N1 = pattern_counts(D1_idx)`, `pi0`, `pi1` | 설계 5 |
| D15.1 | C15 | 지표 | `sz_rec_mass` = $p_1(\hat S\cap\text{SZ})/M_0$, 같은 식으로 `sz1_rec_mass`, `sz2_rec_mass`, `nomaj_rec_mass`(분모는 각 집합의 $p_1$). `sz_rec_count` = 셀 수 비율. `stz_sup_count` = $1-|\hat S\cap\text{STZ}|/|\text{STZ}|$. `stz_sup_maj` = $1-p_0(\hat S\cap\text{STZ})/p_0(\text{STZ})$. `rule_left_<i>` = 규칙 $i$를 포함하는 셀이 $\hat S$에 하나라도 있으면 1. 분모 0이면 nan | 설계 7.2 (a) |
| D16.1 | C16 | 지표 | `Gamma` = $\hat q(\text{SZ})$, `Lambda` = $\hat q(\text{STZ})$, `tv_sz` = $\tfrac12\sum|\hat q(\cdot\mid\text{SZ}) - p_1(\cdot\mid\text{SZ})|$, $\hat q(\text{SZ})=0$이면 nan | 설계 7.2 (b) |
| D17.1 | C17 | 지표 | `leak` = $S_1$ 밖 행 비율. `novel` = $O$ 밖 행 비율. `rec_mass` = 맞힌 서로 다른 SZ 셀의 $p_1$ 합 / $M_0$. `rec_count` = 맞힌 서로 다른 SZ 셀 수 / |SZ|. `rule_rows_<i>` = 규칙 $i$를 포함하는 행 비율. `uni_tv_exact`, `pair_tv_exact` = 생성 행의 1·2차 주변분포와 $\pi_1$의 TV를 열(쌍)마다 구해 평균(8개, 28개) | 설계 7.2 (c) |
| D18.1 | C18 | 수식 | `signal` = $n_0\pi_0/x^*$. 이론 신호 $n_1 r\pi_0/\lambda^*$, $\lambda^*=\tau/\{\kappa\log(1+1/\kappa)\}$, $\kappa=n_0/(n_1 r)$, $\tau=\log(480/0.05)$ | 설계 4.6 |
| D18.2 | C18 | 수식 | 정확 인증 확률 $=(1-\pi_1)^{n_1}\cdot$ `binom.sf(x_star - 1, n0, pi0)` | 설계 7.2 (d) |
| D18.3 | C18 | 지표 | FWER 정확 상한 = $\pi_0>0$, $\pi_1>0$인 조각의 위 값 합. `n_false` = $\hat C$ 중 $\pi_1>0$인 조각 수, `any_false`. `cert_<i>` = 규칙 $i$의 조각이 $\hat C$에 있음. `all_cert` = 넷 다 | 설계 7.2 (d) |
| D18.4 | C18 | 통계량 | Clopper–Pearson 단측 상한 = `beta.ppf(0.95, k+1, n-k)`, $k=n$이면 1 | 설계 7.2 (d) |
| D19.1 | C19 | 판정 | 모든 DGP 점검 통과, 정답 점검 통과, 연결 점검 통과, 신호 배치(frequent $s_{200}<1\le s_{400}$, mid $s_{400}<1\le s_{800}$, rare $s_{800}<1$), 겹침(fav의 `SZ_nomaj` 질량 0, $n_1$마다 avg < unf) | 설계 8 |
| D20.1 | C20 | 경로 | 진입 스크립트 첫 줄에서 프로젝트 루트를 `sys.path`에 넣는다. 인터프리터 `/mnt/data1/calvinkim1027/miniconda3/envs/bomis/bin/python -I` | 계획 GC |
| D21.1 | C21 | 형식 | `vendor/bomis_fa088a9/bomis/{__init__,data,metrics,model,populations}.py`와 `LICENSE`. `SHA256SUMS` 줄 형식 `<sha256>  <상대경로>`. import는 `sys.path.insert(0, <root>/vendor/bomis_fa088a9)` | 설계 6.1 |
| D22.1 | C22 | 정의 | `PopAdapter.K = [4]*8`, `contains(X, c)` = $S_c$[`cell_index(X)`] ($S_0 = p_0>0$), `log_prob(X, c)` = `np.log`($p_c$[`cell_index(X)`]), 0이면 $-\infty$. `ref = {0: 참조 majority 행, 1: 참조 minority 행}` | 설계 6.2 |
| D23.1 | C23 | 호출 | A `Support(D0[:0], D1, K)`, B `Support(D0, D1, K, r=0.0)`, C `Support(D0, D1, K)`. $\hat S$ = `support.contains(CELLS.astype(np.int64))` | 설계 6.3 |
| D23.2 | C23 | 정의 | $\hat C$ = `(N1 == 0) & (N0 >= support.x_star)` (480,). 일치 확인: 표 B마다 박사형 규칙 `((N1>0) | pooled) & ~cert`로 다시 만든 admit이 `support.admit[B]`와 같고, $\hat C$ 개수가 `support.n_certified`와 같음. pooled는 `(N0>0) & ((N1==1).any() or len(B)>1)`이고, 단일 표의 `(N1==1).any()`는 그 열의 4개 수준에서 봄 | 설계 6.4, `model.py` 130행 |
| D24.1 | C24 | 호출 | `BOMiS(seed=sample_seed).fit(D0, D1, [4]*8)`, `G = model.sample(len(D0) - len(D1))`. 기본 인자. 1단계 연결 확인만 `steps=250` | 설계 6.5 |
| D25.1 | C25 | 할당 | `M_hat = model.M_hat`, `M_err = M_hat - M0`, `f0 = model.f0`, `L_prime` = $O$ 밖 생성 셀의 서로 다른 수, `n_sz` = |SZ|. 기록만: `steps = model.steps`, `maj_excluded = len(model.majority) > 0` | 설계 6.5, 7.2 (e) |
| D25.2 | C25 | 할당 | 거리 정답: SZ 셀마다 $O$의 행까지 최소 해밍 거리 $d$. $P_{\text{true}}(s)=\sum_{d=s} p_1/M_0$. `shell_tv` = $\tfrac12\sum_s|\text{shells}(s) - P_{\text{true}}(s)|$(합집합 위에서, 없는 키는 0). `model.shells`가 비면 nan | 설계 7.2 (e) |
| D25.3 | C25 | 할당 | fresh 정답: $D_0\cup D_1$ 행의 3차 조각(열 3개 56조합 × 64값) 개수를 세고, SZ 셀 중 개수 0인 3차 조각을 하나라도 가진 셀의 $p_1$ 합 / $M_0$. `ell_err = model.ell - 정답` | 설계 7.2 (e) |
| D25.4 | C25 | 할당 | `novel_tv` = $O$ 밖 생성 행의 셀별 비율과 $p_1(\cdot\mid\text{SZ})$의 $\tfrac12 L_1$. 새 행이 없으면 nan | 설계 7.2 (e) |
| D25.5 | C25 | 디코더 | `Q = np.exp(model.log_prob(CELLS64, y=1))`, `Qu = np.exp(model.log_prob(CELLS64, y=1, mask=False))`. `Q_sz` = Q[SZ] 합, `Q_stz` = Q[STZ] 합($\hat S$ 밖은 0이라 $Q(\hat S\cap\text{STZ})$와 같다), `Qu_stz` = Qu[STZ] 합, `Q_total`, `Qu_total`. `Q_tv` = $\tfrac12 L_1(Q(\cdot\mid\hat S\setminus O),\ p_1(\cdot\mid\text{SZ}))$ | 설계 7.2 (e) |
| D25.6 | C25 | 분해 | `R1` = $p_1(\hat S\cap\text{SZ})/M_0$. `R2` = $\hat S\cap\text{SZ}$의 $p_1$ 값을 내림차순으로 앞 $L'$개 합 / $M_0$. `R4` = 맞힌 서로 다른 SZ 셀의 $p_1$ 합 / $M_0$. `loss_support = 1-R1`, `loss_count = R1-R2`, `loss_select = R2-R4`. $\hat S$는 `model.support.contains` | 설계 7.2 (e) |
| D25.7 | C25 | 누출 | `n_stz_in_S` = $|\hat S\cap\text{STZ}|$, `p0_stz_in_S` = $p_0(\hat S\cap\text{STZ})$, `Q_stz`, `leak` | 설계 7.2 (e) |
| D25.8 | C25 | 순서 | fit → sample → 진단. `log_prob`는 난수를 쓰지 않는다. 밑줄로 시작하는 메서드는 부르지 않는다 | 설계 6.5 |
| D26.1 | C26 | 정의 | `ros(D1, n, rng)` = `D1[rng.integers(0, len(D1), n)]`, `rng = default_rng([pop, s, 30])` | 설계 10 |
| D27.1 | C27 | 입력 | 학습 표 = $D_0\cup D_1$, 열 `c0`–`c7`(값은 "0"–"3" 문자열)과 `target`("0"/"1"). 하위 프로세스: `python -m categorical_baselines.cli --model <name> --train <csv> --config <models.reference.json> --seed <sample_seed> --rows <n0-n1> --device <cpu|cuda> --output <새 폴더>` | baseline README |
| D27.2 | C27 | 환경 | 인터프리터 `/mnt/data1/calvinkim1027/miniconda3/envs/tabular_foundation/bin/python`. `PYTHONPATH = <baseline src>:<extras 압축 해제 폴더>:/mnt/data1/calvinkim1027/baseline_exp_runtime_v3/envs/generation`. extras는 `/mnt/data1/calvinkim1027/categorical_baselines_runtime_20260929/extras.tar.gz` | baseline `scripts/validate.sh` |
| D27.3 | C27 | 출력 | `<output>/synthetic_minority.csv.gz`를 읽어 `c0`–`c7`을 int64 행으로. 파일이 없거나 종료 코드가 0이 아니면 실패로 기록(`status.json`의 내용 포함)하고 대체 행을 만들지 않는다 | Q9 |
| D27.4 | C27 | 목록 | A단계 `("smoten", "smote_enc", "ctgan")`. `ctgan`만 `--device cuda`. B단계 `("tabsyn", "tabdiff")`는 같은 래퍼 | 설계 10, Q8 |
| D28.1 | C28 | 범위 | 모집단 = 개발용 처음 3개, 환경 `frequent__fav`, $n_1\in\{400, 800\}$, 표본 seed 0–9 (Q6) | 설계 11 |
| D28.2 | C28 | 출력 | 칸마다 `results/easy/<method>/<pop>_<n1>_<seed>.csv` 한 행. method ∈ {A, B, C, bomis, ros, smoten, smote_enc, ctgan}. A/B/C 행은 C15·C18, 생성 방법 행은 `evaluate` 6개 + C17, bomis는 C25도 | 설계 11 |
| D28.3 | C28 | 집계 | (방법, $n_1$)마다 30칸 평균. MCSE = 모집단별 평균 3개의 표준편차 / $\sqrt 3$. bomis − 각 baseline의 짝 차이(recovery, leak, control)도 같은 방식 | 설계 7.3 |

## 3. 모호점 (Q)

| ID | 관련 | 무엇이 불명확한가 | 결정 | 상태 |
|---|---|---|---|---|
| Q1 | C3 | "균등"의 구현 | 사전순 후보 목록에서 `rng.choice` | [임시 결정] (기존 코드) |
| Q2 | D16.1, D25 | TV 정의 | $\tfrac12 L_1$ | [임시 결정] |
| Q4 | D17.1 | 회수율을 서로 다른 셀 기준으로 | 서로 다른 셀 (박사형 `evaluate`와 같음) | [승인] |
| Q6 | D28.1 | 미리보기 표본 seed 수 | 설계 20개 중 0–9. 10–19는 미팅 뒤 같은 스크립트로 채움 | [임시 결정] 랩미팅 시간 때문 |
| Q7 | B단계 | SMOTE-NC는 imbalanced-learn에서 모든 열이 범주형이면 돌지 않음 | 초안에서 어떻게 돌렸는지 박사형에게 확인 | 미결 |
| Q8 | D27.4 | baseline 설정 | `models.reference.json` 그대로(CTGAN 300 epoch, batch 500) | [임시 결정] |
| Q9 | D27.3 | baseline 실패 처리 | 실패로 기록, 대체 행 없음 | [임시 결정] (baseline 코드 원칙) |
| Q10 | D27.1 | baseline seed | 표본 seed (BOMiS와 같은 규칙) | [임시 결정] |
| Q11 | D26.1, D27.1 | baseline이 쓰는 데이터 | ROS는 $D_1$만. 나머지는 CLI 형식대로 $D_0\cup D_1$ 표 | [임시 결정] (초안 G.2와 같음) |

## 4. 대응표

구현자가 채운다. 모든 C와 D에 `파일:함수:줄`을 적는다.

| ID | 코드 위치 | 비고 |
|---|---|---|
| C1 | | |
| … | | |

## 5. 검증 사례 (V)

| ID | 종류 | 입력 | 기대 결과 | 허용 오차 | 잡는 항목 | 계획 테스트 |
|---|---|---|---|---|---|---|
| V1 | 손계산 | 기존 T1 테스트 4개 | 기존 기대값 | 정확 | C1 | `tests/test_space.py` |
| V2 | 이론 성질 | 기존 T2 테스트 4개 | 기존 기대값 | 1e-10 | C2, C3 | `tests/test_laws.py` |
| V3 | 손계산 | $P_0(a')=0.4$, perm=(2,0,3) | 새 행 `[0.075, 0.7625, 0.15, 0.0125]` | 1e-15 | D4.5 | T2 `test_planted_row_hand` |
| V4 | 이론 성질 | 심기 통과 seed | 각 Rule $\pi_0=t_{lv}$, $e=t_{lv}/P_0(a')$, frequent $e\le0.2$, mid $e\le0.1$, rare $e\le1/60$ | 1e-12 | D4.2, D4.7 | T2 `test_planted_pi0_exact` |
| V5 | 이론 성질 | 버린 질량 0인 수준 | 부모 하나 고정 조각의 옮겨 나간 비율 $e$, 두 부모 $1-(1-e_i)(1-e_j)$ | 1e-10 | D7.1 | T2 `test_loss_fractions_exact` |
| V6 | 이론 성질 | 버린 질량 0인 환경 | $\min\pi_1/\pi_0 \ge 0.512$ | 1e-12 | C7, C8, 설계 4.5 | T2 `test_floor_bound_when_no_drop` |
| V7 | 손계산 | $\gamma\in\{0,0.5,1\}$ | $p_1(Z)=0, 0.1, 0.2$ | 1e-12 | D8.1 | T2 `test_mix_z_mass` |
| V8 | 이론 성질 | 9환경 | `pairwise_feasible_cells(pi1) == S1` | 정확 | D9.2 | T2 `test_local_support_all_envs` |
| V9 | 참조 구현 대조 | seed 0–59 | 수락 48개. seed 0–10 중 7만 거절 | 정확 | C10 전체, D4.3 난수 순서 | T3 `test_acceptance_matches_reference` |
| V10 | 성질 | 저장 후 불러오기 | 배열·규칙·0 쌍이 같음 | 정확 | D12.1 | T3 `test_roundtrip` |
| V11 | 극한 사례 | 마지막 셀 확률 0, $u$가 1에 가까움 | 확률 0인 셀이 안 나옴 | 정확 | D13.1 | T4 `test_draw_cells_zero_mass` |
| V12 | 성질 | 같은 (pop, s) | $D_1(400)$ = $D_1(800)$의 앞 400행, 환경 간 같은 균등 난수 | 정확 | D13.2 | T4 `test_nested_and_common_random` |
| V13 | 성질 | 정답 | SZ1, SZ2가 SZ를 나눔, $M_0 = 1-p_1(O)$, fav에서 `SZ_nomaj` 없음 | 1e-12 | D14.1 | T4 `test_truth_partition` |
| V14 | 손계산 | $\hat S = S_1$, $O$, $U$ | 회수율·억제율 (1,1), (0,1), (1,0) | 정확 | D15.1 | T5 `test_score_set_extremes` |
| V15 | 손계산 | STZ 셀 3개 + SZ 셀 2개 × 2 | leak 3/7, rec_count 2/|SZ| | 1e-12 | D17.1 | T5 `test_score_rows_hand` |
| V16 | 손계산 | $\hat q = p_1(\cdot\mid\text{SZ})$ | Gamma 1, Lambda 0, tv 0. $\hat q(\text{SZ})=0$이면 tv nan | 1e-12 | D16.1 | T5 `test_score_law` |
| V17 | 손계산 | $\pi_0=0.5$, $n_0=10$, $x^*=8$ | 56/1024. $\pi_1=0.1$, $n_1=5$면 $0.9^5\cdot56/1024$ | 1e-15 | D18.2 | T5 `test_exact_cert_prob_hand` |
| V18 | 손계산 | CP $k=0$, $n=3000$ | $1-0.05^{1/3000}$ | 1e-12 | D18.4 | T5 `test_clopper_pearson` |
| V19 | 참조 구현 대조 | `certify_threshold(8000, n1, 0.5, 0.05/480)` | 2603, 1346, 680, 340, 170 | 정확 | C21 import, D18.1 | T1 `test_vendor_threshold_values` |
| V20 | 성질 | 고정 사본 | 파일별 SHA-256이 `SHA256SUMS`와 같음 | 정확 | D21.1 | T1 `test_vendor_hashes` |
| V21 | 이론 성질 | 무작위 표본 | $\hat S_A \subseteq \hat S_C \subseteq \hat S_B$, B `n_certified == 0`, A의 $\hat S$ = 36개 조각이 모두 $N_1>0$인 셀 | 정확 | D23.1 | T6 `test_support_nesting` |
| V22 | 참조 구현 대조 | 무작위 표본 | D23.2 재구성이 `support.admit`, `n_certified`와 같음 | 정확 | D23.2 | T6 `test_cert_reconstruction` |
| V23 | 참조 구현 대조 | 무작위 행, SZ 셀 각 1번 | `evaluate`의 recovery·leak = C17 값, SZ 각 1번이면 recovery 1, leak 0 | 1e-12 | D22.1, D17.1 | T6 `test_evaluate_agrees` |
| V24 | 손계산 | $\hat S=S_1$, 새 행 = $p_1$ 큰 SZ 셀 $L'$개 | $R_4 = R_2$. $L' \ge |\text{SZ}|$이면 $R_1=R_2=1$ | 1e-12 | D25.6 | T7 `test_decomposition_hand` |
| V25 | 이론 성질 | BOMiS `steps=250` 한 번 | $R_4\le R_2\le R_1\le 1$, `Qu_total`=1±1e-4, `Q_total`≤1+1e-4, 진단 전후 `model.rng.bit_generator.state` 같음 | 표기 | D25.5–D25.8 | T7 `test_bomis_diag_smoke` (sbatch) |
| V26 | 성질 | ROS | 생성 행이 모두 $D_1$의 행, 개수 $n$ | 정확 | D26.1 | T8 `test_ros` |
| V27 | 끝에서 끝까지 | 외부 래퍼, smoten, $n_1=50$, rows 16 | 16행, 값 0–3, 또는 실패 기록 | 정확 | D27.1–D27.3 | T8 `test_external_smoten_smoke` (sbatch) |
| V28 | 끝에서 끝까지 | `run_check.py --quick` | verdict의 모든 항목 통과 | 정확 | C19 | T9 |
| V29 | 끝에서 끝까지 | `run_easy.py --quick` | 8개 방법 행이 모두 생기고 요약 md가 만들어짐 | 정확 | C28 | T10 |

## 6. 결과 민감 항목

검토에서 먼저 볼 것이다.

- **D4.3, D4.4**: 순열 네 개를 거절 판정 전에 모두 뽑아야 참조 구현과 같은 모집단이 나온다(V9).
- **D4.2**: 부모 확률은 0을 넣은 뒤의 법칙에서 구한다. 0을 넣기 전 값으로 하면 $\pi_0$가 목표와 어긋난다.
- **D13.2**: 환경 간 같은 난수, $n_1$ 사이 포함 관계. 짝지은 비교의 전제다.
- **D18.2**: `binom.sf(x_star - 1, ...)`의 `-1`.
- **D25.6**: $R_2$는 $\hat S\cap\text{SZ}$ 안에서 고른다. $S_1$ 전체에서 고르면 부등식이 깨질 수 있다.
- **D24.1, Q8**: BOMiS와 baseline의 기본 설정. 바꾸면 결과가 달라진다.
- **D27.1**: baseline 학습 표의 열 순서와 문자열 값. 잘못 되면 생성 행이 엉뚱한 열로 들어간다.
