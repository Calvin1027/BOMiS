# 합성 실험 v2 (9환경) 구현 계획

> **For agentic workers:** 이 계획은 Codex가 작업 단위로 구현한다. 각 작업의 테스트를 먼저 쓰고, 실패를 확인한 뒤 구현하고, 통과를 확인한다. 단계는 체크박스(`- [ ]`)로 표시한다. 이 폴더는 git 저장소가 아니므로 커밋 단계는 없다.

**Goal:** 정답을 아는 9환경 합성 모집단에서 박사형 BOMiS(고친 데 없는 사본)와 baseline을 돌리고, support·할당·디코더 단계별로 채점한다. 랩미팅(10/27 14:00) 전에 쉬운 환경 `frequent__fav`의 미리보기 결과를 낸다.

**Architecture:** `dgp/`에 모집단 생성·표본·정답·채점, `vendor/`에 박사형 코드 고정 사본, `dgp/harness.py`와 `dgp/bomis_diag.py`에 박사형 코드 연결과 단계별 진단, `baselines/`에 ROS와 외부 baseline 래퍼를 둔다. 실행은 `scripts/`의 진입 스크립트를 sbatch로 돌린다.

**Tech Stack:** Python 3.11.17 (`bomis` env: numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.9.1, torch 2.5.1), pytest 9.1.1. 외부 baseline은 `tabular_foundation` env + baseline 프로젝트 PYTHONPATH.

**Spec:**
- 설계: `docs/superpowers/specs/2026-10-26-synthetic-study-ademp-design.md`
- 구성 요소 명세서: `docs/algo_specs/stage1_v2_spec.md` (C, D, Q, V ID. 코드 주석에 담당 ID를 적는다)

## Global Constraints

- 프로젝트 루트 `/data1/calvinkim1027/BOMIS_new_수만이형`. 실행 사본 `/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime`(계산 노드는 `/mnt/data1`만 본다). 동기화는 `bash experiments/sync_runtime.sh`.
- Python은 srun/sbatch로만 돈다. 옵션 `-p brl --ntasks=1 --exclude=brl1,brl2`. 테스트는 `--qos=normal`. 본실행 자원은 제출 때 정한다.
- 인터프리터 `/mnt/data1/calvinkim1027/miniconda3/envs/bomis/bin/python -I`. 진입 스크립트는 첫 줄에서 프로젝트 루트를 `sys.path`에 넣는다(D20.1).
- `vendor/bomis_fa088a9/`와 baseline 프로젝트 코드는 고치지 않는다.
- 난수는 모두 `np.random.default_rng([정수, ...])` 형식(명세서 D2.2, D3.1, D4.3, D13.2, D26.1).
- 값: $d=8$, $K=4$, $\eta=1$, $t$ = 0.06/0.03/0.005, `P_MIN=0.3`, $\omega=0.2$, $\gamma$ = 0/0.5/1, $n_0=8000$, $n_1\in\{50,100,200,400,800\}$, $\alpha=0.05$, $r=0.5$, $k=2$, 시험 2000/클래스, 참조 50,000/클래스.
- 테스트 시드는 9000번대, 개발용 모집단은 seed 0부터 처음 수락된 10개.
- 본실행(`run_easy.py` 전체, `run_check.py` 전체)은 Claude 3회 검토와 사용자 승인 뒤에만 돈다. 구현 중에는 `--quick`과 pytest만 돈다.

## Review Focus

1. **버린 질량이 있어도 점검을 통과한 모집단**은 수락되어야 한다(수락은 증명이 아니라 정확 점검으로 판정). → T3 `test_accept_by_check_not_by_drop`
2. **BOMiS가 새 행을 하나도 만들지 않는 경우**($\hat M=0$ 등): 진단이 멈추지 않고 $L'=0$, $R_2=R_4=0$, `novel_tv`는 nan. → T7 `test_diag_no_novel_rows`
3. **외부 baseline 실패**(`CAPACITY_FAILURE`, 출력 없음, 종료 코드 ≠ 0): 그 방법만 실패 행으로 남고 같은 칸의 다른 방법은 계속 돈다. → T8 `test_external_failure_recorded`, T10 `test_cell_continues_after_failure`
4. **다시 제출한 칸**: 결과 csv가 이미 있으면 그 방법은 건너뛴다. 외부 CLI의 출력 폴더는 매번 새로 만든다. → T10 `test_cell_skips_existing`
5. **minority 표본에 어떤 열의 수준이 하나도 없는 경우**: $\hat C$ 재구성이 박사형의 단일 표 규칙(`(N1==1).any()`)과 같아야 한다. → T6 `test_cert_reconstruction`에 그런 표본을 하나 넣는다

---

### Task 1: 박사형 코드 고정 사본과 동기화 (C21, C20)

**Files:**
- Create: `vendor/bomis_fa088a9/bomis/{__init__,data,metrics,model,populations}.py`, `vendor/bomis_fa088a9/LICENSE` (원본 `refs/suman_repo_clone/bomis/`에서 복사), `vendor/bomis_fa088a9/SHA256SUMS`
- Create: `dgp/vendor.py`
- Modify: `experiments/sync_runtime.sh` (동기화 목록에 `vendor`, `baselines` 추가)
- Test: `tests/test_vendor.py`

**Interfaces:**
- Produces: `dgp.vendor.VENDOR_ROOT: Path`, `dgp.vendor.load_bomis() -> module`(`bomis` 패키지, `sys.path` 앞에 `VENDOR_ROOT`를 넣고 import), `dgp.vendor.verify_hashes() -> dict[str, bool]`

- [ ] **Step 1: 테스트 작성**
  - `test_vendor_hashes`: `verify_hashes()`의 값이 모두 True이고 키가 6개(LICENSE 포함).
  - `test_vendor_threshold_values` (V19): `certify_threshold(8000, n1, 0.5, 0.05/480)`가 n1 = 50, 100, 200, 400, 800에서 `[2603, 1346, 680, 340, 170]`.
- [ ] **Step 2: 실패 확인** — `bash experiments/sync_runtime.sh && srun -p brl --ntasks=1 --qos=normal --exclude=brl1,brl2 --cpus-per-task=1 --mem=4G --chdir=/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime /mnt/data1/calvinkim1027/miniconda3/envs/bomis/bin/python -I -m pytest tests/test_vendor.py -v` → import 실패
- [ ] **Step 3: 복사와 구현.** 원본은 `cp`로 복사하고 `sha256sum`으로 `SHA256SUMS`를 만든다(줄 형식 `<sha>  bomis/model.py`).
- [ ] **Step 4: 통과 확인** — 같은 명령, 2 passed

### Task 2: 금지 쌍 심기와 minority 점검 정리 (C4, C6–C9)

**Files:**
- Modify: `dgp/minority.py` — 상수 교체(`T_LEVELS`, `P_MIN`, `OMEGA`, `GAMMAS`, `R_TRUE`), `plant_pairs` 추가, `select_pairs`·`select_triples`·`FREQ_LEVELS`·`E_MAX_TRIPLE`·`TRIPLE_RANGE`·`TRIPLE_PARENTS`·`ASSUMPTIONS` 삭제, `check_minority`에서 triple 분기와 `triples_minimal_ok` 삭제(D9.2)
- Modify: `tests/test_minority.py` — fixture를 심기 기반으로 바꾸고 triple 테스트 삭제

**Interfaces:**
- Consumes: `dgp.laws.TreeTables`, `tree_law`, `apply_zeros`, `ParentZero`; `dgp.space.pattern_probs`, `pattern_id`
- Produces:
  - `T_LEVELS: dict[str, float]` = `{"frequent": 0.06, "mid": 0.03, "rare": 0.005}`, `P_MIN = 0.3`
  - `plant_pairs(t: TreeTables, zeros: list[ParentZero], rng) -> tuple[TreeTables, dict[str, list[Rule]]] | str` — D4.2–D4.6. 실패 시 `"no_plant:<i>"`. 입력 `t`는 바꾸지 않는다
  - 기존 `Rule`, `regions`, `move_mass`, `mix`, `pairwise_feasible_cells`, `rule_mask`, `rule_pid` 유지
  - `check_minority(p0, p1, S1, Z, rules, gamma, r=R_TRUE) -> dict` 키: `sum_ok, support_ok, local_support_ok, infeasible_ok, z_mass_ok, min_ratio, floor_ok, passed`

- [ ] **Step 1: 테스트 작성** (fixture `prep`: seed 9000부터 `plant_pairs`가 성공하고 세 수준의 9환경이 모두 점검을 통과하는 첫 seed. 난수는 명세서 D2.2, D3.1, D4.3)
  - `test_planted_row_hand` (V3): `P_0(a')=0.4`가 되도록 만든 작은 표에서 perm (2,0,3)이면 새 행 `[0.075, 0.7625, 0.15, 0.0125]` (1e-15). 표를 직접 만들 수 없으면 `plant_pairs` 내부의 행 계산을 `_planted_row(p_parent: float, perm: tuple[int,int,int]) -> np.ndarray`로 분리해 그것을 시험한다
  - `test_planted_pi0_exact` (V4): 각 Rule의 $\pi_0$가 목표와 1e-12 이내, $e = t/P_0(a')$, frequent $e\le0.2$, mid $e\le0.1$, rare $e\le1/60$; 입력 표가 바뀌지 않음
  - `test_plant_rejects_low_parent`: 모든 부모 확률이 0.3 미만인 표(뿌리·사슬 행을 균등 0.25로)에서 `"no_plant:0"`
  - `test_loss_fractions_exact` (V5): 버린 질량이 0인 수준에서, 질량 이동 전후(재정규화 전)의 조각 $(P_i=a'_i)$ 손실 비율이 $e_i$, $(P_i=a'_i, P_j=a'_j)$는 $1-(1-e_i)(1-e_j)$ (1e-10)
  - `test_floor_bound_when_no_drop` (V6): 버린 질량 0인 환경의 `min_ratio >= 0.512 - 1e-12`
  - `test_mix_z_mass` (V7), `test_local_support_all_envs` (V8): 9환경 모두 `local_support_ok`
  - `test_check_detects_support_violation`: $p_1$의 $S_1$ 셀 하나를 0으로 하면 `support_ok` False
- [ ] **Step 2: 실패 확인** — pytest `tests/test_minority.py`, `plant_pairs` 없음
- [ ] **Step 3: 구현** (명세서 D4.1–D4.7, D9.2)
- [ ] **Step 4: 통과 확인** — `tests/test_space.py tests/test_laws.py tests/test_minority.py` 모두 통과

### Task 3: 모집단 조립·수락·저장 (C10, C12)

**Files:**
- Create: `dgp/population.py`, `scripts/build_pops.py`
- Test: `tests/test_population.py`

**Interfaces:**
- Consumes: Task 2 전부, `dgp.laws.draw_tree_tables`, `choose_parent_zeros`
- Produces:
  - `ENV_KEYS: list[str]` (D10.1), `LEVELS = ("frequent", "mid", "rare")`
  - `@dataclass Env: p1, S1, R, Z: np.ndarray; gamma: float; level: str; check: dict`
  - `@dataclass Population: seed: int; p0: np.ndarray; h: np.ndarray; shared: list[ParentZero]; major: list[ParentZero]; rules: dict[str, list[Rule]]; dropped: dict[str, float]; envs: dict[str, Env]`
  - `build_population(pop_seed: int) -> Population | str` (D10.2. 순서는 명세서 1절 호출 순서)
  - `generate_accepted(start: int, n: int, max_tries: int = 2000) -> tuple[list[Population], list[tuple[int, str]]]`, 예외 `StopGeneration`
  - `save_population(pop: Population, path: Path) -> None`, `load_population(path: Path) -> Population` (D12.1)
  - `scripts/build_pops.py --start 0 --n 10 --out results/pops` → `results/pops/pop_<seed>.npz`, `results/pops/rejections.csv`

- [ ] **Step 1: 테스트 작성**
  - `test_acceptance_matches_reference` (V9): seed 0–59에서 수락 48개, seed 0–10 중 거절은 7뿐
  - `test_accept_by_check_not_by_drop` (Review Focus 1): 수락된 seed 중 `dropped` 값이 0보다 큰 수준이 있는 모집단이 하나 이상 있고(참조 구현에서 52개 중 34개 seed가 버린 질량 > 0), 그 모집단의 모든 환경 `check["passed"]`가 True
  - `test_env_keys_and_paired`: 9개 키 순서, 같은 수준의 세 환경이 같은 `rules`를 공유, `p1(Z)` = 0, 0.1, 0.2
  - `test_roundtrip` (V10): 저장 후 불러온 배열이 같고(`np.array_equal`), rules·shared·major가 튜플 비교로 같음
- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — V9는 seed 60개 전체 점검이라 약 1분, `--cpus-per-task=4 --mem=16G`

### Task 4: 표본과 정답 (C13, C14)

**Files:**
- Create: `dgp/sampling.py`, `dgp/truth.py`
- Test: `tests/test_sampling_truth.py`

**Interfaces:**
- Produces:
  - `draw_cells(p: np.ndarray, u: np.ndarray) -> np.ndarray` (D13.1)
  - `@dataclass Split: D0_idx, D1_idx, test_X, test_y` + 속성 `D0`, `D1`(행, int64)
  - `make_split(pop: Population, env: str, n1: int, sample_seed: int, n0: int = 8000, n_test: int = 2000) -> Split` (D13.2, D13.3)
  - `reference_rows(pop: Population, env: str, n_ref: int = 50000) -> dict[int, np.ndarray]` (키 0, 1)
  - `@dataclass Truth: env, p0, p1, S1, O, O0, SZ, SZ1, SZ2, SZ_nomaj, STZ, M0, N0, N1, pi0, pi1, rules, n0, n1`
  - `make_truth(pop: Population, env: str, split: Split) -> Truth` (D14.1, D14.2)

- [ ] **Step 1: 테스트 작성**
  - `test_draw_cells_zero_mass` (V11): 마지막 셀 확률 0, `u=[0.0, 0.5, np.nextafter(1.0, 0.0)]`에서 확률 0인 셀이 안 나옴
  - `test_nested_and_common_random` (V12): `make_split(..., 400, s).D1` = `make_split(..., 800, s).D1[:400]`; 두 환경의 $D_1$ 균등 난수가 같음(같은 $p_1$이면 같은 행); $D_0$는 환경과 무관
  - `test_truth_partition` (V13): SZ1 ∪ SZ2 = SZ, 서로소; $M_0 = 1 - p_1(O)$ (1e-12); fav 환경에서 `SZ_nomaj.any()`가 False; `STZ == ~S1`
  - `test_split_format`: `test_X`의 앞 2000행이 minority, `test_y`의 합 2000
- [ ] **Step 2: 실패 확인** — **Step 3: 구현** — **Step 4: 통과 확인**

### Task 5: 채점기 (a)–(d) (C15–C18)

**Files:**
- Create: `dgp/scorer.py`, `dgp/cert_metrics.py`
- Test: `tests/test_scorer.py`

**Interfaces:**
- Produces:
  - `score_set(truth: Truth, S_hat: np.ndarray) -> dict` (D15.1)
  - `score_law(truth: Truth, q: np.ndarray) -> dict` (D16.1)
  - `score_rows(truth: Truth, G_idx: np.ndarray) -> dict` (D17.1)
  - `exact_cert_prob(pi0, pi1, n0, n1, x_star) -> np.ndarray`, `exact_fwer_bound(truth, x_star) -> float`, `clopper_pearson_upper(k, n, level=0.95) -> float`, `signal(pi0, n0, x_star)`, `theory_signal(pi0, n0, n1, r=0.5)`, `power_curve(pi0_grid, n0, n1, x_star) -> np.ndarray`
  - `score_cert(truth: Truth, C_hat: np.ndarray, x_star: int) -> dict` (D18.3; 키 `cert_0..3`, `all_cert`, `n_false`, `any_false`, `fwer_bound`, `signal_0..3`, `exact_power_0..3`)

- [ ] **Step 1: 테스트 작성** (작은 Truth는 fixture 모집단에서 만든다)
  - `test_score_set_extremes` (V14), `test_score_rows_hand` (V15), `test_score_law` (V16), `test_exact_cert_prob_hand` (V17), `test_clopper_pearson` (V18)
  - `test_nan_denominators`: SZ가 빈 Truth에서 `sz_rec_mass`가 nan
- [ ] **Step 2–4**: 실패 확인, 구현, 통과 확인

### Task 6: 박사형 코드 연결 — 연결부, A/B/C, $\hat C$, BOMiS 실행 (C22–C24)

**Files:**
- Create: `dgp/harness.py`
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: Task 1 `load_bomis`, Task 4 `Split`, `Truth`, `reference_rows`
- Produces:
  - `class PopAdapter: K, contains(X, c), log_prob(X, c), ref` (D22.1); `make_pop_adapter(pop, env, ref: dict[int, np.ndarray]) -> PopAdapter`
  - `split_dict(split: Split) -> dict` (키 `D0, D1, test_X, test_y`)
  - `run_support(method: str, split: Split)` → `(S_hat: np.ndarray, support)`; method ∈ `{"A","B","C"}` (D23.1)
  - `cert_set(support, split: Split) -> np.ndarray` (480,) bool, `check_cert_reconstruction(support, split) -> bool` (D23.2)
  - `run_bomis(split: Split, seed: int, **kw) -> tuple[model, np.ndarray]` (생성 행, D24.1)
  - `evaluate_rows(G: np.ndarray, split: Split, adapter: PopAdapter) -> dict` (박사형 `evaluate` 그대로 호출)

- [ ] **Step 1: 테스트 작성**
  - `test_support_nesting` (V21)
  - `test_cert_reconstruction` (V22, Review Focus 5): 일반 표본 하나와, $D_1$에서 열 0의 한 수준을 모두 다른 수준으로 바꾼 표본 하나(그 수준이 minority에 0번)
  - `test_evaluate_agrees` (V23): 무작위 행 500개에서 `evaluate`의 recovery·leak가 `score_rows`의 `rec_mass`·`leak`와 1e-12 이내; SZ 셀 각 1번이면 recovery 1, leak 0
- [ ] **Step 2–4**: 실패 확인, 구현, 통과 확인

### Task 7: BOMiS 단계별 진단 (C25)

**Files:**
- Create: `dgp/bomis_diag.py`
- Test: `tests/test_bomis_diag.py`, `experiments/bomis_diag_smoke.sbatch`

**Interfaces:**
- Consumes: Task 4 `Truth`, Task 6 `run_bomis`
- Produces:
  - `decomposition(truth: Truth, S_hat: np.ndarray, G_idx: np.ndarray) -> dict` (D25.6; `R1, R2, R4, L_prime, loss_*`)
  - `true_shell_law(truth: Truth) -> dict[int, float]` (D25.2), `true_fresh_share(truth: Truth, D0_idx, D1_idx) -> float` (D25.3)
  - `bomis_diagnostics(model, truth: Truth, split: Split, G: np.ndarray) -> dict` (D25.1–D25.8 전부, `decomposition` 포함)

- [ ] **Step 1: 테스트 작성**
  - `test_decomposition_hand` (V24): $\hat S = S_1$, 생성 행 = $p_1$ 큰 SZ 셀 $L'$개 각 1번 + $O$의 행 → `R4 == R2`; $L' \ge |\text{SZ}|$이면 `R1 == R2 == 1`
  - `test_decomposition_order_random`: 무작위 $\hat S \supseteq$ 생성 새 셀에서 `R4 <= R2 + 1e-12 <= R1 + 2e-12`
  - `test_diag_no_novel_rows` (Review Focus 2): 생성 행이 모두 $O$의 행이면 `L_prime == 0`, `R2 == R4 == 0`, `novel_tv`가 nan
  - `test_true_shell_law_sums_to_one`: 값 합 1 (1e-12)
  - `test_bomis_diag_smoke` (V25, `@pytest.mark.slow`, sbatch에서만): fixture 모집단 `frequent__fav`, $n_1=100$, 표본 seed 0, `steps=250`. `R4 <= R2 <= R1 <= 1`, `abs(Qu_total - 1) < 1e-4`, `Q_total <= 1 + 1e-4`, `bomis_diagnostics` 호출 전후 `model.rng.bit_generator.state`가 같음
- [ ] **Step 2: 실패 확인** — 빠른 테스트는 srun, smoke는 `sbatch experiments/bomis_diag_smoke.sbatch`(CPU 4, 16G, 30분)
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**

### Task 8: baseline — ROS와 외부 래퍼 (C26, C27)

**Files:**
- Create: `baselines/__init__.py`, `baselines/ros.py`, `baselines/external.py`, `experiments/sync_baselines.sh`, `experiments/baseline_smoke.sbatch`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Produces:
  - `ros(D1: np.ndarray, n: int, rng) -> np.ndarray` (D26.1)
  - `A_MODELS = ("smoten", "smote_enc", "ctgan")`, `B_MODELS = ("tabsyn", "tabdiff")`, `DEVICE = {"ctgan": "cuda", "tabsyn": "cuda", "tabdiff": "cuda", "smoten": "cpu", "smote_enc": "cpu"}`
  - `run_external(name: str, D0: np.ndarray, D1: np.ndarray, n: int, seed: int, workdir: Path) -> dict` → `{"ok": bool, "rows": np.ndarray | None, "reason": str, "seconds": float}` (D27.1–D27.3)
  - `experiments/sync_baselines.sh`: `/data1/calvinkim1027/과거 BOMIS/TDG_baseline_categorical`의 `src`, `configs`, `third_party`를 `<runtime>/baselines_src/`로 rsync
  - 환경 변수: `BL_PYTHON`(기본 D27.2 인터프리터), `BL_DEPS`(job이 extras를 푼 폴더), `BL_SRC`(`<runtime>/baselines_src`)
  - sbatch에서 extras 풀기: `deps=${SLURM_TMPDIR:-/tmp}/bl_deps_$SLURM_JOB_ID; mkdir -p $deps; tar -xzf /mnt/data1/calvinkim1027/categorical_baselines_runtime_20260929/extras.tar.gz -C $deps; export BL_DEPS=$deps`

- [ ] **Step 1: 테스트 작성**
  - `test_ros` (V26)
  - `test_train_table_format`: 래퍼가 쓰는 학습 표의 열이 `c0..c7, target`, 값이 문자열, minority 행 수 = `len(D1)`
  - `test_external_failure_recorded` (Review Focus 3): 존재하지 않는 모델 이름이나 실패하는 가짜 인터프리터(`BL_PYTHON=/bin/false`)로 `ok == False`, `rows is None`, 예외 없음
  - `test_external_smoten_smoke` (V27, slow, sbatch에서만): fixture 모집단, $n_1=50$, `n=16` → 16행, 값 0–3
- [ ] **Step 2: 실패 확인** — **Step 3: 구현** — **Step 4: 통과 확인** (`sbatch experiments/baseline_smoke.sbatch`: smoten과 ctgan을 각 1번, ctgan은 `--gres=gpu:1`, 소요 시간을 로그에 남김)

### Task 9: 1단계 점검 실행기 (C19, C20)

**Files:**
- Create: `dgp/check.py`(점검·판정 함수), `scripts/run_check.py`(진입), `experiments/dgp_check.sbatch`
- Test: `tests/test_run_check.py`

**Interfaces:**
- Consumes: Task 3–7
- Produces: `scripts/run_check.py --pops results/pops --out results/dgp_check [--quick] [--n-pops N] [--seeds S]`
  - 출력: `env_checks.csv`(모집단×환경), `truth_stats.csv`((모집단, 환경, $n_1$)별 설계 8절 7의 기술 통계 평균), `oracle.csv`(V14–V18, V21–V24의 입력·기대·실제·통과), `connection.csv`(V19–V23, V25), `verdict.json`(D19.1 항목별 True/False와 `stage1_pass`), `reports/stage1_dgp_check.md`
  - `--quick`: 9000번대 seed로 모집단 2개, 표본 seed 2개, $n_1 \in \{100, 800\}$

- [ ] **Step 1: 테스트 작성** — `test_help_exit_zero`: `-I`로 `--help` 종료 코드 0. `test_quick_end_to_end` (V28, slow): `--quick` 실행 뒤 `verdict.json`의 모든 값 True
- [ ] **Step 2–4**: 실패 확인, 구현, 통과 확인 (`--quick`은 sbatch, CPU 4, 16G)

### Task 10: 쉬운 환경 실행과 요약 (C28)

**Files:**
- Create: `scripts/run_easy.py`, `scripts/summarize_easy.py`, `experiments/easy_cpu.sbatch`, `experiments/easy_gpu.sbatch`
- Test: `tests/test_run_easy.py`

**Interfaces:**
- Consumes: Task 3–8
- Produces:
  - `run_cell(pop_path: Path, n1: int, sample_seed: int, methods: list[str], out: Path, env: str = "frequent__fav", bomis_kw: dict | None = None) -> None` — 방법마다 `out/<method>/<pop>_<n1>_<seed>.csv` 한 행(D28.2). 이미 있으면 건너뜀. 한 방법이 실패해도 다음 방법을 돈다(실패 행에 `ok=0`, `reason`)
  - `scripts/run_easy.py --task-id <i> --group cpu|gpu [--quick]`: 칸 목록 = 모집단 3개 × $n_1$ (400, 800) × 표본 seed 0–9 = 60칸, 순서는 $n_1$=400 먼저. `cpu` 그룹은 `A, B, C, bomis, ros, smoten, smote_enc`, `gpu` 그룹은 `ctgan`
  - `experiments/easy_cpu.sbatch`(array 0–59, CPU 4, 16G, 1시간), `experiments/easy_gpu.sbatch`(array 0–59, GPU 1, CPU 6, 24G, 1시간). 자원과 동시 실행 수는 제출 때 확정
  - `scripts/summarize_easy.py --in results/easy --out reports/easy_preview.md`: (방법, $n_1$)별 평균·MCSE(D28.3), bomis − baseline 짝 차이, bomis의 회수율 분해 표($R_1, R_2, R_4$와 손실 셋), 실패 수
  - `--quick`: fixture 모집단 1개, $n_1=400$, seed 0, BOMiS `steps=250`, 외부 baseline은 `configs/models.smoke.json`

- [ ] **Step 1: 테스트 작성**
  - `test_cell_skips_existing` (Review Focus 4): csv가 있는 방법은 다시 돌지 않음(호출 횟수로 확인)
  - `test_cell_continues_after_failure` (Review Focus 3): 외부 방법 하나가 실패해도 뒤의 방법 csv가 생김
  - `test_task_index_mapping`: task id 0–59가 60칸을 하나씩, 겹침 없이 덮음
  - `test_quick_end_to_end` (V29, slow): cpu·gpu 그룹 `--quick` 뒤 8개 방법 csv와 `reports/easy_preview.md`
- [ ] **Step 2–4**: 실패 확인, 구현, 통과 확인

---

## 실행 순서 (구현과 검토가 끝난 뒤, 사용자 승인 후)

1. `scripts/build_pops.py --start 0 --n 10` (CPU 4, 약 2분)
2. `scripts/run_check.py`를 개발용 모집단 3개로(쉬운 환경에 쓰는 것) 먼저, 통과하면 10개로
3. `easy_cpu.sbatch`와 `easy_gpu.sbatch` 제출. 예상: BOMiS fit 60번 × 약 585초, 8개 동시로 약 1.2시간. CTGAN 60번은 GPU 4개로 baseline 스모크 소요 시간 × 15
4. `scripts/summarize_easy.py` → `reports/easy_preview.md`

## B단계 (랩미팅 뒤, 이 계획 밖)

- 표본 seed 10–19 채우기, TabSyn·TabDiff(같은 래퍼, `B_MODELS`), TabDDPM·ForestDiffusion 범주형 어댑터 이식, SMOTE-NC(명세서 Q7), 나머지 축(진행 순서 2단계).
