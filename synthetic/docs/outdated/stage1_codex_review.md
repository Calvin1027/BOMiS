# Stage 1 Codex 구현·검토 기록

## 범위

- 기준: 2026-10-23 수정 설계, 구현 계획, 구성 요소 명세서, stage1_dgp_prompt.md.
- 허용 실행: 계획의 42개 pytest와 테스트에 포함된 `--quick`, CPU Slurm만.
- 예비 실행, 개발용 모집단 본실행, GPU, 최종 보고용 seed는 실행하지 않는다.
- 테스트 기대값과 문서 원문은 수정하지 않는다.
- 수정된 local support 비교는 `F & ~S1 == F & triple`이고 비어 있지 않아야 한다.
- quick oracle은 `2 * 18 * 9 + 3 = 327`행을 확인한다.

## 작업별 RED / GREEN

Task 3 fixture 실패로 중단. 원본 srun 로그는 runtime의 `logs/task<N>_{red,green}.log`에 보존한다.

### Task 1

- RED: job 11795, 종료 코드 2, `ModuleNotFoundError: No module named 'dgp.space'` (계획한 수집 실패).
- 첫 실행은 brl0에서 공유 파일시스템의 Python 모듈 읽기가 지연됨. 환경·인터프리터·테스트 옵션 변경 없이 완료.
- 셀·조각 번호, 36개 조각/셀, bincount 집계 및 실행 뼈대 구현.
- GREEN: job 11807 (brl3), 4 passed. job 11803은 brl0의 반복 I/O 지연으로 종료 후 같은 테스트를 brl3에서 재시도함. 이후 테스트에도 `--nodelist=brl3`를 추가해 실행 위치만 고정한다(과학 설정·자원 제한 동일).

### Task 2

- RED: job 11811, 종료 코드 2, `ModuleNotFoundError: No module named 'dgp.laws'`.
- 공유 0 → Z 쌍 순서, 삽입 전 트리 확률 기준의 사전순 후보 균등 추출, 삽입 후 정규화 구현.
- GREEN: job 11812, 4 passed.

### Task 3

- RED: job 11814, 종료 코드 2, `ModuleNotFoundError: No module named 'dgp.minority'`.
- 0 삽입 후 p0로 규칙 선택, 부모 좌표 동시 이동, R에만 분배, 재정규화 전 drop 기록, gamma 혼합 및 정확 점검 구현.
- 최초 GREEN 시도: job 11819, `10 errors in 45.73s`, 종료 코드 1. seed 9000–9299, eta=0.5에서 모든 규칙을 고를 수 있는 seed가 없어 공통 `prep` fixture에서 실패. 개별 10개 테스트 본문은 실행되지 않음.
- 조건·seed 범위·기대값은 유지한다. 동일 fixture에 실패 사유 계수와 셀 직접 합산을 통한 금지 쌍 선택 독립 대조만 추가해 원인을 확인한다. 진단은 같은 Task 3 pytest 안에서만 수행한다.

## 자체 검토 1: 구현된 범위의 명세 대조

전체 Task 1–8 완료 후의 최종 검토가 아니다. Task 3 선행 fixture 실패 때문에 현재 존재하는 C1–C9와 실행 뼈대만 검토했다.

- C1: 열 0 최상위, 셀 번호와 단일·쌍 사전순 번호, 셀당 조각 36개, int8/int16/int64 형태를 대조. T1의 4개 테스트 통과.
- C2–C3: 뿌리와 사슬·잎 간선 순서, 행별 Dirichlet, 부모 쌍에만 0 삽입, 삽입 전 질량 0.05 이하, 공유 0 먼저 추출, Z 부모 쌍 비복원 균등 추출, 삽입 후 한 번 정규화를 대조. T2의 4개 테스트 통과.
- C4: 0 삽입 후 p0로 pi0/e 계산, 양끝 포함, 목표 거리·사전순 동점 처리, 부모 0→3 순서를 대조. 원인 확인을 위해 동일 seed 범위에서 별도의 셀 직접 합산과 선택 결과를 비교하도록 fixture 진단을 추가함.
- C5: 부모 (0,2), cols=(부모,잎,다음 부모), 세 수준의 부모 값 제외, eT/질량/하위 부모 쌍 조건을 정적 대조. 실행 검증은 fixture 실패로 미도달.
- C6–C9: R/Z 분리, gamma=0의 S1 및 Z 반환, 이동 전 원래 규칙 마스크, 부모 좌표 동시 변경, R 필터, 동일 배분, 재정규화 전 drop, 규정된 혼합식, F 교집합 비교를 정적 대조. 실행 검증은 fixture 실패로 미도달.
- 추가 처리 점검: 확률 클리핑·support epsilon·임의 재시도·규칙 완화 없음. np.divide의 out=0은 목적지가 없는 원천의 분배항에만 쓰며, 해당 원천의 질량은 dropped로 모두 기록한다.
- 진입점과 C10–C20의 전체 호출 경로 점검은 미실시. Task 4–8 미착수.

## 자체 검토 2: 구현된 범위의 수치 경계

- D1.1–D1.3: int8 셀에서 조각 번호를 계산하기 전에 int16으로 승격. 단일 조각의 -1 좌표를 인덱싱에서 제외. T1로 확인.
- D4.2: pi0를 셀마다 36개 조각에 가중합하므로 각 조각의 정의와 일치. 독립 직접 합산 진단으로 900개 선택 결과 일치 확인.
- D5.2: eT 분모는 부모 주변확률, pi0(T)는 세 좌표의 공동 확률이다. 두 값을 혼용하지 않음. 정적 대조만 수행.
- D7.1: 이동 후보는 각 부모 좌표의 (원래 값 + 1,2,3) mod 4를 모두 나열하므로 원래 값을 제외한 정확히 세 값이다. 여러 규칙이 같은 부모를 지목해도 비트 OR로 한 좌표만 바꾼다. 목적지 0개인 행은 drop 합산 후 분배하지 않음. 정적 대조만 수행.
- D8.1: gamma=0에서 h(Z)로 나누지 않음. 혼합 후 별도 재정규화 없음. 정적 대조만 수행.
- D9.2: triple 하위 쌍은 열을 정렬한 뒤 pattern_id를 계산. `extra.any()`와 `array_equal(extra, F & triple_mask)`를 함께 적용. 정적 대조만 수행.
- D9.3: pi0>0 및 pi1>0인 조각에만 floor 적용. 0 판정에 임의 허용 오차를 사용하지 않음. 정적 대조만 수행.
- D13.1/D13.2/D14.1/D18.2 등 Task 5 이후 결과 민감 항목은 미구현으로 미검토.
- move_mass 성능: fixture 실패로 함수 호출에 도달하지 못해 미측정. 1초 목표 충족을 주장하지 않음.

## C 대응표

| ID | 코드 위치 | 비고 |
|---|---|---|
| C1 | dgp/space.py:cell_index:40; pattern_id:45; pattern_probs:52; pattern_counts:59; cells_containing:66 | T1 통과 |
| C2 | dgp/laws.py:TreeTables:15; draw_tree_tables:24; tree_law:35 | T2 통과 |
| C3 | dgp/laws.py:choose_parent_zeros:43; apply_zeros:71 | T2 통과 |
| C4 | dgp/minority.py:Rule:28; select_pairs:50; rule_mask:38; rule_pid:43 | T3 fixture의 선택 단계까지만 실행 |
| C5 | dgp/minority.py:select_triples:73 | 본문 실행 검증 미도달 |
| C6 | dgp/minority.py:regions:111 | 본문 실행 검증 미도달 |
| C7 | dgp/minority.py:move_mass:127 | 본문 실행 검증·시간 측정 미도달 |
| C8 | dgp/minority.py:mix:160 | 본문 실행 검증 미도달 |
| C9 | dgp/minority.py:pairwise_feasible_cells:168; check_minority:173 | 본문 실행 검증 미도달 |
| C10 | 미구현 | Task 4 미착수 |
| C11 | 미구현 | Task 4 미착수; pilot 미실행 |
| C12 | 미구현 | Task 4 미착수 |
| C13 | 미구현 | Task 5 미착수 |
| C14 | 미구현 | Task 5 미착수 |
| C15 | 미구현 | Task 6 미착수 |
| C16 | 미구현 | Task 6 미착수 |
| C17 | 미구현 | Task 6 미착수 |
| C18 | 미구현 | Task 7 미착수 |
| C19 | 미구현 | Task 8 미착수 |
| C20 | experiments/sync_runtime.sh:2; experiments/pytest.sbatch:12; pytest.ini:1 | Task 1 실행 뼈대만. 진입 스크립트와 본실행 sbatch는 미구현 |

## D 대응표

| ID | 코드 위치 | 비고 |
|---|---|---|
| D1.1 | dgp/space.py:<module>:17; cell_index:40 | T1 통과 |
| D1.2 | dgp/space.py:<module>:20; pattern_id:45; cells_containing:66 | T1 통과 |
| D1.3 | dgp/space.py:<module>:31; pattern_probs:52; pattern_counts:59 | T1 통과 |
| D2.1 | dgp/laws.py:draw_tree_tables:24; tree_law:35 | T2 통과 |
| D2.2 | tests/test_minority.py:prep:25 | 테스트 fixture만; 생산 population 미구현 |
| D3.1 | dgp/laws.py:choose_parent_zeros:50 | T2 통과 |
| D3.2 | dgp/laws.py:choose_parent_zeros:61 | T2 통과 |
| D3.3 | tests/test_minority.py:prep:28; dgp/laws.py:choose_parent_zeros:58 | 테스트 fixture RNG만; 생산 population 미구현 |
| D3.4 | dgp/laws.py:apply_zeros:71 | T2 통과 |
| D4.1 | dgp/minority.py:Rule:28; select_pairs:55 | 구현됨 |
| D4.2 | dgp/minority.py:select_pairs:53 | 독립 직접 합산과 대조 |
| D4.3 | dgp/minority.py:select_pairs:67 | 구현됨 |
| D4.4 | dgp/minority.py:<module>:13 | 문서값 동일 |
| D5.1 | dgp/minority.py:select_triples:78 | 실행 검증 미도달 |
| D5.2 | dgp/minority.py:select_triples:79 | 실행 검증 미도달 |
| D5.3 | dgp/minority.py:select_triples:105 | 실행 검증 미도달 |
| D6.1 | dgp/minority.py:regions:111 | 실행 검증 미도달 |
| D7.1 | dgp/minority.py:move_mass:127 | 실행 검증 미도달 |
| D7.2 | 미구현 | Task 4의 재사용 조립 로직 미착수 |
| D8.1 | dgp/minority.py:mix:160 | 실행 검증 미도달 |
| D9.1 | dgp/minority.py:check_minority:195 | 실행 검증 미도달 |
| D9.2 | dgp/minority.py:pairwise_feasible_cells:168; check_minority:177 | 수정된 정의 반영, 실행 검증 미도달 |
| D9.3 | dgp/minority.py:check_minority:191 | 실행 검증 미도달 |
| D10.1 | 미구현 | Task 4 미착수 |
| D10.2 | 미구현 | Task 4 미착수 |
| D10.3 | 미구현 | Task 4 미착수 |
| D11.1 | 미구현 | Task 4 미착수 |
| D12.1 | 미구현 | Task 4 미착수 |
| D13.1 | 미구현 | Task 5 미착수 |
| D13.2 | 미구현 | Task 5 미착수 |
| D14.1 | 미구현 | Task 5 미착수 |
| D14.2 | 미구현 | Task 5 미착수 |
| D15.1 | 미구현 | Task 6 미착수 |
| D16.1 | 미구현 | Task 6 미착수 |
| D17.1 | 미구현 | Task 6 미착수 |
| D18.1 | 미구현 | Task 7 미착수 |
| D18.2 | 미구현 | Task 7 미착수 |
| D18.3 | 미구현 | Task 7 미착수 |
| D18.4 | 미구현 | Task 7 미착수 |
| D19.1 | 미구현 | Task 8 미착수 |
| D20.1 | pytest.ini:1; experiments/sync_runtime.sh:2 | pytest 경로만; scripts/quick 미구현 |

## 신규 파일과 목적

- `dgp/__init__.py`: 빈 패키지 파일.
- `dgp/space.py`: 셀·조각 번호, 조각 확률·개수.
- `dgp/laws.py`: 트리 분포, 부모 쌍 0 선택·삽입.
- `dgp/minority.py`: 규칙 선택·영역·이동·혼합·점검. fixture 이후 함수는 런타임 미검증.
- `tests/test_space.py`, `tests/test_laws.py`, `tests/test_minority.py`: 계획 T1–T3의 4+4+10개 테스트. T3 fixture에는 실패 원인 진단만 추가.
- `pytest.ini`: 계획의 pytest 경로·대상 설정.
- `experiments/sync_runtime.sh`: 코드만 runtime에 동기화. 기존 로그·결과·보고서 보존.
- `experiments/pytest.sbatch`: 계획의 CPU pytest 실행 스크립트. 제출하지 않음.
- `reports/stage1_codex_review.md`: 이 부분 구현·실행·검토·미완료 기록.

설정된 formatter/linter는 발견되지 않았다. shell 파일의 `bash -n`은 통과했다. 패키지 설치, 원문 문서 수정, git 작업은 하지 않았다.

## 기준 문서 SHA-256

```text
1703eb82a2628bc40c7cbba539595922dedbf8cc220d4f70c082f50dd7996ee7  docs/superpowers/specs/2026-10-22-synthetic-dgp-design.md
86e910bda43183b78e6e5941509959d416d1f7fe535f4b9a0def322b79d20003  docs/superpowers/plans/2026-10-22-synthetic-dgp.md
99410f0fdc37a47bcfef94dc0d51415ee7ba125ae85140a24d31e056f4b2ce28  docs/algo_specs/stage1_dgp_spec.md
86ac5778d81dbb81d7c8a09f61c81eff86b65ed656dc1b5f67f8799ebb8084c7  docs/codex_prompts/stage1_dgp_prompt.md
```

## 중단 시점의 확인 결과

- Task 1: 4 passed (job 11807).
- Task 2: 4 passed (job 11812).
- Task 3 최초 시도: 10 fixture errors (job 11819, 45.73초).
- Task 3 동일 범위 독립 대조: 10 fixture errors (job 11826, 48.02초). 900개 `(seed, 수준)`에서 셀 직접 합산과 구현의 선택 결과가 모두 일치했다.
- 300개 seed 모두 `no_pair:frequent`. 0 쌍 선택은 모두 성공했으며, triple 선택·영역·질량 이동·혼합·floor/local-support 점검에는 도달하지 못했다.
- 따라서 현재 결과는 서로 다른 테스트 기준으로 **8 passed + 10 setup errors**다. 42개 전체 테스트를 실행하거나 통과한 결과가 아니다.
- Task 4–8 및 해당 24개 테스트는 미착수. `--quick`, pilot, 개발용/최종용 seed, GPU는 실행하지 않았다.
- 규칙 조건, η, seed 범위, 테스트 기대값을 완화하지 않았다. C1–C9의 부분 구현을 보존하고 여기서 중단한다.

### 직접 합산 진단: 해당 부모의 후보가 없는 seed 수

한 seed가 여러 열·수준에 중복 포함될 수 있다. 아래 수치는 전체 수락률이나 pilot 결과가 아니다.

| 수준 | parent 0 | parent 1 | parent 2 | parent 3 |
|---|---:|---:|---:|---:|
| frequent | 225 | 274 | 268 | 271 |
| mid | 160 | 134 | 107 | 124 |
| rare | 79 | 53 | 48 | 48 |

진단 원본: `/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime/logs/task3_selection_diagnostic.json`. 300개 seed별 거절 사유를 모두 담는다.

### 문서/설계 검토가 필요한 지점

계획 T3 fixture는 seed 9000–9299와 η=0.5에서 모든 수준의 규칙을 고를 수 있는 seed를 찾는다고 가정하지만, 이번 고정된 실행에서 그런 seed가 없었다. frequent는 각 잎 간선에서 pi0>=0.09와 e<=0.2를 동시에 요구하므로 해당 부모 값의 주변확률이 적어도 0.45여야 한다(pi0=e*P0(P=a')). 이 조건을 네 간선이 동시에 충족하는 fixture/생성 설계를 먼저 검토할 필요가 있다.

이는 전체 모집단 생성의 수학적 불가능성이나 다른 η의 수락률을 입증한 결과가 아니다. 예비 실행이나 seed 범위 확대는 수행하지 않았다. 테스트 기대값을 코드에 맞춰 고치거나 조건을 완화하지 않는다.

독립 대조 후 검토 기록 보완: D4.2의 선택 결과 일치가 확인됐다. 1차·2차 검토에서 생산 코드의 조건 변경은 하지 않았고, 발견한 문제는 선행 fixture의 부재다. 전체 진입점 감사·전체 자체 검토·42개 테스트 완료는 여전히 남아 있다.

## srun 원본 출력

아래 출력은 해당 로그 파일의 내용을 생략 없이 옮겼다. 여러 작업별 실행이며 단일 전체 42개 테스트 실행이 아니다.

<details>
<summary>task1_red.log</summary>

```text
srun: job 11795 queued and waiting for resources
srun: job 11795 has been allocated resources
============================= test session starts ==============================
platform linux -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0 -- /mnt/data1/calvinkim1027/miniconda3/envs/tabular_foundation/bin/python
cachedir: .pytest_cache
rootdir: /mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
configfile: pytest.ini
plugins: anyio-4.13.0
collecting ... collected 0 items / 1 error

==================================== ERRORS ====================================
_____________________ ERROR collecting tests/test_space.py _____________________
ImportError while importing test module '/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime/tests/test_space.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../miniconda3/envs/tabular_foundation/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_space.py:3: in <module>
    from dgp.space import (
E   ModuleNotFoundError: No module named 'dgp.space'
=========================== short test summary info ============================
ERROR tests/test_space.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 54.73s ===============================
srun: error: brl0: task 0: Exited with exit code 2

```
</details>

<details>
<summary>task1_green.log</summary>

```text
srun: job 11803 queued and waiting for resources
srun: job 11803 has been allocated resources
srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
slurmstepd: error: *** STEP 11803.0 ON brl0 CANCELLED AT 2026-10-23T01:27:33 ***
srun: error: brl0: task 0: Terminated

```
</details>

<details>
<summary>task1_green_brl3.log</summary>

```text
srun: job 11807 queued and waiting for resources
srun: job 11807 has been allocated resources
============================= test session starts ==============================
platform linux -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0 -- /mnt/data1/calvinkim1027/miniconda3/envs/tabular_foundation/bin/python
cachedir: .pytest_cache
rootdir: /mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
configfile: pytest.ini
plugins: anyio-4.13.0
collecting ... collected 4 items

tests/test_space.py::test_cell_index_order PASSED                        [ 25%]
tests/test_space.py::test_pattern_ids PASSED                             [ 50%]
tests/test_space.py::test_cell_patterns_and_probs PASSED                 [ 75%]
tests/test_space.py::test_pattern_counts_matches_bruteforce PASSED       [100%]

============================== 4 passed in 25.27s ==============================

```
</details>

<details>
<summary>task2_red.log</summary>

```text
srun: job 11811 queued and waiting for resources
srun: job 11811 has been allocated resources
============================= test session starts ==============================
platform linux -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0 -- /mnt/data1/calvinkim1027/miniconda3/envs/tabular_foundation/bin/python
cachedir: .pytest_cache
rootdir: /mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
configfile: pytest.ini
plugins: anyio-4.13.0
collecting ... collected 0 items / 1 error

==================================== ERRORS ====================================
_____________________ ERROR collecting tests/test_laws.py ______________________
ImportError while importing test module '/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime/tests/test_laws.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../miniconda3/envs/tabular_foundation/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_laws.py:4: in <module>
    from dgp.laws import apply_zeros, choose_parent_zeros, draw_tree_tables, tree_law
E   ModuleNotFoundError: No module named 'dgp.laws'
=========================== short test summary info ============================
ERROR tests/test_laws.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 11.26s ===============================
srun: error: brl3: task 0: Exited with exit code 2

```
</details>

<details>
<summary>task2_green.log</summary>

```text
srun: job 11812 queued and waiting for resources
srun: job 11812 has been allocated resources
============================= test session starts ==============================
platform linux -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0 -- /mnt/data1/calvinkim1027/miniconda3/envs/tabular_foundation/bin/python
cachedir: .pytest_cache
rootdir: /mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
configfile: pytest.ini
plugins: anyio-4.13.0
collecting ... collected 4 items

tests/test_laws.py::test_tree_law_normalized_positive PASSED             [ 25%]
tests/test_laws.py::test_parent_zeros_structure PASSED                   [ 50%]
tests/test_laws.py::test_leaf_conditional_independence_after_zeros PASSED [ 75%]
tests/test_laws.py::test_choose_parent_zeros_none_when_impossible PASSED [100%]

============================== 4 passed in 17.07s ==============================

```
</details>

<details>
<summary>task3_red.log</summary>

```text
srun: job 11814 queued and waiting for resources
srun: job 11814 has been allocated resources
============================= test session starts ==============================
platform linux -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0 -- /mnt/data1/calvinkim1027/miniconda3/envs/tabular_foundation/bin/python
cachedir: .pytest_cache
rootdir: /mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
configfile: pytest.ini
plugins: anyio-4.13.0
collecting ... collected 0 items / 1 error

==================================== ERRORS ====================================
___________________ ERROR collecting tests/test_minority.py ____________________
ImportError while importing test module '/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime/tests/test_minority.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../miniconda3/envs/tabular_foundation/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_minority.py:8: in <module>
    from dgp.minority import (
E   ModuleNotFoundError: No module named 'dgp.minority'
=========================== short test summary info ============================
ERROR tests/test_minority.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 20.06s ===============================
srun: error: brl3: task 0: Exited with exit code 2

```
</details>

<details>
<summary>task3_green.log</summary>

```text
srun: job 11819 queued and waiting for resources
srun: job 11819 has been allocated resources
============================= test session starts ==============================
platform linux -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0 -- /mnt/data1/calvinkim1027/miniconda3/envs/tabular_foundation/bin/python
cachedir: .pytest_cache
rootdir: /mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
configfile: pytest.ini
plugins: anyio-4.13.0
collecting ... collected 10 items

tests/test_minority.py::test_select_pairs_rules ERROR                    [ 10%]
tests/test_minority.py::test_select_triples_rules ERROR                  [ 20%]
tests/test_minority.py::test_regions ERROR                               [ 30%]
tests/test_minority.py::test_move_mass_supports ERROR                    [ 40%]
tests/test_minority.py::test_loss_fractions_exact_pairs_only ERROR       [ 50%]
tests/test_minority.py::test_triple_loss_bounded ERROR                   [ 60%]
tests/test_minority.py::test_mix_z_mass ERROR                            [ 70%]
tests/test_minority.py::test_floor_and_checks_all_envs ERROR             [ 80%]
tests/test_minority.py::test_local_support_violation_is_exactly_triples ERROR [ 90%]
tests/test_minority.py::test_check_detects_support_violation ERROR       [100%]

==================================== ERRORS ====================================
__________________ ERROR at setup of test_select_pairs_rules ___________________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            if any(value is None for value in pairs.values()):
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
>       pytest.fail("계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음

tests/test_minority.py:41: Failed
_________________ ERROR at setup of test_select_triples_rules __________________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            if any(value is None for value in pairs.values()):
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
>       pytest.fail("계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음

tests/test_minority.py:41: Failed
________________________ ERROR at setup of test_regions ________________________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            if any(value is None for value in pairs.values()):
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
>       pytest.fail("계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음

tests/test_minority.py:41: Failed
__________________ ERROR at setup of test_move_mass_supports ___________________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            if any(value is None for value in pairs.values()):
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
>       pytest.fail("계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음

tests/test_minority.py:41: Failed
____________ ERROR at setup of test_loss_fractions_exact_pairs_only ____________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            if any(value is None for value in pairs.values()):
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
>       pytest.fail("계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음

tests/test_minority.py:41: Failed
__________________ ERROR at setup of test_triple_loss_bounded __________________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            if any(value is None for value in pairs.values()):
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
>       pytest.fail("계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음

tests/test_minority.py:41: Failed
______________________ ERROR at setup of test_mix_z_mass _______________________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            if any(value is None for value in pairs.values()):
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
>       pytest.fail("계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음

tests/test_minority.py:41: Failed
_______________ ERROR at setup of test_floor_and_checks_all_envs _______________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            if any(value is None for value in pairs.values()):
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
>       pytest.fail("계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음

tests/test_minority.py:41: Failed
______ ERROR at setup of test_local_support_violation_is_exactly_triples _______

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            if any(value is None for value in pairs.values()):
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
>       pytest.fail("계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음

tests/test_minority.py:41: Failed
____________ ERROR at setup of test_check_detects_support_violation ____________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            if any(value is None for value in pairs.values()):
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
>       pytest.fail("계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음

tests/test_minority.py:41: Failed
=========================== short test summary info ============================
ERROR tests/test_minority.py::test_select_pairs_rules - Failed: 계획 T3의 see...
ERROR tests/test_minority.py::test_select_triples_rules - Failed: 계획 T3의 s...
ERROR tests/test_minority.py::test_regions - Failed: 계획 T3의 seed 9000–9299...
ERROR tests/test_minority.py::test_move_mass_supports - Failed: 계획 T3의 see...
ERROR tests/test_minority.py::test_loss_fractions_exact_pairs_only - Failed: ...
ERROR tests/test_minority.py::test_triple_loss_bounded - Failed: 계획 T3의 se...
ERROR tests/test_minority.py::test_mix_z_mass - Failed: 계획 T3의 seed 9000–9...
ERROR tests/test_minority.py::test_floor_and_checks_all_envs - Failed: 계획 T...
ERROR tests/test_minority.py::test_local_support_violation_is_exactly_triples
ERROR tests/test_minority.py::test_check_detects_support_violation - Failed: ...
============================= 10 errors in 45.73s ==============================
srun: error: brl3: task 0: Exited with exit code 1

```
</details>

<details>
<summary>task3_diagnostic.log</summary>

```text
srun: job 11826 queued and waiting for resources
srun: job 11826 has been allocated resources
============================= test session starts ==============================
platform linux -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0 -- /mnt/data1/calvinkim1027/miniconda3/envs/tabular_foundation/bin/python
cachedir: .pytest_cache
rootdir: /mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
configfile: pytest.ini
plugins: anyio-4.13.0
collecting ... collected 10 items

tests/test_minority.py::test_select_pairs_rules ERROR                    [ 10%]
tests/test_minority.py::test_select_triples_rules ERROR                  [ 20%]
tests/test_minority.py::test_regions ERROR                               [ 30%]
tests/test_minority.py::test_move_mass_supports ERROR                    [ 40%]
tests/test_minority.py::test_loss_fractions_exact_pairs_only ERROR       [ 50%]
tests/test_minority.py::test_triple_loss_bounded ERROR                   [ 60%]
tests/test_minority.py::test_mix_z_mass ERROR                            [ 70%]
tests/test_minority.py::test_floor_and_checks_all_envs ERROR             [ 80%]
tests/test_minority.py::test_local_support_violation_is_exactly_triples ERROR [ 90%]
tests/test_minority.py::test_check_detects_support_violation ERROR       [100%]

==================================== ERRORS ====================================
__________________ ERROR at setup of test_select_pairs_rules ___________________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        reasons = Counter()
        skipped = []
        reference_checks = 0
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                reasons["no_zeros"] += 1
                skipped.append((s, "no_zeros"))
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            # 실패 진단: bincount와 독립적인 셀 직접 합산으로 C4 결정을 대조.
            # 원래 fixture 범위·분포·선택 조건·성공 판정은 유지한다.
            for level, (target, lo, hi) in FREQ_LEVELS.items():
                reference = []
                for parent in range(4):
                    candidates = []
                    for ap in range(4):
                        parent_mask = CELLS[:, parent] == ap
                        parent_mass = p0[parent_mask].sum()
                        if parent_mass == 0:
                            continue
                        for a in range(4):
                            mass = p0[parent_mask & (CELLS[:, parent + 4] == a)].sum()
                            if mass / parent_mass <= E_MAX and lo <= mass <= hi:
                                candidates.append((abs(mass - target), (ap, a)))
                    if not candidates:
                        reasons[f"missing:{level}:parent{parent}"] += 1
                        reference.append(None)
                    else:
                        reference.append(min(candidates)[1])
                expected = None if None in reference else reference
                actual = None if pairs[level] is None else [r.vals for r in pairs[level]]
                assert actual == expected, (s, level, actual, expected)
                reference_checks += 1
            if any(value is None for value in pairs.values()):
                failed = next(level for level in FREQ_LEVELS if pairs[level] is None)
                reasons[f"no_pair:{failed}"] += 1
                skipped.append((s, f"no_pair:{failed}"))
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                reasons["no_triple"] += 1
                skipped.append((s, "no_triple"))
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
        diagnostic = {"eta": 0.5, "start_seed": 9000, "stop_seed_exclusive": 9300,
                      "reference_checks": reference_checks,
                      "reasons": dict(reasons), "skipped": skipped}
        diagnostic_path = Path("logs/task3_selection_diagnostic.json")
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
>       pytest.fail(f"계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {dict(reasons)}")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {'missing:frequent:parent0': 225, 'missing:frequent:parent1': 274, 'missing:frequent:parent2': 268, 'missing:frequent:parent3': 271, 'missing:rare:parent1': 53, 'no_pair:frequent': 300, 'missing:mid:parent0': 160, 'missing:mid:parent1': 134, 'missing:mid:parent3': 124, 'missing:rare:parent2': 48, 'missing:mid:parent2': 107, 'missing:rare:parent3': 48, 'missing:rare:parent0': 79}

tests/test_minority.py:84: Failed
_________________ ERROR at setup of test_select_triples_rules __________________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        reasons = Counter()
        skipped = []
        reference_checks = 0
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                reasons["no_zeros"] += 1
                skipped.append((s, "no_zeros"))
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            # 실패 진단: bincount와 독립적인 셀 직접 합산으로 C4 결정을 대조.
            # 원래 fixture 범위·분포·선택 조건·성공 판정은 유지한다.
            for level, (target, lo, hi) in FREQ_LEVELS.items():
                reference = []
                for parent in range(4):
                    candidates = []
                    for ap in range(4):
                        parent_mask = CELLS[:, parent] == ap
                        parent_mass = p0[parent_mask].sum()
                        if parent_mass == 0:
                            continue
                        for a in range(4):
                            mass = p0[parent_mask & (CELLS[:, parent + 4] == a)].sum()
                            if mass / parent_mass <= E_MAX and lo <= mass <= hi:
                                candidates.append((abs(mass - target), (ap, a)))
                    if not candidates:
                        reasons[f"missing:{level}:parent{parent}"] += 1
                        reference.append(None)
                    else:
                        reference.append(min(candidates)[1])
                expected = None if None in reference else reference
                actual = None if pairs[level] is None else [r.vals for r in pairs[level]]
                assert actual == expected, (s, level, actual, expected)
                reference_checks += 1
            if any(value is None for value in pairs.values()):
                failed = next(level for level in FREQ_LEVELS if pairs[level] is None)
                reasons[f"no_pair:{failed}"] += 1
                skipped.append((s, f"no_pair:{failed}"))
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                reasons["no_triple"] += 1
                skipped.append((s, "no_triple"))
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
        diagnostic = {"eta": 0.5, "start_seed": 9000, "stop_seed_exclusive": 9300,
                      "reference_checks": reference_checks,
                      "reasons": dict(reasons), "skipped": skipped}
        diagnostic_path = Path("logs/task3_selection_diagnostic.json")
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
>       pytest.fail(f"계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {dict(reasons)}")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {'missing:frequent:parent0': 225, 'missing:frequent:parent1': 274, 'missing:frequent:parent2': 268, 'missing:frequent:parent3': 271, 'missing:rare:parent1': 53, 'no_pair:frequent': 300, 'missing:mid:parent0': 160, 'missing:mid:parent1': 134, 'missing:mid:parent3': 124, 'missing:rare:parent2': 48, 'missing:mid:parent2': 107, 'missing:rare:parent3': 48, 'missing:rare:parent0': 79}

tests/test_minority.py:84: Failed
________________________ ERROR at setup of test_regions ________________________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        reasons = Counter()
        skipped = []
        reference_checks = 0
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                reasons["no_zeros"] += 1
                skipped.append((s, "no_zeros"))
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            # 실패 진단: bincount와 독립적인 셀 직접 합산으로 C4 결정을 대조.
            # 원래 fixture 범위·분포·선택 조건·성공 판정은 유지한다.
            for level, (target, lo, hi) in FREQ_LEVELS.items():
                reference = []
                for parent in range(4):
                    candidates = []
                    for ap in range(4):
                        parent_mask = CELLS[:, parent] == ap
                        parent_mass = p0[parent_mask].sum()
                        if parent_mass == 0:
                            continue
                        for a in range(4):
                            mass = p0[parent_mask & (CELLS[:, parent + 4] == a)].sum()
                            if mass / parent_mass <= E_MAX and lo <= mass <= hi:
                                candidates.append((abs(mass - target), (ap, a)))
                    if not candidates:
                        reasons[f"missing:{level}:parent{parent}"] += 1
                        reference.append(None)
                    else:
                        reference.append(min(candidates)[1])
                expected = None if None in reference else reference
                actual = None if pairs[level] is None else [r.vals for r in pairs[level]]
                assert actual == expected, (s, level, actual, expected)
                reference_checks += 1
            if any(value is None for value in pairs.values()):
                failed = next(level for level in FREQ_LEVELS if pairs[level] is None)
                reasons[f"no_pair:{failed}"] += 1
                skipped.append((s, f"no_pair:{failed}"))
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                reasons["no_triple"] += 1
                skipped.append((s, "no_triple"))
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
        diagnostic = {"eta": 0.5, "start_seed": 9000, "stop_seed_exclusive": 9300,
                      "reference_checks": reference_checks,
                      "reasons": dict(reasons), "skipped": skipped}
        diagnostic_path = Path("logs/task3_selection_diagnostic.json")
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
>       pytest.fail(f"계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {dict(reasons)}")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {'missing:frequent:parent0': 225, 'missing:frequent:parent1': 274, 'missing:frequent:parent2': 268, 'missing:frequent:parent3': 271, 'missing:rare:parent1': 53, 'no_pair:frequent': 300, 'missing:mid:parent0': 160, 'missing:mid:parent1': 134, 'missing:mid:parent3': 124, 'missing:rare:parent2': 48, 'missing:mid:parent2': 107, 'missing:rare:parent3': 48, 'missing:rare:parent0': 79}

tests/test_minority.py:84: Failed
__________________ ERROR at setup of test_move_mass_supports ___________________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        reasons = Counter()
        skipped = []
        reference_checks = 0
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                reasons["no_zeros"] += 1
                skipped.append((s, "no_zeros"))
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            # 실패 진단: bincount와 독립적인 셀 직접 합산으로 C4 결정을 대조.
            # 원래 fixture 범위·분포·선택 조건·성공 판정은 유지한다.
            for level, (target, lo, hi) in FREQ_LEVELS.items():
                reference = []
                for parent in range(4):
                    candidates = []
                    for ap in range(4):
                        parent_mask = CELLS[:, parent] == ap
                        parent_mass = p0[parent_mask].sum()
                        if parent_mass == 0:
                            continue
                        for a in range(4):
                            mass = p0[parent_mask & (CELLS[:, parent + 4] == a)].sum()
                            if mass / parent_mass <= E_MAX and lo <= mass <= hi:
                                candidates.append((abs(mass - target), (ap, a)))
                    if not candidates:
                        reasons[f"missing:{level}:parent{parent}"] += 1
                        reference.append(None)
                    else:
                        reference.append(min(candidates)[1])
                expected = None if None in reference else reference
                actual = None if pairs[level] is None else [r.vals for r in pairs[level]]
                assert actual == expected, (s, level, actual, expected)
                reference_checks += 1
            if any(value is None for value in pairs.values()):
                failed = next(level for level in FREQ_LEVELS if pairs[level] is None)
                reasons[f"no_pair:{failed}"] += 1
                skipped.append((s, f"no_pair:{failed}"))
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                reasons["no_triple"] += 1
                skipped.append((s, "no_triple"))
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
        diagnostic = {"eta": 0.5, "start_seed": 9000, "stop_seed_exclusive": 9300,
                      "reference_checks": reference_checks,
                      "reasons": dict(reasons), "skipped": skipped}
        diagnostic_path = Path("logs/task3_selection_diagnostic.json")
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
>       pytest.fail(f"계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {dict(reasons)}")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {'missing:frequent:parent0': 225, 'missing:frequent:parent1': 274, 'missing:frequent:parent2': 268, 'missing:frequent:parent3': 271, 'missing:rare:parent1': 53, 'no_pair:frequent': 300, 'missing:mid:parent0': 160, 'missing:mid:parent1': 134, 'missing:mid:parent3': 124, 'missing:rare:parent2': 48, 'missing:mid:parent2': 107, 'missing:rare:parent3': 48, 'missing:rare:parent0': 79}

tests/test_minority.py:84: Failed
____________ ERROR at setup of test_loss_fractions_exact_pairs_only ____________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        reasons = Counter()
        skipped = []
        reference_checks = 0
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                reasons["no_zeros"] += 1
                skipped.append((s, "no_zeros"))
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            # 실패 진단: bincount와 독립적인 셀 직접 합산으로 C4 결정을 대조.
            # 원래 fixture 범위·분포·선택 조건·성공 판정은 유지한다.
            for level, (target, lo, hi) in FREQ_LEVELS.items():
                reference = []
                for parent in range(4):
                    candidates = []
                    for ap in range(4):
                        parent_mask = CELLS[:, parent] == ap
                        parent_mass = p0[parent_mask].sum()
                        if parent_mass == 0:
                            continue
                        for a in range(4):
                            mass = p0[parent_mask & (CELLS[:, parent + 4] == a)].sum()
                            if mass / parent_mass <= E_MAX and lo <= mass <= hi:
                                candidates.append((abs(mass - target), (ap, a)))
                    if not candidates:
                        reasons[f"missing:{level}:parent{parent}"] += 1
                        reference.append(None)
                    else:
                        reference.append(min(candidates)[1])
                expected = None if None in reference else reference
                actual = None if pairs[level] is None else [r.vals for r in pairs[level]]
                assert actual == expected, (s, level, actual, expected)
                reference_checks += 1
            if any(value is None for value in pairs.values()):
                failed = next(level for level in FREQ_LEVELS if pairs[level] is None)
                reasons[f"no_pair:{failed}"] += 1
                skipped.append((s, f"no_pair:{failed}"))
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                reasons["no_triple"] += 1
                skipped.append((s, "no_triple"))
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
        diagnostic = {"eta": 0.5, "start_seed": 9000, "stop_seed_exclusive": 9300,
                      "reference_checks": reference_checks,
                      "reasons": dict(reasons), "skipped": skipped}
        diagnostic_path = Path("logs/task3_selection_diagnostic.json")
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
>       pytest.fail(f"계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {dict(reasons)}")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {'missing:frequent:parent0': 225, 'missing:frequent:parent1': 274, 'missing:frequent:parent2': 268, 'missing:frequent:parent3': 271, 'missing:rare:parent1': 53, 'no_pair:frequent': 300, 'missing:mid:parent0': 160, 'missing:mid:parent1': 134, 'missing:mid:parent3': 124, 'missing:rare:parent2': 48, 'missing:mid:parent2': 107, 'missing:rare:parent3': 48, 'missing:rare:parent0': 79}

tests/test_minority.py:84: Failed
__________________ ERROR at setup of test_triple_loss_bounded __________________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        reasons = Counter()
        skipped = []
        reference_checks = 0
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                reasons["no_zeros"] += 1
                skipped.append((s, "no_zeros"))
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            # 실패 진단: bincount와 독립적인 셀 직접 합산으로 C4 결정을 대조.
            # 원래 fixture 범위·분포·선택 조건·성공 판정은 유지한다.
            for level, (target, lo, hi) in FREQ_LEVELS.items():
                reference = []
                for parent in range(4):
                    candidates = []
                    for ap in range(4):
                        parent_mask = CELLS[:, parent] == ap
                        parent_mass = p0[parent_mask].sum()
                        if parent_mass == 0:
                            continue
                        for a in range(4):
                            mass = p0[parent_mask & (CELLS[:, parent + 4] == a)].sum()
                            if mass / parent_mass <= E_MAX and lo <= mass <= hi:
                                candidates.append((abs(mass - target), (ap, a)))
                    if not candidates:
                        reasons[f"missing:{level}:parent{parent}"] += 1
                        reference.append(None)
                    else:
                        reference.append(min(candidates)[1])
                expected = None if None in reference else reference
                actual = None if pairs[level] is None else [r.vals for r in pairs[level]]
                assert actual == expected, (s, level, actual, expected)
                reference_checks += 1
            if any(value is None for value in pairs.values()):
                failed = next(level for level in FREQ_LEVELS if pairs[level] is None)
                reasons[f"no_pair:{failed}"] += 1
                skipped.append((s, f"no_pair:{failed}"))
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                reasons["no_triple"] += 1
                skipped.append((s, "no_triple"))
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
        diagnostic = {"eta": 0.5, "start_seed": 9000, "stop_seed_exclusive": 9300,
                      "reference_checks": reference_checks,
                      "reasons": dict(reasons), "skipped": skipped}
        diagnostic_path = Path("logs/task3_selection_diagnostic.json")
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
>       pytest.fail(f"계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {dict(reasons)}")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {'missing:frequent:parent0': 225, 'missing:frequent:parent1': 274, 'missing:frequent:parent2': 268, 'missing:frequent:parent3': 271, 'missing:rare:parent1': 53, 'no_pair:frequent': 300, 'missing:mid:parent0': 160, 'missing:mid:parent1': 134, 'missing:mid:parent3': 124, 'missing:rare:parent2': 48, 'missing:mid:parent2': 107, 'missing:rare:parent3': 48, 'missing:rare:parent0': 79}

tests/test_minority.py:84: Failed
______________________ ERROR at setup of test_mix_z_mass _______________________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        reasons = Counter()
        skipped = []
        reference_checks = 0
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                reasons["no_zeros"] += 1
                skipped.append((s, "no_zeros"))
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            # 실패 진단: bincount와 독립적인 셀 직접 합산으로 C4 결정을 대조.
            # 원래 fixture 범위·분포·선택 조건·성공 판정은 유지한다.
            for level, (target, lo, hi) in FREQ_LEVELS.items():
                reference = []
                for parent in range(4):
                    candidates = []
                    for ap in range(4):
                        parent_mask = CELLS[:, parent] == ap
                        parent_mass = p0[parent_mask].sum()
                        if parent_mass == 0:
                            continue
                        for a in range(4):
                            mass = p0[parent_mask & (CELLS[:, parent + 4] == a)].sum()
                            if mass / parent_mass <= E_MAX and lo <= mass <= hi:
                                candidates.append((abs(mass - target), (ap, a)))
                    if not candidates:
                        reasons[f"missing:{level}:parent{parent}"] += 1
                        reference.append(None)
                    else:
                        reference.append(min(candidates)[1])
                expected = None if None in reference else reference
                actual = None if pairs[level] is None else [r.vals for r in pairs[level]]
                assert actual == expected, (s, level, actual, expected)
                reference_checks += 1
            if any(value is None for value in pairs.values()):
                failed = next(level for level in FREQ_LEVELS if pairs[level] is None)
                reasons[f"no_pair:{failed}"] += 1
                skipped.append((s, f"no_pair:{failed}"))
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                reasons["no_triple"] += 1
                skipped.append((s, "no_triple"))
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
        diagnostic = {"eta": 0.5, "start_seed": 9000, "stop_seed_exclusive": 9300,
                      "reference_checks": reference_checks,
                      "reasons": dict(reasons), "skipped": skipped}
        diagnostic_path = Path("logs/task3_selection_diagnostic.json")
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
>       pytest.fail(f"계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {dict(reasons)}")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {'missing:frequent:parent0': 225, 'missing:frequent:parent1': 274, 'missing:frequent:parent2': 268, 'missing:frequent:parent3': 271, 'missing:rare:parent1': 53, 'no_pair:frequent': 300, 'missing:mid:parent0': 160, 'missing:mid:parent1': 134, 'missing:mid:parent3': 124, 'missing:rare:parent2': 48, 'missing:mid:parent2': 107, 'missing:rare:parent3': 48, 'missing:rare:parent0': 79}

tests/test_minority.py:84: Failed
_______________ ERROR at setup of test_floor_and_checks_all_envs _______________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        reasons = Counter()
        skipped = []
        reference_checks = 0
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                reasons["no_zeros"] += 1
                skipped.append((s, "no_zeros"))
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            # 실패 진단: bincount와 독립적인 셀 직접 합산으로 C4 결정을 대조.
            # 원래 fixture 범위·분포·선택 조건·성공 판정은 유지한다.
            for level, (target, lo, hi) in FREQ_LEVELS.items():
                reference = []
                for parent in range(4):
                    candidates = []
                    for ap in range(4):
                        parent_mask = CELLS[:, parent] == ap
                        parent_mass = p0[parent_mask].sum()
                        if parent_mass == 0:
                            continue
                        for a in range(4):
                            mass = p0[parent_mask & (CELLS[:, parent + 4] == a)].sum()
                            if mass / parent_mass <= E_MAX and lo <= mass <= hi:
                                candidates.append((abs(mass - target), (ap, a)))
                    if not candidates:
                        reasons[f"missing:{level}:parent{parent}"] += 1
                        reference.append(None)
                    else:
                        reference.append(min(candidates)[1])
                expected = None if None in reference else reference
                actual = None if pairs[level] is None else [r.vals for r in pairs[level]]
                assert actual == expected, (s, level, actual, expected)
                reference_checks += 1
            if any(value is None for value in pairs.values()):
                failed = next(level for level in FREQ_LEVELS if pairs[level] is None)
                reasons[f"no_pair:{failed}"] += 1
                skipped.append((s, f"no_pair:{failed}"))
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                reasons["no_triple"] += 1
                skipped.append((s, "no_triple"))
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
        diagnostic = {"eta": 0.5, "start_seed": 9000, "stop_seed_exclusive": 9300,
                      "reference_checks": reference_checks,
                      "reasons": dict(reasons), "skipped": skipped}
        diagnostic_path = Path("logs/task3_selection_diagnostic.json")
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
>       pytest.fail(f"계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {dict(reasons)}")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {'missing:frequent:parent0': 225, 'missing:frequent:parent1': 274, 'missing:frequent:parent2': 268, 'missing:frequent:parent3': 271, 'missing:rare:parent1': 53, 'no_pair:frequent': 300, 'missing:mid:parent0': 160, 'missing:mid:parent1': 134, 'missing:mid:parent3': 124, 'missing:rare:parent2': 48, 'missing:mid:parent2': 107, 'missing:rare:parent3': 48, 'missing:rare:parent0': 79}

tests/test_minority.py:84: Failed
______ ERROR at setup of test_local_support_violation_is_exactly_triples _______

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        reasons = Counter()
        skipped = []
        reference_checks = 0
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                reasons["no_zeros"] += 1
                skipped.append((s, "no_zeros"))
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            # 실패 진단: bincount와 독립적인 셀 직접 합산으로 C4 결정을 대조.
            # 원래 fixture 범위·분포·선택 조건·성공 판정은 유지한다.
            for level, (target, lo, hi) in FREQ_LEVELS.items():
                reference = []
                for parent in range(4):
                    candidates = []
                    for ap in range(4):
                        parent_mask = CELLS[:, parent] == ap
                        parent_mass = p0[parent_mask].sum()
                        if parent_mass == 0:
                            continue
                        for a in range(4):
                            mass = p0[parent_mask & (CELLS[:, parent + 4] == a)].sum()
                            if mass / parent_mass <= E_MAX and lo <= mass <= hi:
                                candidates.append((abs(mass - target), (ap, a)))
                    if not candidates:
                        reasons[f"missing:{level}:parent{parent}"] += 1
                        reference.append(None)
                    else:
                        reference.append(min(candidates)[1])
                expected = None if None in reference else reference
                actual = None if pairs[level] is None else [r.vals for r in pairs[level]]
                assert actual == expected, (s, level, actual, expected)
                reference_checks += 1
            if any(value is None for value in pairs.values()):
                failed = next(level for level in FREQ_LEVELS if pairs[level] is None)
                reasons[f"no_pair:{failed}"] += 1
                skipped.append((s, f"no_pair:{failed}"))
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                reasons["no_triple"] += 1
                skipped.append((s, "no_triple"))
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
        diagnostic = {"eta": 0.5, "start_seed": 9000, "stop_seed_exclusive": 9300,
                      "reference_checks": reference_checks,
                      "reasons": dict(reasons), "skipped": skipped}
        diagnostic_path = Path("logs/task3_selection_diagnostic.json")
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
>       pytest.fail(f"계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {dict(reasons)}")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {'missing:frequent:parent0': 225, 'missing:frequent:parent1': 274, 'missing:frequent:parent2': 268, 'missing:frequent:parent3': 271, 'missing:rare:parent1': 53, 'no_pair:frequent': 300, 'missing:mid:parent0': 160, 'missing:mid:parent1': 134, 'missing:mid:parent3': 124, 'missing:rare:parent2': 48, 'missing:mid:parent2': 107, 'missing:rare:parent3': 48, 'missing:rare:parent0': 79}

tests/test_minority.py:84: Failed
____________ ERROR at setup of test_check_detects_support_violation ____________

    @pytest.fixture(scope="module")
    def prep():
        # 계획 T3: seed 9000–9299, eta=0.5. 후보 조건을 바꾸지 않는다.
        reasons = Counter()
        skipped = []
        reference_checks = 0
        for s in range(9000, 9300):
            t = draw_tree_tables(np.random.default_rng(np.random.SeedSequence([s, 0])), 0.5)
            p_tree = tree_law(t)
            zeros = choose_parent_zeros(
                p_tree, np.random.default_rng(np.random.SeedSequence([s, 2]))
            )
            if zeros is None:
                reasons["no_zeros"] += 1
                skipped.append((s, "no_zeros"))
                continue
            shared, major = zeros
            p0 = apply_zeros(p_tree, shared + major)
            pairs = {level: select_pairs(p0, level) for level in FREQ_LEVELS}
            # 실패 진단: bincount와 독립적인 셀 직접 합산으로 C4 결정을 대조.
            # 원래 fixture 범위·분포·선택 조건·성공 판정은 유지한다.
            for level, (target, lo, hi) in FREQ_LEVELS.items():
                reference = []
                for parent in range(4):
                    candidates = []
                    for ap in range(4):
                        parent_mask = CELLS[:, parent] == ap
                        parent_mass = p0[parent_mask].sum()
                        if parent_mass == 0:
                            continue
                        for a in range(4):
                            mass = p0[parent_mask & (CELLS[:, parent + 4] == a)].sum()
                            if mass / parent_mass <= E_MAX and lo <= mass <= hi:
                                candidates.append((abs(mass - target), (ap, a)))
                    if not candidates:
                        reasons[f"missing:{level}:parent{parent}"] += 1
                        reference.append(None)
                    else:
                        reference.append(min(candidates)[1])
                expected = None if None in reference else reference
                actual = None if pairs[level] is None else [r.vals for r in pairs[level]]
                assert actual == expected, (s, level, actual, expected)
                reference_checks += 1
            if any(value is None for value in pairs.values()):
                failed = next(level for level in FREQ_LEVELS if pairs[level] is None)
                reasons[f"no_pair:{failed}"] += 1
                skipped.append((s, f"no_pair:{failed}"))
                continue
            triples = select_triples(p0, pairs)
            if triples is None:
                reasons["no_triple"] += 1
                skipped.append((s, "no_triple"))
                continue
            h = tree_law(draw_tree_tables(
                np.random.default_rng(np.random.SeedSequence([s, 1])), 0.5
            ))
            return SimpleNamespace(
                p0=p0, S0=p0 > 0, h=h, shared=shared, major=major,
                pairs=pairs, triples=triples,
            )
        diagnostic = {"eta": 0.5, "start_seed": 9000, "stop_seed_exclusive": 9300,
                      "reference_checks": reference_checks,
                      "reasons": dict(reasons), "skipped": skipped}
        diagnostic_path = Path("logs/task3_selection_diagnostic.json")
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
>       pytest.fail(f"계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {dict(reasons)}")
E       Failed: 계획 T3의 seed 9000–9299에서 규칙을 고를 수 없음: {'missing:frequent:parent0': 225, 'missing:frequent:parent1': 274, 'missing:frequent:parent2': 268, 'missing:frequent:parent3': 271, 'missing:rare:parent1': 53, 'no_pair:frequent': 300, 'missing:mid:parent0': 160, 'missing:mid:parent1': 134, 'missing:mid:parent3': 124, 'missing:rare:parent2': 48, 'missing:mid:parent2': 107, 'missing:rare:parent3': 48, 'missing:rare:parent0': 79}

tests/test_minority.py:84: Failed
=========================== short test summary info ============================
ERROR tests/test_minority.py::test_select_pairs_rules - Failed: 계획 T3의 see...
ERROR tests/test_minority.py::test_select_triples_rules - Failed: 계획 T3의 s...
ERROR tests/test_minority.py::test_regions - Failed: 계획 T3의 seed 9000–9299...
ERROR tests/test_minority.py::test_move_mass_supports - Failed: 계획 T3의 see...
ERROR tests/test_minority.py::test_loss_fractions_exact_pairs_only - Failed: ...
ERROR tests/test_minority.py::test_triple_loss_bounded - Failed: 계획 T3의 se...
ERROR tests/test_minority.py::test_mix_z_mass - Failed: 계획 T3의 seed 9000–9...
ERROR tests/test_minority.py::test_floor_and_checks_all_envs - Failed: 계획 T...
ERROR tests/test_minority.py::test_local_support_violation_is_exactly_triples
ERROR tests/test_minority.py::test_check_detects_support_violation - Failed: ...
============================= 10 errors in 48.02s ==============================
srun: error: brl3: task 0: Exited with exit code 1

```
</details>
