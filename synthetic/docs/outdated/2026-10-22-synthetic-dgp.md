# 1단계 합성 DGP와 채점기 Implementation Plan

> **For agentic workers:** 이 계획은 Codex가 구현한다. 작업마다 아래 단계를 체크박스(`- [ ]`)로 따라가고, 모든 작업이 끝나면 "Codex 기초 검토 2회"(맨 끝)를 한 뒤 보고한다. 실험 실행(예비 실행, 점검 실행)은 하지 않는다. 그건 Claude 검토 3회와 사용자 승인 뒤에 따로 한다.

**Goal:** 초안의 가정(local support $k=2$, floor $r=1/2$)을 구성상 만족하거나 의도적으로 깨는 8열×4수준 합성 환경 18개(STZ 축 3 × 겹침 축 3 × 가정 축 2)와, 셀 단위·조각 단위 정답 채점기를 만든다.

**Architecture:** 65,536개 셀을 모두 나열한 배열 위에서 $p_0$, $p_1$을 정확히 계산한다. 모집단 seed 하나에서 $p_0$, $h$, 0 쌍, $D_0$를 공유하고, 규칙과 섞기 비율만 바꿔 환경 18개를 만든다(짝지은 설계). 채점기는 정답 분류 위에서 집합, 법칙, 생성 행, 인증 집합을 채점한다.

**Tech Stack:** Python 3.10, numpy 2.2.6, scipy 1.15.3, pandas 2.3.3, pytest 9.1.1 (env `tabular_foundation`에 설치되어 있음)

**Spec:** `/data1/calvinkim1027/BOMIS_new_수만이형/docs/superpowers/specs/2026-10-22-synthetic-dgp-design.md` (구현 전에 끝까지 읽는다. 값과 이유는 spec이 기준이다)

## Global Constraints

**경로와 실행**

- 프로젝트 루트(코드 작성 위치): `/data1/calvinkim1027/BOMIS_new_수만이형`. 로그인 노드 로컬 디스크라 계산 노드에서 보이지 않는다.
- 실행 위치: `/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime`. 실행 전에 `bash experiments/sync_runtime.sh`로 코드를 복사한다.
- Python은 로그인 노드에서 직접 실행할 수 없다. 반드시 srun 또는 sbatch로 실행한다. 아래에서 이 명령을 `RUNPY <args>`로 줄여 쓴다.
  `srun --partition=brl --ntasks=1 --qos=normal --exclude=brl1,brl2 --mem=8G --time=00:30:00 --chdir=/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime /mnt/data1/calvinkim1027/miniconda3/envs/tabular_foundation/bin/python -I <args>`
- `-I`(Python 3.10)는 스크립트 디렉터리와 현재 디렉터리를 `sys.path`에 넣지 않는다.
  - 진입 스크립트(`scripts/*.py`)는 첫 줄에서 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`를 한다.
  - pytest는 `pytest.ini`의 `pythonpath = .`로 해결한다.
- git 저장소가 아니다. commit 단계는 없다. GPU를 쓰지 않는다.

**공간과 번호**

- 열 0–3은 부모 $P_1$–$P_4$, 열 4–7은 잎. 부모 열 $i$의 잎은 열 $4+i$이다.
- 간선: 사슬 (0,1),(1,2),(2,3), 잎 (0,4),(1,5),(2,6),(3,7). 뿌리는 열 0.
- 부모 쌍(0을 두는 곳) 순서: `PARENT_PAIRS = ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))`
- 셀 번호: `index = Σ_j c_j · 4^(7−j)` (열 0이 최상위 자리).
- 조각 번호
  - 단일: `4*j + v` (0–31)
  - 쌍: 열 쌍 $(j,l)$, $j<l$의 사전순 순위 `rank`(0–27)로 `32 + 16*rank + 4*v + w` (32–479)
- $m = 480$, $\alpha = 0.05$, $\tau = \log 9600$.

**모집단 값**

- $r_{\text{true}} = 0.5$(방법도 같은 값), $\omega = 0.2$.
- 공유 0: 부모 쌍마다 하나씩 6개. majority 전용 0($Z$ 쌍): 서로 다른 부모 쌍 2개에 하나씩, 그 쌍의 공유 0과 다른 값 쌍.
  - 두 종류 모두 0으로 만들기 전 쌍 확률이 0.05 이하인 값 쌍 중에서 균등 무작위로 고른다.
  - $Z$ 쌍의 부모 쌍 2개는 6개 중 균등 무작위로 고른다.
- STZ 축 `FREQ_LEVELS`(목표, 하한, 상한, 양 끝 포함): frequent (0.12, 0.09, 0.16), mid (0.03, 0.025, 0.04), rare (0.005, 0.003, 0.008).
- 금지 쌍 선택
  - 잎 간선마다 $(P=a', L=a)$를 고른다. 조건: $e = P_0(L=a \mid P=a') \le 0.2$, $\pi_0$가 범위 안.
  - 그중 $|\pi_0 - \text{target}|$이 가장 작은 것. 같으면 $(a', a)$ 사전순.
- 3변수 규칙 선택(부모 열 0과 2에 하나씩)
  - 형태: $(P=c', L=c, P^{+}=b)$. $L$은 그 부모의 잎(열 4, 6), $P^{+}$는 다음 부모(열 1, 3)이다.
  - 조건 1: $e^T = P_0(L=c \mid P=c') \le 0.1$
  - 조건 2: $c'$는 세 수준 금지 쌍의 $a'$와 모두 다르다
  - 조건 3: $\pi_0 \in [0.01, 0.04]$
  - 조건 4: 하위 쌍 $(P=c', P^{+}=b)$의 $\pi_0 > 0$
  - 그중 $|\pi_0 - 0.02|$가 가장 작은 것. 같으면 $(c', c, b)$ 사전순.
- 겹침 축 `GAMMAS = {"fav": 0.0, "avg": 0.5, "unf": 1.0}`, 가정 축 `ASSUMPTIONS = ("holds", "viol")`.
- 환경 이름 `f"{freq}__{ov}__{asm}"`. 순서는 freq(frequent, mid, rare), ov(fav, avg, unf), asm(holds, viol) 순으로 바깥부터 돈다. 모두 18개.

**표본과 seed**

- $n_0 = 8000$, $n_1 \in \{50, 100, 200, 400, 800\}$, 표본 seed 0–19.
- 모집단 seed
  - 개발용: 0부터 시도해 처음 수락된 10개
  - 최종 보고용: 1000번대, 쓰지 않음
  - 예비 실행: 5000–5099
  - 테스트와 `--quick`: 9000번대
  - 최대 시도 2000번
- $\eta$ 후보 (1.0, 0.5, 0.3). 수락률 0.05 이상인 값 중 가장 큰 것.
- 난수 흐름은 `np.random.default_rng(np.random.SeedSequence(entropy))`. entropy는 다음과 같다.

  | 대상 | entropy |
  |---|---|
  | $p_0$ 표 | `[pop_seed, 0]` |
  | $h$ 표 | `[pop_seed, 1]` |
  | 0 쌍 선택 | `[pop_seed, 2]` |
  | $D_0$ | `[pop_seed, sample_seed, 10]` |
  | $D_1$ 균등 난수 | `[pop_seed, sample_seed, 11]` |

- $D_1$은 균등 난수 800개를 한 번 뽑아 앞의 $n_1$개를 쓴다. 환경 사이에서는 같은 난수, $n_1$ 사이에서는 포함 관계가 된다.
- $D_0$는 (pop_seed, sample_seed)마다 하나다.

**공통 규칙**

- 분모가 0인 지표는 예외 없이 `nan`.
- AUC는 계산하지 않는다.

## Review Focus

1. 부동소수 누적합 때문에 확률 0인 셀이 추출되는 경우: 어떤 표본 셀도 지지집합 밖에 있으면 안 된다 (Task 5).
2. 한 셀이 규칙을 둘 이상 포함하는 경우: 부모 좌표를 모두 바꾼 조합으로 나누고, 손실 비율이 spec 3.4절 값과 맞아야 한다 (Task 3).
3. 규칙이나 0 쌍을 고를 수 없는 seed: 예외 없이 거절 사유를 돌려주고 다음 seed로 넘어가야 한다 (Task 4).
4. 분모가 0인 경우($M_0=0$, 성립 환경의 3변수 STZ, $\hat q(\text{SZ})=0$): `nan` (Task 6).
5. `-I` 모드에서 진입 스크립트의 `import dgp` 실패: `python -I scripts/run_check.py --help`가 성공해야 한다 (Task 8).

---

## 파일 구성

| 파일 | 책임 |
|---|---|
| `dgp/__init__.py` | 빈 파일 |
| `dgp/space.py` | 셀 배열, 셀·조각 번호, 조각 확률과 개수 |
| `dgp/laws.py` | 트리 법칙, 부모 쌍 위의 0 |
| `dgp/minority.py` | 규칙 선택, 영역, 질량 이동, 섞기, minority 점검 |
| `dgp/population.py` | 모집단과 18개 환경 조립, 수락 반복, 저장·불러오기 |
| `dgp/pilot.py` | $\eta$ 예비 실행 |
| `dgp/sampling.py` | $D_0$, $D_1$ 추출 |
| `dgp/truth.py` | 표본별 정답 분류 |
| `dgp/scorer.py` | 채점 (a)(b)(c) |
| `dgp/cert_metrics.py` | 채점 (d), 정확 계산 |
| `dgp/check.py` | spec 8절 점검, 표와 보고서 |
| `scripts/run_pilot.py`, `scripts/run_check.py` | 진입 스크립트 |
| `tests/conftest.py`, `tests/test_*.py` | 테스트 |
| `pytest.ini` | `[pytest]` / `pythonpath = .` / `testpaths = tests` |
| `experiments/sync_runtime.sh` | 코드 복사 |
| `experiments/pytest.sbatch`, `experiments/dgp_pilot.sbatch`, `experiments/dgp_check.sbatch` | 실행용 |

---

### Task 1: 공간과 실행 뼈대

**Files:** Create `dgp/__init__.py`, `dgp/space.py`, `pytest.ini`, `experiments/sync_runtime.sh`, `experiments/pytest.sbatch` / Test `tests/test_space.py`

**Interfaces — Produces (`dgp/space.py`):**
- 상수: `D=8`, `K=4`, `N_CELLS=65536`, `N_PATTERNS=480`, `PARENTS=(0,1,2,3)`, `LEAF_OF={0:4,1:5,2:6,3:7}`, `CHAIN_EDGES`, `LEAF_EDGES`, `PARENT_PAIRS`
- `CELLS: np.ndarray` (65536, 8) int8, 셀 번호 순서
- `cell_index(rows: np.ndarray) -> np.ndarray` — (n,8) → (n,) int64
- `PAT_COLS`, `PAT_VALS`: (480,2) int8. 단일 조각의 둘째 칸은 -1
- `pattern_id(cols: tuple[int, ...], vals: tuple[int, ...]) -> int` — cols 오름차순, 길이 1 또는 2
- `CELL_PATTERNS: np.ndarray` (65536, 36) int16 — 셀마다 단일 8개 + 쌍 28개의 조각 번호
- `pattern_probs(p: np.ndarray) -> np.ndarray` — (65536,) → (480,)
- `pattern_counts(cell_idx: np.ndarray) -> np.ndarray` — (n,) → (480,) int64
- `cells_containing(pid: int) -> np.ndarray` — (65536,) bool

- [ ] **Step 1: 테스트 작성** (`tests/test_space.py`)

```python
def test_cell_index_order():
    assert cell_index(np.zeros((1, 8), int))[0] == 0
    assert cell_index(np.full((1, 8), 3))[0] == 65535
    assert cell_index(np.array([[0,0,0,0,0,0,0,1]]))[0] == 1
    assert cell_index(np.array([[1,0,0,0,0,0,0,0]]))[0] == 16384
    assert np.array_equal(cell_index(CELLS.astype(int)), np.arange(65536))

def test_pattern_ids():
    assert pattern_id((0,), (0,)) == 0 and pattern_id((7,), (3,)) == 31
    assert pattern_id((0, 1), (0, 0)) == 32 and pattern_id((6, 7), (3, 3)) == 479
    assert len({tuple(r) for r in np.c_[PAT_COLS, PAT_VALS]}) == 480

def test_cell_patterns_and_probs():
    assert CELL_PATTERNS.shape == (65536, 36)
    assert all(len(set(row)) == 36 for row in CELL_PATTERNS[::997])
    pp = pattern_probs(np.full(65536, 1 / 65536))
    assert np.allclose(pp[:32], 0.25) and np.allclose(pp[32:], 1 / 16)

def test_pattern_counts_matches_bruteforce():
    idx = np.random.default_rng(0).integers(0, 65536, 500)
    cnt = pattern_counts(idx)
    for pid in (0, 17, 32, 200, 479):
        assert cnt[pid] == cells_containing(pid)[idx].sum()
```

- [ ] **Step 2: 실패 확인** — `pytest.ini`와 `sync_runtime.sh`를 먼저 만든다.
  - `sync_runtime.sh`는 `dgp/ scripts/ tests/ experiments/ pytest.ini`를 `rsync -a --delete`로 runtime에 복사한다.
  - runtime에 `logs/ results/ reports/`가 없으면 만든다. 이미 있으면 내용은 건드리지 않는다.
  - Run: `bash experiments/sync_runtime.sh && RUNPY -m pytest tests/test_space.py -v`
  - Expected: FAIL (`dgp.space` 없음)
- [ ] **Step 3: 구현** — `pattern_probs`와 `pattern_counts`는 `CELL_PATTERNS`와 `np.bincount(..., minlength=480)`로 계산한다. `pytest.sbatch`는 위 srun과 같은 자원으로 `python -I -m pytest -v`를 실행한다.
- [ ] **Step 4: 통과 확인** — 같은 명령. Expected: 4 passed

---

### Task 2: 트리 법칙과 부모 쌍 위의 0

**Files:** Create `dgp/laws.py` / Test `tests/test_laws.py`

**Interfaces**
- Consumes: `dgp.space`
- Produces (`dgp/laws.py`):
  - `@dataclass TreeTables: root: np.ndarray; cond: dict[tuple[int,int], np.ndarray]`
    - `root`: (4,)
    - `cond`: 키는 (부모 열, 자식 열), 값은 (4,4) 표(행 = 부모 값, 행 합 1). 간선 7개
  - `draw_tree_tables(rng, eta: float) -> TreeTables` — 뿌리와 각 행을 `rng.dirichlet(np.full(4, eta))`로 뽑는다
  - `tree_law(t: TreeTables) -> np.ndarray` — (65536,), 합 1
  - `ParentZero = tuple[int, int, int, int]` — (j, l, v, w), j<l
  - `choose_parent_zeros(p_tree, rng, max_mass: float = 0.05, n_major: int = 2) -> tuple[list[ParentZero], list[ParentZero]] | None`
    - 반환: (공유 0 6개, $Z$ 쌍 n_major개)
    - 공유 0은 `PARENT_PAIRS` 순서로 하나씩 고른다.
    - $Z$ 쌍은 서로 다른 부모 쌍 n_major개를 균등 무작위로 고르고, 그 쌍의 공유 0과 다른 값 쌍을 고른다.
    - 후보가 없으면 None
  - `apply_zeros(p, zeros: list[ParentZero]) -> np.ndarray` — 해당 셀을 0으로 하고 다시 정규화

- [ ] **Step 1: 테스트 작성** (`tests/test_laws.py`)

```python
def _tables(seed=0, eta=1.0):
    return draw_tree_tables(np.random.default_rng(seed), eta)

def test_tree_law_normalized_positive():
    p = tree_law(_tables())
    assert abs(p.sum() - 1) < 1e-12 and (p > 0).all()

def test_parent_zeros_structure():
    p_tree = tree_law(_tables(3))
    shared, major = choose_parent_zeros(p_tree, np.random.default_rng(4))
    assert [(z[0], z[1]) for z in shared] == list(PARENT_PAIRS)
    assert len(major) == 2 and len({(z[0], z[1]) for z in major}) == 2
    for z in major:
        assert z not in shared and (z[0], z[1]) in PARENT_PAIRS
    for (j, l, v, w) in shared + major:
        m = (CELLS[:, j] == v) & (CELLS[:, l] == w)
        assert p_tree[m].sum() <= 0.05
        assert apply_zeros(p_tree, shared + major)[m].sum() == 0

def test_leaf_conditional_independence_after_zeros():
    t = _tables(1)
    shared, major = choose_parent_zeros(tree_law(t), np.random.default_rng(2))
    p0 = apply_zeros(tree_law(t), shared + major)
    for i in range(4):
        for x in [c for c in range(8) if c not in (i, 4 + i)]:
            for ap in range(4):
                for xv in range(4):
                    m = (CELLS[:, i] == ap) & (CELLS[:, x] == xv)
                    if p0[m].sum() > 0:
                        for a in range(4):
                            got = p0[m & (CELLS[:, 4 + i] == a)].sum() / p0[m].sum()
                            assert abs(got - t.cond[(i, 4 + i)][ap, a]) < 1e-10

def test_choose_parent_zeros_none_when_impossible():
    assert choose_parent_zeros(tree_law(_tables(5)), np.random.default_rng(0), max_mass=-1.0) is None
```

- [ ] **Step 2: 실패 확인** — Run: `bash experiments/sync_runtime.sh && RUNPY -m pytest tests/test_laws.py -v`. Expected: FAIL
- [ ] **Step 3: 구현** — `tree_law`는 `CELLS` 열로 표를 인덱싱해서 곱한다(반복문 없이).
- [ ] **Step 4: 통과 확인** — Expected: 4 passed

---

### Task 3: 규칙, 영역, minority 법칙

**Files:** Create `dgp/minority.py` / Test `tests/test_minority.py`

**Interfaces**
- Consumes: `dgp.space`, `dgp.laws`
- Produces (`dgp/minority.py`):
  - 상수: `FREQ_LEVELS`, `E_MAX = 0.2`, `E_MAX_TRIPLE = 0.1`, `TRIPLE_RANGE = (0.02, 0.01, 0.04)`, `TRIPLE_PARENTS = (0, 2)`, `OMEGA = 0.2`, `GAMMAS`, `ASSUMPTIONS`, `R_TRUE = 0.5`
  - `class Rule(NamedTuple): kind: str; cols: tuple[int, ...]; vals: tuple[int, ...]; move_col: int; e: float; pi0: float`
    - `kind`는 "pair" 또는 "triple"
    - pair: `cols = (parent, leaf)`
    - triple: `cols = (parent, leaf, parent+1)`
    - `move_col`은 바꿀 부모 열
  - `rule_mask(rule: Rule) -> np.ndarray` — 규칙을 포함하는 셀 (65536,) bool
  - `rule_pid(rule: Rule) -> int` — pair만. `pattern_id(cols, vals)`
  - `select_pairs(p0, level: str) -> list[Rule] | None` — 부모 0→3 순서로 4개
  - `select_triples(p0, pairs_by_level: dict[str, list[Rule]]) -> list[Rule] | None` — 부모 0, 2 순서로 2개
  - `regions(shared, major, rules, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — (R, Z, S1). spec 3.2절
  - `move_mass(p0, R, rules) -> tuple[np.ndarray, float]` — ($p_1^{\text{base}}$, 버린 질량). 대상 셀은 R 안
  - `mix(p1base, h, R, Z, gamma: float, omega: float = OMEGA) -> np.ndarray` — `gamma == 0`이면 $Z$ 항은 없다
  - `pairwise_feasible_cells(pi1: np.ndarray) -> np.ndarray` — `CELL_PATTERNS`의 36개 조각이 모두 `pi1 > 0`인 셀
  - `check_minority(p0, p1, S1, Z, rules, gamma: float, r: float = R_TRUE) -> dict`
    - `sum_ok`
    - `support_ok`: p1>0 ⟺ S1
    - `local_support_ok`: triple이 없으면 `pairwise_feasible_cells(pi1) == S1`. 있으면 `F & ~S1`이 `F & (triple 포함 셀)`과 정확히 같고 비어 있지 않음(`F = pairwise_feasible_cells(pi1)`. triple과 금지 쌍을 함께 포함하는 셀은 F에 없으므로 비교에서 빠짐)
    - `triples_minimal_ok`: 각 triple의 하위 쌍 3개가 모두 π1>0. triple이 없으면 True
    - `infeasible_ok`: π0>0이고 π1=0인 조각 = pair 규칙의 pid 4개
    - `z_mass_ok`: |p1[Z].sum() − OMEGA·gamma| < 1e-12
    - `min_ratio`, `floor_ok`, `passed`

- [ ] **Step 1: 테스트 작성** (`tests/test_minority.py`)

`prep` fixture(module scope) 만드는 법:
- seed $s$를 9000부터 300개까지 올려 가며 다음 순서로 시도한다.
  - `t = draw_tree_tables(default_rng(SeedSequence([s,0])), 0.5)`
  - 0 쌍: `choose_parent_zeros(tree_law(t), default_rng(SeedSequence([s,2])))`
  - `p0`, `S0`
  - 세 수준의 `select_pairs`
  - `select_triples`
- 모두 None이 아닌 첫 seed를 쓴다. 못 찾으면 테스트 실패.
- `h = tree_law(draw_tree_tables(default_rng(SeedSequence([s,1])), 0.5))`
- 필드: `p0, S0, h, shared, major, pairs(dict), triples`

```python
def test_select_pairs_rules(prep):
    for level, (target, lo, hi) in FREQ_LEVELS.items():
        fp = prep.pairs[level]
        assert [f.cols for f in fp] == [(0, 4), (1, 5), (2, 6), (3, 7)]
        for f in fp:
            assert f.kind == "pair" and f.move_col == f.cols[0]
            assert f.e <= E_MAX + 1e-12 and lo <= f.pi0 <= hi
            assert abs(f.pi0 - prep.p0[rule_mask(f)].sum()) < 1e-12

def test_select_triples_rules(prep):
    assert [t.cols for t in prep.triples] == [(0, 4, 1), (2, 6, 3)]
    for t in prep.triples:
        assert t.kind == "triple" and t.move_col == t.cols[0] and t.e <= E_MAX_TRIPLE + 1e-12
        assert 0.01 <= t.pi0 <= 0.04 and abs(t.pi0 - prep.p0[rule_mask(t)].sum()) < 1e-12
        for lv in FREQ_LEVELS:   # c'가 모든 수준의 a'와 다름
            pair = [f for f in prep.pairs[lv] if f.cols[0] == t.cols[0]][0]
            assert t.vals[0] != pair.vals[0]

def test_regions(prep):
    rules = prep.pairs["mid"] + prep.triples
    for g in (0.0, 0.5, 1.0):
        R, Z, S1 = regions(prep.shared, prep.major, rules, g)
        assert (R <= prep.S0).all() and not (Z & prep.S0).any() and Z.any()
        assert np.array_equal(S1, R if g == 0 else R | Z)
        assert not any((rule_mask(r) & S1).any() for r in rules)

def test_move_mass_supports(prep):
    R, Z, S1 = regions(prep.shared, prep.major, prep.pairs["frequent"], 0.5)
    base, dropped = move_mass(prep.p0, R, prep.pairs["frequent"])
    assert abs(base.sum() - 1) < 1e-12 and 0 <= dropped < 1
    assert (base[~R] == 0).all() and (base[R] > 0).all()

def test_loss_fractions_exact_pairs_only(prep):
    pr = prep.pairs["frequent"]          # 금지 쌍만: 부모 값 a'로 들어오는 질량이 없어 손실이 정확함
    R, Z, S1 = regions(prep.shared, prep.major, pr, 0.0)
    base, dropped = move_mass(prep.p0, R, pr)
    if dropped > 0: pytest.skip("drop case")
    pi0, pib = pattern_probs(prep.p0), pattern_probs(base)
    f0, f1 = pr[0], pr[1]
    pid = pattern_id((f0.cols[0],), (f0.vals[0],))                       # 부모 하나: 손실 e
    assert abs(pib[pid] / pi0[pid] - (1 - f0.e)) < 1e-10
    pid2 = pattern_id((f0.cols[0], f1.cols[0]), (f0.vals[0], f1.vals[0]))  # 금지 쌍 부모 둘
    if pi0[pid2] > 0:
        assert abs(pib[pid2] / pi0[pid2] - (1 - f0.e) * (1 - f1.e)) < 1e-10

def test_triple_loss_bounded(prep):
    rules = prep.pairs["frequent"] + prep.triples   # 같은 부모 열의 금지 쌍 이동이 c'로 질량을 보낼 수 있어 하한만 확인
    R, Z, S1 = regions(prep.shared, prep.major, rules, 0.0)
    base, dropped = move_mass(prep.p0, R, rules)
    if dropped > 0: pytest.skip("drop case")
    pi0, pib = pattern_probs(prep.p0), pattern_probs(base)
    for t in prep.triples:
        pid = pattern_id((t.cols[0],), (t.vals[0],))
        assert pib[pid] / pi0[pid] >= 1 - prep.p0[rule_mask(t)].sum() / pi0[pid] - 1e-12
        assert pib[pid] / pi0[pid] >= 1 - E_MAX_TRIPLE - 1e-12

def test_mix_z_mass(prep):
    rules = prep.pairs["rare"]
    for g in (0.0, 0.5, 1.0):
        R, Z, S1 = regions(prep.shared, prep.major, rules, g)
        p1 = mix(move_mass(prep.p0, R, rules)[0], prep.h, R, Z, g)
        assert abs(p1.sum() - 1) < 1e-12 and abs(p1[Z].sum() - OMEGA * g) < 1e-12

def test_floor_and_checks_all_envs(prep):
    for lv in FREQ_LEVELS:
        for asm in ASSUMPTIONS:
            rules = prep.pairs[lv] + (prep.triples if asm == "viol" else [])
            for g in (0.0, 0.5, 1.0):
                R, Z, S1 = regions(prep.shared, prep.major, rules, g)
                base, dropped = move_mass(prep.p0, R, rules)
                res = check_minority(prep.p0, mix(base, prep.h, R, Z, g), S1, Z, rules, g)
                if dropped == 0:
                    assert res["min_ratio"] >= 0.8 * 0.64 - 1e-12 and res["passed"], (lv, asm, g, res)

def test_local_support_violation_is_exactly_triples(prep):
    rules = prep.pairs["mid"] + prep.triples
    R, Z, S1 = regions(prep.shared, prep.major, rules, 0.5)
    p1 = mix(move_mass(prep.p0, R, rules)[0], prep.h, R, Z, 0.5)
    F = pairwise_feasible_cells(pattern_probs(p1))
    tri = rule_mask(prep.triples[0]) | rule_mask(prep.triples[1])
    extra = F & ~S1
    assert extra.any() and np.array_equal(extra, F & tri)
    assert check_minority(prep.p0, p1, S1, Z, rules, 0.5)["local_support_ok"]

def test_check_detects_support_violation(prep):
    rules = prep.pairs["mid"]
    R, Z, S1 = regions(prep.shared, prep.major, rules, 0.0)
    bad = np.where(S1, prep.p0, 0.0); bad[np.flatnonzero(S1)[0]] = 0; bad /= bad.sum()
    assert not check_minority(prep.p0, bad, S1, Z, rules, 0.0)["support_ok"]
```

- [ ] **Step 2: 실패 확인** — Run: `bash experiments/sync_runtime.sh && RUNPY -m pytest tests/test_minority.py -v`. Expected: FAIL
- [ ] **Step 3: 구현**
  - `move_mass`: 규칙을 포함하는 각 셀에서, 포함한 규칙들의 `move_col`을 각각 자기 값이 아닌 3개 값으로 바꾼 모든 조합 중 `R` 안에 오는 셀로 질량을 똑같이 나눈다. 없으면 버린다. 끝에 재정규화한다(반환하는 버린 질량은 재정규화 전 값).
  - 성능 목표: 호출 한 번에 1초 이내.
- [ ] **Step 4: 통과 확인** — Expected: 10 passed

---

### Task 4: 모집단과 환경 조립, 수락, 저장, 예비 실행

**Files:** Create `dgp/population.py`, `dgp/pilot.py` / Test `tests/test_population.py`

**Interfaces**
- Consumes: Task 1–3
- Produces (`dgp/population.py`):
  - `ENV_KEYS: list[str]` — 18개, Global Constraints 순서
  - `parse_env(key: str) -> tuple[str, str, str]`
  - `@dataclass EnvLaw: key: str; gamma: float; rules: list[Rule]; R: np.ndarray; Z: np.ndarray; S1: np.ndarray; p1: np.ndarray; pi1: np.ndarray; dropped_mass: float; min_ratio: float`
  - `@dataclass Population: pop_seed: int; eta: float; tables0: TreeTables; tables_h: TreeTables; shared_zeros: list; major_zeros: list; p0: np.ndarray; S0: np.ndarray; pi0: np.ndarray; h: np.ndarray; pairs: dict[str, list[Rule]]; triples: list[Rule]; envs: dict[str, EnvLaw]`
  - `build_population(pop_seed: int, eta: float) -> tuple[Population | None, str]`
    - 사유: `"ok"`, `"no_zeros"`, `"no_pair:<freq>"`, `"no_triple"`, `"check_fail:<env>:<첫 실패 키>"`
    - `move_mass`는 (freq, asm)마다 한 번만 계산하고 세 $\gamma$에 재사용한다
  - `class StopGeneration(RuntimeError)`
  - `generate_accepted(start_seed: int, n_accept: int, eta: float, max_tries: int = 2000) -> tuple[list[Population], list[tuple[int, str]]]`
  - `save_population(pop, path)`, `load_population(path) -> Population`
    - npz 하나에 저장한다.
    - 환경 배열 이름은 `p1__<env>`, `S1__<env>`, `R__<env>`, `Z__<env>`, `pi1__<env>`이다.
    - 나머지는 JSON 문자열 `meta`에 담는다.
  - `population.py`는 `from dgp.minority import select_pairs, select_triples, regions, move_mass, mix, check_minority` 형태로 가져온다. 테스트가 `dgp.population.select_pairs`, `dgp.population.select_triples`, `dgp.population.build_population`을 monkeypatch한다.
- Produces (`dgp/pilot.py`):
  - `ETA_CANDIDATES = (1.0, 0.5, 0.3)`, `PILOT_SEEDS = range(5000, 5100)`, `MIN_ACCEPT_RATE = 0.05`
  - `acceptance_rate(eta, seeds) -> tuple[float, dict[str, int]]`
  - `choose_eta(rates: dict[float, float]) -> float | None`

- [ ] **Step 1: 테스트 작성** (`tests/test_population.py`)

```python
@pytest.fixture(scope="module")
def pop():
    return generate_accepted(9000, 1, eta=0.5, max_tries=300)[0][0]

def test_paired_design(pop):
    assert list(pop.envs) == ENV_KEYS and len(ENV_KEYS) == 18
    for key, env in pop.envs.items():
        freq, ov, asm = parse_env(key)
        pairs = [r for r in env.rules if r.kind == "pair"]
        tris = [r for r in env.rules if r.kind == "triple"]
        assert pairs == pop.pairs[freq]
        assert tris == (pop.triples if asm == "viol" else [])
        assert env.gamma == {"fav": 0.0, "avg": 0.5, "unf": 1.0}[ov]
        assert (env.R <= pop.S0).all() and env.min_ratio >= 0.5
    pids = [{rule_pid(r) for r in pop.pairs[f]} for f in ("frequent", "mid", "rare")]
    assert not (pids[0] & pids[1]) and not (pids[1] & pids[2]) and not (pids[0] & pids[2])

def test_reproducible():
    a, ra = build_population(9001, 0.5); b, rb = build_population(9001, 0.5)
    assert ra == rb
    if a is not None:
        assert np.array_equal(a.p0, b.p0)
        assert np.array_equal(a.envs["mid__avg__viol"].p1, b.envs["mid__avg__viol"].p1)

def test_rejection_reasons(pop, monkeypatch):
    import dgp.population as P             # pop.pop_seed는 수락된 seed라 앞 단계는 모두 통과함
    monkeypatch.setattr(P, "select_pairs", lambda p0, level: None)
    assert P.build_population(pop.pop_seed, 0.5) == (None, "no_pair:frequent")
    monkeypatch.undo()
    monkeypatch.setattr(P, "select_triples", lambda p0, pairs: None)
    assert P.build_population(pop.pop_seed, 0.5) == (None, "no_triple")

def test_stop_generation(monkeypatch):
    import dgp.population as P
    monkeypatch.setattr(P, "build_population", lambda s, e: (None, "no_triple"))
    with pytest.raises(P.StopGeneration):
        P.generate_accepted(0, 1, eta=0.5, max_tries=5)

def test_save_load_roundtrip(pop, tmp_path):
    save_population(pop, tmp_path / "pop.npz")
    q = load_population(tmp_path / "pop.npz")
    assert np.array_equal(q.p0, pop.p0) and np.array_equal(q.pi0, pop.pi0)
    assert q.shared_zeros == pop.shared_zeros and q.major_zeros == pop.major_zeros and q.eta == pop.eta
    for k in pop.envs:
        assert np.array_equal(q.envs[k].p1, pop.envs[k].p1) and q.envs[k].rules == pop.envs[k].rules
        assert np.array_equal(q.envs[k].Z, pop.envs[k].Z)

def test_z_is_shared_across_envs(pop):
    zs = {pop.envs[k].Z.tobytes() for k in ENV_KEYS if pop.envs[k].gamma > 0 and k.endswith("holds")}
    assert len(zs) <= 3      # Z는 0 쌍과 그 환경의 규칙으로만 정해짐 (STZ 수준 3개)

def test_choose_eta():
    assert choose_eta({1.0: 0.01, 0.5: 0.08, 0.3: 0.2}) == 0.5
    assert choose_eta({1.0: 0.01, 0.5: 0.02, 0.3: 0.04}) is None
```

- [ ] **Step 2: 실패 확인** — Run: `bash experiments/sync_runtime.sh && RUNPY -m pytest tests/test_population.py -v`. Expected: FAIL
- [ ] **Step 3: 구현** — `build_population`은 다음 순서로 진행한다. 어느 단계든 실패하면 해당 사유로 None을 돌려준다.
  1. $p_0$ 표 → 트리 법칙
  2. 0 쌍 선택 → `p0`, `S0`, `pi0`
  3. $h$
  4. 세 수준 `select_pairs` → `select_triples`
  5. 18개 환경마다 `regions` → `move_mass`(재사용) → `mix` → `check_minority`
- [ ] **Step 4: 통과 확인** — Expected: 7 passed

---

### Task 5: 표본 추출과 정답 분류

**Files:** Create `dgp/sampling.py`, `dgp/truth.py`, `tests/conftest.py` / Test `tests/test_sampling_truth.py`

**Interfaces**
- Consumes: `Population`, `EnvLaw`, `dgp.space`, `dgp.minority.rule_mask`
- Produces (`dgp/sampling.py`):
  - `N0 = 8000`, `N1_GRID = (50, 100, 200, 400, 800)`, `SAMPLE_SEEDS = range(20)`
  - `draw_cells(p, u) -> np.ndarray` int64
    - 누적합을 마지막 값으로 나눈 뒤 `searchsorted(..., side="right")`를 쓴다.
    - 확률 0인 셀이 나오면 그 앞에서 가장 가까운 양수 확률 셀로 옮긴다.
  - `draw_D0(pop, sample_seed, n0=N0) -> np.ndarray`
  - `draw_D1(pop, env: str, sample_seed, n1) -> np.ndarray`
- Produces (`dgp/truth.py`):
  - `@dataclass Truth`
    - 셀 마스크(bool): `O, O0, SZ, SZ1, SZ2, SZ_nomaj, STZ, STZ_maj, STZ_common, STZ_triple`
    - 값: `M0: float; N0: np.ndarray; N1: np.ndarray`
  - `make_truth(pop, env: str, D0, D1) -> Truth` — spec 6절 그대로. `STZ_triple`은 STZ_maj 중 triple을 포함하고 pair 규칙은 포함하지 않는 셀
- `tests/conftest.py`: session scope fixture `pop = generate_accepted(9000, 1, eta=0.5, max_tries=300)[0][0]`. Task 6, 7, 8도 쓴다.

- [ ] **Step 1: 테스트 작성** (`tests/test_sampling_truth.py`)

```python
def test_draws_stay_in_support(pop):
    for env in ENV_KEYS:
        assert pop.envs[env].S1[draw_D1(pop, env, 0, 800)].all()
    assert pop.S0[draw_D0(pop, 0)].all()

def test_draw_cells_never_hits_zero_prob():
    p = np.zeros(65536); p[3] = 0.5; p[4464] = 0.5
    u = np.r_[np.linspace(0, 1, 1001)[:-1], 0.9999999999999999, 1.0]
    assert set(draw_cells(p, u).tolist()) <= {3, 4464}

def test_paired_and_nested(pop):
    assert np.array_equal(draw_D0(pop, 3), draw_D0(pop, 3))
    e = "mid__avg__holds"
    assert np.array_equal(draw_D1(pop, e, 3, 100)[:50], draw_D1(pop, e, 3, 50))
    u = np.random.default_rng(np.random.SeedSequence([pop.pop_seed, 3, 11])).random(800)[:100]
    e2 = "rare__unf__viol"
    assert np.array_equal(draw_D1(pop, e2, 3, 100), draw_cells(pop.envs[e2].p1, u))

def test_truth_partitions(pop):
    for env in ("frequent__fav__holds", "rare__unf__viol"):
        tr = make_truth(pop, env, draw_D0(pop, 0), draw_D1(pop, env, 0, 200))
        S1 = pop.envs[env].S1
        assert np.array_equal(tr.SZ | tr.O, S1) and not (tr.SZ & tr.O).any()
        assert np.array_equal(tr.SZ1 | tr.SZ2, tr.SZ) and not (tr.SZ1 & tr.SZ2).any()
        assert np.array_equal(tr.SZ_nomaj, tr.SZ & (pop.p0 == 0)) and (tr.SZ_nomaj <= tr.SZ2).all()
        assert np.array_equal(tr.STZ, ~S1)
        assert np.array_equal(tr.STZ_maj | tr.STZ_common, tr.STZ) and not (tr.STZ_maj & tr.STZ_common).any()
        assert np.array_equal(tr.STZ_maj, pop.S0 & ~S1) and (tr.STZ_triple <= tr.STZ_maj).all()
        assert tr.STZ_triple.any() == env.endswith("viol")
        assert abs(tr.M0 - pop.envs[env].p1[tr.SZ].sum()) < 1e-12
        assert all(tr.N1[rule_pid(r)] == 0 for r in pop.envs[env].rules if r.kind == "pair")
    tr = make_truth(pop, "mid__fav__holds", draw_D0(pop, 0), draw_D1(pop, "mid__fav__holds", 0, 200))
    assert not tr.SZ_nomaj.any()        # fav에서는 majority 불가 SZ 없음
```

- [ ] **Step 2: 실패 확인** — Run: `bash experiments/sync_runtime.sh && RUNPY -m pytest tests/test_sampling_truth.py -v`. Expected: FAIL
- [ ] **Step 3: 구현** — 난수 흐름은 Global Constraints대로.
- [ ] **Step 4: 통과 확인** — Expected: 4 passed

---

### Task 6: 채점 (a) 집합, (b) 법칙, (c) 생성 행

**Files:** Create `dgp/scorer.py` / Test `tests/test_scorer.py`

**Interfaces**
- Consumes: `Population`, `Truth`, `rule_mask`
- Produces (`dgp/scorer.py`). 모두 `dict[str, float]`을 반환하고, 분모가 0이면 `nan`이다.
  - `score_set(pop, env, tr, S_hat) -> dict`
    - SZ 쪽 키: `sz_rec_mass`, `sz_rec_mass_1`, `sz_rec_mass_2`, `sz_rec_mass_nomaj`, `sz_rec_count`
    - STZ 쪽 키: `stz_sup_count`, `stz_sup_count_maj`, `stz_sup_count_common`, `stz_sup_count_triple`, `stz_sup_p0mass_maj`
    - 규칙 키: `rule_left_<j>` (환경의 rules 순서 j. 그 규칙을 포함하는 셀이 S_hat에 남았으면 1.0)
  - `score_law(pop, env, tr, q) -> dict` — 키: `gamma`, `lambda`, `lambda_maj`, `lambda_common`, `lambda_triple`, `tv_in_sz`
  - `score_rows(pop, env, tr, cell_idx) -> dict` — 키: `sz_rec_mass`, `sz_rec_count`, `leak`, `leak_maj`, `leak_common`, `leak_triple`(행 비율), `stz_maj_hit`

- [ ] **Step 1: 테스트 작성** (`tests/test_scorer.py`)

```python
def _case(pop, env, n1=100):
    return make_truth(pop, env, draw_D0(pop, 0), draw_D1(pop, env, 0, n1))

def test_oracle_sets(pop):
    env = "mid__avg__viol"; tr = _case(pop, env); S1 = pop.envs[env].S1
    s = score_set(pop, env, tr, S1)
    assert s["sz_rec_mass"] == 1 and s["stz_sup_count"] == 1 and s["stz_sup_count_triple"] == 1
    s = score_set(pop, env, tr, tr.O)
    assert s["sz_rec_mass"] == 0 and s["stz_sup_count_maj"] == 1
    s = score_set(pop, env, tr, np.ones(65536, bool))
    assert s["sz_rec_mass"] == 1 and s["stz_sup_count"] == 0 and s["stz_sup_p0mass_maj"] == 0
    assert all(s[f"rule_left_{j}"] == 1.0 for j in range(len(pop.envs[env].rules)))
    s = score_set(pop, env, tr, S1 | tr.STZ_triple)
    assert s["stz_sup_count_triple"] == 0 and s["stz_sup_count_common"] == 1

def test_oracle_law(pop):
    env = "frequent__unf__holds"; tr = _case(pop, env); p1 = pop.envs[env].p1
    r = score_law(pop, env, tr, np.where(tr.SZ, p1, 0) / p1[tr.SZ].sum())
    assert abs(r["gamma"] - 1) < 1e-12 and r["lambda"] == 0 and r["tv_in_sz"] < 1e-12

def test_rows(pop):
    env = "rare__fav__holds"; tr = _case(pop, env)
    stz = np.flatnonzero(tr.STZ_maj)[:3]; sz = np.flatnonzero(tr.SZ)[:2]
    r = score_rows(pop, env, tr, np.r_[stz, sz, sz])
    assert abs(r["leak_maj"] - 3 / 7) < 1e-12 and r["leak_common"] == 0
    assert abs(r["sz_rec_count"] - 2 / tr.SZ.sum()) < 1e-12

def test_nan_on_empty_denominators(pop):
    env = "mid__avg__holds"; tr = _case(pop, env)
    assert np.isnan(score_set(pop, env, tr, pop.envs[env].S1)["stz_sup_count_triple"])   # holds: 3변수 STZ 없음
    assert np.isnan(score_law(pop, env, tr, np.where(tr.O, 1.0, 0) / tr.O.sum())["tv_in_sz"])
    tr2 = dataclasses.replace(tr, SZ1=np.zeros(65536, bool))
    assert np.isnan(score_set(pop, env, tr2, pop.envs[env].S1)["sz_rec_mass_1"])

def test_nomaj_recovery(pop):
    env = "mid__unf__holds"; tr = _case(pop, env)
    s = score_set(pop, env, tr, tr.O | (tr.SZ & ~tr.SZ_nomaj))
    assert s["sz_rec_mass_nomaj"] == 0 and s["sz_rec_mass"] < 1
```

- [ ] **Step 2: 실패 확인** — Run: `bash experiments/sync_runtime.sh && RUNPY -m pytest tests/test_scorer.py -v`. Expected: FAIL
- [ ] **Step 3: 구현** (spec 7절 (a)(b)(c) 정의 그대로)
- [ ] **Step 4: 통과 확인** — Expected: 5 passed

---

### Task 7: 채점 (d) 인증 집합과 정확 계산

**Files:** Create `dgp/cert_metrics.py` / Test `tests/test_cert_metrics.py`

**Interfaces**
- Consumes: `Population`, `Truth`, `rule_pid`
- Produces (`dgp/cert_metrics.py`):
  - `TAU = math.log(480 / 0.05)`
  - `lambda_star(n0, n1, r, m=480, alpha=0.05) -> float` — $\tau/\{\kappa\log(1+1/\kappa)\}$, $\kappa = n_0/(n_1 r)$
  - `signal(pi0, n0, n1, r)` — $n_1 r \pi_0/\lambda^*$
  - `exact_cert_prob(pi0, pi1, n0, n1, x_hat) -> np.ndarray` — $(1-\pi_1)^{n_1}$ · `binom.sf(x_hat-1, n0, pi0)`
  - `exact_fwer_bound(pop, env, n0, n1, x_hat, top=10) -> tuple[float, list[int]]` — π0>0, π1>0인 조각만
  - `score_cert(pop, env, tr, C_hat, x_hat, n0, n1, r) -> dict`
    - 대상: 환경의 pair 규칙 4개(3변수 규칙은 대상 아님)
    - 키: `cert_0`–`cert_3`, `power_exact_0`–`power_exact_3`, `s_0`–`s_3`, `any_false`, `n_false`, `all_detected`
  - `clopper_pearson_upper(k, n, level=0.95) -> float` — 단측. `beta.ppf(level, k+1, n-k)`, k=n이면 1
  - `power_curve(n0, n1, r, x_hat, pi0_grid) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: 테스트 작성** (`tests/test_cert_metrics.py`)

```python
def test_lambda_star_matches_spec_table():
    for n1, pistar in [(50, 0.367), (100, 0.184), (200, 0.092), (400, 0.046), (800, 0.023)]:
        assert abs(lambda_star(8000, n1, 0.5) / (n1 * 0.5) - pistar) < 1.5e-3

def test_exact_cert_prob_hand_example():
    assert abs(exact_cert_prob(0.5, 0.0, 10, 5, 8) - 56 / 1024) < 1e-15
    assert abs(exact_cert_prob(0.5, 0.1, 10, 5, 8) - 0.9 ** 5 * 56 / 1024) < 1e-15
    assert exact_cert_prob(0.5, 0.0, 10, 5, 11) == 0.0

def test_clopper_pearson():
    assert abs(clopper_pearson_upper(0, 3000) - (1 - 0.05 ** (1 / 3000))) < 1e-12
    assert clopper_pearson_upper(5, 5) == 1.0

def test_score_cert_oracle(pop):
    env = "frequent__avg__viol"; tr = make_truth(pop, env, draw_D0(pop, 0), draw_D1(pop, env, 0, 400))
    C = np.zeros(480, bool); C[[rule_pid(r) for r in pop.envs[env].rules if r.kind == "pair"]] = True
    r = score_cert(pop, env, tr, C, x_hat=200, n0=8000, n1=400, r=0.5)
    assert r["all_detected"] == 1 and r["any_false"] == 0 and r["n_false"] == 0
    C[np.flatnonzero(pop.envs[env].pi1 > 0)[0]] = True
    r = score_cert(pop, env, tr, C, x_hat=200, n0=8000, n1=400, r=0.5)
    assert r["any_false"] == 1 and r["n_false"] == 1

def test_power_curve_monotone():
    s, pw = power_curve(8000, 200, 0.5, 900, np.linspace(0.01, 0.3, 50))
    assert np.all(np.diff(s) > 0) and np.all(np.diff(pw) >= -1e-15)
```

- [ ] **Step 2: 실패 확인** — Run: `bash experiments/sync_runtime.sh && RUNPY -m pytest tests/test_cert_metrics.py -v`. Expected: FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — Expected: 5 passed

---

### Task 8: 점검 실행기와 진입 스크립트

**Files:** Create `dgp/check.py`, `scripts/run_pilot.py`, `scripts/run_check.py`, `experiments/dgp_pilot.sbatch`, `experiments/dgp_check.sbatch` / Test `tests/test_check.py`

**Interfaces**
- Consumes: Task 1–7
- Produces (`dgp/check.py`). 모두 `pandas.DataFrame`을 반환한다.
  - `population_table(pops) -> DataFrame`
    - 행: (pop_seed, env)
    - 열: `freq`, `overlap`, `assumption`, `eta`, `min_ratio`, `dropped_mass`, `n_S1`, `n_stz_maj`, `n_stz_common`, `n_stz_triple`, `z_mass`, `tv_p0_p1`, 규칙별 `pi0_<j>`·`e_<j>`, `check_minority`의 키 전부(`passed` 포함)
  - `s_table(pops) -> DataFrame`
    - 행: (pop_seed, freq, pair i)
    - 열: `pi0`, `s_50`…`s_800`
  - `sample_table(pops, n1_grid=N1_GRID, sample_seeds=SAMPLE_SEEDS) -> DataFrame`
    - 행: (pop_seed, env, n1, sample_seed)
    - 열: `M0`, `n_SZ`, `sz1_share_count`, `sz1_share_mass`, `nomaj_share_mass`(majority 불가 SZ 질량 / $M_0$), `n_O`, `n_O0`, 규칙별 majority 개수 `N0_<j>`
  - `oracle_table(pops) -> DataFrame` — 설계 8절 채점기 점검. 열: `pop_seed`, `env`, `check`, `value`, `passed`. `value`는 아래 첫 번째 키의 값이다.
    - (pop_seed, env)마다 `tr = make_truth(pop, env, draw_D0(pop, 0), draw_D1(pop, env, 0, 100))`, `x_hat = math.ceil(lambda_star(N0, 100, R_TRUE))`로 다음 9개를 실행한다.

      | check | 호출 | passed 조건 |
      |---|---|---|
      | `set_S1` | `score_set(..., S1)` | `sz_rec_mass == 1`, `stz_sup_count == 1` |
      | `set_O` | `score_set(..., tr.O)` | `sz_rec_mass == 0`, `stz_sup_count == 1` |
      | `set_U` | `score_set(..., 전부 True)` | `sz_rec_mass == 1`, `stz_sup_count == 0` |
      | `law_oracle` | `score_law(..., p1·1[SZ]/M0)` | `gamma`가 1과 1e-12 이내, `lambda == 0`, `tv_in_sz < 1e-12` |
      | `rows_sz` | `score_rows(..., np.flatnonzero(tr.SZ))` | `sz_rec_count == 1`, `leak == 0` |
      | `rows_half` | `score_rows(..., [STZ_maj 첫 셀, SZ 첫 셀])` | `leak == 0.5`, `leak_maj == 0.5` |
      | `cert_exact` | `score_cert(..., C=금지 쌍 4개, x_hat, N0, 100, R_TRUE)` | `all_detected == 1`, `n_false == 0` |
      | `cert_one_false` | 위 C에 `pi1 > 0`인 첫 조각 추가 | `any_false == 1`, `n_false == 1` |
      | `fwer_bound` | `exact_fwer_bound(pop, env, N0, 100, x_hat)` | 값이 유한하고 0 이상(값 기록용) |
    - 모집단과 무관한 3개는 실행마다 한 번, `pop_seed = -1`, `env = "none"`으로 기록한다.

      | check | 호출 | passed 조건 |
      |---|---|---|
      | `exact_hand` | `exact_cert_prob(0.5, 0.0, 10, 5, 8)` | 56/1024와 1e-15 이내 |
      | `cp_hand` | `clopper_pearson_upper(0, 3000)` | $1-0.05^{1/3000}$과 1e-12 이내 |
      | `power_curve_monotone` | `power_curve(8000, 200, 0.5, 900, np.linspace(0.01, 0.3, 50))` | `s` 강증가, `power` 감소 없음(허용 1e-15) |
  - `verdict(pop_t, s_t, sample_t, oracle_t) -> dict`
    - `checks_pass`: pop_t의 `passed`가 모두 참
    - `oracle_pass`
    - `cross_frequent`, `cross_mid`: 해당 수준 모든 쌍이 `s_50 < 1 <= s_800`
    - `overlap_ok`: $n_1$마다 overlap별 `nomaj_share_mass` 평균이 fav에서 0이고, avg보다 unf에서 큼
    - `triple_ok`: `n_stz_triple > 0` ⟺ assumption == viol
    - `stage1_pass`: 앞의 여섯 개 모두 참
  - `write_report(path, pop_t, s_t, sample_t, oracle_t, verdict_d, eta, skipped) -> None`
    - 위치: `reports/stage1_dgp_check.md`
    - 내용: 판정 요약, 축별 평균 표((freq, overlap, assumption, n1)로 평균), 실제 s 표, 거절된 seed 수와 사유별 개수
    - 한국어로 쓰고, 수식은 `$...$`
- `scripts/run_pilot.py`
  - 인자: `--out results/dgp_pilot`, `--help`
  - 동작: η 후보마다 `acceptance_rate`를 계산해 `pilot.json`(`rates`, `reasons`, `chosen_eta`)을 쓴다.
- `scripts/run_check.py`
  - 인자: `--out results/dgp_check`, `--pilot results/dgp_pilot/pilot.json`, `--report reports/stage1_dgp_check.md`, `--quick`(9000번대 seed로 모집단 2개, 표본 seed 2개), `--help`
  - η는 pilot.json의 `chosen_eta`를 쓴다. None이면 멈추고 그 사실을 출력한다.
  - 저장물
    - 수락된 모집단: `pops/pop_<seed>.npz`
    - csv 4개(`populations.csv`, `s_table.csv`, `samples.csv`, `oracle.csv`)
    - `verdict.json`
    - 보고서
  - `StopGeneration`이면 거절 기록을 쓰고 종료 코드 2로 끝난다.
- sbatch 두 개
  - 자원: `--partition=brl --ntasks=1 --qos=normal --exclude=brl1,brl2 --cpus-per-task=1 --mem=8G --time=04:00:00`
  - `--chdir=/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime`
  - 로그: `logs/%x_%j.log`

- [ ] **Step 1: 테스트 작성** (`tests/test_check.py`)

```python
def test_isolated_mode_entry_scripts():
    for s in ("scripts/run_check.py", "scripts/run_pilot.py"):
        r = subprocess.run([sys.executable, "-I", s, "--help"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

def test_quick_run_end_to_end(tmp_path):
    (tmp_path / "pilot.json").write_text(json.dumps({"rates": {}, "reasons": {}, "chosen_eta": 0.5}))
    r = subprocess.run([sys.executable, "-I", "scripts/run_check.py", "--quick",
                        "--out", str(tmp_path / "out"), "--pilot", str(tmp_path / "pilot.json"),
                        "--report", str(tmp_path / "report.md")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for f in ("populations.csv", "s_table.csv", "samples.csv", "oracle.csv", "verdict.json"):
        assert (tmp_path / "out" / f).exists()
    pt = pd.read_csv(tmp_path / "out" / "populations.csv")
    assert pt.groupby("pop_seed").size().eq(18).all() and (pt["min_ratio"] >= 0.5).all()
    oc = pd.read_csv(tmp_path / "out" / "oracle.csv")
    assert len(oc) == 2 * 18 * 9 + 3 and oc["passed"].all()
    assert (tmp_path / "report.md").read_text().strip()

def test_verdict_logic():
    s = pd.DataFrame({"freq": ["frequent", "mid"], "s_50": [0.3, 0.1], "s_800": [5.0, 0.9]})
    smp = pd.DataFrame({"overlap": ["fav", "avg", "unf"], "n1": [50] * 3, "nomaj_share_mass": [0.0, 0.2, 0.4]})
    pt = pd.DataFrame({"passed": [True, True], "assumption": ["holds", "viol"], "n_stz_triple": [0, 10]})
    v = verdict(pt, s, smp, pd.DataFrame({"passed": [True]}))
    assert v["cross_frequent"] and not v["cross_mid"] and v["overlap_ok"] and v["triple_ok"]
    assert not v["stage1_pass"]
```

- [ ] **Step 2: 실패 확인** — Run: `bash experiments/sync_runtime.sh && RUNPY -m pytest tests/test_check.py -v`. Expected: FAIL
- [ ] **Step 3: 구현** — 진입 스크립트는 첫 줄에서 저장소 루트를 `sys.path`에 넣는다.
- [ ] **Step 4: 통과 확인** — Expected: 3 passed
- [ ] **Step 5: 전체 테스트** — Run: `bash experiments/sync_runtime.sh && RUNPY -m pytest -v`. Expected: 42 passed

---

## Codex 기초 검토 2회 (구현 끝난 뒤, 실험 실행 전)

1. **1차: spec 대조.** spec 1–8절을 코드와 한 줄씩 맞춘다. 특히 다음을 본다.
   - 번호: 열 순서, 셀 번호, 조각 번호
   - 0 쌍: 두 종류의 위치(부모 쌍만)와 선택 규칙
   - 규칙 선택: 금지 쌍과 3변수 규칙의 조건(범위, $e$ 상한, $c' \ne a'$, 동점 처리)
   - 영역 $R$, $Z$, $S_1$
   - 질량 이동: 부모 좌표만 바꾸고 대상이 $R$ 안
   - 섞기 식과 $p_1(Z) = \omega\gamma$
   - 난수 entropy, $D_1$의 포함 관계와 환경 간 같은 난수
2. **2차: 수치 경계.** 다음을 본다.
   - 누적합 보정
   - 버린 질량이 있을 때 재정규화
   - 분모 0 처리
   - `binom.sf`의 `x_hat-1` 인자
   - Clopper–Pearson이 단측인지
   - `-I` 모드 import
   - 3변수 STZ 정의(pair 규칙을 포함하지 않는 셀)

검토마다 찾은 문제와 고친 내용을 `reports/stage1_codex_review.md`에 적는다. 끝나면 변경 파일 목록, 테스트 결과 요약, 검토 기록 위치를 보고한다.

## 실행 (Codex 범위 밖: Claude 검토 3회와 사용자 승인 뒤)

1. `bash experiments/sync_runtime.sh && sbatch experiments/dgp_pilot.sbatch` → `results/dgp_pilot/pilot.json`의 `chosen_eta` 확인
2. `sbatch experiments/dgp_check.sbatch` → `verdict.json`의 `stage1_pass` 확인
3. runtime의 `results/`와 `reports/stage1_dgp_check.md`를 프로젝트 루트로 복사하고 해석을 덧붙인다
