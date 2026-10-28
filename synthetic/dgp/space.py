"""C1: 고정 셀 공간과 1·2차 조각 번호 (설계 1절)."""

from itertools import combinations, product

import numpy as np

D = 8
K = 4
N_CELLS = K**D
N_PATTERNS = 480
PARENTS = (0, 1, 2, 3)
LEAF_OF = {i: 4 + i for i in PARENTS}
CHAIN_EDGES = ((0, 1), (1, 2), (2, 3))
LEAF_EDGES = tuple(LEAF_OF.items())
PARENT_PAIRS = tuple(combinations(PARENTS, 2))
_PAIRS = tuple(combinations(range(D), 2))
_WEIGHTS = K ** np.arange(D - 1, -1, -1, dtype=np.int64)
# C1, D1.1: 열 0이 최상위 자리.
CELLS = ((np.arange(N_CELLS)[:, None] // _WEIGHTS) % K).astype(np.int8)
PAT_COLS = np.full((N_PATTERNS, 2), -1, dtype=np.int8)
PAT_VALS = np.full((N_PATTERNS, 2), -1, dtype=np.int8)
for _j, _v in product(range(D), range(K)):
    PAT_COLS[4 * _j + _v, 0] = _j
    PAT_VALS[4 * _j + _v, 0] = _v
for _rank, (_j, _l) in enumerate(_PAIRS):
    for _v, _w in product(range(K), repeat=2):
        _pid = 32 + 16 * _rank + 4 * _v + _w
        PAT_COLS[_pid] = (_j, _l)
        PAT_VALS[_pid] = (_v, _w)
# C1, D1.2, D1.3: int8 덧셈의 넘침을 피하고 조각 번호를 int16에 저장.
CELL_PATTERNS = np.column_stack(
    [4 * j + CELLS[:, j].astype(np.int16) for j in range(D)]
    + [
        32 + 16 * rank + 4 * CELLS[:, j].astype(np.int16) + CELLS[:, l]
        for rank, (j, l) in enumerate(_PAIRS)
    ]
).astype(np.int16)


def cell_index(rows: np.ndarray) -> np.ndarray:
    # C1, D1.1: 계획 T1.
    return np.asarray(rows, dtype=np.int64) @ _WEIGHTS


def pattern_id(cols: tuple[int, ...], vals: tuple[int, ...]) -> int:
    # C1, D1.2: 계획 T1. cols는 오름차순.
    if len(cols) == 1:
        return 4 * cols[0] + vals[0]
    return 32 + 16 * _PAIRS.index(tuple(cols)) + 4 * vals[0] + vals[1]


def pattern_probs(p: np.ndarray) -> np.ndarray:
    # C1, D1.3: 각 셀의 36개 조각에 같은 셀 질량을 더한다.
    return np.bincount(
        CELL_PATTERNS.ravel(), weights=np.repeat(p, 36), minlength=N_PATTERNS
    )


def pattern_counts(cell_idx: np.ndarray) -> np.ndarray:
    # C1, D1.3: 중복 표본도 각각 센다.
    return np.bincount(
        CELL_PATTERNS[cell_idx].ravel(), minlength=N_PATTERNS
    ).astype(np.int64)


def cells_containing(pid: int) -> np.ndarray:
    # C1, D1.2: 단일 조각의 두 번째 좌표(-1)는 쓰지 않는다.
    cols = PAT_COLS[pid]
    vals = PAT_VALS[pid]
    active = cols >= 0
    return (CELLS[:, cols[active]] == vals[active]).all(axis=1)
