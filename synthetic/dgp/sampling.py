"""C13: 표본 추출 (설계 4.8절)."""

from dataclasses import dataclass

import numpy as np

from dgp.space import CELLS, N_CELLS

N0, N1_MAX, N_TEST, N_REF = 8000, 800, 2000, 50000


def draw_cells(p: np.ndarray, u: np.ndarray) -> np.ndarray:
    # C13, D13.1: 고정 셀 순서의 누적분포에 균등 난수를 넣는다. 확률 0인 셀은 나오지 않는다.
    c = np.cumsum(p)
    c = c / c[-1]
    return np.minimum(np.searchsorted(c, u, side="right"), N_CELLS - 1).astype(np.int64)


def rows(idx: np.ndarray) -> np.ndarray:
    return CELLS[idx].astype(np.int64)


@dataclass
class Split:
    D0_idx: np.ndarray
    D1_idx: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray

    @property
    def D0(self) -> np.ndarray:
        return rows(self.D0_idx)

    @property
    def D1(self) -> np.ndarray:
        return rows(self.D1_idx)


def _u(n: int, *entropy: int) -> np.ndarray:
    return np.random.default_rng(list(entropy)).random(n)


def make_split(pop, env: str, n1: int, sample_seed: int, n0: int = N0, n_test: int = N_TEST) -> Split:
    # C13, D13.2–D13.3. D1은 800개 균등 난수의 앞 n1개라 n1 사이에 포함 관계다.
    s, p0, p1 = pop.seed, pop.p0, pop.envs[env].p1
    D0_idx = draw_cells(p0, _u(n0, s, sample_seed, 10))
    D1_idx = draw_cells(p1, _u(N1_MAX, s, sample_seed, 11)[:n1])
    T0 = draw_cells(p0, _u(n_test, s, sample_seed, 12))
    T1 = draw_cells(p1, _u(n_test, s, sample_seed, 13))
    test_X = np.vstack([rows(T1), rows(T0)])  # 박사형 known_split처럼 minority 먼저
    test_y = np.r_[np.ones(n_test), np.zeros(n_test)].astype(int)
    return Split(D0_idx, D1_idx, test_X, test_y)


def reference_rows(pop, env: str, n_ref: int = N_REF) -> dict[int, np.ndarray]:
    # C13, D13.2: 참조 표본은 모집단 단위 난수.
    return {
        0: rows(draw_cells(pop.p0, _u(n_ref, pop.seed, 14))),
        1: rows(draw_cells(pop.envs[env].p1, _u(n_ref, pop.seed, 15))),
    }
