"""C2–C3: 트리 법칙과 부모 쌍 위의 0 (설계 2절)."""

from dataclasses import dataclass
from itertools import product

import numpy as np

from dgp.space import (
    CELLS, CHAIN_EDGES, K, LEAF_EDGES, PARENT_PAIRS,
    cells_containing, pattern_id, pattern_probs,
)


@dataclass
class TreeTables:
    # C2, D2.1: 조건부 표의 행은 부모 값, 열은 자식 값.
    root: np.ndarray
    cond: dict[tuple[int, int], np.ndarray]


ParentZero = tuple[int, int, int, int]


def draw_tree_tables(rng, eta: float) -> TreeTables:
    # C2, D2.1: 뿌리 → 사슬 간선 → 잎 간선. 각 행을 독립 추출.
    alpha = np.full(K, eta)
    root = rng.dirichlet(alpha)
    cond = {
        edge: np.stack([rng.dirichlet(alpha) for _ in range(K)])
        for edge in CHAIN_EDGES + LEAF_EDGES
    }
    return TreeTables(root, cond)


def tree_law(t: TreeTables) -> np.ndarray:
    # C2, D2.1: 셀을 순회하지 않고 표를 배열 인덱싱해 곱한다.
    factors = np.stack(
        [t.cond[j, l][CELLS[:, j], CELLS[:, l]] for j, l in CHAIN_EDGES + LEAF_EDGES]
    )
    return t.root[CELLS[:, 0]] * np.prod(factors, axis=0)


def choose_parent_zeros(
    p_tree, rng, max_mass: float = 0.05, n_major: int = 2
) -> tuple[list[ParentZero], list[ParentZero]] | None:
    # C3, D3.1–D3.3, Q1: 삽입 전 확률, 사전순 후보에서 균등 추출.
    pi_tree = pattern_probs(p_tree)
    shared = []
    candidates_by_pair = {}
    for j, l in PARENT_PAIRS:
        candidates = [
            (j, l, v, w)
            for v, w in product(range(K), repeat=2)
            if pi_tree[pattern_id((j, l), (v, w))] <= max_mass
        ]
        if not candidates:
            return None
        shared.append(candidates[int(rng.choice(len(candidates)))])
        candidates_by_pair[j, l] = candidates
    major = []
    selected = rng.choice(len(PARENT_PAIRS), size=n_major, replace=False)
    for rank in selected:
        pair = PARENT_PAIRS[int(rank)]
        candidates = [z for z in candidates_by_pair[pair] if z != shared[int(rank)]]
        if not candidates:
            return None
        major.append(candidates[int(rng.choice(len(candidates)))])
    return shared, major


def apply_zeros(p, zeros: list[ParentZero]) -> np.ndarray:
    # C3, D3.4: 지정된 셀을 0으로 만든 뒤 한 번만 정규화.
    out = p.copy()
    for j, l, v, w in zeros:
        out[cells_containing(pattern_id((j, l), (v, w)))] = 0.0
    return out / out.sum()
