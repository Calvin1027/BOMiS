"""C4, C6–C9: 금지 쌍 심기, 영역, 질량 이동, 섞기와 정확 점검 (설계 4.3–4.5, 8절)."""

import copy
from itertools import product
from typing import NamedTuple

import numpy as np

from dgp.laws import ParentZero, TreeTables, apply_zeros, tree_law
from dgp.space import (
    CELLS, CELL_PATTERNS, K, N_CELLS, PARENTS,
    cells_containing, pattern_id, pattern_probs,
)

# C4, D4.1: 수준별 목표 pi0와 심기 조건 (0.06 / 0.2).
T_LEVELS = {"frequent": 0.06, "mid": 0.03, "rare": 0.005}
P_MIN = 0.3
E_MAX = 0.2
OMEGA = 0.2
GAMMAS = {"fav": 0.0, "avg": 0.5, "unf": 1.0}
R_TRUE = 0.5


class Rule(NamedTuple):
    # C4, D4.6: 금지 쌍 (P_i = a', L_i = a). move_col은 부모 열 i.
    kind: str
    cols: tuple[int, ...]
    vals: tuple[int, ...]
    move_col: int
    e: float
    pi0: float


def rule_mask(rule: Rule) -> np.ndarray:
    # C4: 규칙을 포함하는 셀.
    return (CELLS[:, rule.cols] == rule.vals).all(axis=1)


def rule_pid(rule: Rule) -> int:
    # C4: 금지 쌍의 2차 조각 번호.
    return pattern_id(rule.cols, rule.vals)


def _planted_row(p_parent: float, perm) -> np.ndarray:
    # C4, D4.5: perm은 (frequent, mid, rare) 순서의 잎 값. 남은 한 값이 나머지 질량을 받는다.
    row = np.zeros(K)
    for t, a in zip(T_LEVELS.values(), perm):
        row[int(a)] = t / p_parent
    rest = [v for v in range(K) if v not in {int(a) for a in perm}][0]
    row[rest] = 1.0 - row.sum()
    return row


def plant_pairs(
    t: TreeTables, zeros: list[ParentZero], rng
) -> tuple[TreeTables, dict[str, list[Rule]]] | str:
    # C4, D4.2–D4.6. rng = default_rng([pop_seed, 3]).
    pi = pattern_probs(apply_zeros(tree_law(t), zeros))  # D4.2: 0을 넣은 뒤의 부모 확률
    perms, a_prime, p_parent = {}, {}, {}
    for i in PARENTS:  # D4.3: 거절 판정 전에 네 순열을 모두 뽑는다
        perms[i] = rng.permutation(K)[:3]
        probs = [pi[pattern_id((i,), (v,))] for v in range(K)]
        a_prime[i] = int(np.argmax(probs))
        p_parent[i] = float(probs[a_prime[i]])
    for i in PARENTS:  # D4.4
        if p_parent[i] < P_MIN:
            return f"no_plant:{i}"
    planted = copy.deepcopy(t)
    rules = {lv: [] for lv in T_LEVELS}
    for i in PARENTS:
        planted.cond[(i, 4 + i)][a_prime[i]] = _planted_row(p_parent[i], perms[i])
        for (lv, target), a in zip(T_LEVELS.items(), perms[i]):
            rules[lv].append(Rule(
                "pair", (i, 4 + i), (a_prime[i], int(a)), i,
                float(target / p_parent[i]), float(target),
            ))
    return planted, rules


def regions(shared, major, rules, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # C6, D6.1: gamma=0이어도 Z 자체는 계산한다.
    shared_mask = np.zeros(N_CELLS, dtype=bool)
    major_mask = np.zeros(N_CELLS, dtype=bool)
    forbidden = np.zeros(N_CELLS, dtype=bool)
    for j, l, v, w in shared:
        shared_mask |= cells_containing(pattern_id((j, l), (v, w)))
    for j, l, v, w in major:
        major_mask |= cells_containing(pattern_id((j, l), (v, w)))
    for rule in rules:
        forbidden |= rule_mask(rule)
    R = ~shared_mask & ~major_mask & ~forbidden
    Z = major_mask & ~shared_mask & ~forbidden
    return R, Z, R.copy() if gamma == 0 else R | Z


def move_mass(p0, R, rules) -> tuple[np.ndarray, float]:
    # C7, D7.1: 원래 셀의 규칙으로 이동 좌표를 확정하고 동시에 바꾼다.
    moving = np.zeros(N_CELLS, dtype=np.uint8)
    for rule in rules:
        moving[rule_mask(rule)] |= np.uint8(1 << rule.move_col)
    out = np.where(R, p0, 0.0)
    dropped = 0.0
    for bits in range(1, 1 << len(PARENTS)):
        source = np.flatnonzero((p0 > 0) & (moving == bits))
        if source.size == 0:
            continue
        columns = [j for j in PARENTS if bits & (1 << j)]
        # 각 좌표의 원래 값 이외 3개 값을 모두 나열한다.
        shifts = np.asarray(list(product((1, 2, 3), repeat=len(columns))))
        destinations = np.broadcast_to(source[:, None], (source.size, len(shifts))).copy()
        for axis, col in enumerate(columns):
            old = CELLS[source, col].astype(np.int64)[:, None]
            new = (old + shifts[None, :, axis]) % K
            destinations += (new - old) * K ** (7 - col)
        valid = R[destinations]
        counts = valid.sum(axis=1)
        dropped += float(p0[source[counts == 0]].sum())
        shares = np.divide(
            p0[source], counts, out=np.zeros(source.size), where=counts > 0
        )
        out += np.bincount(
            destinations[valid],
            weights=np.broadcast_to(shares[:, None], destinations.shape)[valid],
            minlength=N_CELLS,
        )
    return out / out.sum(), dropped


def mix(p1base, h, R, Z, gamma: float, omega: float = OMEGA) -> np.ndarray:
    # C8, D8.1: gamma=0에서는 h(Z)로 나누지 않는다. 추가 정규화 없음.
    out = (1 - omega) * p1base + omega * (1 - gamma) * np.where(R, h, 0.0) / h[R].sum()
    if gamma != 0:
        out += omega * gamma * np.where(Z, h, 0.0) / h[Z].sum()
    return out


def pairwise_feasible_cells(pi1: np.ndarray) -> np.ndarray:
    # C9, D9.2: 단일 8개와 쌍 28개가 모두 가능한 셀.
    return (pi1[CELL_PATTERNS] > 0).all(axis=1)


def check_minority(p0, p1, S1, Z, rules, gamma: float, r: float = R_TRUE) -> dict:
    # C9, D9.1–D9.2: 설계 8절 1–5. local support는 F2 == S1.
    pi0, pi1 = pattern_probs(p0), pattern_probs(p1)
    possible = (pi0 > 0) & (pi1 > 0)
    min_ratio = float((pi1[possible] / pi0[possible]).min()) if possible.any() else float("nan")
    expected = sorted(rule_pid(f) for f in rules)
    result = {
        "sum_ok": bool(abs(p0.sum() - 1) < 1e-12 and abs(p1.sum() - 1) < 1e-12),
        "support_ok": bool(np.array_equal(p1 > 0, S1)),
        "local_support_ok": bool(np.array_equal(pairwise_feasible_cells(pi1), S1)),
        "infeasible_ok": np.flatnonzero((pi0 > 0) & (pi1 == 0)).tolist() == expected,
        "z_mass_ok": bool(abs(p1[Z].sum() - OMEGA * gamma) < 1e-12),
        "min_ratio": min_ratio,
        "floor_ok": bool(min_ratio >= r),
    }
    result["passed"] = all(value for key, value in result.items() if key.endswith("_ok"))
    return result
