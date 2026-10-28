"""C14: 정답 (설계 5절). STZ는 하나로만 정의한다."""

from dataclasses import dataclass

import numpy as np

from dgp.space import N_CELLS, pattern_counts, pattern_probs


@dataclass
class Truth:
    env: str
    p0: np.ndarray
    p1: np.ndarray
    S1: np.ndarray
    O: np.ndarray
    O0: np.ndarray
    SZ: np.ndarray
    SZ1: np.ndarray
    SZ2: np.ndarray
    SZ_nomaj: np.ndarray
    STZ: np.ndarray
    M0: float
    N0: np.ndarray
    N1: np.ndarray
    pi0: np.ndarray
    pi1: np.ndarray
    rules: list
    n0: int
    n1: int


def _present(idx: np.ndarray) -> np.ndarray:
    out = np.zeros(N_CELLS, dtype=bool)
    out[np.unique(idx)] = True
    return out


def make_truth(pop, env: str, split) -> Truth:
    # C14, D14.1–D14.2
    e = pop.envs[env]
    O, O0 = _present(split.D1_idx), _present(split.D0_idx)
    SZ = e.S1 & ~O
    return Truth(
        env=env, p0=pop.p0, p1=e.p1, S1=e.S1, O=O, O0=O0,
        SZ=SZ, SZ1=SZ & O0, SZ2=SZ & ~O0, SZ_nomaj=SZ & (pop.p0 == 0), STZ=~e.S1,
        M0=float(e.p1[SZ].sum()),
        N0=pattern_counts(split.D0_idx), N1=pattern_counts(split.D1_idx),
        pi0=pattern_probs(pop.p0), pi1=pattern_probs(e.p1),
        rules=list(pop.rules[e.level]), n0=len(split.D0_idx), n1=len(split.D1_idx),
    )
