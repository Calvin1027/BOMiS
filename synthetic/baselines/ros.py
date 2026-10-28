"""C26: random oversampling. minority 행을 복원 추출한다."""

import numpy as np


def ros(D1: np.ndarray, n: int, rng) -> np.ndarray:
    # C26, D26.1: rng는 default_rng([pop_seed, sample_seed, 30]).
    return np.asarray(D1, dtype=np.int64)[rng.integers(0, len(D1), n)]
