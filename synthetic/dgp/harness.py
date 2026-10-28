"""C22–C24: 박사형 코드 연결. 모집단 연결부, A/B/C support, 인증 집합, BOMiS 실행 (설계 6.2–6.5)."""

import numpy as np

from dgp.space import CELLS, K, cell_index, pattern_counts, pattern_id
from dgp.vendor import load_bomis

load_bomis()
from bomis import BOMiS, Support  # noqa: E402
from bomis.metrics import evaluate  # noqa: E402

KS = [K] * 8
CELLS64 = CELLS.astype(np.int64)


class PopAdapter:
    # C22, D22.1: evaluate가 쓰는 K, contains, log_prob, ref만 구현한다.
    def __init__(self, p0: np.ndarray, p1: np.ndarray, ref: dict[int, np.ndarray]):
        self.K = list(KS)
        self._p = {0: p0, 1: p1}
        self.ref = ref

    def contains(self, X, c: int) -> np.ndarray:
        return self._p[c][cell_index(X)] > 0

    def log_prob(self, X, c: int) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(self._p[c][cell_index(X)])


def make_pop_adapter(pop, env: str, ref: dict[int, np.ndarray]) -> PopAdapter:
    return PopAdapter(pop.p0, pop.envs[env].p1, ref)


def split_dict(split) -> dict:
    return {"D0": split.D0, "D1": split.D1, "test_X": split.test_X, "test_y": split.test_y}


def run_support(method: str, split):
    # C23, D23.1: 박사형 Support를 인자만 바꿔 부른다. 돌려주는 값: (S_hat, support, D0_idx_used).
    D0, D1 = split.D0, split.D1
    if method == "A":
        support, used = Support(D0[:0], D1, KS), split.D0_idx[:0]
    elif method == "B":
        support, used = Support(D0, D1, KS, r=0.0), split.D0_idx
    elif method == "C":
        support, used = Support(D0, D1, KS), split.D0_idx
    else:
        raise ValueError(method)
    return support.contains(CELLS64), support, used


def cert_set(support, D0_idx: np.ndarray, D1_idx: np.ndarray) -> np.ndarray:
    # C23, D23.2: Support는 조각별 인증을 저장하지 않으므로 개수로 다시 만든다.
    N0, N1 = pattern_counts(D0_idx), pattern_counts(D1_idx)
    return (N1 == 0) & (N0 >= support.x_star)


def check_cert_reconstruction(support, D0_idx: np.ndarray, D1_idx: np.ndarray) -> bool:
    # C23, D23.2: 박사형 허용 규칙을 표마다 다시 계산해 support.admit과 비교한다.
    N0, N1 = pattern_counts(D0_idx), pattern_counts(D1_idx)
    C_hat = (N1 == 0) & (N0 >= support.x_star)
    for B, table in support.admit.items():
        if len(B) == 1:
            pids = np.array([pattern_id(B, (v,)) for v in range(K)])
        else:
            pids = np.array([pattern_id(B, (v, w)) for v in range(K) for w in range(K)])
        n0, n1, cert = N0[pids], N1[pids], C_hat[pids]
        pooled = (n0 > 0) & ((n1 == 1).any() or len(B) > 1)
        admit = ((n1 > 0) | pooled) & ~cert
        if not np.array_equal(admit, table.reshape(-1)):
            return False
    return int(C_hat.sum()) == int(support.n_certified)


def run_bomis(split, seed: int, **kw):
    # C24, D24.1: 기본 인자. 생성 행 수는 n0 - n1.
    D0, D1 = split.D0, split.D1
    model = BOMiS(seed=seed, **kw).fit(D0, D1, KS)
    G = model.sample(len(D0) - len(D1))
    return model, np.asarray(G, dtype=np.int64)


def evaluate_rows(G: np.ndarray, split, adapter: PopAdapter) -> dict:
    # C22: 박사형 evaluate 그대로.
    return evaluate(np.asarray(G, dtype=np.int64), split_dict(split), adapter)
