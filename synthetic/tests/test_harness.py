import numpy as np

from dgp.harness import (
    CELLS64, check_cert_reconstruction, evaluate_rows, make_pop_adapter, run_support,
)
from dgp.sampling import Split, make_split, reference_rows
from dgp.scorer import score_rows
from dgp.space import CELL_PATTERNS, cell_index, pattern_counts
from dgp.truth import make_truth


def test_support_nesting(pop):
    # V21
    s = make_split(pop, "frequent__fav", 400, 0)
    SA, A, _ = run_support("A", s)
    SB, B, _ = run_support("B", s)
    SC, C, _ = run_support("C", s)
    assert (SA <= SC).all() and (SC <= SB).all()
    assert B.n_certified == 0
    N1 = pattern_counts(s.D1_idx)
    assert np.array_equal(SA, (N1[CELL_PATTERNS] > 0).all(axis=1))


def test_cert_reconstruction(pop):
    # V22, Review Focus 5: 열 0의 수준 0이 minority에 한 번도 없는 표본도 넣는다.
    s = make_split(pop, "frequent__fav", 800, 0)
    D1 = s.D1.copy()
    D1[D1[:, 0] == 0, 0] = 1
    s2 = Split(s.D0_idx, cell_index(D1), s.test_X, s.test_y)
    for split in (s, s2):
        for m in ("A", "B", "C"):
            _, sup, used = run_support(m, split)
            assert check_cert_reconstruction(sup, used, split.D1_idx)


def test_evaluate_agrees(pop):
    # V23
    env = "frequent__avg"
    s = make_split(pop, env, 400, 0)
    t = make_truth(pop, env, s)
    adapter = make_pop_adapter(pop, env, reference_rows(pop, env))
    G = np.random.default_rng(0).integers(0, 4, (500, 8))
    ev, ours = evaluate_rows(G, s, adapter), score_rows(t, cell_index(G))
    assert abs(ev["recovery"] - ours["rec_mass"]) < 1e-12 and abs(ev["leak"] - ours["leak"]) < 1e-12
    G2 = CELLS64[t.SZ]
    ev2 = evaluate_rows(G2, s, adapter)
    assert abs(ev2["recovery"] - 1) < 1e-12 and ev2["leak"] == 0
