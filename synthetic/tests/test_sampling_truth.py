import numpy as np

from dgp.sampling import draw_cells, make_split
from dgp.space import N_CELLS
from dgp.truth import make_truth


def test_draw_cells_zero_mass():
    # V11
    p = np.ones(N_CELLS)
    p[[0, 5, N_CELLS - 1]] = 0.0
    u = np.array([0.0, 5 / (N_CELLS - 3), 0.5, np.nextafter(1.0, 0.0)])
    idx = draw_cells(p, u)
    assert not np.isin(idx, [0, 5, N_CELLS - 1]).any()


def test_nested_and_common_random(pop):
    # V12
    a = make_split(pop, "frequent__fav", 400, 0)
    b = make_split(pop, "frequent__fav", 800, 0)
    c = make_split(pop, "frequent__avg", 400, 0)
    assert np.array_equal(a.D1_idx, b.D1_idx[:400])
    assert np.array_equal(a.D0_idx, b.D0_idx) and np.array_equal(a.D0_idx, c.D0_idx)
    assert len(a.D0_idx) == 8000 and len(b.D1_idx) == 800


def test_truth_partition(pop):
    # V13
    for env in ("frequent__fav", "mid__unf"):
        s = make_split(pop, env, 400, 1)
        t = make_truth(pop, env, s)
        assert np.array_equal(t.SZ1 | t.SZ2, t.SZ) and not (t.SZ1 & t.SZ2).any()
        assert abs(t.M0 - (1 - t.p1[t.O].sum())) < 1e-12
        assert np.array_equal(t.STZ, ~t.S1)
        if env.endswith("fav"):
            assert not t.SZ_nomaj.any()


def test_split_format(pop):
    s = make_split(pop, "frequent__fav", 400, 0)
    assert s.test_X.shape == (4000, 8) and s.test_y.sum() == 2000 and s.test_y[:2000].all()
    assert s.D0.dtype == np.int64 and s.D1.shape == (400, 8)
