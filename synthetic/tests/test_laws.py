import numpy as np

from dgp.space import CELLS, PARENT_PAIRS
from dgp.laws import apply_zeros, choose_parent_zeros, draw_tree_tables, tree_law

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
