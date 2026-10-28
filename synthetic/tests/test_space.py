import numpy as np

from dgp.space import (
    CELLS, CELL_PATTERNS, PAT_COLS, PAT_VALS, cell_index,
    cells_containing, pattern_counts, pattern_id, pattern_probs,
)

def test_cell_index_order():
    assert cell_index(np.zeros((1, 8), int))[0] == 0
    assert cell_index(np.full((1, 8), 3))[0] == 65535
    assert cell_index(np.array([[0,0,0,0,0,0,0,1]]))[0] == 1
    assert cell_index(np.array([[1,0,0,0,0,0,0,0]]))[0] == 16384
    assert np.array_equal(cell_index(CELLS.astype(int)), np.arange(65536))

def test_pattern_ids():
    assert pattern_id((0,), (0,)) == 0 and pattern_id((7,), (3,)) == 31
    assert pattern_id((0, 1), (0, 0)) == 32 and pattern_id((6, 7), (3, 3)) == 479
    assert len({tuple(r) for r in np.c_[PAT_COLS, PAT_VALS]}) == 480

def test_cell_patterns_and_probs():
    assert CELL_PATTERNS.shape == (65536, 36)
    assert all(len(set(row)) == 36 for row in CELL_PATTERNS[::997])
    pp = pattern_probs(np.full(65536, 1 / 65536))
    assert np.allclose(pp[:32], 0.25) and np.allclose(pp[32:], 1 / 16)

def test_pattern_counts_matches_bruteforce():
    idx = np.random.default_rng(0).integers(0, 65536, 500)
    cnt = pattern_counts(idx)
    for pid in (0, 17, 32, 200, 479):
        assert cnt[pid] == cells_containing(pid)[idx].sum()
