import math

import numpy as np
import pytest

from dgp.bomis_diag import _tv_to_sz, bomis_diagnostics, decomposition, true_shell_law
from dgp.sampling import make_split
from dgp.truth import make_truth


def _truth(pop, n1=400):
    s = make_split(pop, "frequent__fav", n1, 0)
    return s, make_truth(pop, "frequent__fav", s)


def test_decomposition_hand(pop):
    # V24
    _, t = _truth(pop)
    order = np.argsort(np.where(t.SZ, t.p1, -1.0))[::-1]
    top5, obs = order[:5], np.flatnonzero(t.O)[:10]
    d = decomposition(t, t.S1, np.r_[top5, obs, top5])
    assert d["L_prime"] == 5 and abs(d["R4"] - d["R2"]) < 1e-12 and abs(d["R1"] - 1) < 1e-12
    allsz = np.flatnonzero(t.SZ)
    d2 = decomposition(t, t.S1, allsz)
    assert abs(d2["R1"] - 1) < 1e-12 and abs(d2["R2"] - 1) < 1e-12 and abs(d2["R4"] - 1) < 1e-12


def test_decomposition_order_random(pop):
    _, t = _truth(pop)
    rng = np.random.default_rng(1)
    S_hat = t.S1 & (rng.random(t.S1.size) < 0.7) | (rng.random(t.S1.size) < 0.05)
    G = rng.choice(np.flatnonzero(S_hat), 3000)
    d = decomposition(t, S_hat, G)
    assert d["R4"] <= d["R2"] + 1e-12 <= d["R1"] + 2e-12 and d["R1"] <= 1 + 1e-12


def test_diag_no_novel_rows(pop):
    # Review Focus 2
    _, t = _truth(pop)
    d = decomposition(t, t.S1, np.flatnonzero(t.O))
    assert d["L_prime"] == 0 and d["R2"] == 0 and d["R4"] == 0
    assert math.isnan(_tv_to_sz(np.zeros(t.S1.size), t))


def test_true_shell_law_sums_to_one(pop):
    _, t = _truth(pop)
    law = true_shell_law(t)
    assert abs(sum(law.values()) - 1) < 1e-12 and min(law) >= 1


@pytest.mark.slow
def test_bomis_diag_smoke(pop):
    # V25
    from dgp.harness import run_bomis
    s, t = _truth(pop, n1=100)
    model, G = run_bomis(s, 0, steps=250)
    state = model.rng.bit_generator.state
    d = bomis_diagnostics(model, t, s, G)
    assert model.rng.bit_generator.state == state
    assert d["R4"] <= d["R2"] + 1e-12 <= d["R1"] + 2e-12 <= 1 + 3e-12
    assert abs(d["Qu_total"] - 1) < 1e-4 and d["Q_total"] <= 1 + 1e-4
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()})
