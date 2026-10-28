import math

import numpy as np

from dgp.cert_metrics import clopper_pearson_upper, exact_cert_prob
from dgp.sampling import make_split
from dgp.scorer import score_law, score_rows, score_set
from dgp.space import N_CELLS
from dgp.truth import make_truth


def _truth(pop):
    return make_truth(pop, "frequent__avg", make_split(pop, "frequent__avg", 400, 0))


def test_score_set_extremes(pop):
    # V14
    t = _truth(pop)
    a, b, c = score_set(t, t.S1), score_set(t, t.O.copy()), score_set(t, np.ones(N_CELLS, bool))
    assert (a["sz_rec_mass"], a["stz_sup_count"], a["stz_sup_maj"]) == (1.0, 1.0, 1.0)
    assert (b["sz_rec_mass"], b["stz_sup_count"]) == (0.0, 1.0)
    assert (c["sz_rec_mass"], c["stz_sup_count"], c["stz_sup_maj"]) == (1.0, 0.0, 0.0)


def test_score_rows_hand(pop):
    # V15
    t = _truth(pop)
    stz, sz = np.flatnonzero(t.STZ)[:3], np.flatnonzero(t.SZ)[:2]
    out = score_rows(t, np.r_[stz, sz, sz])
    assert abs(out["leak"] - 3 / 7) < 1e-12
    assert abs(out["rec_count"] - 2 / t.SZ.sum()) < 1e-12
    assert abs(out["rec_mass"] - t.p1[sz].sum() / t.M0) < 1e-12


def test_score_law(pop):
    # V16
    t = _truth(pop)
    q = np.where(t.SZ, t.p1, 0.0) / t.M0
    out = score_law(t, q)
    assert abs(out["Gamma"] - 1) < 1e-12 and out["Lambda"] == 0 and out["tv_sz"] < 1e-12
    assert math.isnan(score_law(t, np.where(t.SZ, 0.0, 1.0))["tv_sz"])


def test_exact_cert_prob_hand():
    # V17
    assert abs(exact_cert_prob(0.5, 0.0, 10, 5, 8) - 56 / 1024) < 1e-15
    assert abs(exact_cert_prob(0.5, 0.1, 10, 5, 8) - 0.9 ** 5 * 56 / 1024) < 1e-15


def test_clopper_pearson():
    # V18
    assert abs(clopper_pearson_upper(0, 3000) - (1 - 0.05 ** (1 / 3000))) < 1e-12
    assert clopper_pearson_upper(5, 5) == 1.0
