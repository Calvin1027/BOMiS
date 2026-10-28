import copy

import numpy as np

from dgp.laws import TreeTables, draw_tree_tables
from dgp.minority import (
    T_LEVELS, _planted_row, check_minority, move_mass, plant_pairs, regions, rule_pid,
)
from dgp.space import pattern_id, pattern_probs


def test_planted_row_hand():
    # V3
    row = _planted_row(0.4, (2, 0, 3))
    assert np.allclose(row, [0.075, 0.7625, 0.15, 0.0125], atol=1e-15)


def test_planted_pi0_exact(pop):
    # V4
    pi0 = pattern_probs(pop.p0)
    bound = {"frequent": 0.2, "mid": 0.1, "rare": 1 / 60}
    for lv, rules in pop.rules.items():
        assert [r.move_col for r in rules] == [0, 1, 2, 3]
        for r in rules:
            assert abs(pi0[rule_pid(r)] - T_LEVELS[lv]) < 1e-12
            p_parent = pi0[pattern_id((r.cols[0],), (r.vals[0],))]
            assert abs(r.e - T_LEVELS[lv] / p_parent) < 1e-12
            assert r.e <= bound[lv] + 1e-15


def test_plant_does_not_mutate_input():
    t = draw_tree_tables(np.random.default_rng([5, 0]), 1.0)
    before = copy.deepcopy(t)
    out = plant_pairs(t, [], np.random.default_rng([5, 3]))
    assert all(np.array_equal(t.cond[k], before.cond[k]) for k in t.cond)
    if not isinstance(out, str):
        assert any(not np.array_equal(out[0].cond[k], t.cond[k]) for k in t.cond)


def test_plant_rejects_low_parent():
    uniform = np.full((4, 4), 0.25)
    t = TreeTables(np.full(4, 0.25), {k: uniform.copy() for k in draw_tree_tables(np.random.default_rng(0), 1.0).cond})
    assert plant_pairs(t, [], np.random.default_rng(0)) == "no_plant:0"


def test_loss_fractions_exact(pop):
    # V5: 버린 질량이 0인 수준에서 재정규화 전 손실 비율.
    pi0 = pattern_probs(pop.p0)
    checked = 0
    for lv, rules in pop.rules.items():
        if pop.dropped[lv] != 0:
            continue
        R, _, _ = regions(pop.shared, pop.major, rules, 0.0)
        base, dropped = move_mass(pop.p0, R, rules)
        assert dropped == 0
        pib = pattern_probs(base)
        for r in rules:
            pid = pattern_id((r.cols[0],), (r.vals[0],))
            assert abs(pib[pid] / pi0[pid] - (1 - r.e)) < 1e-10
        for a in range(4):
            for b in range(a + 1, 4):
                ra, rb = rules[a], rules[b]
                pid = pattern_id((ra.cols[0], rb.cols[0]), (ra.vals[0], rb.vals[0]))
                if pi0[pid] > 0:
                    assert abs(pib[pid] / pi0[pid] - (1 - ra.e) * (1 - rb.e)) < 1e-10
        checked += 1
    assert checked > 0


def test_floor_bound_when_no_drop(pop):
    # V6
    for env in pop.envs.values():
        if pop.dropped[env.level] == 0:
            assert env.check["min_ratio"] >= 0.512 - 1e-12


def test_mix_z_mass(pop):
    # V7
    for env in pop.envs.values():
        assert abs(env.p1[env.Z].sum() - 0.2 * env.gamma) < 1e-12


def test_local_support_all_envs(pop):
    # V8
    assert all(env.check["local_support_ok"] for env in pop.envs.values())


def test_check_detects_support_violation(pop):
    env = pop.envs["frequent__fav"]
    p1 = env.p1.copy()
    p1[np.flatnonzero(env.S1)[0]] = 0.0
    p1 /= p1.sum()
    res = check_minority(pop.p0, p1, env.S1, env.Z, pop.rules["frequent"], env.gamma)
    assert not res["support_ok"] and not res["passed"]
