import numpy as np

from dgp.population import ENV_KEYS, build_population, load_population, save_population


def test_acceptance_matches_reference_and_drop():
    # V9 (참조 구현 accept_v3.py), Review Focus 1
    results = {s: build_population(s) for s in range(60)}
    accepted = {s: p for s, p in results.items() if not isinstance(p, str)}
    assert len(accepted) == 48
    assert [s for s in range(11) if s not in accepted] == [7]
    with_drop = [p for p in accepted.values() if any(v > 0 for v in p.dropped.values())]
    assert with_drop
    assert all(env.check["passed"] for p in with_drop for env in p.envs.values())


def test_env_keys_and_paired(pop):
    assert list(pop.envs) == ENV_KEYS and len(ENV_KEYS) == 9
    assert ENV_KEYS[0] == "frequent__fav" and ENV_KEYS[-1] == "rare__unf"
    for key, env in pop.envs.items():
        assert key.startswith(env.level)
    assert [pop.envs[k].gamma for k in ENV_KEYS[:3]] == [0.0, 0.5, 1.0]


def test_roundtrip(pop, tmp_path):
    # V10
    path = tmp_path / "pop.npz"
    save_population(pop, path)
    back = load_population(path)
    assert back.seed == pop.seed and np.array_equal(back.p0, pop.p0) and np.array_equal(back.h, pop.h)
    assert back.shared == [tuple(z) for z in pop.shared] and back.major == [tuple(z) for z in pop.major]
    assert back.rules == pop.rules and back.dropped == pop.dropped
    for key in ENV_KEYS:
        a, b = pop.envs[key], back.envs[key]
        assert all(np.array_equal(getattr(a, f), getattr(b, f)) for f in ("p1", "S1", "R", "Z"))
