"""C10, C12: 모집단 조립·수락·저장 (설계 4.7, 9절)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dgp.laws import apply_zeros, choose_parent_zeros, draw_tree_tables, tree_law
from dgp.minority import (
    E_MAX, GAMMAS, T_LEVELS, Rule, check_minority, mix, move_mass, plant_pairs, regions, rule_pid,
)
from dgp.space import pattern_probs

LEVELS = tuple(T_LEVELS)
OVERLAPS = tuple(GAMMAS)
# C10, D10.1: 바깥부터 STZ 수준, 겹침 수준.
ENV_KEYS = [f"{lv}__{ov}" for lv in LEVELS for ov in OVERLAPS]


class StopGeneration(RuntimeError):
    pass


@dataclass
class Env:
    level: str
    gamma: float
    p1: np.ndarray
    S1: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    check: dict


@dataclass
class Population:
    seed: int
    p0: np.ndarray
    h: np.ndarray
    shared: list
    major: list
    rules: dict
    dropped: dict
    envs: dict


def _rng(*entropy: int):
    return np.random.default_rng(list(entropy))


def build_population(pop_seed: int) -> Population | str:
    # C10, D10.2. 순서: 표 → 0(심기 전 법칙) → 심기 → p0 → h → 수준별 이동 → γ별 섞기·점검.
    t = draw_tree_tables(_rng(pop_seed, 0), 1.0)
    z = choose_parent_zeros(tree_law(t), _rng(pop_seed, 2))
    if z is None:
        return "no_zeros"
    shared, major = z
    zeros = shared + major
    planted = plant_pairs(t, zeros, _rng(pop_seed, 3))
    if isinstance(planted, str):
        return planted
    t_planted, rules = planted
    p0 = apply_zeros(tree_law(t_planted), zeros)
    pi0 = pattern_probs(p0)
    for rs in rules.values():  # D4.7: 구성상 항상 참. 어긋나면 구현 오류다.
        for r in rs:
            if abs(pi0[rule_pid(r)] - r.pi0) > 1e-12 or r.e > E_MAX + 1e-15:
                raise RuntimeError(f"planting check failed for seed {pop_seed}: {r}")
    h = tree_law(draw_tree_tables(_rng(pop_seed, 1), 1.0))
    envs, dropped = {}, {}
    for lv in LEVELS:
        R0, _, _ = regions(shared, major, rules[lv], 0.0)
        base, dropped[lv] = move_mass(p0, R0, rules[lv])  # D7.1: 세 γ에 재사용
        for ov, g in GAMMAS.items():
            R, Z, S1 = regions(shared, major, rules[lv], g)
            p1 = mix(base, h, R, Z, g)
            envs[f"{lv}__{ov}"] = Env(lv, g, p1, S1, R, Z, check_minority(p0, p1, S1, Z, rules[lv], g))
    for key in ENV_KEYS:
        for name, value in envs[key].check.items():
            if name.endswith("_ok") and not value:
                return f"check_fail:{key}:{name}"
    return Population(pop_seed, p0, h, list(shared), list(major), rules, dropped, envs)


def generate_accepted(start: int, n: int, max_tries: int = 2000) -> tuple[list[Population], list[tuple[int, str]]]:
    # C10, D10.3
    pops, rejected = [], []
    for seed in range(start, start + max_tries):
        out = build_population(seed)
        if isinstance(out, str):
            rejected.append((seed, out))
        else:
            pops.append(out)
            if len(pops) == n:
                return pops, rejected
    raise StopGeneration(f"{len(pops)}/{n} accepted in {max_tries} tries from {start}")


def _rule_to_json(r: Rule) -> list:
    return [r.kind, list(r.cols), list(r.vals), r.move_col, r.e, r.pi0]


def _rule_from_json(x: list) -> Rule:
    return Rule(x[0], tuple(int(v) for v in x[1]), tuple(int(v) for v in x[2]), int(x[3]), float(x[4]), float(x[5]))


def save_population(pop: Population, path: Path) -> None:
    # C12, D12.1
    arrays = {"p0": pop.p0, "h": pop.h}
    for key, env in pop.envs.items():
        arrays.update({f"p1__{key}": env.p1, f"S1__{key}": env.S1, f"R__{key}": env.R, f"Z__{key}": env.Z})
    meta = {
        "seed": pop.seed,
        "shared": [list(z) for z in pop.shared],
        "major": [list(z) for z in pop.major],
        "rules": {lv: [_rule_to_json(r) for r in rs] for lv, rs in pop.rules.items()},
        "dropped": pop.dropped,
        "envs": {key: {"level": env.level, "gamma": env.gamma, "check": env.check} for key, env in pop.envs.items()},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load_population(path: Path) -> Population:
    # C12, D12.1
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        envs = {
            key: Env(info["level"], float(info["gamma"]), f[f"p1__{key}"], f[f"S1__{key}"],
                     f[f"R__{key}"], f[f"Z__{key}"], info["check"])
            for key, info in meta["envs"].items()
        }
        return Population(
            int(meta["seed"]), f["p0"], f["h"],
            [tuple(int(v) for v in z) for z in meta["shared"]],
            [tuple(int(v) for v in z) for z in meta["major"]],
            {lv: [_rule_from_json(x) for x in rs] for lv, rs in meta["rules"].items()},
            {lv: float(v) for lv, v in meta["dropped"].items()},
            envs,
        )
