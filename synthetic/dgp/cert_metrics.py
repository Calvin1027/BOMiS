"""C18: 채점 (d) 인증과 정확 계산 (설계 4.6, 7.2절 (d))."""

import math

import numpy as np
from scipy.stats import beta, binom

from dgp.minority import rule_pid

TAU = math.log(480 / 0.05)


def exact_cert_prob(pi0, pi1, n0: int, n1: int, x_star: int):
    # C18, D18.2: P(N1 = 0) P(N0 >= x*), N0와 N1은 독립.
    return (1.0 - np.asarray(pi1)) ** n1 * binom.sf(x_star - 1, n0, pi0)


def exact_fwer_bound(truth, x_star: int) -> float:
    # C18, D18.3: 가능한 조각에서 인증 확률의 합.
    ok = (truth.pi0 > 0) & (truth.pi1 > 0)
    return float(exact_cert_prob(truth.pi0[ok], truth.pi1[ok], truth.n0, truth.n1, x_star).sum())


def clopper_pearson_upper(k: int, n: int, level: float = 0.95) -> float:
    # C18, D18.4
    return 1.0 if k == n else float(beta.ppf(level, k + 1, n - k))


def signal(pi0, n0: int, x_star: int):
    # C18, D18.1: 코드 기준 신호.
    return n0 * np.asarray(pi0) / x_star


def theory_signal(pi0, n0: int, n1: int, r: float = 0.5):
    # C18, D18.1: 이론 신호 n1 r pi0 / lambda*.
    kappa = n0 / (n1 * r)
    lam = TAU / (kappa * math.log1p(1 / kappa))
    return n1 * r * np.asarray(pi0) / lam


def power_curve(pi0_grid, n0: int, x_star: int):
    # C18: 금지 쌍(pi1 = 0)의 정확 인증 확률 곡선.
    return binom.sf(x_star - 1, n0, np.asarray(pi0_grid))


def score_cert(truth, C_hat: np.ndarray, x_star: int) -> dict:
    # C18, D18.3
    out = {"x_star": int(x_star), "n_cert": int(C_hat.sum())}
    feasible = truth.pi1 > 0
    out["n_false"] = int((C_hat & feasible).sum())
    out["any_false"] = int(out["n_false"] > 0)
    out["fwer_bound"] = exact_fwer_bound(truth, x_star)
    certs = []
    for i, r in enumerate(truth.rules):
        pid = rule_pid(r)
        certs.append(bool(C_hat[pid]))
        out[f"cert_{i}"] = int(C_hat[pid])
        out[f"signal_{i}"] = float(signal(truth.pi0[pid], truth.n0, x_star))
        out[f"theory_signal_{i}"] = float(theory_signal(truth.pi0[pid], truth.n0, truth.n1))
        out[f"exact_power_{i}"] = float(binom.sf(x_star - 1, truth.n0, truth.pi0[pid]))
    out["all_cert"] = int(all(certs))
    return out
