"""C15–C17: 채점 (a) 집합, (b) 법칙, (c) 생성 행 (설계 7.2절)."""

import numpy as np

from dgp.minority import rule_mask
from dgp.space import N_CELLS, pattern_counts


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def score_set(truth, S_hat: np.ndarray) -> dict:
    # C15, D15.1
    p0, p1, SZ, STZ = truth.p0, truth.p1, truth.SZ, truth.STZ
    out = {
        "sz_rec_mass": _ratio(p1[S_hat & SZ].sum(), truth.M0),
        "sz1_rec_mass": _ratio(p1[S_hat & truth.SZ1].sum(), p1[truth.SZ1].sum()),
        "sz2_rec_mass": _ratio(p1[S_hat & truth.SZ2].sum(), p1[truth.SZ2].sum()),
        "nomaj_rec_mass": _ratio(p1[S_hat & truth.SZ_nomaj].sum(), p1[truth.SZ_nomaj].sum()),
        "sz_rec_count": _ratio((S_hat & SZ).sum(), SZ.sum()),
        "stz_sup_count": 1.0 - _ratio((S_hat & STZ).sum(), STZ.sum()),
        "stz_sup_maj": 1.0 - _ratio(p0[S_hat & STZ].sum(), p0[STZ].sum()),
        "S_size": int(S_hat.sum()),
    }
    for i, r in enumerate(truth.rules):
        out[f"rule_left_{i}"] = int((S_hat & rule_mask(r)).any())
    return out


def score_law(truth, q: np.ndarray) -> dict:
    # C16, D16.1
    gamma = float(q[truth.SZ].sum())
    tv = float("nan")
    if gamma > 0 and truth.M0 > 0:
        tv = 0.5 * float(np.abs(np.where(truth.SZ, q, 0.0) / gamma - np.where(truth.SZ, truth.p1, 0.0) / truth.M0).sum())
    return {"Gamma": gamma, "Lambda": float(q[truth.STZ].sum()), "tv_sz": tv}


def score_rows(truth, G_idx: np.ndarray) -> dict:
    # C17, D17.1. 회수율은 맞힌 서로 다른 SZ 셀 기준.
    G_idx = np.asarray(G_idx, dtype=np.int64)
    n = len(G_idx)
    hit = np.unique(G_idx[truth.SZ[G_idx]])
    out = {
        "leak": float((~truth.S1[G_idx]).mean()) if n else float("nan"),
        "novel": float((~truth.O[G_idx]).mean()) if n else float("nan"),
        "rec_mass": _ratio(truth.p1[hit].sum(), truth.M0),
        "rec_count": _ratio(len(hit), truth.SZ.sum()),
    }
    for i, r in enumerate(truth.rules):
        out[f"rule_rows_{i}"] = float(rule_mask(r)[G_idx].mean()) if n else float("nan")
    if n:
        freq = pattern_counts(G_idx) / n
        uni = 0.5 * np.abs(freq[:32] - truth.pi1[:32]).reshape(8, 4).sum(axis=1)
        pair = 0.5 * np.abs(freq[32:] - truth.pi1[32:]).reshape(28, 16).sum(axis=1)
        out["uni_tv_exact"], out["pair_tv_exact"] = float(uni.mean()), float(pair.mean())
    else:
        out["uni_tv_exact"] = out["pair_tv_exact"] = float("nan")
    return out


def cell_mask(idx: np.ndarray) -> np.ndarray:
    out = np.zeros(N_CELLS, dtype=bool)
    out[np.asarray(idx, dtype=np.int64)] = True
    return out
