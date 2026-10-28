"""C25: BOMiS 단계별 진단 (설계 7.2절 (e)). fit 객체의 공개 속성과 log_prob만 쓴다."""

from itertools import combinations

import numpy as np

from dgp.space import CELLS, N_CELLS, cell_index

CELLS64 = CELLS.astype(np.int64)
_TRIPLES = np.array(list(combinations(range(8), 3)), dtype=np.int64)  # 56개
_TRIPLE_CODES = None


def _triple_codes() -> np.ndarray:
    # (65536, 56): 셀마다 3차 조각 코드(열 3개의 값 16a + 4b + c).
    global _TRIPLE_CODES
    if _TRIPLE_CODES is None:
        c = CELLS64[:, _TRIPLES]
        _TRIPLE_CODES = (c[:, :, 0] * 16 + c[:, :, 1] * 4 + c[:, :, 2]).astype(np.int16)
    return _TRIPLE_CODES


def _nan_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def decomposition(truth, S_hat: np.ndarray, G_idx: np.ndarray) -> dict:
    # C25, D25.6: 1 >= R1 >= R2 >= R4.
    G_idx = np.asarray(G_idx, dtype=np.int64)
    novel = np.unique(G_idx[~truth.O[G_idx]])
    L = len(novel)
    cand = truth.p1[S_hat & truth.SZ]
    R1 = _nan_ratio(cand.sum(), truth.M0)
    R2 = _nan_ratio(np.sort(cand)[::-1][:L].sum(), truth.M0)
    R4 = _nan_ratio(truth.p1[novel[truth.SZ[novel]]].sum(), truth.M0)
    return {"R1": R1, "R2": R2, "R4": R4, "L_prime": L,
            "loss_support": 1.0 - R1, "loss_count": R1 - R2, "loss_select": R2 - R4}


def true_shell_law(truth) -> dict[int, float]:
    # C25, D25.2: SZ 셀에서 O까지 최소 해밍 거리의 p1 가중 분포.
    sz = np.flatnonzero(truth.SZ)
    O_rows = CELLS[np.flatnonzero(truth.O)]
    if not len(sz) or not len(O_rows) or truth.M0 <= 0:
        return {}
    dist = np.empty(len(sz), dtype=np.int64)
    for i in range(0, len(sz), 2048):
        block = CELLS[sz[i:i + 2048]]
        dist[i:i + 2048] = (block[:, None, :] != O_rows[None]).sum(axis=2).min(axis=1)
    w = truth.p1[sz]
    return {int(s): float(w[dist == s].sum() / truth.M0) for s in np.unique(dist)}


def true_fresh_share(truth, D0_idx: np.ndarray, D1_idx: np.ndarray) -> float:
    # C25, D25.3: 두 표본 합쳐 한 번도 안 나온 3차 조각을 가진 SZ 셀의 p1 몫.
    codes = _triple_codes()
    pooled = codes[np.concatenate([D0_idx, D1_idx])]
    counts = np.stack([np.bincount(pooled[:, t], minlength=64) for t in range(len(_TRIPLES))])  # (56, 64)
    sz = np.flatnonzero(truth.SZ)
    if not len(sz) or truth.M0 <= 0:
        return float("nan")
    unseen = (counts[np.arange(len(_TRIPLES))[None, :], codes[sz]] == 0).any(axis=1)
    return float(truth.p1[sz][unseen].sum() / truth.M0)


def _tv_to_sz(q: np.ndarray, truth) -> float:
    # q(.|A)와 p1(.|SZ)의 TV. q는 이미 A 밖에서 0인 벡터.
    total = q.sum()
    if total <= 0 or truth.M0 <= 0:
        return float("nan")
    return 0.5 * float(np.abs(q / total - np.where(truth.SZ, truth.p1, 0.0) / truth.M0).sum())


def bomis_diagnostics(model, truth, split, G: np.ndarray) -> dict:
    # C25, D25.1–D25.8. 순서는 fit -> sample -> 이 함수. log_prob는 난수를 쓰지 않는다.
    G_idx = cell_index(G)
    S_hat = model.support.contains(CELLS64)
    out = {
        "M_hat": float(model.M_hat), "M0": float(truth.M0),
        "M_err": float(model.M_hat - truth.M0), "M_abs_err": float(abs(model.M_hat - truth.M0)),
        "f0": float(model.f0), "n_sz": int(truth.SZ.sum()),
        "steps": int(model.steps), "maj_excluded": int(len(model.majority) > 0),
    }
    out.update(decomposition(truth, S_hat, G_idx))
    true_shell = true_shell_law(truth)
    shells = dict(model.shells)
    if shells and true_shell:
        keys = set(shells) | set(true_shell)
        out["shell_tv"] = 0.5 * float(sum(abs(shells.get(s, 0.0) - true_shell.get(s, 0.0)) for s in keys))
    else:
        out["shell_tv"] = float("nan")
    fresh = true_fresh_share(truth, split.D0_idx, split.D1_idx)
    out["ell"], out["ell_true"] = float(model.ell), fresh
    out["ell_err"] = float(model.ell - fresh)
    Gn = G_idx[~truth.O[G_idx]]
    out["novel_tv"] = _tv_to_sz(np.bincount(Gn, minlength=N_CELLS).astype(float), truth) if len(Gn) else float("nan")
    Q = np.exp(model.log_prob(CELLS64, y=1))
    Qu = np.exp(model.log_prob(CELLS64, y=1, mask=False))
    out.update({
        "Q_total": float(Q.sum()), "Qu_total": float(Qu.sum()),
        "Q_sz": float(Q[truth.SZ].sum()), "Q_stz": float(Q[truth.STZ].sum()),
        "Qu_sz": float(Qu[truth.SZ].sum()), "Qu_stz": float(Qu[truth.STZ].sum()),
        "Q_tv": _tv_to_sz(np.where(S_hat & ~truth.O, Q, 0.0), truth),
        "n_stz_in_S": int((S_hat & truth.STZ).sum()),
        "p0_stz_in_S": float(truth.p0[S_hat & truth.STZ].sum()),
        "leak": float((~truth.S1[G_idx]).mean()),
    })
    return out
