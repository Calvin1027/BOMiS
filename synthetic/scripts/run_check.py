"""C19 진입: 1단계 점검. 모집단 점검, 정답 기술 통계, 신호 배치·겹침 판정, 박사형 사본 해시를 모은다.
정답 점검(V14–V18)과 연결 점검(V19–V25)은 pytest로 돌고, 그 결과는 --pytest-log로 받아 판정에 넣는다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dgp.minority import T_LEVELS  # noqa: E402
from dgp.population import ENV_KEYS, load_population  # noqa: E402
from dgp.sampling import make_split  # noqa: E402
from dgp.truth import make_truth  # noqa: E402
from dgp.vendor import load_bomis, verify_hashes  # noqa: E402

N1S = (50, 100, 200, 400, 800)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", default="results/pops")
    ap.add_argument("--n-pops", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="results/dgp_check")
    ap.add_argument("--pytest-log", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pops = [load_population(p) for p in sorted(Path(args.pops).glob("pop_*.npz"))[: args.n_pops]]
    env_rows, stat_rows = [], []
    for pop in pops:
        for key in ENV_KEYS:
            e = pop.envs[key]
            env_rows.append({"pop": pop.seed, "env": key, "dropped": pop.dropped[e.level], **e.check,
                             "tv_p0_p1": 0.5 * float(np.abs(pop.p0 - e.p1).sum()), "S1_size": int(e.S1.sum())})
            for n1 in N1S:
                for s in args.seeds:
                    t = make_truth(pop, key, make_split(pop, key, n1, s))
                    p1sz = t.M0
                    stat_rows.append({
                        "pop": pop.seed, "env": key, "n1": n1, "seed": s, "n_stz": int(t.STZ.sum()),
                        "M0": t.M0, "n_sz": int(t.SZ.sum()),
                        "sz1_share_count": t.SZ1.sum() / max(t.SZ.sum(), 1),
                        "sz1_share_mass": t.p1[t.SZ1].sum() / p1sz if p1sz > 0 else np.nan,
                        "nomaj_share_mass": t.p1[t.SZ_nomaj].sum() / p1sz if p1sz > 0 else np.nan,
                    })
    env_df, stat_df = pd.DataFrame(env_rows), pd.DataFrame(stat_rows)
    env_df.to_csv(out / "env_checks.csv", index=False)
    stats = stat_df.groupby(["env", "n1"]).mean(numeric_only=True).drop(columns=["pop", "seed"]).reset_index()
    stats.to_csv(out / "truth_stats.csv", index=False)
    # 신호 배치(D19.1): 코드 기준값 x*로 s = n0 pi0 / x*.
    thr = load_bomis().model.certify_threshold
    s = {lv: {n1: 8000 * t / thr(8000, n1, 0.5, 0.05 / 480) for n1 in N1S} for lv, t in T_LEVELS.items()}
    nomaj = stats.assign(ov=stats.env.str.split("__").str[1]).groupby(["n1", "ov"]).nomaj_share_mass.mean().unstack()
    verdict = {
        "env_checks_pass": bool(env_df.passed.all()),
        "min_ratio_ge_half": bool((env_df.min_ratio >= 0.5).all()),
        "vendor_hashes": bool(all(verify_hashes().values())),
        "signal_frequent": bool(s["frequent"][200] < 1 <= s["frequent"][400]),
        "signal_mid": bool(s["mid"][400] < 1 <= s["mid"][800]),
        "signal_rare": bool(s["rare"][800] < 1),
        "overlap_fav_zero": bool((nomaj["fav"] == 0).all()),
        "overlap_avg_lt_unf": bool((nomaj["avg"] < nomaj["unf"]).all()),
    }
    if args.pytest_log:
        tail = Path(args.pytest_log).read_text().strip().splitlines()[-1]
        verdict["pytest_pass"] = "passed" in tail and "failed" not in tail and "error" not in tail
        verdict["pytest_summary"] = tail
    verdict["stage1_pass"] = all(v for k, v in verdict.items() if isinstance(v, bool))
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
    lines = ["# 1단계 점검 결과", "", f"모집단: {[p.seed for p in pops]}, 표본 seed {args.seeds}", "",
             "## 판정", ""] + [f"- {k}: {v}" for k, v in verdict.items()] + [
             "", "## 신호 s = n0 pi0 / x*", "", "```\n" + pd.DataFrame(s).T.round(2).to_string() + "\n```", "",
             "## 환경별 floor 최솟값과 버린 질량", "",
             "```\n" + env_df.groupby("env")[["min_ratio", "dropped", "tv_p0_p1"]].agg(["min", "mean", "max"]).round(3).to_string() + "\n```", "",
             "## 정답 기술 통계 (env, n1 평균)", "", "```\n" + stats.round(4).to_string(index=False) + "\n```"]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/stage1_dgp_check.md").write_text("\n".join(lines))
    print(json.dumps(verdict, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
