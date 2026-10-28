"""BOMiS와 생성 baseline의 개수 회수·개수 손실·선택 손실·금지 쌍 행을 같은 정의로 비교한다.

생성 행은 저장하지 않았으므로 칸별 csv의 값과 정답(모집단에서 다시 만듦)만 쓴다.
- k = 맞힌 서로 다른 SZ 셀 수 = rec_count × |SZ|. BOMiS는 k = L'(새 셀이 모두 SZ).
- 셀 k개로 얻을 수 있는 최대 회수 R2 = 질량 큰 SZ 셀 k개의 p1 합 / M0. 개수 손실 = 1 − R2, 선택 손실 = R2 − rec_mass.
  support 손실은 0으로 둔다(baseline은 support 단계가 없고, BOMiS의 support 손실은 0.004 이하라 4절 표와 거의 같다).
- 금지 쌍 행 비율 = rule_rows_0..3의 합(규칙 4개에 놓인 생성 행 몫), 금지 쌍 회피 = 행이 하나도 없는 규칙의 몫.
사용: compare_counts.py [--level mid --prefix mid] (결과: reports/<prefix>_counts.csv, reports/<prefix>_counts.md)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dgp.population import load_population  # noqa: E402
from dgp.sampling import make_split  # noqa: E402
from dgp.truth import make_truth  # noqa: E402

FREQ_OUTS = {"fav": "results/easy", "avg": "results/easy_avg", "unf": "results/easy_unf"}
METHODS = {"bomis": "BOMiS", "tabsyn": "TabSyn", "tabdiff": "TabDiff", "tabddpm": "TabDDPM", "arf": "ARF",
           "ctgan": "CTGAN", "smoten": "SMOTE-N", "smote_enc": "SMOTE-ENC"}
COLS = ["rec_mass", "k", "rec_count", "loss_count", "loss_select", "stz_rows", "rule_free"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="frequent", choices=["frequent", "mid", "rare"])
    ap.add_argument("--prefix", default="easy", help="결과 파일 이름 앞부분(reports/<prefix>_counts.md·csv)")
    args = ap.parse_args()
    OUTS = FREQ_OUTS if args.level == "frequent" else {ov: f"results/{args.level}_{ov}" for ov in FREQ_OUTS}
    pops = {int(p.stem[-4:]): load_population(p) for p in sorted(Path("results/pops").glob("pop_*.npz"))[:3]}
    rows = []
    for ov, out in OUTS.items():
        env = f"{args.level}__{ov}"
        frames = {m: pd.concat([pd.read_csv(f) for f in fs]) for m in METHODS
                  if (fs := sorted(Path(out, m).glob("*.csv")))}  # 돌리지 않은 방법(mid·rare의 TabSyn·TabDiff)은 건너뛴다
        for (ps, n1, s), _ in frames["bomis"].groupby(["pop", "n1", "seed"]):
            truth = make_truth(pops[ps], env, make_split(pops[ps], env, n1, s))
            n_sz = int(truth.SZ.sum())
            top = np.concatenate([[0.0], np.cumsum(np.sort(truth.p1[truth.SZ])[::-1])]) / truth.M0
            for m, df in frames.items():
                r = df[(df["pop"] == ps) & (df["n1"] == n1) & (df["seed"] == s)]
                if len(r) != 1:
                    continue
                r = r.iloc[0]
                k = int(round(r["rec_count"] * n_sz))
                rule = np.array([r[f"rule_rows_{i}"] for i in range(4)])
                rows.append({"env": ov, "method": m, "pop": ps, "n1": n1, "seed": s, "n_sz": n_sz,
                             "rec_mass": r["rec_mass"], "k": k, "rec_count": r["rec_count"],
                             "loss_count": 1.0 - top[k], "loss_select": top[k] - r["rec_mass"],
                             "stz_rows": rule.sum(), "rule_free": (rule == 0).mean()})
    df = pd.DataFrame(rows)
    df.to_csv(f"reports/{args.prefix}_counts.csv", index=False)
    pm = df.groupby(["env", "n1", "method", "pop"])[COLS].mean()
    g = pm.groupby(["env", "n1", "method"])
    mean, mcse = g.mean(), g.std(ddof=1) / np.sqrt(3)
    n = df.groupby(["env", "n1", "method"]).size()
    lines = [f"# 생성 방법별 개수 회수·손실 분해·금지 쌍 행 ({args.level}, scripts/compare_counts.py)", "",
             "| 환경 | $n_1$ | 방법 | 칸 | recovery | 맞힌 SZ 셀 수 $k$ | SZ 개수 회수 | 개수 손실 | 선택 손실 | 금지 쌍 행 비율 | 금지 쌍 회피 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ov in OUTS:
        for n1 in (400, 800):
            for m, name in METHODS.items():
                if (ov, n1, m) not in mean.index:
                    continue
                a, e = mean.loc[(ov, n1, m)], mcse.loc[(ov, n1, m)]
                cells = [f"{a[c]:.3f} ± {e[c]:.3f}" for c in ("rec_mass",)] + [f"{a['k']:.0f} ± {e['k']:.0f}"] + \
                        [f"{a[c]:.3f} ± {e[c]:.3f}" for c in ("rec_count", "loss_count", "loss_select", "stz_rows", "rule_free")]
                lines.append(f"| {ov} | {n1} | {name} | {n[(ov, n1, m)]} | " + " | ".join(cells) + " |")
    Path(f"reports/{args.prefix}_counts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
