"""Chao1 상한 ablation(results/ablation_cap) 집계: (env, n1)별 base·nocap 평균 ± MCSE와 짝 차이(nocap - base).

MCSE = 모집단별 평균의 표준편차 / √모집단 수 (make_easy_compare4.py와 같은 정의).
검산: base recovery가 기존 results/<env>/bomis의 recovery와 같은지.
--level frequent(기본): reports/ablation_cap_table.md, reports/ablation_cap_cells.csv
--level mid|rare: reports/ablation_cap_<level>_table.md, reports/ablation_cap_<level>_cells.csv
  (mid·rare는 같은 칸의 다른 생성 방법 평균도 함께 적는다)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = [("recovery", "SZ 질량 회수"), ("rec_count", "SZ 개수 회수"), ("leak", "STZ leak"), ("control", "control"),
           ("diag_L_prime", "새 셀 수 $L'$"), ("diag_loss_count", "개수 손실"), ("diag_loss_select", "선택 손실"),
           ("uni_tv", "uni_tv"), ("pair_tv", "pair_tv")]
OTHERS = {"tabddpm": "TabDDPM", "arf": "ARF", "ctgan": "CTGAN", "ros": "ROS"}  # 같은 칸에서 비교할 다른 방법
OTHER_COLS = ["recovery", "rec_count", "leak", "pair_tv"]


def env_out(level):
    if level == "frequent":
        return {"frequent__fav": "results/easy", "frequent__avg": "results/easy_avg", "frequent__unf": "results/easy_unf"}
    return {f"{level}__{ov}": f"results/{level}_{ov}" for ov in ("fav", "avg", "unf")}


def mean_mcse(g, col):
    by_pop = g.groupby("pop")[col].mean()
    return g[col].mean(), by_pop.std(ddof=1) / np.sqrt(len(by_pop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="frequent", choices=["frequent", "mid", "rare"])
    args = ap.parse_args()
    ENV_OUT = env_out(args.level)
    suffix = "" if args.level == "frequent" else f"_{args.level}"

    df = pd.concat([pd.read_csv(p) for env in ENV_OUT for p in Path("results/ablation_cap", env).glob("*.csv")],
                   ignore_index=True)
    key = ["env", "pop", "n1", "seed"]
    saved = []
    for env, out in ENV_OUT.items():
        for p in Path(out, "bomis").glob("*.csv"):
            r = pd.read_csv(p).iloc[0]
            saved.append({"env": env, "pop": int(r["pop"]), "n1": int(r["n1"]), "seed": int(r["seed"]), "old_recovery": r["recovery"]})
    base = df[df["cond"] == "base"].merge(pd.DataFrame(saved), on=key, how="left")
    diff = (base["recovery"] - base["old_recovery"]).abs()
    wide = df.pivot_table(index=key, columns="cond", values=[c for c, _ in METRICS]).reset_index()
    for c, _ in METRICS:
        wide[(c, "diff")] = wide[(c, "nocap")] - wide[(c, "base")]
    wide.columns = ["_".join(x for x in col if x) for col in wide.columns]
    wide.to_csv(f"reports/ablation_cap{suffix}_cells.csv", index=False)

    groups = list(wide.groupby([pd.Categorical(wide["env"], categories=list(ENV_OUT)), "n1"], observed=True))
    lines = [f"# Chao1 상한 ablation, {args.level} (scripts/summarize_ablation.py --level {args.level})", "",
             f"칸 수: base {int((df['cond'] == 'base').sum())}, nocap {int((df['cond'] == 'nocap').sum())}. "
             f"검산: base recovery와 기존 결과의 차이 최대 {diff.max():.2e}, 0이 아닌 칸 {int((diff > 1e-9).sum())}개, "
             f"기존 결과 없음 {int(base['old_recovery'].isna().sum())}칸.", "",
             "값은 칸 평균 ± MCSE. 차이 = nocap − base (같은 칸끼리 짝).", ""]
    for c, name in METRICS:
        lines += [f"## {name} (`{c}`)", "", "| 환경 | $n_1$ | 칸 | base | nocap | 차이 |", "|---|---|---|---|---|---|"]
        for (env, n1), g in groups:
            vals = [mean_mcse(g, f"{c}_{k}") for k in ("base", "nocap", "diff")]
            lines.append(f"| {env.split('__')[1]} | {n1} | {len(g)} | " + " | ".join(f"{m:.3f} ± {s:.3f}" for m, s in vals) + " |")
        lines.append("")

    if args.level != "frequent":
        rows = []
        for env, out in ENV_OUT.items():
            for m in OTHERS:
                for p in Path(out, m).glob("*.csv"):
                    r = pd.read_csv(p).iloc[0]
                    rows.append({"env": env, "method": m, "pop": int(r["pop"]), "n1": int(r["n1"]), "seed": int(r["seed"]),
                                 **{c: r[c] for c in OTHER_COLS}})
        oth = pd.DataFrame(rows)
        lines += ["## nocap과 다른 생성 방법 (같은 칸 평균 ± MCSE)", "",
                  "| 환경 | $n_1$ | 방법 | 칸 | " + " | ".join(OTHER_COLS) + " |", "|" + "---|" * (4 + len(OTHER_COLS))]
        for (env, n1), g in groups:
            for cond in ("base", "nocap"):
                vals = [mean_mcse(g, f"{c}_{cond}") for c in OTHER_COLS]
                lines.append(f"| {env.split('__')[1]} | {n1} | BOMiS {cond} | {len(g)} | "
                             + " | ".join(f"{m:.3f} ± {s:.3f}" for m, s in vals) + " |")
            for m, name in OTHERS.items():
                h = oth[(oth.env == env) & (oth.n1 == n1) & (oth.method == m)]
                if not len(h):
                    continue
                vals = [mean_mcse(h, c) for c in OTHER_COLS]
                lines.append(f"| {env.split('__')[1]} | {n1} | {name} | {len(h)} | "
                             + " | ".join(f"{m_:.3f} ± {s:.3f}" for m_, s in vals) + " |")
        lines.append("")

    Path(f"reports/ablation_cap{suffix}_table.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
