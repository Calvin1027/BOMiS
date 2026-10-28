"""BOMiS 개수 손실을 '생성 예산 몫'과 'Chao1 상한 몫'으로 나눈다. 모델 학습 없이 support C만 다시 만든다.

R2(L) = (S_hat ∩ SZ에서 p1이 큰 셀 L개의 질량) / M0. bomis_diag.decomposition의 R2와 같은 정의.
  L' = 실제 만든 서로 다른 새 셀 수 (diag_L_prime)
  n_new = 새 행 수 = novel × n_gen  (새 행은 모두 O 밖, 나머지 행은 O 안)
분해: 1 = (1-R1) + [R1 - R2(n_new)] + [R2(n_new) - R2(L')] + [R2(L') - R4] + R4
         support    예산(프로토콜상 불가피)   Chao1 상한          셀 선택
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dgp.harness import run_support  # noqa: E402
from dgp.population import load_population  # noqa: E402
from dgp.sampling import make_split  # noqa: E402
from dgp.truth import make_truth  # noqa: E402

OUTS = {"frequent__fav": "results/easy", "frequent__avg": "results/easy_avg", "frequent__unf": "results/easy_unf"}


def main():
    rows = []
    for env, out in OUTS.items():
        for f in sorted(Path(out, "bomis").glob("*.csv")):
            b = pd.read_csv(f).iloc[0]
            pop = load_population(Path("results/pops") / f"pop_{int(b['pop']):04d}.npz")
            split = make_split(pop, env, int(b["n1"]), int(b["seed"]))
            truth = make_truth(pop, env, split)
            S_hat = run_support("C", split)[0]
            cand = np.sort(truth.p1[S_hat & truth.SZ])[::-1]
            n_gen = len(split.D0_idx) - len(split.D1_idx)
            n_new = int(round(b["novel"] * n_gen))
            L = int(b["diag_L_prime"])
            r2 = lambda k: cand[:k].sum() / truth.M0  # noqa: E731
            rows.append({"env": env.split("__")[1], "n1": int(b["n1"]), "pop": int(b["pop"]), "seed": int(b["seed"]),
                         "n_sz": int(truth.SZ.sum()), "n_cand": len(cand), "L_prime": L, "n_new": n_new, "n_gen": n_gen,
                         "R1": cand.sum() / truth.M0, "R2_L": r2(L), "R2_new": r2(n_new), "R2_gen": r2(n_gen),
                         "R4": b["diag_R4"], "R2_L_diag": b["diag_R2"]})
    df = pd.DataFrame(rows)
    df["chk"] = (df["R2_L"] - df["R2_L_diag"]).abs()
    df["loss_support"] = 1 - df["R1"]
    df["loss_budget"] = df["R1"] - df["R2_new"]
    df["loss_chao1"] = df["R2_new"] - df["R2_L"]
    df["loss_select"] = df["R2_L"] - df["R4"]
    df["count_ceiling_new"] = df["n_new"] / df["n_sz"]
    Path("reports").mkdir(exist_ok=True)
    df.to_csv("reports/r2_budget_cells.csv", index=False)
    cols = ["n_sz", "L_prime", "n_new", "R1", "R2_L", "R2_new", "R2_gen", "R4",
            "loss_support", "loss_budget", "loss_chao1", "loss_select", "count_ceiling_new"]
    g = df.groupby(["env", "n1"], sort=False)
    summ = g[cols].mean()
    summ["cells"] = g.size()
    print("cells", len(df), "max |R2(L') - diag_R2|", df["chk"].max())
    print(summ.round(4).to_string())
    summ.round(4).to_csv("reports/r2_budget_summary.csv")


if __name__ == "__main__":
    main()
