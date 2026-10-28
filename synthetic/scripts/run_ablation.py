"""Chao1 상한 ablation: 한 칸(env, pop, n1, seed)에서 BOMiS를 한 번 fit하고 두 조건으로 sample한다.

  base  : 박사형 코드 그대로 (새 셀 수 L = min(n_new, ceil(f0)))
  nocap : fit 뒤 model.f0 = 1e9. f0는 _novel 안에서만 쓰이므로(model.py 446, 506, 529행)
          전체 상한 L = n_new가 되고 거리별 Chao1 상한도 풀린다. 다른 것은 그대로다.
두 조건은 같은 학습 모델을 쓰고, sample 전마다 난수 상태를 fit 직후 값으로 되돌린다.
원저자 코드와 run_easy.py는 고치지 않는다. 출력: <out>/<env>/<pop>_<n1>_<seed>.csv (조건마다 한 줄).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse  # noqa: E402
import copy  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ENV_OUT = {"frequent__fav": "results/easy", "frequent__avg": "results/easy_avg", "frequent__unf": "results/easy_unf",
           **{f"{lv}__{ov}": f"results/{lv}_{ov}" for lv in ("mid", "rare") for ov in ("fav", "avg", "unf")}}  # 기존 BOMiS 결과 위치(검산용)
NOCAP_F0 = 1e9


def run_cell(env, pop_seed, n1, seed, out):
    from dgp.bomis_diag import bomis_diagnostics
    from dgp.harness import BOMiS, KS, evaluate_rows, make_pop_adapter
    from dgp.population import load_population
    from dgp.sampling import make_split, reference_rows
    from dgp.scorer import score_rows
    from dgp.space import cell_index
    from dgp.truth import make_truth

    path = Path(out) / env / f"{pop_seed:04d}_{n1}_{seed}.csv"
    if path.exists():
        print("skip", path, flush=True)
        return
    pop = load_population(Path("results/pops") / f"pop_{pop_seed:04d}.npz")
    split = make_split(pop, env, n1, seed)
    truth = make_truth(pop, env, split)
    adapter = make_pop_adapter(pop, env, reference_rows(pop, env))
    n_gen = len(split.D0_idx) - len(split.D1_idx)

    t0 = time.monotonic()
    model = BOMiS(seed=seed).fit(split.D0, split.D1, KS)  # run_bomis와 같은 기본 인자
    t_fit = time.monotonic() - t0
    f0, rng_state = model.f0, copy.deepcopy(model.rng.bit_generator.state)

    saved = Path(ENV_OUT[env]) / "bomis" / f"{pop_seed:04d}_{n1}_{seed}.csv"
    saved_rec = float(pd.read_csv(saved)["recovery"].iloc[0]) if saved.exists() else np.nan

    rows = []
    for cond, f0_used in (("base", f0), ("nocap", NOCAP_F0)):
        model.rng.bit_generator.state = copy.deepcopy(rng_state)
        model.f0 = f0_used
        t1 = time.monotonic()
        G = np.asarray(model.sample(n_gen), dtype=np.int64)
        t_sample = time.monotonic() - t1
        row = {"env": env, "pop": pop_seed, "n1": n1, "seed": seed, "cond": cond, "f0_chao1": f0,
               "f0_used": f0_used, "n_gen": n_gen, "t_fit": t_fit, "t_sample": t_sample, "saved_recovery": saved_rec}
        row.update({f"diag_{k}": v for k, v in bomis_diagnostics(model, truth, split, G).items()})
        row.update(evaluate_rows(G, split, adapter))
        row.update(score_rows(truth, cell_index(G)))
        row["t_total"] = time.monotonic() - t1
        rows.append(row)
        print(cond, "recovery", round(row["recovery"], 4), "leak", round(row["leak"], 4),
              "L'", row["diag_L_prime"], "sample s", round(t_sample, 1), flush=True)
    model.f0 = f0
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print("fit s", round(t_fit, 1), "saved recovery", saved_rec, "->", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="frequent__fav", choices=list(ENV_OUT))
    ap.add_argument("--pop", type=int, default=0)
    ap.add_argument("--n1", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ablation_cap")
    args = ap.parse_args()
    print("env", args.env, "pop", args.pop, "n1", args.n1, "seed", args.seed,
          "job", os.environ.get("SLURM_JOB_ID"), "node", os.environ.get("SLURMD_NODENAME"), flush=True)
    run_cell(args.env, args.pop, args.n1, args.seed, args.out)


if __name__ == "__main__":
    main()
