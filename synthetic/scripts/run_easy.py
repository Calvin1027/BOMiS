"""C28 진입: 쉬운 환경 frequent__fav의 한 칸(모집단, n1, 표본 seed)에서 방법들을 돌려 칸별 csv를 쓴다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ENV = "frequent__fav"
N1S = (400, 800)
CPU_METHODS = ["A", "B", "C", "bomis", "ros", "smoten", "smote_enc"]
GPU_METHODS = ["ctgan"]
OVERLAPS = ("fav", "avg", "unf")


def env_list(levels):
    return [f"{lv}__{ov}" for lv in levels for ov in OVERLAPS]


def out_dir(env):
    """환경별 결과 폴더. frequent는 처음 쓴 이름(results/easy, easy_avg, easy_unf)을 그대로 쓴다."""
    lv, ov = env.split("__")
    if lv == "frequent":
        return "results/easy" if ov == "fav" else f"results/easy_{ov}"
    return f"results/{lv}_{ov}"


def cells(pop_paths, seeds):
    # D28.1: n1=400 먼저, 그다음 모집단, 표본 seed 순서.
    return [(p, n1, s) for n1 in N1S for p in pop_paths for s in seeds]


def _row_path(out, method, pop_seed, n1, s):
    return Path(out) / method / f"{pop_seed:04d}_{n1}_{s}.csv"


def run_cell(pop_path, n1, sample_seed, methods, out, env=ENV, bomis_kw=None, base_config=None):
    """C28, D28.2. 이미 csv가 있는 방법은 건너뛰고, 한 방법이 실패해도 다음 방법을 돈다."""
    from baselines.external import run_external
    from baselines.ros import ros
    from dgp.bomis_diag import bomis_diagnostics
    from dgp.cert_metrics import score_cert
    from dgp.harness import (
        cert_set, check_cert_reconstruction, evaluate_rows, make_pop_adapter, run_bomis, run_support,
    )
    from dgp.population import load_population
    from dgp.sampling import make_split, reference_rows
    from dgp.scorer import score_rows, score_set
    from dgp.space import cell_index
    from dgp.truth import make_truth

    pop = load_population(pop_path)
    todo = [m for m in methods if not _row_path(out, m, pop.seed, n1, sample_seed).exists()]
    if not todo:
        return
    split = make_split(pop, env, n1, sample_seed)
    truth = make_truth(pop, env, split)
    adapter = None
    n_gen = len(split.D0_idx) - len(split.D1_idx)
    for m in todo:
        row = {"pop": pop.seed, "n1": n1, "seed": sample_seed, "env": env, "method": m, "M0": truth.M0}
        t0 = time.monotonic()
        try:
            if m in ("A", "B", "C"):
                S_hat, sup, used = run_support(m, split)
                row.update(score_set(truth, S_hat))
                row.update(score_cert(truth, cert_set(sup, used, split.D1_idx), sup.x_star))
                row["cert_consistent"] = int(check_cert_reconstruction(sup, used, split.D1_idx))
                G = None
            elif m == "bomis":
                model, G = run_bomis(split, sample_seed, **(bomis_kw or {}))
                row.update({f"diag_{k}": v for k, v in bomis_diagnostics(model, truth, split, G).items()})
            elif m == "ros":
                G = ros(split.D1, n_gen, np.random.default_rng([pop.seed, sample_seed, 30]))
            else:
                with tempfile.TemporaryDirectory(dir=os.environ.get("SLURM_TMPDIR") or "/tmp") as tmp:
                    res = run_external(m, split.D0, split.D1, n_gen, sample_seed, Path(tmp) / m, base_config=base_config)
                row.update({"attempt": res["attempt"], "options": str(res["options"]), "fail_log": res["reason"][-500:]})
                if not res["ok"]:
                    raise RuntimeError(res["reason"][-1500:])
                G = res["rows"]
            if G is not None:
                adapter = adapter or make_pop_adapter(pop, env, reference_rows(pop, env))
                row.update(evaluate_rows(G, split, adapter))
                row.update(score_rows(truth, cell_index(G)))
            row["ok"] = 1
        except Exception as error:  # Review Focus 3: 이 방법만 실패로 남긴다
            row.update({"ok": 0, "error": f"{type(error).__name__}: {error}"[-2000:]})
            traceback.print_exc()
        row["seconds"] = time.monotonic() - t0
        path = _row_path(out, m, pop.seed, n1, sample_seed)
        if row["ok"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(path, index=False)
        else:
            fail = Path(out) / "_failed" / m / path.name
            fail.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(fail, index=False)
        print(m, pop.seed, n1, sample_seed, "ok" if row["ok"] else "FAIL", round(row["seconds"], 1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--group", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--pops", default="results/pops")
    ap.add_argument("--n-pops", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default="results/easy")
    ap.add_argument("--env", default=ENV, choices=env_list(("frequent", "mid", "rare")))
    ap.add_argument("--levels", nargs="+", default=["frequent"], choices=["frequent", "mid", "rare"],
                    help="--all-envs에서 돌릴 STZ 수준. 수준마다 fav·avg·unf 3환경")
    ap.add_argument("--methods", nargs="+", help="그룹 기본 방법 대신 돌릴 방법들")
    ap.add_argument("--all-envs", action="store_true", help="--env·--out 대신 작업 번호로 3환경을 나눠 돈다")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    pop_paths = sorted(Path(args.pops).glob("pop_*.npz"))[: args.n_pops]
    methods = args.methods or (CPU_METHODS if args.group == "cpu" else GPU_METHODS)
    bomis_kw, base_config = None, None
    if args.quick:
        pop_paths, args.seeds = pop_paths[:1], [0]
        bomis_kw = {"steps": 250}
        from baselines.external import bl_src
        base_config = bl_src() / "configs" / "models.smoke.json"
    todo = cells(pop_paths, args.seeds)
    if args.quick:
        todo = todo[:1]
    if args.all_envs:  # 배열 하나로 여러 환경: 작업 번호 = 환경 순번 × 칸 수 + 칸 번호, 결과는 환경별 폴더
        args.env = env_list(args.levels)[args.task_id // len(todo)]
        args.out = out_dir(args.env)
        args.task_id %= len(todo)
    p, n1, s = todo[args.task_id]
    print("task", args.task_id, "pop", p.name, "n1", n1, "seed", s, "group", args.group,
          "env", args.env, "job", os.environ.get("SLURM_JOB_ID"), "node", os.environ.get("SLURMD_NODENAME"), flush=True)
    run_cell(p, n1, s, methods, args.out, env=args.env, bomis_kw=bomis_kw, base_config=base_config)


if __name__ == "__main__":
    main()
