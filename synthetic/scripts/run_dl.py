"""DL baseline(TabDDPM, TabSyn) 배열 진입. STZ 수준마다 3환경 × 모집단 3 × n1 2 × 표본 seed 10 = 모델당 180칸.

배열 작업 하나 = 한 모델의 (seed, n1, 환경) 묶음 하나 = 모집단 3칸. job 안에서 --slot 0–2가 모집단 하나씩 동시에 돈다(GPU 1장 공유).
순서는 seed 먼저라서 일찍 끝난 결과가 9개 데이터(환경 × 모집단)와 n1 둘을 고루 덮는다. 결과는 쉬운 환경 결과 폴더에 방법 이름으로 쌓인다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse  # noqa: E402
import os  # noqa: E402

from run_easy import N1S, env_list, out_dir, run_cell  # noqa: E402

MODELS = ["tabddpm", "tabsyn"]


def units(pop_paths, seeds, models=MODELS, envs=env_list(["frequent"])):
    groups = [[(env, p, n1, s) for p in pop_paths] for s in seeds for n1 in N1S for env in envs]
    return [(m, g) for g in groups for m in models]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--pops", default="results/pops")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--levels", nargs="+", default=["frequent"], choices=["frequent", "mid", "rare"])
    ap.add_argument("--quick", action="store_true", help="연결 확인: fav, 모집단 0, n1=400, seed 0에서 두 모델을 스모크 설정으로")
    args = ap.parse_args()
    pop_paths = sorted(Path(args.pops).glob("pop_*.npz"))[:3]
    if args.quick:
        smoke = Path(os.environ.get("BL_SRC", "/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime/baselines_src")) / "configs" / "models.smoke.json"
        run_cell(pop_paths[0], 400, 0, args.models, "results/dl_smoke", env="frequent__fav", base_config=smoke)
        return
    model, group = units(pop_paths, args.seeds, args.models, env_list(args.levels))[args.task_id]
    env, p, n1, s = group[args.slot]
    print("task", args.task_id, "slot", args.slot, "model", model, "env", env, "pop", p.name, "n1", n1, "seed", s,
          "job", os.environ.get("SLURM_JOB_ID"), "node", os.environ.get("SLURMD_NODENAME"), flush=True)
    run_cell(p, n1, s, [model], out_dir(env), env=env)


if __name__ == "__main__":
    main()
