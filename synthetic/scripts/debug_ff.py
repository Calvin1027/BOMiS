"""ForestDiffusion 실패 원인 확인: fav 모집단 0, n1=400, seed 0의 학습표로 forestflow_cli를 본 설정으로 돌리고 전체 출력을 로그로 남긴다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

from baselines import external as ex  # noqa: E402
from dgp.population import load_population  # noqa: E402
from dgp.sampling import make_split  # noqa: E402

work = Path(os.environ.get("SLURM_TMPDIR") or "/tmp") / f"ff_debug_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
work.mkdir(parents=True, exist_ok=True)
pop = load_population("results/pops/pop_0000.npz")
split = make_split(pop, "frequent__fav", 400, 0)
ex.train_table(split.D0, split.D1).to_csv(work / "train.csv", index=False)
(work / "config.json").write_text(json.dumps({"models": {"forest_flow": ex.FORESTFLOW["reference"]}}))
env = dict(os.environ, PYTHONPATH=f"{ex.V4_SRC}:{ex.TABDDPM_SITE}:{ex.GEN_SITE}", PYTHONDONTWRITEBYTECODE="1")
for k in ("JOBLIB_TEMP_FOLDER",):
    print(k, env.get(k), flush=True)
cmd = [ex.BL_PYTHON, str(Path(ex.__file__).resolve().with_name("forestflow_cli.py")), "--train", str(work / "train.csv"),
       "--config", str(work / "config.json"), "--seed", "0", "--rows", str(len(split.D0_idx) - len(split.D1_idx)),
       "--output", str(work / "out")]
code = subprocess.run(cmd, cwd=work, env=env).returncode
print("exit", code, flush=True)
print((work / "out" / "status.json").read_text()[:4000] if (work / "out" / "status.json").exists() else "no status")
