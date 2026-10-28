"""avg/unf 환경 연결 확인: 모집단 0, n1=400, seed 0에서 A, C, ros, BOMiS(50 step)를 돌려 env별 값을 찍는다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

from run_easy import run_cell  # noqa: E402

OUT = Path("results/plumb")
ENVS = ["frequent__fav", "frequent__avg", "frequent__unf"]
for env in ENVS:
    run_cell(Path("results/pops/pop_0000.npz"), 400, 0, ["A", "C", "ros", "bomis"], OUT / env, env=env, bomis_kw={"steps": 50})
cols = ["env", "M0", "stz_sup_count", "sz_rec_mass", "all_cert", "n_false", "recovery", "leak", "diag_Q_stz", "diag_R1"]
for env in ENVS:
    for m in ["A", "C", "bomis"]:
        d = pd.read_csv(OUT / env / m / "0000_400_0.csv")
        print(env, m, d[[c for c in cols if c in d]].to_dict("records"), flush=True)
fails = list((OUT).glob("*/_failed/*/*.csv"))
print("PLUMB", "FAIL" if fails else "OK", [str(p) for p in fails])
