"""C27 스모크: 실제 크기(n0=8000, n1=400, 7600행)로 외부 baseline 한 번씩 돌려 시간과 성공 여부를 본다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse  # noqa: E402

import numpy as np  # noqa: E402

from baselines.external import run_external  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--work", default="results/baseline_smoke")
args = ap.parse_args()
rng = np.random.default_rng([9000, 0])
D0 = rng.integers(0, 4, (8000, 8))
D1 = rng.integers(0, 4, (400, 8))
for name in args.models:
    res = run_external(name, D0, D1, 7600, seed=0, workdir=Path(args.work) / name)
    rows = res["rows"]
    print(name, "ok", res["ok"], "attempt", res["attempt"], "secs", round(res["seconds"], 1),
          "shape", None if rows is None else rows.shape, "reason", res["reason"][:800], flush=True)
