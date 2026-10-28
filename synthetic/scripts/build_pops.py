"""C10, C12 진입: 개발용 모집단을 seed start부터 n개 만들어 저장한다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse  # noqa: E402
import csv  # noqa: E402

from dgp.population import generate_accepted, save_population  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--out", default="results/pops")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
pops, rejected = generate_accepted(args.start, args.n)
for p in pops:
    save_population(p, out / f"pop_{p.seed:04d}.npz")
with open(out / "rejections.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed", "reason"])
    w.writerows(rejected)
print("accepted", [p.seed for p in pops], "rejected", rejected, flush=True)
