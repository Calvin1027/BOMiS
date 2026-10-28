"""ForestDiffusion(Forest-Flow) 실행기. 과거 파이프라인(Baseline_exp_v4)의 ForestFlowAdapter를 수정 없이 그대로 쓴다.

과거 어댑터는 범주형 열만 있는 표도 받는다(범주를 정수 코드로 바꿔 cat_indexes로 넘기고, minority 갈래에서 직접 생성).
출력은 categorical_baselines CLI와 같은 계약(synthetic_minority.csv.gz, status.json)을 따른다.
과거 파이프라인 env(PYTHONPATH에 Baseline_exp_v4/src, generation)에서 돈다.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd
from baseline_exp.adapters.forest_flow import ForestFlowAdapter
from baseline_exp.contracts import DatasetSchema


def fit_sample(train: pd.DataFrame, opts: dict, seed: int, n: int, work: Path) -> pd.DataFrame:
    features = [c for c in train.columns if c != "target"]
    schema = DatasetSchema.from_outer_train(train, target="target", positive_label="1",
                                            numerical_columns=(), categorical_columns=tuple(features))
    params = {k: v for k, v in opts.items() if k not in ("source_dir", "patch_file")}
    adapter = ForestFlowAdapter(Path(opts["source_dir"]), Path(opts["patch_file"]), **params)
    work.mkdir(parents=True)
    adapter.fit(train, schema, model_seed=seed, work_dir=work)
    # 과거 어댑터 규칙상 학습 seed와 생성 seed는 달라야 한다
    return adapter.sample_minority(n, sampling_seed=seed + 1).frame[features]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        opts = json.loads(args.config.read_text())["models"]["forest_flow"]
        train = pd.read_csv(args.train, dtype=str, keep_default_na=False)
        frame = fit_sample(train, opts, args.seed, args.rows, args.output / "work")
        frame.to_csv(args.output / "synthetic_minority.csv.gz", index=False)
        status = {"status": "PASS", "options": opts}
    except Exception as error:
        traceback.print_exc()
        status = {"status": "FAIL", "error_type": type(error).__name__, "error": traceback.format_exc()[-3000:]}
    (args.output / "status.json").write_text(json.dumps(status))
    return 0 if status["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
