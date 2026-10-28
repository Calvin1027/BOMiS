"""TabDDPM 실행기. 과거 파이프라인(Baseline_exp_v4)의 TabDDPM 설정 파일 생성, 안정화 패치, 학습률 재시도 규칙을 그대로 쓴다.

과거 어댑터는 숫자 열이 있는 표를 전제로 X_num 파일을 늘 쓰고 생성 뒤 X_num_train.npy를 읽는다. 이번 표는 범주형 8열뿐이라
X_num 파일을 만들지 않는다(TabDDPM 원본은 X_num 파일이 없으면 숫자 열 0개로 처리한다). 과거 코드는 수정하지 않고 import만 한다.
출력은 categorical_baselines CLI와 같은 계약(synthetic_minority.csv.gz, status.json)을 따른다.
과거 파이프라인 env(PYTHONPATH에 Baseline_exp_v4/src, tabddpm_extras, generation)에서 돈다.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from baseline_exp.adapters._external import isolate_source, run_logged, write_text_new
from baseline_exp.adapters.tabddpm import _apply_gradient_clip_patch, _config


def fit_sample(train: pd.DataFrame, opts: dict, seed: int, n: int, work: Path) -> pd.DataFrame:
    repo, _ = isolate_source(Path(opts["source_dir"]), work / "_source")
    if opts.get("gradient_clip_norm") is not None:
        _apply_gradient_clip_patch(repo, float(opts["gradient_clip_norm"]), int(opts["max_nonfinite_gradient_skips"]))
    features = [c for c in train.columns if c != "target"]
    data = work / "data"
    data.mkdir(parents=True)
    x_cat = train[features].astype(str).to_numpy(dtype=object)
    y = (train["target"].astype(str) == "1").to_numpy(dtype=np.int64)
    for split in ("train", "val", "test"):  # 과거 어댑터와 같이 val/test 자리는 train 사본
        np.save(data / f"X_cat_{split}.npy", x_cat)
        np.save(data / f"y_{split}.npy", y)
    (data / "info.json").write_text(json.dumps({"name": "data", "task_type": "binclass", "n_classes": 2}))
    common = dict(data=data, numerical_count=0, train_steps=int(opts["train_steps"]),
                  diffusion_steps=int(opts["diffusion_steps"]), batch_size=int(opts["batch_size"]), model_seed=seed)
    selected = None
    for i, lr in enumerate([float(opts["learning_rate"]), *map(float, opts.get("learning_rate_backoff", []))]):
        attempt = work / f"attempt-{i:02d}"
        experiment = attempt / "experiment"
        experiment.mkdir(parents=True)
        config = attempt / "train.toml"
        write_text_new(config, _config(experiment=experiment, learning_rate=lr, sampling_seed=seed, n=1, **common))
        run_logged([sys.executable, "scripts/pipeline.py", "--config", str(config), "--train"],
                   cwd=repo, log_path=attempt / "train.log", python_path=repo)
        losses = pd.read_csv(experiment / "loss.csv").select_dtypes(include=[np.number]).to_numpy(dtype=float)
        status = json.loads((experiment / "training_status.json").read_text())
        steps = common["train_steps"]
        # 과거 어댑터의 성공 기준: 유한 loss, 정해진 업데이트 수 완료, 비유한 기울기 건너뜀 수 한도 이내
        ok = (losses.size > 0 and np.isfinite(losses).all() and status.get("status") == "SUCCESS"
              and int(status.get("completed_steps", -1)) == steps
              and int(status.get("completed_optimizer_updates", steps)) == steps
              and status.get("finite_loss") is True
              and int(status.get("skipped_nonfinite_gradient_steps", 0)) <= int(opts.get("max_nonfinite_gradient_skips", 0)))
        print("attempt", i, "lr", lr, "ok", ok, "status", status.get("status"), flush=True)
        if ok:
            selected = (experiment, lr)
            break
    if selected is None:
        raise RuntimeError("TRAINING_STABILITY_LIMIT_EXCEEDED: no learning rate attempt completed")
    experiment, lr = selected
    config = work / "sample.toml"
    # 과거 어댑터처럼 학습 seed와 생성 seed를 다르게 둔다
    write_text_new(config, _config(experiment=experiment, learning_rate=lr, sampling_seed=seed + 1, n=n, **common))
    run_logged([sys.executable, "scripts/pipeline.py", "--config", str(config), "--sample"],
               cwd=repo, log_path=work / "sample.log", python_path=repo, stage="sample")
    x = np.load(experiment / "X_cat_train.npy", allow_pickle=True)
    y_gen = np.load(experiment / "y_train.npy", allow_pickle=True).reshape(-1)
    if x.shape != (n, len(features)) or y_gen.shape != (n,):
        raise RuntimeError(f"RAW_SCHEMA_MISMATCH x={x.shape} y={y_gen.shape}")
    if not np.array_equal(y_gen.astype(float), np.ones(n)):
        raise RuntimeError("CONDITION_FAILED: TabDDPM returned nonminority rows")
    return pd.DataFrame(x, columns=features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"status": "FAIL"}
    try:
        opts = json.loads(args.config.read_text())["models"]["tabddpm"]
        train = pd.read_csv(args.train, dtype=str, keep_default_na=False)
        frame = fit_sample(train, opts, args.seed, args.rows, args.output / "work")
        frame.to_csv(args.output / "synthetic_minority.csv.gz", index=False)
        status = {"status": "PASS", "options": opts}
    except Exception as error:
        traceback.print_exc()
        logs = sorted((args.output / "work").rglob("*.log"))
        tail = logs[-1].read_text()[-1500:] if logs else ""
        status = {"status": "FAIL", "error_type": type(error).__name__, "error": f"{error} | log tail: {tail}"}
    (args.output / "status.json").write_text(json.dumps(status))
    return 0 if status["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
