import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from .contracts import (
    AdapterFailure,
    sample_result_metadata,
    validate_fit_artifacts,
    validate_fit_contract,
)
from .data import categorical_schema, read_train
from .registry import MODELS, make_model
from .seeds import semantic_seed


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fit one categorical-only generator; emit minority X"
    )
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        config = json.loads(args.config.read_text())
        options = dict(config["models"][args.model])
        if args.model in {"ctgan", "tvae"}:
            options["require_cuda"] = args.device == "cuda"
        elif args.model not in {"smote", "smoten", "smote_enc"}:
            options["device"] = args.device
        import torch

        torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))))
        train = read_train(args.train)
        schema = categorical_schema(train)
        code_hash = hashlib.sha256()
        package_root = Path(__file__).parent
        for source in sorted(package_root.rglob("*.py")):
            code_hash.update(str(source.relative_to(package_root)).encode())
            code_hash.update(source.read_bytes())
        model_seed = semantic_seed(args.seed, args.model, "fit")
        sample_seed = semantic_seed(args.seed, args.model, "sample")
        metadata = {
            "model": args.model,
            "model_variant": (
                "SMOTE-OHE-argmax"
                if args.model == "smote"
                else "TVAE-probabilistic-target"
                if args.model == "tvae" and options.get("target_decode") == "probabilistic_softmax"
                else args.model
            ),
            "adapter_source_sha256": code_hash.hexdigest(),
            "source_train": str(args.train.resolve()),
            "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "options": options,
            "root_seed": args.seed,
            "fit_seed": model_seed,
            "sample_seed": sample_seed,
            "n_rows": len(train),
            "n_minority": int(train.target.sum()),
            "n_features": len(schema.feature_columns),
            "n_numerical": 0,
            "validation_or_test_access": False,
            "purpose": config["purpose"],
            "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "packages": {
                p: importlib.metadata.version(p)
                for p in ("torch", "numpy", "pandas", "scikit-learn", "ctgan", "imbalanced-learn")
            },
        }
        write_json(args.output / "run.json", metadata)
        adapter = make_model(args.model, **options)
        result = adapter.fit(train, schema, model_seed=model_seed, work_dir=args.output / "fit")
        validate_fit_contract(result, expected_model=args.model, expected_outer_train_n=len(train))
        validate_fit_artifacts(result, work_dir=args.output / "fit")
        write_json(args.output / "fit.json", asdict(result))
        fitted = time.monotonic()
        generated = adapter.sample_minority(args.rows, sampling_seed=sample_seed)
        generated.frame.to_csv(args.output / "synthetic_minority.csv.gz", index=False)
        write_json(args.output / "sample.json", sample_result_metadata(generated))
        write_json(
            args.output / "status.json",
            {
                "status": "PASS",
                "rows": len(generated.frame),
                "fit_seconds": fitted - start,
                "sample_seconds": time.monotonic() - fitted,
            },
        )
        print(json.dumps({"status": "PASS", "model": args.model, "rows": args.rows}), flush=True)
    except Exception as error:
        write_json(
            args.output / "status.json",
            {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "stage": getattr(error, "stage", None),
                "code": getattr(error, "code", None),
                "context": error.context if isinstance(error, AdapterFailure) else {},
                "seconds": time.monotonic() - start,
            },
        )
        traceback.print_exc()
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
