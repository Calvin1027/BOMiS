from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .provenance import sha256_file, write_json_new
from .revision import PROJECT_REVISION, PROTOCOL_ID
from .adapters._external import tree_sha256
from .final_integrity import SUMMARY_METRICS, validate_cell_integrity


REQUIRED_FINITE_METRICS = (
    "fidelity.mean_numerical_ks",
    "fidelity.mean_categorical_tvd",
    "fidelity.mean_category_coverage",
    "fidelity.categorical_tuple_validity",
    "fidelity.alpha_precision",
    "fidelity.beta_recall",
    "downstream.validation_mcc",
    "downstream.test_mcc",
    "downstream.test_pr_auc",
)


def _flatten_numeric(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten_numeric(f"{prefix}.{key}" if prefix else str(key), item, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)


def aggregate(
    experiment_path: Path,
    generation_root: Path,
    evaluation_root: Path,
    output_dir: Path,
) -> Path:
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("status") != "FROZEN_FOR_EXECUTION":
        raise RuntimeError("final experiment is not frozen")
    protocol_id = PROTOCOL_ID
    if (
        experiment.get("protocol_id") != protocol_id
        or experiment.get("project_revision") != PROJECT_REVISION
    ):
        raise RuntimeError("final experiment protocol is not full-access v1")
    models_reference = experiment.get("models_config")
    if not isinstance(models_reference, str):
        raise RuntimeError("final experiment has no bound model config")
    models_path = experiment_path.parent / models_reference
    if not models_path.is_file():
        raise RuntimeError(f"bound model config is missing: {models_path}")
    models = json.loads(models_path.read_text(encoding="utf-8"))
    assigned_models = (
        list(models.get("resource_groups", {}).get("gpu", []))
        + list(models.get("resource_groups", {}).get("cpu_light", []))
        + list(models.get("resource_groups", {}).get("cpu_forest", []))
    )
    if (
        models.get("status") != "FROZEN_FOR_EXECUTION"
        or models.get("protocol_id") != protocol_id
        or models.get("project_revision") != PROJECT_REVISION
        or len(assigned_models) != len(set(assigned_models))
        or set(assigned_models) != set(experiment["models"])
        or set(models.get("models", {})) != set(experiment["models"])
    ):
        raise RuntimeError("bound model config does not exactly match the experiment")
    expected_models_hash = sha256_file(models_path)
    expected_hash = sha256_file(experiment_path)
    rows: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    evaluator_source_hashes: set[str] = set()
    frozen_source_hash = tree_sha256(Path(__file__).resolve().parent)
    for seed in experiment["split"]["seeds"]:
        for dataset in experiment["datasets"]:
            for model in experiment["models"]:
                cell = evaluation_root / dataset / model / f"split-{seed}"
                generated = generation_root / dataset / model / f"split-{seed}"
                marker_path, metrics_path = cell / "SUCCESS.json", cell / "metrics.json"
                generated_marker_path = generated / "GENERATED.json"
                if not marker_path.is_file() or not metrics_path.is_file() or not generated_marker_path.is_file():
                    missing.append(str(cell))
                    continue
                generated_marker, marker, _ = validate_cell_integrity(
                    generated, cell, dataset=dataset, model=model, seed=int(seed),
                    experiment_hash=expected_hash, models_hash=expected_models_hash,
                    source_hash=frozen_source_hash,
                )
                if (
                    marker.get("status") != "SUCCESS"
                    or marker.get("dataset") != dataset
                    or marker.get("model") != model
                    or int(marker.get("split_seed", -1)) != int(seed)
                    or marker.get("experiment_sha256") != expected_hash
                    or marker.get("protocol_id") != protocol_id
                    or marker.get("project_revision") != PROJECT_REVISION
                    or marker.get("models_config_sha256") != expected_models_hash
                    or marker.get("generation_marker_sha256") != sha256_file(generated_marker_path)
                ):
                    raise RuntimeError(f"invalid evaluation provenance: {cell}")
                csv_path = generated / "synthetic_minority.csv"
                parquet_path = generated / "synthetic_minority.parquet"
                if (
                    generated_marker.get("status") != "GENERATED_UNVERIFIED"
                    or generated_marker.get("dataset") != dataset
                    or generated_marker.get("model") != model
                    or generated_marker.get("project_revision") != PROJECT_REVISION
                    or generated_marker.get("protocol_id") != protocol_id
                    or generated_marker.get("config_sha256", {}).get("experiment") != expected_hash
                    or generated_marker.get("config_sha256", {}).get("models") != expected_models_hash
                    or not csv_path.is_file()
                    or not parquet_path.is_file()
                    or sha256_file(csv_path) != generated_marker.get("synthetic_csv_file_sha256")
                    or sha256_file(parquet_path) != generated_marker.get("synthetic_file_sha256")
                ):
                    raise RuntimeError(f"invalid generation provenance: {generated}")
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                source_hash = marker.get("benchmark_source_tree_sha256")
                if not isinstance(source_hash, str) or len(source_hash) != 64:
                    raise RuntimeError(f"missing evaluator source identity: {cell}")
                evaluator_source_hashes.add(source_hash)
                row: dict[str, Any] = {"dataset": dataset, "model": model, "split_seed": int(seed)}
                _flatten_numeric("", metrics, row)
                for required in REQUIRED_FINITE_METRICS:
                    value = row.get(required)
                    if value is None or not math.isfinite(float(value)):
                        raise RuntimeError(f"required metric is missing/nonfinite: {required}: {cell}")
                catcat = metrics.get("fidelity", {}).get("mean_pair_similarity_categorical_categorical")
                catcat_pairs = metrics.get("fidelity", {}).get("defined_pairs_categorical_categorical")
                if (catcat is None) != (int(catcat_pairs) == 0):
                    raise RuntimeError(f"categorical-categorical undefinedness mismatch: {cell}")
                rows.append(row)
                generated_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "split_seed": int(seed),
                        "M_d": int(generated_marker["M_d"]),
                        "relative_csv_path": str(csv_path.relative_to(generation_root.parent)),
                        "relative_parquet_path": str(parquet_path.relative_to(generation_root.parent)),
                        "synthetic_csv_sha256": generated_marker["synthetic_csv_file_sha256"],
                        "synthetic_parquet_sha256": generated_marker["synthetic_file_sha256"],
                        "synthetic_dataframe_sha256": generated_marker["synthetic_sha256"],
                        "generator_source_tree_sha256": generated_marker["benchmark_source_tree_sha256"],
                    }
                )
    expected = len(experiment["split"]["seeds"]) * len(experiment["datasets"]) * len(experiment["models"])
    if missing or len(rows) != expected:
        raise RuntimeError(f"final evaluation incomplete: expected={expected}, found={len(rows)}, missing={len(missing)}")
    if len(evaluator_source_hashes) != 1:
        raise RuntimeError(
            f"final evaluation mixes evaluator source identities: {sorted(evaluator_source_hashes)}"
        )
    if len(generated_rows) != expected:
        raise RuntimeError(f"final generated-sample index incomplete: expected={expected}, found={len(generated_rows)}")
    cells = pd.DataFrame(rows).sort_values(["dataset", "model", "split_seed"]).reset_index(drop=True)
    # Do not average seeds, row counts, or fitted decision thresholds as performance metrics.
    metric_columns = list(SUMMARY_METRICS)
    for column in metric_columns:
        if column not in cells:
            cells[column] = float("nan")
    grouped = cells.groupby(["dataset", "model"], sort=True)[metric_columns]
    means = grouped.mean().add_suffix("__mean")
    stds = grouped.std(ddof=1).add_suffix("__sd")
    counts = grouped.count().add_suffix("__n")
    dataset_summary = pd.concat([means, stds, counts], axis=1).reset_index()
    per_split_macro = cells.groupby(["model", "split_seed"], sort=True)[metric_columns].mean().reset_index()
    defined_counts = cells.groupby(["model", "split_seed"], sort=True)[metric_columns].count()
    per_split_denominators = defined_counts.add_suffix("__n_datasets").reset_index()
    macro_grouped = per_split_macro.groupby("model", sort=True)[metric_columns]
    macro_summary = pd.concat(
        [
            macro_grouped.mean().add_suffix("__dataset_macro_mean"),
            macro_grouped.std(ddof=1).add_suffix("__between_split_sd"),
            macro_grouped.count().add_suffix("__n_splits"),
            defined_counts.groupby("model").min().add_suffix("__n_datasets_min"),
            defined_counts.groupby("model").max().add_suffix("__n_datasets_max"),
        ],
        axis=1,
    ).reset_index()
    generated_index = pd.DataFrame(generated_rows).sort_values(
        ["dataset", "model", "split_seed"]
    ).reset_index(drop=True)
    # Claim the final aggregate namespace only after every input has passed validation.
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in (
        ("cell_metrics.csv", cells),
        ("dataset_model_summary.csv", dataset_summary),
        ("model_dataset_macro_summary.csv", macro_summary),
        ("generated_samples_index.csv", generated_index),
        ("model_split_defined_dataset_counts.csv", per_split_denominators),
    ):
        with (output_dir / name).open("x", encoding="utf-8") as handle:
            frame.to_csv(handle, index=False)
    write_json_new(
        output_dir / "AGGREGATED.json",
        {
            "status": "AGGREGATED_DESCRIPTIVE",
            "project_revision": PROJECT_REVISION,
            "expected_cells": expected,
            "observed_cells": len(cells),
            "datasets": experiment["datasets"],
            "models": experiment["models"],
            "split_seeds": experiment["split"]["seeds"],
            "inference_note": experiment["inference_note"],
            "undefined_metric_policy": experiment["undefined_metric_policy"],
            "experiment_sha256": expected_hash,
            "protocol_id": protocol_id,
            "models_config_sha256": expected_models_hash,
            "benchmark_source_tree_sha256": next(iter(evaluator_source_hashes)),
            "artifacts": {
                name: sha256_file(output_dir / name)
                for name in (
                    "cell_metrics.csv",
                    "dataset_model_summary.csv",
                    "model_dataset_macro_summary.csv",
                    "generated_samples_index.csv",
                    "model_split_defined_dataset_counts.csv",
                )
            },
        },
    )
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(aggregate(args.experiment, args.generation_root, args.evaluation_root, args.output_dir))
