from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .contracts import DatasetSchema
from .provenance import sha256_file, write_json_new
from .registry import build_adapter
from .revision import PROJECT_REVISION, PROTOCOL_ID
from .seeds import assert_distinct_seed_roles, semantic_seed
from .source_validation import verify_external_source


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(
    experiment_path: Path,
    models_path: Path,
    execution_path: Path,
    data_root: Path,
    output_path: Path,
) -> dict:
    experiment, models, execution = map(_json, (experiment_path, models_path, execution_path))
    if experiment.get("status") != "FROZEN_FOR_EXECUTION":
        raise ValueError("experiment is not frozen")
    if models.get("status") != "FROZEN_FOR_EXECUTION":
        raise ValueError("models are not frozen")
    if execution.get("status") != "READY_NOT_SUBMITTED":
        raise ValueError("execution is not ready")
    protocol_id = PROTOCOL_ID
    if (
        experiment.get("protocol_id") != protocol_id
        or models.get("protocol_id") != protocol_id
        or experiment.get("project_revision") != PROJECT_REVISION
        or models.get("project_revision") != PROJECT_REVISION
        or execution.get("project_revision") != PROJECT_REVISION
    ):
        raise ValueError("split-1 preflight requires full-access protocol v1")
    if experiment.get("models_config") != models_path.name:
        raise ValueError("experiment is not bound to the supplied model config")
    if execution.get("experiment") != experiment_path.name or execution.get("models") != models_path.name:
        raise ValueError("execution references different configs")
    if len(experiment["datasets"]) != 13 or not experiment["models"]:
        raise ValueError("split-1 experiment must contain 13 datasets and at least one model")
    seeds = list(experiment["split"]["seeds"])
    if len(seeds) != 1:
        raise ValueError("split-1 preflight requires exactly one seed")
    groups = models["resource_groups"]
    assigned = groups["gpu"] + groups["cpu_light"] + groups.get("cpu_forest", [])
    if len(assigned) != len(set(assigned)) or set(assigned) != set(experiment["models"]):
        raise ValueError("model resource groups are not an exact partition")
    if set(models["models"]) != set(experiment["models"]):
        raise ValueError("model specs do not match experiment")
    project_root = experiment_path.parent.parent
    source_lock_name = experiment.get("source_lock")
    if not isinstance(source_lock_name, str) or not source_lock_name:
        raise ValueError("experiment has no bound source lock")
    source_lock = _json(project_root / "configs" / source_lock_name)
    if (
        source_lock.get("status") != "FROZEN_FOR_EXECUTION"
        or source_lock.get("protocol_id") != protocol_id
        or source_lock.get("project_revision") != PROJECT_REVISION
    ):
        raise ValueError("source lock is not frozen for full-access protocol v1")
    for model in experiment["models"]:
        spec = models["models"][model]
        verify_external_source(model, spec, source_lock, project_root=project_root)
        adapter = build_adapter(model, spec, project_root=project_root)
        if adapter.name != model:
            raise ValueError(f"adapter identity mismatch: {model}")

    if "forest_flow" in experiment["models"]:
        forest = models["models"]["forest_flow"]
        forest_resource = execution["cpu_resources"]["forest"]
        if int(forest["n_jobs"]) != int(forest_resource["cpus"]):
            raise ValueError("Forest-Flow n_jobs must equal allocated CPUs")
        if forest_resource["memory"] != "128G" or int(forest["n_batch"]) != 1:
            raise ValueError("Forest-Flow memory-safe resource contract changed")
    elif groups.get("cpu_forest"):
        raise ValueError("cpu_forest must be empty when Forest-Flow is excluded")
    if (
        "tabddpm" in experiment["models"]
        and (
            int(models["models"]["tabddpm"].get("train_steps", -1)) != 10_000
            or int(models["models"]["tabddpm"].get("diffusion_steps", -1)) != 1_000
            or float(models["models"]["tabddpm"]["learning_rate"]) > 0.0002
        )
    ):
        raise ValueError("TabDDPM frozen 10000-update/1000-timestep budget changed")
    if "tabbyflow" in experiment["models"]:
        tabby = models["models"]["tabbyflow"]
        semantics = models.get("tabbyflow_budget_semantics", {})
        if (
            int(tabby.get("train_steps", -1)) != 8_000
            or int(tabby.get("raw_batch_size", -1)) != 100_000
            or semantics.get("effective_training_unit") != "epoch"
            or int(semantics.get("effective_training_count", -1)) != 8_000
            or int(semantics.get("training_batch_size", -1)) != 4_096
            or int(semantics.get("rejection_request_chunk_size", -1)) != 100_000
            or int(semantics.get("effective_upstream_sampling_batch_size", -1)) != 10_000
        ):
            raise ValueError("TabbyFlow effective epoch and sampling-batch semantics changed")

    dataset_rows: list[dict[str, object]] = []
    seed = int(seeds[0])
    for name in experiment["datasets"]:
        dataset_path = data_root / name / f"split-{seed}" / "dataset.json"
        dataset = _json(dataset_path)
        if dataset.get("status") != "FROZEN_FOR_EXECUTION":
            raise ValueError(f"dataset not frozen: {name}")
        if dataset.get("project_revision") != PROJECT_REVISION:
            raise ValueError(f"dataset project revision mismatch: {name}")
        if dataset.get("name") != name or int(dataset.get("split_seed")) != seed:
            raise ValueError(f"dataset identity mismatch: {name}")
        if dataset.get("experiment_sha256") != sha256_file(experiment_path):
            raise ValueError(f"dataset experiment hash mismatch: {name}")
        outer_entry = dataset["splits"]["outer_train"]
        outer_path = Path(outer_entry["path"])
        if sha256_file(outer_path) != outer_entry["sha256"]:
            raise ValueError(f"outer-train hash mismatch: {name}")
        outer = pd.read_csv(outer_path)
        schema = DatasetSchema.from_outer_train(
            outer,
            target=str(dataset["target"]),
            positive_label=dataset["positive_label"],
            numerical_columns=tuple(dataset["numerical_columns"]),
            categorical_columns=tuple(dataset["categorical_columns"]),
        )
        if not schema.numerical_columns or not schema.categorical_columns:
            raise ValueError(f"dataset is not mixed-type: {name}")
        y = schema.binary_target(outer)
        minority = int(y.sum())
        majority = len(y) - minority
        expected_m = majority
        if minority <= 10 or int(dataset["M_d"]) != expected_m or expected_m < 1:
            raise ValueError(f"invalid minority-bank geometry: {name}")
        # CTAB-GAN+ concatenates transformed data and its conditional vector.
        # This conservative upper bound must fit the patched 64x64 image side.
        categorical_width = sum(len(schema.category_domains[c]) for c in schema.categorical_columns) + 2
        # A continuous column contributes at most one scalar plus ten mixture
        # components to data_dim, and those ten components may also enter n_opt.
        ctab_width_upper_bound = 2 * categorical_width + 21 * len(schema.numerical_columns)
        if ctab_width_upper_bound > 64 * 64:
            raise ValueError(f"CTAB-GAN+ transformed width exceeds 64x64: {name}")
        for model in experiment["models"]:
            model_seed = semantic_seed(
                int(experiment["model_seed_root"]), name, model, seed, "fit"
            )
            sampling_seed = semantic_seed(
                int(experiment["sampling_seed_root"]), name, model, seed, "sample"
            )
            assert_distinct_seed_roles(model_seed, sampling_seed)
        dataset_rows.append(
            {
                "dataset": name,
                "outer_train_rows": len(outer),
                "minority_rows": minority,
                "M_d": expected_m,
                "numerical_columns": len(schema.numerical_columns),
                "categorical_columns": len(schema.categorical_columns),
                "ctab_width_upper_bound": ctab_width_upper_bound,
            }
        )
    payload = {
        "status": "PASS",
        "project_revision": PROJECT_REVISION,
        "protocol_id": protocol_id,
        "expected_cells": len(experiment["datasets"]) * len(experiment["models"]),
        "split_seed": seed,
        "datasets": dataset_rows,
        "config_sha256": {
            "experiment": sha256_file(experiment_path),
            "models": sha256_file(models_path),
            "execution": sha256_file(execution_path),
        },
    }
    write_json_new(output_path, payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.experiment, args.models, args.execution, args.data_root, args.output), sort_keys=True))
