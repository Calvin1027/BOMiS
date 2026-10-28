from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .job_matrix import matrix
from .provenance import sha256_file, write_json_new
from .revision import PROJECT_REVISION, PROTOCOL_ID


Cell = tuple[str, str, int]


def _write_tsv(path: Path, cells: Iterable[Cell]) -> int:
    rows = list(cells)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for dataset, model, seed in rows:
            handle.write(f"{dataset}\t{model}\t{seed}\n")
    return len(rows)


def _pool(cells: list[Cell], pool: str, shares: dict[str, int]) -> list[Cell]:
    """Partition cells deterministically using the execution-configured pool shares."""
    if set(shares) != {"default", "normal"} or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in shares.values()
    ):
        raise ValueError("GPU pool shares must be positive integers for default and normal")
    if pool not in shares:
        raise ValueError(f"unknown GPU pool: {pool}")
    total = shares["default"] + shares["normal"]
    if pool == "default":
        accepted = range(0, shares["default"])
    else:
        accepted = range(shares["default"], total)
    return [cell for index, cell in enumerate(cells) if index % total in accepted]


def build(experiment_path: Path, models_path: Path, execution_path: Path, output_dir: Path) -> dict:
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    models = json.loads(models_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    protocol_id = PROTOCOL_ID
    if (
        experiment.get("protocol_id") != protocol_id
        or models.get("protocol_id") != protocol_id
        or experiment.get("project_revision") != PROJECT_REVISION
        or models.get("project_revision") != PROJECT_REVISION
        or execution.get("project_revision") != PROJECT_REVISION
    ):
        raise ValueError("experiment and model configs must use full-access protocol v1")
    if experiment.get("models_config") != models_path.name:
        raise ValueError("experiment is not bound to the supplied model config")
    assigned_models = (
        list(models.get("resource_groups", {}).get("gpu", []))
        + list(models.get("resource_groups", {}).get("cpu_light", []))
        + list(models.get("resource_groups", {}).get("cpu_forest", []))
    )
    if (
        experiment.get("status") != "FROZEN_FOR_EXECUTION"
        or models.get("status") != "FROZEN_FOR_EXECUTION"
        or len(assigned_models) != len(set(assigned_models))
        or set(assigned_models) != set(experiment.get("models", []))
        or set(models.get("models", {})) != set(experiment.get("models", []))
    ):
        raise ValueError("model resources/specs do not exactly partition the experiment")
    if (
        execution.get("status") != "READY_NOT_SUBMITTED"
        or execution.get("protocol_id") != protocol_id
    ):
        raise ValueError("execution config must be READY_NOT_SUBMITTED")
    if execution.get("experiment") != experiment_path.name or execution.get("models") != models_path.name:
        raise ValueError("execution config does not reference the supplied configs")
    output_dir.mkdir(parents=True, exist_ok=False)

    gate = execution["scale_gate"]
    gate_seed = int(gate["split_seed"])
    gate_datasets = set(gate["datasets"])
    gate_models = set(gate["models"])
    if gate_seed not in experiment["split"]["seeds"] or not gate_datasets.issubset(experiment["datasets"]):
        raise ValueError("scale gate is outside the frozen experiment")

    counts: dict[str, int] = {}
    gpu_models = list(models["resource_groups"]["gpu"])
    forest_models = list(models["resource_groups"].get("cpu_forest", []))
    pool_specs = execution.get("gpu_pools", {})
    pool_shares = {
        pool: spec.get("share")
        for pool, spec in pool_specs.items()
        if isinstance(spec, dict)
    }
    if set(pool_specs) != {"default", "normal"} or set(pool_shares) != {"default", "normal"}:
        raise ValueError("execution must define exactly the default and normal GPU pools")
    if (
        gate_models != set(gpu_models + forest_models)
        or set(execution.get("gpu_submission_order", [])) != set(gpu_models)
        or set(execution.get("gpu_resources", {})) != set(gpu_models)
    ):
        raise ValueError("scale gate or GPU resources do not exactly match model groups")
    for model in gpu_models:
        all_cells = [cell for cell in matrix(experiment_path, models_path, "gpu") if cell[1] == model]
        canary = [cell for cell in all_cells if cell[0] in gate_datasets and cell[2] == gate_seed]
        canary_set = set(canary)
        pilot_remainder = [cell for cell in all_cells if cell[2] == gate_seed and cell not in canary_set]
        full_remainder = [cell for cell in all_cells if cell[2] != gate_seed]
        if model not in gate_models or len(canary) != len(gate_datasets):
            raise ValueError(f"invalid GPU scale gate for {model}")
        for phase, cells in (("canary", canary), ("pilot_remainder", pilot_remainder), ("full_remainder", full_remainder)):
            for pool in ("default", "normal"):
                relative = Path(phase) / "gpu" / model / f"{pool}.tsv"
                counts[str(relative)] = _write_tsv(
                    output_dir / relative,
                    _pool(cells, pool, pool_shares),
                )

    light = matrix(experiment_path, models_path, "cpu_light")
    light_pilot = [cell for cell in light if cell[2] == gate_seed]
    light_full = [cell for cell in light if cell[2] != gate_seed]
    counts["pilot_remainder/cpu_light.tsv"] = _write_tsv(output_dir / "pilot_remainder/cpu_light.tsv", light_pilot)
    counts["full_remainder/cpu_light.tsv"] = _write_tsv(output_dir / "full_remainder/cpu_light.tsv", light_full)

    forest = (
        matrix(experiment_path, models_path, "cpu_forest")
        if forest_models
        else []
    )
    forest_canary = [cell for cell in forest if cell[0] in gate_datasets and cell[2] == gate_seed]
    forest_canary_set = set(forest_canary)
    forest_pilot = [cell for cell in forest if cell[2] == gate_seed and cell not in forest_canary_set]
    forest_full = [cell for cell in forest if cell[2] != gate_seed]
    expected_forest_canary = len(forest_models) * len(gate_datasets)
    if (
        not set(forest_models).issubset(gate_models)
        or len(forest_canary) != expected_forest_canary
    ):
        raise ValueError("invalid CPU-forest scale gate")
    counts["canary/cpu_forest.tsv"] = _write_tsv(output_dir / "canary/cpu_forest.tsv", forest_canary)
    counts["pilot_remainder/cpu_forest.tsv"] = _write_tsv(output_dir / "pilot_remainder/cpu_forest.tsv", forest_pilot)
    counts["full_remainder/cpu_forest.tsv"] = _write_tsv(output_dir / "full_remainder/cpu_forest.tsv", forest_full)

    total = sum(counts.values())
    expected = len(experiment["datasets"]) * len(experiment["models"]) * len(experiment["split"]["seeds"])
    if total != expected:
        raise RuntimeError(f"manifest coverage mismatch: expected={expected}, found={total}")
    all_rows: list[Cell] = []
    for relative in counts:
        for line in (output_dir / relative).read_text(encoding="utf-8").splitlines():
            dataset, model, seed = line.split("\t")
            all_rows.append((dataset, model, int(seed)))
    if len(set(all_rows)) != expected:
        raise RuntimeError("task manifests contain duplicate or missing cells")

    scope_dir = output_dir / "scopes"
    pilot_scope = [cell for cell in all_rows if cell[2] == gate_seed]
    _write_tsv(scope_dir / "split1.tsv", sorted(pilot_scope))
    _write_tsv(scope_dir / "full.tsv", sorted(all_rows))
    expected_pilot = len(experiment["datasets"]) * len(experiment["models"])
    if len(pilot_scope) != expected_pilot:
        raise RuntimeError(f"split-1 scope mismatch: expected={expected_pilot}, found={len(pilot_scope)}")

    # Submission aliases are deliberately excluded from `counts`: they provide
    # one resource-homogeneous task list per model for the clean split-1 pilot,
    # while the canonical manifests above remain a disjoint partition of all cells.
    for model in gpu_models:
        model_pilot = [cell for cell in pilot_scope if cell[1] == model]
        for pool in ("default", "normal"):
            _write_tsv(
                output_dir / "split1_tasks/gpu" / model / f"{pool}.tsv",
                _pool(model_pilot, pool, pool_shares),
            )
    _write_tsv(output_dir / "split1_tasks/cpu_light.tsv", [cell for cell in pilot_scope if cell[1] in set(models["resource_groups"]["cpu_light"])])
    _write_tsv(output_dir / "split1_tasks/cpu_forest.tsv", [cell for cell in pilot_scope if cell[1] in set(models["resource_groups"]["cpu_forest"])])

    # Full-run aliases combine the split-1 non-canary and later-split phases.
    # They are deliberately excluded from ``counts`` so canonical coverage
    # remains a disjoint partition while submission receives one list per
    # resource-homogeneous model/pool.
    canary_set = set(
        cell
        for cell in all_rows
        if cell[0] in gate_datasets and cell[1] in gate_models and cell[2] == gate_seed
    )
    remainder = [cell for cell in all_rows if cell not in canary_set]
    for model in gpu_models:
        model_remainder = [cell for cell in remainder if cell[1] == model]
        for pool in ("default", "normal"):
            _write_tsv(
                output_dir / "remainder/gpu" / model / f"{pool}.tsv",
                _pool(model_remainder, pool, pool_shares),
            )
    _write_tsv(
        output_dir / "all/cpu_light.tsv",
        [cell for cell in all_rows if cell[1] in set(models["resource_groups"]["cpu_light"])],
    )
    _write_tsv(
        output_dir / "remainder/cpu_forest.tsv",
        [cell for cell in remainder if cell[1] in set(models["resource_groups"]["cpu_forest"])],
    )

    payload = {
        "status": "MANIFESTS_READY_NOT_SUBMITTED",
        "project_revision": PROJECT_REVISION,
        "protocol_id": protocol_id,
        "expected_cells": expected,
        "canary_cells": len(gpu_models) * len(gate_datasets) + len(forest_canary),
        "split1_cells": len(pilot_scope),
        "counts": counts,
        "config_sha256": {
            "experiment": sha256_file(experiment_path),
            "models": sha256_file(models_path),
            "execution": sha256_file(execution_path),
        },
    }
    write_json_new(output_dir / "MANIFESTS.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.experiment, args.models, args.execution, args.output_dir), sort_keys=True))
