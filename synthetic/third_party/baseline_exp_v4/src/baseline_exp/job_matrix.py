from __future__ import annotations

import argparse
import json
from pathlib import Path

from .revision import PROJECT_REVISION, PROTOCOL_ID


def matrix(experiment_path: Path, models_path: Path, group: str) -> list[tuple[str, str, int]]:
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    models = json.loads(models_path.read_text(encoding="utf-8"))
    if experiment.get("status") != "FROZEN_FOR_EXECUTION" or models.get("status") != "FROZEN_FOR_EXECUTION":
        raise ValueError("experiment and model configs must be frozen")
    protocol_id = PROTOCOL_ID
    if (
        experiment.get("protocol_id") != protocol_id
        or models.get("protocol_id") != protocol_id
        or experiment.get("project_revision") != PROJECT_REVISION
        or models.get("project_revision") != PROJECT_REVISION
    ):
        raise ValueError("experiment and models must use full-access protocol v1")
    if experiment.get("models_config") != models_path.name:
        raise ValueError("experiment is not bound to the supplied model config")
    selected_models = list(models["resource_groups"][group])
    if not selected_models or not set(selected_models).issubset(experiment["models"]):
        raise ValueError(f"invalid model resource group {group!r}")
    if not set(selected_models).issubset(models["models"]):
        raise ValueError(f"resource group {group!r} references missing model")
    return [
        (dataset, model, int(seed))
        for seed in experiment["split"]["seeds"]
        for dataset in experiment["datasets"]
        for model in selected_models
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--group", choices=("gpu", "cpu"), required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--count", action="store_true")
    action.add_argument("--index", type=int)
    args = parser.parse_args()
    cells = matrix(args.experiment, args.models, args.group)
    if args.count:
        print(len(cells))
    else:
        try:
            print("\t".join(map(str, cells[args.index])))
        except IndexError as error:
            raise SystemExit(f"array index {args.index} outside [0, {len(cells) - 1}]") from error
