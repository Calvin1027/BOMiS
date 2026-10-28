from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import DatasetSchema
from .provenance import sha256_file, write_json_new
from .revision import PROJECT_REVISION, PROTOCOL_ID


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load(entry: dict[str, Any], data_root: Path) -> pd.DataFrame:
    path = Path(entry["path"]).resolve()
    try:
        path.relative_to(data_root.resolve())
    except ValueError as error:
        raise RuntimeError(f"prepared split escapes frozen data root: {path}") from error
    if not path.is_file() or sha256_file(path) != entry.get("sha256"):
        raise RuntimeError(f"prepared split hash mismatch: {path}")
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def validate(experiment_path: Path, models_path: Path, data_root: Path, output: Path) -> Path:
    experiment, models = _read(experiment_path), _read(models_path)
    if (
        experiment.get("status") != "FROZEN_FOR_EXECUTION"
        or models.get("status") != "FROZEN_FOR_EXECUTION"
        or experiment.get("project_revision") != PROJECT_REVISION
        or models.get("project_revision") != PROJECT_REVISION
        or experiment.get("protocol_id") != PROTOCOL_ID
        or models.get("protocol_id") != PROTOCOL_ID
        or experiment.get("models_config") != models_path.name
    ):
        raise RuntimeError("prepared-data gate received incompatible frozen configs")
    prepared = _read(data_root / "PREPARED.json")
    expected_count = len(experiment["datasets"]) * len(experiment["split"]["seeds"])
    registry_path = experiment_path.parent / experiment["dataset_registry"]
    if (
        prepared.get("status") != "PREPARED"
        or prepared.get("project_revision") != PROJECT_REVISION
        or prepared.get("manifest_count") != expected_count
        or prepared.get("experiment_sha256") != sha256_file(experiment_path)
        or prepared.get("registry_sha256") != sha256_file(registry_path)
    ):
        raise RuntimeError("PREPARED.json does not certify the complete frozen matrix")

    neighbour_requirements = [
        int(models["models"]["smotenc"]["k_neighbors"]),
        int(models["models"]["smote_enc"]["k_neighbors"]),
        int(models["models"]["borderline_smote"]["k_neighbors"]),
        int(models["models"]["adasyn"]["n_neighbors"]),
    ]
    minimum_minority = max(neighbour_requirements) + 1
    records: list[dict[str, Any]] = []
    for dataset_name in experiment["datasets"]:
        for seed in experiment["split"]["seeds"]:
            manifest_path = data_root / dataset_name / f"split-{int(seed)}" / "dataset.json"
            manifest = _read(manifest_path)
            if (
                manifest.get("status") != "FROZEN_FOR_EXECUTION"
                or manifest.get("project_revision") != PROJECT_REVISION
                or manifest.get("name") != dataset_name
                or int(manifest.get("split_seed", -1)) != int(seed)
                or manifest.get("experiment_sha256") != sha256_file(experiment_path)
                or manifest.get("registry_sha256") != sha256_file(registry_path)
            ):
                raise RuntimeError(f"invalid dataset manifest provenance: {manifest_path}")
            frames = {role: _load(manifest["splits"][role], data_root) for role in ("outer_train", "validation", "test")}
            train = frames["outer_train"]
            schema = DatasetSchema.from_outer_train(
                train,
                target=str(manifest["target"]),
                positive_label=manifest["positive_label"],
                numerical_columns=tuple(manifest["numerical_columns"]),
                categorical_columns=tuple(manifest["categorical_columns"]),
            )
            if tuple(manifest.get("feature_columns", [])) != schema.feature_columns:
                raise RuntimeError(f"feature order mismatch: {dataset_name}/{seed}")
            if not schema.numerical_columns or not schema.categorical_columns:
                raise RuntimeError(f"dataset is not mixed type: {dataset_name}/{seed}")
            counts: dict[str, dict[str, int]] = {}
            for role, frame in frames.items():
                if tuple(frame.columns) != (*schema.feature_columns, schema.target):
                    raise RuntimeError(f"split schema mismatch: {dataset_name}/{seed}/{role}")
                y = schema.binary_target(frame)
                minority, majority = int(y.sum()), int((1 - y).sum())
                required_minority = minimum_minority if role == "outer_train" else 2
                if minority < required_minority or majority < 2 or (role == "outer_train" and minority >= majority):
                    raise RuntimeError(
                        f"class/neighbour precondition failed: {dataset_name}/{seed}/{role}"
                    )
                for column in schema.numerical_columns:
                    if not np.isfinite(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)).all():
                        raise RuntimeError(f"nonfinite prepared value: {dataset_name}/{seed}/{role}/{column}")
                if frame.loc[:, list(schema.categorical_columns)].isna().any().any():
                    raise RuntimeError(f"missing prepared category: {dataset_name}/{seed}/{role}")
                counts[role] = {"rows": len(frame), "minority": minority, "majority": majority}
            if int(manifest.get("M_d", -1)) != counts["outer_train"]["majority"]:
                raise RuntimeError(f"M_d mismatch: {dataset_name}/{seed}")
            records.append(
                {
                    "dataset": dataset_name,
                    "split_seed": int(seed),
                    "M_d": int(manifest["M_d"]),
                    "outer_train_minority": counts["outer_train"]["minority"],
                    "outer_train_majority": counts["outer_train"]["majority"],
                    "numerical_columns": len(schema.numerical_columns),
                    "categorical_columns": len(schema.categorical_columns),
                    "manifest_sha256": sha256_file(manifest_path),
                }
            )
    identities = {(item["dataset"], item["split_seed"]) for item in records}
    if len(records) != expected_count or len(identities) != expected_count:
        raise RuntimeError("prepared-data coverage is missing or duplicated")
    write_json_new(
        output,
        {
            "status": "PASS",
            "project_revision": PROJECT_REVISION,
            "protocol_id": PROTOCOL_ID,
            "expected_manifests": expected_count,
            "validated_manifests": len(records),
            "minimum_outer_train_minority_required": minimum_minority,
            "minimum_evaluation_class_count_required": 2,
            "experiment_sha256": sha256_file(experiment_path),
            "models_config_sha256": sha256_file(models_path),
            "records": records,
        },
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(validate(args.experiment, args.models, args.data_root, args.output))
