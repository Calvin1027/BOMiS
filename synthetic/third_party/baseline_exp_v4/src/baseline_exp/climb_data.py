from __future__ import annotations

import argparse
import io
import json
import traceback
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .provenance import sha256_file, write_json_new
from .revision import PROJECT_REVISION, PROTOCOL_ID


MISSING_TOKEN = "__MISSING__"
SOURCE_MISSING_TOKENS = {"?", "", "nan", "NaN", "<NA>", "None", "null", "NULL"}


def _normalize_categorical(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .replace({token: MISSING_TOKEN for token in SOURCE_MISSING_TOKENS})
        .fillna(MISSING_TOKEN)
        .astype(str)
    )


@dataclass(frozen=True)
class SourceBundle:
    description: bytes
    features: bytes
    parquet: bytes


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "tdg-baseline-exp/0.1"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def fetch_source_bundle(spec: dict[str, Any]) -> SourceBundle:
    did = int(spec["openml_id"])
    return SourceBundle(
        description=_fetch(f"https://www.openml.org/api/v1/json/data/{did}"),
        features=_fetch(f"https://www.openml.org/api/v1/json/data/features/{did}"),
        parquet=_fetch(spec["parquet_url"]),
    )


def frozen_source_bundle(name: str, lock: dict[str, Any], project_root: Path) -> SourceBundle:
    """Read the same hash-locked raw OpenML bytes for every split, never old split tables."""
    entry = lock["datasets"][name]
    blobs = {}
    for kind in ("description", "features", "parquet"):
        path = project_root / entry[kind]["path"]
        if sha256_file(path) != entry[kind]["sha256"]:
            raise RuntimeError(f"frozen raw source hash mismatch: {name}/{kind}")
        blobs[kind] = path.read_bytes()
    return SourceBundle(**blobs)


def _canonical_target(values: pd.Series, dataset_name: str) -> tuple[pd.Series, str, dict[str, int]]:
    target = values.astype("string").str.strip()
    if target.isna().any():
        raise ValueError(f"{dataset_name}: target contains missing values")
    counts = target.value_counts(dropna=False)
    if len(counts) != 2:
        raise ValueError(f"{dataset_name}: expected binary target, found {len(counts)} classes")
    if counts.iloc[0] == counts.iloc[1]:
        raise ValueError(f"{dataset_name}: minority class is not unique")
    minority = str(counts.idxmin())
    binary = target.eq(minority).astype(np.int8)
    return binary, minority, {str(key): int(value) for key, value in counts.items()}


def _apply_transforms(
    frame: pd.DataFrame, spec: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series | None, dict[str, str]]:
    result = frame.copy()
    groups: pd.Series | None = None
    group_column = spec.get("group_column")
    if group_column:
        if group_column not in result:
            raise ValueError(f"{spec['name']}: missing group column {group_column!r}")
        groups = result[group_column].astype("string")
        if groups.isna().any():
            raise ValueError(f"{spec['name']}: group column contains missing values")

    parsed_dates: dict[str, pd.Series] = {}
    new_types: dict[str, str] = {}
    for transform in spec.get("datetime_transforms", []):
        column = transform["column"]
        if column not in result:
            raise ValueError(f"{spec['name']}: missing datetime column {column!r}")
        parsed = pd.to_datetime(result[column], errors="coerce", utc=True)
        if parsed.isna().any():
            raise ValueError(f"{spec['name']}: invalid datetime values in {column!r}")
        parsed_dates[column] = parsed
        prefix = transform["prefix"]
        for component in transform["components"]:
            output = f"{prefix}_{component}"
            if component == "month":
                result[output] = parsed.dt.month.astype("string")
            elif component == "day_of_week":
                result[output] = parsed.dt.dayofweek.astype("string")
            elif component == "hour":
                result[output] = parsed.dt.hour.astype("string")
            else:
                raise ValueError(f"unsupported datetime component: {component}")
            new_types[output] = "categorical"

    for transform in spec.get("datetime_differences", []):
        left = transform["left"]
        right = transform["right"]
        if left not in parsed_dates or right not in parsed_dates:
            raise ValueError("datetime difference columns must also have datetime_transforms")
        output = transform["output"]
        result[output] = (
            (parsed_dates[left] - parsed_dates[right]).dt.total_seconds() / 86400.0
        )
        new_types[output] = "numerical"

    drops = set(spec.get("drop_columns", []))
    drops.update(item["column"] for item in spec.get("datetime_transforms", []))
    if group_column:
        drops.add(group_column)
    missing_drops = sorted(drops.difference(result.columns))
    if missing_drops:
        raise ValueError(f"{spec['name']}: requested drop columns absent: {missing_drops}")
    result = result.drop(columns=list(drops))
    if result.columns.duplicated().any():
        raise ValueError(f"{spec['name']}: duplicate columns after transforms")
    return result, groups, new_types


def _assign_types(
    frame: pd.DataFrame,
    feature_metadata: dict[str, dict[str, Any]],
    spec: dict[str, Any],
    new_types: dict[str, str],
) -> tuple[list[str], list[str]]:
    categorical_overrides = set(spec.get("categorical_overrides", []))
    numerical_overrides = set(spec.get("numerical_overrides", []))
    if categorical_overrides.intersection(numerical_overrides):
        raise ValueError(f"{spec['name']}: overlapping type overrides")
    unknown_overrides = (categorical_overrides | numerical_overrides).difference(frame.columns)
    if unknown_overrides:
        raise ValueError(f"{spec['name']}: type overrides absent: {sorted(unknown_overrides)}")

    numerical: list[str] = []
    categorical: list[str] = []
    for column in frame.columns:
        if column in categorical_overrides:
            categorical.append(column)
        elif column in numerical_overrides:
            numerical.append(column)
        elif column in new_types:
            (categorical if new_types[column] == "categorical" else numerical).append(column)
        else:
            metadata = feature_metadata.get(column)
            if metadata is None:
                raise ValueError(f"{spec['name']}: no OpenML metadata for {column!r}")
            data_type = metadata["data_type"]
            (categorical if data_type in {"nominal", "string"} else numerical).append(column)
    if not numerical or not categorical:
        raise ValueError(
            f"{spec['name']}: not mixed-type after audit "
            f"(numerical={len(numerical)}, categorical={len(categorical)})"
        )
    return numerical, categorical


def _split_indices(
    target: pd.Series,
    groups: pd.Series | None,
    seed: int,
) -> dict[str, np.ndarray]:
    indices = np.arange(len(target))
    if groups is None:
        train_valid, test = train_test_split(
            indices, test_size=0.15, random_state=seed, stratify=target
        )
        train, validation = train_test_split(
            train_valid,
            test_size=0.15 / 0.85,
            random_state=seed + 1,
            stratify=target.iloc[train_valid],
        )
        return {"outer_train": train, "validation": validation, "test": test}

    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold = np.full(len(target), -1, dtype=int)
    placeholder = np.zeros((len(target), 1), dtype=np.int8)
    for fold_id, (_, held_out) in enumerate(splitter.split(placeholder, target, groups)):
        fold[held_out] = fold_id
    if (fold < 0).any():
        raise RuntimeError("group fold assignment was incomplete")
    return {
        "outer_train": indices[fold >= 6],
        "validation": indices[(fold >= 3) & (fold < 6)],
        "test": indices[fold < 3],
    }


def _subsample(
    frame: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series | None,
    max_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    if max_rows is None or len(frame) <= max_rows:
        return frame.reset_index(drop=True), target.reset_index(drop=True), (
            None if groups is None else groups.reset_index(drop=True)
        )
    if groups is not None:
        raise ValueError("row subsampling is forbidden for grouped datasets")
    selected, _ = train_test_split(
        np.arange(len(frame)), train_size=max_rows, random_state=seed, stratify=target
    )
    return (
        frame.iloc[selected].reset_index(drop=True),
        target.iloc[selected].reset_index(drop=True),
        None,
    )


def prepare_dataset(
    spec: dict[str, Any],
    destination: Path,
    split_seed: int,
    *,
    registry_path: Path,
    experiment_path: Path,
    max_rows: int | None,
    source_bundle: SourceBundle | None = None,
) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    did = int(spec["openml_id"])
    bundle = source_bundle if source_bundle is not None else fetch_source_bundle(spec)
    description_bytes = bundle.description
    features_bytes = bundle.features
    parquet_bytes = bundle.parquet
    for name, payload in (
        ("openml_description.json", description_bytes),
        ("openml_features.json", features_bytes),
        ("openml_source.parquet", parquet_bytes),
    ):
        with (destination / name).open("xb") as handle:
            handle.write(payload)

    description = json.loads(description_bytes)["data_set_description"]
    if (
        description["name"] != spec["openml_name"]
        or int(description["version"]) != int(spec["openml_version"])
        or description["default_target_attribute"] != spec["target"]
        or description["status"] != "active"
    ):
        raise ValueError(f"{spec['name']}: OpenML identity mismatch")

    feature_items = json.loads(features_bytes)["data_features"]["feature"]
    feature_metadata = {item["name"]: item for item in feature_items}
    source = pd.read_parquet(io.BytesIO(parquet_bytes))
    if spec["target"] not in source:
        raise ValueError(f"{spec['name']}: target missing from parquet")
    binary_target, raw_positive, raw_counts = _canonical_target(
        source[spec["target"]], spec["name"]
    )
    features = source.drop(columns=[spec["target"]])
    transformed, groups, new_types = _apply_transforms(features, spec)
    transformed, binary_target, groups = _subsample(
        transformed, binary_target, groups, max_rows, split_seed
    )
    numerical, categorical = _assign_types(
        transformed, feature_metadata, spec, new_types
    )
    # Preserve the transformed source feature order. Numerical/categorical lists remain separate
    # semantic annotations and must not silently reorder the user-visible table.
    feature_order = list(transformed.columns)
    clean = transformed.loc[:, feature_order].copy()
    for column in numerical:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean.loc[~np.isfinite(clean[column]), column] = np.nan
    for column in categorical:
        clean[column] = _normalize_categorical(clean[column])
    clean["target"] = binary_target.to_numpy(dtype=np.int8)

    split_indices = _split_indices(clean["target"], groups, split_seed)
    if set(np.concatenate(list(split_indices.values()))) != set(range(len(clean))):
        raise RuntimeError(f"{spec['name']}: split coverage failure")
    if sum(len(value) for value in split_indices.values()) != len(clean):
        raise RuntimeError(f"{spec['name']}: split overlap failure")
    if groups is not None:
        group_sets = {role: set(groups.iloc[idx]) for role, idx in split_indices.items()}
        if any(
            group_sets[left].intersection(group_sets[right])
            for left, right in (("outer_train", "validation"), ("outer_train", "test"), ("validation", "test"))
        ):
            raise RuntimeError(f"{spec['name']}: group leakage across splits")

    train = clean.iloc[split_indices["outer_train"]].copy()
    medians = {column: float(train[column].median()) for column in numerical}
    if not all(np.isfinite(value) for value in medians.values()):
        raise ValueError(f"{spec['name']}: nonfinite outer-train median")
    train_domains = {column: set(train[column].astype(str)) for column in categorical}
    splits: dict[str, dict[str, Any]] = {}
    category_audit: dict[str, dict[str, int]] = {}
    for role in ("outer_train", "validation", "test"):
        part = clean.iloc[split_indices[role]].copy()
        for column, median in medians.items():
            part[column] = part[column].fillna(median)
        if not set(part["target"].unique()) == {0, 1}:
            raise ValueError(f"{spec['name']}: {role} is not binary")
        if int(part["target"].sum()) >= int((part["target"] == 0).sum()):
            raise ValueError(f"{spec['name']}: positive class is not the minority in {role}")
        part = part.loc[:, [*feature_order, "target"]].reset_index(drop=True)
        path = destination / f"{role}.csv.gz"
        with path.open("xb") as handle:
            part.to_csv(handle, index=False, compression="gzip")
        splits[role] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": len(part),
            "minority_rows": int(part["target"].sum()),
        }
        if role != "outer_train":
            category_audit[role] = {
                column: int((~part[column].astype(str).isin(train_domains[column])).sum())
                for column in categorical
            }

    majority = int((train["target"] == 0).sum())
    manifest = {
        "schema_version": "dataset.climb.v1",
        "project_revision": PROJECT_REVISION,
        "status": "FROZEN_FOR_EXECUTION",
        "purpose": "CLIMB-selected mixed-type minority-generator benchmark",
        "name": spec["name"],
        "openml": {
            "id": did,
            "version": int(spec["openml_version"]),
            "name": spec["openml_name"],
            "target": spec["target"],
            "parquet_url": spec["parquet_url"],
            "description_sha256": sha256_file(destination / "openml_description.json"),
            "features_sha256": sha256_file(destination / "openml_features.json"),
            "source_sha256": sha256_file(destination / "openml_source.parquet"),
        },
        "registry_sha256": sha256_file(registry_path),
        "experiment_sha256": sha256_file(experiment_path),
        "preparation_code_sha256": sha256_file(Path(__file__)),
        "split_seed": split_seed,
        "split_strategy": "stratified_group_20fold_70_15_15" if groups is not None else "stratified_random_70_15_15",
        "group_column": spec.get("group_column"),
        "target": "target",
        "positive_label": 1,
        "raw_positive_label": raw_positive,
        "raw_target_counts": raw_counts,
        "feature_columns": feature_order,
        "numerical_columns": numerical,
        "categorical_columns": categorical,
        "transformations": {
            "drop_columns": spec.get("drop_columns", []),
            "categorical_overrides": spec.get("categorical_overrides", []),
            "datetime_transforms": spec.get("datetime_transforms", []),
            "datetime_differences": spec.get("datetime_differences", []),
        },
        "M_d": majority,
        "minority_bank_rule": "match outer-train majority count",
        "max_rows_engineering_only": max_rows,
        "imputation": {
            "fit_on": "outer_train_only",
            "numerical_medians": medians,
            "categorical_missing_token": MISSING_TOKEN,
        },
        "unseen_category_rows": category_audit,
        "splits": splits,
    }
    write_json_new(destination / "dataset.json", manifest)
    return destination / "dataset.json"


def prepare_experiment(
    experiment_path: Path,
    output_root: Path,
    registry_path: Path | None = None,
) -> list[Path]:
    experiment = _read_json(experiment_path)
    if registry_path is None:
        registry_reference = experiment.get("dataset_registry")
        if not isinstance(registry_reference, str) or not registry_reference:
            raise ValueError("experiment must declare dataset_registry")
        registry_path = (experiment_path.parent / registry_reference).resolve()
    registry = _read_json(registry_path)
    if (
        registry.get("status") != "FROZEN_FOR_EXECUTION"
        or registry.get("project_revision") != PROJECT_REVISION
    ):
        raise ValueError("dataset registry is not frozen")
    if experiment.get("status") != "FROZEN_FOR_EXECUTION":
        raise ValueError("experiment is not frozen")
    if (
        experiment.get("protocol_id") != PROTOCOL_ID
        or experiment.get("project_revision") != PROJECT_REVISION
    ):
        raise ValueError("experiment does not use the full outer-train access protocol")
    specs = {item["name"]: item for item in registry["datasets"]}
    registry_names = [item["name"] for item in registry["datasets"]]
    registry_ids = [int(item["openml_id"]) for item in registry["datasets"]]
    if len(registry_names) != len(set(registry_names)) or len(registry_ids) != len(set(registry_ids)):
        raise ValueError("dataset registry has duplicate names or OpenML IDs")
    requested = list(experiment["datasets"])
    if len(requested) != len(set(requested)):
        raise ValueError("duplicate datasets in experiment")
    unknown = sorted(set(requested).difference(specs))
    if unknown:
        raise ValueError(f"datasets absent from registry: {unknown}")
    split = experiment.get("split", {})
    fractions = (
        float(split.get("train_fraction", -1)),
        float(split.get("validation_fraction", -1)),
        float(split.get("test_fraction", -1)),
    )
    if not np.allclose(fractions, (0.70, 0.15, 0.15), rtol=0, atol=1e-12):
        raise ValueError(f"unsupported split fractions: {fractions}")
    if experiment.get("minority_bank_rule") != "match outer-train majority count":
        raise ValueError("unsupported minority_bank_rule")
    output_root.mkdir(parents=True, exist_ok=False)
    manifests: list[Path] = []
    max_rows = experiment.get("dataset_preparation", {}).get("max_rows")
    if max_rows is not None and (not isinstance(max_rows, int) or max_rows < 100):
        raise ValueError("engineering max_rows must be an integer >= 100 or null")
    try:
        frozen_name = experiment.get("dataset_preparation", {}).get("frozen_source_lock")
        frozen_lock = _read_json(experiment_path.parent / frozen_name) if frozen_name else None
        if frozen_lock is not None and set(frozen_lock.get("datasets", {})) != set(requested):
            raise RuntimeError("frozen raw source scope differs from experiment")
        for name in requested:
            source_bundle = (frozen_source_bundle(name, frozen_lock, experiment_path.parent.parent)
                             if frozen_lock is not None else fetch_source_bundle(specs[name]))
            for seed in experiment["split"]["seeds"]:
                destination = output_root / name / f"split-{int(seed)}"
                manifests.append(
                    prepare_dataset(
                        specs[name], destination, int(seed), registry_path=registry_path,
                        experiment_path=experiment_path, max_rows=max_rows,
                        source_bundle=source_bundle,
                    )
                )
        write_json_new(
            output_root / "PREPARED.json",
            {
                "status": "PREPARED",
                "project_revision": PROJECT_REVISION,
                "dataset_count": len(requested),
                "split_count": len(experiment["split"]["seeds"]),
                "manifest_count": len(manifests),
                "registry_sha256": sha256_file(registry_path),
                "experiment_sha256": sha256_file(experiment_path),
                "preparation_code_sha256": sha256_file(Path(__file__)),
            },
        )
    except Exception as error:
        write_json_new(
            output_root / "PREPARATION_FAILED.json",
            {"status": "FAILED", "error_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        )
        raise
    return manifests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for item in prepare_experiment(args.experiment, args.output_root, args.registry):
        print(item)
