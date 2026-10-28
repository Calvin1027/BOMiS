from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import (
    AdapterFailure,
    DatasetSchema,
    sample_result_metadata,
    validate_fit_artifacts,
    validate_fit_contract,
)
from .downstream import evaluate_downstream
from .metrics import evaluate_fidelity
from .provenance import (
    claim_new_directory,
    dataframe_sha256,
    environment_manifest,
    sha256_file,
    write_failure_new,
    write_json_new,
)
from .registry import build_adapter
from .revision import PROJECT_REVISION, PROTOCOL_ID
from .seeds import assert_distinct_seed_roles, semantic_seed
from .source_validation import verify_external_source


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_split(spec: dict[str, Any], role: str) -> pd.DataFrame:
    entry = spec["splits"][role]
    path = Path(entry["path"]).resolve()
    if sha256_file(path) != entry["sha256"]:
        raise AdapterFailure("configuration", "DATA_HASH_MISMATCH", f"{role}: {path}")
    frame = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    if frame.empty:
        raise AdapterFailure("configuration", "EMPTY_SPLIT", role)
    return frame


def run_cell(
    *,
    project_root: Path,
    experiment_config: Path,
    model_config: Path,
    dataset_manifest: Path,
    model_name: str,
    split_seed: int,
    output_root: Path,
) -> Path:
    experiment, models, dataset = map(_load_json, (experiment_config, model_config, dataset_manifest))
    source_lock_name = experiment.get("source_lock")
    if not isinstance(source_lock_name, str) or not source_lock_name:
        raise AdapterFailure("configuration", "SOURCE_LOCK_REFERENCE_MISSING", "experiment")
    source_lock_path = project_root / "configs" / source_lock_name
    source_lock = _load_json(source_lock_path)
    if (
        source_lock.get("status") != "FROZEN_FOR_EXECUTION"
        or source_lock.get("protocol_id") != PROTOCOL_ID
        or source_lock.get("project_revision") != PROJECT_REVISION
    ):
        raise AdapterFailure("configuration", "SOURCES_NOT_FROZEN", str(source_lock_path))
    if experiment.get("status") != "FROZEN_FOR_EXECUTION":
        raise AdapterFailure("configuration", "EXPERIMENT_NOT_FROZEN", str(experiment_config))
    if models.get("status") != "FROZEN_FOR_EXECUTION":
        raise AdapterFailure("configuration", "MODELS_NOT_FROZEN", str(model_config))
    if dataset.get("status") != "FROZEN_FOR_EXECUTION":
        raise AdapterFailure("configuration", "DATASET_NOT_FROZEN", str(dataset_manifest))
    protocol_id = PROTOCOL_ID
    if (
        experiment.get("protocol_id") != protocol_id
        or models.get("protocol_id") != protocol_id
        or experiment.get("project_revision") != PROJECT_REVISION
        or models.get("project_revision") != PROJECT_REVISION
        or dataset.get("project_revision") != PROJECT_REVISION
    ):
        raise AdapterFailure("configuration", "PROTOCOL_MISMATCH", protocol_id)
    if experiment.get("models_config") != model_config.name:
        raise AdapterFailure("configuration", "MODEL_CONFIG_NOT_BOUND", model_config.name)
    assigned_models = (
        list(models.get("resource_groups", {}).get("gpu", []))
        + list(models.get("resource_groups", {}).get("cpu_light", []))
        + list(models.get("resource_groups", {}).get("cpu_forest", []))
    )
    expected_models = list(experiment.get("models", []))
    if (
        len(assigned_models) != len(set(assigned_models))
        or set(assigned_models) != set(expected_models)
        or set(models.get("models", {})) != set(expected_models)
    ):
        raise AdapterFailure("configuration", "MODEL_SET_MISMATCH", "resource/spec/experiment")
    dataset_name = str(dataset["name"])
    if model_name not in models["models"]:
        raise AdapterFailure("configuration", "MODEL_NOT_IN_EXPERIMENT", model_name)
    cell = output_root / dataset_name / model_name / f"split-{int(split_seed)}"
    claim_new_directory(cell)
    try:
        if dataset_name not in experiment["datasets"]:
            raise AdapterFailure("configuration", "DATASET_NOT_IN_EXPERIMENT", dataset_name)
        if dataset.get("experiment_sha256") != sha256_file(experiment_config):
            raise AdapterFailure(
                "configuration", "DATASET_EXPERIMENT_HASH_MISMATCH", dataset_name
            )
        registry_reference = experiment.get("dataset_registry")
        if not isinstance(registry_reference, str):
            raise AdapterFailure("configuration", "DATASET_REGISTRY_MISSING", dataset_name)
        registry_path = experiment_config.parent / registry_reference
        if (
            not registry_path.is_file()
            or dataset.get("registry_sha256") != sha256_file(registry_path)
        ):
            raise AdapterFailure(
                "configuration", "DATASET_REGISTRY_HASH_MISMATCH", dataset_name
            )
        outer_train = _load_split(dataset, "outer_train")
        validation = _load_split(dataset, "validation")
        test = _load_split(dataset, "test")
        if int(dataset["split_seed"]) != int(split_seed) or int(split_seed) not in experiment["split"]["seeds"]:
            raise AdapterFailure("configuration", "SPLIT_SEED_MISMATCH", str(split_seed))
        schema = DatasetSchema.from_outer_train(
            outer_train,
            target=str(dataset["target"]),
            positive_label=dataset["positive_label"],
            numerical_columns=tuple(dataset["numerical_columns"]),
            categorical_columns=tuple(dataset["categorical_columns"]),
        )
        expected_columns = [*schema.feature_columns, schema.target]
        for role, frame in (("validation", validation), ("test", test)):
            if list(frame.columns) != expected_columns:
                raise AdapterFailure("configuration", "SPLIT_SCHEMA_MISMATCH", role)
            schema.binary_target(frame)
        m_d = int(dataset["M_d"])
        if m_d < 1:
            raise AdapterFailure("configuration", "INVALID_M_D", str(m_d))
        model_seed = semantic_seed(int(experiment["model_seed_root"]), dataset_name, model_name, split_seed, "fit")
        sampling_seed = semantic_seed(int(experiment["sampling_seed_root"]), dataset_name, model_name, split_seed, "sample")
        classifier_seed = semantic_seed(int(experiment["classifier_seed_root"]), dataset_name, model_name, split_seed, "classifier")
        assert_distinct_seed_roles(model_seed, sampling_seed, classifier_seed)
        source_identity = verify_external_source(
            model_name, models["models"][model_name], source_lock, project_root=project_root
        )
        adapter = build_adapter(model_name, models["models"][model_name], project_root=project_root)
        fit = adapter.fit(outer_train.copy(deep=True), schema, model_seed=model_seed, work_dir=cell / "model")
        # Preserve the adapter's fit record even if contract validation or sampling fails.
        write_json_new(cell / "fit.json", {"fit": fit})
        validate_fit_contract(
            fit, expected_model=model_name, expected_outer_train_n=len(outer_train)
        )
        validate_fit_artifacts(fit, work_dir=cell / "model")
        sample = adapter.sample_minority(m_d, sampling_seed=sampling_seed)
        synthetic_path = cell / "synthetic_minority.parquet"
        if synthetic_path.exists():
            raise AdapterFailure("validation", "ARTIFACT_EXISTS", str(synthetic_path))
        sample.frame.to_parquet(synthetic_path, index=False)
        reference = test.loc[schema.binary_target(test).astype(bool), list(schema.feature_columns)].copy()
        if reference.empty:
            raise AdapterFailure("metric", "NO_TEST_MINORITY", dataset_name)
        metric_seed = semantic_seed(int(experiment["metric_seed_root"]), dataset_name, model_name, split_seed, "fidelity")
        fidelity, components = evaluate_fidelity(reference, sample.frame, schema, metric_seed=metric_seed)
        for name, frame in components.items():
            path = cell / f"{name}.csv"
            with path.open("x", encoding="utf-8") as handle:
                frame.to_csv(handle, index=False)
        downstream = evaluate_downstream(
            outer_train, validation, test, sample.frame, schema,
            classifier_seed=classifier_seed, classifier_spec=experiment["classifier"],
        )
        write_json_new(cell / "sample.json", {"sample": sample_result_metadata(sample)})
        write_json_new(cell / "metrics.json", {"fidelity": fidelity, "downstream": downstream})
        write_json_new(
            cell / "SUCCESS.json",
            {
                "status": "SUCCESS", "protocol_id": protocol_id, "dataset": dataset_name, "model": model_name,
                "project_revision": PROJECT_REVISION,
                "split_seed": int(split_seed), "M_d": m_d, "fallback_used": False,
                "outer_train_sha256": dataframe_sha256(outer_train),
                "synthetic_sha256": dataframe_sha256(sample.frame),
                "fit_file_sha256": sha256_file(cell / "fit.json"),
                "sample_file_sha256": sha256_file(cell / "sample.json"),
                "synthetic_file_sha256": sha256_file(synthetic_path),
                "config_sha256": {"experiment": sha256_file(experiment_config), "models": sha256_file(model_config), "dataset": sha256_file(dataset_manifest), "sources": sha256_file(source_lock_path)},
                "source_identity": source_identity,
                "environment": environment_manifest(),
            },
        )
    except Exception as error:
        if isinstance(error, AdapterFailure):
            stage, code = error.stage, error.code
        else:
            stage, code = "fit", type(error).__name__.upper()
        write_failure_new(cell, model=model_name, dataset=dataset_name, split_seed=split_seed, stage=stage, code=code, message=str(error), context=(error.context if isinstance(error, AdapterFailure) else None))
        write_json_new(cell / "TRACEBACK.json", {"traceback": traceback.format_exc()})
        raise
    return cell
