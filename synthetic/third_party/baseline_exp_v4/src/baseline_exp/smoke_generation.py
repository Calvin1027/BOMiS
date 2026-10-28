from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd

from .contracts import (
    AdapterFailure,
    DatasetSchema,
    sample_result_metadata,
    validate_fit_artifacts,
    validate_fit_contract,
)
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
from .adapters._external import tree_sha256


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _split(dataset: dict, role: str) -> pd.DataFrame:
    entry = dataset["splits"][role]
    path = Path(entry["path"])
    if sha256_file(path) != entry["sha256"]:
        raise AdapterFailure("configuration", "DATA_HASH_MISMATCH", role)
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def generate(
    *,
    project_root: Path,
    experiment_path: Path,
    models_path: Path,
    dataset_path: Path,
    model_name: str,
    output_root: Path,
) -> Path:
    # Capture the exact wrapper source before importing/building the adapter.
    # This remains truthful even if an operator accidentally edits a mutable
    # checkout while a long fit is running.
    benchmark_source_hash = tree_sha256(project_root / "src" / "baseline_exp")
    experiment, models, dataset = _json(experiment_path), _json(models_path), _json(dataset_path)
    source_lock_name = experiment.get("source_lock")
    if not isinstance(source_lock_name, str) or not source_lock_name:
        raise AdapterFailure("configuration", "SOURCE_LOCK_REFERENCE_MISSING", "experiment")
    source_path = project_root / "configs" / source_lock_name
    sources = _json(source_path)
    protocol_id = PROTOCOL_ID
    for payload, label in ((experiment, "experiment"), (models, "models"), (dataset, "dataset"), (sources, "sources")):
        if payload.get("status") != "FROZEN_FOR_EXECUTION":
            raise AdapterFailure("configuration", "NOT_FROZEN", label)
    if (
        sources.get("protocol_id") != protocol_id
        or sources.get("project_revision") != PROJECT_REVISION
    ):
        raise AdapterFailure("configuration", "SOURCE_PROTOCOL_MISMATCH", source_lock_name)
    if (
        experiment.get("protocol_id") != protocol_id
        or models.get("protocol_id") != protocol_id
        or experiment.get("project_revision") != PROJECT_REVISION
        or models.get("project_revision") != PROJECT_REVISION
        or dataset.get("project_revision") != PROJECT_REVISION
    ):
        raise AdapterFailure("configuration", "PROTOCOL_MISMATCH", protocol_id)
    if experiment.get("models_config") != models_path.name:
        raise AdapterFailure("configuration", "MODEL_CONFIG_NOT_BOUND", models_path.name)
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
    if model_name not in models["models"]:
        raise AdapterFailure("configuration", "MODEL_NOT_IN_SMOKE", model_name)
    dataset_name, split_seed = str(dataset["name"]), int(dataset["split_seed"])
    if dataset_name not in experiment["datasets"] or split_seed not in experiment["split"]["seeds"]:
        raise AdapterFailure("configuration", "DATASET_NOT_IN_EXPERIMENT", dataset_name)
    if dataset.get("experiment_sha256") != sha256_file(experiment_path):
        raise AdapterFailure("configuration", "DATASET_EXPERIMENT_HASH_MISMATCH", dataset_name)
    registry_reference = experiment.get("dataset_registry")
    if not isinstance(registry_reference, str):
        raise AdapterFailure("configuration", "DATASET_REGISTRY_MISSING", dataset_name)
    registry_path = experiment_path.parent / registry_reference
    if not registry_path.is_file() or dataset.get("registry_sha256") != sha256_file(registry_path):
        raise AdapterFailure("configuration", "DATASET_REGISTRY_HASH_MISMATCH", dataset_name)
    cell = output_root / dataset_name / model_name / f"split-{split_seed}"
    claim_new_directory(cell)
    try:
        outer_train = _split(dataset, "outer_train")
        schema = DatasetSchema.from_outer_train(
            outer_train,
            target=str(dataset["target"]),
            positive_label=dataset["positive_label"],
            numerical_columns=tuple(dataset["numerical_columns"]),
            categorical_columns=tuple(dataset["categorical_columns"]),
        )
        model_seed = semantic_seed(int(experiment["model_seed_root"]), dataset_name, model_name, split_seed, "fit")
        sampling_seed = semantic_seed(int(experiment["sampling_seed_root"]), dataset_name, model_name, split_seed, "sample")
        assert_distinct_seed_roles(model_seed, sampling_seed)
        spec = models["models"][model_name]
        source_identity = verify_external_source(model_name, spec, sources, project_root=project_root)
        adapter = build_adapter(model_name, spec, project_root=project_root)
        fit = adapter.fit(outer_train.copy(deep=True), schema, model_seed=model_seed, work_dir=cell / "model")
        # Preserve the adapter's fit record even if contract validation or sampling fails.
        write_json_new(cell / "fit.json", {"fit": fit})
        validate_fit_contract(
            fit, expected_model=model_name, expected_outer_train_n=len(outer_train)
        )
        validate_fit_artifacts(fit, work_dir=cell / "model")
        sample = adapter.sample_minority(int(dataset["M_d"]), sampling_seed=sampling_seed)
        # Parquet preserves the exact validated dtypes/float bits used by the dataframe hash.
        # CSV round-tripping can change floating-point representations and create false hash failures.
        output = cell / "synthetic_minority.parquet"
        if output.exists():
            raise AdapterFailure("validation", "ARTIFACT_EXISTS", str(output))
        sample.frame.to_parquet(output, index=False)
        csv_output = cell / "synthetic_minority.csv"
        if csv_output.exists():
            raise AdapterFailure("validation", "ARTIFACT_EXISTS", str(csv_output))
        with csv_output.open("x", encoding="utf-8", newline="") as handle:
            sample.frame.to_csv(handle, index=False)
        write_json_new(cell / "sample.json", {"sample": sample_result_metadata(sample)})
        fit_path = cell / "fit.json"
        sample_path = cell / "sample.json"
        write_json_new(
            cell / "GENERATED.json",
            {
                "status": "GENERATED_UNVERIFIED",
                "project_revision": PROJECT_REVISION,
                "protocol_id": protocol_id,
                "purpose": str(experiment.get("purpose", "generation run")),
                "model": model_name,
                "dataset": dataset_name,
                "split_seed": split_seed,
                "M_d": int(dataset["M_d"]),
                "outer_train_sha256": dataframe_sha256(outer_train),
                "synthetic_sha256": dataframe_sha256(sample.frame),
                "synthetic_file_sha256": sha256_file(output),
                "synthetic_csv_file_sha256": sha256_file(csv_output),
                "fit_file_sha256": sha256_file(fit_path),
                "sample_file_sha256": sha256_file(sample_path),
                "source_identity": source_identity,
                "benchmark_source_tree_sha256": benchmark_source_hash,
                "config_sha256": {
                    "experiment": sha256_file(experiment_path),
                    "models": sha256_file(models_path),
                    "dataset": sha256_file(dataset_path),
                    "sources": sha256_file(source_path),
                },
                "environment": environment_manifest(),
                "fallback_used": False,
                "backfill_used": False,
            },
        )
    except Exception as error:
        stage = error.stage if isinstance(error, AdapterFailure) else "fit"
        code = error.code if isinstance(error, AdapterFailure) else type(error).__name__.upper()
        write_failure_new(cell, model=model_name, dataset=dataset_name, split_seed=split_seed, stage=stage, code=code, message=str(error), context=(error.context if isinstance(error, AdapterFailure) else None))
        write_json_new(cell / "TRACEBACK.json", {"traceback": traceback.format_exc()})
        raise
    return cell


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(generate(project_root=root, experiment_path=args.experiment, models_path=args.models, dataset_path=args.dataset, model_name=args.model, output_root=args.output_root))
