from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd

from .contracts import AdapterFailure, DatasetSchema
from .downstream import evaluate_downstream
from .metrics import evaluate_fidelity
from .provenance import claim_new_directory, dataframe_sha256, environment_manifest, sha256_file, write_failure_new, write_json_new
from .seeds import semantic_seed
from .adapters._external import tree_sha256
from .final_integrity import EVALUATION_ARTIFACTS, validate_metrics
from .revision import PROJECT_REVISION, PROTOCOL_ID


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_numeric(prefix: str, value: object, output: dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten_numeric(f"{prefix}.{key}" if prefix else str(key), item, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = value


def load_split(dataset: dict, role: str) -> pd.DataFrame:
    entry = dataset["splits"][role]
    path = Path(entry["path"])
    if sha256_file(path) != entry["sha256"]:
        raise AdapterFailure("metric", "DATA_HASH_MISMATCH", role)
    return (
        pd.read_parquet(path)
        if path.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(path)
    )


def evaluate(*, experiment_path: Path, dataset_path: Path, model: str, generation_root: Path, output_root: Path) -> Path:
    evaluator_source_hash = tree_sha256(Path(__file__).resolve().parent)
    experiment, dataset = load_json(experiment_path), load_json(dataset_path)
    if experiment.get("status") != "FROZEN_FOR_EXECUTION" or dataset.get("status") != "FROZEN_FOR_EXECUTION":
        raise AdapterFailure("configuration", "NOT_FROZEN", "experiment or dataset")
    protocol_id = PROTOCOL_ID
    if (
        experiment.get("protocol_id") != protocol_id
        or experiment.get("project_revision") != PROJECT_REVISION
        or dataset.get("project_revision") != PROJECT_REVISION
    ):
        raise AdapterFailure("configuration", "PROTOCOL_MISMATCH", protocol_id)
    models_reference = experiment.get("models_config")
    if not isinstance(models_reference, str):
        raise AdapterFailure("configuration", "MODEL_CONFIG_REFERENCE_MISSING", "experiment")
    models_path = experiment_path.parent / models_reference
    if not models_path.is_file():
        raise AdapterFailure("configuration", "MODEL_CONFIG_MISSING", str(models_path))
    models_payload = load_json(models_path)
    if (
        models_payload.get("status") != "FROZEN_FOR_EXECUTION"
        or models_payload.get("protocol_id") != protocol_id
        or models_payload.get("project_revision") != PROJECT_REVISION
    ):
        raise AdapterFailure("configuration", "MODEL_PROTOCOL_MISMATCH", models_reference)
    assigned_models = (
        list(models_payload.get("resource_groups", {}).get("gpu", []))
        + list(models_payload.get("resource_groups", {}).get("cpu_light", []))
        + list(models_payload.get("resource_groups", {}).get("cpu_forest", []))
    )
    expected_models = list(experiment.get("models", []))
    if (
        len(assigned_models) != len(set(assigned_models))
        or set(assigned_models) != set(expected_models)
        or set(models_payload.get("models", {})) != set(expected_models)
        or model not in models_payload.get("models", {})
    ):
        raise AdapterFailure("configuration", "MODEL_SET_MISMATCH", model)
    expected_models_hash = sha256_file(models_path)
    name, seed = str(dataset["name"]), int(dataset["split_seed"])
    if name not in experiment["datasets"] or seed not in experiment["split"]["seeds"]:
        raise AdapterFailure("configuration", "DATASET_NOT_IN_EXPERIMENT", name)
    if dataset.get("experiment_sha256") != sha256_file(experiment_path):
        raise AdapterFailure("configuration", "DATASET_EXPERIMENT_HASH_MISMATCH", name)
    registry_reference = experiment.get("dataset_registry")
    if not isinstance(registry_reference, str):
        raise AdapterFailure("configuration", "DATASET_REGISTRY_MISSING", name)
    registry_path = experiment_path.parent / registry_reference
    if not registry_path.is_file() or dataset.get("registry_sha256") != sha256_file(registry_path):
        raise AdapterFailure("configuration", "DATASET_REGISTRY_HASH_MISMATCH", name)
    generated = generation_root / name / model / f"split-{seed}"
    cell = output_root / name / model / f"split-{seed}"
    claim_new_directory(cell)
    try:
        upstream_failure = generated / "FAILURE.json"
        if upstream_failure.is_file():
            failure = load_json(upstream_failure)
            raise AdapterFailure(
                "metric",
                "UPSTREAM_GENERATION_FAILED",
                f"{failure.get('stage', 'unknown')}:{failure.get('code', 'unknown')}",
            )
        if not generated.is_dir():
            raise AdapterFailure(
                "metric", "UPSTREAM_GENERATION_MISSING", str(generated)
            )
        marker = load_json(generated / "GENERATED.json")
        synthetic_path = generated / "synthetic_minority.parquet"
        fit_path, sample_path = generated / "fit.json", generated / "sample.json"
        if (
            marker.get("status") != "GENERATED_UNVERIFIED"
            or not synthetic_path.is_file()
            or not fit_path.is_file()
            or not sample_path.is_file()
        ):
            raise AdapterFailure("metric", "GENERATION_NOT_SUCCESSFUL", str(generated))
        if (
            marker.get("protocol_id") != protocol_id
            or marker.get("project_revision") != PROJECT_REVISION
        ):
            raise AdapterFailure("metric", "GENERATION_PROTOCOL_MISMATCH", model)
        if marker.get("config_sha256", {}).get("experiment") != sha256_file(experiment_path):
            raise AdapterFailure("metric", "GENERATION_EXPERIMENT_HASH_MISMATCH", model)
        if marker.get("config_sha256", {}).get("dataset") != sha256_file(dataset_path):
            raise AdapterFailure("metric", "GENERATION_DATASET_HASH_MISMATCH", model)
        if marker.get("config_sha256", {}).get("models") != expected_models_hash:
            raise AdapterFailure("metric", "GENERATION_MODEL_CONFIG_HASH_MISMATCH", model)
        if marker.get("benchmark_source_tree_sha256") != evaluator_source_hash:
            raise AdapterFailure("metric", "GENERATION_EVALUATOR_SOURCE_MISMATCH", model)
        if sha256_file(fit_path) != marker.get("fit_file_sha256"):
            raise AdapterFailure("metric", "FIT_ARTIFACT_HASH_MISMATCH", model)
        if sha256_file(sample_path) != marker.get("sample_file_sha256"):
            raise AdapterFailure("metric", "SAMPLE_ARTIFACT_HASH_MISMATCH", model)
        outer_train, validation, test = (load_split(dataset, role) for role in ("outer_train", "validation", "test"))
        schema = DatasetSchema.from_outer_train(
            outer_train, target=str(dataset["target"]), positive_label=dataset["positive_label"],
            numerical_columns=tuple(dataset["numerical_columns"]), categorical_columns=tuple(dataset["categorical_columns"]),
        )
        if sha256_file(synthetic_path) != marker.get("synthetic_file_sha256"):
            raise AdapterFailure("metric", "SYNTHETIC_FILE_HASH_MISMATCH", model)
        synthetic = pd.read_parquet(synthetic_path)
        synthetic = schema.restore_and_validate(synthetic, expected_n=int(dataset["M_d"]))
        csv_hash = marker.get("synthetic_csv_file_sha256")
        if csv_hash is not None:
            csv_path = generated / "synthetic_minority.csv"
            if not csv_path.is_file() or sha256_file(csv_path) != csv_hash:
                raise AdapterFailure("metric", "SYNTHETIC_CSV_HASH_MISMATCH", model)
            csv_frame = pd.read_csv(csv_path)
            if tuple(csv_frame.columns) != schema.feature_columns or len(csv_frame) != int(
                dataset["M_d"]
            ):
                raise AdapterFailure("metric", "SYNTHETIC_CSV_SCHEMA_MISMATCH", model)
        fit_payload = load_json(fit_path).get("fit", {})
        expected_model_seed = semantic_seed(
            int(experiment["model_seed_root"]), name, model, seed, "fit"
        )
        expected_sampling_seed = semantic_seed(
            int(experiment["sampling_seed_root"]), name, model, seed, "sample"
        )
        if (
            fit_payload.get("model") != model
            or fit_payload.get("training_access") != "full_outer_train"
            or int(fit_payload.get("n_outer_train", -1)) != len(outer_train)
            or int(fit_payload.get("model_seed", -1)) != expected_model_seed
        ):
            raise AdapterFailure("metric", "FIT_ACCESS_CONTRACT_MISMATCH", model)
        sample_payload = load_json(sample_path).get("sample", {})
        if int(sample_payload.get("sampling_seed", -1)) != expected_sampling_seed:
            raise AdapterFailure("metric", "SAMPLE_PROVENANCE_INVALID", model)
        if sample_payload.get("rejection_used") is True:
            raw = int(sample_payload.get("raw_draw_count", -1))
            invalid_target = int(sample_payload.get("invalid_target_rows", -1))
            rejected_y0 = int(sample_payload.get("rejected_y0_rows", -1))
            decoded_y1 = int(sample_payload.get("decoded_y1_rows", -1))
            valid_y1 = int(sample_payload.get("valid_y1_rows", -1))
            invalid_y1 = int(sample_payload.get("invalid_y1_rows", -1))
            selected = int(sample_payload.get("selected_output_rows", -1))
            if (
                min(raw, invalid_target, rejected_y0, decoded_y1, valid_y1, invalid_y1, selected) < 0
                or raw != invalid_target + rejected_y0 + decoded_y1
                or decoded_y1 != valid_y1 + invalid_y1
                or selected != int(dataset["M_d"])
                or selected > valid_y1
            ):
                raise AdapterFailure("metric", "REJECTION_CONSERVATION_MISMATCH", model)
        if dataframe_sha256(synthetic) != marker["synthetic_sha256"]:
            raise AdapterFailure("metric", "SYNTHETIC_HASH_MISMATCH", model)
        reference = test.loc[schema.binary_target(test).astype(bool), list(schema.feature_columns)].copy()
        metric_seed = semantic_seed(int(experiment["metric_seed_root"]), name, model, seed, "fidelity")
        fidelity, components = evaluate_fidelity(reference, synthetic, schema, metric_seed=metric_seed)
        for component, frame in components.items():
            with (cell / f"{component}.csv").open("x", encoding="utf-8") as handle:
                frame.to_csv(handle, index=False)
        classifier_seed = semantic_seed(int(experiment["classifier_seed_root"]), name, model, seed, "classifier")
        downstream = evaluate_downstream(outer_train, validation, test, synthetic, schema, classifier_seed=classifier_seed, classifier_spec=experiment["classifier"])
        metrics_payload = {"fidelity": fidelity, "downstream": downstream}
        validate_metrics(metrics_payload)
        write_json_new(cell / "metrics.json", metrics_payload)
        metric_row: dict[str, object] = {
            "dataset": name,
            "model": model,
            "split_seed": seed,
        }
        _flatten_numeric("", metrics_payload, metric_row)
        with (cell / "metrics.csv").open("x", encoding="utf-8", newline="") as handle:
            pd.DataFrame([metric_row]).to_csv(handle, index=False)
        write_json_new(cell / "SUCCESS.json", {
            "status": "SUCCESS", "purpose": experiment["purpose"], "dataset": name, "model": model,
            "project_revision": PROJECT_REVISION,
            "protocol_id": protocol_id, "models_config_sha256": expected_models_hash,
            "split_seed": seed, "M_d": int(dataset["M_d"]), "generation_marker_sha256": sha256_file(generated / "GENERATED.json"),
            "synthetic_sha256": marker["synthetic_sha256"], "experiment_sha256": sha256_file(experiment_path),
            "dataset_sha256": sha256_file(dataset_path), "environment": environment_manifest(),
            "benchmark_source_tree_sha256": evaluator_source_hash,
            "evaluation_artifacts_sha256": {
                name: sha256_file(cell / name) for name in EVALUATION_ARTIFACTS
            },
        })
    except Exception as error:
        stage = error.stage if isinstance(error, AdapterFailure) else "metric"
        code = error.code if isinstance(error, AdapterFailure) else type(error).__name__.upper()
        write_failure_new(cell, model=model, dataset=name, split_seed=seed, stage=stage, code=code, message=str(error), context=(error.context if isinstance(error, AdapterFailure) else None))
        write_json_new(cell / "TRACEBACK.json", {"traceback": traceback.format_exc()})
        raise
    return cell


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate(experiment_path=args.experiment, dataset_path=args.dataset, model=args.model, generation_root=args.generation_root, output_root=args.output_root))
