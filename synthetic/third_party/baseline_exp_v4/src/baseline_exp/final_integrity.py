"""Fail-closed integrity checks shared by all final-run consumers."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .provenance import sha256_file
from .revision import PROJECT_REVISION, PROTOCOL_ID


EVALUATION_ARTIFACTS = (
    "metrics.json", "metrics.csv", "marginal_components.csv", "pairwise_components.csv",
)
PRIMARY_METRICS = (
    "fidelity.mean_numerical_ks", "fidelity.mean_categorical_tvd",
    "fidelity.mean_category_coverage", "fidelity.categorical_tuple_validity",
    "fidelity.alpha_precision", "fidelity.beta_recall", "downstream.validation_mcc",
    "downstream.test_mcc", "downstream.test_pr_auc",
)
PAIR_TYPES = ("numerical_numerical", "numerical_categorical", "categorical_categorical")
SUMMARY_METRICS = PRIMARY_METRICS + tuple(
    "fidelity.mean_pair_similarity_" + kind for kind in PAIR_TYPES
)


def validate_metrics(metrics: dict) -> None:
    for key in PRIMARY_METRICS:
        section, name = key.split(".")
        value = metrics.get(section, {}).get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError(f"missing/nonfinite primary metric: {key}")
    for kind in PAIR_TYPES:
        fidelity = metrics.get("fidelity", {})
        count = fidelity.get("defined_pairs_" + kind)
        value = fidelity.get("mean_pair_similarity_" + kind)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise RuntimeError(f"invalid pair denominator: {kind}")
        if count == 0:
            if value is not None:
                raise RuntimeError(f"undefined pair metric must be null: {kind}")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError(f"defined pair metric is missing/nonfinite: {kind}")


def validate_rejection(sample: dict, expected_n: int) -> None:
    if sample.get("rejection_used") is not True:
        return
    keys = ("raw_draw_count", "invalid_target_rows", "rejected_y0_rows", "decoded_y1_rows",
            "valid_y1_rows", "invalid_y1_rows", "selected_output_rows")
    values = [sample.get(key) for key in keys]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise RuntimeError("invalid rejection counters")
    raw, invalid_target, y0, decoded, valid, invalid, selected = values
    if raw != invalid_target + y0 + decoded or decoded != valid + invalid or not selected == expected_n <= valid:
        raise RuntimeError("rejection conservation mismatch")
    if raw <= 0 or sample.get("accepted_minority_count") != valid or raw > sample.get("raw_draw_cap", -1):
        raise RuntimeError("rejection cap/accepted count mismatch")
    for key, expected in (("y1_rate", decoded / raw),
                          ("valid_given_y1_rate", valid / decoded if decoded else 0.0),
                          ("overall_acceptance_rate", valid / raw)):
        actual = sample.get(key)
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(f"rejection rate mismatch: {key}")


def validate_cell_integrity(generation: Path, evaluation: Path, *, dataset: str, model: str,
                            seed: int, experiment_hash: str, models_hash: str,
                            source_hash: str) -> tuple[dict, dict, dict]:
    generated_path = generation / "GENERATED.json"
    generated = json.loads(generated_path.read_text())
    success = json.loads((evaluation / "SUCCESS.json").read_text())
    for marker, status in ((generated, "GENERATED_UNVERIFIED"), (success, "SUCCESS")):
        if any(marker.get(key) != expected for key, expected in {
            "status": status, "dataset": dataset, "model": model, "split_seed": seed,
            "project_revision": PROJECT_REVISION, "protocol_id": PROTOCOL_ID,
            "benchmark_source_tree_sha256": source_hash,
        }.items()):
            raise RuntimeError(f"foreign cell/source provenance: {dataset}/{model}/{seed}")
    if (generated.get("config_sha256", {}).get("experiment") != experiment_hash
        or generated.get("config_sha256", {}).get("models") != models_hash
        or success.get("experiment_sha256") != experiment_hash
        or success.get("models_config_sha256") != models_hash
        or success.get("generation_marker_sha256") != sha256_file(generated_path)
        or generated.get("fallback_used") is not False or generated.get("backfill_used") is not False):
        raise RuntimeError("cell configuration/lineage mismatch")
    for name, field in (("synthetic_minority.csv", "synthetic_csv_file_sha256"),
                        ("synthetic_minority.parquet", "synthetic_file_sha256"),
                        ("fit.json", "fit_file_sha256"), ("sample.json", "sample_file_sha256")):
        if sha256_file(generation / name) != generated.get(field):
            raise RuntimeError(f"generation artifact hash mismatch: {name}")
    hashes = success.get("evaluation_artifacts_sha256", {})
    if set(hashes) != set(EVALUATION_ARTIFACTS):
        raise RuntimeError("evaluation artifact manifest is incomplete")
    for name in EVALUATION_ARTIFACTS:
        if sha256_file(evaluation / name) != hashes[name]:
            raise RuntimeError(f"evaluation artifact hash mismatch: {name}")
    expected_n = generated.get("M_d")
    if isinstance(expected_n, bool) or not isinstance(expected_n, int) or expected_n <= 0 or success.get("M_d") != expected_n:
        raise RuntimeError("cell output count mismatch")
    validate_rejection(json.loads((generation / "sample.json").read_text())["sample"], expected_n)
    metrics = json.loads((evaluation / "metrics.json").read_text())
    validate_metrics(metrics)
    return generated, success, metrics
