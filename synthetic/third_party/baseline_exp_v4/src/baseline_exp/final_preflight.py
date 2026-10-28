from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .adapters._external import tree_sha256
from .final_manifests import build
from .provenance import sha256_file, write_json_new
from .revision import PROJECT_REVISION, PROTOCOL_ID
from .source_validation import verify_external_source
from .climb_data import frozen_source_bundle


EXPECTED_DATASETS = 13
EXPECTED_MODELS = 13
EXPECTED_SEEDS = (20261001, 20261002, 20261003, 20261004, 20261005)
EXPECTED_CELLS = EXPECTED_DATASETS * EXPECTED_MODELS * len(EXPECTED_SEEDS)
TRACKED_ROOTS = ("src", "configs", "slurm", "scripts", "tests")
TRACKED_TOP_LEVEL = ("README.md", "PROTOCOL.md", "pyproject.toml")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tracked_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for name in TRACKED_ROOTS:
        root = project_root / name
        if not root.is_dir():
            raise RuntimeError(f"missing tracked directory: {root}")
        if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")
                                   if not any(part in EXCLUDED_PARTS for part in path.parts)):
            raise RuntimeError(f"symlink in tracked runtime: {root}")
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and not any(part in EXCLUDED_PARTS for part in path.relative_to(project_root).parts)
        )
    for name in TRACKED_TOP_LEVEL:
        path = project_root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing tracked file: {path}")
        files.append(path)
    return sorted(set(files), key=lambda path: str(path.relative_to(project_root)))


def _manifest_digest(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, value in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def _validate_split1_reference(project_root: Path, reference: dict[str, Any]) -> dict[str, str]:
    if reference.get("status") != "VERIFIED_ENGINEERING_GATE":
        raise RuntimeError("split-1 gate reference is not verified")
    matrix = reference.get("matrix", {})
    validation = reference.get("validation", {})
    if (
        matrix.get("expected_cells") != 169
        or validation.get("pass") != 169
        or validation.get("fail") != 0
        or validation.get("missing") != 0
        or validation.get("duplicate_cells") != 0
        or validation.get("primary_nonfinite_values") != 0
    ):
        raise RuntimeError("split-1 gate reference is incomplete")
    evidence = reference.get("evidence", {})
    verified: dict[str, str] = {}
    for label in ("pass", "audit"):
        raw = evidence.get(f"{label}_artifact")
        expected = evidence.get(f"{label}_artifact_sha256")
        if not isinstance(raw, str) or not isinstance(expected, str):
            raise RuntimeError(f"split-1 {label} evidence is missing")
        path = (project_root / "configs" / raw).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"split-1 {label} evidence hash mismatch")
        verified[label] = expected
    pass_payload = _read((project_root / "configs" / evidence["pass_artifact"]).resolve())
    audit_payload = _read((project_root / "configs" / evidence["audit_artifact"]).resolve())
    if (
        pass_payload.get("status") != "PASS"
        or pass_payload.get("validated_generation_cells") != 169
        or pass_payload.get("validated_metric_cells") != 169
        or pass_payload.get("synthetic_csv_files") != 169
        or audit_payload.get("counts") != {"PASS": 169, "FAIL": 0, "MISSING": 0}
    ):
        raise RuntimeError("split-1 evidence payload does not prove 169/169")
    return verified


def preflight(
    *,
    project_root: Path,
    experiment_path: Path,
    models_path: Path,
    execution_path: Path,
    output_dir: Path,
    unit_evidence: Path,
    data_evidence: Path,
) -> Path:
    project_root = project_root.resolve()
    experiment_path = experiment_path.resolve()
    models_path = models_path.resolve()
    execution_path = execution_path.resolve()
    unit_evidence = unit_evidence.resolve()
    data_evidence = data_evidence.resolve()
    try:
        unit_evidence.relative_to(output_dir.resolve())
        data_evidence.relative_to(project_root / "reviews")
    except ValueError as error:
        raise RuntimeError("review evidence is outside its approved runtime namespace") from error
    experiment, models, execution = map(_read, (experiment_path, models_path, execution_path))
    suites = list(ET.parse(unit_evidence).getroot().iter("testsuite"))
    if not suites or sum(int(s.get("tests", "0")) for s in suites) < 65 or any(
        int(s.get(key, "0")) for s in suites for key in ("failures", "errors", "skipped")
    ):
        raise RuntimeError("unit evidence is incomplete or has failed/skipped tests")
    data_pass = _read(data_evidence)
    if (data_pass.get("status") != "PASS" or data_pass.get("validated_manifests") != 65
        or data_pass.get("experiment_sha256") != sha256_file(experiment_path)
        or data_pass.get("models_config_sha256") != sha256_file(models_path)):
        raise RuntimeError("65-split data evidence is missing or stale")
    if project_root.name != PROJECT_REVISION:
        raise RuntimeError(f"runtime project name is not {PROJECT_REVISION}")
    expected_project = execution.get("runtime_contract", {}).get("compute_project")
    if str(project_root) != expected_project:
        raise RuntimeError(f"runtime path mismatch: {project_root} != {expected_project}")
    if any(payload.get("project_revision") != PROJECT_REVISION for payload in (experiment, models, execution)):
        raise RuntimeError("configuration project revision mismatch")
    if any(payload.get("protocol_id") != PROTOCOL_ID for payload in (experiment, models, execution)):
        raise RuntimeError("configuration protocol mismatch")
    if experiment.get("status") != "FROZEN_FOR_EXECUTION" or models.get("status") != "FROZEN_FOR_EXECUTION":
        raise RuntimeError("experiment or model config is not frozen")
    if execution.get("status") != "READY_NOT_SUBMITTED":
        raise RuntimeError("execution config is not ready")
    if execution.get("experiment") != experiment_path.name or execution.get("models") != models_path.name:
        raise RuntimeError("execution config does not bind supplied experiment/model configs")
    if experiment.get("models_config") != models_path.name:
        raise RuntimeError("experiment does not bind supplied model config")
    if tuple(map(int, experiment.get("split", {}).get("seeds", []))) != EXPECTED_SEEDS:
        raise RuntimeError("five prespecified split seeds changed")
    if len(experiment.get("datasets", [])) != EXPECTED_DATASETS or len(set(experiment["datasets"])) != EXPECTED_DATASETS:
        raise RuntimeError("dataset matrix is not exactly 13 unique datasets")
    if len(experiment.get("models", [])) != EXPECTED_MODELS or len(set(experiment["models"])) != EXPECTED_MODELS:
        raise RuntimeError("model matrix is not exactly 13 unique models")
    groups = models.get("resource_groups", {})
    assigned = list(groups.get("gpu", [])) + list(groups.get("cpu_light", [])) + list(groups.get("cpu_forest", []))
    if (
        len(assigned) != len(set(assigned))
        or set(assigned) != set(experiment["models"])
        or set(models.get("models", {})) != set(experiment["models"])
        or groups.get("cpu_forest") != []
        or "forest_flow" in assigned
    ):
        raise RuntimeError("resource groups do not exactly partition the 13 active models")
    if models["models"]["tabddpm"].get("train_steps") != 10_000 or models["models"]["tabddpm"].get("diffusion_steps") != 1_000:
        raise RuntimeError("TabDDPM budget changed")
    gpu_models = list(groups.get("gpu", []))
    gpu_resources = execution.get("gpu_resources", {})
    gpu_pools = execution.get("gpu_pools", {})
    expected_order = ["tabbyflow", "tabsyn", "tabddpm", "ctabgan_plus", "ctgan", "tvae"]
    if (
        set(gpu_resources) != set(gpu_models)
        or any(resource.get("memory") != "32G" for resource in gpu_resources.values())
        or execution.get("gpu_submission_order") != expected_order
        or set(gpu_pools) != {"default", "normal"}
        or gpu_pools.get("default", {}).get("share") != 2
        or gpu_pools.get("normal", {}).get("share") != 1
        or gpu_pools.get("default", {}).get("qos") != "default_qos"
        or gpu_pools.get("normal", {}).get("qos") != "normal"
    ):
        raise RuntimeError("32G GPU resource or 2:1 pool-allocation contract changed")

    for key in ("dataset_registry", "source_lock", "scope_decision", "protocol_decision", "split1_gate_reference"):
        name = experiment.get(key) if key in experiment else execution.get(key)
        if not isinstance(name, str) or not (project_root / "configs" / name).is_file():
            raise RuntimeError(f"missing bound config: {key}")
    registry = _read(project_root / "configs" / experiment["dataset_registry"])
    if (
        registry.get("project_revision") != PROJECT_REVISION
        or registry.get("status") != "FROZEN_FOR_EXECUTION"
        or [item["name"] for item in registry.get("datasets", [])] != experiment["datasets"]
    ):
        raise RuntimeError("dataset registry does not exactly match experiment order")
    input_lock_name = experiment["dataset_preparation"]["frozen_source_lock"]
    input_lock = _read(project_root / "configs" / input_lock_name)
    if set(input_lock["datasets"]) != set(experiment["datasets"]):
        raise RuntimeError("frozen input source scope mismatch")
    for name in experiment["datasets"]:
        frozen_source_bundle(name, input_lock, project_root)

    split1_reference = _read(project_root / "configs" / experiment["split1_gate_reference"])
    split1_hashes = _validate_split1_reference(project_root, split1_reference)
    source_lock = _read(project_root / "configs" / experiment["source_lock"])
    external_sources: dict[str, dict[str, str]] = {}
    for model, spec in models["models"].items():
        identity = verify_external_source(model, spec, source_lock, project_root=project_root)
        if identity:
            external_sources[model] = identity

    manifest_dir = output_dir / "task_manifests"
    payload = build(experiment_path, models_path, execution_path, manifest_dir)
    if payload.get("expected_cells") != EXPECTED_CELLS:
        raise RuntimeError("task-manifest builder did not produce 845 cells")
    full_rows = (manifest_dir / "scopes/full.tsv").read_text(encoding="utf-8").splitlines()
    if len(full_rows) != len(set(full_rows)) or len(full_rows) != EXPECTED_CELLS:
        raise RuntimeError("full task scope is incomplete or duplicated")

    tracked = {
        str(path.relative_to(project_root)): sha256_file(path)
        for path in _tracked_files(project_root)
    }
    forbidden = [relative for relative in tracked if "repair" in Path(relative).name.lower()]
    if forbidden:
        raise RuntimeError(f"repair-only files are forbidden in v4 tracked runtime: {forbidden}")
    stale_hits: list[str] = []
    stale_revision = "Baseline_exp_" + "v3"
    stale_project_path = "/TDG/" + stale_revision
    for relative in tracked:
        path = project_root / relative
        if path.suffix.lower() in {".py", ".json", ".sh", ".sbatch", ".md", ".toml"}:
            text = path.read_text(encoding="utf-8")
            if stale_revision in text or stale_project_path in text:
                stale_hits.append(relative)
    if stale_hits:
        raise RuntimeError(f"stale predecessor binding remains: {stale_hits}")

    config_names = {
        "experiment": experiment_path.name,
        "models": models_path.name,
        "execution": execution_path.name,
        "dataset_registry": experiment["dataset_registry"],
        "source_lock": experiment["source_lock"],
        "scope_decision": experiment["scope_decision"],
        "protocol_decision": experiment["protocol_decision"],
        "split1_gate_reference": experiment["split1_gate_reference"],
        "input_sources": input_lock_name,
    }
    config_hashes = {
        key: sha256_file(project_root / "configs" / name)
        for key, name in config_names.items()
    }
    preflight_path = output_dir / "PREFLIGHT_PASS.json"
    write_json_new(
        preflight_path,
        {
            "status": "PASS_READY_FOR_APPROVAL",
            "project_revision": PROJECT_REVISION,
            "protocol_id": PROTOCOL_ID,
            "compute_project": str(project_root),
            "expected_cells": EXPECTED_CELLS,
            "datasets": EXPECTED_DATASETS,
            "models": EXPECTED_MODELS,
            "split_seeds": list(EXPECTED_SEEDS),
            "manifest_cells": len(full_rows),
            "manifest_unique_cells": len(set(full_rows)),
            "config_names": config_names,
            "config_sha256": config_hashes,
            "source_tree_sha256": tree_sha256(project_root / "src" / "baseline_exp"),
            "tracked_manifest_sha256": _manifest_digest(tracked),
            "tracked_files_sha256": tracked,
            "external_sources": external_sources,
            "split1_evidence_sha256": split1_hashes,
            "unit_evidence_sha256": sha256_file(unit_evidence),
            "data65_evidence_sha256": sha256_file(data_evidence),
            "unit_tests": sum(int(s.get("tests", "0")) for s in suites),
            "data65_evidence_path": str(data_evidence),
            "algorithm_guards": {
                "ctabgan_plus_constant_numerical_general_scaler": True,
                "tabddpm_training_optimizer_updates": 10_000,
                "tabddpm_diffusion_timesteps": 1_000,
                "fallback_generator_forbidden": True,
                "overwrite_forbidden": True,
                "structural_na_retained": True,
            },
            "scheduler_guards": {
                "gpu_host_memory_per_cell": "32G",
                "gpu_pool_shares": {"default": 2, "normal": 1},
                "gpu_submission_order": expected_order,
            },
            "authorization": "this artifact is evidence only and does not authorize submission",
        },
    )
    write_json_new(
        output_dir / "APPROVAL.template.json",
        {
            "status": "PENDING_USER_APPROVAL",
            "project_revision": PROJECT_REVISION,
            "run_tag": None,
            "compute_project": str(project_root),
            "result_root": None,
            "export_root": None,
            "expected_cells": EXPECTED_CELLS,
            "preflight_path": str(preflight_path),
            "preflight_sha256": sha256_file(preflight_path),
            "tracked_manifest_sha256": _manifest_digest(tracked),
            "config_sha256": config_hashes,
            "note": "copy to a new file and set APPROVED plus the fresh run-specific paths only after review",
        },
    )
    return preflight_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unit-evidence", type=Path, required=True)
    parser.add_argument("--data-evidence", type=Path, required=True)
    args = parser.parse_args()
    print(
        preflight(
            project_root=args.project_root,
            experiment_path=args.experiment,
            models_path=args.models,
            execution_path=args.execution,
            output_dir=args.output_dir,
            unit_evidence=args.unit_evidence,
            data_evidence=args.data_evidence,
        )
    )
