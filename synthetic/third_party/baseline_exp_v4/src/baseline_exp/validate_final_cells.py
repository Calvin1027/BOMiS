from __future__ import annotations

import argparse
import json
from pathlib import Path

from .provenance import sha256_file, write_json_new
from .revision import PROJECT_REVISION, PROTOCOL_ID
from .adapters._external import tree_sha256
from .final_integrity import validate_cell_integrity


def _cells(manifest_root: Path) -> list[tuple[str, str, int]]:
    cells: list[tuple[str, str, int]] = []
    manifests = [manifest_root] if manifest_root.is_file() else sorted(manifest_root.rglob("*.tsv"))
    for manifest in manifests:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            dataset, model, seed = line.split("\t")
            cells.append((dataset, model, int(seed)))
    if not cells:
        raise RuntimeError("empty task manifest scope")
    if len(cells) != len(set(cells)):
        raise RuntimeError("duplicate cells across task manifests")
    return cells


def validate(experiment_path: Path, manifest_root: Path, run_root: Path, output_dir: Path, label: str) -> Path:
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    protocol_id = PROTOCOL_ID
    if (
        experiment.get("status") != "FROZEN_FOR_EXECUTION"
        or experiment.get("protocol_id") != protocol_id
        or experiment.get("project_revision") != PROJECT_REVISION
    ):
        raise RuntimeError("cell gate requires the frozen full-access protocol")
    models_reference = experiment.get("models_config")
    if not isinstance(models_reference, str):
        raise RuntimeError("experiment has no bound model config")
    models_path = experiment_path.parent / models_reference
    if not models_path.is_file():
        raise RuntimeError(f"bound model config is missing: {models_path}")
    models = json.loads(models_path.read_text(encoding="utf-8"))
    if (
        models.get("status") != "FROZEN_FOR_EXECUTION"
        or models.get("protocol_id") != protocol_id
        or models.get("project_revision") != PROJECT_REVISION
        or set(models.get("models", {})) != set(experiment.get("models", []))
    ):
        raise RuntimeError("bound model config does not match Baseline_exp_v4")
    models_hash = sha256_file(models_path)
    expected_hash = sha256_file(experiment_path)
    cells = _cells(manifest_root)
    if label.startswith("FINAL_"):
        expected_cells = {(d, m, int(s)) for d in experiment["datasets"]
                          for m in experiment["models"] for s in experiment["split"]["seeds"]}
        if set(cells) != expected_cells:
            raise RuntimeError("final manifest is not the complete experiment Cartesian product")
    frozen_source_hash = tree_sha256(Path(__file__).resolve().parent)
    failures: list[dict[str, object]] = []
    evaluator_source_hashes: set[str] = set()
    for dataset, model, seed in cells:
        generation = run_root / "generation" / dataset / model / f"split-{seed}"
        evaluation = run_root / "evaluation" / dataset / model / f"split-{seed}"
        generated_marker = generation / "GENERATED.json"
        success_marker = evaluation / "SUCCESS.json"
        problem = None
        if (
            dataset not in experiment["datasets"]
            or model not in experiment["models"]
            or seed not in experiment["split"]["seeds"]
        ):
            problem = "CELL_OUTSIDE_EXPERIMENT"
        elif not generated_marker.is_file():
            problem = "GENERATION_MISSING_OR_FAILED"
        elif not success_marker.is_file():
            problem = "EVALUATION_MISSING_OR_FAILED"
        else:
            try:
                _, marker, _ = validate_cell_integrity(
                    generation, evaluation, dataset=dataset, model=model, seed=seed,
                    experiment_hash=expected_hash, models_hash=models_hash, source_hash=frozen_source_hash,
                )
            except Exception as error:
                failures.append({"dataset": dataset, "model": model, "split_seed": seed,
                                 "code": "CELL_INTEGRITY_FAILED", "detail": str(error)})
                continue
            if (
                marker.get("status") != "SUCCESS"
                or marker.get("experiment_sha256") != expected_hash
                or marker.get("protocol_id") != protocol_id
                or marker.get("project_revision") != PROJECT_REVISION
                or marker.get("models_config_sha256") != models_hash
            ):
                problem = "INVALID_EVALUATION_PROVENANCE"
            else:
                source_hash = marker.get("benchmark_source_tree_sha256")
                if not isinstance(source_hash, str) or len(source_hash) != 64:
                    problem = "INVALID_EVALUATOR_SOURCE_IDENTITY"
                else:
                    evaluator_source_hashes.add(source_hash)
        if problem:
            failures.append({"dataset": dataset, "model": model, "split_seed": seed, "code": problem})

    output_dir.mkdir(parents=True, exist_ok=False)
    unsuccessful_cells = len(failures)
    if len(evaluator_source_hashes) > 1:
        failures.append(
            {
                "dataset": "__gate__",
                "model": "__gate__",
                "split_seed": -1,
                "code": "MIXED_EVALUATOR_SOURCE_IDENTITIES",
            }
        )
    passed = not failures
    write_json_new(
        output_dir / ("PASS.json" if passed else "FAILURE.json"),
        {
            "status": "PASS" if passed else "FAILED",
            "project_revision": PROJECT_REVISION,
            "label": label,
            "expected_cells": len(cells),
            "successful_cells": len(cells) - unsuccessful_cells,
            "experiment_sha256": expected_hash,
            "protocol_id": protocol_id,
            "models_config_sha256": models_hash,
            "benchmark_source_tree_sha256": (
                next(iter(evaluator_source_hashes)) if len(evaluator_source_hashes) == 1 else None
            ),
            "failures": failures,
        },
    )
    if not passed:
        raise RuntimeError(f"{label} failed: {len(failures)}/{len(cells)} cells unsuccessful")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    print(validate(args.experiment, args.manifest_root, args.run_root, args.output_dir, args.label))
