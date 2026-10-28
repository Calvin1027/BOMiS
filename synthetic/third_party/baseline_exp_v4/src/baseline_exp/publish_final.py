from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .provenance import sha256_file, write_json_new
from .revision import PROJECT_REVISION, PROTOCOL_ID


PUBLISHED_FILES = (
    "cell_metrics.csv",
    "dataset_model_summary.csv",
    "model_dataset_macro_summary.csv",
    "generated_samples_index.csv",
    "model_split_defined_dataset_counts.csv",
    "AGGREGATED.json",
)


def publish(run_root: Path, destination: Path, expected_cells: int) -> Path:
    aggregate = run_root / "aggregate"
    final_gate = run_root / "validation" / "final" / "PASS.json"
    deep_audit = run_root / "validation" / "deep_audit.json"
    aggregated_marker = aggregate / "AGGREGATED.json"
    for path in (final_gate, deep_audit, aggregated_marker):
        if not path.is_file():
            raise RuntimeError(f"required final artifact is missing: {path}")
    gate = json.loads(final_gate.read_text(encoding="utf-8"))
    audit = json.loads(deep_audit.read_text(encoding="utf-8"))
    aggregated = json.loads(aggregated_marker.read_text(encoding="utf-8"))
    if (
        gate.get("status") != "PASS"
        or gate.get("expected_cells") != expected_cells
        or gate.get("successful_cells") != expected_cells
        or audit.get("counts") != {"PASS": expected_cells, "FAIL": 0, "MISSING": 0}
        or aggregated.get("status") != "AGGREGATED_DESCRIPTIVE"
        or aggregated.get("project_revision") != PROJECT_REVISION
        or aggregated.get("protocol_id") != PROTOCOL_ID
        or aggregated.get("expected_cells") != expected_cells
        or aggregated.get("observed_cells") != expected_cells
    ):
        raise RuntimeError("final validation chain is not complete")
    for field in ("experiment_sha256", "benchmark_source_tree_sha256"):
        if not aggregated.get(field) or not gate.get(field) == audit.get(field) == aggregated[field]:
            raise RuntimeError(f"final validation chain provenance mismatch: {field}")
    for name in PUBLISHED_FILES:
        if name != "AGGREGATED.json" and sha256_file(aggregate / name) != aggregated.get("artifacts", {}).get(name):
            raise RuntimeError(f"aggregate artifact changed after validation: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    copied: dict[str, str] = {}
    for name in PUBLISHED_FILES:
        source = aggregate / name
        if not source.is_file():
            raise RuntimeError(f"aggregate artifact is missing: {source}")
        target = destination / name
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        copied[name] = sha256_file(target)
        if name != "AGGREGATED.json" and copied[name] != aggregated["artifacts"][name]:
            raise RuntimeError(f"aggregate artifact changed during publication: {name}")
    generation_link = destination / "generation"
    relative_target = os.path.relpath(run_root / "generation", destination)
    generation_link.symlink_to(relative_target, target_is_directory=True)
    write_json_new(
        destination / f"COMPLETE_{expected_cells}.json",
        {
            "status": "COMPLETE",
            "project_revision": PROJECT_REVISION,
            "protocol_id": PROTOCOL_ID,
            "expected_cells": expected_cells,
            "validated_generation_cells": expected_cells,
            "validated_metric_cells": expected_cells,
            "generation_directory": str(generation_link),
            "final_gate_sha256": sha256_file(final_gate),
            "deep_audit_sha256": sha256_file(deep_audit),
            "published_artifacts": copied,
        },
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-cells", type=int, required=True)
    args = parser.parse_args()
    print(publish(args.run_root, args.destination, args.expected_cells))
