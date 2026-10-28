from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .adapters._external import tree_sha256
from .contracts import AdapterFailure
from .provenance import sha256_file


SOURCE_KEYS = {
    "ctabgan_plus": "ctabgan_plus",
    "tabddpm": "tabddpm",
    "tabsyn": "tabsyn",
    "tabbyflow": "tabbyflow",
    "forest_flow": "forest_flow",
}


def verify_external_source(
    model: str,
    model_spec: dict[str, Any],
    source_lock: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, str]:
    if model not in SOURCE_KEYS:
        return {}
    raw_path = Path(model_spec["source_dir"])
    source = raw_path if raw_path.is_absolute() else project_root / raw_path
    source = source.resolve()
    if not source.is_dir():
        raise AdapterFailure("configuration", "SOURCE_MISSING", str(source))
    locked = source_lock["sources"][SOURCE_KEYS[model]]
    if "patch_file" in model_spec:
        patch = Path(model_spec["patch_file"])
        patch = patch if patch.is_absolute() else (project_root / patch)
        if not patch.is_file() or sha256_file(patch) != locked.get("sampling_patch_sha256"):
            raise AdapterFailure("configuration", "SAMPLING_PATCH_HASH_MISMATCH", model)
    for relative, expected in locked.get("patched_files_sha256", {}).items():
        path = source / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise AdapterFailure("configuration", "PATCH_HASH_MISMATCH", f"{model}: {relative}")
    expected_tree = locked.get("source_tree_sha256")
    if expected_tree is not None and tree_sha256(source) != expected_tree:
        raise AdapterFailure("configuration", "SOURCE_TREE_MISMATCH", model)
    git_dir = source / ".git"
    if git_dir.exists():
        head = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if head.returncode != 0 or head.stdout.strip() != locked["commit"]:
            raise AdapterFailure("configuration", "SOURCE_COMMIT_MISMATCH", model)
    elif expected_tree is None and not locked.get("patched_files_sha256"):
        raise AdapterFailure(
            "configuration", "SOURCE_IDENTITY_UNVERIFIABLE", f"freeze a tree hash for {model}"
        )
    return {"path": str(source), "tree_sha256": tree_sha256(source)}
