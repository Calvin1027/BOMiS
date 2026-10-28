from __future__ import annotations

import hashlib
import json
import os
import stat
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts import AdapterFailure


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    excluded = {
        ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        "data", "ckpt", "synthetic", "outputs", "runs", "wandb",
    }
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if any(part in excluded for part in path.relative_to(root).parts):
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def isolate_source(source: Path, destination: Path) -> tuple[Path, str]:
    source = source.resolve()
    if not source.is_dir():
        raise AdapterFailure("configuration", "SOURCE_MISSING", f"source is missing: {source}")
    if destination.exists():
        raise AdapterFailure("fit", "SOURCE_DESTINATION_EXISTS", f"refusing to reuse {destination}")
    identity = tree_sha256(source)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
            "data", "ckpt", "synthetic", "outputs", "runs", "wandb",
        ),
    )
    # The frozen project snapshot is intentionally read-only.  copytree/copy2
    # preserves those permission bits, but integration patches must be applied
    # to the per-cell isolated copy.  Make only that disposable copy writable;
    # the frozen source tree itself remains immutable.
    for path in (destination, *destination.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
    if tree_sha256(destination) != identity:
        raise AdapterFailure("fit", "SOURCE_COPY_MISMATCH", "isolated source hash mismatch")
    return destination, identity


def write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    write_text_new(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    python_path: Path | None = None,
    stage: str = "fit",
) -> None:
    if log_path.exists():
        raise AdapterFailure("fit", "LOG_EXISTS", f"refusing to overwrite {log_path}")
    environment = os.environ.copy()
    if python_path is not None:
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(python_path) if not existing else f"{python_path}{os.pathsep}{existing}"
        )
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise AdapterFailure(
            stage, "UPSTREAM_PROCESS_FAILED", f"command failed with code {completed.returncode}"
        )


def read_exact_csv(path: Path, *, columns: tuple[str, ...], n: int) -> pd.DataFrame:
    if not path.is_file():
        raise AdapterFailure("sample", "SAMPLE_FILE_MISSING", f"missing sample file {path}")
    frame = pd.read_csv(path)
    if tuple(frame.columns) != columns:
        if frame.shape[1] == len(columns):
            frame.columns = list(columns)
        else:
            raise AdapterFailure("sample", "SAMPLE_SCHEMA_MISMATCH", "sample CSV schema mismatch")
    if len(frame) != n:
        raise AdapterFailure("sample", "RAW_SAMPLE_COUNT_MISMATCH", "sample CSV row mismatch")
    return frame.loc[:, list(columns)]
