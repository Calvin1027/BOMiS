from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from .revision import PROJECT_REVISION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_sha256(frame: Any) -> str:
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, frame.columns)).encode("utf-8"))
    digest.update("\x1f".join(map(str, frame.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")


def claim_new_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment_manifest() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "sdv",
        "ctgan", "sdmetrics", "synthcity", "torch", "xgboost",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "packages": packages,
    }


def write_failure_new(
    work_dir: Path,
    *,
    model: str,
    dataset: str,
    split_seed: int,
    stage: str,
    code: str,
    message: str,
    context: dict[str, Any] | None = None,
) -> None:
    write_json_new(
        work_dir / "FAILURE.json",
        {
            "status": "FAILED",
            "project_revision": PROJECT_REVISION,
            "model": model,
            "dataset": dataset,
            "split_seed": split_seed,
            "stage": stage,
            "code": code,
            "message": message,
            "failure_context": dict(context or {}),
            "fallback_used": False,
            "backfill_used": False,
        },
    )
