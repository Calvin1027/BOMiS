from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .provenance import sha256_file, write_json_new


URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
NUMERICAL = ("age", "balance", "day", "campaign", "pdays", "previous")
CATEGORICAL = (
    "job", "marital", "education", "default", "housing", "loan", "contact", "month",
    "poutcome",
)
TARGET = "__target__"


def _write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8") as handle:
        frame.to_csv(handle, index=False)


def prepare(output_dir: Path, *, seed: int = 20260904, max_rows: int = 800) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    archive_path = output_dir / "bank_marketing.zip"
    request = urllib.request.Request(URL, headers={"User-Agent": "TDG-Baseline-Smoke/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, archive_path.open("xb") as handle:
        handle.write(response.read())
    with zipfile.ZipFile(archive_path) as archive:
        nested = [name for name in archive.namelist() if name.endswith("bank.zip")]
        if len(nested) != 1:
            raise RuntimeError(f"expected one nested bank.zip, found {nested!r}")
        with zipfile.ZipFile(io.BytesIO(archive.read(nested[0]))) as bank_archive:
            candidates = [name for name in bank_archive.namelist() if name.endswith("bank-full.csv")]
            if len(candidates) != 1:
                raise RuntimeError(f"expected one bank-full.csv, found {candidates!r}")
            frame = pd.read_csv(io.BytesIO(bank_archive.read(candidates[0])), sep=";")
    selected = frame.loc[:, [*NUMERICAL, *CATEGORICAL, "y"]].copy()
    selected[TARGET] = selected.pop("y").eq("yes").astype(np.int8)
    subset, _ = train_test_split(
        selected,
        train_size=max_rows,
        random_state=seed,
        shuffle=True,
        stratify=selected[TARGET],
    )
    train_valid, test = train_test_split(
        subset,
        test_size=0.15,
        random_state=seed,
        shuffle=True,
        stratify=subset[TARGET],
    )
    train, validation = train_test_split(
        train_valid,
        test_size=0.15 / 0.85,
        random_state=seed + 1,
        shuffle=True,
        stratify=train_valid[TARGET],
    )
    paths: dict[str, dict[str, str]] = {}
    for role, part in (("outer_train", train), ("validation", validation), ("test", test)):
        path = output_dir / f"{role}.csv"
        _write_csv_new(path, part.reset_index(drop=True))
        paths[role] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    manifest = output_dir / "dataset.json"
    write_json_new(
        manifest,
        {
            "schema_version": "dataset.v1",
            "status": "FROZEN_FOR_EXECUTION",
            "name": "bank_marketing_smoke_n800",
            "source": {"url": URL, "archive_sha256": sha256_file(archive_path)},
            "split_seed": seed,
            "target": TARGET,
            "positive_label": 1,
            "numerical_columns": list(NUMERICAL),
            "categorical_columns": list(CATEGORICAL),
            "M_d": 64,
            "M_d_rule": "engineering smoke fixed bank; not a scientific estimand",
            "splits": paths,
            "counts": {
                role: {"rows": len(part), "minority": int(part[TARGET].sum())}
                for role, part in (("outer_train", train), ("validation", validation), ("test", test))
            },
        },
    )
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-rows", type=int, default=800)
    arguments = parser.parse_args()
    print(prepare(arguments.output_dir, seed=arguments.seed, max_rows=arguments.max_rows))
