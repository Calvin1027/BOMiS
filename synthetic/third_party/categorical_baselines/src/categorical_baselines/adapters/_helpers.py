from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import AdapterFailure, DatasetSchema, SampleResult


def seed_everything(seed: int, *, require_torch: bool = False, require_cuda: bool = False) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
    except ImportError:
        if require_torch:
            raise
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    elif require_cuda:
        raise AdapterFailure("configuration", "CUDA_UNAVAILABLE", "CUDA is required")


def canonical_joint(train: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    table = train.loc[:, [*schema.feature_columns, schema.target]].copy().reset_index(drop=True)
    table[schema.target] = schema.binary_target(table)
    return table


def new_artifact_path(work_dir: Path, name: str) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / name
    if path.exists():
        raise AdapterFailure("fit", "ARTIFACT_EXISTS", f"refusing to overwrite {path}")
    return path


def bounded_minority_rejection(
    draw: Callable[[int, int], pd.DataFrame],
    schema: DatasetSchema,
    *,
    n: int,
    sampling_seed: int,
    raw_multiplier: int,
    raw_batch_size: int,
    raw_cap: int | None = None,
) -> SampleResult:
    if raw_multiplier < 1 or raw_batch_size < 1:
        raise AdapterFailure("configuration", "INVALID_REJECTION_CAP", "invalid rejection cap")
    raw_cap = n * raw_multiplier if raw_cap is None else int(raw_cap)
    if raw_cap < n:
        raise ValueError("raw cap must be at least the requested output count")
    accepted: list[pd.DataFrame] = []
    valid_y1_rows = 0
    decoded_y1_rows = 0
    invalid_y1_rows = 0
    invalid_target_rows = 0
    rejected_y0_rows = 0
    raw_count = 0
    requested_raw_count = 0
    chunk = 0
    while valid_y1_rows < n and raw_count < raw_cap:
        request = min(raw_batch_size, raw_cap - raw_count)
        raw = draw(request, sampling_seed + chunk)
        requested_raw_count += request
        if not isinstance(raw, pd.DataFrame):
            raise AdapterFailure(
                "sample",
                "RAW_SAMPLE_INVALID",
                "upstream did not return a DataFrame",
                context={"requested_rows": int(request), "raw_chunk": int(chunk)},
            )
        if len(raw) < 1 or len(raw) > request:
            raise AdapterFailure(
                "sample",
                "RAW_SAMPLE_COUNT_INVALID",
                f"upstream returned {len(raw)} rows for a request of {request}",
                context={
                    "requested_rows": int(request),
                    "returned_rows": len(raw),
                    "raw_chunk": int(chunk),
                },
            )
        expected = {*schema.feature_columns, schema.target}
        if raw.columns.duplicated().any() or set(raw.columns) != expected:
            raise AdapterFailure(
                "sample", "RAW_SCHEMA_MISMATCH", "joint raw sample has unexpected columns"
            )
        raw = raw.loc[:, [*schema.feature_columns, schema.target]]
        # Every joint generator is fitted to ``canonical_joint``, whose target is
        # strictly 0/1. Partition every raw row by target before checking feature
        # validity so the accounting identities are exact and exhaustive.
        target = pd.to_numeric(raw[schema.target], errors="coerce")
        target_values = target.to_numpy(dtype=float)
        finite_target = np.isfinite(target_values)
        valid_target = finite_target & np.isin(target_values, [0.0, 1.0])
        y1_mask = valid_target & (target_values == 1.0)
        y0_mask = valid_target & (target_values == 0.0)
        invalid_target_rows += int((~valid_target).sum())
        rejected_y0_rows += int(y0_mask.sum())
        decoded_y1_rows += int(y1_mask.sum())

        y1_features = raw.loc[y1_mask, list(schema.feature_columns)].copy()
        feature_valid = schema.feature_validity_mask(y1_features)
        valid_chunk = y1_features.loc[feature_valid].copy()
        invalid_y1_rows += int((~feature_valid).sum())
        valid_y1_rows += len(valid_chunk)
        if len(valid_chunk):
            accepted.append(valid_chunk)
        # Some official generators legitimately return a short final batch.
        # Count only rows that were actually decoded; never invent or silently
        # truncate rows.  A zero or oversized batch remains terminal above.
        raw_count += len(raw)
        chunk += 1
    counters = {
        "required_M_d": int(n),
        "total_raw_draws": int(raw_count),
        "total_requested_raw_rows": int(requested_raw_count),
        "invalid_target_rows": int(invalid_target_rows),
        "rejected_y0_rows": int(rejected_y0_rows),
        "decoded_y1_rows": int(decoded_y1_rows),
        "valid_y1_rows": int(valid_y1_rows),
        "invalid_y1_rows": int(invalid_y1_rows),
        "invalid_decoded_rows": int(invalid_target_rows + invalid_y1_rows),
        "selected_output_rows": int(n if valid_y1_rows >= n else 0),
        "raw_cap": int(raw_cap),
        "raw_chunks": int(chunk),
        "y1_rate": decoded_y1_rows / raw_count if raw_count else None,
        "valid_given_y1_rate": valid_y1_rows / decoded_y1_rows if decoded_y1_rows else None,
        "overall_acceptance_rate": valid_y1_rows / raw_count if raw_count else None,
        "selection_policy": "stable_generation_order_first_M_d",
        "termination_reason": (
            "required_count_reached" if valid_y1_rows >= n else "raw_cap_exhausted"
        ),
    }
    if raw_count != invalid_target_rows + rejected_y0_rows + decoded_y1_rows:
        raise AdapterFailure(
            "validation", "RAW_DRAW_CONSERVATION_FAILED", repr(counters), context=counters
        )
    if decoded_y1_rows != valid_y1_rows + invalid_y1_rows:
        raise AdapterFailure(
            "validation", "Y1_CONSERVATION_FAILED", repr(counters), context=counters
        )
    if valid_y1_rows < n:
        raise AdapterFailure(
            "sample",
            "CAPACITY_FAILURE",
            f"accepted {valid_y1_rows} valid minority rows from raw cap {raw_cap}; required {n}",
            context=counters,
        )
    frame = pd.concat(accepted, ignore_index=True).iloc[:n].copy().reset_index(drop=True)
    return SampleResult(
        frame=frame,
        sampling_seed=sampling_seed,
        raw_draw_count=raw_count,
        minority_acceptance_rate=valid_y1_rows / raw_count,
        raw_draw_cap=raw_cap,
        rejection_used=True,
        accepted_minority_count=valid_y1_rows,
        details={
            "sampling_mode": "bounded_joint_rejection",
            **counters,
            "selected_output_rate": n / raw_count,
        },
        decoded_y1_rows=decoded_y1_rows,
        valid_y1_rows=valid_y1_rows,
        invalid_y1_rows=invalid_y1_rows,
        invalid_target_rows=invalid_target_rows,
        rejected_y0_rows=rejected_y0_rows,
        y1_rate=decoded_y1_rows / raw_count,
        valid_given_y1_rate=(valid_y1_rows / decoded_y1_rows if decoded_y1_rows else None),
        overall_acceptance_rate=valid_y1_rows / raw_count,
        selected_output_rows=n,
        termination_reason="required_count_reached",
    )
