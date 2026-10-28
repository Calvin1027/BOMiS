from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype


FailureStage = Literal["configuration", "fit", "sample", "validation", "metric"]


class AdapterFailure(RuntimeError):
    """Expected, attributable failure of one model x dataset x split cell."""

    def __init__(
        self,
        stage: FailureStage,
        code: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.context = dict(context or {})


@dataclass(frozen=True)
class DatasetSchema:
    target: str
    positive_label: Any
    target_domain: tuple[Any, Any]
    feature_columns: tuple[str, ...]
    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    dtypes: dict[str, str]
    category_domains: dict[str, tuple[Any, ...]]
    category_dtype_domains: dict[str, tuple[Any, ...]]
    category_ordered: dict[str, bool]

    @classmethod
    def from_outer_train(
        cls,
        train: pd.DataFrame,
        *,
        target: str,
        positive_label: Any,
        numerical_columns: tuple[str, ...],
        categorical_columns: tuple[str, ...],
    ) -> DatasetSchema:
        if train.columns.duplicated().any():
            raise AdapterFailure("configuration", "DUPLICATE_COLUMNS", "column names are not unique")
        if target not in train:
            raise AdapterFailure("configuration", "TARGET_MISSING", f"missing target {target!r}")
        features = tuple(column for column in train.columns if column != target)
        if not features or not all(isinstance(column, str) for column in train.columns):
            raise AdapterFailure(
                "configuration", "INVALID_COLUMNS", "nonempty string-named features are required"
            )
        if set(numerical_columns).intersection(categorical_columns):
            raise AdapterFailure("configuration", "SCHEMA_OVERLAP", "feature types overlap")
        if set(features) != set(numerical_columns).union(categorical_columns):
            raise AdapterFailure(
                "configuration", "SCHEMA_COVERAGE", "feature types do not exactly cover features"
            )
        if train[target].isna().any():
            raise AdapterFailure("configuration", "TARGET_NA", "outer-train target contains missing values")
        target_order = tuple(pd.unique(train[target]).tolist())
        target_values = set(target_order)
        if len(target_values) != 2 or positive_label not in target_values:
            raise AdapterFailure(
                "configuration", "TARGET_NOT_BINARY", "outer-train target must contain two classes"
            )
        dtypes = {column: str(train[column].dtype) for column in features}
        domains: dict[str, tuple[Any, ...]] = {}
        dtype_domains: dict[str, tuple[Any, ...]] = {}
        ordered: dict[str, bool] = {}
        for column in categorical_columns:
            if train[column].isna().any():
                raise AdapterFailure(
                    "configuration", "UNRESOLVED_MISSING", f"missing category in {column!r}"
                )
            domains[column] = tuple(pd.unique(train[column]).tolist())
            if not domains[column]:
                raise AdapterFailure(
                    "configuration", "EMPTY_CATEGORY_DOMAIN", f"empty domain for {column!r}"
                )
            tokens = [str(value) for value in domains[column]]
            if len(tokens) != len(set(tokens)):
                raise AdapterFailure(
                    "configuration",
                    "AMBIGUOUS_CATEGORY_TOKENS",
                    f"string conversion is non-injective for {column!r}",
                )
            dtype = train[column].dtype
            ordered[column] = bool(dtype.ordered) if isinstance(dtype, pd.CategoricalDtype) else False
            dtype_domains[column] = (
                tuple(dtype.categories.tolist())
                if isinstance(dtype, pd.CategoricalDtype)
                else domains[column]
            )
        for column in numerical_columns:
            numeric = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                raise AdapterFailure(
                    "configuration", "NONFINITE_TRAIN", f"non-finite numerical value in {column!r}"
                )
        return cls(
            target=target,
            positive_label=positive_label,
            target_domain=target_order,  # type: ignore[arg-type]
            feature_columns=features,
            numerical_columns=tuple(numerical_columns),
            categorical_columns=tuple(categorical_columns),
            dtypes=dtypes,
            category_domains=domains,
            category_dtype_domains=dtype_domains,
            category_ordered=ordered,
        )

    def binary_target(self, frame: pd.DataFrame) -> np.ndarray:
        if self.target not in frame:
            raise AdapterFailure("validation", "TARGET_MISSING", "target missing from input")
        values = frame[self.target]
        if values.isna().any():
            raise AdapterFailure("validation", "TARGET_NA", "target contains missing values")
        unknown = set(values.unique().tolist()).difference(self.target_domain)
        if unknown:
            raise AdapterFailure("validation", "UNKNOWN_TARGET", f"unknown target values: {unknown!r}")
        return (values == self.positive_label).to_numpy(dtype=np.int8)

    def restore_and_validate(self, generated: pd.DataFrame, *, expected_n: int) -> pd.DataFrame:
        if not isinstance(generated, pd.DataFrame):
            raise AdapterFailure("validation", "NOT_DATAFRAME", "generator output is not a DataFrame")
        if generated.columns.duplicated().any():
            raise AdapterFailure("validation", "DUPLICATE_COLUMNS", "generated columns are not unique")
        if tuple(generated.columns) != self.feature_columns:
            raise AdapterFailure(
                "validation",
                "SCHEMA_MISMATCH",
                f"expected columns {self.feature_columns!r}; received {tuple(generated.columns)!r}",
            )
        if len(generated) != expected_n:
            raise AdapterFailure(
                "validation", "ROW_COUNT_MISMATCH", f"expected {expected_n}; received {len(generated)}"
            )
        result = generated.copy().reset_index(drop=True)
        for column in self.numerical_columns:
            numeric = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                raise AdapterFailure(
                    "validation", "NONFINITE_OUTPUT", f"non-finite output in {column!r}"
                )
            original_dtype = self.dtypes[column]
            dtype = pd.api.types.pandas_dtype(original_dtype)
            if is_integer_dtype(dtype) or is_bool_dtype(dtype):
                # Several official generators model integer-valued measurements on a
                # continuous latent scale. Nearest-integer projection is the shared,
                # deterministic inverse-domain map before exact dtype restoration.
                rounded = np.rint(numeric)
                if is_bool_dtype(dtype):
                    valid_range = (rounded >= 0) & (rounded <= 1)
                else:
                    limits = np.iinfo(dtype)
                    # Use an exclusive power-of-two upper bound: float64 cannot
                    # exactly represent the largest int64/uint64 value.
                    valid_range = (rounded >= limits.min) & (rounded < int(limits.max) + 1)
                if not valid_range.all():
                    raise AdapterFailure("validation", "NUMERIC_DTYPE_OVERFLOW", column)
                result[column] = rounded.astype(dtype)
            elif is_float_dtype(dtype):
                result[column] = numeric.astype(dtype)
            else:
                result[column] = numeric
            if not np.isfinite(result[column].to_numpy(dtype=float)).all():
                raise AdapterFailure("validation", "NUMERIC_DTYPE_OVERFLOW", column)
        for column in self.categorical_columns:
            if result[column].isna().any():
                raise AdapterFailure(
                    "validation", "MISSING_CATEGORY", f"missing generated category in {column!r}"
                )
            values = result[column].astype(str)
            lookup = {str(value): value for value in self.category_domains[column]}
            unknown = sorted(set(values.unique()).difference(lookup))
            if unknown:
                raise AdapterFailure(
                    "validation",
                    "UNKNOWN_CATEGORY",
                    f"unknown categories in {column!r}: {unknown[:10]!r}",
                )
            original_dtype = pd.api.types.pandas_dtype(self.dtypes[column])
            if isinstance(original_dtype, pd.CategoricalDtype):
                result[column] = pd.Categorical(
                    values.map(lookup),
                    categories=list(self.category_dtype_domains[column]),
                    ordered=self.category_ordered[column],
                )
            else:
                # Restore the exact outer-train values, rather than their serialization tokens.
                restored = values.map(lookup)
                try:
                    result[column] = restored.astype(original_dtype)
                except (TypeError, ValueError) as error:
                    raise AdapterFailure(
                        "validation",
                        "CATEGORY_DTYPE_RESTORATION_FAILED",
                        f"cannot restore dtype for {column!r}: {error}",
                    ) from error
        return result.loc[:, list(self.feature_columns)]

    def feature_validity_mask(self, generated: pd.DataFrame) -> np.ndarray:
        """Return rows that can pass the feature-only restoration contract.

        Joint generators use this before minority rejection so a decoded ``Y=1``
        row with an invalid feature is never counted as an accepted minority row.
        Structural schema errors remain terminal rather than being treated as
        ordinary decoder rejection.
        """
        if not isinstance(generated, pd.DataFrame):
            raise AdapterFailure("sample", "RAW_NOT_DATAFRAME", "raw features are not a DataFrame")
        if generated.columns.duplicated().any():
            raise AdapterFailure("sample", "RAW_DUPLICATE_COLUMNS", "raw feature columns are duplicated")
        if tuple(generated.columns) != self.feature_columns:
            raise AdapterFailure(
                "sample",
                "RAW_SCHEMA_MISMATCH",
                f"expected raw features {self.feature_columns!r}; received {tuple(generated.columns)!r}",
            )
        valid = np.ones(len(generated), dtype=bool)
        for column in self.numerical_columns:
            numeric = pd.to_numeric(generated[column], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(numeric)
            valid &= finite
            dtype = pd.api.types.pandas_dtype(self.dtypes[column])
            if is_integer_dtype(dtype) or is_bool_dtype(dtype):
                rounded = np.rint(np.where(finite, numeric, 0.0))
                if is_bool_dtype(dtype):
                    valid &= (rounded >= 0) & (rounded <= 1)
                else:
                    limits = np.iinfo(dtype)
                    valid &= (rounded >= limits.min) & (rounded < int(limits.max) + 1)
            elif is_float_dtype(dtype):
                with np.errstate(over="ignore", invalid="ignore"):
                    restored = numeric.astype(dtype)
                valid &= np.isfinite(restored.astype(float))
        for column in self.categorical_columns:
            present = generated[column].notna().to_numpy()
            domain = {str(value) for value in self.category_domains[column]}
            known = generated[column].astype(str).isin(domain).to_numpy()
            valid &= present & known
        return valid


@dataclass(frozen=True)
class FitResult:
    model: str
    model_seed: int
    n_outer_train: int
    n_minority_train: int
    sampling_mode: str
    loss_artifacts: tuple[str, ...] = ()
    checkpoint_artifacts: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)
    training_access: str = ""
    actual_training_population: str = ""
    cross_class_usage: str = ""
    minority_sampling_semantics: str = ""


@dataclass(frozen=True)
class SampleResult:
    frame: pd.DataFrame
    sampling_seed: int
    raw_draw_count: int | None
    minority_acceptance_rate: float | None
    raw_draw_cap: int | None
    rejection_used: bool | None = None
    accepted_minority_count: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
    decoded_y1_rows: int | None = None
    valid_y1_rows: int | None = None
    invalid_y1_rows: int | None = None
    invalid_target_rows: int | None = None
    rejected_y0_rows: int | None = None
    y1_rate: float | None = None
    valid_given_y1_rate: float | None = None
    overall_acceptance_rate: float | None = None
    selected_output_rows: int | None = None
    termination_reason: str | None = None


def sample_result_metadata(result: SampleResult) -> dict[str, Any]:
    """Serializable sample provenance without the potentially large output frame."""
    return {
        "sampling_seed": result.sampling_seed,
        "raw_draw_count": result.raw_draw_count,
        "minority_acceptance_rate": result.minority_acceptance_rate,
        "raw_draw_cap": result.raw_draw_cap,
        "rejection_used": result.rejection_used,
        "accepted_minority_count": result.accepted_minority_count,
        "decoded_y1_rows": result.decoded_y1_rows,
        "valid_y1_rows": result.valid_y1_rows,
        "invalid_y1_rows": result.invalid_y1_rows,
        "invalid_target_rows": result.invalid_target_rows,
        "rejected_y0_rows": result.rejected_y0_rows,
        "y1_rate": result.y1_rate,
        "valid_given_y1_rate": result.valid_given_y1_rate,
        "overall_acceptance_rate": result.overall_acceptance_rate,
        "selected_output_rows": result.selected_output_rows,
        "termination_reason": result.termination_reason,
        "details": result.details,
    }


_ACTUAL_TRAINING_POPULATIONS = {
    "full_joint",
    "full_labeled_outer_train",
    "minority_rows_for_resampling",
    "minority_conditional_marginals",
}
_CROSS_CLASS_USAGE = {
    "none",
    "minority_neighbourhood",
    "class_association_and_minority_neighbourhood",
    "local_boundary",
    "local_density",
    "shared_joint_model",
}
_MINORITY_SAMPLING_SEMANTICS = {
    "bootstrap",
    "direct_conditional",
    "direct_oversampling",
    "native_conditioning",
    "bounded_joint_rejection",
}


def validate_fit_contract(
    result: FitResult,
    *,
    expected_model: str,
    expected_outer_train_n: int,
) -> None:
    """Validate the common full-access input contract for every active adapter."""
    if result.model != expected_model:
        raise AdapterFailure("fit", "FIT_MODEL_MISMATCH", result.model)
    if result.n_outer_train != expected_outer_train_n:
        raise AdapterFailure(
            "fit",
            "FIT_ACCESS_ROW_MISMATCH",
            f"adapter recorded {result.n_outer_train}; full outer-train has {expected_outer_train_n}",
        )
    if result.training_access != "full_outer_train":
        raise AdapterFailure(
            "fit", "FIT_ACCESS_NOT_FULL_OUTER_TRAIN", result.training_access or "missing"
        )
    if result.actual_training_population not in _ACTUAL_TRAINING_POPULATIONS:
        raise AdapterFailure(
            "fit", "ACTUAL_TRAINING_POPULATION_INVALID", result.actual_training_population or "missing"
        )
    if result.cross_class_usage not in _CROSS_CLASS_USAGE:
        raise AdapterFailure(
            "fit", "CROSS_CLASS_USAGE_INVALID", result.cross_class_usage or "missing"
        )
    if result.minority_sampling_semantics not in _MINORITY_SAMPLING_SEMANTICS:
        raise AdapterFailure(
            "fit",
            "MINORITY_SAMPLING_SEMANTICS_INVALID",
            result.minority_sampling_semantics or "missing",
        )
    if not result.loss_artifacts and not result.details.get("loss_unavailable_reason"):
        raise AdapterFailure(
            "fit", "LOSS_PROVENANCE_MISSING", "loss artifact or unavailability reason required"
        )
    if not result.checkpoint_artifacts and not result.details.get(
        "checkpoint_unavailable_reason"
    ):
        raise AdapterFailure(
            "fit",
            "CHECKPOINT_PROVENANCE_MISSING",
            "checkpoint artifact or unavailability reason required",
        )


def validate_fit_artifacts(result: FitResult, *, work_dir: Path) -> None:
    """Require every declared fit artifact to exist inside the cell model directory."""
    root = work_dir.resolve()
    for kind, entries in (
        ("loss", result.loss_artifacts),
        ("checkpoint", result.checkpoint_artifacts),
    ):
        for entry in entries:
            relative = Path(entry)
            candidate = (root / relative).resolve()
            if relative.is_absolute() or root not in candidate.parents:
                raise AdapterFailure("fit", "FIT_ARTIFACT_PATH_INVALID", f"{kind}: {entry}")
            if not candidate.exists():
                raise AdapterFailure("fit", "FIT_ARTIFACT_MISSING", f"{kind}: {entry}")


class GeneratorAdapter(ABC):
    name: str

    def __init__(self) -> None:
        self.schema: DatasetSchema | None = None
        self.fit_result: FitResult | None = None

    @abstractmethod
    def fit(
        self,
        train: pd.DataFrame,
        schema: DatasetSchema,
        *,
        model_seed: int,
        work_dir: Path,
    ) -> FitResult:
        """Fit using outer-train only."""

    @abstractmethod
    def _sample_minority(self, n: int, *, sampling_seed: int) -> SampleResult:
        """Return an unvalidated feature-only minority table."""

    def sample_minority(self, n: int, *, sampling_seed: int) -> SampleResult:
        if n < 1:
            raise AdapterFailure("sample", "INVALID_SAMPLE_SIZE", "requested sample size must be positive")
        if self.schema is None or self.fit_result is None:
            raise AdapterFailure("sample", "NOT_FITTED", f"{self.name} has not been fitted")
        if sampling_seed == self.fit_result.model_seed:
            raise AdapterFailure(
                "configuration", "SEED_ROLE_COLLISION", "fit seed and sampling seed must differ"
            )
        result = self._sample_minority(n, sampling_seed=sampling_seed)
        try:
            validated = self.schema.restore_and_validate(result.frame, expected_n=n)
        except AdapterFailure as error:
            context = sample_result_metadata(result)
            context.update(error.context)
            raise AdapterFailure(
                error.stage, error.code, str(error), context=context
            ) from error
        if result.sampling_seed != sampling_seed:
            raise AdapterFailure("validation", "SEED_MISMATCH", "adapter reported a different sample seed")
        if result.rejection_used is None:
            raise AdapterFailure(
                "validation", "REJECTION_MODE_MISSING", "adapter did not declare rejection mode"
            )
        if result.rejection_used:
            if result.raw_draw_count is None:
                raise AdapterFailure(
                    "validation", "RAW_COUNT_MISSING", "rejection sampling requires raw count"
                )
            if result.raw_draw_count < n:
                raise AdapterFailure(
                    "validation", "RAW_COUNT_INVALID", "raw draw count is smaller than output count"
                )
            accepted = result.accepted_minority_count
            if accepted is None:
                accepted = n
            if accepted < n or accepted > result.raw_draw_count:
                raise AdapterFailure(
                    "validation", "ACCEPTED_COUNT_INVALID", "accepted minority count is invalid"
                )
            expected_rate = accepted / result.raw_draw_count
            if result.minority_acceptance_rate is None or not np.isclose(
                result.minority_acceptance_rate, expected_rate, rtol=1e-10, atol=1e-12
            ):
                raise AdapterFailure(
                    "validation", "ACCEPTANCE_RATE_INVALID", "acceptance rate is inconsistent"
                )
            counters = {
                "decoded_y1_rows": result.decoded_y1_rows,
                "valid_y1_rows": result.valid_y1_rows,
                "invalid_y1_rows": result.invalid_y1_rows,
                "invalid_target_rows": result.invalid_target_rows,
                "rejected_y0_rows": result.rejected_y0_rows,
                "selected_output_rows": result.selected_output_rows,
            }
            if any(value is None for value in counters.values()):
                raise AdapterFailure(
                    "validation", "REJECTION_COUNTER_MISSING", repr(counters)
                )
            decoded_y1 = int(result.decoded_y1_rows)
            valid_y1 = int(result.valid_y1_rows)
            invalid_y1 = int(result.invalid_y1_rows)
            invalid_target = int(result.invalid_target_rows)
            rejected_y0 = int(result.rejected_y0_rows)
            selected_output = int(result.selected_output_rows)
            if min(decoded_y1, valid_y1, invalid_y1, invalid_target, rejected_y0, selected_output) < 0:
                raise AdapterFailure("validation", "REJECTION_COUNTER_NEGATIVE", repr(counters))
            if result.raw_draw_count != invalid_target + rejected_y0 + decoded_y1:
                raise AdapterFailure(
                    "validation", "RAW_DRAW_CONSERVATION_FAILED", repr(counters)
                )
            if decoded_y1 != valid_y1 + invalid_y1:
                raise AdapterFailure(
                    "validation", "Y1_CONSERVATION_FAILED", repr(counters)
                )
            if selected_output != n or not selected_output <= valid_y1:
                raise AdapterFailure(
                    "validation", "SELECTED_OUTPUT_INVARIANT_FAILED", repr(counters)
                )
            if accepted != valid_y1:
                raise AdapterFailure(
                    "validation", "ACCEPTED_COUNT_ALIAS_MISMATCH", repr(counters)
                )
            expected_y1_rate = decoded_y1 / result.raw_draw_count
            expected_valid_given_y1 = valid_y1 / decoded_y1 if decoded_y1 else None
            expected_overall = valid_y1 / result.raw_draw_count
            if result.y1_rate is None or not np.isclose(
                result.y1_rate, expected_y1_rate, rtol=1e-10, atol=1e-12
            ):
                raise AdapterFailure("validation", "Y1_RATE_INVALID", repr(counters))
            if expected_valid_given_y1 is None:
                if result.valid_given_y1_rate is not None:
                    raise AdapterFailure(
                        "validation", "VALID_GIVEN_Y1_RATE_INVALID", repr(counters)
                    )
            elif result.valid_given_y1_rate is None or not np.isclose(
                result.valid_given_y1_rate, expected_valid_given_y1, rtol=1e-10, atol=1e-12
            ):
                raise AdapterFailure(
                    "validation", "VALID_GIVEN_Y1_RATE_INVALID", repr(counters)
                )
            if result.overall_acceptance_rate is None or not np.isclose(
                result.overall_acceptance_rate, expected_overall, rtol=1e-10, atol=1e-12
            ):
                raise AdapterFailure(
                    "validation", "OVERALL_ACCEPTANCE_RATE_INVALID", repr(counters)
                )
            if result.minority_acceptance_rate != result.overall_acceptance_rate:
                raise AdapterFailure(
                    "validation", "ACCEPTANCE_RATE_ALIAS_MISMATCH", repr(counters)
                )
            if result.termination_reason != "required_count_reached":
                raise AdapterFailure(
                    "validation", "TERMINATION_REASON_INVALID", str(result.termination_reason)
                )
            if result.raw_draw_cap is None or result.raw_draw_count > result.raw_draw_cap:
                raise AdapterFailure("validation", "RAW_CAP_INVARIANT_FAILED", repr(counters))
            detail_expectations = {
                "total_raw_draws": result.raw_draw_count,
                "invalid_target_rows": invalid_target,
                "rejected_y0_rows": rejected_y0,
                "decoded_y1_rows": decoded_y1,
                "valid_y1_rows": valid_y1,
                "invalid_y1_rows": invalid_y1,
                "selected_output_rows": selected_output,
                "y1_rate": result.y1_rate,
                "valid_given_y1_rate": result.valid_given_y1_rate,
                "overall_acceptance_rate": result.overall_acceptance_rate,
                "termination_reason": result.termination_reason,
            }
            if any(result.details.get(key) != value for key, value in detail_expectations.items()):
                raise AdapterFailure(
                    "validation", "REJECTION_DETAIL_MISMATCH", repr(detail_expectations)
                )
        else:
            if result.minority_acceptance_rate is not None or result.accepted_minority_count is not None:
                raise AdapterFailure(
                    "validation", "SPURIOUS_ACCEPTANCE", "non-rejection sampling has no acceptance rate"
                )
            if result.raw_draw_count is not None and result.raw_draw_count < n:
                raise AdapterFailure(
                    "validation", "RAW_COUNT_INVALID", "direct sampling raw count is below output count"
                )
            direct_only_fields = (
                result.decoded_y1_rows,
                result.valid_y1_rows,
                result.invalid_y1_rows,
                result.invalid_target_rows,
                result.rejected_y0_rows,
                result.y1_rate,
                result.valid_given_y1_rate,
                result.overall_acceptance_rate,
                result.selected_output_rows,
                result.termination_reason,
            )
            if any(value is not None for value in direct_only_fields) or result.raw_draw_cap is not None:
                raise AdapterFailure(
                    "validation", "SPURIOUS_REJECTION_METADATA", "direct sampler reported rejection fields"
                )
        return SampleResult(
            frame=validated,
            sampling_seed=result.sampling_seed,
            raw_draw_count=result.raw_draw_count,
            minority_acceptance_rate=result.minority_acceptance_rate,
            raw_draw_cap=result.raw_draw_cap,
            rejection_used=result.rejection_used,
            accepted_minority_count=result.accepted_minority_count,
            details=result.details,
            decoded_y1_rows=result.decoded_y1_rows,
            valid_y1_rows=result.valid_y1_rows,
            invalid_y1_rows=result.invalid_y1_rows,
            invalid_target_rows=result.invalid_target_rows,
            rejected_y0_rows=result.rejected_y0_rows,
            y1_rate=result.y1_rate,
            valid_given_y1_rate=result.valid_given_y1_rate,
            overall_acceptance_rate=result.overall_acceptance_rate,
            selected_output_rows=result.selected_output_rows,
            termination_reason=result.termination_reason,
        )
