from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse

from ..contracts import AdapterFailure, DatasetSchema, FitResult, GeneratorAdapter, SampleResult


def _validate_original_prefix(
    original_x: pd.DataFrame,
    original_y: np.ndarray,
    resampled_x: Any,
    resampled_y: Any,
    *,
    appended_n: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(resampled_x, columns=original_x.columns)
    labels = np.asarray(resampled_y, dtype=np.int8)
    if len(frame) != len(original_x) + appended_n or len(labels) != len(frame):
        raise AdapterFailure("sample", "ROW_COUNT_MISMATCH", "unexpected resampler size")
    prefix = frame.iloc[: len(original_x)].reset_index(drop=True)
    original = original_x.reset_index(drop=True)
    for column in original.columns:
        left, right = prefix[column], original[column]
        try:
            equal = np.allclose(left.astype(float), right.astype(float), rtol=0.0, atol=0.0)
        except (TypeError, ValueError):
            equal = np.array_equal(left.astype(str), right.astype(str))
        if not equal:
            raise AdapterFailure("sample", "PREFIX_MUTATED", f"resampler changed {column!r}")
    if not np.array_equal(labels[: len(original)], original_y):
        raise AdapterFailure("sample", "PREFIX_LABEL_MUTATED", "resampler changed labels")
    if not np.all(labels[len(original) :] == 1):
        raise AdapterFailure("sample", "NONMINORITY_APPEND", "resampler appended another class")
    return frame.iloc[len(original) :].copy().reset_index(drop=True)


class SMOTENCAdapter(GeneratorAdapter):
    name = "smotenc"

    def __init__(self, *, k_neighbors: int = 5) -> None:
        super().__init__()
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be positive")
        self.k_neighbors = int(k_neighbors)
        self._train: pd.DataFrame | None = None

    def fit(
        self,
        train: pd.DataFrame,
        schema: DatasetSchema,
        *,
        model_seed: int,
        work_dir: Path,
    ) -> FitResult:
        del work_dir
        if not schema.categorical_columns or not schema.numerical_columns:
            raise AdapterFailure(
                "fit", "INAPPLICABLE_SCHEMA", "SMOTE-NC requires mixed numerical/categorical data"
            )
        y = schema.binary_target(train)
        n1 = int(y.sum())
        if n1 <= self.k_neighbors:
            raise AdapterFailure("fit", "TOO_FEW_MINORITY", "insufficient minority neighbours")
        self.schema = schema
        self._train = train.loc[:, [*schema.feature_columns, schema.target]].copy()
        self.fit_result = FitResult(
            model=self.name,
            model_seed=model_seed,
            n_outer_train=len(train),
            n_minority_train=n1,
            sampling_mode="official_resampler",
            details={
                "algorithm": "imblearn.over_sampling.SMOTENC",
                "k_neighbors": self.k_neighbors,
                "loss_unavailable_reason": "non-iterative resampler has no training loss",
                "checkpoint_unavailable_reason": "sampling is recomputed from the stored outer-train table",
            },
            training_access="full_outer_train",
            actual_training_population="full_labeled_outer_train",
            cross_class_usage="minority_neighbourhood",
            minority_sampling_semantics="direct_oversampling",
        )
        return self.fit_result

    def _sample_minority(self, n: int, *, sampling_seed: int) -> SampleResult:
        from imblearn.over_sampling import SMOTENC

        assert self._train is not None and self.schema is not None
        features = self._train.loc[:, list(self.schema.feature_columns)]
        y = self.schema.binary_target(self._train)
        n1 = int(y.sum())
        categorical = [features.columns.get_loc(name) for name in self.schema.categorical_columns]
        sampler = SMOTENC(
            categorical_features=categorical,
            sampling_strategy={1: n1 + n},
            random_state=sampling_seed,
            k_neighbors=self.k_neighbors,
        )
        x_resampled, y_resampled = sampler.fit_resample(features, y)
        appended = _validate_original_prefix(
            features, y, x_resampled, y_resampled, appended_n=n
        )
        return SampleResult(
            frame=appended,
            sampling_seed=sampling_seed,
            raw_draw_count=int(appended.shape[0]),
            minority_acceptance_rate=None,
            raw_draw_cap=None,
            rejection_used=False,
            details={"sampling_mode": "native_mixed_smote", "k_neighbors": self.k_neighbors},
        )


class _MixedTypeOHECodec:
    """Outer-train-fitted representation for numeric-only adaptive samplers."""

    def __init__(self, schema: DatasetSchema) -> None:
        self.schema = schema
        self.scaler: StandardScaler | None = None
        self.encoder: OneHotEncoder | None = None
        self.category_tokens: list[tuple[str, ...]] = []
        self.encoded_dimension = 0

    def _categorical_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Map semantic categories to injective string tokens before sklearn OHE.

        sklearn requires explicitly supplied numerical categories to be sorted.  The
        schema order is intentionally the outer-train encounter order, so coercing to
        the already-validated injective string representation preserves that order
        without introducing an ordinal interpretation.
        """
        return frame.loc[:, list(self.schema.categorical_columns)].astype(str)

    def fit(self, frame: pd.DataFrame) -> _MixedTypeOHECodec:
        dimension = 0
        if self.schema.numerical_columns:
            numerical = frame.loc[:, list(self.schema.numerical_columns)].to_numpy(dtype=float)
            if not np.isfinite(numerical).all():
                raise AdapterFailure("fit", "NONFINITE_TRAIN", "invalid numerical OHE input")
            self.scaler = StandardScaler().fit(numerical)
            dimension += len(self.schema.numerical_columns)
        if self.schema.categorical_columns:
            self.category_tokens = [
                tuple(str(level) for level in self.schema.category_domains[column])
                for column in self.schema.categorical_columns
            ]
            self.encoder = OneHotEncoder(
                categories=[list(levels) for levels in self.category_tokens],
                handle_unknown="error",
                sparse_output=True,
                dtype=np.float64,
            ).fit(self._categorical_frame(frame))
            dimension += sum(len(levels) for levels in self.encoder.categories_)
        if dimension < 1:
            raise AdapterFailure("fit", "EMPTY_REPRESENTATION", "OHE sampler has no features")
        self.encoded_dimension = int(dimension)
        return self

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        blocks: list[sparse.csr_matrix] = []
        if self.schema.numerical_columns:
            assert self.scaler is not None
            numerical = frame.loc[:, list(self.schema.numerical_columns)].to_numpy(dtype=float)
            blocks.append(sparse.csr_matrix(self.scaler.transform(numerical)))
        if self.schema.categorical_columns:
            assert self.encoder is not None
            blocks.append(
                self.encoder.transform(
                    self._categorical_frame(frame)
                ).tocsr()
            )
        encoded = sparse.hstack(blocks, format="csr", dtype=np.float64)
        if encoded.shape[1] != self.encoded_dimension or not np.isfinite(encoded.data).all():
            raise AdapterFailure("fit", "ENCODING_FAILED", "invalid OHE representation")
        return encoded

    def inverse_transform(self, encoded: Any) -> pd.DataFrame:
        matrix = encoded.toarray() if sparse.issparse(encoded) else np.asarray(encoded)
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.encoded_dimension:
            raise AdapterFailure("sample", "RAW_SCHEMA_MISMATCH", "invalid encoded width")
        if not np.isfinite(matrix).all():
            raise AdapterFailure("sample", "NONFINITE_OUTPUT", "non-finite encoded sample")
        values: dict[str, Any] = {}
        offset = 0
        n_num = len(self.schema.numerical_columns)
        if n_num:
            assert self.scaler is not None
            numerical = self.scaler.inverse_transform(matrix[:, :n_num])
            for index, column in enumerate(self.schema.numerical_columns):
                values[column] = numerical[:, index]
            offset = n_num
        if self.schema.categorical_columns:
            assert self.encoder is not None
            for column, levels in zip(
                self.schema.categorical_columns, self.encoder.categories_, strict=True
            ):
                width = len(levels)
                block = matrix[:, offset : offset + width]
                # SMOTE interpolation produces soft one-hot blocks. Argmax is the frozen,
                # deterministic projection back to one valid outer-train category.
                values[column] = np.asarray(levels, dtype=object)[np.argmax(block, axis=1)]
                offset += width
        if offset != matrix.shape[1]:
            raise AdapterFailure("sample", "RAW_SCHEMA_MISMATCH", "unconsumed OHE columns")
        return pd.DataFrame(values).loc[:, list(self.schema.feature_columns)]


class _OHEAdaptiveSamplerAdapter(GeneratorAdapter):
    algorithm: str

    def __init__(self) -> None:
        super().__init__()
        self._train: pd.DataFrame | None = None
        self._codec: _MixedTypeOHECodec | None = None

    def _minority_neighbour_requirement(self) -> int:
        raise NotImplementedError

    def _sampler(self, final_minority_count: int, seed: int) -> Any:
        raise NotImplementedError

    def _requested_increment(self, n: int, minority_count: int) -> int:
        del minority_count
        return n

    def _parameters(self) -> dict[str, Any]:
        raise NotImplementedError

    def fit(
        self, train: pd.DataFrame, schema: DatasetSchema, *, model_seed: int, work_dir: Path
    ) -> FitResult:
        del work_dir
        if not schema.numerical_columns or not schema.categorical_columns:
            raise AdapterFailure("fit", "INAPPLICABLE_SCHEMA", "OHE adapter requires mixed data")
        y = schema.binary_target(train)
        n1 = int(y.sum())
        if n1 <= self._minority_neighbour_requirement():
            raise AdapterFailure("fit", "TOO_FEW_MINORITY", "insufficient minority neighbours")
        stored = train.loc[:, [*schema.feature_columns, schema.target]].copy()
        codec = _MixedTypeOHECodec(schema).fit(stored.loc[:, list(schema.feature_columns)])
        self.schema, self._train, self._codec = schema, stored, codec
        self.fit_result = FitResult(
            model=self.name,
            model_seed=model_seed,
            n_outer_train=len(stored),
            n_minority_train=n1,
            sampling_mode="official_numeric_sampler_with_frozen_ohe_projection",
            details={
                "algorithm": self.algorithm,
                "adaptation": "outer-train StandardScaler plus categorical one-hot",
                "categorical_decoder": "per-column argmax with outer-train category order",
                "encoded_dimension": codec.encoded_dimension,
                "parameters": self._parameters(),
                "loss_unavailable_reason": "non-iterative resampler has no training loss",
                "checkpoint_unavailable_reason": "sampling is recomputed from the stored outer-train table",
            },
            training_access="full_outer_train",
            actual_training_population="full_labeled_outer_train",
            cross_class_usage=(
                "local_boundary" if self.name == "borderline_smote" else "local_density"
            ),
            minority_sampling_semantics="direct_oversampling",
        )
        return self.fit_result

    def _sample_minority(self, n: int, *, sampling_seed: int) -> SampleResult:
        assert self.schema is not None and self._train is not None and self._codec is not None
        features = self._train.loc[:, list(self.schema.feature_columns)]
        y = self.schema.binary_target(self._train)
        n1 = int(y.sum())
        increment = self._requested_increment(n, n1)
        encoded = self._codec.transform(features)
        sampler = self._sampler(n1 + increment, sampling_seed)
        resampled, labels = sampler.fit_resample(encoded, y)
        labels = np.asarray(labels, dtype=np.int8)
        original_n = len(features)
        if len(labels) != resampled.shape[0] or resampled.shape[0] < original_n:
            raise AdapterFailure("sample", "ROW_COUNT_MISMATCH", "inconsistent resampler output")
        prefix = resampled[:original_n]
        difference = prefix - encoded
        if sparse.issparse(difference):
            if difference.nnz and np.max(np.abs(difference.data)) > 1e-12:
                raise AdapterFailure("sample", "PREFIX_MUTATED", "encoded prefix changed")
        elif not np.allclose(prefix, encoded.toarray(), rtol=0.0, atol=1e-12):
            raise AdapterFailure("sample", "PREFIX_MUTATED", "encoded prefix changed")
        if not np.array_equal(labels[:original_n], y):
            raise AdapterFailure("sample", "PREFIX_LABEL_MUTATED", "label prefix changed")
        appended = resampled[original_n:]
        appended_labels = labels[original_n:]
        if appended.shape[0] < n:
            raise AdapterFailure(
                "sample", "ROW_COUNT_MISMATCH", f"generated {appended.shape[0]}; required {n}"
            )
        if not np.all(appended_labels == 1):
            raise AdapterFailure("sample", "NONMINORITY_APPEND", "nonminority row appended")
        rng = np.random.default_rng(sampling_seed)
        selected = rng.choice(appended.shape[0], size=n, replace=False)
        frame = self._codec.inverse_transform(appended[selected])
        return SampleResult(
            frame=frame,
            sampling_seed=sampling_seed,
            raw_draw_count=int(appended.shape[0]),
            minority_acceptance_rate=None,
            raw_draw_cap=None,
            rejection_used=False,
            details={
                "sampling_mode": "numeric_sampler_ohe_argmax_projection",
                "generated_before_thinning": int(appended.shape[0]),
                "selected_output_rate": n / int(appended.shape[0]),
                "exact_size_policy": "deterministic seeded thinning without replacement",
            },
        )


class BorderlineSMOTEOHEAdapter(_OHEAdaptiveSamplerAdapter):
    name = "borderline_smote"
    algorithm = "imblearn.over_sampling.BorderlineSMOTE"

    def __init__(self, *, k_neighbors: int = 5, m_neighbors: int = 10) -> None:
        if k_neighbors < 1 or m_neighbors < 2:
            raise ValueError("invalid Borderline-SMOTE neighbour parameters")
        super().__init__()
        self.k_neighbors, self.m_neighbors = int(k_neighbors), int(m_neighbors)

    def _minority_neighbour_requirement(self) -> int:
        return self.k_neighbors

    def _sampler(self, final_minority_count: int, seed: int) -> Any:
        from imblearn.over_sampling import BorderlineSMOTE

        return BorderlineSMOTE(
            sampling_strategy={1: final_minority_count}, random_state=seed,
            k_neighbors=self.k_neighbors, m_neighbors=self.m_neighbors, kind="borderline-1",
        )

    def _parameters(self) -> dict[str, Any]:
        return {"kind": "borderline-1", "k_neighbors": self.k_neighbors, "m_neighbors": self.m_neighbors}


class ADASYNOHEAdapter(_OHEAdaptiveSamplerAdapter):
    name = "adasyn"
    algorithm = "imblearn.over_sampling.ADASYN"

    def __init__(self, *, n_neighbors: int = 5) -> None:
        if n_neighbors < 1:
            raise ValueError("ADASYN n_neighbors must be positive")
        super().__init__()
        self.n_neighbors = int(n_neighbors)

    def _minority_neighbour_requirement(self) -> int:
        return self.n_neighbors

    def _requested_increment(self, n: int, minority_count: int) -> int:
        # imbalanced-learn's per-observation integer allocation may undershoot.
        # A prespecified surplus followed by seeded thinning enforces the common M_d contract.
        return n + minority_count

    def _sampler(self, final_minority_count: int, seed: int) -> Any:
        from imblearn.over_sampling import ADASYN

        return ADASYN(
            sampling_strategy={1: final_minority_count}, random_state=seed,
            n_neighbors=self.n_neighbors,
        )

    def _parameters(self) -> dict[str, Any]:
        return {"n_neighbors": self.n_neighbors, "surplus_rule": "requested_n_plus_n_minority"}


class SMOTEENCAdapter(GeneratorAdapter):
    """Paper-faithful generalized SMOTE-ENC implementation."""

    name = "smote_enc"

    def __init__(self, *, k_neighbors: int = 5) -> None:
        super().__init__()
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be positive")
        self.k_neighbors = int(k_neighbors)
        self._minority_num_scaled: np.ndarray | None = None
        self._minority_cat: np.ndarray | None = None
        self._neighbours: np.ndarray | None = None
        self._scaler: StandardScaler | None = None
        self._category_orders: list[tuple[str, ...]] = []
        self._association_maps: list[dict[str, float]] = []
        self._categorical_scale = 1.0

    @staticmethod
    def _mode_with_order(values: np.ndarray, order: tuple[str, ...]) -> str:
        counts = Counter(str(item) for item in values.tolist())
        maximum = max(counts.values())
        return next(level for level in order if counts.get(level, 0) == maximum)

    def fit(
        self,
        train: pd.DataFrame,
        schema: DatasetSchema,
        *,
        model_seed: int,
        work_dir: Path,
    ) -> FitResult:
        del work_dir
        if not schema.categorical_columns:
            raise AdapterFailure(
                "fit", "INAPPLICABLE_SCHEMA", "SMOTE-ENC requires at least one nominal feature"
            )
        y = schema.binary_target(train)
        n1 = int(y.sum())
        if n1 <= self.k_neighbors:
            raise AdapterFailure("fit", "TOO_FEW_MINORITY", "insufficient minority neighbours")
        prevalence = float(y.mean())
        association_maps: list[dict[str, float]] = []
        category_orders: list[tuple[str, ...]] = []
        encoded_columns: list[np.ndarray] = []
        for column in schema.categorical_columns:
            values = train[column].astype(str)
            order = tuple(str(level) for level in schema.category_domains[column])
            mapping: dict[str, float] = {}
            for level in order:
                level_mask = values.eq(level).to_numpy()
                observed = int(np.count_nonzero(level_mask & (y == 1)))
                expected = int(np.count_nonzero(level_mask)) * prevalence
                if expected <= 0.0:
                    raise AdapterFailure(
                        "fit", "ZERO_EXPECTED_COUNT", f"undefined association for {column}={level}"
                    )
                mapping[level] = (observed - expected) / expected
            encoded = values.map(mapping)
            if encoded.isna().any():
                raise AdapterFailure("fit", "ENCODING_FAILED", f"failed to encode {column!r}")
            association_maps.append(mapping)
            category_orders.append(order)
            encoded_columns.append(encoded.to_numpy(dtype=float))

        minority = y == 1
        if schema.numerical_columns:
            numerical = train.loc[:, list(schema.numerical_columns)].to_numpy(dtype=float)
            scaler = StandardScaler().fit(numerical)
            numerical_scaled = scaler.transform(numerical)
            minority_num = numerical_scaled[minority]
            continuous_sd = minority_num.std(axis=0, ddof=0)
            categorical_scale = float(np.median(continuous_sd))
            if not np.isfinite(categorical_scale) or categorical_scale <= 0.0:
                raise AdapterFailure(
                    "fit", "DEGENERATE_CONTINUOUS_SCALE", "SMOTE-ENC continuous scale is nonpositive"
                )
            self._scaler = scaler
        else:
            minority_num = np.empty((n1, 0), dtype=float)
            categorical_scale = 1.0
            self._scaler = None
        encoded_cat = np.column_stack(encoded_columns) * categorical_scale
        minority_cat_encoded = encoded_cat[minority]
        neighbour_space = np.hstack([minority_num, minority_cat_encoded])
        neighbours = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        neighbours.fit(neighbour_space)
        indices = neighbours.kneighbors(neighbour_space, return_distance=False)[:, 1:]
        self.schema = schema
        self._minority_num_scaled = minority_num
        self._minority_cat = train.loc[minority, list(schema.categorical_columns)].astype(str).to_numpy()
        self._neighbours = indices
        self._category_orders = category_orders
        self._association_maps = association_maps
        self._categorical_scale = categorical_scale
        self.fit_result = FitResult(
            model=self.name,
            model_seed=model_seed,
            n_outer_train=len(train),
            n_minority_train=n1,
            sampling_mode="paper_faithful_resampler",
            details={
                "algorithm": "Mukherjee-Khushi SMOTE-ENC (2021)",
                "reference_commit": "c5b28be552d393bfb73647ea8c3d7362df690f14",
                "k_neighbors": self.k_neighbors,
                "minority_prevalence": prevalence,
                "categorical_scale": categorical_scale,
                "association_maps": association_maps,
                "tie_policy": "train-domain order",
                "loss_unavailable_reason": "non-iterative resampler has no training loss",
                "checkpoint_unavailable_reason": "sampling is recomputed from fitted in-memory transforms",
            },
            training_access="full_outer_train",
            actual_training_population="full_labeled_outer_train",
            cross_class_usage="class_association_and_minority_neighbourhood",
            minority_sampling_semantics="direct_oversampling",
        )
        return self.fit_result

    def _sample_minority(self, n: int, *, sampling_seed: int) -> SampleResult:
        assert self.schema is not None
        assert self._minority_num_scaled is not None
        assert self._minority_cat is not None
        assert self._neighbours is not None
        rng = np.random.default_rng(sampling_seed)
        n_minority = len(self._minority_num_scaled)
        base_rows = rng.integers(0, n_minority, size=n)
        neighbour_slots = rng.integers(0, self.k_neighbors, size=n)
        neighbour_rows = self._neighbours[base_rows, neighbour_slots]
        steps = rng.uniform(0.0, 1.0, size=(n, 1))
        base_num = self._minority_num_scaled[base_rows]
        neighbour_num = self._minority_num_scaled[neighbour_rows]
        generated_num_scaled = base_num + steps * (neighbour_num - base_num)
        generated: dict[str, Any] = {}
        if self.schema.numerical_columns:
            assert self._scaler is not None
            numerical = self._scaler.inverse_transform(generated_num_scaled)
            for index, column in enumerate(self.schema.numerical_columns):
                generated[column] = numerical[:, index]
        for index, column in enumerate(self.schema.categorical_columns):
            modes = [
                self._mode_with_order(
                    self._minority_cat[self._neighbours[row], index],
                    self._category_orders[index],
                )
                for row in base_rows
            ]
            generated[column] = np.asarray(modes, dtype=object)
        frame = pd.DataFrame(generated).loc[:, list(self.schema.feature_columns)]
        return SampleResult(
            frame=frame,
            sampling_seed=sampling_seed,
            raw_draw_count=n,
            minority_acceptance_rate=None,
            raw_draw_cap=None,
            rejection_used=False,
            details={
                "sampling_mode": "smote_enc_interpolation",
                "k_neighbors": self.k_neighbors,
                "pair_selection": "uniform base row and uniform neighbor slot",
            },
        )
