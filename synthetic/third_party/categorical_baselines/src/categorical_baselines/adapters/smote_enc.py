from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..contracts import AdapterFailure, DatasetSchema, FitResult, GeneratorAdapter, SampleResult
from ..data import require_categorical


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
        require_categorical(schema)
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
                    "fit",
                    "DEGENERATE_CONTINUOUS_SCALE",
                    "SMOTE-ENC continuous scale is nonpositive",
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
        raw_indices = neighbours.kneighbors(neighbour_space, return_distance=False)
        # Equal encoded distances (and duplicate rows) need not put self first.
        indices = np.asarray(
            [row[row != i][: self.k_neighbors] for i, row in enumerate(raw_indices)]
        )
        self.schema = schema
        self._minority_num_scaled = minority_num
        self._minority_cat = (
            train.loc[minority, list(schema.categorical_columns)].astype(str).to_numpy()
        )
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
