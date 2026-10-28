from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import AdapterFailure, DatasetSchema, FitResult, GeneratorAdapter, SampleResult


class RandomOverSamplerAdapter(GeneratorAdapter):
    name = "random_over_sampler"

    def __init__(self) -> None:
        super().__init__()
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
        self.schema = schema
        self._train = train.loc[:, [*schema.feature_columns, schema.target]].copy()
        n1 = int(schema.binary_target(self._train).sum())
        if n1 == 0:
            raise AdapterFailure("fit", "NO_MINORITY", "outer-train contains no minority row")
        self.fit_result = FitResult(
            model=self.name,
            model_seed=model_seed,
            n_outer_train=len(train),
            n_minority_train=n1,
            sampling_mode="official_resampler",
            details={
                "algorithm": "imblearn.over_sampling.RandomOverSampler",
                "shrinkage": None,
                "loss_unavailable_reason": "non-iterative resampler has no training loss",
                "checkpoint_unavailable_reason": "nonparametric resampler has no fitted checkpoint",
            },
            training_access="full_outer_train",
            actual_training_population="minority_rows_for_resampling",
            cross_class_usage="none",
            minority_sampling_semantics="bootstrap",
        )
        return self.fit_result

    def _sample_minority(self, n: int, *, sampling_seed: int) -> SampleResult:
        from imblearn.over_sampling import RandomOverSampler

        assert self._train is not None and self.schema is not None
        features = self._train.loc[:, list(self.schema.feature_columns)]
        labels = self.schema.binary_target(self._train)
        n1 = int(labels.sum())
        sampler = RandomOverSampler(
            sampling_strategy={1: n1 + n}, random_state=sampling_seed, shrinkage=None
        )
        resampled_x, resampled_y = sampler.fit_resample(features, labels)
        result = pd.DataFrame(resampled_x, columns=features.columns)
        resampled_y = np.asarray(resampled_y, dtype=np.int8)
        if len(result) != len(features) + n:
            raise AdapterFailure("sample", "ROW_COUNT_MISMATCH", "unexpected resampler size")
        if not result.iloc[: len(features)].reset_index(drop=True).equals(
            features.reset_index(drop=True)
        ) or not np.array_equal(resampled_y[: len(features)], labels):
            raise AdapterFailure("sample", "PREFIX_MUTATED", "resampler changed original rows")
        appended = result.iloc[len(features) :].copy().reset_index(drop=True)
        if not np.all(resampled_y[len(features) :] == 1):
            raise AdapterFailure("sample", "NONMINORITY_APPEND", "resampler appended another class")
        return SampleResult(
            frame=appended,
            sampling_seed=sampling_seed,
            raw_draw_count=n,
            minority_acceptance_rate=None,
            raw_draw_cap=None,
            rejection_used=False,
            details={
                "sampling_mode": "minority_row_bootstrap",
                "n_unique_rows": int(appended.drop_duplicates().shape[0]),
            },
        )


class IndependentMarginalAdapter(GeneratorAdapter):
    name = "independent_marginal"

    def __init__(self) -> None:
        super().__init__()
        self._minority: pd.DataFrame | None = None

    def fit(
        self,
        train: pd.DataFrame,
        schema: DatasetSchema,
        *,
        model_seed: int,
        work_dir: Path,
    ) -> FitResult:
        del work_dir
        mask = schema.binary_target(train).astype(bool)
        self._minority = train.loc[mask, list(schema.feature_columns)].copy().reset_index(drop=True)
        if self._minority.empty:
            raise AdapterFailure("fit", "NO_MINORITY", "outer-train contains no minority row")
        self.schema = schema
        self.fit_result = FitResult(
            model=self.name,
            model_seed=model_seed,
            n_outer_train=len(train),
            n_minority_train=len(self._minority),
            sampling_mode="independent_empirical_marginals",
            details={
                "algorithm": "custom negative control",
                "public_fit_input": "full_outer_train_X_y",
                "conditional_estimator": "product_j empirical P(X_j | Y=1)",
                "loss_unavailable_reason": "empirical estimator has no training loss",
                "checkpoint_unavailable_reason": "empirical estimator has no fitted checkpoint",
            },
            training_access="full_outer_train",
            actual_training_population="minority_conditional_marginals",
            cross_class_usage="none",
            minority_sampling_semantics="direct_conditional",
        )
        return self.fit_result

    def _sample_minority(self, n: int, *, sampling_seed: int) -> SampleResult:
        assert self._minority is not None and self.schema is not None
        rng = np.random.default_rng(sampling_seed)
        sampled = {
            column: self._minority[column].to_numpy()[
                rng.integers(0, len(self._minority), size=n)
            ]
            for column in self.schema.feature_columns
        }
        return SampleResult(
            frame=pd.DataFrame(sampled).loc[:, list(self.schema.feature_columns)],
            sampling_seed=sampling_seed,
            raw_draw_count=n,
            minority_acceptance_rate=None,
            raw_draw_cap=None,
            rejection_used=False,
            details={"sampling_mode": "independent_column_bootstrap"},
        )
