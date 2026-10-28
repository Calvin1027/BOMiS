from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts import AdapterFailure, DatasetSchema, FitResult, GeneratorAdapter, SampleResult
from ._helpers import canonical_joint, new_artifact_path


class GaussianCopulaAdapter(GeneratorAdapter):
    name = "gaussian_copula"

    def __init__(self) -> None:
        super().__init__()
        self._model: Any | None = None

    def fit(
        self,
        train: pd.DataFrame,
        schema: DatasetSchema,
        *,
        model_seed: int,
        work_dir: Path,
    ) -> FitResult:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer

        table = canonical_joint(train, schema)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(table)
        for column in schema.categorical_columns:
            metadata.update_column(column_name=column, sdtype="categorical")
        metadata.update_column(column_name=schema.target, sdtype="categorical")
        model = GaussianCopulaSynthesizer(
            metadata,
            default_distribution="beta",
            enforce_min_max_values=True,
            enforce_rounding=True,
        )
        # SDV currently exposes no public per-fit RNG argument.
        import numpy as np

        np.random.seed(model_seed)
        model.fit(table)
        checkpoint = new_artifact_path(work_dir, "gaussian_copula.pkl")
        model.save(filepath=checkpoint)
        self.schema = schema
        self._model = model
        n1 = int(table[schema.target].sum())
        self.fit_result = FitResult(
            model=self.name,
            model_seed=model_seed,
            n_outer_train=len(table),
            n_minority_train=n1,
            sampling_mode="sdv_exact_condition",
            checkpoint_artifacts=(checkpoint.name,),
            details={
                "algorithm": "sdv.single_table.GaussianCopulaSynthesizer",
                "default_distribution": "beta",
                "fit_seed_mechanism": "numpy global state required by pinned SDV",
                "loss_unavailable_reason": "closed-form copula fit has no iterative loss history",
            },
            training_access="full_outer_train",
            actual_training_population="full_joint",
            cross_class_usage="shared_joint_model",
            minority_sampling_semantics="native_conditioning",
        )
        return self.fit_result

    def _sample_minority(self, n: int, *, sampling_seed: int) -> SampleResult:
        from sdv.sampling import Condition

        assert self._model is not None and self.schema is not None
        if not hasattr(self._model, "_set_random_state"):
            raise AdapterFailure(
                "sample", "RNG_HOOK_MISSING", "pinned SDV random-state hook is unavailable"
            )
        self._model._set_random_state(sampling_seed)
        condition = Condition(num_rows=n, column_values={self.schema.target: 1})
        sampled = self._model.sample_from_conditions([condition])
        if len(sampled) != n or not sampled[self.schema.target].eq(1).all():
            raise AdapterFailure("sample", "CONDITION_FAILED", "SDV did not satisfy Y=1")
        return SampleResult(
            frame=sampled.loc[:, list(self.schema.feature_columns)].copy(),
            sampling_seed=sampling_seed,
            raw_draw_count=None,
            minority_acceptance_rate=None,
            raw_draw_cap=None,
            rejection_used=False,
            details={
                "sampling_mode": "sdv_sample_from_conditions",
                "raw_count_unavailable_reason": "SDV does not expose internal rejection draws",
            },
        )
