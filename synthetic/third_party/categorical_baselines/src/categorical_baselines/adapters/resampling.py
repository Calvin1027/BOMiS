from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ..contracts import AdapterFailure, FitResult, GeneratorAdapter, SampleResult
from ..data import require_categorical


class NominalResampler(GeneratorAdapter):
    """Official SMOTEN or explicitly labelled SMOTE with OHE/argmax projection."""

    def __init__(self, name="smoten", k_neighbors=5):
        super().__init__()
        if name not in {"smoten", "smote"} or k_neighbors < 1:
            raise ValueError("invalid sampler or neighbor count")
        self.name, self.k_neighbors = name, int(k_neighbors)

    def fit(self, train, schema, *, model_seed, work_dir: Path):
        require_categorical(schema)
        self.schema = schema
        self.x = train.loc[:, list(schema.feature_columns)].astype(str).reset_index(drop=True)
        self.y = schema.binary_target(train)
        n1 = int(self.y.sum())
        if n1 <= self.k_neighbors:
            raise AdapterFailure("fit", "TOO_FEW_MINORITY", "k must be smaller than minority n")
        if self.name == "smote":
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="error")
            self.encoded = self.encoder.fit_transform(self.x)
        self.fit_result = FitResult(
            self.name,
            model_seed,
            len(train),
            n1,
            "direct_oversampling",
            details={
                "algorithm": "imblearn.SMOTEN" if self.name == "smoten" else "imblearn.SMOTE",
                "representation": "nominal" if self.name == "smoten" else "one_hot_argmax",
                "k_neighbors": self.k_neighbors,
                "projection_degeneracy": self.name == "smote",
                "loss_unavailable_reason": "non-iterative resampler",
                "checkpoint_unavailable_reason": "resampler stores the training observations",
            },
            training_access="full_outer_train",
            actual_training_population="full_labeled_outer_train",
            cross_class_usage="class_association_and_minority_neighbourhood"
            if self.name == "smoten"
            else "minority_neighbourhood",
            minority_sampling_semantics="direct_oversampling",
        )
        return self.fit_result

    def _sample_minority(self, n, *, sampling_seed):
        from imblearn.over_sampling import SMOTE, SMOTEN

        options = {
            "sampling_strategy": {1: int(self.y.sum()) + n},
            "random_state": sampling_seed,
            "k_neighbors": self.k_neighbors,
        }
        values = self.encoded if self.name == "smote" else self.x
        sampler = SMOTE(**options) if self.name == "smote" else SMOTEN(**options)
        sampled, labels = sampler.fit_resample(values, self.y)
        if not np.array_equal(labels[: len(self.y)], self.y) or not np.all(
            labels[len(self.y) :] == 1
        ):
            raise AdapterFailure("sample", "PREFIX_LABEL_MUTATED", "unexpected resampler labels")
        if not np.array_equal(np.asarray(sampled)[: len(self.y)], np.asarray(values)):
            raise AdapterFailure("sample", "PREFIX_MUTATED", "resampler changed original rows")
        tail = sampled[len(self.y) :]
        if self.name == "smote":
            tail = self.encoder.inverse_transform(tail)
        frame = pd.DataFrame(np.asarray(tail), columns=self.schema.feature_columns)
        return SampleResult(
            frame,
            sampling_seed,
            n,
            None,
            None,
            rejection_used=False,
            details={"representation": self.fit_result.details["representation"]},
        )
