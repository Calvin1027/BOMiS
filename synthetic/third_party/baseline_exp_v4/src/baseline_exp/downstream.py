from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .contracts import AdapterFailure, DatasetSchema


def _best_mcc_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    # Evaluate the same thresholds and tie rule in O(n log n), rather than
    # constructing and scoring n full prediction vectors (O(n squared)).
    y, probability = np.asarray(y), np.asarray(probability)
    if len(y) == 0 or len(y) != len(probability) or not np.isfinite(probability).all():
        raise AdapterFailure("metric", "INVALID_PROBABILITY", "invalid threshold inputs")
    if not np.isin(y, [0, 1]).all():
        raise AdapterFailure("metric", "INVALID_TARGET", "threshold target must be binary")
    order = np.argsort(probability, kind="stable")
    sorted_p, sorted_y = probability[order], y[order]
    values, first = np.unique(sorted_p, return_index=True)
    prefix_positive = np.concatenate(([0.0], np.cumsum(sorted_y, dtype=float)))
    fn = prefix_positive[first]
    tn = first.astype(float) - fn
    tp = prefix_positive[-1] - fn
    fp = len(y) - prefix_positive[-1] - tn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    interior = np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator > 0)
    candidates = np.concatenate(([-np.inf], values, [np.inf]))
    scores = np.concatenate(([0.0], interior, [0.0]))
    if not np.isfinite(scores).all():
        raise AdapterFailure("metric", "MCC_NONFINITE", "validation MCC is non-finite")
    best = float(scores.max())
    # Deterministic conservative tie break: largest threshold among equal maxima.
    threshold = float(candidates[np.flatnonzero(np.isclose(scores, best))[-1]])
    return threshold, best


def evaluate_downstream(
    outer_train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    synthetic_minority: pd.DataFrame,
    schema: DatasetSchema,
    *,
    classifier_seed: int,
    classifier_spec: dict[str, Any],
) -> dict[str, float | int | str]:
    synthetic = schema.restore_and_validate(synthetic_minority, expected_n=len(synthetic_minority))
    outer_y = schema.binary_target(outer_train)
    training_policy = str(classifier_spec.get("training_policy", "augment_original"))
    if training_policy == "replace_observed_minority_with_synthetic_bank":
        majority = outer_train.loc[outer_y == 0, list(schema.feature_columns)]
        if len(synthetic) != len(majority):
            raise AdapterFailure(
                "configuration",
                "SYNTHETIC_BANK_NOT_BALANCED",
                f"expected {len(majority)} synthetic rows; received {len(synthetic)}",
            )
        augmented_x = pd.concat([majority, synthetic], ignore_index=True)
        augmented_y = np.concatenate(
            [np.zeros(len(majority), dtype=np.int8), np.ones(len(synthetic), dtype=np.int8)]
        )
        n_observed_minority_used = 0
    elif training_policy == "augment_original":
        augmented_x = pd.concat(
            [outer_train.loc[:, list(schema.feature_columns)], synthetic], ignore_index=True
        )
        augmented_y = np.concatenate([outer_y, np.ones(len(synthetic), dtype=np.int8)])
        n_observed_minority_used = int(outer_y.sum())
    else:
        raise AdapterFailure(
            "configuration", "DOWNSTREAM_POLICY_NOT_FROZEN", training_policy
        )
    # sklearn rejects explicitly supplied numerical categories unless sorted.  These
    # are nominal schema domains whose order is frozen by outer-train, so use the
    # schema's injective string tokens consistently at fit and prediction time.
    categorical_columns = list(schema.categorical_columns)
    for column in categorical_columns:
        augmented_x[column] = augmented_x[column].astype(str)
    validation_x = validation.loc[:, list(schema.feature_columns)].copy()
    test_x = test.loc[:, list(schema.feature_columns)].copy()
    for column in categorical_columns:
        validation_x[column] = validation_x[column].astype(str)
        test_x[column] = test_x[column].astype(str)
    if classifier_spec.get("name") != "logistic_regression":
        raise AdapterFailure("configuration", "CLASSIFIER_NOT_FROZEN", "unsupported classifier")
    preprocessing = ColumnTransformer(
        [
            ("numerical", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(schema.numerical_columns)),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(categories=[[str(level) for level in schema.category_domains[c]] for c in schema.categorical_columns], handle_unknown="ignore"))]), categorical_columns),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=float(classifier_spec["C"]),
        max_iter=int(classifier_spec["max_iter"]),
        class_weight=classifier_spec.get("class_weight"),
        solver=str(classifier_spec.get("solver", "lbfgs")),
        random_state=classifier_seed,
    )
    pipeline = Pipeline([("preprocess", preprocessing), ("classifier", classifier)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(augmented_x, augmented_y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise AdapterFailure(
            "metric", "CLASSIFIER_DID_NOT_CONVERGE", "logistic regression hit its iteration limit"
        )
    validation_probability = pipeline.predict_proba(validation_x)[:, 1]
    test_probability = pipeline.predict_proba(test_x)[:, 1]
    for role, probability in (("validation", validation_probability), ("test", test_probability)):
        if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
            raise AdapterFailure("metric", "INVALID_PROBABILITY", role)
    validation_y, test_y = schema.binary_target(validation), schema.binary_target(test)
    threshold, validation_mcc = _best_mcc_threshold(validation_y, validation_probability)
    return {
        "classifier": "logistic_regression",
        "classifier_seed": int(classifier_seed),
        "training_policy": training_policy,
        "n_observed_minority_used": n_observed_minority_used,
        "mcc_threshold_selected_on": "outer_validation",
        "mcc_threshold": threshold,
        "validation_mcc": validation_mcc,
        "test_mcc": _finite_mcc(test_y, test_probability >= threshold),
        "test_pr_auc": _finite_pr_auc(test_y, test_probability),
        "n_augmented_train": len(augmented_x),
    }


def _finite_mcc(y: np.ndarray, predicted: np.ndarray) -> float:
    value = float(matthews_corrcoef(y, predicted))
    if not np.isfinite(value):
        raise AdapterFailure("metric", "MCC_NONFINITE", "test MCC is non-finite")
    return value


def _finite_pr_auc(y: np.ndarray, probability: np.ndarray) -> float:
    value = float(average_precision_score(y, probability))
    if not np.isfinite(value):
        raise AdapterFailure("metric", "PR_AUC_NONFINITE", "test PR-AUC is non-finite")
    return value
