from __future__ import annotations

from itertools import combinations
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors

from .contracts import AdapterFailure, DatasetSchema


def _finite(value: float, name: str) -> float:
    if not np.isfinite(value):
        raise AdapterFailure("metric", "NONFINITE_METRIC", name)
    return float(value)


def marginal_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, schema: DatasetSchema) -> tuple[pd.DataFrame, dict[str, float | None]]:
    rows: list[dict[str, Any]] = []
    for column in schema.numerical_columns:
        value = ks_2samp(real[column].to_numpy(), synthetic[column].to_numpy()).statistic
        rows.append({"column": column, "kind": "numerical", "discrepancy": _finite(value, column)})
    for column in schema.categorical_columns:
        levels = list(pd.unique(pd.concat([real[column], synthetic[column]], ignore_index=True)))
        p = real[column].value_counts(normalize=True).reindex(levels, fill_value=0.0)
        q = synthetic[column].value_counts(normalize=True).reindex(levels, fill_value=0.0)
        value = 0.5 * np.abs(p.to_numpy() - q.to_numpy()).sum()
        rows.append({"column": column, "kind": "categorical", "discrepancy": _finite(value, column)})
    components = pd.DataFrame(rows)
    summary: dict[str, float | None] = {}
    for kind, key in (("numerical", "mean_numerical_ks"), ("categorical", "mean_categorical_tvd")):
        values = components.loc[components["kind"].eq(kind), "discrepancy"]
        summary[key] = float(values.mean()) if len(values) else None
    return components, summary


def _eta_squared(numeric: pd.Series, categorical: pd.Series, levels: list[Any]) -> float:
    x = numeric.to_numpy(dtype=float)
    total = float(np.square(x - x.mean()).sum())
    if total == 0.0:
        return 0.0
    between = 0.0
    for level in levels:
        group = x[categorical.eq(level).to_numpy()]
        if len(group):
            between += len(group) * float((group.mean() - x.mean()) ** 2)
    return between / total


def pairwise_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, schema: DatasetSchema) -> tuple[pd.DataFrame, dict[str, float | None]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(schema.feature_columns, 2):
        left_num, right_num = left in schema.numerical_columns, right in schema.numerical_columns
        if left_num and right_num:
            r = real[[left, right]].corr(method="pearson").iloc[0, 1]
            s = synthetic[[left, right]].corr(method="pearson").iloc[0, 1]
            if pd.isna(r) or pd.isna(s):
                similarity, defined, pair_type = None, False, "numerical_numerical"
            else:
                similarity, defined, pair_type = 1.0 - abs(float(r) - float(s)) / 2.0, True, "numerical_numerical"
        elif not left_num and not right_num:
            l_levels = list(pd.unique(pd.concat([real[left], synthetic[left]], ignore_index=True)))
            r_levels = list(pd.unique(pd.concat([real[right], synthetic[right]], ignore_index=True)))
            index = pd.MultiIndex.from_product([l_levels, r_levels])
            p = real.groupby([left, right], observed=False).size().reindex(index, fill_value=0).to_numpy(dtype=float) / len(real)
            q = synthetic.groupby([left, right], observed=False).size().reindex(index, fill_value=0).to_numpy(dtype=float) / len(synthetic)
            similarity, defined, pair_type = 1.0 - 0.5 * np.abs(p - q).sum(), True, "categorical_categorical"
        else:
            num, cat = (left, right) if left_num else (right, left)
            levels = list(pd.unique(pd.concat([real[cat], synthetic[cat]], ignore_index=True)))
            r, s = _eta_squared(real[num], real[cat], levels), _eta_squared(synthetic[num], synthetic[cat], levels)
            similarity, defined, pair_type = 1.0 - abs(r - s), True, "numerical_categorical"
        rows.append({"left": left, "right": right, "pair_type": pair_type, "similarity": similarity, "defined": defined})
    components = pd.DataFrame(rows)
    summary: dict[str, float | None] = {}
    for pair_type in ("numerical_numerical", "numerical_categorical", "categorical_categorical"):
        subset = components.loc[components["pair_type"].eq(pair_type) & components["defined"], "similarity"] if len(components) else pd.Series(dtype=float)
        summary[f"mean_pair_similarity_{pair_type}"] = float(subset.mean()) if len(subset) else None
        summary[f"defined_pairs_{pair_type}"] = int(len(subset))
    return components, summary


def support_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, schema: DatasetSchema) -> dict[str, float | int | None]:
    per_column: list[float] = []
    for column in schema.categorical_columns:
        real_levels = set(real[column].tolist())
        generated_levels = set(synthetic[column].tolist())
        per_column.append(len(real_levels & generated_levels) / len(real_levels))
    if schema.categorical_columns:
        columns = list(schema.categorical_columns)
        real_tuples = set(map(tuple, real[columns].itertuples(index=False, name=None)))
        synth_tuples = list(map(tuple, synthetic[columns].itertuples(index=False, name=None)))
        tuple_validity = float(np.mean([item in real_tuples for item in synth_tuples]))
    else:
        tuple_validity = None
    return {
        "mean_category_coverage": float(np.mean(per_column)) if per_column else None,
        "categorical_tuple_validity": tuple_validity,
        "categorical_column_count": len(schema.categorical_columns),
    }


def alpha_precision_beta_recall(real: pd.DataFrame, synthetic: pd.DataFrame, schema: DatasetSchema, *, metric_seed: int) -> dict[str, float | int]:
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

    combined = pd.concat([real, synthetic], ignore_index=True)
    numerical = combined.loc[:, list(schema.numerical_columns)].to_numpy(dtype=float)
    if schema.categorical_columns:
        # The generator is restricted to outer-train categories, but an untouched
        # test-minority reference can legitimately contain a level not observed in
        # outer-train.  Give those real-reference levels an evaluation-only one-hot
        # coordinate.  Never learn a category from synthetic output.
        categories: list[list[str]] = []
        for column in schema.categorical_columns:
            levels = [str(level) for level in schema.category_domains[column]]
            known = set(levels)
            levels.extend(
                level
                for level in pd.unique(real[column].astype(str)).tolist()
                if level not in known
            )
            categories.append(levels)
        encoder = OneHotEncoder(categories=categories, sparse_output=False, handle_unknown="error")
        categorical = encoder.fit_transform(
            combined.loc[:, list(schema.categorical_columns)].astype(str)
        )
    else:
        categorical = np.empty((len(combined), 0), dtype=float)
    matrix = np.column_stack([numerical, categorical])
    real_x, synth_x = matrix[: len(real)], matrix[len(real) :]
    scaler = MinMaxScaler().fit(real_x)
    real_x, synth_x = scaler.transform(real_x), scaler.transform(synth_x)
    n = min(len(real_x), len(synth_x))
    if n < 2:
        raise AdapterFailure("metric", "ALPHA_PRECISION_TOO_SMALL", str(n))
    rng = np.random.default_rng(metric_seed)
    real_x = real_x[rng.choice(len(real_x), size=n, replace=False)]
    synth_x = synth_x[rng.choice(len(synth_x), size=n, replace=False)]
    center = np.mean(real_x, axis=0)
    alphas = np.linspace(0.0, 1.0, 30)
    radii = np.quantile(np.linalg.norm(real_x - center, axis=1), alphas)
    synth_center = np.mean(synth_x, axis=0)
    synth_to_center = np.linalg.norm(synth_x - center, axis=1)
    metric_jobs = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    real_to_real = NearestNeighbors(n_neighbors=2, n_jobs=metric_jobs, p=2).fit(real_x).kneighbors(real_x, return_distance=True)[0][:, 1]
    real_to_synth_distance, real_to_synth_index = NearestNeighbors(n_neighbors=1, n_jobs=metric_jobs, p=2).fit(synth_x).kneighbors(real_x)
    real_to_synth_distance, real_to_synth_index = real_to_synth_distance[:, 0], real_to_synth_index[:, 0]
    closest = synth_x[real_to_synth_index]
    closest_radius = np.linalg.norm(closest - synth_center, axis=1)
    closest_quantiles = np.quantile(closest_radius, alphas)
    precision_curve = np.asarray([np.mean(synth_to_center <= radius) for radius in radii])
    coverage_curve = np.asarray([
        np.mean((real_to_synth_distance <= real_to_real) & (closest_radius <= radius))
        for radius in closest_quantiles
    ])
    denominator = float(np.sum(alphas))
    alpha_precision = 1.0 - float(np.sum(np.abs(alphas - precision_curve))) / denominator
    beta_recall = 1.0 - float(np.sum(np.abs(alphas - coverage_curve))) / denominator
    return {
        "alpha_precision": _finite(alpha_precision, "alpha_precision"),
        "beta_recall": _finite(beta_recall, "beta_recall"),
        "alpha_precision_n": int(n),
        "alpha_precision_metric_seed": int(metric_seed),
    }


def evaluate_fidelity(real: pd.DataFrame, synthetic: pd.DataFrame, schema: DatasetSchema, *, metric_seed: int) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    if tuple(real.columns) != schema.feature_columns or real.isna().any().any():
        raise AdapterFailure("metric", "REFERENCE_SCHEMA_INVALID", "test-minority reference is invalid")
    for column in schema.numerical_columns:
        if not np.isfinite(pd.to_numeric(real[column], errors="coerce").to_numpy(dtype=float)).all():
            raise AdapterFailure("metric", "REFERENCE_NONFINITE", column)
    synthetic = schema.restore_and_validate(synthetic, expected_n=len(synthetic))
    marginal, marginal_summary = marginal_metrics(real, synthetic, schema)
    pairwise, pairwise_summary = pairwise_metrics(real, synthetic, schema)
    summary = {**marginal_summary, **pairwise_summary, **support_metrics(real, synthetic, schema), **alpha_precision_beta_recall(real, synthetic, schema, metric_seed=metric_seed)}
    return summary, {"marginal_components": marginal, "pairwise_components": pairwise}
