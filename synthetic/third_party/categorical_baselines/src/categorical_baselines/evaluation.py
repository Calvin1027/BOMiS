"""Post-generation evaluation; generators never import held-out observations."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import sha256, write_json
from .data import categorical_schema, read_train
from .downstream import evaluate_downstream
from .metrics import alpha_precision_beta_recall, marginal_metrics, pairwise_metrics
from .seeds import semantic_seed
from .worker import audit_bank


def evaluate_cell(task, cell, bank_name, variant, data, output):
    output.mkdir(parents=True, exist_ok=False)
    audit_bank(task, cell, bank_name)
    manifest = json.loads(
        (data / task["dataset"] / f"split-{task['seed']}" / "manifest.json").read_text()
    )
    frames = {}
    for role in ("train", "validation", "test"):
        entry = manifest["files"][role]
        path = data / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"{role} checksum mismatch")
        frames[role] = read_train(path)
    schema = categorical_schema(frames["train"])
    bank = pd.read_csv(
        cell / bank_name / "synthetic_minority.csv.gz", dtype=str, keep_default_na=False
    )
    real = frames["test"].loc[frames["test"].target.eq(1), list(schema.feature_columns)]
    marginal, marginal_summary = marginal_metrics(real, bank, schema)
    pairs, pair_summary = pairwise_metrics(real, bank, schema)
    marginal.to_csv(output / "marginal_components.csv", index=False)
    pairs.to_csv(output / "pairwise_components.csv", index=False)
    coverage = [len(set(real[c]) & set(bank[c])) / real[c].nunique() for c in real]
    pd.DataFrame({"column": list(real.columns), "coverage": coverage}).to_csv(
        output / "coverage_components.csv", index=False
    )
    train_minority = frames["train"].loc[frames["train"].target.eq(1), list(schema.feature_columns)]
    train_tuples = set(train_minority.itertuples(index=False, name=None))
    generated = list(bank.itertuples(index=False, name=None))
    fidelity = {
        **marginal_summary,
        **pair_summary,
        "mean_category_coverage": float(np.mean(coverage)),
        "domain_validity": 1.0,
        "unique_fraction": len(set(generated)) / len(generated),
        "train_exact_match": float(np.mean([row in train_tuples for row in generated])),
    }
    n = min(2048, len(real), len(bank))
    real_rng = np.random.default_rng(
        semantic_seed(task["seed"], task["dataset"], "metric_reference")
    )
    synth_rng = np.random.default_rng(
        semantic_seed(task["seed"], task["dataset"], task["model"], variant, "metric_generated")
    )
    real_indices = real_rng.choice(len(real), n, replace=False)
    synth_indices = synth_rng.choice(len(bank), n, replace=False)
    alpha = alpha_precision_beta_recall(
        real.iloc[real_indices],
        bank.iloc[synth_indices],
        schema,
        metric_seed=semantic_seed(task["seed"], task["dataset"], "alpha_beta"),
    )
    alpha.update(
        reference_indices=real_indices.tolist(),
        synthetic_indices=synth_indices.tolist(),
        reference_duplicate_fraction=1 - real.iloc[real_indices].drop_duplicates().shape[0] / n,
        synthetic_duplicate_fraction=1 - bank.iloc[synth_indices].drop_duplicates().shape[0] / n,
        distance_ties="discrete ties use <= as in V4",
    )
    write_json(output / "alpha_beta.json", alpha)
    downstream = evaluate_downstream(
        frames["train"],
        frames["validation"],
        frames["test"],
        bank,
        schema,
        classifier_seed=semantic_seed(task["seed"], task["dataset"], "classifier"),
        classifier_spec={
            "name": "logistic_regression",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": None,
            "solver": "lbfgs",
            "training_policy": "replace_observed_minority_with_synthetic_bank",
        },
        prediction_dir=output,
    )
    unknowns = {
        role: {c: int((~frames[role][c].isin(schema.category_domains[c])).sum()) for c in real}
        for role in ("validation", "test")
    }
    write_json(
        output / "metrics.json",
        {
            "status": "PASS",
            "dataset": task["dataset"],
            "seed": task["seed"],
            "model": task["model"],
            "variant": variant,
            "fidelity": fidelity,
            "downstream": downstream,
            "unknown_category_counts": unknowns,
            "bank_sha256": sha256(cell / bank_name / "synthetic_minority.csv.gz"),
        },
    )


def evaluate(root, source):
    if not (root / "primary/GATE_PASS.json").exists():
        raise ValueError("independent generation gate has not passed")
    tasks = json.loads((root / "primary/manifest.json").read_text())
    data = source / "prepared_data/categorical_imbalance_v1"
    worker = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    for i, task in enumerate(tasks):
        if i % count != worker:
            continue
        cell = root / "primary/cells" / task["id"]
        status = json.loads((cell / "status.json").read_text())
        variants = status.get("variants", {"primary": {"status": "PASS", "bank": status["bank"]}})
        for variant, record in variants.items():
            output = root / "evaluation" / f"{task['id']}__{variant}"
            if record["status"] != "PASS":
                write_json(
                    output / "status.json",
                    {"status": record["status"], "sampling": record},
                )
                continue
            try:
                evaluate_cell(task, cell, record["bank"], variant, data, output)
                write_json(output / "status.json", {"status": "PASS"})
            except Exception as error:
                write_json(output / "status.json", {"status": "FAIL", "error": str(error)})


def aggregate(root):
    tasks = json.loads((root / "primary/manifest.json").read_text())
    rows, failures, diagnostics = [], [], []
    for task in tasks:
        variants = ("primary", "official") if task["model"] == "tvae" else ("primary",)
        for variant in variants:
            path = root / "evaluation" / f"{task['id']}__{variant}"
            try:
                status = json.loads((path / "status.json").read_text())
                if variant == "official" and status["status"] == "CAPACITY_LIMITED":
                    diagnostics.append({"id": task["id"], "status": status["status"]})
                    continue
                if status["status"] != "PASS":
                    raise ValueError(str(status))
                metrics = json.loads((path / "metrics.json").read_text())
                marginal = pd.read_csv(path / "marginal_components.csv")
                pairs = pd.read_csv(path / "pairwise_components.csv")
                coverage = pd.read_csv(path / "coverage_components.csv")
                checks = {
                    "mean_categorical_tvd": marginal.discrepancy.mean(),
                    "mean_pair_similarity_categorical_categorical": pairs.similarity.mean(),
                    "mean_category_coverage": coverage.coverage.mean(),
                }
                for key, value in checks.items():
                    if not np.isclose(value, metrics["fidelity"][key], rtol=1e-12, atol=1e-12):
                        raise ValueError(f"component reconciliation: {key}")
                from sklearn.metrics import average_precision_score, matthews_corrcoef
                from .downstream import _best_mcc_threshold

                val = pd.read_csv(
                    path / "validation_predictions.csv.gz", float_precision="round_trip"
                )
                test = pd.read_csv(path / "test_predictions.csv.gz", float_precision="round_trip")
                threshold, _ = _best_mcc_threshold(
                    val.target.to_numpy(), val.probability.to_numpy()
                )
                for key, value in {
                    "test_pr_auc": average_precision_score(test.target, test.probability),
                    "test_mcc": matthews_corrcoef(test.target, test.probability >= threshold),
                }.items():
                    if not np.isclose(value, metrics["downstream"][key], rtol=1e-10, atol=1e-10):
                        raise ValueError(f"prediction reconciliation: {key}")
                rows.append(
                    {
                        "dataset": task["dataset"],
                        "seed": task["seed"],
                        "model": task["model"],
                        "variant": variant,
                        **{k: v for k, v in metrics["fidelity"].items() if v is not None},
                        "test_pr_auc": metrics["downstream"]["test_pr_auc"],
                        "test_mcc": metrics["downstream"]["test_mcc"],
                    }
                )
            except Exception as error:
                failures.append({"id": task["id"], "variant": variant, "error": str(error)})
    target = root / "aggregate"
    target.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.to_csv(target / "cells.csv", index=False)
        values = [c for c in frame.select_dtypes(include="number") if c != "seed"]
        per_dataset = frame.groupby(["dataset", "model", "variant"])[values].agg(
            ["mean", "std", "count"]
        )
        per_dataset.to_csv(target / "dataset_summary.csv")
        frame.groupby(["dataset", "model", "variant"])[values].mean().groupby(
            ["model", "variant"]
        ).mean().to_csv(target / "dataset_macro_summary.csv")
    write_json(
        target / "verification.json",
        {
            "status": "PASS" if not failures else "FAIL",
            "evaluated": len(rows),
            "expected_output_cells": 520,
            "failed": failures,
            "official_tvae_capacity_limited": diagnostics,
            "macro_policy": "dataset means then dataset macro; inspect completion denominators",
        },
    )
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("evaluate", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.operation == "evaluate":
        evaluate(args.root, args.source)
    else:
        aggregate(args.root)


if __name__ == "__main__":
    main()
