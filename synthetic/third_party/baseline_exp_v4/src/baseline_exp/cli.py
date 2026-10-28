from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run-cell", choices=["run-cell"])
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    run_cell(project_root=project, experiment_config=args.experiment, model_config=args.models, dataset_manifest=args.dataset, model_name=args.model, split_seed=args.split_seed, output_root=args.output_root)
