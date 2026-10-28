"""Manifest-driven final dataset preparation.

This module intentionally contains no dataset names. Dataset identity, transforms, and split seeds
are supplied by the experiment and its referenced dataset registry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .climb_data import prepare_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for manifest in prepare_experiment(args.experiment, args.output_root):
        print(manifest)
