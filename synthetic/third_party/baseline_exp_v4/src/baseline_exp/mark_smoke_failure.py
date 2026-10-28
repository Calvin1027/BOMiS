from __future__ import annotations

import argparse
from pathlib import Path

from .provenance import write_json_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    args.cell.mkdir(parents=True, exist_ok=True)
    if not (args.cell / "GENERATED.json").exists() and not (args.cell / "FAILURE.json").exists():
        dataset = args.cell.parent.parent.name
        split_token = args.cell.name.removeprefix("split-")
        write_json_new(
            args.cell / "FAILURE.json",
            {
                "status": "FAILED",
                "stage": "external_process",
                "code": "TIMEOUT_OR_UNCAUGHT_PROCESS_FAILURE",
                "model": args.model,
                "dataset": dataset,
                "split_seed": int(split_token) if split_token.isdigit() else None,
                "exit_code": args.exit_code,
                "failure_context": {},
                "fallback_used": False,
                "backfill_used": False,
            },
        )
