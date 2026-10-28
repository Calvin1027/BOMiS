"""Slurm workers consume frozen cells through atomic directory claims."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from .benchmark import sha256, write_json
from .data import categorical_schema, read_train


def worker(root, phase, pool):
    phase_root = root / phase
    tasks = json.loads((phase_root / "manifest.json").read_text())
    if phase == "primary":

        def measured(task):
            identifier = task["id"].replace(str(task["seed"]), "20261001")
            prior = root / "engineering" / "cells" / identifier / "status.json"
            return (
                json.loads(prior.read_text()).get("seconds", task["cost_hint"])
                if prior.exists()
                else task["cost_hint"]
            )

        tasks.sort(key=lambda t: (-measured(t), t["id"]))
    worker_id = (
        f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    )
    claims = phase_root / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    logs = phase_root / "workers"
    logs.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stop_at = started + int(os.environ.get("CB_WALL_SECONDS", 259200)) - 1800
    failures, completed = [], []
    telemetry = None
    with (logs / f"{worker_id}.gpu.csv").open("w") as gpu_log:
        if pool == "gpu":
            telemetry = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,uuid,name,memory.total,memory.used,utilization.gpu",
                    "--format=csv",
                    "--loop=5",
                ],
                stdout=gpu_log,
                stderr=subprocess.STDOUT,
            )
        try:
            for task in tasks:
                if task["pool"] != pool or time.monotonic() >= stop_at:
                    continue
                claim = claims / task["id"]
                try:
                    claim.mkdir()
                except FileExistsError:
                    continue
                write_json(
                    claim / "owner.json",
                    {
                        "worker": worker_id,
                        "host": socket.gethostname(),
                        "pid": os.getpid(),
                        "claimed_at": time.time(),
                    },
                )
                output = phase_root / "cells" / task["id"]
                output.parent.mkdir(exist_ok=True)
                task_path = phase_root / "tasks" / f"{task['id']}.json"
                with (claim / "worker.log").open("w") as log:
                    proc = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "categorical_baselines.benchmark",
                            "generate",
                            "--task",
                            str(task_path),
                            "--output",
                            str(output),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    while proc.poll() is None:
                        write_json(claim / "heartbeat.json", {"time": time.time(), "pid": proc.pid})
                        time.sleep(5)
                status_path = output / "status.json"
                if not status_path.exists():
                    write_json(
                        status_path,
                        {"status": "FAIL", "stage": "process", "exit_code": proc.returncode},
                    )
                status = json.loads(status_path.read_text())
                tolerated = (
                    phase == "smoke"
                    and status.get("status") == "CAPACITY_LIMITED"
                    and status.get("reload_verified")
                )
                if status["status"] != "PASS" and not tolerated:
                    failures.append(task["id"])
                completed.append(task["id"])
                write_json(
                    claim / "TERMINAL.json",
                    {"status": status["status"], "exit_code": proc.returncode},
                )
                print(json.dumps({"cell": task["id"], "status": status["status"]}), flush=True)
        finally:
            if telemetry is not None:
                telemetry.terminate()
                telemetry.wait(timeout=20)
    write_json(
        logs / f"{worker_id}.json",
        {
            "completed": completed,
            "failed": failures,
            "seconds": time.monotonic() - started,
            "gpu_sharing": False,
        },
    )
    # The independent gate, not a worker exit code, decides experiment completion.


def audit_bank(task, cell, bank_name):
    bank = cell / bank_name
    sample_marker = json.loads((bank / "SAMPLE_COMPLETE.json").read_text())
    csv = bank / "synthetic_minority.csv.gz"
    if sha256(csv) != sample_marker["csv_sha256"]:
        raise ValueError("bank checksum mismatch")
    train = read_train(task["train"])
    schema = categorical_schema(train)
    frame = pd.read_csv(csv, dtype=str, keep_default_na=False)
    schema.restore_and_validate(frame, expected_n=task["rows"])
    counter_path = bank / "sample.json"
    if counter_path.exists():
        counters = json.loads(counter_path.read_text())
        if counters["rejection_used"]:
            raw = counters["raw_draw_count"]
            if raw > task["raw_cap"] or raw != sum(
                counters[key]
                for key in ("invalid_target_rows", "rejected_y0_rows", "decoded_y1_rows")
            ):
                raise ValueError("raw draw accounting failed")
            if (
                counters["decoded_y1_rows"]
                != counters["valid_y1_rows"] + counters["invalid_y1_rows"]
            ):
                raise ValueError("minority draw accounting failed")
            if counters["selected_output_rows"] != task["rows"]:
                raise ValueError("selected output count mismatch")
    return {"bank": str(bank), "rows": len(frame), "sha256": sample_marker["csv_sha256"]}


def audit(root, phase):
    tasks = json.loads((root / phase / "manifest.json").read_text())
    failed, verified, capacity = [], [], []
    for task in tasks:
        cell = root / phase / "cells" / task["id"]
        try:
            status = json.loads((cell / "status.json").read_text())
            if status["status"] == "PASS":
                verified.append({"id": task["id"], **audit_bank(task, cell, status["bank"])})
            elif (
                phase == "smoke"
                and status["status"] == "CAPACITY_LIMITED"
                and status.get("reload_verified")
            ):
                capacity.append(task["id"])
            else:
                raise ValueError(f"primary bank status: {status['status']}")
            official = status.get("variants", {}).get("official")
            if official:
                if official["status"] == "PASS":
                    audit_bank(task, cell, official["bank"])
                elif official["status"] != "CAPACITY_LIMITED" or not official.get(
                    "reload_verified"
                ):
                    raise ValueError("official TVAE diagnostic failed unexpectedly")
        except Exception as error:
            failed.append({"id": task["id"], "error": str(error)})
    result = {
        "status": "PASS" if not failed else "FAIL",
        "expected": len(tasks),
        "verified": verified,
        "short_fit_capacity_limited": capacity,
        "failed": failed,
    }
    write_json(root / phase / "AUDIT.json", result)
    if failed:
        raise SystemExit(1)
    write_json(
        root / phase / "GATE_PASS.json",
        {
            "status": "PASS",
            "expected": len(tasks),
            "manifest_sha256": sha256(root / phase / "manifest.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("work", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "engineering", "primary"), required=True)
    parser.add_argument("--pool", choices=("gpu", "cpu"))
    args = parser.parse_args()
    if args.operation == "work":
        worker(args.root, args.phase, args.pool)
    else:
        audit(args.root, args.phase)


if __name__ == "__main__":
    main()
