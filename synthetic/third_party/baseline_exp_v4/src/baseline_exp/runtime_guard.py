"""Recheck run authorization and frozen file identity when queued tasks start."""
import argparse
import json
from pathlib import Path
from .provenance import sha256_file
from .final_preflight import _tracked_files


def verify_run(project: Path, run_root: Path) -> None:
    approval = json.loads((run_root / "APPROVAL.json").read_text())
    preflight_path = run_root / "PREFLIGHT_PASS.json"
    preflight = json.loads(preflight_path.read_text())
    frozen = json.loads((project / "RUNTIME_FROZEN.json").read_text())
    if (approval.get("status") != "APPROVED" or approval.get("compute_project") != str(project)
        or approval.get("result_root") != str(run_root)
        or run_root.name != "final_" + str(approval.get("run_tag"))
        or approval.get("preflight_sha256") != sha256_file(preflight_path)
        or frozen.get("preflight_sha256") != sha256_file(preflight_path)
        or preflight.get("compute_project") != str(project)
        or preflight.get("status") != "PASS_READY_FOR_APPROVAL"):
        raise RuntimeError("queued job authorization/runtime mismatch")
    current = {str(p.relative_to(project)): sha256_file(p) for p in _tracked_files(project)}
    if current != preflight.get("tracked_files_sha256"):
        raise RuntimeError("frozen runtime drifted after submission")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    verify_run(args.project, args.run_root)
