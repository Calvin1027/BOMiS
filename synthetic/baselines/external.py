"""C27: 외부 baseline 래퍼. baseline 프로젝트 CLI를 하위 프로세스로 부른다(코드 수정 없음)."""

import json
import os
import shutil
import subprocess
import tarfile
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


def _path(name: str, default: str) -> str:
    """경로 설정(config/paths.sh에서 export). 없으면 원래 서버 경로. 빈 문자열이면 '쓰지 않음'."""
    return os.environ.get(name, default)


RUNTIME = Path(_path("BOMIS_RUNTIME", "/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime"))
BL_PYTHON = _path("BL_PYTHON", "/mnt/data1/calvinkim1027/miniconda3/envs/tabular_foundation/bin/python")
GEN_SITE = _path("BL_GEN_SITE", "/mnt/data1/calvinkim1027/baseline_exp_runtime_v3/envs/generation")
EXTRAS = _path("BL_EXTRAS_TAR", "/mnt/data1/calvinkim1027/categorical_baselines_runtime_20260929/extras.tar.gz")
COLS = [f"c{j}" for j in range(8)]
A_MODELS = ("smoten", "smote_enc", "ctgan")
B_MODELS = ("tabsyn", "tabdiff")
DEVICE = {"smoten": "cpu", "smote_enc": "cpu", "smote": "cpu", "arf": "cpu", "ctgan": "cuda", "tabsyn": "cuda",
          "tabdiff": "cuda", "tabddpm": "cuda", "forest_flow": "cpu"}
# TabDDPM은 categorical_baselines에 없어 과거 파이프라인(Baseline_exp_v4) 코드를 tabddpm_cli.py로 부른다.
V4_SRC = _path("BL_V4_SRC", "/mnt/data1/calvinkim1027/baseline_exp_runtime_v4_32g_20260910b/TDG/Baseline_exp_v4/src")
TABDDPM_SITE = _path("BL_TABDDPM_SITE", "/mnt/data1/calvinkim1027/baseline_exp_runtime_v3/envs/tabddpm_extras")
_TABDDPM_SOURCE = _path("BL_TABDDPM_SOURCE", "/mnt/data1/calvinkim1027/baseline_exp_runtime_v3/TDG/MCMC_in_generative model/generator_stage_v1/third_party/tab-ddpm")
# reference = 과거 최종 설정(configs/models.climb_final.v8.full_access.no_forest.json), smoke = 연결 확인용
TABDDPM = {
    "reference": {"source_dir": _TABDDPM_SOURCE, "train_steps": 10000, "diffusion_steps": 1000, "batch_size": 1024,
                  "learning_rate": 0.0002, "learning_rate_backoff": [0.0001, 0.00005], "gradient_clip_norm": 1.0,
                  "max_nonfinite_gradient_skips": 20},
    "smoke": {"source_dir": _TABDDPM_SOURCE, "train_steps": 200, "diffusion_steps": 100, "batch_size": 1024,
              "learning_rate": 0.0002, "learning_rate_backoff": [0.0001, 0.00005], "gradient_clip_norm": 1.0,
              "max_nonfinite_gradient_skips": 20},
}
# ForestDiffusion(Forest-Flow)도 과거 파이프라인 어댑터를 forestflow_cli.py로 부른다.
# reference = 과거 최종 설정(Baseline_exp_v2/configs/models.climb_final.v4.json), smoke = 과거 models.smoke.v3.json
_FF = {"source_dir": _path("BL_FF_SOURCE", "/mnt/data1/calvinkim1027/baseline_exp_runtime_v3/TDG/Baseline_exp_v2/third_party/ForestDiffusion"),
       "patch_file": _path("BL_FF_PATCH", "/mnt/data1/calvinkim1027/baseline_exp_runtime_v3/TDG/Baseline_exp_v2/patches/forest_flow_target_label.patch")}
FORESTFLOW = {
    "reference": {**_FF, "n_t": 50, "duplicate_k": 100, "n_batch": 1, "n_jobs": 2, "n_estimators": 100, "max_depth": 7},
    "smoke": {**_FF, "n_t": 5, "duplicate_k": 2, "n_batch": 1, "n_jobs": 4, "n_estimators": 20, "max_depth": 4},
}
OLD_PIPELINE = {"tabddpm": ("tabddpm_cli.py", TABDDPM), "forest_flow": ("forestflow_cli.py", FORESTFLOW)}
# 실패하면 앞에서부터 차례로 덮어쓴 설정으로 다시 돈다(사용자 지시 10/27: 실패 시 설정을 바꿔 성공시킨다).
RETRY = {
    "smoten": [{"k_neighbors": 3}],
    "smote_enc": [{"k_neighbors": 3}],
    "smote": [{"k_neighbors": 3}],
    "arf": [{"max_raw_multiplier": 2000}],
    "ctgan": [{"max_raw_multiplier": 2000}, {"max_raw_multiplier": 2000, "batch_size": 250}],
    "tabsyn": [{"max_raw_multiplier": 2000}],
    "tabdiff": [{"max_raw_multiplier": 2000}],
}


def bl_src() -> Path:
    return Path(os.environ.get("BL_SRC", RUNTIME / "baselines_src"))


def _pythonpath(*parts) -> str:
    # 빈 값(설정에서 끈 경로)은 건너뛴다.
    return ":".join(str(p) for p in parts if p)


def ensure_deps() -> Path | None:
    # D27.2: extras는 job마다 노드 로컬 폴더에 한 번 푼다. BL_EXTRAS_TAR가 빈 값이면 쓰지 않는다(env에 이미 설치된 경우).
    if os.environ.get("BL_DEPS"):
        return Path(os.environ["BL_DEPS"])
    if not EXTRAS:
        return None
    tag = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    deps = Path(os.environ.get("SLURM_TMPDIR") or "/tmp") / f"bl_deps_{tag}"
    marker = deps / ".unpacked"
    if not marker.exists():
        deps.mkdir(parents=True, exist_ok=True)
        with tarfile.open(EXTRAS) as tar:
            tar.extractall(deps)
        marker.touch()
    return deps


def train_table(D0: np.ndarray, D1: np.ndarray) -> pd.DataFrame:
    # D27.1: 열 c0–c7과 target, 값은 문자열.
    X = np.vstack([D0, D1]).astype(np.int64)
    frame = pd.DataFrame(X.astype(str), columns=COLS)
    frame["target"] = np.r_[np.zeros(len(D0), int), np.ones(len(D1), int)].astype(str)
    return frame


def _attempt(name, options, train_csv, n, seed, device, workdir, timeout):
    config = workdir / f"config_{uuid.uuid4().hex[:8]}.json"
    config.write_text(json.dumps({"purpose": "bomis_synthetic_v2", "models": {name: options}}))
    out = workdir / f"out_{uuid.uuid4().hex[:8]}"
    src = bl_src()
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    python = os.environ.get("BL_PYTHON", BL_PYTHON)
    if name in OLD_PIPELINE:
        env["PYTHONPATH"] = _pythonpath(V4_SRC, TABDDPM_SITE, GEN_SITE)
        cmd = [python, str(Path(__file__).resolve().with_name(OLD_PIPELINE[name][0])), "--train", str(train_csv),
               "--config", str(config), "--seed", str(seed), "--rows", str(n), "--output", str(out)]
        cwd = workdir
    else:
        env["PYTHONPATH"] = _pythonpath(src / "src", ensure_deps(), GEN_SITE)
        cmd = [python, "-m", "categorical_baselines.cli",
               "--model", name, "--train", str(train_csv), "--config", str(config),
               "--seed", str(seed), "--rows", str(n), "--device", device, "--output", str(out)]
        cwd = src
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        code, tail = proc.returncode, (proc.stdout + proc.stderr)[-2000:]
    except (subprocess.TimeoutExpired, OSError) as error:
        code, tail = -1, f"{type(error).__name__}: {error}"
    seconds = time.monotonic() - t0
    status = {}
    if (out / "status.json").exists():
        status = json.loads((out / "status.json").read_text())
    csv = out / "synthetic_minority.csv.gz"
    if code == 0 and csv.exists() and status.get("status") == "PASS":
        rows = pd.read_csv(csv, dtype=str, keep_default_na=False)[COLS].astype(np.int64).to_numpy()
        if len(rows) == n and rows.min() >= 0 and rows.max() <= 3:
            return dict(ok=True, rows=rows, reason="", seconds=seconds)
        return dict(ok=False, rows=None, reason=f"bad_rows shape={rows.shape}", seconds=seconds)
    reason = status.get("error_type", "") + ": " + status.get("error", "") if status else f"exit {code}: {tail}"
    return dict(ok=False, rows=None, reason=reason[-1500:], seconds=seconds)


def run_external(name: str, D0: np.ndarray, D1: np.ndarray, n: int, seed: int, workdir: Path,
                 device: str | None = None, base_config: Path | None = None, timeout: float = 3 * 3600) -> dict:
    """C27, D27.1–D27.3. 실패하면 RETRY의 설정으로 다시 돈다. 돌려주는 dict:
    ok, rows, reason(마지막 실패 사유), seconds(시도 합), attempt(성공한 시도 번호, 0이 기본), options."""
    workdir = Path(workdir).resolve()  # 하위 프로세스는 cwd=BL_SRC라 절대 경로가 필요하다
    workdir.mkdir(parents=True, exist_ok=True)
    train_csv = workdir / "train.csv"
    train_table(D0, D1).to_csv(train_csv, index=False)
    base_config = Path(base_config).resolve() if base_config else bl_src() / "configs" / "models.reference.json"
    if name in OLD_PIPELINE:
        base = dict(OLD_PIPELINE[name][1]["smoke" if "smoke" in base_config.name else "reference"])
    else:
        base = dict(json.loads(Path(base_config).read_text())["models"][name])
    device = device or DEVICE[name]
    total, reasons = 0.0, []
    for attempt, override in enumerate([{}] + RETRY.get(name, [])):
        options = {**base, **override}
        res = _attempt(name, options, train_csv, n, seed, device, workdir, timeout)
        total += res["seconds"]
        if res["ok"]:
            shutil.rmtree(workdir, ignore_errors=True)
            return dict(ok=True, rows=res["rows"], reason="; ".join(reasons), seconds=total,
                        attempt=attempt, options=options)
        reasons.append(f"[{attempt}] {res['reason']}")
    return dict(ok=False, rows=None, reason="; ".join(reasons)[-3000:], seconds=total,
                attempt=-1, options=None)
