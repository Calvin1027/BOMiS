"""Frozen categorical experiment cells, with independent fit/sample artifacts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import platform
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters.vendor import add_source
from .contracts import (
    sample_result_metadata,
    validate_fit_artifacts,
    validate_fit_contract,
)
from .data import categorical_schema, read_train
from .registry import MODELS, make_model
from .seeds import semantic_seed

GPU_MODELS = ("ctgan", "tvae", "tabsyn", "tabnat", "stasy", "tabdiff")
NUMERICAL_FAILURES = {
    "NONFINITE_LOSS",
    "NONFINITE_GRADIENT",
    "NONFINITE_PARAMETERS",
    "NONFINITE_LOSS_OR_STATE",
    "NONFINITE_LIKELIHOOD",
    "LOSS_NONFINITE",
    "LOSS_MISSING_OR_NONFINITE",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def recipe(model, base, n0, n1, phase):
    options = dict(base)
    n = n0 + n1
    steps = 0
    if model in GPU_MODELS:
        batch = options["batch_size"]
        steps = max(1, n // batch) if model == "ctgan" else math.ceil(n / batch)
        if model in {"ctgan", "tvae"}:
            options["epochs"] = max(options["epochs"], math.ceil(1000 / steps))
        if phase == "smoke":
            options["epochs"] = max(1, math.ceil(50 / steps))
            if model == "tabsyn":
                options["vae_epochs"] = max(1, math.ceil(50 / steps))
            if model == "stasy":
                options.update(fine_tune_epochs=1, eps_iters=1)
        if model == "stasy":
            options["warmup"] = min(5000, max(1, math.ceil(0.05 * options["epochs"] * steps)))
    rows = min(16, n0) if phase == "smoke" else n0
    cap = max(200 * rows, (10 * rows * n + n1 - 1) // n1)
    return options, rows, cap, steps


def prepare(root, source):
    root, source = Path(root), Path(source)
    data = source / "prepared_data/categorical_imbalance_v1"
    index = json.loads((data / "index.json").read_text())
    config_path = source / "configs/models.generation_v1.json"
    config = json.loads(config_path.read_text())
    checked = set()
    manifests = []
    for relative in index["manifests"]:
        manifest = json.loads((data / relative).read_text())
        for entry in [*manifest["files"].values(), manifest["schema"], manifest["membership"]]:
            path = data / entry["path"]
            if str(path) not in checked:
                if sha256(path) != entry["sha256"]:
                    raise ValueError(f"input hash mismatch: {path}")
                checked.add(str(path))
        counts = manifest["files"]["train"]["class_counts"]
        if counts["1"] <= 5 or counts["0"] < 10 * counts["1"]:
            raise ValueError("invalid training class counts")
        manifests.append(manifest)
    if len(manifests) != 40 or len(checked) != 164:
        raise ValueError("incomplete prepared dataset")
    for phase in ("smoke", "engineering", "primary"):
        tasks = []
        for manifest in manifests:
            if phase != "primary" and manifest["seed"] != 20261001:
                continue
            n0, n1 = (manifest["files"]["train"]["class_counts"][str(c)] for c in (0, 1))
            models = list(MODELS) + (["ros", "independent"] if phase == "primary" else [])
            for model in models:
                options, rows, cap, steps = recipe(model, config["models"][model], n0, n1, phase)
                task = {
                    "id": f"{manifest['dataset']}__{manifest['seed']}__{model}",
                    "dataset": manifest["dataset"],
                    "seed": manifest["seed"],
                    "model": model,
                    "phase": phase,
                    "options": options,
                    "pool": "gpu" if model in GPU_MODELS else "cpu",
                    "train": str(data / manifest["files"]["train"]["path"]),
                    "train_sha256": manifest["files"]["train"]["sha256"],
                    "n0": n0,
                    "n1": n1,
                    "rows": rows,
                    "raw_cap": cap,
                    "steps_per_epoch": steps,
                    "config_sha256": sha256(config_path),
                    "lr_multipliers": config["lr_multipliers"] if model in GPU_MODELS else [1.0],
                }
                # Training counts, never held-out scores, provide a provisional cost ordering.
                task["cost_hint"] = max(1, steps) * options.get("epochs", 1)
                tasks.append(task)
        tasks.sort(key=lambda t: (-t["cost_hint"], t["id"]))
        for task in tasks:
            write_json(root / phase / "tasks" / f"{task['id']}.json", task)
        write_json(root / phase / "manifest.json", tasks)
    write_json(
        root / "INPUTS_VERIFIED.json",
        {
            "status": "PASS",
            "split_count": 40,
            "artifact_count": len(checked),
            "config_sha256": sha256(config_path),
            "generation_reads": "retained train only",
            "gpu_sharing": "disabled until paired throughput and reproducibility qualification",
        },
    )


def load_adapter(path, model):
    if model in {"tabsyn", "tabnat", "stasy", "tabdiff", "arf"}:
        add_source("arfpy" if model == "arf" else model)
    with Path(path).open("rb") as stream:
        return pickle.load(stream)  # trusted, locally generated inference artifact


def save_adapter(adapter, path):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    with temporary.open("xb") as stream:
        pickle.dump(adapter, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def seeds(task, variant="primary"):
    base = (task["dataset"], task["phase"], task["model"])
    return (
        semantic_seed(task["seed"], *base, "fit"),
        semantic_seed(task["seed"], *base, variant, "sample"),
    )


def fit_attempt(task, directory, multiplier):
    import torch

    started = time.monotonic()
    if task["pool"] == "gpu":
        properties = torch.cuda.get_device_properties(0)
        write_json(
            directory / "hardware.json",
            {
                "gpu": properties.name,
                "vram_bytes": properties.total_memory,
                "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
        )
        torch.cuda.reset_peak_memory_stats()
    if sha256(task["train"]) != task["train_sha256"]:
        raise ValueError("train hash mismatch")
    train = read_train(task["train"])
    schema = categorical_schema(train)
    if len(train) != task["n0"] + task["n1"] or int(train.target.sum()) != task["n1"]:
        raise ValueError("train count mismatch")
    options = dict(task["options"])
    if task["model"] in GPU_MODELS:
        options["lr_multiplier"] = multiplier
    if task["model"] in {"ctgan", "tvae"}:
        options["require_cuda"] = True
    elif task["model"] not in {"smote", "smoten", "smote_enc"}:
        options["device"] = "cuda" if task["pool"] == "gpu" else "cpu"
    adapter = make_model(task["model"], **options)
    fit_seed, _ = seeds(task)
    result = adapter.fit(train, schema, model_seed=fit_seed, work_dir=directory / "fit")
    validate_fit_contract(result, expected_model=task["model"], expected_outer_train_n=len(train))
    validate_fit_artifacts(result, work_dir=directory / "fit")
    write_json(directory / "fit.json", asdict(result))
    adapter.raw_draw_cap = task["raw_cap"]
    save_adapter(adapter, directory / "inference.pkl")
    del adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # A separate command subsequently loads this artifact and tests seeded replay.
    write_json(
        directory / "FIT_COMPLETE.json",
        {
            "status": "PASS",
            "seconds": time.monotonic() - started,
            "inference_sha256": sha256(directory / "inference.pkl"),
            "lr_multiplier": multiplier,
            "training_resume": "unsupported; restart complete fit",
            "peak_allocated_vram": torch.cuda.max_memory_allocated()
            if task["pool"] == "gpu"
            else None,
        },
    )


def paired_tvae(adapter, directory, seed):
    """Decode one shared latent bank both ways without intervening RNG changes."""
    import torch
    from .adapters._helpers import seed_everything

    model = adapter._model
    info = model.transformer._column_transform_info_list
    if info[-1].column_name != adapter.schema.target or info[-1].output_dimensions != 2:
        raise ValueError("TVAE target span changed")
    start = sum(c.output_dimensions for c in info[:-1])
    seed_everything(seed, require_torch=True, require_cuda=True)
    model.decoder.eval()
    with torch.no_grad():
        noise = torch.randn(1024, model.embedding_dim, device=model._device)
        logits, sigma = model.decoder(noise)
        official = torch.tanh(logits)
        sampled = official.clone()
        probabilities = logits[:, start : start + 2].softmax(1)
        choices = torch.multinomial(probabilities, 1)
        sampled[:, start : start + 2] = 0
        sampled[:, start : start + 2].scatter_(1, choices, 1)
        frames = [
            model.transformer.inverse_transform(x.cpu().numpy(), sigma.cpu().numpy())
            for x in (official, sampled)
        ]
    pd.testing.assert_frame_equal(
        frames[0].drop(columns="target"), frames[1].drop(columns="target")
    )
    for name, frame in zip(("official", "sampled"), frames, strict=True):
        frame.to_csv(directory / f"paired_{name}.csv.gz", index=False)
    write_json(
        directory / "paired_decoder.json",
        {
            "rows": 1024,
            "seed": seed,
            "same_latent_logits": True,
            "features_identical": True,
            "official_y1": int(frames[0].target.sum()),
            "sampled_y1": int(frames[1].target.sum()),
            "latent_sha256": hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
        },
    )


def sample_attempt(task, fitted, directory, chunk, variant):
    started = time.monotonic()
    marker = json.loads((fitted / "FIT_COMPLETE.json").read_text())
    if sha256(fitted / "inference.pkl") != marker["inference_sha256"]:
        raise ValueError("inference checkpoint hash mismatch")
    adapter = load_adapter(fitted / "inference.pkl", task["model"])
    adapter.raw_draw_cap = task["raw_cap"]
    if hasattr(adapter, "raw_batch_size"):
        adapter.raw_batch_size = chunk
    if variant == "official":
        adapter.target_decode = "official_argmax"
    _, sample_seed = seeds(task, variant)
    # Check native outputs independently of whether a short fit emits minority Y.
    if task["model"] in GPU_MODELS or task["model"] == "arf":
        from .adapters._helpers import seed_everything

        def draw():
            if task["model"] in {"ctgan", "tvae"}:
                return adapter._draw(7, sample_seed)
            seed_everything(sample_seed, require_torch=task["model"] != "arf")
            return adapter.codec.decode(adapter._draw_codes(7))

        first = draw()
        # Reload, rather than comparing two calls on an unverified mutable object.
        del adapter
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        adapter = load_adapter(fitted / "inference.pkl", task["model"])
        adapter.raw_draw_cap = task["raw_cap"]
        adapter.raw_batch_size = chunk
        if variant == "official":
            adapter.target_decode = "official_argmax"
        pd.testing.assert_frame_equal(first, draw())
        adapter.schema.restore_and_validate(first.drop(columns="target"), expected_n=7)
        if not first.target.isin([0, 1]).all():
            raise ValueError("invalid native joint labels")
    write_json(directory / "RELOAD_VERIFIED.json", {"status": "PASS"})
    if task["model"] == "tvae" and variant == "primary":
        paired_tvae(adapter, directory, semantic_seed(sample_seed, "paired"))
    result = adapter.sample_minority(task["rows"], sampling_seed=sample_seed)
    result.frame.to_csv(directory / "synthetic_minority.csv.gz", index=False)
    write_json(directory / "sample.json", sample_result_metadata(result))
    write_json(
        directory / "SAMPLE_COMPLETE.json",
        {
            "status": "PASS",
            "rows": len(result.frame),
            "seconds": time.monotonic() - started,
            "sample_seed": sample_seed,
            "raw_batch_size": chunk,
            "csv_sha256": sha256(directory / "synthetic_minority.csv.gz"),
        },
    )


def child(task_path, operation, output, fitted=None, multiplier=1.0, chunk=1, variant="primary"):
    command = [
        sys_executable(),
        "-m",
        "categorical_baselines.benchmark",
        operation,
        "--task",
        str(task_path),
        "--output",
        str(output),
        "--multiplier",
        str(multiplier),
        "--chunk",
        str(chunk),
        "--variant",
        variant,
    ]
    if fitted is not None:
        command += ["--fitted", str(fitted)]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "process.log").open("w") as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    error_path = output / "ERROR.json"
    error = (
        json.loads(error_path.read_text())
        if error_path.exists()
        else {"code": "PROCESS_EXIT", "exit_code": code}
    )
    return code, error


def sys_executable():
    import sys

    return sys.executable


def generate_cell(task_path, output):
    task = json.loads(Path(task_path).read_text())
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "task.json", task)
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "threads": os.environ.get("OMP_NUM_THREADS"),
        },
    )
    if task["model"] in {"ros", "independent"}:
        train = read_train(task["train"])
        schema = categorical_schema(train)
        minority = train.loc[train.target.eq(1), list(schema.feature_columns)]
        rng = np.random.default_rng(seeds(task)[1])
        if task["model"] == "ros":
            frame = minority.iloc[rng.integers(len(minority), size=task["rows"])].reset_index(
                drop=True
            )
        else:
            frame = pd.DataFrame(
                {c: rng.choice(minority[c].to_numpy(), task["rows"]) for c in minority}
            )
        schema.restore_and_validate(frame, expected_n=task["rows"])
        bank = output / "primary"
        bank.mkdir()
        frame.to_csv(bank / "synthetic_minority.csv.gz", index=False)
        write_json(
            bank / "SAMPLE_COMPLETE.json",
            {
                "status": "PASS",
                "rows": len(frame),
                "csv_sha256": sha256(bank / "synthetic_minority.csv.gz"),
            },
        )
        write_json(
            output / "status.json",
            {"status": "PASS", "bank": "primary", "seconds": time.monotonic() - started},
        )
        return
    fitted = None
    for attempt, multiplier in enumerate(task["lr_multipliers"]):
        fit_dir = output / f"fit_attempt_{attempt}"
        code, error = child(task_path, "fit", fit_dir, multiplier=multiplier)
        if code == 0:
            fitted = fit_dir
            break
        if error.get("code") not in NUMERICAL_FAILURES:
            break
    if fitted is None:
        write_json(output / "status.json", {"status": "FAIL", "stage": "fit", "error": error})
        return
    variants = ["primary"] + (["official"] if task["model"] == "tvae" else [])
    statuses = {}
    for variant in variants:
        chunk = task["options"].get("raw_batch_size", 1)
        while True:
            bank = output / f"{variant}_chunk_{chunk}"
            code, error = child(
                task_path, "sample", bank, fitted=fitted, chunk=chunk, variant=variant
            )
            if code == 0:
                statuses[variant] = {"status": "PASS", "bank": bank.name}
                break
            if error.get("code") == "CUDA_OOM" and chunk > 1:
                chunk = max(1, chunk // 2)
                continue
            reload_ok = (bank / "RELOAD_VERIFIED.json").exists()
            statuses[variant] = {
                "status": "CAPACITY_LIMITED" if error.get("code") == "CAPACITY_FAILURE" else "FAIL",
                "error": error,
                "reload_verified": reload_ok,
                "bank": bank.name,
            }
            break
    primary = statuses["primary"]
    write_json(
        output / "status.json",
        {
            **primary,
            "variants": statuses,
            "fit": fitted.name,
            "seconds": time.monotonic() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "generate", "fit", "sample"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fitted", type=Path)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--variant", default="primary")
    args = parser.parse_args()
    try:
        if args.operation == "prepare":
            prepare(args.root, args.source)
        elif args.operation == "generate":
            generate_cell(args.task, args.output)
        else:
            import torch

            torch.set_num_threads(int(os.environ.get("CB_THREADS", "3")))
            task = json.loads(args.task.read_text())
            if args.operation == "fit":
                fit_attempt(task, args.output, args.multiplier)
            else:
                sample_attempt(task, args.fitted, args.output, args.chunk, args.variant)
    except Exception as error:
        if args.output:
            code = getattr(error, "code", type(error).__name__)
            if "out of memory" in str(error).lower():
                code = "CUDA_OOM" if "cuda" in str(error).lower() else "HOST_OOM"
            write_json(
                args.output / "ERROR.json",
                {"code": code, "message": str(error), "context": getattr(error, "context", {})},
            )
        traceback.print_exc()
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
