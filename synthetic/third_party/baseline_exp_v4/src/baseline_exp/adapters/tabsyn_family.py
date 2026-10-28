from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ..contracts import AdapterFailure, DatasetSchema, FitResult, GeneratorAdapter, SampleResult
from ..seeds import semantic_seed
from ._external import isolate_source, run_logged, write_json_new
from ._helpers import bounded_minority_rejection, canonical_joint


# These are effective values in the source-locked EF-VFM config. ``train_steps``
# is consumed by an outer ``for epoch`` loop, while ``sample.batch_size`` is the
# actual GPU ODE minibatch. The adapter's ``raw_batch_size`` is instead the
# rejection wrapper's requested row count per upstream process invocation.
TABBYFLOW_TRAINING_BATCH_SIZE = 4096
TABBYFLOW_UPSTREAM_SAMPLING_BATCH_SIZE = 10_000


def _dataset_id(model: str, work_dir: Path) -> str:
    digest = hashlib.sha256(str(work_dir.resolve()).encode()).hexdigest()[:16]
    return f"baseline_{model}_{digest}"


def _prepare(
    repo: Path,
    model: str,
    table: pd.DataFrame,
    schema: DatasetSchema,
    work: Path,
    *,
    n_classes: int = 2,
) -> str:
    dataset = _dataset_id(model, work)
    data_dir, info_path = repo / "data" / dataset, repo / "data" / "Info" / f"{dataset}.json"
    if data_dir.exists() or info_path.exists():
        raise AdapterFailure("fit", "UPSTREAM_NAMESPACE_EXISTS", dataset)
    data_dir.mkdir(parents=True)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [*schema.numerical_columns, *schema.categorical_columns, schema.target]
    raw = table.loc[:, ordered].copy()
    raw_path, test_path = data_dir / f"{dataset}.csv", data_dir / "outer_train_copy.csv"
    raw.to_csv(raw_path, index=False)
    raw.to_csv(test_path, index=False)
    n_num, n_cat = len(schema.numerical_columns), len(schema.categorical_columns)
    write_json_new(
        info_path,
        {
            "name": dataset,
            "task_type": "binclass",
            "header": "infer",
            "column_names": ordered,
            "num_col_idx": list(range(n_num)),
            "cat_col_idx": list(range(n_num, n_num + n_cat)),
            "target_col_idx": [n_num + n_cat],
            "file_type": "csv",
            "data_path": str(raw_path),
            "test_path": str(test_path),
            "val_path": None,
            "n_classes": int(n_classes),
        },
    )
    run_logged(
        [sys.executable, "process_dataset.py", "--dataname", dataset],
        cwd=repo,
        log_path=work / "preprocess.log",
    )
    processed = json.loads((data_dir / "info.json").read_text(encoding="utf-8"))
    if int(processed.get("train_num", -1)) != len(table):
        raise AdapterFailure("fit", "PREPROCESS_ROW_DRIFT", "upstream changed outer-train size")
    return dataset


def _read_joint(path: Path, schema: DatasetSchema, n: int) -> pd.DataFrame:
    if not path.is_file():
        raise AdapterFailure("sample", "SAMPLE_FILE_MISSING", str(path))
    raw = pd.read_csv(path)
    expected = [*schema.numerical_columns, *schema.categorical_columns, schema.target]
    if set(raw.columns) == set(expected):
        raw = raw.loc[:, expected]
    elif list(raw.columns) == [str(i) for i in range(len(expected))]:
        raw.columns = expected
    else:
        raise AdapterFailure("sample", "RAW_SCHEMA_MISMATCH", "external CSV width mismatch")
    if len(raw) != n:
        raise AdapterFailure("sample", "RAW_SAMPLE_COUNT_MISMATCH", "external CSV row mismatch")
    return raw.loc[:, [*schema.feature_columns, schema.target]]


class TabSynAdapter(GeneratorAdapter):
    name = "tabsyn"

    def __init__(
        self,
        source_dir: Path,
        *,
        vae_epochs: int = 4000,
        diffusion_epochs: int = 10001,
        sampling_steps: int = 50,
        max_raw_multiplier: int = 80,
        raw_batch_size: int = 100000,
    ) -> None:
        super().__init__()
        self.source_dir = source_dir
        self.vae_epochs = int(vae_epochs)
        self.diffusion_epochs = int(diffusion_epochs)
        self.sampling_steps = int(sampling_steps)
        self.max_raw_multiplier = int(max_raw_multiplier)
        self.raw_batch_size = int(raw_batch_size)
        self._repo: Path | None = None
        self._dataset: str | None = None
        self._work: Path | None = None

    def fit(self, train: pd.DataFrame, schema: DatasetSchema, *, model_seed: int, work_dir: Path) -> FitResult:
        if min(
            self.vae_epochs,
            self.diffusion_epochs,
            self.sampling_steps,
            self.max_raw_multiplier,
            self.raw_batch_size,
        ) < 1:
            raise AdapterFailure("configuration", "INVALID_BUDGET", "invalid TabSyn budget")
        repo, source_hash = isolate_source(self.source_dir, work_dir / "_source")
        table = canonical_joint(train, schema)
        dataset = _prepare(repo, self.name, table, schema, work_dir)
        vae_ckpt, diff_ckpt = repo / "tabsyn" / "vae" / "ckpt" / dataset, repo / "tabsyn" / "ckpt" / dataset
        if vae_ckpt.exists() or diff_ckpt.exists():
            raise AdapterFailure("fit", "CHECKPOINT_EXISTS", "TabSyn checkpoint exists")
        common = [sys.executable, "main.py", "--dataname", dataset, "--gpu", "0"]
        run_logged(common + ["--method", "vae", "--mode", "train", "--seed", str(model_seed), "--vae-epochs", str(self.vae_epochs)], cwd=repo, log_path=work_dir / "vae-train.log")
        run_logged(common + ["--method", "tabsyn", "--mode", "train", "--seed", str(model_seed), "--diffusion-epochs", str(self.diffusion_epochs)], cwd=repo, log_path=work_dir / "diffusion-train.log")
        status_paths = [vae_ckpt / "training_status.json", diff_ckpt / "training_status.json"]
        if not all(path.is_file() for path in status_paths):
            raise AdapterFailure("fit", "TRAIN_STATUS_MISSING", "TabSyn status missing")
        statuses = [json.loads(path.read_text(encoding="utf-8")) for path in status_paths]
        vae_requested = int(statuses[0].get("requested_epochs", self.vae_epochs))
        vae_completed = int(statuses[0].get("completed_epochs", -1))
        diffusion_requested = int(statuses[1].get("requested_epochs", -1))
        diffusion_completed = int(statuses[1].get("completed_epochs", -1))
        diffusion_early_stopped = statuses[1].get("early_stopped")
        if (
            vae_requested != self.vae_epochs
            or vae_completed != self.vae_epochs
            or diffusion_requested != self.diffusion_epochs
            or not 1 <= diffusion_completed <= self.diffusion_epochs
            or (
                diffusion_completed < self.diffusion_epochs
                and diffusion_early_stopped is not True
            )
        ):
            raise AdapterFailure("fit", "TRAINING_INCOMPLETE", "TabSyn budget incomplete")
        for key, status in (("best_validation_categorical_loss", statuses[0]), ("best_loss", statuses[1])):
            if not math.isfinite(float(status.get(key, math.nan))):
                raise AdapterFailure("fit", "LOSS_INVALID", f"invalid {key}")
        self.schema, self._repo, self._dataset, self._work = schema, repo, dataset, work_dir
        self.fit_result = FitResult(
            model=self.name, model_seed=model_seed, n_outer_train=len(table),
            n_minority_train=int(table[schema.target].sum()), sampling_mode="bounded_joint_rejection",
            loss_artifacts=(str(status_paths[0].relative_to(work_dir)), str(status_paths[1].relative_to(work_dir))),
            checkpoint_artifacts=(str(vae_ckpt.relative_to(work_dir)), str(diff_ckpt.relative_to(work_dir))),
            details={"algorithm": "amazon-science/tabsyn", "vae_epochs": self.vae_epochs, "diffusion_epochs": self.diffusion_epochs, "sampling_steps": self.sampling_steps, "source_tree_sha256": source_hash, "training_status": statuses},
            training_access="full_outer_train",
            actual_training_population="full_joint",
            cross_class_usage="shared_joint_model",
            minority_sampling_semantics="bounded_joint_rejection",
        )
        return self.fit_result

    def _sample_minority(self, n: int, *, sampling_seed: int) -> SampleResult:
        assert self.schema is not None and self._repo is not None and self._dataset is not None and self._work is not None
        chunk_index = 0
        def draw(size: int, _: int) -> pd.DataFrame:
            nonlocal chunk_index
            seed = semantic_seed(sampling_seed, "tabsyn_raw_chunk", chunk_index)
            output = self._work / f"raw-seed-{sampling_seed}-chunk-{chunk_index:04d}.csv"
            run_logged([sys.executable, "main.py", "--dataname", self._dataset, "--gpu", "0", "--method", "tabsyn", "--mode", "sample", "--seed", str(seed), "--num-samples", str(size), "--sample-batch-size", str(self.raw_batch_size), "--steps", str(self.sampling_steps), "--save_path", str(output)], cwd=self._repo, log_path=self._work / f"sample-chunk-{chunk_index:04d}.log", stage="sample")
            frame = _read_joint(output, self.schema, size)
            chunk_index += 1
            return frame
        return bounded_minority_rejection(draw, self.schema, n=n, sampling_seed=sampling_seed, raw_multiplier=self.max_raw_multiplier, raw_batch_size=self.raw_batch_size)


class TabbyFlowAdapter(GeneratorAdapter):
    """Official EF-VFM fitted to full outer-train X,Y and sampled by rejection."""
    name = "tabbyflow"

    def __init__(
        self,
        source_dir: Path,
        *,
        train_steps: int = 8000,
        max_raw_multiplier: int = 80,
        raw_batch_size: int = 100000,
        sample_retry_attempts: int = 1,
    ) -> None:
        super().__init__()
        self.source_dir, self.train_steps = source_dir, int(train_steps)
        self.max_raw_multiplier = int(max_raw_multiplier)
        self.raw_batch_size = int(raw_batch_size)
        self.sample_retry_attempts = int(sample_retry_attempts)
        self._repo: Path | None = None
        self._dataset: str | None = None
        self._experiment: str | None = None
        self._work: Path | None = None

    def fit(self, train: pd.DataFrame, schema: DatasetSchema, *, model_seed: int, work_dir: Path) -> FitResult:
        if min(
            self.train_steps,
            self.max_raw_multiplier,
            self.raw_batch_size,
            self.sample_retry_attempts,
        ) < 1:
            raise AdapterFailure("configuration", "INVALID_BUDGET", "invalid TabbyFlow budget")
        repo, source_hash = isolate_source(self.source_dir, work_dir / "_source")
        table = canonical_joint(train, schema)
        minority_count = int(table[schema.target].sum())
        if minority_count < 1:
            raise AdapterFailure("fit", "NO_MINORITY", "outer-train contains no minority row")
        dataset = _prepare(repo, self.name, table, schema, work_dir, n_classes=2)
        experiment = f"{dataset}_m{model_seed}"
        checkpoint = repo / "ef_vfm" / "ckpt" / dataset / experiment
        if checkpoint.exists():
            raise AdapterFailure("fit", "CHECKPOINT_EXISTS", str(checkpoint))
        run_logged([sys.executable, "main.py", "--dataname", dataset, "--mode", "train", "--exp_name", experiment, "--seed", str(model_seed), "--gpu", "0", "--no_wandb", "--disable_upstream_eval", "--train_steps", str(self.train_steps)], cwd=repo, log_path=work_dir / "train.log")
        status_path = checkpoint / "training_status.json"
        if not status_path.is_file():
            raise AdapterFailure("fit", "TRAIN_STATUS_MISSING", str(status_path))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if bool(status.get("stopped_due_to_nan")) or int(status.get("completed_epochs", -1)) != self.train_steps:
            raise AdapterFailure("fit", "TRAINING_INCOMPLETE", "TabbyFlow status failed")
        self.schema, self._repo, self._dataset, self._experiment, self._work = schema, repo, dataset, experiment, work_dir
        self.fit_result = FitResult(
            model=self.name, model_seed=model_seed, n_outer_train=len(table),
            n_minority_train=minority_count, sampling_mode="bounded_joint_rejection",
            loss_artifacts=("train.log",), checkpoint_artifacts=(str(checkpoint.relative_to(work_dir)),),
            details={
                "algorithm": "andresguzco/ef-vfm TabbyFlow",
                "training_population": "outer-train joint X,Y",
                "n_fit": len(table),
                "target_classes": 2,
                "target_role": "joint_generated_categorical_variable",
                "train_steps": self.train_steps,
                "configured_training_budget_key": "train_steps",
                "effective_training_unit": "epoch",
                "effective_training_count": self.train_steps,
                "training_batch_size": TABBYFLOW_TRAINING_BATCH_SIZE,
                "effective_optimizer_updates": self.train_steps
                * math.ceil(len(table) / TABBYFLOW_TRAINING_BATCH_SIZE),
                "max_raw_multiplier": self.max_raw_multiplier,
                "raw_batch_size": self.raw_batch_size,
                "raw_batch_size_semantics": "rejection_request_chunk_size",
                "rejection_request_chunk_size": self.raw_batch_size,
                "effective_upstream_sampling_batch_size": (
                    TABBYFLOW_UPSTREAM_SAMPLING_BATCH_SIZE
                ),
                "sample_retry_attempts": self.sample_retry_attempts,
                "source_tree_sha256": source_hash,
                "training_status": status,
                "training_log": "train.log",
            },
            training_access="full_outer_train",
            actual_training_population="full_joint",
            cross_class_usage="shared_joint_model",
            minority_sampling_semantics="bounded_joint_rejection",
        )
        return self.fit_result

    def _sample_minority(self, n: int, *, sampling_seed: int) -> SampleResult:
        assert self.schema is not None and self._repo is not None and self._dataset is not None and self._experiment is not None and self._work is not None
        chunk_index = 0
        sampling_attempts: list[dict[str, object]] = []

        def draw(size: int, _: int) -> pd.DataFrame:
            nonlocal chunk_index
            failures: list[dict[str, object]] = []
            frame: pd.DataFrame | None = None
            for attempt in range(self.sample_retry_attempts):
                seed = (
                    semantic_seed(sampling_seed, "tabbyflow_raw_chunk", chunk_index)
                    if attempt == 0
                    else semantic_seed(
                        sampling_seed,
                        "tabbyflow_raw_chunk",
                        chunk_index,
                        "retry",
                        attempt,
                    )
                )
                stem = (
                    f"raw-seed-{sampling_seed}-chunk-{chunk_index:04d}"
                    f"-attempt-{attempt:02d}"
                )
                output = self._work / f"{stem}.csv"
                try:
                    run_logged(
                        [
                            sys.executable, "main.py", "--dataname", self._dataset,
                            "--mode", "test", "--exp_name", self._experiment,
                            "--seed", str(seed), "--gpu", "0", "--no_wandb",
                            "--sample_only", "--num_samples_to_generate", str(size),
                            "--save_path", str(output),
                        ],
                        cwd=self._repo,
                        log_path=self._work / f"sample-{stem}.log",
                        stage="sample",
                    )
                    metadata = output.with_suffix(".metadata.json")
                    if not metadata.is_file():
                        raise AdapterFailure(
                            "sample", "SAMPLE_METADATA_MISSING", str(metadata)
                        )
                    payload = json.loads(metadata.read_text(encoding="utf-8"))
                    if (
                        int(payload.get("requested_samples", -1)) != size
                        or int(payload.get("ode_nan_rows_zero_filled", -1)) != 0
                    ):
                        raise AdapterFailure(
                            "sample",
                            "SAMPLE_METADATA_INVALID",
                            "TabbyFlow sampler status failed",
                        )
                    frame = _read_joint(output, self.schema, size)
                    sampling_attempts.append(
                        {
                            "raw_chunk": int(chunk_index),
                            "attempt": attempt,
                            "seed": int(seed),
                            "requested_rows": int(size),
                            "status": "SUCCESS",
                        }
                    )
                    break
                except AdapterFailure as error:
                    record = {
                        "raw_chunk": int(chunk_index),
                        "attempt": attempt,
                        "seed": int(seed),
                        "requested_rows": int(size),
                        "status": "FAILED",
                        "code": error.code,
                        "message": str(error),
                    }
                    failures.append(record)
                    sampling_attempts.append(record)
            if frame is None:
                raise AdapterFailure(
                    "sample",
                    "TABBYFLOW_SAMPLE_RETRIES_EXHAUSTED",
                    f"all {self.sample_retry_attempts} sampling attempts failed",
                    context={
                        "raw_chunk": int(chunk_index),
                        "requested_rows": int(size),
                        "attempts": failures,
                    },
                )
            chunk_index += 1
            return frame

        result = bounded_minority_rejection(
            draw,
            self.schema,
            n=n,
            sampling_seed=sampling_seed,
            raw_multiplier=self.max_raw_multiplier,
            raw_batch_size=self.raw_batch_size,
        )
        details = dict(result.details)
        details.update(
            {
                "raw_batch_size_semantics": "rejection_request_chunk_size",
                "rejection_request_chunk_size": self.raw_batch_size,
                "effective_upstream_sampling_batch_size": (
                    TABBYFLOW_UPSTREAM_SAMPLING_BATCH_SIZE
                ),
                "sample_retry_attempts": self.sample_retry_attempts,
                "sampling_attempts": sampling_attempts,
                "failed_sampling_attempts": sum(
                    item["status"] == "FAILED" for item in sampling_attempts
                ),
            }
        )
        return replace(result, details=details)
