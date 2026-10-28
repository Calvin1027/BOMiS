from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import AdapterFailure, DatasetSchema, FitResult, GeneratorAdapter, SampleResult
from ._external import isolate_source, run_logged, tree_sha256, write_text_new
from ._helpers import canonical_joint


def _apply_gradient_clip_patch(source_root: Path, max_norm: float, max_skips: int = 20) -> str:
    """Apply a narrow, auditable stability patch to the isolated trainer."""
    trainer = source_root / "scripts" / "train.py"
    source = trainer.read_text(encoding="utf-8")
    replacements = {
        """        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
""": """        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.skipped_nonfinite_gradient_steps = 0
        self.completed_optimizer_updates = 0
""",
        """        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss
""": f"""        if not torch.isfinite(loss):
            self.skipped_nonfinite_gradient_steps += 1
            return None, None
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm={max_norm!r})
        if torch.isfinite(grad_norm):
            self.optimizer.step()
            self.completed_optimizer_updates += 1
            return loss_multi, loss_gauss
        self.skipped_nonfinite_gradient_steps += 1
        self.optimizer.zero_grad()
        return None, None
""",
        """            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)
""": f"""            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)
            if batch_loss_multi is None:
                if self.skipped_nonfinite_gradient_steps > {max_skips!r}:
                    break
                continue
""",
        """            'status': 'SUCCESS',
            'completed_steps': int(steps),
""": """            'status': ('SUCCESS' if trainer.completed_optimizer_updates == steps else 'STABILITY_LIMIT'),
            'completed_steps': int(trainer.completed_optimizer_updates),
            'completed_optimizer_updates': int(trainer.completed_optimizer_updates),
""",
        """            'finite_loss': bool(np.isfinite(trainer.loss_history[['mloss', 'gloss', 'loss']].to_numpy()).all()),
""": """            'finite_loss': bool(np.isfinite(trainer.loss_history[['mloss', 'gloss', 'loss']].to_numpy()).all()),
            'skipped_nonfinite_gradient_steps': int(trainer.skipped_nonfinite_gradient_steps),
""",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise AdapterFailure(
                "fit", "INTEGRATION_PATCH_CONTEXT_MISMATCH", old.splitlines()[0].strip()
            )
        source = source.replace(old, new)
    trainer.write_text(source, encoding="utf-8")
    return tree_sha256(source_root)


def _config(
    *,
    experiment: Path,
    data: Path,
    numerical_count: int,
    train_steps: int,
    diffusion_steps: int,
    batch_size: int,
    learning_rate: float,
    model_seed: int,
    sampling_seed: int,
    n: int,
) -> str:
    return f'''seed = {model_seed}
parent_dir = "{experiment}"
real_data_path = "{data}"
model_type = "mlp"
num_numerical_features = {numerical_count}
device = "cuda:0"

[model_params]
num_classes = 2
is_y_cond = true
[model_params.rtdl_params]
d_layers = [256, 256]
dropout = 0.0
[diffusion_params]
num_timesteps = {diffusion_steps}
gaussian_loss_type = "mse"
scheduler = "cosine"
[train.main]
steps = {train_steps}
lr = {learning_rate}
weight_decay = 0.00001
batch_size = {batch_size}
seed = {model_seed}
[train.T]
seed = {model_seed}
normalization = "quantile"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"
[sample]
num_samples = {n}
batch_size = 10000
seed = {sampling_seed}
class_dist = [0.0, 1.0]
'''


class TabDDPMAdapter(GeneratorAdapter):
    """Official TabDDPM with the source-locked class-distribution/status patch."""

    name = "tabddpm"

    def __init__(
        self,
        source_dir: Path,
        *,
        train_steps: int = 10000,
        diffusion_steps: int = 1000,
        batch_size: int = 1024,
        learning_rate: float = 0.0002,
        learning_rate_backoff: list[float] | tuple[float, ...] = (),
        gradient_clip_norm: float | None = None,
        max_nonfinite_gradient_skips: int = 10,
    ) -> None:
        super().__init__()
        self.source_dir = source_dir
        self.train_steps = int(train_steps)
        self.diffusion_steps = int(diffusion_steps)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.learning_rate_backoff = tuple(float(value) for value in learning_rate_backoff)
        self.gradient_clip_norm = (
            None if gradient_clip_norm is None else float(gradient_clip_norm)
        )
        self.max_nonfinite_gradient_skips = int(max_nonfinite_gradient_skips)
        self._repo: Path | None = None
        self._work_dir: Path | None = None
        self._model_seed: int | None = None
        self._experiment: Path | None = None
        self._selected_learning_rate: float | None = None

    def fit(
        self,
        train: pd.DataFrame,
        schema: DatasetSchema,
        *,
        model_seed: int,
        work_dir: Path,
    ) -> FitResult:
        learning_rates = (self.learning_rate, *self.learning_rate_backoff)
        if (
            min(self.train_steps, self.diffusion_steps, self.batch_size) < 1
            or any(not 0.0 < rate < 1.0 for rate in learning_rates)
            or any(right >= left for left, right in zip(learning_rates, learning_rates[1:]))
            or len(set(learning_rates)) != len(learning_rates)
            or (self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0)
            or self.max_nonfinite_gradient_skips < 0
        ):
            raise AdapterFailure("configuration", "INVALID_BUDGET", "invalid TabDDPM budget")
        repo, source_hash = isolate_source(self.source_dir, work_dir / "_source")
        patched_source_hash = (
            _apply_gradient_clip_patch(repo, self.gradient_clip_norm, self.max_nonfinite_gradient_skips)
            if self.gradient_clip_norm is not None
            else source_hash
        )
        table = canonical_joint(train, schema)
        data = work_dir / "data"
        data.mkdir(parents=True, exist_ok=False)
        x_num = table.loc[:, list(schema.numerical_columns)].to_numpy(dtype=np.float64)
        x_cat = table.loc[:, list(schema.categorical_columns)].astype(str).to_numpy(dtype=object)
        y = table[schema.target].to_numpy(dtype=np.int64)
        for split in ("train", "val", "test"):
            np.save(data / f"X_num_{split}.npy", x_num)
            np.save(data / f"X_cat_{split}.npy", x_cat)
            np.save(data / f"y_{split}.npy", y)
        with (data / "info.json").open("x", encoding="utf-8") as handle:
            json.dump({"name": data.name, "task_type": "binclass", "n_classes": 2}, handle)
        attempts: list[dict[str, object]] = []
        selected: tuple[Path, float, dict[str, object], str] | None = None
        required = ("model.pt", "model_ema.pt", "loss.csv", "training_status.json")
        for attempt_index, learning_rate in enumerate(learning_rates):
            attempt_name = f"attempt-{attempt_index:02d}-lr-{learning_rate:.10g}"
            attempt_dir = work_dir / attempt_name
            experiment = attempt_dir / "experiment"
            experiment.mkdir(parents=True, exist_ok=False)
            config = attempt_dir / "train.toml"
            write_text_new(
                config,
                _config(
                    experiment=experiment,
                    data=data,
                    numerical_count=len(schema.numerical_columns),
                    train_steps=self.train_steps,
                    diffusion_steps=self.diffusion_steps,
                    batch_size=self.batch_size,
                    learning_rate=learning_rate,
                    model_seed=model_seed,
                    sampling_seed=model_seed,
                    n=1,
                ),
            )
            run_logged(
                [sys.executable, "scripts/pipeline.py", "--config", str(config), "--train"],
                cwd=repo,
                log_path=attempt_dir / "train.log",
                python_path=repo,
            )
            missing = [name for name in required if not (experiment / name).is_file()]
            if missing:
                raise AdapterFailure("fit", "TRAIN_ARTIFACT_MISSING", repr(missing))
            losses = pd.read_csv(experiment / "loss.csv")
            numeric = losses.select_dtypes(include=[np.number]).to_numpy(dtype=float)
            status = json.loads((experiment / "training_status.json").read_text(encoding="utf-8"))
            finite = bool(not losses.empty and numeric.size > 0 and np.isfinite(numeric).all())
            complete = bool(
                status.get("status") == "SUCCESS"
                and int(status.get("completed_steps", -1)) == self.train_steps
                and int(status.get("completed_optimizer_updates", -1)) == self.train_steps
                and status.get("finite_loss") is True
                and int(status.get("skipped_nonfinite_gradient_steps", 0))
                <= self.max_nonfinite_gradient_skips
            )
            attempt = {
                "attempt": attempt_index,
                "learning_rate": learning_rate,
                "finite_loss": finite,
                "complete": complete,
                "loss_artifact": f"{attempt_name}/experiment/loss.csv",
                "training_log": f"{attempt_name}/train.log",
                "training_status": status,
            }
            attempts.append(attempt)
            write_text_new(
                attempt_dir / "ATTEMPT.json",
                json.dumps(attempt, indent=2, sort_keys=True) + "\n",
            )
            if finite and complete:
                selected = experiment, learning_rate, status, attempt_name
                break
        if selected is None:
            raise AdapterFailure(
                "fit",
                "TRAINING_STABILITY_LIMIT_EXCEEDED",
                (
                    "no TabDDPM attempt satisfied finite-loss, completion, and "
                    f"nonfinite-gradient-skip limit {self.max_nonfinite_gradient_skips}; "
                    f"learning rates={learning_rates!r}"
                ),
                context={"attempts": attempts},
            )
        experiment, selected_learning_rate, status, attempt_name = selected
        self.schema, self._repo, self._work_dir, self._model_seed = (
            schema,
            repo,
            work_dir,
            model_seed,
        )
        self._experiment = experiment
        self._selected_learning_rate = selected_learning_rate
        self.fit_result = FitResult(
            model=self.name,
            model_seed=model_seed,
            n_outer_train=len(table),
            n_minority_train=int(y.sum()),
            sampling_mode="native_class_conditioning",
            loss_artifacts=(f"{attempt_name}/experiment/loss.csv",),
            checkpoint_artifacts=(
                f"{attempt_name}/experiment/model.pt",
                f"{attempt_name}/experiment/model_ema.pt",
            ),
            details={
                "algorithm": "yandex-research/tab-ddpm",
                "train_steps": self.train_steps,
                "diffusion_steps": self.diffusion_steps,
                "training_optimizer_updates": self.train_steps,
                "diffusion_timesteps": self.diffusion_steps,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "learning_rate_backoff": list(self.learning_rate_backoff),
                "selected_learning_rate": selected_learning_rate,
                "gradient_clip_norm": self.gradient_clip_norm,
                "max_nonfinite_gradient_skips": self.max_nonfinite_gradient_skips,
                "training_attempts": attempts,
                "class_dist": [0.0, 1.0],
                "upstream_split_arrays": {
                    "train": "outer_train",
                    "val": "outer_train_copy",
                    "test": "outer_train_copy",
                },
                "upstream_val_test_role": (
                    "interface_placeholders_loaded_and_transformed_only; "
                    "preprocessors_fit_on_train_and_optimizer_consumes_train_only"
                ),
                "source_tree_sha256": source_hash,
                "patched_source_tree_sha256": patched_source_hash,
                "training_status": status,
            },
            training_access="full_outer_train",
            actual_training_population="full_joint",
            cross_class_usage="shared_joint_model",
            minority_sampling_semantics="native_conditioning",
        )
        return self.fit_result

    def _sample_minority(self, n: int, *, sampling_seed: int) -> SampleResult:
        assert self.schema is not None and self._repo is not None
        assert self._work_dir is not None and self._model_seed is not None
        assert self._experiment is not None and self._selected_learning_rate is not None
        experiment, data = self._experiment, self._work_dir / "data"
        config = self._work_dir / f"sample-seed-{sampling_seed}.toml"
        write_text_new(
            config,
            _config(
                experiment=experiment,
                data=data,
                numerical_count=len(self.schema.numerical_columns),
                train_steps=self.train_steps,
                diffusion_steps=self.diffusion_steps,
                batch_size=self.batch_size,
                learning_rate=self._selected_learning_rate,
                model_seed=self._model_seed,
                sampling_seed=sampling_seed,
                n=n,
            ),
        )
        run_logged(
            [sys.executable, "scripts/pipeline.py", "--config", str(config), "--sample"],
            cwd=self._repo,
            log_path=self._work_dir / f"sample-seed-{sampling_seed}.log",
            python_path=self._repo,
            stage="sample",
        )
        x_num = np.load(experiment / "X_num_train.npy", allow_pickle=True)
        x_cat = np.load(experiment / "X_cat_train.npy", allow_pickle=True)
        y = np.load(experiment / "y_train.npy", allow_pickle=True).reshape(-1)
        if x_num.shape != (n, len(self.schema.numerical_columns)):
            raise AdapterFailure("sample", "RAW_SCHEMA_MISMATCH", "numeric array shape mismatch")
        if x_cat.shape != (n, len(self.schema.categorical_columns)) or y.shape != (n,):
            raise AdapterFailure("sample", "RAW_SCHEMA_MISMATCH", "categorical/target shape mismatch")
        if not np.array_equal(y.astype(float), np.ones(n, dtype=float)):
            raise AdapterFailure("sample", "CONDITION_FAILED", "TabDDPM returned nonminority")
        values: dict[str, object] = {}
        for index, column in enumerate(self.schema.numerical_columns):
            values[column] = x_num[:, index]
        for index, column in enumerate(self.schema.categorical_columns):
            values[column] = x_cat[:, index]
        frame = pd.DataFrame(values).loc[:, list(self.schema.feature_columns)]
        return SampleResult(
            frame=frame,
            sampling_seed=sampling_seed,
            raw_draw_count=n,
            minority_acceptance_rate=None,
            raw_draw_cap=None,
            rejection_used=False,
            details={
                "sampling_mode": "native_class_conditioning",
                "class_dist": [0.0, 1.0],
                "selected_learning_rate": self._selected_learning_rate,
            },
        )
