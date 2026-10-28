from __future__ import annotations

import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import AdapterFailure, DatasetSchema, FitResult, GeneratorAdapter, SampleResult
from ..provenance import sha256_file
from ._external import isolate_source, tree_sha256
from ._helpers import canonical_joint, new_artifact_path, seed_everything


class ForestFlowAdapter(GeneratorAdapter):
    name = "forest_flow"

    def __init__(
        self,
        source_dir: Path,
        patch_file: Path,
        *,
        n_t: int = 50,
        duplicate_k: int = 100,
        n_batch: int = 1,
        n_jobs: int = 8,
        n_estimators: int = 100,
        max_depth: int = 7,
        gpu_hist: bool = False,
        tree_method: str = "hist",
        nthread: int = 4,
    ) -> None:
        super().__init__()
        self.source_dir = source_dir
        self.patch_file = patch_file
        self.n_t = int(n_t)
        self.duplicate_k = int(duplicate_k)
        self.n_batch = int(n_batch)
        self.n_jobs = int(n_jobs)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.gpu_hist = bool(gpu_hist)
        self.tree_method = str(tree_method)
        self.nthread = int(nthread)
        if self.gpu_hist and self.n_jobs != 1:
            raise AdapterFailure(
                "configuration",
                "GPU_PARALLELISM_UNSAFE",
                "Forest-Flow gpu_hist requires n_jobs=1 to avoid concurrent XGBoost workers on one GPU",
            )
        self._model: Any | None = None

    def fit(
        self,
        train: pd.DataFrame,
        schema: DatasetSchema,
        *,
        model_seed: int,
        work_dir: Path,
    ) -> FitResult:
        isolated, source_hash = isolate_source(self.source_dir, work_dir / "_source")
        if not self.patch_file.is_file():
            raise AdapterFailure("configuration", "PATCH_MISSING", str(self.patch_file))
        patch_sha256 = sha256_file(self.patch_file)
        for arguments in (
            ["git", "apply", "--check", "--ignore-space-change", str(self.patch_file)],
            ["git", "apply", "--ignore-space-change", str(self.patch_file)],
        ):
            completed = subprocess.run(arguments, cwd=isolated, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise AdapterFailure("fit", "SAMPLING_PATCH_FAILED", completed.stderr.strip())
        preserved_patch = new_artifact_path(work_dir, "forest_flow_target_label.patch")
        shutil.copyfile(self.patch_file, preserved_patch)
        implementation = isolated / "Python-Package/base-ForestDiffusion/ForestDiffusion/diffusion_with_trees_class.py"
        source = implementation.read_text()
        edits = {
            "    data_iterator = xgb.QuantileDMatrix(it)":
            "    print(f'FOREST_BEGIN pid={os.getpid()} t={t} rows={sum(len(x) for x in X1_splitted)} cuda={self.gpu_hist}', flush=True)\n    data_iterator = xgb.QuantileDMatrix(it, nthread=self.xgboost_kwargs.get('nthread', 4))\n    print(f'FOREST_MATRIX_READY pid={os.getpid()} t={t}', flush=True)",
            "    out = xgb.train(xgb_dict, data_iterator, num_boost_round=self.n_estimators)":
            "    out = xgb.train(xgb_dict, data_iterator, num_boost_round=self.n_estimators)\n    config = json.loads(out.save_config())\n    device = config['learner']['generic_param'].get('device', '')\n    if self.gpu_hist and not device.startswith('cuda'):\n      raise RuntimeError(f'ForestFlow silently failed to use CUDA: {device}')\n    print(f'FOREST_DONE pid={os.getpid()} t={t} device={device}', flush=True)",
        }
        for old, new in edits.items():
            if source.count(old) != 1:
                raise AdapterFailure("fit", "INTEGRATION_PATCH_CONTEXT_MISMATCH", old)
            source = source.replace(old, new)
        implementation.write_text("import os\nimport json\n" + source)
        patched_source_hash = tree_sha256(isolated)
        package_root = isolated / "Python-Package" / "base-ForestDiffusion"
        if not package_root.is_dir():
            raise AdapterFailure("configuration", "SOURCE_LAYOUT", "ForestDiffusion package missing")
        table = canonical_joint(train, schema)
        x = table.loc[:, list(schema.feature_columns)].copy()
        for column in schema.categorical_columns:
            mapping = {
                str(level): index for index, level in enumerate(schema.category_domains[column])
            }
            x[column] = x[column].astype(str).map(mapping)
            if x[column].isna().any():
                raise AdapterFailure("fit", "CATEGORY_ENCODING", f"cannot encode {column!r}")
        seed_everything(model_seed, require_torch=self.gpu_hist, require_cuda=self.gpu_hist)
        # Loky workers unpickle bound ForestDiffusion methods in fresh interpreters, so the
        # isolated package must remain importable for the fitted model's entire lifetime.
        package_text = str(package_root)
        if package_text not in sys.path:
            sys.path.insert(0, package_text)
        from ForestDiffusion import ForestDiffusionModel
        cat_indices = [x.columns.get_loc(column) for column in schema.categorical_columns]
        int_indices = [
            x.columns.get_loc(column)
            for column in schema.numerical_columns
            if pd.api.types.is_integer_dtype(pd.api.types.pandas_dtype(schema.dtypes[column]))
        ]
        model = ForestDiffusionModel(
            x.to_numpy(dtype=float),
            label_y=table[schema.target].to_numpy(dtype=np.int64),
            n_t=self.n_t,
            duplicate_K=self.duplicate_k,
            n_batch=self.n_batch,
            bin_indexes=[],
            cat_indexes=cat_indices,
            int_indexes=int_indices,
            diffusion_type="flow",
            seed=model_seed,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            gpu_hist=self.gpu_hist,
            tree_method=self.tree_method,
            nthread=self.nthread,
        )
        checkpoint = new_artifact_path(work_dir, "forest_flow.pkl")
        with checkpoint.open("xb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.schema = schema
        self._model = model
        self.fit_result = FitResult(
            model=self.name,
            model_seed=model_seed,
            n_outer_train=len(table),
            n_minority_train=int(table[schema.target].sum()),
            sampling_mode="direct_class_branch",
            checkpoint_artifacts=(checkpoint.name,),
            details={
                "algorithm": "SamsungSAILMontreal/ForestDiffusion Forest-Flow",
                "diffusion_type": "flow",
                "n_t": self.n_t,
                "duplicate_k": self.duplicate_k,
                "n_batch": self.n_batch,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "compute_backend": "xgboost_cuda" if self.gpu_hist else "xgboost_cpu_parallel",
                "gpu_hist": self.gpu_hist,
                "tree_method": self.tree_method,
                "n_jobs": self.n_jobs,
                "nthread": self.nthread,
                "source_tree_sha256": source_hash,
                "patched_source_tree_sha256": patched_source_hash,
                "sampling_patch_sha256": patch_sha256,
                "sampling_patch_artifact": preserved_patch.name,
                "loss_unavailable_reason": "official implementation exposes no common loss table",
                "label_handling": "class_specific",
                "cross_class_parameter_sharing": False,
                "target_label": 1,
            },
        )
        return self.fit_result

    def _sample_minority(self, n: int, *, sampling_seed: int) -> SampleResult:
        assert self._model is not None and self.schema is not None
        seed_everything(sampling_seed)
        generated = np.asarray(self._model.generate(batch_size=n, target_label=1))
        expected_width = len(self.schema.feature_columns) + 1
        if generated.shape != (n, expected_width):
            raise AdapterFailure("sample", "RAW_SCHEMA_MISMATCH", "Forest-Flow joint output shape mismatch")
        labels = generated[:, -1]
        if not np.isfinite(labels.astype(float)).all() or not np.array_equal(
            np.rint(labels.astype(float)).astype(np.int64), np.ones(n, dtype=np.int64)
        ):
            raise AdapterFailure("sample", "CONDITION_FAILED", "Forest-Flow returned nonminority")
        frame = pd.DataFrame(generated[:, :-1], columns=self.schema.feature_columns)
        for column in self.schema.categorical_columns:
            order = self.schema.category_domains[column]
            values = frame[column].to_numpy(dtype=float)
            codes = np.rint(values).astype(int)
            if not np.isclose(values, codes).all() or np.any((codes < 0) | (codes >= len(order))):
                raise AdapterFailure("sample", "UNKNOWN_CATEGORY", f"invalid code for {column!r}")
            frame[column] = np.asarray(order, dtype=object)[codes]
        return SampleResult(
            frame=frame,
            sampling_seed=sampling_seed,
            raw_draw_count=n,
            minority_acceptance_rate=None,
            raw_draw_cap=None,
            rejection_used=False,
            details={
                "sampling_mode": "direct_class_branch",
                "target_label": 1,
                "label_handling": "class_specific",
                "cross_class_parameter_sharing": False,
            },
        )
