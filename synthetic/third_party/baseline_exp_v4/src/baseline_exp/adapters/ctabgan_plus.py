from __future__ import annotations

import pickle
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import AdapterFailure, DatasetSchema, FitResult, GeneratorAdapter, SampleResult
from ._external import isolate_source
from ._external import tree_sha256
from ._helpers import (
    bounded_minority_rejection,
    canonical_joint,
    new_artifact_path,
    seed_everything,
)


def _general_columns(numerical_columns: tuple[str, ...]) -> list[str]:
    """Route every numerical feature through the patched finite min-max scaler.

    The isolated upstream scaler patch maps a constant column with denominator
    one during transform and restores it with its zero range during inverse
    transform.  Excluding constants instead sends them through the mixture
    decoder, whose range filter can reject every generated row.
    """
    return list(numerical_columns)


def _apply_sampling_patch(source_root: Path) -> str:
    """Finite constant scaling and exactly counted raw draws, without nested retries."""
    transformer = source_root / "model/synthesizer/transformer.py"
    text = transformer.read_text(encoding="utf-8")
    old = "current = (current - (info['min'])) / (info['max'] - info['min'])"
    new = "current = (current - info['min']) / (info['max'] - info['min'] if info['max'] != info['min'] else 1.0)"
    if text.count(old) != 1:
        raise AdapterFailure("fit", "INTEGRATION_PATCH_CONTEXT_MISMATCH", "constant scaler")
    transformer.write_text(text.replace(old, new), encoding="utf-8")
    path = source_root / "model/synthesizer/ctabgan_synthesizer.py"
    text = path.read_text(encoding="utf-8")
    start = text.index("    @torch.no_grad()\n    def sample(self, n, seed=0):")
    # Preserve the official generator, conditioning distribution, activation and decoder.
    # Only batch sizing and retry ownership change. The adapter owns the finite budget.
    method = '''    @torch.no_grad()
    def sample(self, n, seed=0):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.generator.eval()
        data = []
        for offset in range(0, n, 500):
            size = min(500, n - offset)
            noisez = torch.randn(size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample(size)
            c = torch.from_numpy(condvec).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez = noisez.view(size, self.random_dim + self.cond_generator.n_opt, 1, 1)
            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            data.append(apply_activate(faket, self.transformer.output_info).detach().cpu().numpy())
        result, _ = self.transformer.inverse_transform(np.concatenate(data, axis=0))
        return result
'''
    if "    def " in text[start + len("    @torch.no_grad()\n    def sample"):]:
        raise AdapterFailure("fit", "INTEGRATION_PATCH_CONTEXT_MISMATCH", "sample is not last method")
    path.write_text(text[:start] + method, encoding="utf-8")
    return tree_sha256(source_root)


def _apply_integration_patch(source_root: Path) -> str:
    """Apply two narrow, fail-closed compatibility fixes to an isolated source tree."""
    synthesizer_path = source_root / "model" / "synthesizer" / "ctabgan_synthesizer.py"
    source_text = synthesizer_path.read_text(encoding="utf-8")
    replacements = {
        """def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1)).view(val.size(0), 1)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so)*low + (torch.sin(val*omega)/so) * high
""": """def slerp(val, low, high):
    # Stable spherical interpolation for the official gradient penalty.
    eps = torch.finfo(low.dtype).eps
    low_norm = low/torch.norm(low, dim=1, keepdim=True).clamp_min(eps)
    high_norm = high/torch.norm(high, dim=1, keepdim=True).clamp_min(eps)
    cosine = (low_norm*high_norm).sum(1).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(cosine).view(val.size(0), 1)
    so = torch.sin(omega)
    spherical = (torch.sin((1.0-val)*omega)/so.clamp_min(eps))*low + (torch.sin(val*omega)/so.clamp_min(eps)) * high
    linear = (1.0-val)*low + val*high
    res = torch.where(so.abs() <= eps, linear, spherical)
""",
        "sides = [4, 8, 16, 24, 32]": "sides = [4, 8, 16, 24, 32, 64]",
    }
    for old, new in replacements.items():
        expected = 2 if old.startswith("sides =") else 1
        if source_text.count(old) != expected:
            raise AdapterFailure(
                "fit", "INTEGRATION_PATCH_CONTEXT_MISMATCH", old.splitlines()[0]
            )
        source_text = source_text.replace(old, new)
    synthesizer_path.write_text(source_text, encoding="utf-8")
    return tree_sha256(source_root)


class CTABGANPlusAdapter(GeneratorAdapter):
    name = "ctabgan_plus"

    def __init__(
        self,
        source_dir: Path,
        *,
        epochs: int = 150,
        batch_size: int = 500,
        device: str = "cuda",
        max_raw_multiplier: int = 80,
        raw_batch_size: int = 10000,
    ) -> None:
        super().__init__()
        self.source_dir = source_dir
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.device = device
        self.max_raw_multiplier = int(max_raw_multiplier)
        self.raw_batch_size = int(raw_batch_size)
        self._model: Any | None = None

    def fit(
        self,
        train: pd.DataFrame,
        schema: DatasetSchema,
        *,
        model_seed: int,
        work_dir: Path,
    ) -> FitResult:
        if (
            self.epochs < 1
            or self.batch_size < 2
            or self.max_raw_multiplier < 1
            or self.raw_batch_size < 1
        ):
            raise AdapterFailure("configuration", "INVALID_BUDGET", "invalid CTAB-GAN+ budget")
        isolated, source_hash = isolate_source(self.source_dir, work_dir / "_source")
        _apply_integration_patch(isolated)
        patched_source_hash = _apply_sampling_patch(isolated)
        joint = canonical_joint(train, schema)
        table = joint
        minority = joint[schema.target].to_numpy(dtype=bool)
        general_columns = _general_columns(schema.numerical_columns)
        constant_numerical_columns = [
            c for c in schema.numerical_columns if table[c].nunique(dropna=False) == 1
        ]
        seed_everything(model_seed, require_torch=True, require_cuda=self.device.startswith("cuda"))
        sys.path.insert(0, str(isolated))
        try:
            from model.ctabgan import CTABGAN
        finally:
            sys.path.pop(0)
        model = CTABGAN(
            df=table,
            test_ratio=0.0,
            categorical_columns=[*schema.categorical_columns, schema.target],
            log_columns=[],
            mixed_columns={},
            general_columns=general_columns,
            non_categorical_columns=[],
            integer_columns=[
                column
                for column in schema.numerical_columns
                if pd.api.types.is_integer_dtype(pd.api.types.pandas_dtype(schema.dtypes[column]))
            ],
            problem_type={"Classification": schema.target},
            batch_size=self.batch_size,
            epochs=self.epochs,
            device=self.device,
        )
        train_log = new_artifact_path(work_dir, "train.log")
        with train_log.open("x", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
            model.fit()
        checkpoint = new_artifact_path(work_dir, "ctabgan_plus.pkl")
        with checkpoint.open("xb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # The official implementation does not expose an epoch-wise loss history.
        self.schema = schema
        self._model = model
        self.fit_result = FitResult(
            model=self.name,
            model_seed=model_seed,
            n_outer_train=len(joint),
            n_minority_train=int(minority.sum()),
            sampling_mode="bounded_joint_rejection",
            checkpoint_artifacts=(checkpoint.name,),
            details={
                "algorithm": "Team-TUD/CTAB-GAN-Plus",
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "max_raw_multiplier": self.max_raw_multiplier,
                "raw_batch_size": self.raw_batch_size,
                "problem_type": {"Classification": schema.target},
                "general_columns": general_columns,
                "constant_numerical_columns": constant_numerical_columns,
                "fit_population": "outer_train_joint",
                "target_is_generated_variable": True,
                "source_tree_sha256": source_hash,
                "patched_source_tree_sha256": patched_source_hash,
                "integration_patch_scope": "stable_slerp_64x64_finite_constant_general_scaler_exact_raw_draws",
                "loss_unavailable_reason": "official implementation exposes no loss history",
                "training_log": train_log.name,
            },
            training_access="full_outer_train",
            actual_training_population="full_joint",
            cross_class_usage="shared_joint_model",
            minority_sampling_semantics="bounded_joint_rejection",
        )
        return self.fit_result

    def _draw(self, size: int, seed: int) -> pd.DataFrame:
        assert self._model is not None and self.schema is not None
        seed_everything(seed, require_torch=True, require_cuda=self.device.startswith("cuda"))
        frame = self._model.generate_samples(size, seed=seed)
        expected = [*self.schema.feature_columns, self.schema.target]
        if set(frame.columns) != set(expected):
            raise AdapterFailure("sample", "RAW_SCHEMA_MISMATCH", "CTAB-GAN+ schema mismatch")
        return frame.loc[:, expected]

    def _sample_minority(self, n: int, *, sampling_seed: int) -> SampleResult:
        assert self.schema is not None
        batch_index = 0

        def draw(size: int, _: int) -> pd.DataFrame:
            nonlocal batch_index
            draw_seed = int(
                np.random.SeedSequence([sampling_seed, batch_index]).generate_state(1)[0]
            )
            frame = self._draw(size, draw_seed)
            batch_index += 1
            return frame

        return bounded_minority_rejection(
            draw,
            self.schema,
            n=n,
            sampling_seed=sampling_seed,
            raw_multiplier=self.max_raw_multiplier,
            raw_batch_size=self.raw_batch_size,
        )
