from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import AdapterFailure, DatasetSchema, FitResult, GeneratorAdapter, SampleResult
from ._helpers import (
    bounded_minority_rejection,
    canonical_joint,
    new_artifact_path,
    seed_everything,
)


class _CTGANFamilyAdapter(GeneratorAdapter):
    synthesizer_class: str

    def __init__(
        self,
        *,
        epochs: int = 300,
        batch_size: int = 500,
        max_raw_multiplier: int = 80,
        raw_batch_size: int = 10_000,
        require_cuda: bool = True,
        target_decode: str = "official_argmax",
    ) -> None:
        super().__init__()
        if epochs < 1 or batch_size < 2 or batch_size % 2:
            raise ValueError("epochs must be positive and batch_size must be positive and even")
        if max_raw_multiplier < 1 or raw_batch_size < 1:
            raise ValueError("rejection budgets must be positive")
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.max_raw_multiplier = int(max_raw_multiplier)
        self.raw_batch_size = int(raw_batch_size)
        self.require_cuda = bool(require_cuda)
        if target_decode not in {"official_argmax", "probabilistic_softmax"}:
            raise ValueError("target_decode must be official_argmax or probabilistic_softmax")
        if self.synthesizer_class != "TVAE" and target_decode != "official_argmax":
            raise ValueError("probabilistic target decoding is TVAE-only")
        self.target_decode = target_decode
        self._model: Any | None = None

    def _new_model(self) -> Any:
        from ctgan import CTGAN, TVAE

        cls = CTGAN if self.synthesizer_class == "CTGAN" else TVAE
        return cls(
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=False,
            enable_gpu=self.require_cuda,
        )

    def fit(
        self,
        train: pd.DataFrame,
        schema: DatasetSchema,
        *,
        model_seed: int,
        work_dir: Path,
    ) -> FitResult:
        joint = canonical_joint(train, schema)
        table = joint
        discrete = [*schema.categorical_columns, schema.target]
        sampling_mode = (
            "bounded_joint_rejection"
            if self.synthesizer_class == "TVAE"
            else "native_probability_tilt_plus_bounded_rejection"
        )
        seed_everything(model_seed, require_torch=True, require_cuda=self.require_cuda)
        model = self._new_model()
        if not hasattr(model, "set_random_state"):
            raise AdapterFailure("fit", "RNG_HOOK_MISSING", "CTGAN RNG hook is unavailable")
        model.set_random_state(model_seed)
        model.fit(table, discrete_columns=discrete)
        losses = getattr(model, "loss_values", None)
        if not isinstance(losses, pd.DataFrame) or losses.empty:
            raise AdapterFailure("fit", "LOSS_MISSING", f"{self.name} did not expose losses")
        if "Epoch" in losses:
            completed_epochs = int(losses["Epoch"].nunique())
        else:
            completed_epochs = len(losses)
        if completed_epochs != self.epochs:
            raise AdapterFailure(
                "fit", "TRAINING_INCOMPLETE",
                f"{self.name} reported {completed_epochs} of {self.epochs} epochs",
            )
        numeric_loss = losses.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if numeric_loss.size == 0 or not np.isfinite(numeric_loss).all():
            raise AdapterFailure("fit", "LOSS_NONFINITE", f"{self.name} loss is invalid")
        loss_path = new_artifact_path(work_dir, "training_loss.csv")
        checkpoint_path = new_artifact_path(work_dir, f"{self.name}.pt")
        losses.to_csv(loss_path, index=False)
        model.save(checkpoint_path)
        self.schema = schema
        self._model = model
        self.fit_result = FitResult(
            model=self.name,
            model_seed=model_seed,
            n_outer_train=len(joint),
            n_minority_train=int(joint[schema.target].sum()),
            sampling_mode=sampling_mode,
            loss_artifacts=(loss_path.name,),
            checkpoint_artifacts=(checkpoint_path.name,),
            details={
                "algorithm": f"ctgan.{self.synthesizer_class}",
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "discrete_columns": discrete,
                "completed_epochs": completed_epochs,
                "fit_population": (
                    "outer_train_joint"
                ),
                "target_decode": self.target_decode,
            },
            training_access="full_outer_train",
            actual_training_population="full_joint",
            cross_class_usage="shared_joint_model",
            minority_sampling_semantics=(
                "bounded_joint_rejection"
                if self.synthesizer_class == "TVAE"
                else "native_conditioning"
            ),
        )
        return self.fit_result

    def _draw(self, size: int, seed: int) -> pd.DataFrame:
        assert self._model is not None and self.schema is not None
        seed_everything(seed, require_torch=True, require_cuda=self.require_cuda)
        self._model.set_random_state(seed)
        if self.synthesizer_class == "CTGAN":
            return self._model.sample(
                size, condition_column=self.schema.target, condition_value=1
            ).loc[:, [*self.schema.feature_columns, self.schema.target]]
        if self.target_decode == "official_argmax":
            return self._model.sample(size).loc[
                :, [*self.schema.feature_columns, self.schema.target]
            ]
        return self._draw_tvae_probabilistic_target(size, seed)

    def _draw_tvae_probabilistic_target(self, size: int, seed: int) -> pd.DataFrame:
        """Sample TVAE while drawing Y from its decoder softmax.

        Upstream TVAE applies a column-wise argmax to every categorical decoder
        span.  For a rare binary target this can map every latent draw to Y=0,
        even when the learned decoder assigns nonzero probability to Y=1.  We
        retain the official prior, decoder, feature decoding, and inverse
        transformer, but sample the target category from its learned softmax.
        The resulting joint rows still pass through the common bounded
        rejection sampler; no rows from another generator are substituted.
        """
        import torch

        assert self._model is not None and self.schema is not None
        transformer = self._model.transformer
        columns = transformer._column_transform_info_list
        target_info = next(
            (item for item in columns if item.column_name == self.schema.target), None
        )
        if target_info is None or target_info.column_type != "discrete":
            raise AdapterFailure(
                "sample", "TVAE_TARGET_SPAN_MISSING", self.schema.target
            )
        target_start = sum(
            item.output_dimensions
            for item in columns
            if item.column_name != self.schema.target
        )
        # canonical_joint always places the target last. Fail closed if an
        # upstream transformer changes that ordering assumption.
        if columns[-1].column_name != self.schema.target:
            raise AdapterFailure(
                "sample", "TVAE_TARGET_NOT_LAST", self.schema.target
            )
        target_end = target_start + target_info.output_dimensions
        if target_info.output_dimensions != 2:
            raise AdapterFailure(
                "sample", "TVAE_TARGET_NOT_BINARY", str(target_info.output_dimensions)
            )

        seed_everything(seed, require_torch=True, require_cuda=self.require_cuda)
        self._model.decoder.eval()
        batch_size = int(self._model.batch_size)
        pieces: list[np.ndarray] = []
        sigmas = None
        remaining = int(size)
        while remaining:
            batch = min(batch_size, remaining)
            noise = torch.randn(
                batch,
                int(self._model.embedding_dim),
                device=self._model._device,
            )
            logits, sigmas = self._model.decoder(noise)
            decoded = torch.tanh(logits)
            probabilities = torch.softmax(logits[:, target_start:target_end], dim=1)
            selected = torch.multinomial(probabilities, num_samples=1)
            decoded[:, target_start:target_end] = 0.0
            decoded[:, target_start:target_end].scatter_(1, selected, 1.0)
            pieces.append(decoded.detach().cpu().numpy())
            remaining -= batch
        assert sigmas is not None
        matrix = np.concatenate(pieces, axis=0)
        frame = transformer.inverse_transform(
            matrix, sigmas.detach().cpu().numpy()
        )
        if not isinstance(frame, pd.DataFrame):
            raise AdapterFailure(
                "sample", "RAW_NOT_DATAFRAME", "TVAE inverse transform returned an array"
            )
        return frame.loc[:, [*self.schema.feature_columns, self.schema.target]]

    def _sample_minority(self, n: int, *, sampling_seed: int) -> SampleResult:
        assert self.schema is not None
        result = bounded_minority_rejection(
            self._draw,
            self.schema,
            n=n,
            sampling_seed=sampling_seed,
            raw_multiplier=self.max_raw_multiplier,
            raw_batch_size=self.raw_batch_size,
        )
        details = dict(result.details)
        if self.synthesizer_class == "CTGAN":
            details["native_condition"] = {"column": self.schema.target, "value": 1}
        elif self.target_decode == "probabilistic_softmax":
            details.update(
                {
                    "target_decode": "probabilistic_softmax",
                    "target_decode_scope": "target_span_only",
                    "feature_decode": "official_tvae",
                }
            )
        return SampleResult(
            frame=result.frame,
            sampling_seed=result.sampling_seed,
            raw_draw_count=result.raw_draw_count,
            minority_acceptance_rate=result.minority_acceptance_rate,
            raw_draw_cap=result.raw_draw_cap,
            rejection_used=result.rejection_used,
            accepted_minority_count=result.accepted_minority_count,
            details=details,
            decoded_y1_rows=result.decoded_y1_rows,
            valid_y1_rows=result.valid_y1_rows,
            invalid_y1_rows=result.invalid_y1_rows,
            invalid_target_rows=result.invalid_target_rows,
            rejected_y0_rows=result.rejected_y0_rows,
            y1_rate=result.y1_rate,
            valid_given_y1_rate=result.valid_given_y1_rate,
            overall_acceptance_rate=result.overall_acceptance_rate,
            selected_output_rows=result.selected_output_rows,
            termination_reason=result.termination_reason,
        )


class CTGANAdapter(_CTGANFamilyAdapter):
    name = "ctgan"
    synthesizer_class = "CTGAN"


class TVAEAdapter(_CTGANFamilyAdapter):
    name = "tvae"
    synthesizer_class = "TVAE"
