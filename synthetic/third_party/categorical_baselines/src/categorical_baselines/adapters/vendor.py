"""Shared categorical transport, not a replacement generative algorithm."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import AdapterFailure, FitResult, GeneratorAdapter
from ..data import JointCodec, require_categorical
from ._helpers import (
    bounded_minority_rejection,
    canonical_joint,
    new_artifact_path,
    seed_everything,
)


def add_source(name, source_root=None):
    root = Path(source_root) if source_root else Path(__file__).resolve().parents[3] / "third_party"
    repo = root / name
    if not repo.is_dir():
        raise AdapterFailure("configuration", "SOURCE_MISSING", str(repo))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def finite_step(loss, optimizer, model, *, clip=None):
    import torch

    if loss.numel() != 1 or not torch.isfinite(loss):
        raise AdapterFailure("fit", "NONFINITE_LOSS", str(loss.detach().cpu()))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters or not torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all():
        raise AdapterFailure("fit", "NONFINITE_GRADIENT", "training stopped")
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(parameters, clip)
    optimizer.step()
    if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
        raise AdapterFailure("fit", "NONFINITE_PARAMETERS", "optimizer produced non-finite weights")


class VendorJointAdapter(GeneratorAdapter):
    def __init__(
        self,
        *,
        source_root=None,
        device="cpu",
        epochs=300,
        batch_size=256,
        sampling_steps=50,
        raw_batch_size=1024,
        max_raw_multiplier=200,
        **options,
    ):
        super().__init__()
        if min(epochs, batch_size, sampling_steps, raw_batch_size, max_raw_multiplier) < 1:
            raise ValueError("budgets must be positive")
        if self.name in {"tabsyn", "tabdiff", "stasy"} and sampling_steps < 2:
            raise ValueError("diffusion sampling_steps must be at least two")
        self.source_root, self.device = source_root, device
        self.epochs, self.batch_size, self.sampling_steps = (
            int(epochs),
            int(batch_size),
            int(sampling_steps),
        )
        self.raw_batch_size, self.max_raw_multiplier = int(raw_batch_size), int(max_raw_multiplier)
        self.options = options
        self.logs = []

    def fit(self, train, schema, *, model_seed, work_dir):
        require_categorical(schema)
        schema.binary_target(train)
        seed_everything(
            model_seed,
            require_torch=self.name != "arf",
            require_cuda=self.device.startswith("cuda"),
        )
        self.repo = add_source("arfpy" if self.name == "arf" else self.name, self.source_root)
        self.schema, self.codec = schema, JointCodec.fit(train, schema)
        self.work = Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        new_artifact_path(self.work, "checkpoint.pt")
        self.logs = []
        codes = self.codec.encode(canonical_joint(train, schema))
        self.model_seed = model_seed
        self._fit_codes(codes)
        if not self.logs or not np.isfinite([row["loss"] for row in self.logs]).all():
            raise AdapterFailure("fit", "LOSS_MISSING_OR_NONFINITE", self.name)
        pd.DataFrame(self.logs).to_csv(
            new_artifact_path(self.work, "training_loss.csv"), index=False
        )
        codec_path = new_artifact_path(self.work, "codec.json")
        codec_path.write_text(json.dumps(self.codec.__dict__, indent=2) + "\n")
        source_hash = hashlib.sha256()
        for path in sorted(self.repo.rglob("*.py")):
            source_hash.update(str(path.relative_to(self.repo)).encode())
            source_hash.update(path.read_bytes())
        details = {
            "upstream_source": str(self.repo),
            "upstream_python_sha256": source_hash.hexdigest(),
            "zero_numerical_features": True,
            "epochs": self.epochs,
            "epoch_semantics": "complete shuffled passes over retained outer-training rows",
            "batch_size": self.batch_size,
            "sampling_steps": self.sampling_steps,
            "options": self.options,
            "codec": "train-only nominal symbols, target last",
            "native_conditioning": False,
            **getattr(self, "fit_details", {}),
        }
        self.fit_result = FitResult(
            self.name,
            model_seed,
            len(train),
            int(schema.binary_target(train).sum()),
            "bounded_joint_rejection",
            loss_artifacts=("training_loss.csv",),
            checkpoint_artifacts=("checkpoint.pt", "codec.json"),
            details=details,
            training_access="full_outer_train",
            actual_training_population="full_joint",
            cross_class_usage="shared_joint_model",
            minority_sampling_semantics="bounded_joint_rejection",
        )
        return self.fit_result

    def _sample_minority(self, n, *, sampling_seed):
        def draw(size, seed):
            seed_everything(seed, require_torch=self.name != "arf")
            return self.codec.decode(self._draw_codes(size))

        return bounded_minority_rejection(
            draw,
            self.schema,
            n=n,
            sampling_seed=sampling_seed,
            raw_multiplier=self.max_raw_multiplier,
            raw_batch_size=self.raw_batch_size,
            raw_cap=getattr(self, "raw_draw_cap", None),
        )

    def loader(self, codes):
        import torch

        return torch.utils.data.DataLoader(
            torch.as_tensor(codes),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

    def save_torch(self, **state):
        import torch

        torch.save(
            {
                "model": self.name,
                "codec": self.codec.__dict__,
                "options": self.options,
                "model_seed": self.model_seed,
                **state,
            },
            self.work / "checkpoint.pt",
        )
