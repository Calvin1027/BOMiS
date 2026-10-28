"""Official NCSN++, VE-SDE, self-paced loss, likelihood fine-tuning, and ODE sampling."""

import json

import numpy as np
import torch
import torch.nn.functional as F

from ..contracts import AdapterFailure
from .vendor import VendorJointAdapter


class ScalarLog:
    def __init__(self):
        self.rows = []

    def add_scalars(self, name, values, step):
        self.rows.append({"name": name, "step": int(step), **values})


class STaSyAdapter(VendorJointAdapter):
    name = "stasy"

    def _fit_codes(self, codes):
        import likelihood
        import losses
        from configs.shoppers import get_config
        from models import ncsnpp_tabular, utils
        from models.ema import ExponentialMovingAverage
        from sde_lib import VESDE

        del ncsnpp_tabular  # importing registers the official model
        config = get_config()
        config.device = torch.device(self.device)
        config.data.image_size = sum(self.codec.cardinalities)
        config.training.batch_size = self.batch_size
        config.model.num_scales = self.sampling_steps
        config.optim.warmup = int(self.options.get("warmup", config.optim.warmup))
        config.optim.lr *= float(self.options.get("lr_multiplier", 1.0))
        if "hidden_dims" in self.options:
            config.model.hidden_dims = tuple(self.options["hidden_dims"])
        if "nf" in self.options:
            config.model.nf = int(self.options["nf"])
        self.config = config
        # The upstream factory always enables DataParallel, which routes a CPU
        # model to CUDA whenever GPUs are visible. Keep the native architecture
        # on exactly the requested device.
        self.model = utils.get_model(config.model.name)(config).to(config.device)
        self.sde = VESDE(config.model.sigma_min, config.model.sigma_max, config.model.num_scales)
        # Upstream denoising indexes this table with GPU timesteps before moving it.
        self.sde.discrete_sigmas = self.sde.discrete_sigmas.to(self.device)
        train = torch.cat(
            [
                F.one_hot(torch.as_tensor(codes[:, j]), k)
                for j, k in enumerate(self.codec.cardinalities)
            ],
            dim=1,
        ).float()
        ema = ExponentialMovingAverage(self.model.parameters(), decay=config.model.ema_rate)
        optimizer = losses.get_optimizer(config, self.model.parameters())
        state = {"model": self.model, "optimizer": optimizer, "ema": ema, "step": 0}
        writer = ScalarLog()
        step_fn = losses.get_step_fn(
            self.sde,
            True,
            optimize_fn=losses.optimization_manager(config),
            reduce_mean=True,
            continuous=True,
            likelihood_weighting=False,
            spl=True,
            writer=writer,
            alpha0=config.model.alpha0,
            beta0=config.model.beta0,
        )
        for epoch in range(self.epochs):
            self._train_epoch(train, state, step_fn, "self_paced", epoch)
        ema.copy_to(self.model.parameters())
        ft_epochs = int(self.options.get("fine_tune_epochs", 50))
        if ft_epochs < 0:
            raise ValueError("fine_tune_epochs cannot be negative")
        if ft_epochs:
            likelihood_fn = likelihood.get_likelihood_fn(
                self.sde,
                lambda x: x,
                "Rademacher",
                rtol=1e-3,
                atol=1e-3,
            )

            def log_likelihood():
                parts = []
                self.model.eval()
                for start in range(0, len(train), self.batch_size):
                    ll, _, _ = likelihood_fn(
                        self.model,
                        train[start : start + self.batch_size].to(self.device),
                        eps_iters=int(self.options.get("eps_iters", 50)),
                    )
                    parts.append(ll.detach().cpu())
                values = torch.cat(parts)
                if not torch.isfinite(values).all():
                    raise AdapterFailure("fit", "NONFINITE_LIKELIHOOD", "STaSy fine-tuning")
                return values

            initial_ll = log_likelihood()
            chosen = initial_ll <= initial_ll.median()
            config.optim.lr = float(self.options.get("fine_tune_lr", 2e-7)) * float(
                self.options.get("lr_multiplier", 1.0)
            )
            config.optim.warmup = 0
            state["optimizer"] = losses.get_optimizer(config, self.model.parameters())
            # Keep model, EMA, and update count together; no hard-coded upstream seed reset.
            fine_step = losses.get_step_fn(
                self.sde,
                True,
                optimize_fn=losses.optimization_manager(config),
                reduce_mean=True,
                continuous=True,
                likelihood_weighting=False,
                spl=False,
            )
            for epoch in range(ft_epochs):
                if not chosen.any():
                    break
                self._train_epoch(train[chosen], state, fine_step, "likelihood_fine_tune", epoch)
                chosen = log_likelihood() - initial_ll < -0.1
        # Upstream evaluation installs the checkpoint's EMA even after fine-tuning.
        # Likelihood selection above intentionally uses the current fine-tune model.
        ema.copy_to(self.model.parameters())
        self.model.eval()
        self.save_torch(
            state_dict=self.model.state_dict(),
            ema=ema.state_dict(),
            optimizer=state["optimizer"].state_dict(),
            config=config.to_dict(),
            completed_updates=state["step"],
        )
        (self.work / "self_paced_scalars.json").write_text(json.dumps(writer.rows, indent=2))
        self.fit_details = {
            "parallelism": "single requested device; native architecture",
            "self_paced_learning": True,
            "fine_tune_epochs_requested": ft_epochs,
            "fine_tune_selection": "official likelihood rule, retained train only",
            "checkpoint_selection": "final prescribed stage; no test-based selection",
            "sampling_weights": "final-stage EMA, including fine-tuning updates",
            "representation": "categorical one-hot; softmax argmax inverse",
        }

    def _train_epoch(self, train, state, step_fn, stage, epoch):
        self.model.train()
        total = count = 0
        for batch in self.loader(train):
            loss = step_fn(state, batch.float().to(self.device))
            if (
                not torch.isfinite(loss)
                or not torch.stack([torch.isfinite(p).all() for p in self.model.parameters()]).all()
            ):
                raise AdapterFailure("fit", "NONFINITE_LOSS_OR_STATE", "STaSy")
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        self.logs.append(
            {
                "stage": stage,
                "epoch": epoch + 1,
                "loss": total / count,
                "updates": int(state["step"]),
                "lr": state["optimizer"].param_groups[0]["lr"],
            }
        )

    def _draw_codes(self, n):
        import sampling

        shape = (n, sum(self.codec.cardinalities))
        sampler = sampling.get_ode_sampler(
            self.sde,
            shape,
            lambda x: x,
            denoise=True,
            rtol=float(self.options.get("ode_tolerance", 1e-5)),
            atol=float(self.options.get("ode_tolerance", 1e-5)),
            eps=1e-5,
            device=self.device,
        )
        raw, _ = sampler(self.model, sampling_shape=shape)
        if not torch.isfinite(raw).all():
            raise AdapterFailure("sample", "NONFINITE_OUTPUT", "STaSy ODE")
        columns, offset = [], 0
        for size in self.codec.cardinalities:
            columns.append(raw[:, offset : offset + size].argmax(1).cpu().numpy())
            offset += size
        return np.column_stack(columns)
