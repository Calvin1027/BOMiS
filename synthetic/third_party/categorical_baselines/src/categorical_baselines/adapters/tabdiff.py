import copy

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import numpy as np
import torch

from .vendor import VendorJointAdapter, finite_step


class TabDiffAdapter(VendorJointAdapter):
    name = "tabdiff"

    def _fit_codes(self, codes):
        from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
        from tabdiff.modules.main_modules import Model, UniModMLP

        with (self.repo / "tabdiff/configs/tabdiff_configs.toml").open("rb") as stream:
            config = tomllib.load(stream)
        network = config["unimodmlp_params"]
        network.update(d_numerical=0, categories=[c + 1 for c in self.codec.cardinalities])
        for key in ("dim_t", "num_layers", "factor", "d_token"):
            if key in self.options:
                network[key] = int(self.options[key])
        diffusion = config["diffusion_params"]
        diffusion.update(
            num_timesteps=self.sampling_steps,
            scheduler="power_mean_per_column",
            cat_scheduler="log_linear_per_column",
        )
        denoiser = Model(UniModMLP(**network), **diffusion["edm_params"]).to(self.device)
        self.model = UnifiedCtimeDiffusion(
            np.asarray(self.codec.cardinalities),
            0,
            denoiser,
            None,
            **diffusion,
            device=torch.device(self.device),
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001 * float(self.options.get("lr_multiplier", 1.0)),
            weight_decay=0,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=50)
        ema = copy.deepcopy(self.model)
        decay = 0.997
        for epoch in range(self.epochs):
            self.model.train()
            total = count = 0
            for batch in self.loader(codes):
                categorical_loss, _ = self.model.mixed_loss(batch.float().to(self.device))
                # The official diffusion already omits the absent numerical branch.
                finite_step(categorical_loss, optimizer, self.model)
                total += float(categorical_loss.detach()) * len(batch)
                count += len(batch)
            scheduler.step(total / count)
            # The pinned official Trainer updates EMA once per complete epoch.
            with torch.no_grad():
                for target, source in zip(ema.parameters(), self.model.parameters(), strict=True):
                    target.lerp_(source, 1 - decay)
            self.logs.append(
                {
                    "stage": "masked_diffusion",
                    "epoch": epoch + 1,
                    "loss": total / count,
                    "lr": optimizer.param_groups[0]["lr"],
                    "updates": (epoch + 1)
                    * ((len(codes) + self.batch_size - 1) // self.batch_size),
                }
            )
        self.save_torch(
            state_dict=self.model.state_dict(),
            ema=ema.state_dict(),
            config=config,
            optimizer=optimizer.state_dict(),
        )
        self.model = ema.eval()
        self.fit_details = {
            "ema_decay": decay,
            "ema_update_interval": "epoch",
            "learnable_categorical_schedule": True,
            "sampling": "official joint masked diffusion; no classifier guidance",
        }

    def _draw_codes(self, n):
        with torch.no_grad():
            return self.model.sample(n).numpy()
