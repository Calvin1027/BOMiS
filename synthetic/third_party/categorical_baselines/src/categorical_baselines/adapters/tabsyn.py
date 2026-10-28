"""V4's full-joint TabSyn route with official VAE/EDM and empty numeric loss removed."""

import numpy as np
import torch
import torch.nn.functional as F

from ..contracts import AdapterFailure
from .vendor import VendorJointAdapter, finite_step


class TabSynAdapter(VendorJointAdapter):
    name = "tabsyn"

    def _fit_codes(self, codes):
        from tabsyn.model import MLPDiffusion, Model
        from tabsyn.vae.model import Decoder_model, Encoder_model, Model_VAE

        layers = int(self.options.get("num_layers", 2))
        self.token_dim = int(self.options.get("d_token", 4))
        factor = int(self.options.get("factor", 32))
        args = (layers, 0, self.codec.cardinalities, self.token_dim)
        vae = Model_VAE(*args, n_head=1, factor=factor).to(self.device)
        multiplier = float(self.options.get("lr_multiplier", 1.0))
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3 * multiplier, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=10)
        beta, minimum_beta = 0.01, 1e-5
        best, patience = float("inf"), 0
        vae_epochs = int(self.options.get("vae_epochs", 4000))
        if vae_epochs < 1:
            raise ValueError("vae_epochs must be positive")
        best_state = None
        for epoch in range(vae_epochs):
            vae.train()
            total = total_ce = count = 0
            for batch in self.loader(codes):
                batch = batch.long().to(self.device)
                empty_num = torch.empty((len(batch), 0), device=self.device)
                _, recon, mu, logvar = vae(empty_num, batch)
                ce = sum(
                    F.cross_entropy(logits, batch[:, j]) for j, logits in enumerate(recon)
                ) / len(recon)
                kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
                # The numerical reconstruction term is exactly zero for d_num=0.
                loss = ce + beta * kl
                finite_step(loss, optimizer, vae)
                total += float(loss.detach()) * len(batch)
                total_ce += float(ce.detach()) * len(batch)
                count += len(batch)
            value, ce_value = total / count, total_ce / count
            scheduler.step(ce_value)
            self.logs.append(
                {
                    "stage": "vae",
                    "epoch": epoch + 1,
                    "loss": value,
                    "ce": ce_value,
                    "beta": beta,
                    "lr": optimizer.param_groups[0]["lr"],
                    "updates": (epoch + 1)
                    * ((len(codes) + self.batch_size - 1) // self.batch_size),
                }
            )
            if ce_value < best:
                best, patience = ce_value, 0
                best_state = {k: v.detach().cpu().clone() for k, v in vae.state_dict().items()}
            else:
                patience += 1
                if patience == 10 and beta > minimum_beta:
                    beta *= 0.7
        vae.load_state_dict(best_state)
        encoder = Encoder_model(*args, n_head=1, factor=factor).to(self.device)
        self.decoder = Decoder_model(*args, n_head=1, factor=factor).to(self.device)
        encoder.load_weights(vae)
        self.decoder.load_weights(vae)
        encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            chunks = []
            for start in range(0, len(codes), self.batch_size):
                batch = torch.as_tensor(
                    codes[start : start + self.batch_size], device=self.device
                ).long()
                z = encoder(torch.empty((len(batch), 0), device=self.device), batch)[:, 1:]
                chunks.append(z.flatten(1).cpu())
            latent = torch.cat(chunks)
        self.mean = latent.mean(0).to(self.device)
        latent = (latent - self.mean.cpu()) / 2
        self.latent_dim = latent.shape[1]
        self.model = Model(
            MLPDiffusion(self.latent_dim, int(self.options.get("hidden_dim", 1024))),
            hid_dim=self.latent_dim,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3 * multiplier)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=20)
        best, patience, best_state = float("inf"), 0, None
        for epoch in range(self.epochs):
            self.model.train()
            total = count = 0
            for batch in self.loader(latent):
                loss = self.model(batch.float().to(self.device)).mean()
                finite_step(loss, optimizer, self.model)
                total += float(loss.detach()) * len(batch)
                count += len(batch)
            value = total / count
            scheduler.step(value)
            self.logs.append(
                {
                    "stage": "diffusion",
                    "epoch": epoch + 1,
                    "loss": value,
                    "lr": optimizer.param_groups[0]["lr"],
                    "updates": (epoch + 1)
                    * ((len(latent) + self.batch_size - 1) // self.batch_size),
                }
            )
            if value < best:
                best, patience = value, 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience += 1
            if patience >= 500:
                break
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.save_torch(
            vae=vae.state_dict(),
            diffusion=self.model.state_dict(),
            decoder=self.decoder.state_dict(),
            latent_mean=self.mean.cpu(),
        )
        self.fit_details = {
            "compatibility": "empty numerical reconstruction term omitted",
            "decoder": "official categorical argmax",
            "vae_epochs": vae_epochs,
            "vae_scheduler_reference": "training CE; no held-out observations read",
        }

    def _draw_codes(self, n):
        from tabsyn.diffusion_utils import sample

        with torch.no_grad():
            latent = sample(
                self.model.denoise_fn_D,
                n,
                self.latent_dim,
                num_steps=self.sampling_steps,
                device=self.device,
            ).float()
            latent = (latent * 2 + self.mean).reshape(n, len(self.codec.columns), self.token_dim)
            _, logits = self.decoder(latent)
            if (
                not torch.isfinite(latent).all()
                or not torch.stack([torch.isfinite(values).all() for values in logits]).all()
            ):
                raise AdapterFailure("sample", "NONFINITE_OUTPUT", "TabSyn latent or decoder")
            return np.column_stack([x.argmax(1).cpu().numpy() for x in logits])
