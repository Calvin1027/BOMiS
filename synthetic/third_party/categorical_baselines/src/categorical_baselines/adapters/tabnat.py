import torch

from .vendor import VendorJointAdapter, finite_step


class TabNATAdapter(VendorJointAdapter):
    name = "tabnat"

    def _fit_codes(self, codes):
        from tabnat.model import TabNAT

        self.model = TabNAT(
            n_num=0,
            n_cat=len(self.codec.columns),
            categories=self.codec.cardinalities,
            embed_dim=int(self.options.get("embed_dim", 32)),
            buffer_size=int(self.options.get("buffer_size", 8)),
            depth=int(self.options.get("depth", 6)),
            dropout_rate=float(self.options.get("dropout_rate", 0.0)),
            device=self.device,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.options.get("lr", 1e-4)) * float(self.options.get("lr_multiplier", 1.0)),
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=50)
        for epoch in range(self.epochs):
            self.model.train()
            total = count = 0
            for batch in self.loader(codes):
                batch = batch.long().to(self.device)
                loss, _, _ = self.model(None, batch)
                finite_step(loss, optimizer, self.model)
                total += float(loss.detach()) * len(batch)
                count += len(batch)
            # Match the upstream scheduler's summed epoch loss.
            scheduler.step(total)
            self.logs.append(
                {
                    "stage": "masked_transformer",
                    "epoch": epoch + 1,
                    "loss": total / count,
                    "lr": optimizer.param_groups[0]["lr"],
                    "updates": (epoch + 1)
                    * ((len(codes) + self.batch_size - 1) // self.batch_size),
                }
            )
        self.model.eval()
        self.save_torch(state_dict=self.model.state_dict(), optimizer=optimizer.state_dict())
        self.fit_details = {
            "categorical_path": "x_num=None; official masked categorical CE",
            "generation": "official order-agnostic joint sampling then Y rejection",
        }

    def _draw_codes(self, n):
        with torch.no_grad():
            _, categorical = self.model.sample(n, device=self.device)
        return categorical.cpu().numpy()
