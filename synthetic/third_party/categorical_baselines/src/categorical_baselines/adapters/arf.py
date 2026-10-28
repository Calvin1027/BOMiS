import pickle

import pandas as pd

from .vendor import VendorJointAdapter


class ARFAdapter(VendorJointAdapter):
    name = "arf"

    def _fit_codes(self, codes):
        from arfpy.arf import arf

        # Every column, including Y, must reach FORDE as a factor, never a number.
        frame = pd.DataFrame(codes, columns=self.codec.columns).astype(str).astype("category")
        self.model = arf(
            frame,
            num_trees=int(self.options.get("num_trees", 30)),
            max_iters=int(self.options.get("max_iters", 10)),
            min_node_size=int(self.options.get("min_node_size", 5)),
            verbose=False,
            random_state=self.model_seed,
            n_jobs=int(self.options.get("n_jobs", 1)),
        )
        self.model.forde()
        # OOB accuracy is the native adversarial stopping diagnostic, not a neural loss.
        self.logs = [{"stage": "arf", "epoch": 0, "loss": float(self.model.clf.oob_score_)}]
        with (self.work / "checkpoint.pt").open("wb") as stream:
            pickle.dump(self.model, stream)
        self.fit_details = {
            "loss_meaning": "final discriminator OOB accuracy",
            "checkpoint_format": "pickle",
            "factor_columns": self.codec.columns,
        }

    def _draw_codes(self, n):
        return self.model.forge(n).loc[:, self.codec.columns].astype(int).to_numpy()
