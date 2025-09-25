"""
LightningModule wrapper for XGBoost baseline.
"""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import xgboost as xgb


class XGBoostModelLightning(pl.LightningModule):
    def __init__(
        self,
        max_depth: int,
        learning_rate: float,
        n_estimators: int,
        flatten_sequence: bool = True,
        seq_len: int = 10,
        feature_dim: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.models = None

    def _flatten(self, x: torch.Tensor) -> np.ndarray:
        arr = x.cpu().numpy()
        if self.hparams.flatten_sequence and arr.ndim == 3:
            B, T, F = arr.shape
            return arr.reshape(B, T * F)
        return arr

    def fit_models(self, train_inputs: torch.Tensor, train_targets: torch.Tensor):
        X = self._flatten(train_inputs)
        y = train_targets.cpu().numpy()
        self.models = []
        if y.ndim == 3:
            T, F = y.shape[1], y.shape[2]
            for t in range(T):
                step_models = []
                for f in range(F):
                    m = xgb.XGBRegressor(
                        max_depth=self.hparams.max_depth,
                        learning_rate=self.hparams.learning_rate,
                        n_estimators=self.hparams.n_estimators,
                        objective="reg:squarederror",
                        n_jobs=-1,
                        verbosity=0,
                    )
                    m.fit(X, y[:, t, f])
                    step_models.append(m)
                self.models.append(step_models)
        else:
            F = y.shape[1]
            for f in range(F):
                m = xgb.XGBRegressor(
                    max_depth=self.hparams.max_depth,
                    learning_rate=self.hparams.learning_rate,
                    n_estimators=self.hparams.n_estimators,
                    objective="reg:squarederror",
                    n_jobs=-1,
                    verbosity=0,
                )
                m.fit(X, y[:, f])
                self.models.append(m)

    def _predict_numpy(self, X: np.ndarray) -> np.ndarray:
        if self.models is None:
            raise RuntimeError("XGBoost models not fit yet")
        if isinstance(self.models[0], list):
            preds = []
            for step_models in self.models:
                cols = [m.predict(X) for m in step_models]
                preds.append(np.stack(cols, axis=1))
            preds = np.stack(preds, axis=1)
        else:
            cols = [m.predict(X) for m in self.models]
            preds = np.stack(cols, axis=1)
        return preds

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Use fit_models() outside Lightning loop")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        X = self._flatten(x)
        preds = torch.tensor(
            self._predict_numpy(X), dtype=torch.float32, device=self.device
        )
        loss = F.mse_loss(preds, y)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        X = self._flatten(x)
        preds = torch.tensor(
            self._predict_numpy(X), dtype=torch.float32, device=self.device
        )
        loss = F.mse_loss(preds, y)
        self.log("test_loss", loss, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, tuple) else batch
        X = self._flatten(x)
        preds = self._predict_numpy(X)
        return torch.tensor(preds, dtype=torch.float32, device=self.device)

    def configure_optimizers(self):
        return None
