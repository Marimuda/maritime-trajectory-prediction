import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def freq_to_seconds(freq: str) -> float:
    """Convert frequency string (e.g., '1min') to seconds."""
    unit = "".join([c for c in freq.lower() if not c.isdigit()])
    value = int("".join([c for c in freq if c.isdigit()]))
    mapping = {"s": 1, "min": 60, "h": 3600}
    if unit not in mapping:
        raise ValueError(f"Unsupported freq unit: {unit}")
    return value * mapping[unit]


def extract_feature_keys(scalers_cfg: DictConfig) -> List[str]:
    """Return feature names from scalers config, excluding meta keys."""
    return [k for k in scalers_cfg.keys() if not k.startswith("_")]


class AISDataset(Dataset):
    """
    PyTorch Dataset for sequence-based AIS data.
    Each item is a (past, future) tensor pair.
    """

    def __init__(
        self,
        data: np.ndarray,
        history: int,
        horizon: int,
    ):
        """
        Args:
            data: 2D array shape [T, F] for a single vessel.
            history: number of input timesteps.
            horizon: number of output timesteps.
        """
        self.history = history
        self.horizon = horizon
        self.data = data
        self.length = data.shape[0] - history - horizon + 1

    def __len__(self) -> int:
        return max(self.length, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        past = self.data[start : start + self.history]
        future = self.data[
            start + self.history : start + self.history + self.horizon, :2
        ]
        return torch.from_numpy(past).float(), torch.from_numpy(future).float()


class AISDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for AIS trajectory prediction.

    - Preprocesses raw AIS Parquet files: resamples, interpolates, computes velocities.
    - Splits data into train/val/test.
    - Fits StandardScaler on train features (vectorized) and applies to all.
    - Generates torch Datasets of sequence pairs.
    """

    REQUIRED_FEATURES = ["lat", "lon", "sog", "cog", "heading"]
    DERIVED_FEATURES = ["vx", "vy"]

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg.data
        # Determine feature order from config
        self.features: List[str] = extract_feature_keys(self.cfg.scalers)
        self.history = self.cfg.history
        self.horizon = self.cfg.horizon
        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None
        self.scalers: Dict[str, StandardScaler] = {}

    def _load_preprocess(self) -> pd.DataFrame:
        df = pd.concat([pd.read_parquet(p) for p in self.cfg.paths], ignore_index=True)
        # Validate core columns
        for col in ["mmsi", "timestamp"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Resolve feature aliases: sog<->speed, cog<->course
        alias_map = {}
        for feat, candidates in {
            "sog": ["sog", "speed"],
            "cog": ["cog", "course"],
        }.items():
            found = next((c for c in candidates if c in df.columns), None)
            if found is None:
                raise ValueError(
                    f"Missing required feature '{feat}', tried aliases: {candidates}"
                )
            alias_map[found] = feat
        if alias_map:
            df = df.rename(columns=alias_map)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values(["mmsi", "timestamp"], inplace=True)

        # Alias and interpolate required features per vessel"
        frames = []
        for mmsi, grp in df.groupby("mmsi"):
            g = grp.set_index("timestamp").resample(self.cfg.freq).first()
            g[self.REQUIRED_FEATURES] = g[self.REQUIRED_FEATURES].interpolate(
                method="linear", limit_direction="both"
            )
            g["mmsi"] = mmsi
            frames.append(g.reset_index())
        full = pd.concat(frames, ignore_index=True)

        # Compute velocities
        full["dt"] = (
            full.groupby("mmsi")["timestamp"]
            .diff()
            .dt.total_seconds()
            .fillna(freq_to_seconds(self.cfg.freq))
        )
        full["dlat"] = full.groupby("mmsi")["lat"].diff().fillna(0)
        full["dlon"] = full.groupby("mmsi")["lon"].diff().fillna(0)
        full["vx"] = full["dlat"] / full["dt"] * 1852 / 3600
        full["vy"] = full["dlon"] / full["dt"] * 1852 / 3600

        # Ensure all features present
        missing = [f for f in self.features if f not in full.columns]
        if missing:
            raise ValueError(f"Missing features after preprocess: {missing}")

        return full

    def setup(self, stage: Optional[str] = None) -> None:
        # 1. Load and preprocess dataframe
        df = self._load_preprocess()
        data_array = df[self.features].to_numpy()

        # 2. Split time series into sequence windows per vessel
        sequences: List[np.ndarray] = []
        vessel_idx = df["mmsi"].to_numpy()
        # Use sliding window over entire array, BUT reset at vessel boundaries
        start = 0
        for m in np.unique(vessel_idx):
            mask = vessel_idx == m
            arr = data_array[mask]
            T = arr.shape[0]
            L = self.history + self.horizon
            for i in range(T - L + 1):
                sequences.append(arr[i : i + L])
            start += T
        sequences = np.stack(sequences)  # [N_seq, L, F]

        # 3. Train/val/test split
        total = len(sequences)
        t = int(total * self.cfg.test_split)
        v = int(total * self.cfg.val_split)
        tr = total - v - t
        train_seqs = sequences[:tr]
        val_seqs = sequences[tr : tr + v]
        test_seqs = sequences[tr + v :]

        # 4. Fit scalers on train past only (vectorized)
        L = self.history + self.horizon
        flat_past = train_seqs[:, : self.history].reshape(-1, len(self.features))
        self.scalers = instantiate(self.cfg.scalers)
        for idx, feat in enumerate(self.features):
            scaler: StandardScaler = self.scalers[feat]
            scaler.fit(flat_past[:, idx].reshape(-1, 1))

        # 5. Apply batch scaling to all splits
        def scale_batch(arr: np.ndarray) -> np.ndarray:
            b = arr.copy()
            F = b.shape[2]
            b = b.reshape(-1, F)
            for idx, feat in enumerate(self.features):
                b[:, idx] = (
                    self.scalers[feat].transform(b[:, idx].reshape(-1, 1)).flatten()
                )
            return b.reshape(arr.shape)

        train_seqs = scale_batch(train_seqs)
        val_seqs = scale_batch(val_seqs)
        test_seqs = scale_batch(test_seqs)

        # 6. Create Datasets
        self.train_set = AISDataset(train_seqs, self.history, self.horizon)
        self.val_set = AISDataset(val_seqs, self.history, self.horizon)
        self.test_set = AISDataset(test_seqs, self.history, self.horizon)

        logger.info(
            f"AISDataModule setup: train={len(self.train_set)}, val={len(self.val_set)}, test={len(self.test_set)}"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
        )
