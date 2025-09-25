# tests/test_datamodule.py

import pytest
import torch
import pathlib
from hydra import compose, initialize_config_dir

from ais_project.ais_datamodule import AISDataModule


@pytest.fixture(scope="session")
def cfg():
    # Use non-experimental initialize_config_dir from Hydra 1.2+
    project_root = pathlib.Path(__file__).parent.parent
    config_dir = project_root / "conf"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        yield compose(config_name="config")


@pytest.mark.parametrize(
    "loader_fn", ["train_dataloader", "val_dataloader", "test_dataloader"]
)
def test_loader_shapes(cfg, loader_fn):
    dm = AISDataModule(cfg)
    dm.setup()
    loader = getattr(dm, loader_fn)()
    x, y = next(iter(loader))
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 3, f"Expected input shape (B, history, features), got {x.shape}"
    assert (
        y.ndim == 3 and y.shape[2] == 2
    ), f"Expected target shape (B, horizon, 2), got {y.shape}"


def test_split_sum(cfg):
    dm = AISDataModule(cfg)
    dm.setup()
    total = len(dm._dataset)
    splits = sum(len(s) for s in (dm.train_set, dm.val_set, dm.test_set))
    assert total == splits, f"Sum of splits ({splits}) != total ({total})"
