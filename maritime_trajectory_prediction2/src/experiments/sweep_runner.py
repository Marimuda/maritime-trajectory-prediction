import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger


class SweepRunner:
    def __init__(self, config: DictConfig):
        """
        Initialize the sweep runner with configuration

        Args:
            config: Hydra configuration
        """
        self.config = config
        self.setup_environment()

    def setup_environment(self):
        """Set up random seeds and other environment variables"""
        pl.seed_everything(self.config.seed)
        os.environ["WANDB_PROJECT"] = self.config.get(
            "wandb_project", "ais_trajectory_prediction"
        )

    def run(self):
        """Run a single experiment with the given configuration"""
        # Print configuration
        print(OmegaConf.to_yaml(self.config))

        # Create model
        model = hydra.utils.instantiate(self.config.model)

        # Create data module
        datamodule = hydra.utils.instantiate(self.config.data)

        # Create logger
        logger = WandbLogger(
            name=self.config.name,
            save_dir=self.config.output_dir,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # Create callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=os.path.join(self.config.output_dir, "checkpoints"),
                filename="{epoch}-{val_loss:.4f}",
                monitor="val_loss",
                save_top_k=3,
                mode="min",
            ),
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        # Create trainer
        trainer = pl.Trainer(logger=logger, callbacks=callbacks, **self.config.trainer)

        # Fit model
        trainer.fit(model, datamodule=datamodule)

        # Test model
        trainer.test(model, datamodule=datamodule)

        # Return best validation score for hyperparameter optimization
        return trainer.callback_metrics.get("val_loss")


@hydra.main(config_path="../../configs", config_name="experiment/base")
def main(config: DictConfig):
    """Main entry point for running a sweep experiment"""
    runner = SweepRunner(config)
    return runner.run()


if __name__ == "__main__":
    main()
