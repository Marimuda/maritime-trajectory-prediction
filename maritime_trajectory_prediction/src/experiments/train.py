#\!/usr/bin/env python3
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)

@hydra.main(config_path="../../configs", config_name="experiment/base")
def main(config: DictConfig):
    """
    Main training function
    
    Args:
        config: Hydra configuration
    """
    # Print resolved config
    print(OmegaConf.to_yaml(config))
    
    # Set random seed
    pl.seed_everything(config.seed)
    
    # Create model
    model = hydra.utils.instantiate(config.model)
    print(f"Created model: {model.__class__.__name__}")
    
    # Create data module
    datamodule = hydra.utils.instantiate(config.data)
    print(f"Created data module: {datamodule.__class__.__name__}")
    
    # Create logger based on configuration
    if config.get("logger", {}).get("name", "") == "wandb":
        logger = WandbLogger(
            name=config.name,
            save_dir=config.output_dir,
            project=config.get("wandb_project", "ais_trajectory_prediction"),
            config=OmegaConf.to_container(config, resolve=True)
        )
    else:
        logger = TensorBoardLogger(
            save_dir=config.output_dir,
            name=config.name
        )
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(config.output_dir, "checkpoints"),
            filename="{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=3,
            mode="min"
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config.get("early_stopping_patience", 10),
            mode="min"
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **config.trainer
    )
    
    # Train model
    trainer.fit(model, datamodule=datamodule)
    
    # Test model
    trainer.test(model, datamodule=datamodule)
    
    # Save final model
    trainer.save_checkpoint(
        os.path.join(config.output_dir, "final_model.ckpt")
    )
    
    return {
        "val_loss": trainer.callback_metrics.get("val_loss", 0),
        "test_loss": trainer.callback_metrics.get("test_loss", 0)
    }

if __name__ == "__main__":
    main()
