import torch
import hydra
from omegaconf import DictConfig

def create_model(config):
    """
    Factory function for creating model instances
    
    Args:
        config: Configuration for the model
        
    Returns:
        Instantiated model
    """
    if hasattr(config, "_target_"):
        # Use Hydra instantiation if target is specified
        return hydra.utils.instantiate(config)
    
    # Fallback to manual instantiation
    model_type = config.get("type", "traisformer")
    
    if model_type == "traisformer":
        from src.models.traisformer import TrAISformer
        return TrAISformer(config)
    
    elif model_type == "ais_fuser":
        from src.models.ais_fuser import AISFuserLightning
        return AISFuserLightning(config)
    
    elif model_type == "xgboost":
        from src.models.baselines import XGBoostModel
        return XGBoostModel(config)
    
    elif model_type == "lstm":
        from src.models.baselines import LSTMModel
        return LSTMModel(config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_model(checkpoint_path, config=None):
    """
    Load a model from checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: Optional configuration for overrides
        
    Returns:
        Loaded model
    """
    if config is not None:
        # Create model with config then load weights
        model = create_model(config)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])
        return model
    else:
        # Let Lightning handle it
        from pytorch_lightning import LightningModule
        return LightningModule.load_from_checkpoint(checkpoint_path)

def get_model_class(model_type):
    """
    Get model class by type name
    
    Args:
        model_type: Type of model as string
        
    Returns:
        Model class
    """
    if model_type == "traisformer":
        from src.models.traisformer import TrAISformer
        return TrAISformer
    
    elif model_type == "ais_fuser":
        from src.models.ais_fuser import AISFuserLightning
        return AISFuserLightning
    
    elif model_type == "xgboost":
        from src.models.baselines import XGBoostModel
        return XGBoostModel
    
    elif model_type == "lstm":
        from src.models.baselines import LSTMModel
        return LSTMModel
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
