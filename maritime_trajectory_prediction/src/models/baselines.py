import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import xgboost as xgb

class LSTMModel(pl.LightningModule):
    """LSTM baseline model for AIS trajectory prediction"""
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Model parameters
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.output_size = config.output_size
        self.dropout = config.dropout
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor (batch_size, seq_len, output_size)
        """
        # LSTM forward pass
        outputs, _ = self.lstm(x)
        
        # Apply output layer to each time step
        predictions = self.fc(outputs)
        
        return predictions
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        inputs, targets = batch
        
        # Forward pass
        outputs = self(inputs)
        
        # Calculate loss
        loss = F.mse_loss(outputs, targets)
        
        # Log metrics
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        inputs, targets = batch
        
        # Forward pass
        outputs = self(inputs)
        
        # Calculate loss
        loss = F.mse_loss(outputs, targets)
        
        # Log metrics
        self.log('val_loss', loss)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        inputs, targets = batch
        
        # Forward pass
        outputs = self(inputs)
        
        # Calculate loss
        loss = F.mse_loss(outputs, targets)
        
        # Log metrics
        self.log('test_loss', loss)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        """Prediction step"""
        if isinstance(batch, tuple):
            inputs = batch[0]
        else:
            inputs = batch
        
        # Forward pass
        outputs = self(inputs)
        
        return outputs
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

class XGBoostModel(pl.LightningModule):
    """XGBoost baseline model for AIS trajectory prediction"""
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # XGBoost parameters
        self.model_params = {
            'max_depth': config.max_depth,
            'learning_rate': config.learning_rate,
            'n_estimators': config.n_estimators,
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Feature processing
        self.flatten_sequence = config.get('flatten_sequence', True)
        self.seq_len = config.get('seq_len', 10)
        self.feature_dim = config.get('feature_dim', 4)
        
        # Initialize models (one for each output dimension)
        self.models = None
    
    def flatten_features(self, x):
        """
        Flatten sequence features for XGBoost
        
        Args:
            x: Input tensor (batch_size, seq_len, feature_dim)
            
        Returns:
            Flattened features (batch_size, seq_len * feature_dim)
        """
        return x.reshape(x.shape[0], -1)
    
    def fit(self, train_inputs, train_targets):
        """
        Fit XGBoost models
        
        Args:
            train_inputs: Training inputs
            train_targets: Training targets
        """
        # Convert to numpy arrays
        if isinstance(train_inputs, torch.Tensor):
            train_inputs = train_inputs.numpy()
        if isinstance(train_targets, torch.Tensor):
            train_targets = train_targets.numpy()
        
        # Flatten inputs if needed
        if self.flatten_sequence and len(train_inputs.shape) > 2:
            train_inputs = self.flatten_features(train_inputs)
        
        # Create and train models (one for each output dimension)
        self.models = []
        
        # Typically targets have shape (batch_size, seq_len, feature_dim)
        # We need to predict each dimension separately
        target_shape = train_targets.shape
        
        if len(target_shape) == 3:
            # For sequence-to-sequence prediction
            for i in range(target_shape[1]):  # For each time step
                step_models = []
                for j in range(target_shape[2]):  # For each feature
                    model = xgb.XGBRegressor(**self.model_params)
                    model.fit(train_inputs, train_targets[:, i, j])
                    step_models.append(model)
                self.models.append(step_models)
        else:
            # For single-step prediction
            for i in range(target_shape[1]):  # For each feature
                model = xgb.XGBRegressor(**self.model_params)
                model.fit(train_inputs, train_targets[:, i])
                self.models.append(model)
    
    def predict(self, inputs):
        """
        Make predictions with XGBoost models
        
        Args:
            inputs: Input features
            
        Returns:
            Predictions
        """
        # Convert to numpy array
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.numpy()
        
        # Flatten inputs if needed
        if self.flatten_sequence and len(inputs.shape) > 2:
            inputs = self.flatten_features(inputs)
        
        # Check if models are trained
        if self.models is None:
            raise RuntimeError("Models must be fit before prediction")
        
        # Make predictions
        predictions = []
        
        if isinstance(self.models[0], list):
            # For sequence-to-sequence prediction
            for step_models in self.models:
                step_preds = []
                for model in step_models:
                    step_preds.append(model.predict(inputs))
                predictions.append(np.column_stack(step_preds))
            
            # Reshape to (batch_size, seq_len, feature_dim)
            predictions = np.array(predictions)
            predictions = np.transpose(predictions, (1, 0, 2))
        else:
            # For single-step prediction
            for model in self.models:
                predictions.append(model.predict(inputs))
            
            # Reshape to (batch_size, feature_dim)
            predictions = np.column_stack(predictions)
        
        return torch.tensor(predictions, dtype=torch.float32)
    
    def training_step(self, batch, batch_idx):
        """Not used in XGBoost - we use fit() directly"""
        raise NotImplementedError("XGBoost training is handled outside of Lightning loop")
    
    def validation_step(self, batch, batch_idx):
        """Validation step for XGBoost"""
        inputs, targets = batch
        
        # Ensure models are fit
        if self.models is None:
            raise RuntimeError("Models must be fit before validation")
        
        # Get predictions
        predictions = self.predict(inputs)
        
        # Calculate loss
        loss = F.mse_loss(predictions, targets)
        
        # Log metrics
        self.log('val_loss', loss)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step for XGBoost"""
        inputs, targets = batch
        
        # Ensure models are fit
        if self.models is None:
            raise RuntimeError("Models must be fit before testing")
        
        # Get predictions
        predictions = self.predict(inputs)
        
        # Calculate loss
        loss = F.mse_loss(predictions, targets)
        
        # Log metrics
        self.log('test_loss', loss)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        """Prediction step for XGBoost"""
        if isinstance(batch, tuple):
            inputs = batch[0]
        else:
            inputs = batch
        
        # Get predictions
        predictions = self.predict(inputs)
        
        return predictions
    
    def configure_optimizers(self):
        """Not used in XGBoost"""
        return None
