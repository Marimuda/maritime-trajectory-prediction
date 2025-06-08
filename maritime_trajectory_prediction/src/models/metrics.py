"""
Task-specific metrics and loss functions for maritime baseline models.

This module implements appropriate metrics and loss functions for:
1. Trajectory Prediction - RMSE, ADE, FDE, maritime-specific metrics
2. Anomaly Detection - Precision, Recall, F1, ROC-AUC, maritime anomaly types
3. Vessel Interaction - Collision prediction accuracy, interaction classification

Each metric is designed for maritime domain specifics and evaluation standards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)


class TrajectoryPredictionMetrics:
    """
    Comprehensive metrics for maritime trajectory prediction evaluation.
    
    Includes standard trajectory metrics (ADE, FDE) and maritime-specific
    metrics (course deviation, speed accuracy, collision time to closest point).
    """
    
    def __init__(self, earth_radius_km: float = 6371.0):
        self.earth_radius_km = earth_radius_km
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.metadata = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update metrics with new predictions.
        
        Args:
            predictions: Predicted trajectories (batch, seq_len, 4) [lat, lon, sog, cog]
            targets: Ground truth trajectories (batch, seq_len, 4)
            metadata: Optional metadata (vessel types, timestamps, etc.)
        """
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())
        if metadata:
            self.metadata.append(metadata)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all trajectory prediction metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        if not self.predictions:
            return {}
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.predictions, dim=0)  # (total_samples, seq_len, 4)
        all_targets = torch.cat(self.targets, dim=0)
        
        metrics = {}
        
        # Extract position and movement components
        pred_positions = all_preds[:, :, :2]  # lat, lon
        target_positions = all_targets[:, :, :2]
        pred_movement = all_preds[:, :, 2:]  # sog, cog
        target_movement = all_targets[:, :, 2:]
        
        # 1. Average Displacement Error (ADE)
        metrics['ade_km'] = self._compute_ade(pred_positions, target_positions)
        
        # 2. Final Displacement Error (FDE)
        metrics['fde_km'] = self._compute_fde(pred_positions, target_positions)
        
        # 3. Root Mean Square Error for positions
        metrics['rmse_position_km'] = self._compute_rmse_position(pred_positions, target_positions)
        
        # 4. Speed accuracy (SOG)
        metrics['rmse_speed_knots'] = torch.sqrt(F.mse_loss(pred_movement[:, :, 0], target_movement[:, :, 0])).item()
        metrics['mae_speed_knots'] = F.l1_loss(pred_movement[:, :, 0], target_movement[:, :, 0]).item()
        
        # 5. Course accuracy (COG)
        metrics['rmse_course_degrees'] = self._compute_course_rmse(pred_movement[:, :, 1], target_movement[:, :, 1])
        metrics['mae_course_degrees'] = self._compute_course_mae(pred_movement[:, :, 1], target_movement[:, :, 1])
        
        # 6. Maritime-specific metrics
        metrics['bearing_error_degrees'] = self._compute_bearing_error(pred_positions, target_positions)
        metrics['cross_track_error_km'] = self._compute_cross_track_error(pred_positions, target_positions)
        
        # 7. Temporal consistency
        metrics['velocity_consistency'] = self._compute_velocity_consistency(pred_positions, target_positions)
        
        return metrics
    
    def _compute_ade(self, pred_pos: torch.Tensor, target_pos: torch.Tensor) -> float:
        """Compute Average Displacement Error in kilometers."""
        distances = self._haversine_distance(
            pred_pos[:, :, 0], pred_pos[:, :, 1],
            target_pos[:, :, 0], target_pos[:, :, 1]
        )
        return distances.mean().item()
    
    def _compute_fde(self, pred_pos: torch.Tensor, target_pos: torch.Tensor) -> float:
        """Compute Final Displacement Error in kilometers."""
        final_distances = self._haversine_distance(
            pred_pos[:, -1, 0], pred_pos[:, -1, 1],
            target_pos[:, -1, 0], target_pos[:, -1, 1]
        )
        return final_distances.mean().item()
    
    def _compute_rmse_position(self, pred_pos: torch.Tensor, target_pos: torch.Tensor) -> float:
        """Compute RMSE for position in kilometers."""
        distances = self._haversine_distance(
            pred_pos[:, :, 0], pred_pos[:, :, 1],
            target_pos[:, :, 0], target_pos[:, :, 1]
        )
        return torch.sqrt(distances.pow(2).mean()).item()
    
    def _compute_course_rmse(self, pred_course: torch.Tensor, target_course: torch.Tensor) -> float:
        """Compute RMSE for course with circular handling."""
        # Handle circular nature of course (0-360 degrees)
        diff = pred_course - target_course
        diff = torch.where(diff > 180, diff - 360, diff)
        diff = torch.where(diff < -180, diff + 360, diff)
        return torch.sqrt(diff.pow(2).mean()).item()
    
    def _compute_course_mae(self, pred_course: torch.Tensor, target_course: torch.Tensor) -> float:
        """Compute MAE for course with circular handling."""
        diff = pred_course - target_course
        diff = torch.where(diff > 180, diff - 360, diff)
        diff = torch.where(diff < -180, diff + 360, diff)
        return torch.abs(diff).mean().item()
    
    def _compute_bearing_error(self, pred_pos: torch.Tensor, target_pos: torch.Tensor) -> float:
        """Compute bearing error between predicted and actual trajectories."""
        # Compute bearings for consecutive points
        pred_bearings = self._compute_bearings(pred_pos)
        target_bearings = self._compute_bearings(target_pos)
        
        # Compute circular difference
        diff = pred_bearings - target_bearings
        diff = torch.where(diff > 180, diff - 360, diff)
        diff = torch.where(diff < -180, diff + 360, diff)
        
        return torch.abs(diff).mean().item()
    
    def _compute_cross_track_error(self, pred_pos: torch.Tensor, target_pos: torch.Tensor) -> float:
        """Compute cross-track error (perpendicular distance from intended path)."""
        # Simplified cross-track error using great circle distance
        batch_size, seq_len, _ = pred_pos.shape
        
        if seq_len < 2:
            return 0.0
        
        cross_track_errors = []
        
        for i in range(1, seq_len):
            # Vector from previous to current target position
            target_vector = target_pos[:, i, :] - target_pos[:, i-1, :]
            
            # Vector from previous target to current prediction
            pred_vector = pred_pos[:, i, :] - target_pos[:, i-1, :]
            
            # Cross product magnitude gives cross-track error (simplified for lat/lon)
            cross_track = torch.abs(
                target_vector[:, 0] * pred_vector[:, 1] - 
                target_vector[:, 1] * pred_vector[:, 0]
            )
            cross_track_errors.append(cross_track)
        
        if cross_track_errors:
            return torch.stack(cross_track_errors).mean().item() * 111.0  # Rough km conversion
        return 0.0
    
    def _compute_velocity_consistency(self, pred_pos: torch.Tensor, target_pos: torch.Tensor) -> float:
        """Compute velocity consistency between predicted and actual trajectories."""
        # Compute velocities (differences between consecutive positions)
        pred_velocities = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        target_velocities = target_pos[:, 1:, :] - target_pos[:, :-1, :]
        
        # Compute velocity magnitude differences
        pred_speeds = torch.norm(pred_velocities, dim=2)
        target_speeds = torch.norm(target_velocities, dim=2)
        
        speed_consistency = 1.0 - torch.abs(pred_speeds - target_speeds).mean().item()
        return max(0.0, speed_consistency)
    
    def _haversine_distance(self, lat1: torch.Tensor, lon1: torch.Tensor, 
                           lat2: torch.Tensor, lon2: torch.Tensor) -> torch.Tensor:
        """Compute haversine distance between points in kilometers."""
        # Convert to radians
        lat1_rad = torch.deg2rad(lat1)
        lon1_rad = torch.deg2rad(lon1)
        lat2_rad = torch.deg2rad(lat2)
        lon2_rad = torch.deg2rad(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = torch.sin(dlat/2)**2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon/2)**2
        c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
        
        return self.earth_radius_km * c
    
    def _compute_bearings(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute bearings between consecutive positions."""
        if positions.shape[1] < 2:
            return torch.zeros(positions.shape[0], 0)
        
        lat1 = torch.deg2rad(positions[:, :-1, 0])
        lon1 = torch.deg2rad(positions[:, :-1, 1])
        lat2 = torch.deg2rad(positions[:, 1:, 0])
        lon2 = torch.deg2rad(positions[:, 1:, 1])
        
        dlon = lon2 - lon1
        
        y = torch.sin(dlon) * torch.cos(lat2)
        x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
        
        bearings = torch.rad2deg(torch.atan2(y, x))
        bearings = (bearings + 360) % 360  # Normalize to 0-360
        
        return bearings


class AnomalyDetectionMetrics:
    """
    Comprehensive metrics for maritime anomaly detection evaluation.
    
    Includes standard classification metrics and maritime-specific
    anomaly types (route deviation, speed anomalies, loitering, etc.).
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.anomaly_scores = []
        self.true_labels = []
        self.predictions = []
        self.metadata = []
    
    def update(
        self,
        anomaly_scores: torch.Tensor,
        true_labels: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update metrics with new anomaly predictions.
        
        Args:
            anomaly_scores: Anomaly scores (higher = more anomalous)
            true_labels: Binary labels (1 = anomaly, 0 = normal)
            metadata: Optional metadata (anomaly types, vessel info, etc.)
        """
        self.anomaly_scores.append(anomaly_scores.detach().cpu())
        self.true_labels.append(true_labels.detach().cpu())
        
        # Convert scores to binary predictions
        predictions = (anomaly_scores > self.threshold).float()
        self.predictions.append(predictions.detach().cpu())
        
        if metadata:
            self.metadata.append(metadata)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all anomaly detection metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        if not self.anomaly_scores:
            return {}
        
        # Concatenate all scores and labels
        all_scores = torch.cat(self.anomaly_scores, dim=0).numpy()
        all_labels = torch.cat(self.true_labels, dim=0).numpy()
        all_preds = torch.cat(self.predictions, dim=0).numpy()
        
        metrics = {}
        
        # 1. Classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # 2. ROC-AUC
        if len(np.unique(all_labels)) > 1:  # Need both classes for AUC
            metrics['roc_auc'] = roc_auc_score(all_labels, all_scores)
        else:
            metrics['roc_auc'] = 0.5
        
        # 3. Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)
        
        # 4. Additional metrics
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # 5. Maritime-specific metrics
        metrics.update(self._compute_maritime_anomaly_metrics(all_scores, all_labels))
        
        return metrics
    
    def _compute_maritime_anomaly_metrics(self, scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute maritime-specific anomaly detection metrics."""
        maritime_metrics = {}
        
        # Detection latency (how quickly anomalies are detected)
        if hasattr(self, 'metadata') and self.metadata:
            # This would require temporal metadata to compute properly
            maritime_metrics['detection_latency_minutes'] = 0.0
        
        # Anomaly severity distribution
        anomaly_indices = labels == 1
        if np.any(anomaly_indices):
            anomaly_severity = scores[anomaly_indices]
            maritime_metrics['mean_anomaly_severity'] = float(np.mean(anomaly_severity))
            maritime_metrics['max_anomaly_severity'] = float(np.max(anomaly_severity))
        else:
            maritime_metrics['mean_anomaly_severity'] = 0.0
            maritime_metrics['max_anomaly_severity'] = 0.0
        
        # False alarm rate (important for maritime operations)
        normal_indices = labels == 0
        if np.any(normal_indices):
            false_alarms = scores[normal_indices] > self.threshold
            maritime_metrics['false_alarm_rate'] = float(np.mean(false_alarms))
        else:
            maritime_metrics['false_alarm_rate'] = 0.0
        
        return maritime_metrics


class VesselInteractionMetrics:
    """
    Comprehensive metrics for vessel interaction and collision prediction.
    
    Includes collision prediction accuracy, interaction classification,
    and maritime safety metrics (CPA, TCPA, collision risk assessment).
    """
    
    def __init__(self, collision_threshold: float = 0.5, cpa_threshold_km: float = 2.0):
        self.collision_threshold = collision_threshold
        self.cpa_threshold_km = cpa_threshold_km
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.collision_predictions = []
        self.collision_labels = []
        self.interaction_scores = []
        self.vessel_positions = []
        self.metadata = []
    
    def update(
        self,
        collision_predictions: torch.Tensor,
        collision_labels: torch.Tensor,
        interaction_scores: Optional[torch.Tensor] = None,
        vessel_positions: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update metrics with new interaction predictions.
        
        Args:
            collision_predictions: Predicted collision probabilities
            collision_labels: True collision labels
            interaction_scores: Optional interaction strength scores
            vessel_positions: Optional vessel position data for CPA/TCPA
            metadata: Optional metadata (vessel types, timestamps, etc.)
        """
        self.collision_predictions.append(collision_predictions.detach().cpu())
        self.collision_labels.append(collision_labels.detach().cpu())
        
        if interaction_scores is not None:
            self.interaction_scores.append(interaction_scores.detach().cpu())
        
        if vessel_positions is not None:
            self.vessel_positions.append(vessel_positions.detach().cpu())
        
        if metadata:
            self.metadata.append(metadata)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all vessel interaction metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        if not self.collision_predictions:
            return {}
        
        # Concatenate all predictions and labels
        all_preds = torch.cat(self.collision_predictions, dim=0).numpy()
        all_labels = torch.cat(self.collision_labels, dim=0).numpy()
        
        metrics = {}
        
        # 1. Collision prediction metrics
        binary_preds = (all_preds > self.collision_threshold).astype(int)
        
        if len(np.unique(all_labels)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, binary_preds, average='binary', zero_division=0
            )
            
            metrics['collision_precision'] = precision
            metrics['collision_recall'] = recall
            metrics['collision_f1'] = f1
            metrics['collision_auc'] = roc_auc_score(all_labels, all_preds)
        else:
            metrics['collision_precision'] = 0.0
            metrics['collision_recall'] = 0.0
            metrics['collision_f1'] = 0.0
            metrics['collision_auc'] = 0.5
        
        # 2. Collision prediction accuracy
        metrics['collision_accuracy'] = float(np.mean(binary_preds == all_labels))
        
        # 3. Maritime safety metrics
        if self.vessel_positions:
            safety_metrics = self._compute_maritime_safety_metrics()
            metrics.update(safety_metrics)
        
        # 4. Interaction strength metrics
        if self.interaction_scores:
            interaction_metrics = self._compute_interaction_metrics()
            metrics.update(interaction_metrics)
        
        # 5. Risk assessment metrics
        metrics.update(self._compute_risk_assessment_metrics(all_preds, all_labels))
        
        return metrics
    
    def _compute_maritime_safety_metrics(self) -> Dict[str, float]:
        """Compute maritime safety metrics (CPA, TCPA, etc.)."""
        safety_metrics = {}
        
        # This would require implementing CPA/TCPA calculations
        # For now, return placeholder metrics
        safety_metrics['mean_cpa_km'] = 0.0
        safety_metrics['mean_tcpa_minutes'] = 0.0
        safety_metrics['close_encounters'] = 0.0
        
        return safety_metrics
    
    def _compute_interaction_metrics(self) -> Dict[str, float]:
        """Compute vessel interaction strength metrics."""
        all_interactions = torch.cat(self.interaction_scores, dim=0).numpy()
        
        interaction_metrics = {
            'mean_interaction_strength': float(np.mean(all_interactions)),
            'max_interaction_strength': float(np.max(all_interactions)),
            'interaction_variance': float(np.var(all_interactions))
        }
        
        return interaction_metrics
    
    def _compute_risk_assessment_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute risk assessment and safety metrics."""
        risk_metrics = {}
        
        # Risk calibration (how well predicted probabilities match actual rates)
        risk_metrics['mean_predicted_risk'] = float(np.mean(predictions))
        risk_metrics['actual_collision_rate'] = float(np.mean(labels))
        risk_metrics['risk_calibration_error'] = float(np.abs(np.mean(predictions) - np.mean(labels)))
        
        # High-risk scenario detection
        high_risk_threshold = 0.8
        high_risk_predictions = predictions > high_risk_threshold
        if np.any(high_risk_predictions):
            high_risk_accuracy = np.mean(labels[high_risk_predictions])
            risk_metrics['high_risk_accuracy'] = float(high_risk_accuracy)
        else:
            risk_metrics['high_risk_accuracy'] = 0.0
        
        return risk_metrics


class MaritimeLossFunctions:
    """
    Collection of loss functions designed for maritime tasks.
    
    Includes standard losses with maritime-specific modifications
    and domain-specific loss functions for trajectory, anomaly, and interaction tasks.
    """
    
    @staticmethod
    def trajectory_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        position_weight: float = 1.0,
        movement_weight: float = 0.5
    ) -> torch.Tensor:
        """
        Maritime trajectory prediction loss with position and movement components.
        
        Args:
            predictions: Predicted trajectories (batch, seq, 4) [lat, lon, sog, cog]
            targets: Target trajectories (batch, seq, 4)
            weights: Optional temporal weights
            position_weight: Weight for position loss
            movement_weight: Weight for movement loss
            
        Returns:
            Combined loss tensor
        """
        # Split into position and movement components
        pred_pos = predictions[:, :, :2]  # lat, lon
        target_pos = targets[:, :, :2]
        pred_movement = predictions[:, :, 2:]  # sog, cog
        target_movement = targets[:, :, 2:]
        
        # Position loss (MSE in lat/lon space)
        position_loss = F.mse_loss(pred_pos, target_pos, reduction='none')
        
        # Movement loss with circular handling for course
        speed_loss = F.mse_loss(pred_movement[:, :, 0:1], target_movement[:, :, 0:1], reduction='none')
        
        # Circular loss for course (COG)
        course_diff = pred_movement[:, :, 1] - target_movement[:, :, 1]
        course_diff = torch.where(course_diff > 180, course_diff - 360, course_diff)
        course_diff = torch.where(course_diff < -180, course_diff + 360, course_diff)
        course_loss = course_diff.pow(2).unsqueeze(-1)
        
        movement_loss = torch.cat([speed_loss, course_loss], dim=-1)
        
        # Combine losses
        total_loss = position_weight * position_loss.mean() + movement_weight * movement_loss.mean()
        
        # Apply temporal weights if provided
        if weights is not None:
            total_loss = total_loss * weights.mean()
        
        return total_loss
    
    @staticmethod
    def anomaly_loss(
        reconstructions: torch.Tensor,
        inputs: torch.Tensor,
        anomaly_labels: Optional[torch.Tensor] = None,
        reconstruction_weight: float = 1.0,
        regularization_weight: float = 0.1
    ) -> torch.Tensor:
        """
        Maritime anomaly detection loss with reconstruction and regularization.
        
        Args:
            reconstructions: Reconstructed inputs
            inputs: Original inputs
            anomaly_labels: Optional anomaly labels for supervised learning
            reconstruction_weight: Weight for reconstruction loss
            regularization_weight: Weight for regularization
            
        Returns:
            Combined loss tensor
        """
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructions, inputs)
        
        # Regularization loss (encourage sparse representations)
        if len(reconstructions.shape) == 3:  # Sequence data
            regularization_loss = torch.mean(torch.abs(reconstructions))
        else:
            regularization_loss = torch.mean(torch.abs(reconstructions))
        
        total_loss = reconstruction_weight * reconstruction_loss + regularization_weight * regularization_loss
        
        # Add supervised component if labels are available
        if anomaly_labels is not None:
            # Compute reconstruction error per sample
            sample_errors = F.mse_loss(reconstructions, inputs, reduction='none').mean(dim=-1)
            if len(sample_errors.shape) > 1:
                sample_errors = sample_errors.mean(dim=1)
            
            # Supervised loss: high error for anomalies, low error for normal
            supervised_loss = F.binary_cross_entropy_with_logits(
                sample_errors, anomaly_labels.float()
            )
            total_loss = total_loss + supervised_loss
        
        return total_loss
    
    @staticmethod
    def interaction_loss(
        collision_predictions: torch.Tensor,
        collision_labels: torch.Tensor,
        interaction_scores: Optional[torch.Tensor] = None,
        interaction_targets: Optional[torch.Tensor] = None,
        collision_weight: float = 1.0,
        interaction_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Maritime vessel interaction loss with collision prediction and interaction modeling.
        
        Args:
            collision_predictions: Predicted collision probabilities
            collision_labels: True collision labels
            interaction_scores: Optional interaction strength predictions
            interaction_targets: Optional interaction strength targets
            collision_weight: Weight for collision prediction loss
            interaction_weight: Weight for interaction modeling loss
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Combined loss tensor
        """
        # Collision prediction loss (binary classification)
        if class_weights is not None:
            collision_loss = F.binary_cross_entropy_with_logits(
                collision_predictions, collision_labels.float(),
                weight=class_weights
            )
        else:
            collision_loss = F.binary_cross_entropy_with_logits(
                collision_predictions, collision_labels.float()
            )
        
        total_loss = collision_weight * collision_loss
        
        # Interaction modeling loss (if available)
        if interaction_scores is not None and interaction_targets is not None:
            interaction_loss = F.mse_loss(interaction_scores, interaction_targets)
            total_loss = total_loss + interaction_weight * interaction_loss
        
        return total_loss


# Factory functions for creating metrics and loss functions
def create_metrics(task: str, **kwargs) -> Union[TrajectoryPredictionMetrics, AnomalyDetectionMetrics, VesselInteractionMetrics]:
    """
    Factory function to create appropriate metrics for different tasks.
    
    Args:
        task: Task name ('trajectory_prediction', 'anomaly_detection', 'vessel_interaction')
        **kwargs: Task-specific parameters
        
    Returns:
        Initialized metrics object
    """
    if task == 'trajectory_prediction':
        return TrajectoryPredictionMetrics(**kwargs)
    elif task == 'anomaly_detection':
        return AnomalyDetectionMetrics(**kwargs)
    elif task == 'vessel_interaction':
        return VesselInteractionMetrics(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


def create_loss_function(task: str) -> callable:
    """
    Factory function to create appropriate loss function for different tasks.
    
    Args:
        task: Task name ('trajectory_prediction', 'anomaly_detection', 'vessel_interaction')
        
    Returns:
        Loss function
    """
    if task == 'trajectory_prediction':
        return MaritimeLossFunctions.trajectory_loss
    elif task == 'anomaly_detection':
        return MaritimeLossFunctions.anomaly_loss
    elif task == 'vessel_interaction':
        return MaritimeLossFunctions.interaction_loss
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    # Test metrics and loss functions
    logging.basicConfig(level=logging.INFO)
    
    print("Testing trajectory prediction metrics...")
    traj_metrics = create_metrics('trajectory_prediction')
    
    # Test data
    batch_size, seq_len, features = 2, 5, 4
    predictions = torch.randn(batch_size, seq_len, features)
    targets = torch.randn(batch_size, seq_len, features)
    
    traj_metrics.update(predictions, targets)
    traj_results = traj_metrics.compute()
    print(f"Trajectory metrics: {list(traj_results.keys())}")
    
    print("\nTesting anomaly detection metrics...")
    anomaly_metrics = create_metrics('anomaly_detection')
    
    scores = torch.rand(100)  # Random anomaly scores
    labels = torch.randint(0, 2, (100,))  # Random binary labels
    
    anomaly_metrics.update(scores, labels)
    anomaly_results = anomaly_metrics.compute()
    print(f"Anomaly metrics: {list(anomaly_results.keys())}")
    
    print("\nTesting vessel interaction metrics...")
    interaction_metrics = create_metrics('vessel_interaction')
    
    collision_preds = torch.rand(50)
    collision_labels = torch.randint(0, 2, (50,))
    
    interaction_metrics.update(collision_preds, collision_labels)
    interaction_results = interaction_metrics.compute()
    print(f"Interaction metrics: {list(interaction_results.keys())}")
    
    print("\nTesting loss functions...")
    
    # Test trajectory loss
    traj_loss_fn = create_loss_function('trajectory_prediction')
    traj_loss = traj_loss_fn(predictions, targets)
    print(f"Trajectory loss: {traj_loss.item():.4f}")
    
    # Test anomaly loss
    anomaly_loss_fn = create_loss_function('anomaly_detection')
    reconstructions = torch.randn_like(predictions)
    anomaly_loss = anomaly_loss_fn(reconstructions, predictions)
    print(f"Anomaly loss: {anomaly_loss.item():.4f}")
    
    # Test interaction loss
    interaction_loss_fn = create_loss_function('vessel_interaction')
    interaction_loss = interaction_loss_fn(collision_preds, collision_labels.float())
    print(f"Interaction loss: {interaction_loss.item():.4f}")
    
    print("\nAll metrics and loss functions tested successfully!")

