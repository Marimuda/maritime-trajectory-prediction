"""
Trajectory prediction metrics using torchmetrics for maritime applications.
Defines:
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)
- RMSEPosition (Root Mean Square Error over positions)
- CourseRMSE (circular RMSE for course)

Usage:
```python
from torchmetrics import MetricCollection
from metrics.trajectory_metrics import ADE, FDE, RMSEPosition, CourseRMSE

metrics = MetricCollection({
    'ade_km': ADE(),
    'fde_km': FDE(),
    'rmse_position_km': RMSEPosition(),
    'rmse_course_deg': CourseRMSE(),
})

# In LightningModule:
# self.train_metrics = metrics.clone(prefix='train/')
# self.val_metrics   = metrics.clone(prefix='val/')
```
"""

import pandas as pd
import torch
from torch import tensor
from torchmetrics import Metric

EARTH_RADIUS_KM = 6371.0
DEGREES_HALF_CIRCLE = 180
DEGREES_FULL_CIRCLE = 360
MIN_TRAJECTORY_POINTS = 2


def _haversine(lat1, lon1, lat2, lon2):
    # Convert to radians
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c


class ADE(Metric):
    """Average Displacement Error in kilometers."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_dist", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):  # both shape [B, T, 2]
        lat1, lon1 = preds[..., 0], preds[..., 1]
        lat2, lon2 = target[..., 0], target[..., 1]
        dist = _haversine(lat1, lon1, lat2, lon2)
        self.sum_dist += dist.sum()
        self.count += dist.numel()

    def compute(self):
        return self.sum_dist / self.count if self.count > 0 else tensor(0.0)


class FDE(Metric):
    """Final Displacement Error in kilometers."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_dist", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):  # both shape [B, T, 2]
        pred_last = preds[:, -1, :]
        tgt_last = target[:, -1, :]
        dist = _haversine(
            pred_last[..., 0], pred_last[..., 1], tgt_last[..., 0], tgt_last[..., 1]
        )
        self.sum_dist += dist.sum()
        self.count += dist.numel()

    def compute(self):
        return self.sum_dist / self.count if self.count > 0 else tensor(0.0)


class RMSEPosition(Metric):
    """Root Mean Square Error for positions in kilometers."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_sq", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):  # [B, T, 2]
        lat1, lon1 = preds[..., 0], preds[..., 1]
        lat2, lon2 = target[..., 0], target[..., 1]
        dist = _haversine(lat1, lon1, lat2, lon2)
        self.sum_sq += (dist**2).sum()
        self.count += dist.numel()

    def compute(self):
        return torch.sqrt(self.sum_sq / self.count) if self.count > 0 else tensor(0.0)


class CourseRMSE(Metric):
    """Circular RMSE for course (degrees)."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_sq", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):  # [B, T]
        diff = preds - target
        diff = torch.where(diff > DEGREES_HALF_CIRCLE, diff - DEGREES_FULL_CIRCLE, diff)
        diff = torch.where(
            diff < -DEGREES_HALF_CIRCLE, diff + DEGREES_FULL_CIRCLE, diff
        )
        self.sum_sq += (diff**2).sum()
        self.count += diff.numel()

    def compute(self):
        return torch.sqrt(self.sum_sq / self.count) if self.count > 0 else tensor(0.0)


class TrajectoryMetrics:
    """
    Unified trajectory metrics interface expected by tests.
    Composes existing modular metrics following repository philosophy.
    """

    def __init__(self):
        """Initialize metrics collection."""
        self.ade = ADE()
        self.fde = FDE()
        self.rmse_position = RMSEPosition()
        self.course_rmse = CourseRMSE()

    def calculate_trajectory_length(self, trajectory):
        """
        Calculate total trajectory length.

        Args:
            trajectory: Array-like of (lat, lon) points

        Returns:
            Total length in kilometers or NaN if calculation fails
        """
        try:
            if len(trajectory) < MIN_TRAJECTORY_POINTS:
                return 0.0

            # Convert to tensor if needed
            if not torch.is_tensor(trajectory):
                trajectory = torch.tensor(trajectory, dtype=torch.float32)

            total_length = 0.0
            for i in range(1, len(trajectory)):
                lat1, lon1 = trajectory[i - 1, 0], trajectory[i - 1, 1]
                lat2, lon2 = trajectory[i, 0], trajectory[i, 1]
                total_length += _haversine(lat1, lon1, lat2, lon2).item()

            return total_length
        except Exception:
            return float("nan")

    def calculate_average_speed(self, trajectory, timestamps=None):
        """
        Calculate average speed along trajectory.

        Args:
            trajectory: Array-like of (lat, lon) points
            timestamps: Optional timestamps for speed calculation

        Returns:
            Average speed in km/h or NaN if calculation fails
        """
        try:
            if len(trajectory) < MIN_TRAJECTORY_POINTS:
                return 0.0

            total_length = self.calculate_trajectory_length(trajectory)
            if pd.isna(total_length) or total_length == 0.0:
                return 0.0

            if timestamps is not None and len(timestamps) == len(trajectory):
                # Calculate based on actual time differences
                total_time = timestamps[-1] - timestamps[0]
                if hasattr(total_time, "total_seconds"):
                    total_hours = total_time.total_seconds() / 3600.0
                else:
                    total_hours = total_time / 3600.0  # Assume seconds
            else:
                # Assume 1-minute intervals if no timestamps
                total_hours = (len(trajectory) - 1) / 60.0

            return total_length / total_hours if total_hours > 0 else 0.0
        except Exception:
            return float("nan")

    def calculate_displacement(self, trajectory):
        """
        Calculate total displacement (straight-line distance).

        Args:
            trajectory: Array-like of (lat, lon) points

        Returns:
            Displacement in kilometers or NaN if calculation fails
        """
        try:
            if len(trajectory) < MIN_TRAJECTORY_POINTS:
                return 0.0

            if not torch.is_tensor(trajectory):
                trajectory = torch.tensor(trajectory, dtype=torch.float32)

            start_lat, start_lon = trajectory[0, 0], trajectory[0, 1]
            end_lat, end_lon = trajectory[-1, 0], trajectory[-1, 1]

            return _haversine(start_lat, start_lon, end_lat, end_lon).item()
        except Exception:
            return float("nan")
