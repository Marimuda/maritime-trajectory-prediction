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

import torch
from torch import tensor
from torchmetrics import Metric

EARTH_RADIUS_KM = 6371.0


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
        diff = torch.where(diff > 180, diff - 360, diff)
        diff = torch.where(diff < -180, diff + 360, diff)
        self.sum_sq += (diff**2).sum()
        self.count += diff.numel()

    def compute(self):
        return torch.sqrt(self.sum_sq / self.count) if self.count > 0 else tensor(0.0)
