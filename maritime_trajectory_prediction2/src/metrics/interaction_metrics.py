"""
Vessel interaction and collision prediction metrics using torchmetrics.
Defines:
- Collision precision, recall, F1 score, ROC AUC
- Mean interaction strength (continuous)

Usage:
```python
from torchmetrics import MetricCollection
from metrics.interaction_metrics import create_interaction_metrics

metrics = create_interaction_metrics(threshold=0.5)
# In LightningModule:
# self.train_metrics = metrics.clone(prefix='train/')
# self.val_metrics   = metrics.clone(prefix='val/')
```
"""

from torchmetrics import AUROC, F1Score, MeanMetric, MetricCollection, Precision, Recall


def create_interaction_metrics(
    threshold: float = 0.5, dist_sync_on_step: bool = False
) -> MetricCollection:
    """
    Create a collection of vessel interaction metrics.

    Args:
        threshold: Decision threshold for collision prediction.
        dist_sync_on_step: Sync metrics across devices on each step.

    Returns:
        MetricCollection with:
            - collision_precision
            - collision_recall
            - collision_f1
            - collision_auc
            - mean_interaction_strength
    """
    metrics = MetricCollection(
        {
            "collision_precision": Precision(
                threshold=threshold,
                dist_sync_on_step=dist_sync_on_step,
                average="binary",
                zero_division=0,
            ),
            "collision_recall": Recall(
                threshold=threshold,
                dist_sync_on_step=dist_sync_on_step,
                average="binary",
                zero_division=0,
            ),
            "collision_f1": F1Score(
                threshold=threshold,
                dist_sync_on_step=dist_sync_on_step,
                average="binary",
                zero_division=0,
            ),
            "collision_auc": AUROC(pos_label=1, dist_sync_on_step=dist_sync_on_step),
            "mean_interaction_strength": MeanMetric(
                dist_sync_on_step=dist_sync_on_step
            ),
        }
    )
    return metrics
