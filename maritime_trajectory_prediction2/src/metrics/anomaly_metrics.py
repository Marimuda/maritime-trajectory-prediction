"""
Anomaly detection metrics using torchmetrics for maritime applications.
Defines:
- Precision
- Recall
- F1 Score
- ROC AUC

Usage:
```python
from torchmetrics import MetricCollection
from metrics.anomaly_metrics import create_anomaly_metrics

metrics = create_anomaly_metrics(threshold=0.5)
# In LightningModule:
# self.train_metrics = metrics.clone(prefix='train/')
# self.val_metrics   = metrics.clone(prefix='val/')
```
"""

from torchmetrics import AUROC, F1Score, MetricCollection, Precision, Recall


def create_anomaly_metrics(
    threshold: float = 0.5, zero_division: int = 0, dist_sync_on_step: bool = False
) -> MetricCollection:
    """
    Create a collection of anomaly detection metrics.

    Args:
        threshold: Decision threshold for binary classification.
        zero_division: Value to return when there is no positive/negative sample.
        dist_sync_on_step: Whether to sync metrics across devices on each step.

    Returns:
        MetricCollection with precision, recall, f1_score, and roc_auc.
    """
    metrics = MetricCollection(
        {
            "precision": Precision(
                threshold=threshold,
                zero_division=zero_division,
                dist_sync_on_step=dist_sync_on_step,
            ),
            "recall": Recall(
                threshold=threshold,
                zero_division=zero_division,
                dist_sync_on_step=dist_sync_on_step,
            ),
            "f1_score": F1Score(
                threshold=threshold,
                zero_division=zero_division,
                dist_sync_on_step=dist_sync_on_step,
            ),
            "roc_auc": AUROC(dist_sync_on_step=dist_sync_on_step),
        }
    )
    return metrics
