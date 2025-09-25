"""
Maritime anomaly detection loss function.

Combines reconstruction loss with optional supervised anomaly labeling.

Function:
- anomaly_loss
"""

import torch
import torch.nn.functional as F


def anomaly_loss(
    reconstructions: torch.Tensor,
    inputs: torch.Tensor,
    anomaly_labels: torch.Tensor | None = None,
    reconstruction_weight: float = 1.0,
    regularization_weight: float = 0.1,
) -> torch.Tensor:
    """
    Compute loss for maritime anomaly detection.

    Args:
        reconstructions: Model reconstructions, shape (B, *)
        inputs: Original inputs, same shape as reconstructions
        anomaly_labels: Optional binary labels (1=anomaly, 0=normal), shape (B,)
        reconstruction_weight: Weight for reconstruction MSE
        regularization_weight: Weight for L1 regularization on reconstructions

    Returns:
        Scalar loss tensor
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructions, inputs)

    # Regularization: encourage small absolute reconstructions (sparsity)
    reg_loss = torch.mean(torch.abs(reconstructions))

    total_loss = reconstruction_weight * recon_loss + regularization_weight * reg_loss

    # Add supervised anomaly component if labels provided
    if anomaly_labels is not None:
        # Compute per-sample reconstruction error
        # Flatten features per sample
        err = F.mse_loss(reconstructions, inputs, reduction="none")
        err = err.view(err.size(0), -1).mean(dim=1)
        # Binary classification on reconstruction error
        supervised_loss = F.binary_cross_entropy_with_logits(
            err, anomaly_labels.float()
        )
        total_loss = total_loss + supervised_loss

    return total_loss
