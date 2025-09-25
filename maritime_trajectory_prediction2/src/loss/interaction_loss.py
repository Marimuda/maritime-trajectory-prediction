"""
Maritime vessel interaction loss function.

Combines collision prediction and optional interaction strength modeling with configurable weights.

Function:
- interaction_loss
"""

import torch
import torch.nn.functional as F


def interaction_loss(
    collision_preds: torch.Tensor,
    collision_labels: torch.Tensor,
    interaction_scores: torch.Tensor | None = None,
    interaction_targets: torch.Tensor | None = None,
    collision_weight: float = 1.0,
    interaction_weight: float = 0.5,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute maritime vessel interaction loss.

    Args:
        collision_preds: Predicted collision logits or probabilities, shape (B, ...)
        collision_labels: True collision labels, same shape as preds
        interaction_scores: Optional predicted interaction strength, shape (B, ...)
        interaction_targets: Optional true interaction strength targets, same shape as scores
        collision_weight: Weight for collision classification loss
        interaction_weight: Weight for interaction regression loss
        class_weights: Optional weights for binary classification loss

    Returns:
        Scalar loss tensor
    """
    # Collision prediction loss (binary classification)
    if class_weights is not None:
        coll_loss = F.binary_cross_entropy_with_logits(
            collision_preds, collision_labels.float(), weight=class_weights
        )
    else:
        coll_loss = F.binary_cross_entropy_with_logits(
            collision_preds, collision_labels.float()
        )
    loss = collision_weight * coll_loss

    # Interaction modeling loss (optional regression)
    if interaction_scores is not None and interaction_targets is not None:
        int_loss = F.mse_loss(interaction_scores, interaction_targets)
        loss = loss + interaction_weight * int_loss

    return loss
