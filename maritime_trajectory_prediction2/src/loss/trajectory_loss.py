"""
Maritime trajectory prediction loss function.

This module provides a loss that combines position and movement components,
with configurable weighting and optional temporal weights.

Function:
- trajectory_loss
"""

import torch
import torch.nn.functional as F


def trajectory_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    position_weight: float = 1.0,
    movement_weight: float = 0.5,
) -> torch.Tensor:
    """
    Compute maritime trajectory prediction loss.

    Args:
        predictions: Tensor of shape (B, T, 4) [lat, lon, sog, cog]
        targets:     Tensor of same shape as predictions
        weights:     Optional tensor of shape (B, T) for temporal weighting
        position_weight: weight for position MSE component
        movement_weight: weight for movement component

    Returns:
        Scalar loss tensor
    """
    # Split into position and movement
    pred_pos = predictions[..., :2]
    tgt_pos = targets[..., :2]
    pred_mov = predictions[..., 2:]
    tgt_mov = targets[..., 2:]

    # Position loss (MSE across lat/lon)
    pos_loss = F.mse_loss(pred_pos, tgt_pos, reduction="none")  # shape [B, T, 2]

    # Speed (SOG) loss
    speed_loss = F.mse_loss(
        pred_mov[..., 0:1], tgt_mov[..., 0:1], reduction="none"
    )  # shape [B, T, 1]

    # Course (COG) loss with circular handling
    diff = pred_mov[..., 1] - tgt_mov[..., 1]
    diff = torch.where(diff > 180, diff - 360, diff)
    diff = torch.where(diff < -180, diff + 360, diff)
    course_loss = diff.pow(2).unsqueeze(-1)  # shape [B, T, 1]

    mov_loss = torch.cat([speed_loss, course_loss], dim=-1)  # [B, T, 2]

    # Mean losses per element
    mean_pos_loss = pos_loss.mean()  # scalar
    mean_mov_loss = mov_loss.mean()  # scalar

    # Weighted sum
    loss = position_weight * mean_pos_loss + movement_weight * mean_mov_loss

    # Apply temporal weights if provided
    if weights is not None:
        # Expect weights shape [B, T]
        w = weights.unsqueeze(-1)  # [B, T, 1]
        # Recompute weighted position loss
        p = (pos_loss * w).mean()
        m = (mov_loss * w).mean()
        loss = position_weight * p + movement_weight * m

    return loss
