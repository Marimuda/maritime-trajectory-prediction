"""
Maritime trajectory prediction loss function.

This module provides a loss that combines position and movement components,
with configurable weighting, optional temporal weights, and maritime safety features.

Functions:
- trajectory_loss: Basic trajectory prediction loss
- maritime_safety_loss: Safety-aware loss with CPA/TCPA and vessel dynamics
"""

import torch
import torch.nn.functional as F

# Constants for maritime calculations
HALF_CIRCLE_DEG = 180
MIN_VELOCITY_THRESHOLD = 1e-6
MIN_SEQUENCE_LENGTH = 2
VESSEL_SPEC_TURN_IDX = 2
VESSEL_SPEC_ACCEL_IDX = 3
DEFAULT_MAX_TURN_RATE = 5.0
DEFAULT_MAX_ACCELERATION = 0.5
TIMESTEP_SECONDS = 5.0


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
    diff = torch.where(diff > HALF_CIRCLE_DEG, diff - 360, diff)
    diff = torch.where(diff < -HALF_CIRCLE_DEG, diff + 360, diff)
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


def maritime_safety_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    neighbor_trajectories: torch.Tensor | None = None,
    vessel_specs: torch.Tensor | None = None,
    position_weight: float = 1.0,
    movement_weight: float = 0.5,
    collision_weight: float = 10.0,
    feasibility_weight: float = 5.0,
    safety_horizon_steps: int = 6,
    risk_threshold_meters: float = 500.0,
) -> tuple[torch.Tensor, dict]:
    """
    Maritime safety-aware trajectory prediction loss.

    Extends basic trajectory loss with:
    - Collision risk based on CPA/TCPA
    - Vessel dynamics feasibility
    - Safety-critical horizon weighting

    Args:
        predictions: Tensor of shape (B, T, 4) [lat, lon, sog, cog]
        targets: Tensor of same shape as predictions
        weights: Optional tensor of shape (B, T) for temporal weighting
        neighbor_trajectories: Optional (B, N, T, 4) for collision risk
        vessel_specs: Optional (B, 4) [length, beam, max_turn_rate, max_accel]
        position_weight: Weight for position component
        movement_weight: Weight for movement component
        collision_weight: Weight for collision risk component
        feasibility_weight: Weight for dynamics feasibility
        safety_horizon_steps: Number of steps to prioritize for safety
        risk_threshold_meters: CPA threshold for collision risk

    Returns:
        total_loss: Combined safety-aware loss
        components: Dictionary of individual loss components
    """
    components = {}

    # 1. Basic trajectory loss with safety weighting
    if weights is None:
        # Create safety-weighted temporal weights
        batch_size, seq_len = predictions.shape[:2]
        time_steps = torch.arange(seq_len, device=predictions.device)
        weights = torch.exp(-0.1 * time_steps).unsqueeze(0).repeat(batch_size, 1)
        # Boost weights for safety-critical horizon
        weights[:, :safety_horizon_steps] *= 2.0

    basic_loss = trajectory_loss(
        predictions, targets, weights, position_weight, movement_weight
    )
    components["trajectory"] = basic_loss
    total_loss = basic_loss

    # 2. Collision risk loss
    if neighbor_trajectories is not None:
        collision_loss = _compute_collision_risk(
            predictions,
            neighbor_trajectories,
            safety_horizon_steps,
            risk_threshold_meters,
        )
        components["collision_risk"] = collision_loss
        total_loss += collision_weight * collision_loss

    # 3. Vessel dynamics feasibility
    if vessel_specs is not None:
        feasibility_loss = _compute_feasibility_loss(predictions, vessel_specs)
        components["feasibility"] = feasibility_loss
        total_loss += feasibility_weight * feasibility_loss

    return total_loss, components


def _compute_collision_risk(
    predictions: torch.Tensor,
    neighbors: torch.Tensor,
    safety_horizon: int,
    risk_threshold: float,
) -> torch.Tensor:
    """Compute collision risk based on CPA/TCPA."""
    batch_size, seq_len = predictions.shape[:2]
    n_neighbors = neighbors.shape[1]

    collision_risks = []

    for t in range(1, min(seq_len, safety_horizon * 2)):
        # Current and previous positions for velocity estimation
        own_pos = predictions[:, t, :2]
        own_prev = predictions[:, t - 1, :2]
        neighbor_pos = neighbors[:, :, t, :2]
        neighbor_prev = neighbors[:, :, t - 1, :2]

        # Compute velocities
        own_vel = own_pos - own_prev
        neighbor_vel = neighbor_pos - neighbor_prev

        # Compute CPA for each neighbor
        for n in range(n_neighbors):
            rel_pos = own_pos - neighbor_pos[:, n, :]
            rel_vel = own_vel - neighbor_vel[:, n, :]

            # TCPA calculation
            vel_dot = (rel_vel * rel_vel).sum(dim=-1)
            tcpa = torch.where(
                vel_dot > MIN_VELOCITY_THRESHOLD,
                -(rel_pos * rel_vel).sum(dim=-1) / vel_dot,
                torch.zeros_like(vel_dot),
            )
            tcpa = torch.clamp(tcpa, min=0.0)

            # CPA calculation
            future_rel_pos = rel_pos + tcpa.unsqueeze(-1) * rel_vel
            cpa_distance = torch.norm(future_rel_pos, dim=-1)

            # Convert to meters (approximate)
            cpa_meters = cpa_distance * 111000.0

            # Collision risk score
            risk_score = torch.sigmoid((risk_threshold - cpa_meters) / 100.0)
            time_weight = torch.exp(-0.1 * tcpa)
            weighted_risk = risk_score * time_weight

            collision_risks.append(weighted_risk)

    if collision_risks:
        return torch.stack(collision_risks).mean()
    return torch.tensor(0.0, device=predictions.device)


def _compute_feasibility_loss(
    predictions: torch.Tensor,
    vessel_specs: torch.Tensor,
) -> torch.Tensor:
    """Penalize physically impossible vessel movements."""
    if predictions.shape[1] < MIN_SEQUENCE_LENGTH:
        return torch.tensor(0.0, device=predictions.device)

    # Extract constraints (with defaults)
    max_turn_rate = (
        vessel_specs[:, VESSEL_SPEC_TURN_IDX : VESSEL_SPEC_TURN_IDX + 1]
        if vessel_specs.shape[1] > VESSEL_SPEC_TURN_IDX
        else DEFAULT_MAX_TURN_RATE
    )
    max_acceleration = (
        vessel_specs[:, VESSEL_SPEC_ACCEL_IDX : VESSEL_SPEC_ACCEL_IDX + 1]
        if vessel_specs.shape[1] > VESSEL_SPEC_ACCEL_IDX
        else DEFAULT_MAX_ACCELERATION
    )

    feasibility_penalties = []

    for t in range(1, predictions.shape[1]):
        # Course change rate
        course_change = predictions[:, t, 3] - predictions[:, t - 1, 3]
        # Handle circular wraparound
        course_change = torch.where(
            course_change > HALF_CIRCLE_DEG, course_change - 360, course_change
        )
        course_change = torch.where(
            course_change < -HALF_CIRCLE_DEG, course_change + 360, course_change
        )

        # Penalize excessive turn rates
        turn_rate = torch.abs(course_change) / TIMESTEP_SECONDS
        turn_penalty = F.relu(turn_rate - max_turn_rate.squeeze())

        # Speed change rate
        speed_change = torch.abs(predictions[:, t, 2] - predictions[:, t - 1, 2])
        accel_penalty = F.relu(
            speed_change / TIMESTEP_SECONDS - max_acceleration.squeeze()
        )

        total_penalty = turn_penalty + accel_penalty
        feasibility_penalties.append(total_penalty)

    return torch.stack(feasibility_penalties).mean()
