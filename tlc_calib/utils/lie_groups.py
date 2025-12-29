"""Lie group utilities for SO(3) and SE(3) operations.

This module provides differentiable operations for rotation and pose optimization
on the SO(3) and SE(3) manifolds using axis-angle parameterization.
"""

import torch
from torch import Tensor


def skew_symmetric(v: Tensor) -> Tensor:
    """Create a skew-symmetric matrix from a 3D vector.

    Args:
        v: 3D vector of shape (..., 3)

    Returns:
        Skew-symmetric matrix of shape (..., 3, 3) such that [v]_x @ u = v x u
    """
    # Handle batched inputs
    batch_shape = v.shape[:-1]
    v = v.reshape(-1, 3)

    zero = torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)
    skew = torch.stack(
        [
            torch.stack([zero, -v[:, 2], v[:, 1]], dim=-1),
            torch.stack([v[:, 2], zero, -v[:, 0]], dim=-1),
            torch.stack([-v[:, 1], v[:, 0], zero], dim=-1),
        ],
        dim=-2,
    )

    return skew.reshape(*batch_shape, 3, 3)


def axis_angle_to_rotation_matrix(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle representation to rotation matrix using Rodrigues' formula.

    The axis-angle representation is a 3D vector where the direction is the rotation
    axis and the magnitude is the rotation angle in radians.

    Args:
        axis_angle: Axis-angle vector of shape (..., 3)

    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    batch_shape = axis_angle.shape[:-1]
    axis_angle = axis_angle.reshape(-1, 3)

    # Compute rotation angle (magnitude of axis-angle vector)
    theta = torch.norm(axis_angle, dim=-1, keepdim=True)  # (N, 1)
    theta_sq = theta * theta

    # Normalized rotation axis (handle zero rotation case)
    safe_theta = torch.where(theta > 1e-8, theta, torch.ones_like(theta))
    k = axis_angle / safe_theta  # (N, 3)

    # Skew-symmetric matrix [k]_x
    K = skew_symmetric(k)  # (N, 3, 3)

    # Rodrigues' formula: R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    # For small angles, use Taylor expansion for numerical stability
    sin_theta = torch.sin(theta).unsqueeze(-1)  # (N, 1, 1)
    cos_theta = torch.cos(theta).unsqueeze(-1)  # (N, 1, 1)

    # Identity matrix
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    I = I.unsqueeze(0).expand(axis_angle.shape[0], -1, -1)

    # For small angles (theta < 1e-8), R ≈ I + theta * K
    # For larger angles, use full Rodrigues formula
    R = I + sin_theta * K + (1 - cos_theta) * (K @ K)

    # Handle zero rotation case
    zero_rot_mask = (theta < 1e-8).squeeze(-1)
    if zero_rot_mask.any():
        R[zero_rot_mask] = I[zero_rot_mask]

    return R.reshape(*batch_shape, 3, 3)


def rotation_matrix_to_axis_angle(R: Tensor) -> Tensor:
    """Convert rotation matrix to axis-angle representation.

    Args:
        R: Rotation matrix of shape (..., 3, 3)

    Returns:
        Axis-angle vector of shape (..., 3)
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    # Compute rotation angle from trace: trace(R) = 1 + 2*cos(theta)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)  # Numerical stability
    theta = torch.acos(cos_theta)  # (N,)

    # Extract rotation axis from skew-symmetric part: [k]_x = (R - R^T) / (2*sin(theta))
    skew = (R - R.transpose(-1, -2)) / 2  # (N, 3, 3)
    k = torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1)  # (N, 3)

    # Normalize axis for non-zero rotations
    k_norm = torch.norm(k, dim=-1, keepdim=True)
    safe_k_norm = torch.where(k_norm > 1e-8, k_norm, torch.ones_like(k_norm))
    k = k / safe_k_norm

    # Handle special cases:
    # 1. theta ≈ 0: axis is arbitrary, use [0, 0, 0]
    # 2. theta ≈ pi: need to extract axis from R + I
    zero_mask = theta < 1e-8
    pi_mask = (theta - torch.pi).abs() < 1e-8

    # For theta ≈ pi, axis can be found from columns of R + I
    if pi_mask.any():
        R_plus_I = R[pi_mask] + torch.eye(3, device=R.device, dtype=R.dtype)
        # Find the column with largest norm
        col_norms = torch.norm(R_plus_I, dim=-2)  # (M, 3)
        max_col = torch.argmax(col_norms, dim=-1)  # (M,)
        # Extract and normalize
        for i, (mat, col_idx) in enumerate(zip(R_plus_I, max_col)):
            k[pi_mask][i] = mat[:, col_idx] / torch.norm(mat[:, col_idx])

    # Compute axis-angle vector
    axis_angle = k * theta.unsqueeze(-1)

    # Zero rotation case
    axis_angle[zero_mask] = 0.0

    return axis_angle.reshape(*batch_shape, 3)


def quaternion_to_axis_angle(q: Tensor) -> Tensor:
    """Convert quaternion to axis-angle representation.

    Args:
        q: Quaternion of shape (..., 4) in (x, y, z, w) format

    Returns:
        Axis-angle vector of shape (..., 3)
    """
    batch_shape = q.shape[:-1]
    q = q.reshape(-1, 4)

    # Normalize quaternion
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

    # Extract components (x, y, z, w)
    xyz = q[:, :3]
    w = q[:, 3:4]

    # Ensure w >= 0 for unique representation
    sign = torch.sign(w)
    sign[sign == 0] = 1
    xyz = xyz * sign
    w = w * sign

    # Compute angle: theta = 2 * acos(w)
    w = torch.clamp(w, -1, 1)
    theta = 2 * torch.acos(w)  # (N, 1)

    # Compute axis: k = xyz / sin(theta/2)
    sin_half_theta = torch.sin(theta / 2)
    safe_sin = torch.where(sin_half_theta.abs() > 1e-8, sin_half_theta, torch.ones_like(sin_half_theta))
    k = xyz / safe_sin

    # Normalize axis
    k_norm = torch.norm(k, dim=-1, keepdim=True)
    safe_k_norm = torch.where(k_norm > 1e-8, k_norm, torch.ones_like(k_norm))
    k = k / safe_k_norm

    # Compute axis-angle
    axis_angle = k * theta

    # Handle zero rotation
    zero_mask = theta.squeeze(-1) < 1e-8
    axis_angle[zero_mask] = 0.0

    return axis_angle.reshape(*batch_shape, 3)


def axis_angle_to_quaternion(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle to quaternion representation.

    Args:
        axis_angle: Axis-angle vector of shape (..., 3)

    Returns:
        Quaternion of shape (..., 4) in (x, y, z, w) format
    """
    batch_shape = axis_angle.shape[:-1]
    axis_angle = axis_angle.reshape(-1, 3)

    # Compute angle and axis
    theta = torch.norm(axis_angle, dim=-1, keepdim=True)  # (N, 1)
    safe_theta = torch.where(theta > 1e-8, theta, torch.ones_like(theta))
    k = axis_angle / safe_theta  # (N, 3)

    # Quaternion components
    half_theta = theta / 2
    w = torch.cos(half_theta)  # (N, 1)
    xyz = k * torch.sin(half_theta)  # (N, 3)

    # Handle zero rotation
    zero_mask = theta.squeeze(-1) < 1e-8
    xyz[zero_mask] = 0.0
    w[zero_mask] = 1.0

    q = torch.cat([xyz, w], dim=-1)  # (N, 4)
    return q.reshape(*batch_shape, 4)


def compose_rotations(R1: Tensor, R2: Tensor) -> Tensor:
    """Compose two rotations: R = R1 @ R2.

    Args:
        R1: First rotation matrix of shape (..., 3, 3)
        R2: Second rotation matrix of shape (..., 3, 3)

    Returns:
        Composed rotation matrix of shape (..., 3, 3)
    """
    return R1 @ R2


def invert_rotation(R: Tensor) -> Tensor:
    """Invert a rotation matrix: R^(-1) = R^T.

    Args:
        R: Rotation matrix of shape (..., 3, 3)

    Returns:
        Inverted rotation matrix of shape (..., 3, 3)
    """
    return R.transpose(-1, -2)


def transform_points(points: Tensor, R: Tensor, t: Tensor) -> Tensor:
    """Transform points by rotation R and translation t.

    Args:
        points: Points of shape (..., N, 3) or (..., 3)
        R: Rotation matrix of shape (..., 3, 3)
        t: Translation vector of shape (..., 3)

    Returns:
        Transformed points of shape (..., N, 3) or (..., 3)
    """
    if points.dim() == 1:
        return R @ points + t
    elif points.dim() == 2:
        # (N, 3) -> (N, 3)
        return (R @ points.unsqueeze(-1)).squeeze(-1) + t
    else:
        # (..., N, 3) -> (..., N, 3)
        return (R @ points.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(-2)


def invert_transform(R: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Invert a rigid transformation.

    Given T = [R | t], compute T^(-1) = [R^T | -R^T @ t]

    Args:
        R: Rotation matrix of shape (..., 3, 3)
        t: Translation vector of shape (..., 3)

    Returns:
        Tuple of (R_inv, t_inv) where:
            R_inv: Inverted rotation of shape (..., 3, 3)
            t_inv: Inverted translation of shape (..., 3)
    """
    R_inv = R.transpose(-1, -2)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
    return R_inv, t_inv
