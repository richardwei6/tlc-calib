"""Coordinate transformation utilities for TLC-Calib.

This module provides functions for converting between different rotation
representations (Euler angles, quaternions, rotation matrices) and computing
extrinsic transformations.
"""

import numpy as np
import torch
from torch import Tensor


def euler_to_rotation_matrix(
    roll: float | Tensor,
    pitch: float | Tensor,
    yaw: float | Tensor,
    degrees: bool = True,
) -> Tensor:
    """Convert Euler angles (XYZ extrinsic convention) to rotation matrix.

    This matches the scipy 'xyz' extrinsic convention used in convert_lucid_calib.py.

    Args:
        roll: Rotation around X axis
        pitch: Rotation around Y axis
        yaw: Rotation around Z axis
        degrees: If True, angles are in degrees; if False, in radians

    Returns:
        Rotation matrix of shape (3, 3)
    """
    # Convert to tensors if needed
    if isinstance(roll, (int, float)):
        roll = torch.tensor(roll)
    if isinstance(pitch, (int, float)):
        pitch = torch.tensor(pitch)
    if isinstance(yaw, (int, float)):
        yaw = torch.tensor(yaw)

    # Convert to radians if needed
    if degrees:
        roll = torch.deg2rad(roll)
        pitch = torch.deg2rad(pitch)
        yaw = torch.deg2rad(yaw)

    # Rotation matrices for each axis
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    # X rotation (roll)
    Rx = torch.tensor([
        [1, 0, 0],
        [0, cos_r, -sin_r],
        [0, sin_r, cos_r],
    ], dtype=roll.dtype, device=roll.device if roll.is_cuda else None)

    # Y rotation (pitch)
    Ry = torch.tensor([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p],
    ], dtype=pitch.dtype, device=pitch.device if pitch.is_cuda else None)

    # Z rotation (yaw)
    Rz = torch.tensor([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1],
    ], dtype=yaw.dtype, device=yaw.device if yaw.is_cuda else None)

    # Extrinsic XYZ: R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx
    return R


def quaternion_to_rotation_matrix(q: Tensor | dict) -> Tensor:
    """Convert quaternion to rotation matrix.

    Args:
        q: Quaternion as tensor of shape (4,) in (x, y, z, w) format,
           or dict with keys 'x', 'y', 'z', 'w'

    Returns:
        Rotation matrix of shape (3, 3)
    """
    # Handle dict input (from JSON calibration files)
    if isinstance(q, dict):
        q = torch.tensor([q['x'], q['y'], q['z'], q['w']])

    # Normalize quaternion
    q = q / (torch.norm(q) + 1e-8)
    x, y, z, w = q[0], q[1], q[2], q[3]

    # Rotation matrix from quaternion
    R = torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=q.dtype, device=q.device if q.is_cuda else None)

    return R


def compute_lidar_to_camera_transform(
    roll: float,
    pitch: float,
    yaw: float,
    px: float,
    py: float,
    pz: float,
) -> Tensor:
    """Compute the 4x4 transformation matrix from LiDAR to camera frame.

    This matches the transformation in convert_lucid_calib.py:
        - Input: Camera-to-LiDAR extrinsics (roll, pitch, yaw, px, py, pz)
        - Output: LiDAR-to-Camera transformation matrix

    The transformation is: P_cam = R^(-1) * (P_lidar - t)
    So T_lidar_to_cam = [R^(-1) | -R^(-1)*t]

    Args:
        roll: Roll angle in degrees
        pitch: Pitch angle in degrees
        yaw: Yaw angle in degrees
        px: X translation (camera to LiDAR)
        py: Y translation (camera to LiDAR)
        pz: Z translation (camera to LiDAR)

    Returns:
        4x4 transformation matrix T_lidar_to_cam
    """
    # Compute rotation matrix (camera-to-lidar)
    rot_cam_to_lidar = euler_to_rotation_matrix(roll, pitch, yaw, degrees=True)
    t_cam_to_lidar = torch.tensor([px, py, pz], dtype=rot_cam_to_lidar.dtype)

    # Compute inverse transformation (lidar-to-camera)
    rot_lidar_to_cam = rot_cam_to_lidar.T
    t_lidar_to_cam = -rot_lidar_to_cam @ t_cam_to_lidar

    # Build 4x4 transformation matrix
    T = torch.eye(4, dtype=rot_lidar_to_cam.dtype)
    T[:3, :3] = rot_lidar_to_cam
    T[:3, 3] = t_lidar_to_cam

    return T


def compute_camera_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    is_wide: bool = False,
    wide_offset_y: float = 0.0,
    downsample: int = 1,
) -> Tensor:
    """Compute camera intrinsic matrix K with optional wide camera handling.

    For wide cameras, applies the transformation:
        cx = 2 * cx_original
        cy = 2 * cy_original - wide_offset_y

    Then scales by downsample factor if needed.

    Args:
        fx: Focal length X
        fy: Focal length Y
        cx: Principal point X
        cy: Principal point Y
        is_wide: Whether this is a wide camera
        wide_offset_y: Y offset for wide cameras
        downsample: Downsample factor (1 = no downsampling)

    Returns:
        3x3 camera intrinsic matrix K
    """
    if is_wide:
        # Wide camera intrinsics handling
        cx_adj = 2 * cx
        cy_adj = 2 * cy - wide_offset_y
    else:
        cx_adj = cx
        cy_adj = cy

    # Apply downsampling
    fx_scaled = fx / downsample
    fy_scaled = fy / downsample
    cx_scaled = cx_adj / downsample
    cy_scaled = cy_adj / downsample

    K = torch.tensor([
        [fx_scaled, 0.0, cx_scaled],
        [0.0, fy_scaled, cy_scaled],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    return K


def project_points(
    points: Tensor,
    K: Tensor,
    T_lidar_to_cam: Tensor,
) -> tuple[Tensor, Tensor]:
    """Project 3D points to 2D image coordinates.

    Args:
        points: 3D points in LiDAR frame of shape (N, 3)
        K: Camera intrinsic matrix of shape (3, 3)
        T_lidar_to_cam: Transformation matrix of shape (4, 4)

    Returns:
        Tuple of:
            - 2D pixel coordinates of shape (N, 2)
            - Depth values of shape (N,)
    """
    N = points.shape[0]

    # Transform to camera coordinates
    R = T_lidar_to_cam[:3, :3]
    t = T_lidar_to_cam[:3, 3]
    points_cam = (R @ points.T).T + t  # (N, 3)

    # Extract depth (Z in camera frame)
    depth = points_cam[:, 2]

    # Project to image plane
    # u = fx * X/Z + cx
    # v = fy * Y/Z + cy
    x = points_cam[:, 0] / (depth + 1e-8)
    y = points_cam[:, 1] / (depth + 1e-8)

    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]

    pixels = torch.stack([u, v], dim=-1)  # (N, 2)

    return pixels, depth


def get_view_direction(
    T_lidar_to_cam: Tensor,
    points: Tensor | None = None,
) -> Tensor:
    """Compute view direction from camera center to points (or camera forward direction).

    Args:
        T_lidar_to_cam: Transformation matrix of shape (4, 4)
        points: Optional 3D points of shape (N, 3) in LiDAR frame.
                If None, returns camera forward direction.

    Returns:
        Normalized view direction(s) of shape (3,) or (N, 3)
    """
    # Camera center in LiDAR frame: o_c = -R^T @ t
    R = T_lidar_to_cam[:3, :3]
    t = T_lidar_to_cam[:3, 3]
    camera_center = -R.T @ t  # (3,)

    if points is None:
        # Return camera forward direction (negative Z in camera frame -> LiDAR frame)
        # Forward in camera frame is [0, 0, 1], transform to LiDAR: R^T @ [0,0,1]
        forward = R.T @ torch.tensor([0.0, 0.0, 1.0], dtype=R.dtype, device=R.device)
        return forward / (torch.norm(forward) + 1e-8)
    else:
        # View direction from camera to each point
        view_dirs = points - camera_center.unsqueeze(0)  # (N, 3)
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-8)
        return view_dirs


def normalize_points(points: Tensor, center: bool = True, scale: bool = True) -> tuple[Tensor, dict]:
    """Normalize point cloud to unit sphere.

    Args:
        points: Points of shape (N, 3)
        center: Whether to center points at origin
        scale: Whether to scale to unit sphere

    Returns:
        Tuple of:
            - Normalized points of shape (N, 3)
            - Dict with normalization parameters for inverse transform
    """
    params = {}

    if center:
        centroid = points.mean(dim=0)
        points = points - centroid
        params['centroid'] = centroid
    else:
        params['centroid'] = torch.zeros(3, dtype=points.dtype, device=points.device)

    if scale:
        max_dist = torch.norm(points, dim=-1).max()
        if max_dist > 1e-8:
            points = points / max_dist
            params['scale'] = max_dist
        else:
            params['scale'] = torch.tensor(1.0, dtype=points.dtype, device=points.device)
    else:
        params['scale'] = torch.tensor(1.0, dtype=points.dtype, device=points.device)

    return points, params


def denormalize_points(points: Tensor, params: dict) -> Tensor:
    """Reverse normalization of point cloud.

    Args:
        points: Normalized points of shape (N, 3)
        params: Dict with normalization parameters from normalize_points

    Returns:
        Denormalized points of shape (N, 3)
    """
    points = points * params['scale']
    points = points + params['centroid']
    return points
