"""Camera projection utilities with differentiable gradients for TLC-Calib.

This module provides differentiable projection operations that enable
gradient flow from rendered images back to camera pose parameters.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class CameraProjection:
    """Differentiable camera projection with pose gradients.

    This class implements the projection of 3D Gaussians to 2D,
    maintaining gradient flow to the camera pose parameters T^e.

    From the paper:
        - Gradients flow through: 2D mean, 2D covariance, view-dependent color
        - Uses analytical Jacobians for SE(3) pose updates
    """

    @staticmethod
    def transform_points(
        points: Tensor,
        T_lidar_to_cam: Tensor,
    ) -> Tensor:
        """Transform points from LiDAR to camera frame.

        Args:
            points: 3D points of shape (N, 3) in LiDAR frame
            T_lidar_to_cam: 4x4 transformation matrix

        Returns:
            Transformed points of shape (N, 3) in camera frame
        """
        R = T_lidar_to_cam[:3, :3]
        t = T_lidar_to_cam[:3, 3]
        return (R @ points.T).T + t

    @staticmethod
    def project_to_image(
        points_cam: Tensor,
        K: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Project camera-frame points to image coordinates.

        Args:
            points_cam: Points in camera frame of shape (N, 3)
            K: 3x3 camera intrinsic matrix

        Returns:
            Tuple of:
                - pixels: 2D pixel coordinates of shape (N, 2)
                - depth: Depth values of shape (N,)
        """
        # Extract depth (Z coordinate)
        depth = points_cam[:, 2]

        # Avoid division by zero
        safe_depth = torch.where(
            depth.abs() > 1e-8,
            depth,
            torch.ones_like(depth) * 1e-8 * torch.sign(depth + 1e-8)
        )

        # Normalize by depth
        x = points_cam[:, 0] / safe_depth
        y = points_cam[:, 1] / safe_depth

        # Apply intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = fx * x + cx
        v = fy * y + cy

        return torch.stack([u, v], dim=-1), depth

    @staticmethod
    def compute_2d_covariance(
        cov_3d: Tensor,
        points_cam: Tensor,
        K: Tensor,
        T_lidar_to_cam: Tensor,
    ) -> Tensor:
        """Compute 2D covariance matrices for Gaussian splatting.

        Following the EWA splatting formulation, the 2D covariance is:
            Σ_2D = J @ W @ Σ_3D @ W^T @ J^T

        where:
            - J is the Jacobian of the projection
            - W is the rotation component of T_lidar_to_cam

        Args:
            cov_3d: 3D covariance matrices of shape (N, 3, 3)
            points_cam: Points in camera frame of shape (N, 3)
            K: Camera intrinsic matrix of shape (3, 3)
            T_lidar_to_cam: 4x4 transformation matrix

        Returns:
            2D covariance matrices of shape (N, 2, 2)
        """
        N = points_cam.shape[0]
        device = points_cam.device
        dtype = points_cam.dtype

        # Extract focal lengths
        fx, fy = K[0, 0], K[1, 1]

        # Depth (z coordinate)
        z = points_cam[:, 2]
        safe_z = torch.where(z.abs() > 1e-8, z, torch.ones_like(z) * 1e-8)
        z_sq = safe_z ** 2

        # Jacobian of projection: d(u,v)/d(x,y,z)
        # u = fx * x / z + cx  =>  du/dx = fx/z, du/dy = 0, du/dz = -fx*x/z^2
        # v = fy * y / z + cy  =>  dv/dx = 0, dv/dy = fy/z, dv/dz = -fy*y/z^2
        x, y = points_cam[:, 0], points_cam[:, 1]

        J = torch.zeros(N, 2, 3, device=device, dtype=dtype)
        J[:, 0, 0] = fx / safe_z
        J[:, 0, 2] = -fx * x / z_sq
        J[:, 1, 1] = fy / safe_z
        J[:, 1, 2] = -fy * y / z_sq

        # Rotation component
        W = T_lidar_to_cam[:3, :3]  # (3, 3)

        # Transform covariance: J @ W @ Σ @ W^T @ J^T
        # Σ_world = W @ Σ_3d @ W^T (covariance in camera frame)
        # Σ_2d = J @ Σ_world @ J^T
        cov_world = W @ cov_3d @ W.T  # (N, 3, 3)
        cov_2d = J @ cov_world @ J.transpose(-1, -2)  # (N, 2, 2)

        return cov_2d

    @staticmethod
    def build_covariance_from_scale_rotation(
        scales: Tensor,
        rotations: Tensor,
    ) -> Tensor:
        """Build 3D covariance matrices from scale and rotation.

        Σ = R @ S @ S^T @ R^T

        where S = diag(scales)

        Args:
            scales: Scale parameters of shape (N, 3)
            rotations: Quaternions of shape (N, 4) in (x, y, z, w) format

        Returns:
            Covariance matrices of shape (N, 3, 3)
        """
        N = scales.shape[0]
        device = scales.device
        dtype = scales.dtype

        # Convert quaternion to rotation matrix
        x, y, z, w = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]

        R = torch.zeros(N, 3, 3, device=device, dtype=dtype)
        R[:, 0, 0] = 1 - 2*(y*y + z*z)
        R[:, 0, 1] = 2*(x*y - z*w)
        R[:, 0, 2] = 2*(x*z + y*w)
        R[:, 1, 0] = 2*(x*y + z*w)
        R[:, 1, 1] = 1 - 2*(x*x + z*z)
        R[:, 1, 2] = 2*(y*z - x*w)
        R[:, 2, 0] = 2*(x*z - y*w)
        R[:, 2, 1] = 2*(y*z + x*w)
        R[:, 2, 2] = 1 - 2*(x*x + y*y)

        # Build scale matrix (actually S @ S^T = diag(scales^2))
        # For numerical stability, we can use exp(scales) if scales are in log space
        s = scales.exp() if scales.min() < 0 else scales  # Assume log space if negative
        S = torch.diag_embed(s)  # (N, 3, 3)

        # Σ = R @ S @ S^T @ R^T = R @ diag(s^2) @ R^T
        cov = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)

        return cov

    @staticmethod
    def filter_visible_gaussians(
        points_cam: Tensor,
        pixels: Tensor,
        depth: Tensor,
        image_height: int,
        image_width: int,
        near: float = 0.1,
        far: float = 100.0,
    ) -> Tensor:
        """Filter Gaussians to those visible in the image.

        Args:
            points_cam: Points in camera frame of shape (N, 3)
            pixels: 2D pixel coordinates of shape (N, 2)
            depth: Depth values of shape (N,)
            image_height: Image height in pixels
            image_width: Image width in pixels
            near: Near clipping plane
            far: Far clipping plane

        Returns:
            Boolean mask of shape (N,) for visible Gaussians
        """
        # Depth check
        in_depth = (depth > near) & (depth < far)

        # Image bounds check (with margin for Gaussian extent)
        margin = 100  # pixels
        in_bounds = (
            (pixels[:, 0] > -margin) &
            (pixels[:, 0] < image_width + margin) &
            (pixels[:, 1] > -margin) &
            (pixels[:, 1] < image_height + margin)
        )

        return in_depth & in_bounds


def compute_pose_jacobian(
    points_lidar: Tensor,
    T_lidar_to_cam: Tensor,
) -> Tensor:
    """Compute Jacobian of camera-frame points w.r.t. pose parameters.

    The pose is parameterized as se(3) = [translation, rotation].
    This enables gradient computation for pose optimization.

    From the paper (Eq. 4):
        ∂μ_i^c / ∂T^c = [I | -[μ_i^c]_×]

    Args:
        points_lidar: Points in LiDAR frame of shape (N, 3)
        T_lidar_to_cam: 4x4 transformation matrix

    Returns:
        Jacobian of shape (N, 3, 6) where the last dim is [t_x, t_y, t_z, r_x, r_y, r_z]
    """
    N = points_lidar.shape[0]
    device = points_lidar.device
    dtype = points_lidar.dtype

    # Transform points to camera frame
    R = T_lidar_to_cam[:3, :3]
    t = T_lidar_to_cam[:3, 3]
    points_cam = (R @ points_lidar.T).T + t  # (N, 3)

    # Jacobian w.r.t. translation: ∂p_cam/∂t = I
    # Jacobian w.r.t. rotation: ∂p_cam/∂ω = -[p_cam]_×

    J = torch.zeros(N, 3, 6, device=device, dtype=dtype)

    # Translation part (first 3 columns)
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 1.0
    J[:, 2, 2] = 1.0

    # Rotation part (last 3 columns) - negative skew-symmetric of points_cam
    # -[p]_× = [[0, p_z, -p_y], [-p_z, 0, p_x], [p_y, -p_x, 0]]
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    J[:, 0, 4] = z   # ∂x/∂ω_y
    J[:, 0, 5] = -y  # ∂x/∂ω_z
    J[:, 1, 3] = -z  # ∂y/∂ω_x
    J[:, 1, 5] = x   # ∂y/∂ω_z
    J[:, 2, 3] = y   # ∂z/∂ω_x
    J[:, 2, 4] = -x  # ∂z/∂ω_y

    return J


def gaussian_2d(
    pixels: Tensor,
    means_2d: Tensor,
    cov_2d: Tensor,
) -> Tensor:
    """Evaluate 2D Gaussian density at pixel locations.

    Args:
        pixels: Pixel coordinates of shape (H, W, 2) or (N, 2)
        means_2d: Gaussian centers of shape (M, 2)
        cov_2d: 2D covariance matrices of shape (M, 2, 2)

    Returns:
        Density values of shape (H, W, M) or (N, M)
    """
    # Compute inverse covariance
    cov_inv = torch.linalg.inv(cov_2d + 1e-6 * torch.eye(2, device=cov_2d.device))  # (M, 2, 2)

    # Compute difference
    if pixels.dim() == 3:  # (H, W, 2)
        H, W = pixels.shape[:2]
        diff = pixels.reshape(-1, 1, 2) - means_2d.unsqueeze(0)  # (H*W, M, 2)
    else:  # (N, 2)
        diff = pixels.unsqueeze(1) - means_2d.unsqueeze(0)  # (N, M, 2)

    # Mahalanobis distance: (x - μ)^T @ Σ^-1 @ (x - μ)
    mahal = torch.einsum('nmi,mij,nmj->nm', diff, cov_inv, diff)

    # Gaussian density
    density = torch.exp(-0.5 * mahal)

    if pixels.dim() == 3:
        density = density.reshape(H, W, -1)

    return density
