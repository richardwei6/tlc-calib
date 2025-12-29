"""Adaptive Voxel Control for TLC-Calib.

This module implements the adaptive voxel size selection algorithm from the paper,
which determines the optimal voxel size for creating anchor Gaussians based on
the point cloud density and trajectory length.
"""

from typing import Tuple

import numpy as np
import open3d as o3d
import torch
from torch import Tensor

from ..config import VoxelConfig


class AdaptiveVoxelControl:
    """Adaptive Voxel Control for determining optimal anchor density.

    The algorithm uses binary search to find voxel size ε* that matches
    a target number of voxels N_target = β × trajectory_distance.

    From the paper:
        - β = 5000 (proportionality factor)
        - Binary search to find ε* such that |N(ε) - N_target| < tolerance
    """

    def __init__(self, config: VoxelConfig):
        """Initialize Adaptive Voxel Control.

        Args:
            config: Voxel configuration with parameters
        """
        self.config = config
        self.beta = config.beta
        self.tolerance = config.tolerance
        self.eps_min = config.eps_min
        self.eps_max = config.eps_max
        self.max_iterations = config.max_iterations

    def compute_target_voxels(self, trajectory_distance: float) -> int:
        """Compute target number of voxels.

        Args:
            trajectory_distance: Total trajectory distance in meters

        Returns:
            Target number of voxels N_target
        """
        return int(self.beta * trajectory_distance)

    def count_voxels(self, points: np.ndarray, voxel_size: float) -> int:
        """Count number of occupied voxels for a given voxel size.

        Args:
            points: Point cloud of shape (N, 3)
            voxel_size: Voxel edge length

        Returns:
            Number of occupied voxels
        """
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Create voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

        return len(voxel_grid.get_voxels())

    def find_voxel_size(
        self,
        points: Tensor | np.ndarray,
        n_target: int,
    ) -> float:
        """Find optimal voxel size using binary search.

        Args:
            points: Point cloud of shape (N, 3)
            n_target: Target number of voxels

        Returns:
            Optimal voxel size ε*
        """
        # Convert to numpy if tensor
        if isinstance(points, Tensor):
            points = points.cpu().numpy()

        eps_min = self.eps_min
        eps_max = self.eps_max

        # Binary search
        for _ in range(self.max_iterations):
            eps_mid = (eps_min + eps_max) / 2
            n_voxels = self.count_voxels(points, eps_mid)

            if abs(n_voxels - n_target) < self.tolerance:
                return eps_mid
            elif n_voxels > n_target:
                # Too many voxels, increase voxel size
                eps_min = eps_mid
            else:
                # Too few voxels, decrease voxel size
                eps_max = eps_mid

        # Return best estimate
        return (eps_min + eps_max) / 2

    def compute_voxel_size(
        self,
        points: Tensor | np.ndarray,
        trajectory_distance: float,
    ) -> float:
        """Compute optimal voxel size for given point cloud and trajectory.

        This is the main entry point that combines target computation and binary search.

        Args:
            points: Point cloud of shape (N, 3)
            trajectory_distance: Total trajectory distance in meters

        Returns:
            Optimal voxel size ε*
        """
        # Check for fixed voxel size override
        if self.config.fixed_voxel_size is not None:
            return self.config.fixed_voxel_size

        # Compute target
        n_target = self.compute_target_voxels(trajectory_distance)

        # Find optimal voxel size
        return self.find_voxel_size(points, n_target)

    def voxelize_pointcloud(
        self,
        points: Tensor | np.ndarray,
        voxel_size: float,
    ) -> Tuple[Tensor, Tensor]:
        """Voxelize point cloud and return voxel centers.

        Args:
            points: Point cloud of shape (N, 3)
            voxel_size: Voxel edge length

        Returns:
            Tuple of:
                - voxel_centers: Centers of occupied voxels, shape (M, 3)
                - voxel_indices: Original point indices for each voxel, shape (M, K)
        """
        # Convert to numpy if tensor
        if isinstance(points, Tensor):
            points_np = points.cpu().numpy()
            device = points.device
            dtype = points.dtype
        else:
            points_np = points
            device = torch.device('cpu')
            dtype = torch.float32

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        # Create voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

        # Get voxel centers
        # Each voxel has a grid_index, compute center as: grid_index * voxel_size + voxel_size/2
        voxels = voxel_grid.get_voxels()
        origin = np.array(voxel_grid.origin)

        centers = []
        for voxel in voxels:
            grid_idx = np.array(voxel.grid_index)
            center = origin + (grid_idx + 0.5) * voxel_size
            centers.append(center)

        centers = np.array(centers, dtype=np.float32)

        return torch.from_numpy(centers).to(device=device, dtype=dtype)


def voxelize_with_features(
    points: Tensor,
    features: Tensor | None,
    voxel_size: float,
) -> Tuple[Tensor, Tensor | None]:
    """Voxelize point cloud with optional feature averaging.

    Args:
        points: Point cloud of shape (N, 3)
        features: Optional features of shape (N, C)
        voxel_size: Voxel edge length

    Returns:
        Tuple of:
            - voxel_centers: Shape (M, 3)
            - voxel_features: Shape (M, C) or None
    """
    device = points.device
    dtype = points.dtype

    # Compute voxel indices for each point
    voxel_indices = torch.floor(points / voxel_size).long()  # (N, 3)

    # Create unique voxel keys
    # Use a large multiplier to create unique hash
    max_idx = voxel_indices.max().item() + 1
    keys = (voxel_indices[:, 0] * max_idx * max_idx +
            voxel_indices[:, 1] * max_idx +
            voxel_indices[:, 2])

    # Get unique voxels
    unique_keys, inverse_indices = torch.unique(keys, return_inverse=True)

    # Compute voxel centers by averaging points in each voxel
    num_voxels = len(unique_keys)
    voxel_centers = torch.zeros(num_voxels, 3, device=device, dtype=dtype)
    voxel_counts = torch.zeros(num_voxels, device=device, dtype=dtype)

    # Accumulate points
    voxel_centers.index_add_(0, inverse_indices, points)
    voxel_counts.index_add_(0, inverse_indices, torch.ones(len(points), device=device, dtype=dtype))

    # Average
    voxel_centers = voxel_centers / voxel_counts.unsqueeze(-1)

    # Handle features if provided
    voxel_features = None
    if features is not None:
        voxel_features = torch.zeros(num_voxels, features.shape[1], device=device, dtype=dtype)
        voxel_features.index_add_(0, inverse_indices, features)
        voxel_features = voxel_features / voxel_counts.unsqueeze(-1)

    return voxel_centers, voxel_features


def estimate_trajectory_distance(points: Tensor) -> float:
    """Estimate trajectory distance from point cloud extent.

    When explicit poses are not available, we estimate the trajectory
    distance from the bounding box diagonal of the point cloud.

    Args:
        points: Point cloud of shape (N, 3)

    Returns:
        Estimated trajectory distance in meters
    """
    bounds_min = points.min(dim=0).values
    bounds_max = points.max(dim=0).values
    diagonal = torch.norm(bounds_max - bounds_min).item()
    return diagonal
