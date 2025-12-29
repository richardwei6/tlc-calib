"""Anchor Gaussians for TLC-Calib.

Anchor Gaussians are created from voxelized LiDAR point clouds and serve as
fixed geometric references. Their positions are FROZEN during training to
preserve global scale and structure.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..optimization.adaptive_voxel import (
    AdaptiveVoxelControl,
    estimate_trajectory_distance,
    voxelize_with_features,
)
from ..config import ModelConfig, VoxelConfig


class AnchorGaussians(nn.Module):
    """Anchor Gaussians with frozen positions and learnable features.

    From the paper:
        - Anchor positions are voxel centers from LiDAR point cloud
        - Positions remain FIXED throughout training
        - Learnable: anchor features f_i and learned scales ℓ_i
    """

    def __init__(
        self,
        positions: Tensor,
        feature_dim: int = 32,
        initial_scale: float = 1.0,
    ):
        """Initialize Anchor Gaussians.

        Args:
            positions: Anchor positions of shape (N, 3) - will be frozen
            feature_dim: Dimension of learnable anchor features
            initial_scale: Initial value for learned scales
        """
        super().__init__()

        num_anchors = positions.shape[0]
        self.num_anchors = num_anchors
        self.feature_dim = feature_dim

        # FROZEN positions - use register_buffer, not nn.Parameter
        self.register_buffer('positions', positions)

        # Learnable anchor features
        self.anchor_features = nn.Parameter(torch.randn(num_anchors, feature_dim) * 0.1)

        # Learnable scales (one per anchor, used for auxiliary Gaussian generation)
        self.learned_scales = nn.Parameter(torch.ones(num_anchors, 1) * initial_scale)

    @classmethod
    def from_pointcloud(
        cls,
        points: Tensor,
        voxel_config: VoxelConfig,
        model_config: ModelConfig,
        trajectory_distance: Optional[float] = None,
    ) -> "AnchorGaussians":
        """Create Anchor Gaussians from a point cloud using adaptive voxelization.

        Args:
            points: Point cloud of shape (N, 3)
            voxel_config: Voxel configuration
            model_config: Model configuration
            trajectory_distance: Optional trajectory distance (estimated if not provided)

        Returns:
            AnchorGaussians instance
        """
        # Estimate trajectory distance if not provided
        if trajectory_distance is None:
            trajectory_distance = estimate_trajectory_distance(points)

        # Compute optimal voxel size
        avc = AdaptiveVoxelControl(voxel_config)
        voxel_size = avc.compute_voxel_size(points, trajectory_distance)

        print(f"Adaptive Voxel Control:")
        print(f"  Trajectory distance: {trajectory_distance:.2f}m")
        print(f"  Target voxels: {avc.compute_target_voxels(trajectory_distance)}")
        print(f"  Optimal voxel size: {voxel_size:.4f}m")

        # Voxelize point cloud
        voxel_centers = avc.voxelize_pointcloud(points, voxel_size)

        print(f"  Actual voxels (anchors): {len(voxel_centers)}")

        return cls(
            positions=voxel_centers,
            feature_dim=model_config.feature_dim,
        )

    @classmethod
    def from_pointcloud_simple(
        cls,
        points: Tensor,
        voxel_size: float,
        feature_dim: int = 32,
    ) -> "AnchorGaussians":
        """Create Anchor Gaussians with fixed voxel size.

        Args:
            points: Point cloud of shape (N, 3)
            voxel_size: Fixed voxel size
            feature_dim: Feature dimension

        Returns:
            AnchorGaussians instance
        """
        # Simple voxelization
        voxel_centers, _ = voxelize_with_features(points, None, voxel_size)

        print(f"Created {len(voxel_centers)} anchors with voxel size {voxel_size:.4f}m")

        return cls(
            positions=voxel_centers,
            feature_dim=feature_dim,
        )

    def get_positions(self) -> Tensor:
        """Get anchor positions (frozen).

        Returns:
            Positions of shape (N, 3)
        """
        return self.positions

    def get_features(self) -> Tensor:
        """Get learnable anchor features.

        Returns:
            Features of shape (N, feature_dim)
        """
        return self.anchor_features

    def get_scales(self) -> Tensor:
        """Get learnable scales.

        Returns:
            Scales of shape (N, 1)
        """
        return self.learned_scales

    def forward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass returning all anchor properties.

        Returns:
            Tuple of (positions, features, scales)
        """
        return self.positions, self.anchor_features, self.learned_scales

    def __len__(self) -> int:
        return self.num_anchors

    def __repr__(self) -> str:
        return (
            f"AnchorGaussians(num_anchors={self.num_anchors}, "
            f"feature_dim={self.feature_dim})"
        )


def filter_anchors_by_visibility(
    anchors: AnchorGaussians,
    camera_positions: Tensor,
    max_distance: float = 100.0,
    min_distance: float = 0.1,
) -> Tensor:
    """Get mask of anchors visible from camera positions.

    Args:
        anchors: AnchorGaussians instance
        camera_positions: Camera positions of shape (C, 3) or (3,)
        max_distance: Maximum distance for visibility
        min_distance: Minimum distance for visibility

    Returns:
        Boolean mask of shape (N,) indicating visible anchors
    """
    positions = anchors.get_positions()  # (N, 3)

    if camera_positions.dim() == 1:
        camera_positions = camera_positions.unsqueeze(0)  # (1, 3)

    # Compute distances to all cameras
    # (N, 3) - (C, 1, 3) -> (C, N, 3) -> (C, N)
    distances = torch.norm(
        positions.unsqueeze(0) - camera_positions.unsqueeze(1),
        dim=-1
    )  # (C, N)

    # Anchor is visible if in range for at least one camera
    in_range = (distances >= min_distance) & (distances <= max_distance)
    visible = in_range.any(dim=0)  # (N,)

    return visible
