"""Combined Gaussian Scene Model for TLC-Calib.

This module combines Anchor Gaussians and Auxiliary Gaussians into a complete
neural scene representation that can be rendered via differentiable splatting.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .anchor_gaussians import AnchorGaussians
from .auxiliary_mlp import AuxiliaryMLP, SimplifiedAuxiliaryMLP
from ..config import ModelConfig, VoxelConfig


class GaussianSceneModel(nn.Module):
    """Combined scene representation using anchor and auxiliary Gaussians.

    The scene is represented as:
        - Anchor Gaussians: Fixed positions from voxelized LiDAR, learnable features
        - Auxiliary Gaussians: MLP-predicted offsets around each anchor

    From the paper:
        - Anchor positions v_i are frozen
        - Auxiliary positions: m_{i,k} = v_i + δ_{i,k}
        - K=5 auxiliaries per anchor
    """

    def __init__(
        self,
        anchors: AnchorGaussians,
        model_config: ModelConfig,
        use_simplified_mlp: bool = False,
    ):
        """Initialize Gaussian Scene Model.

        Args:
            anchors: Pre-initialized AnchorGaussians
            model_config: Model configuration
            use_simplified_mlp: If True, use RGB output instead of SH
        """
        super().__init__()

        self.anchors = anchors
        self.config = model_config
        self.use_simplified_mlp = use_simplified_mlp

        # Initialize auxiliary MLP
        if use_simplified_mlp:
            self.auxiliary_mlp = SimplifiedAuxiliaryMLP(
                feature_dim=model_config.feature_dim,
                hidden_dim=model_config.hidden_dim,
                num_auxiliary=model_config.num_auxiliary,
                num_layers=model_config.num_layers,
            )
        else:
            self.auxiliary_mlp = AuxiliaryMLP(
                feature_dim=model_config.feature_dim,
                hidden_dim=model_config.hidden_dim,
                num_auxiliary=model_config.num_auxiliary,
                sh_degree=model_config.sh_degree,
                num_layers=model_config.num_layers,
            )

        self.num_auxiliary = model_config.num_auxiliary

    @classmethod
    def from_pointcloud(
        cls,
        points: Tensor,
        model_config: ModelConfig,
        voxel_config: VoxelConfig,
        trajectory_distance: Optional[float] = None,
        use_simplified_mlp: bool = False,
    ) -> "GaussianSceneModel":
        """Create Gaussian Scene Model from point cloud.

        Args:
            points: Point cloud of shape (N, 3)
            model_config: Model configuration
            voxel_config: Voxel configuration
            trajectory_distance: Optional trajectory distance
            use_simplified_mlp: If True, use simplified MLP

        Returns:
            GaussianSceneModel instance
        """
        # Create anchor Gaussians
        anchors = AnchorGaussians.from_pointcloud(
            points=points,
            voxel_config=voxel_config,
            model_config=model_config,
            trajectory_distance=trajectory_distance,
        )

        return cls(
            anchors=anchors,
            model_config=model_config,
            use_simplified_mlp=use_simplified_mlp,
        )

    def forward(
        self,
        view_direction: Tensor,
        anchor_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Generate all Gaussian parameters for rendering.

        Args:
            view_direction: Viewing direction of shape (3,) or (N, 3)
                If (3,), broadcasts to all anchors
            anchor_mask: Optional boolean mask of shape (N,) to select anchors

        Returns:
            Dict containing:
                - means: 3D positions of shape (M, 3) where M = N_anchors * K
                - scales: Scale parameters of shape (M, 3)
                - rotations: Quaternions of shape (M, 4)
                - colors: Color parameters of shape (M, C)
                - opacities: Opacity values of shape (M,)
        """
        # Get anchor properties
        anchor_pos, anchor_features, anchor_scales = self.anchors()

        # Apply mask if provided
        if anchor_mask is not None:
            anchor_pos = anchor_pos[anchor_mask]
            anchor_features = anchor_features[anchor_mask]
            anchor_scales = anchor_scales[anchor_mask]

        N = anchor_pos.shape[0]
        K = self.num_auxiliary

        # Broadcast view direction to all anchors if needed
        if view_direction.dim() == 1:
            view_dir = view_direction.unsqueeze(0).expand(N, 3)
        else:
            view_dir = view_direction
            if anchor_mask is not None:
                view_dir = view_dir[anchor_mask]

        # Get auxiliary Gaussian attributes from MLP
        aux_attrs = self.auxiliary_mlp(anchor_features, view_dir, anchor_scales)

        # Compute final positions: anchor + offset
        # anchor_pos: (N, 3) -> (N, 1, 3)
        # offsets: (N, K, 3)
        # means: (N, K, 3) -> (N*K, 3)
        means = (anchor_pos.unsqueeze(1) + aux_attrs['offsets']).flatten(0, 1)

        # Flatten other attributes
        scales = aux_attrs['scales'].flatten(0, 1)  # (N*K, 3)
        rotations = aux_attrs['rotations'].flatten(0, 1)  # (N*K, 4)
        colors = aux_attrs['colors'].flatten(0, 1)  # (N*K, C)
        opacities = aux_attrs['opacities'].flatten()  # (N*K,)

        return {
            'means': means,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'opacities': opacities,
        }

    def get_gaussian_count(self) -> int:
        """Get total number of Gaussians (anchors * auxiliaries).

        Returns:
            Total Gaussian count
        """
        return len(self.anchors) * self.num_auxiliary

    def get_anchor_count(self) -> int:
        """Get number of anchor Gaussians.

        Returns:
            Anchor count
        """
        return len(self.anchors)

    def get_scene_bounds(self) -> Tuple[Tensor, Tensor]:
        """Get bounding box of anchor positions.

        Returns:
            Tuple of (min_bounds, max_bounds), each of shape (3,)
        """
        positions = self.anchors.get_positions()
        return positions.min(dim=0).values, positions.max(dim=0).values

    def prune_low_opacity_anchors(self, threshold: float = 0.01) -> None:
        """Prune anchors with consistently low opacity (floaters).

        This removes anchors that contribute little to rendering,
        typically caused by noise in the LiDAR data.

        Note: This modifies the model in-place.

        Args:
            threshold: Opacity threshold for pruning
        """
        # This would require re-creating the anchor module with fewer anchors
        # For now, we'll implement opacity-based filtering during rendering
        pass

    def get_optimizer_param_groups(self, lr: float = 0.001) -> list:
        """Get parameter groups for optimizer.

        Args:
            lr: Base learning rate

        Returns:
            List of parameter group dicts
        """
        return [
            {
                'params': self.anchors.parameters(),
                'lr': lr,
                'name': 'anchors',
            },
            {
                'params': self.auxiliary_mlp.parameters(),
                'lr': lr,
                'name': 'auxiliary_mlp',
            },
        ]

    def __repr__(self) -> str:
        return (
            f"GaussianSceneModel(\n"
            f"  anchors={self.anchors},\n"
            f"  num_auxiliary={self.num_auxiliary},\n"
            f"  total_gaussians={self.get_gaussian_count()}\n"
            f")"
        )


def compute_view_direction_for_camera(
    T_lidar_to_cam: Tensor,
    anchor_positions: Optional[Tensor] = None,
) -> Tensor:
    """Compute view direction from camera for anchor positions.

    Args:
        T_lidar_to_cam: 4x4 transformation matrix
        anchor_positions: Optional anchor positions of shape (N, 3)
            If None, returns camera forward direction

    Returns:
        View direction(s) of shape (3,) or (N, 3)
    """
    # Camera center in LiDAR frame: o_c = -R^T @ t
    R = T_lidar_to_cam[:3, :3]
    t = T_lidar_to_cam[:3, 3]
    camera_center = -R.T @ t  # (3,)

    if anchor_positions is None:
        # Return camera forward direction (Z-axis in camera frame transformed to LiDAR)
        forward = R.T @ torch.tensor([0., 0., 1.], device=R.device, dtype=R.dtype)
        return forward / (torch.norm(forward) + 1e-8)
    else:
        # Direction from camera to each anchor
        view_dirs = anchor_positions - camera_center.unsqueeze(0)  # (N, 3)
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-8)
        return view_dirs


# Alias for backwards compatibility
GaussianModel = GaussianSceneModel
