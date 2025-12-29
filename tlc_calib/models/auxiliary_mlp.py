"""Auxiliary MLP for TLC-Calib.

The Auxiliary MLP predicts attributes for auxiliary Gaussians that surround
each anchor Gaussian. These auxiliary Gaussians help adapt local geometry
and improve pose convergence.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AuxiliaryMLP(nn.Module):
    """MLP that predicts auxiliary Gaussian attributes.

    From the paper:
        - 2-layer MLP with ReLU activation and 32 hidden units
        - Input: anchor_feature (32) + view_direction (3) + learned_scale (1) = 36
        - Output: K=5 auxiliary Gaussians per anchor with:
            - Positional offsets δ (3 * K)
            - Scales (3 * K)
            - Rotations as quaternions (4 * K)
            - Spherical harmonics coefficients for color
            - Opacities (K)
    """

    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dim: int = 32,
        num_auxiliary: int = 5,
        sh_degree: int = 3,
        num_layers: int = 2,
    ):
        """Initialize Auxiliary MLP.

        Args:
            feature_dim: Dimension of input anchor features
            hidden_dim: Hidden layer dimension
            num_auxiliary: Number of auxiliary Gaussians per anchor (K)
            sh_degree: Spherical harmonics degree for color
            num_layers: Number of MLP layers
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_auxiliary = num_auxiliary
        self.sh_degree = sh_degree

        # Input dimension: feature + view_dir + scale
        input_dim = feature_dim + 3 + 1

        # Shared backbone
        layers = []
        current_dim = input_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            current_dim = hidden_dim

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        backbone_out_dim = hidden_dim if layers else input_dim

        # Output heads
        # Positional offsets: K * 3
        self.offset_head = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_auxiliary * 3),
        )

        # Scale parameters (log-space): K * 3
        self.scale_head = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_auxiliary * 3),
        )

        # Rotation quaternions: K * 4
        self.rotation_head = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_auxiliary * 4),
        )

        # Spherical harmonics coefficients: K * (sh_degree + 1)^2 * 3
        sh_coeffs = (sh_degree + 1) ** 2 * 3
        self.color_head = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_auxiliary * sh_coeffs),
        )

        # Opacity (logit): K
        self.opacity_head = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_auxiliary),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize offset head to produce small offsets
        nn.init.normal_(self.offset_head[-1].weight, std=0.01)
        nn.init.zeros_(self.offset_head[-1].bias)

        # Initialize rotation head to produce identity quaternions
        # Identity quaternion: (0, 0, 0, 1)
        nn.init.zeros_(self.rotation_head[-1].weight)
        with torch.no_grad():
            bias = torch.zeros(self.num_auxiliary * 4)
            bias[3::4] = 1.0  # Set w component to 1
            self.rotation_head[-1].bias.copy_(bias)

        # Initialize opacity to moderate values
        nn.init.constant_(self.opacity_head[-1].bias, 0.0)  # sigmoid(0) = 0.5

    def forward(
        self,
        anchor_features: Tensor,
        view_directions: Tensor,
        learned_scales: Tensor,
    ) -> Dict[str, Tensor]:
        """Predict auxiliary Gaussian attributes.

        Args:
            anchor_features: Anchor features of shape (N, feature_dim)
            view_directions: Normalized view directions of shape (N, 3)
            learned_scales: Learned scales of shape (N, 1)

        Returns:
            Dict containing:
                - offsets: Position offsets of shape (N, K, 3)
                - scales: Log-scale parameters of shape (N, K, 3)
                - rotations: Quaternions of shape (N, K, 4)
                - colors: SH coefficients of shape (N, K, sh_coeffs)
                - opacities: Opacity values of shape (N, K)
        """
        N = anchor_features.shape[0]
        K = self.num_auxiliary

        # Concatenate inputs
        x = torch.cat([anchor_features, view_directions, learned_scales], dim=-1)

        # Shared backbone
        features = self.backbone(x)

        # Predict attributes
        offsets = self.offset_head(features).view(N, K, 3)
        scales = self.scale_head(features).view(N, K, 3)
        rotations = self.rotation_head(features).view(N, K, 4)
        colors = self.color_head(features).view(N, K, -1)
        opacities = self.opacity_head(features).view(N, K)

        # Normalize quaternions
        rotations = F.normalize(rotations, dim=-1)

        # Apply sigmoid to opacities
        opacities = torch.sigmoid(opacities)

        return {
            'offsets': offsets,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'opacities': opacities,
        }


class SimplifiedAuxiliaryMLP(nn.Module):
    """Simplified MLP that directly outputs RGB color instead of SH.

    This is useful for initial testing or when view-dependent effects
    are not critical.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dim: int = 32,
        num_auxiliary: int = 5,
        num_layers: int = 2,
    ):
        """Initialize Simplified Auxiliary MLP.

        Args:
            feature_dim: Dimension of input anchor features
            hidden_dim: Hidden layer dimension
            num_auxiliary: Number of auxiliary Gaussians per anchor
            num_layers: Number of MLP layers
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_auxiliary = num_auxiliary

        input_dim = feature_dim + 3 + 1

        # Single MLP for all outputs
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])

        self.mlp = nn.Sequential(*layers)

        # Output dimensions per auxiliary Gaussian:
        # offset (3) + scale (3) + rotation (4) + rgb (3) + opacity (1) = 14
        self.output_head = nn.Linear(hidden_dim, num_auxiliary * 14)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        anchor_features: Tensor,
        view_directions: Tensor,
        learned_scales: Tensor,
    ) -> Dict[str, Tensor]:
        """Predict auxiliary Gaussian attributes.

        Args:
            anchor_features: Shape (N, feature_dim)
            view_directions: Shape (N, 3)
            learned_scales: Shape (N, 1)

        Returns:
            Dict with offsets, scales, rotations, colors, opacities
        """
        N = anchor_features.shape[0]
        K = self.num_auxiliary

        x = torch.cat([anchor_features, view_directions, learned_scales], dim=-1)
        features = self.mlp(x)
        output = self.output_head(features).view(N, K, 14)

        # Split outputs
        offsets = output[..., :3]  # (N, K, 3)
        scales = output[..., 3:6]  # (N, K, 3)
        rotations = F.normalize(output[..., 6:10], dim=-1)  # (N, K, 4)
        colors = torch.sigmoid(output[..., 10:13])  # (N, K, 3) in [0, 1]
        opacities = torch.sigmoid(output[..., 13:14]).squeeze(-1)  # (N, K)

        return {
            'offsets': offsets,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'opacities': opacities,
        }
