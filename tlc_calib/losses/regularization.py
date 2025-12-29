"""Regularization losses for TLC-Calib.

This module implements regularization terms from the paper:
    - Scale regularization to prevent scale collapse
"""

import torch
import torch.nn as nn
from torch import Tensor


class ScaleRegularization(nn.Module):
    """Scale regularization loss to prevent scale collapse.

    From the paper (Section 3.2):
        L_scale = mean(max(max(s)/min(s) - σ, 0))

    This penalizes Gaussians with high scale anisotropy (ratio > σ),
    preventing degenerate needle-like or pancake Gaussians.

    Args:
        sigma_threshold: Maximum allowed scale ratio (default 10.0)
    """

    def __init__(self, sigma_threshold: float = 10.0):
        super().__init__()
        self.sigma_threshold = sigma_threshold

    def forward(self, scales: Tensor) -> dict:
        """Compute scale regularization loss.

        Args:
            scales: Gaussian scales of shape (N, 3) or (N, K, 3)

        Returns:
            Dict containing:
                - loss: Scale regularization loss
                - mean_ratio: Mean scale ratio for monitoring
                - max_ratio: Maximum scale ratio
        """
        # Handle different input shapes
        if scales.dim() == 3:
            # (N, K, 3) -> (N*K, 3)
            scales = scales.reshape(-1, 3)

        # Ensure positive scales (they may be in log space)
        if scales.min() < 0:
            scales = scales.exp()

        # Compute scale ratios
        scale_max = scales.max(dim=-1).values  # (N,)
        scale_min = scales.min(dim=-1).values  # (N,)

        # Avoid division by zero
        scale_min = torch.clamp(scale_min, min=1e-8)

        ratio = scale_max / scale_min  # (N,)

        # Penalize ratios exceeding threshold
        excess = torch.clamp(ratio - self.sigma_threshold, min=0.0)
        loss = excess.mean()

        return {
            'loss': loss,
            'mean_ratio': ratio.mean(),
            'max_ratio': ratio.max(),
        }


class OpacityRegularization(nn.Module):
    """Opacity regularization to encourage sparse representations.

    Penalizes very low opacities to prevent "ghost" Gaussians.
    Not used in the original paper but can be useful for optimization.
    """

    def __init__(self, min_opacity: float = 0.01):
        super().__init__()
        self.min_opacity = min_opacity

    def forward(self, opacities: Tensor) -> dict:
        """Compute opacity regularization loss.

        Args:
            opacities: Gaussian opacities of shape (N,) or (N, 1)

        Returns:
            Dict with loss value
        """
        if opacities.dim() == 2:
            opacities = opacities.squeeze(-1)

        # Penalize opacities below minimum
        low_opacity_penalty = torch.clamp(self.min_opacity - opacities, min=0.0)
        loss = low_opacity_penalty.mean()

        return {
            'loss': loss,
            'mean_opacity': opacities.mean(),
            'num_low_opacity': (opacities < self.min_opacity).sum(),
        }


class OffsetRegularization(nn.Module):
    """Offset regularization to keep auxiliary Gaussians near anchors.

    Penalizes large offsets that move auxiliary Gaussians far from
    their anchor positions.
    """

    def __init__(self, max_offset: float = 1.0):
        super().__init__()
        self.max_offset = max_offset

    def forward(self, offsets: Tensor) -> dict:
        """Compute offset regularization loss.

        Args:
            offsets: Gaussian offsets of shape (N, K, 3)

        Returns:
            Dict with loss value
        """
        # Compute offset magnitudes
        offset_norms = torch.norm(offsets, dim=-1)  # (N, K)

        # Penalize offsets exceeding maximum
        excess = torch.clamp(offset_norms - self.max_offset, min=0.0)
        loss = excess.mean()

        return {
            'loss': loss,
            'mean_offset': offset_norms.mean(),
            'max_offset': offset_norms.max(),
        }


class TotalVariationLoss(nn.Module):
    """Total variation loss for smooth rendered images.

    Not used in the original paper but can help with noisy renders.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, image: Tensor) -> dict:
        """Compute total variation loss.

        Args:
            image: Image tensor of shape (H, W, 3) or (B, H, W, 3)

        Returns:
            Dict with loss value
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Convert to (B, C, H, W) if needed
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)

        # Compute differences
        diff_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
        diff_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])

        loss = self.weight * (diff_h.mean() + diff_w.mean())

        return {'loss': loss, 'tv_loss': loss}


class CombinedRegularization(nn.Module):
    """Combined regularization module for convenience.

    Combines multiple regularization terms with configurable weights.
    """

    def __init__(
        self,
        scale_weight: float = 1.0,
        scale_threshold: float = 10.0,
        opacity_weight: float = 0.0,
        opacity_min: float = 0.01,
        offset_weight: float = 0.0,
        offset_max: float = 1.0,
    ):
        """Initialize combined regularization.

        Args:
            scale_weight: Weight for scale regularization
            scale_threshold: Scale ratio threshold (σ)
            opacity_weight: Weight for opacity regularization
            opacity_min: Minimum opacity threshold
            offset_weight: Weight for offset regularization
            offset_max: Maximum offset magnitude
        """
        super().__init__()

        self.scale_weight = scale_weight
        self.opacity_weight = opacity_weight
        self.offset_weight = offset_weight

        self.scale_reg = ScaleRegularization(scale_threshold)
        self.opacity_reg = OpacityRegularization(opacity_min)
        self.offset_reg = OffsetRegularization(offset_max)

    def forward(
        self,
        scales: Tensor,
        opacities: Tensor = None,
        offsets: Tensor = None,
    ) -> dict:
        """Compute combined regularization loss.

        Args:
            scales: Gaussian scales of shape (N, 3) or (N, K, 3)
            opacities: Optional opacities of shape (N,) or (N, K)
            offsets: Optional offsets of shape (N, K, 3)

        Returns:
            Dict containing total loss and individual components
        """
        total_loss = torch.tensor(0.0, device=scales.device)
        result = {}

        # Scale regularization
        if self.scale_weight > 0:
            scale_result = self.scale_reg(scales)
            total_loss = total_loss + self.scale_weight * scale_result['loss']
            result['scale_loss'] = scale_result['loss']
            result['scale_ratio'] = scale_result['mean_ratio']

        # Opacity regularization
        if self.opacity_weight > 0 and opacities is not None:
            opacity_result = self.opacity_reg(opacities)
            total_loss = total_loss + self.opacity_weight * opacity_result['loss']
            result['opacity_loss'] = opacity_result['loss']
            result['mean_opacity'] = opacity_result['mean_opacity']

        # Offset regularization
        if self.offset_weight > 0 and offsets is not None:
            offset_result = self.offset_reg(offsets)
            total_loss = total_loss + self.offset_weight * offset_result['loss']
            result['offset_loss'] = offset_result['loss']
            result['mean_offset'] = offset_result['mean_offset']

        result['loss'] = total_loss
        return result
