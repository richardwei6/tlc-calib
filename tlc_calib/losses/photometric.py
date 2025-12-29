"""Photometric loss functions for TLC-Calib.

This module implements the photometric loss from the paper:
    L_photo = (1 - λ) * L1 + λ * D-SSIM

where λ = 0.2 by default.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from pytorch_msssim import ssim, ms_ssim
    PYTORCH_MSSSIM_AVAILABLE = True
except ImportError:
    PYTORCH_MSSSIM_AVAILABLE = False


class PhotometricLoss(nn.Module):
    """Combined L1 + D-SSIM photometric loss.

    From the paper (Section 3.3):
        L_photo = (1 - λ) * L1 + λ * D-SSIM

    where D-SSIM = (1 - SSIM) / 2

    Args:
        lambda_dssim: Weight for D-SSIM term (default 0.2)
        use_ms_ssim: Use multi-scale SSIM instead of SSIM
    """

    def __init__(
        self,
        lambda_dssim: float = 0.2,
        use_ms_ssim: bool = False,
    ):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        self.use_ms_ssim = use_ms_ssim

        if not PYTORCH_MSSSIM_AVAILABLE:
            print("Warning: pytorch_msssim not available, using custom SSIM implementation")

    def forward(
        self,
        rendered: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> dict:
        """Compute photometric loss.

        Args:
            rendered: Rendered image of shape (H, W, 3) or (B, H, W, 3)
            target: Target image of shape (H, W, 3) or (B, H, W, 3)
            mask: Optional validity mask of shape (H, W) or (B, H, W)

        Returns:
            Dict containing:
                - loss: Total photometric loss
                - l1_loss: L1 component
                - dssim_loss: D-SSIM component
        """
        # Ensure batch dimension
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
            target = target.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        # Convert from (B, H, W, C) to (B, C, H, W) for SSIM
        rendered_bchw = rendered.permute(0, 3, 1, 2)
        target_bchw = target.permute(0, 3, 1, 2)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
            rendered_bchw = rendered_bchw * mask
            target_bchw = target_bchw * mask

        # Compute L1 loss
        l1_loss = self._compute_l1(rendered, target, mask)

        # Compute D-SSIM loss
        dssim_loss = self._compute_dssim(rendered_bchw, target_bchw)

        # Combined loss
        total_loss = (1 - self.lambda_dssim) * l1_loss + self.lambda_dssim * dssim_loss

        return {
            'loss': total_loss,
            'l1_loss': l1_loss,
            'dssim_loss': dssim_loss,
        }

    def _compute_l1(
        self,
        rendered: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute L1 loss."""
        diff = torch.abs(rendered - target)

        if mask is not None:
            if mask.dim() == 3:  # (B, H, W)
                mask = mask.unsqueeze(-1)  # (B, H, W, 1)
            diff = diff * mask
            return diff.sum() / (mask.sum() * 3 + 1e-8)
        else:
            return diff.mean()

    def _compute_dssim(
        self,
        rendered: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute D-SSIM loss: (1 - SSIM) / 2."""
        if PYTORCH_MSSSIM_AVAILABLE:
            if self.use_ms_ssim:
                ssim_val = ms_ssim(
                    rendered, target,
                    data_range=1.0,
                    size_average=True,
                )
            else:
                ssim_val = ssim(
                    rendered, target,
                    data_range=1.0,
                    size_average=True,
                )
        else:
            ssim_val = self._compute_ssim_custom(rendered, target)

        dssim = (1 - ssim_val) / 2
        return dssim

    def _compute_ssim_custom(
        self,
        rendered: Tensor,
        target: Tensor,
        window_size: int = 11,
        sigma: float = 1.5,
    ) -> Tensor:
        """Custom SSIM implementation when pytorch_msssim is unavailable."""
        device = rendered.device
        dtype = rendered.dtype
        C = rendered.shape[1]

        # Create Gaussian window
        coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        # 2D window
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(C, 1, window_size, window_size)

        # Compute means
        mu1 = F.conv2d(rendered, window, padding=window_size // 2, groups=C)
        mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Compute variances
        sigma1_sq = F.conv2d(rendered ** 2, window, padding=window_size // 2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=C) - mu2_sq
        sigma12 = F.conv2d(rendered * target, window, padding=window_size // 2, groups=C) - mu1_mu2

        # SSIM constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()


class L1Loss(nn.Module):
    """Simple L1 loss for ablation studies."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        rendered: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> dict:
        """Compute L1 loss."""
        diff = torch.abs(rendered - target)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            elif mask.dim() == 3 and rendered.dim() == 4:
                mask = mask.unsqueeze(-1)
            diff = diff * mask
            loss = diff.sum() / (mask.sum() * rendered.shape[-1] + 1e-8)
        else:
            loss = diff.mean()

        return {'loss': loss, 'l1_loss': loss, 'dssim_loss': torch.tensor(0.0)}


class PerceptualLoss(nn.Module):
    """Optional perceptual loss using VGG features.

    This is not used in the paper but can be useful for experiments.
    """

    def __init__(
        self,
        layers: tuple = ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'),
        weights: Optional[tuple] = None,
    ):
        super().__init__()
        self.layers = layers
        self.weights = weights or (1.0,) * len(layers)

        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except ImportError:
            print("Warning: torchvision not available for perceptual loss")
            self.vgg = None
            return

        # Layer indices for VGG16
        layer_indices = {
            'relu1_2': 4,
            'relu2_2': 9,
            'relu3_3': 16,
            'relu4_3': 23,
        }

        self.slices = nn.ModuleList()
        prev_idx = 0
        for layer in layers:
            idx = layer_indices[layer]
            self.slices.append(nn.Sequential(*list(vgg.children())[prev_idx:idx]))
            prev_idx = idx

        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self,
        rendered: Tensor,
        target: Tensor,
    ) -> dict:
        """Compute perceptual loss."""
        if not hasattr(self, 'slices') or self.slices is None:
            return {'loss': torch.tensor(0.0), 'perceptual_loss': torch.tensor(0.0)}

        # Ensure (B, C, H, W) format
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
            target = target.unsqueeze(0)

        if rendered.shape[-1] == 3:
            rendered = rendered.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)

        # Normalize
        rendered = (rendered - self.mean) / self.std
        target = (target - self.mean) / self.std

        # Compute features and loss
        total_loss = 0.0
        x_rendered = rendered
        x_target = target

        for i, slice_layer in enumerate(self.slices):
            x_rendered = slice_layer(x_rendered)
            x_target = slice_layer(x_target)
            total_loss += self.weights[i] * F.l1_loss(x_rendered, x_target)

        return {'loss': total_loss, 'perceptual_loss': total_loss}
