"""Differentiable Gaussian Splatting rasterizer for TLC-Calib.

This module wraps gsplat for CUDA-accelerated rasterization with
gradient flow to camera pose parameters.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    import gsplat
    from gsplat import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False

from .projection import CameraProjection, compute_pose_jacobian


class GaussianRasterizer(nn.Module):
    """Differentiable Gaussian rasterizer with camera pose gradients.

    This rasterizer enables gradient flow from the rendered image back to:
    - Gaussian parameters (positions, scales, rotations, colors, opacities)
    - Camera extrinsic parameters (via analytical Jacobians)

    From the paper:
        - Gradients propagate through 2D mean, 2D covariance, view-dependent color
        - Uses chain rule through projection for SE(3) pose updates
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        near_plane: float = 0.1,
        far_plane: float = 100.0,
        sh_degree: int = 0,
        use_gsplat: bool = True,
    ):
        """Initialize the rasterizer.

        Args:
            image_height: Output image height
            image_width: Output image width
            background_color: RGB background color
            near_plane: Near clipping plane distance
            far_plane: Far clipping plane distance
            sh_degree: Spherical harmonics degree for view-dependent color
            use_gsplat: Whether to use gsplat (if available) or fallback
        """
        super().__init__()

        self.image_height = image_height
        self.image_width = image_width
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.sh_degree = sh_degree
        self.use_gsplat = use_gsplat and GSPLAT_AVAILABLE

        self.register_buffer(
            'background',
            torch.tensor(background_color, dtype=torch.float32)
        )

        if not GSPLAT_AVAILABLE and use_gsplat:
            print("Warning: gsplat not available, using fallback software rasterizer")

    def forward(
        self,
        means_3d: Tensor,
        scales: Tensor,
        rotations: Tensor,
        colors: Tensor,
        opacities: Tensor,
        K: Tensor,
        T_lidar_to_cam: Tensor,
        active_sh_degree: int = 0,
    ) -> Dict[str, Tensor]:
        """Render Gaussians to an image.

        Args:
            means_3d: Gaussian centers in LiDAR frame of shape (N, 3)
            scales: Gaussian scales of shape (N, 3)
            rotations: Gaussian rotations as quaternions of shape (N, 4)
            colors: Gaussian colors/SH coefficients of shape (N, 3) or (N, K, 3)
            opacities: Gaussian opacities of shape (N, 1) or (N,)
            K: Camera intrinsic matrix of shape (3, 3)
            T_lidar_to_cam: Extrinsic transformation of shape (4, 4)
            active_sh_degree: Active SH degree for rendering

        Returns:
            Dict containing:
                - rendered_image: RGB image of shape (H, W, 3)
                - depth: Depth map of shape (H, W)
                - alpha: Alpha/opacity map of shape (H, W)
                - radii: Screen-space radii of shape (N,)
        """
        if self.use_gsplat:
            return self._render_gsplat(
                means_3d, scales, rotations, colors, opacities,
                K, T_lidar_to_cam, active_sh_degree
            )
        else:
            return self._render_fallback(
                means_3d, scales, rotations, colors, opacities,
                K, T_lidar_to_cam
            )

    def _render_gsplat(
        self,
        means_3d: Tensor,
        scales: Tensor,
        rotations: Tensor,
        colors: Tensor,
        opacities: Tensor,
        K: Tensor,
        T_lidar_to_cam: Tensor,
        active_sh_degree: int,
    ) -> Dict[str, Tensor]:
        """Render using gsplat CUDA implementation."""
        device = means_3d.device
        N = means_3d.shape[0]

        # Transform points to camera frame
        means_cam = CameraProjection.transform_points(means_3d, T_lidar_to_cam)

        # Build view matrix (world to camera)
        # gsplat expects viewmat as camera-to-world, so we need the inverse
        viewmat = torch.eye(4, device=device, dtype=means_3d.dtype)
        viewmat[:3, :3] = T_lidar_to_cam[:3, :3]
        viewmat[:3, 3] = T_lidar_to_cam[:3, 3]

        # Ensure proper shapes
        if opacities.dim() == 1:
            opacities = opacities.unsqueeze(-1)

        # Ensure colors have proper shape for gsplat
        if colors.dim() == 2:
            # Simple RGB colors (N, 3) -> (N, 1, 3) for SH degree 0
            sh_coeffs = colors.unsqueeze(1)
        else:
            sh_coeffs = colors

        # Build covariance matrices from scale and rotation
        cov_3d = CameraProjection.build_covariance_from_scale_rotation(scales, rotations)

        # Call gsplat rasterization
        renders, alphas, meta = rasterization(
            means=means_cam,  # Points in camera space
            quats=rotations,
            scales=scales,
            opacities=opacities.squeeze(-1),
            colors=sh_coeffs,
            viewmats=viewmat.unsqueeze(0),  # (1, 4, 4)
            Ks=K.unsqueeze(0),  # (1, 3, 3)
            width=self.image_width,
            height=self.image_height,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            sh_degree=active_sh_degree,
            backgrounds=self.background.unsqueeze(0),
            render_mode="RGB+D",
        )

        # Extract outputs
        rendered_image = renders[0, ..., :3]  # (H, W, 3)
        depth = renders[0, ..., 3]  # (H, W)
        alpha = alphas[0]  # (H, W)

        # Get radii from meta if available
        radii = meta.get("radii", torch.zeros(N, device=device))
        if isinstance(radii, Tensor) and radii.dim() == 2:
            radii = radii[0]  # Take first batch

        return {
            'rendered_image': rendered_image,
            'depth': depth,
            'alpha': alpha,
            'radii': radii,
        }

    def _render_fallback(
        self,
        means_3d: Tensor,
        scales: Tensor,
        rotations: Tensor,
        colors: Tensor,
        opacities: Tensor,
        K: Tensor,
        T_lidar_to_cam: Tensor,
    ) -> Dict[str, Tensor]:
        """Fallback software rasterizer (slower, for CPU or when gsplat unavailable)."""
        device = means_3d.device
        dtype = means_3d.dtype
        N = means_3d.shape[0]

        # Transform to camera frame
        means_cam = CameraProjection.transform_points(means_3d, T_lidar_to_cam)

        # Project to 2D
        pixels, depth = CameraProjection.project_to_image(means_cam, K)

        # Compute 2D covariances
        cov_3d = CameraProjection.build_covariance_from_scale_rotation(scales, rotations)
        cov_2d = CameraProjection.compute_2d_covariance(
            cov_3d, means_cam, K, T_lidar_to_cam
        )

        # Filter visible Gaussians
        visible_mask = CameraProjection.filter_visible_gaussians(
            means_cam, pixels, depth,
            self.image_height, self.image_width,
            self.near_plane, self.far_plane
        )

        # Initialize output buffers
        rendered_image = self.background.unsqueeze(0).unsqueeze(0).expand(
            self.image_height, self.image_width, 3
        ).clone()
        depth_map = torch.zeros(self.image_height, self.image_width, device=device, dtype=dtype)
        alpha_map = torch.zeros(self.image_height, self.image_width, device=device, dtype=dtype)

        if visible_mask.sum() == 0:
            return {
                'rendered_image': rendered_image,
                'depth': depth_map,
                'alpha': alpha_map,
                'radii': torch.zeros(N, device=device),
            }

        # Get visible Gaussians
        vis_pixels = pixels[visible_mask]
        vis_depth = depth[visible_mask]
        vis_cov_2d = cov_2d[visible_mask]
        vis_colors = colors[visible_mask] if colors.dim() == 2 else colors[visible_mask, 0]
        vis_opacities = opacities[visible_mask]
        if vis_opacities.dim() == 2:
            vis_opacities = vis_opacities.squeeze(-1)

        # Sort by depth (back to front for proper blending)
        sort_indices = torch.argsort(vis_depth, descending=True)
        vis_pixels = vis_pixels[sort_indices]
        vis_depth = vis_depth[sort_indices]
        vis_cov_2d = vis_cov_2d[sort_indices]
        vis_colors = vis_colors[sort_indices]
        vis_opacities = vis_opacities[sort_indices]

        # Create pixel grid
        y_coords = torch.arange(self.image_height, device=device, dtype=dtype)
        x_coords = torch.arange(self.image_width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        pixel_coords = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

        # Render each Gaussian (vectorized per-Gaussian for efficiency)
        # Note: This is still slower than CUDA but works on CPU
        T_accum = torch.ones(self.image_height, self.image_width, device=device, dtype=dtype)

        for i in range(len(vis_pixels)):
            mean = vis_pixels[i]
            cov = vis_cov_2d[i]
            color = vis_colors[i]
            opacity = vis_opacities[i]
            d = vis_depth[i]

            # Compute Gaussian extent (3 sigma)
            eigenvalues = torch.linalg.eigvalsh(cov)
            radius = 3.0 * torch.sqrt(eigenvalues.max())
            radius_int = int(radius.item()) + 1

            # Skip if radius too small
            if radius_int < 1:
                continue

            # Get bounding box
            cx, cy = int(mean[0].item()), int(mean[1].item())
            x_min = max(0, cx - radius_int)
            x_max = min(self.image_width, cx + radius_int + 1)
            y_min = max(0, cy - radius_int)
            y_max = min(self.image_height, cy + radius_int + 1)

            if x_min >= x_max or y_min >= y_max:
                continue

            # Get local pixel coordinates
            local_pixels = pixel_coords[y_min:y_max, x_min:x_max]  # (h, w, 2)

            # Compute Gaussian weights
            diff = local_pixels - mean  # (h, w, 2)
            cov_inv = torch.linalg.inv(cov + 1e-6 * torch.eye(2, device=device, dtype=dtype))
            mahal = torch.einsum('...i,ij,...j->...', diff, cov_inv, diff)
            gaussian_weight = torch.exp(-0.5 * mahal)  # (h, w)

            # Alpha compositing
            alpha = opacity * gaussian_weight
            alpha = torch.clamp(alpha, 0.0, 0.99)

            # Get accumulated transmittance for this region
            T_local = T_accum[y_min:y_max, x_min:x_max]

            # Update color
            weight = alpha * T_local
            rendered_image[y_min:y_max, x_min:x_max] += weight.unsqueeze(-1) * color

            # Update depth (first hit)
            depth_update = (depth_map[y_min:y_max, x_min:x_max] == 0) & (weight > 0.01)
            depth_map[y_min:y_max, x_min:x_max] = torch.where(
                depth_update, d, depth_map[y_min:y_max, x_min:x_max]
            )

            # Update alpha
            alpha_map[y_min:y_max, x_min:x_max] += weight

            # Update transmittance
            T_accum[y_min:y_max, x_min:x_max] = T_local * (1 - alpha)

        # Clamp output
        rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
        alpha_map = torch.clamp(alpha_map, 0.0, 1.0)

        # Compute approximate radii
        radii = torch.zeros(N, device=device)
        radii[visible_mask] = 3.0 * torch.sqrt(
            torch.linalg.eigvalsh(cov_2d[visible_mask]).max(dim=-1).values
        )

        return {
            'rendered_image': rendered_image,
            'depth': depth_map,
            'alpha': alpha_map,
            'radii': radii,
        }


class DifferentiableRenderer(nn.Module):
    """High-level renderer with camera pose gradient support.

    This class wraps the rasterizer and adds explicit gradient computation
    for camera extrinsic parameters using analytical Jacobians.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        **rasterizer_kwargs,
    ):
        """Initialize the differentiable renderer.

        Args:
            image_height: Output image height
            image_width: Output image width
            **rasterizer_kwargs: Additional arguments for GaussianRasterizer
        """
        super().__init__()

        self.rasterizer = GaussianRasterizer(
            image_height=image_height,
            image_width=image_width,
            **rasterizer_kwargs,
        )

    def render(
        self,
        gaussian_model,
        camera_id: str,
        camera_rig,
        K: Tensor,
        view_direction: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Render scene from a camera viewpoint.

        Args:
            gaussian_model: GaussianModel instance
            camera_id: Camera identifier
            camera_rig: CameraRig with learnable extrinsics
            K: Camera intrinsic matrix
            view_direction: Optional view direction for auxiliary computation

        Returns:
            Rendering results dict
        """
        # Get current extrinsic transformation
        T_lidar_to_cam = camera_rig.get_extrinsic(camera_id)

        # Get Gaussian parameters from model
        gaussians = gaussian_model.get_all_gaussians(view_direction)

        # Render
        result = self.rasterizer(
            means_3d=gaussians['means'],
            scales=gaussians['scales'],
            rotations=gaussians['rotations'],
            colors=gaussians['colors'],
            opacities=gaussians['opacities'],
            K=K,
            T_lidar_to_cam=T_lidar_to_cam,
        )

        return result

    def render_with_pose_gradients(
        self,
        means_3d: Tensor,
        scales: Tensor,
        rotations: Tensor,
        colors: Tensor,
        opacities: Tensor,
        K: Tensor,
        T_lidar_to_cam: Tensor,
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """Render with explicit pose Jacobians for gradient computation.

        This method computes analytical Jacobians for the camera pose,
        enabling gradient-based optimization of extrinsic parameters.

        Returns:
            Tuple of (render_result, pose_jacobian)
        """
        # Compute pose Jacobian
        pose_jacobian = compute_pose_jacobian(means_3d, T_lidar_to_cam)

        # Standard rendering
        result = self.rasterizer(
            means_3d=means_3d,
            scales=scales,
            rotations=rotations,
            colors=colors,
            opacities=opacities,
            K=K,
            T_lidar_to_cam=T_lidar_to_cam,
        )

        return result, pose_jacobian
