"""Main training loop for TLC-Calib.

This module implements the training procedure from the paper:
    - 30K iterations with random image sampling
    - AdamW optimizer with weight decay scheduling
    - Combined photometric + regularization loss
    - Separate learning rates for rotation/translation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TLCCalibConfig, TrainingConfig
from ..data.dataset import TLCCalibDataset, SingleCameraDataset
from ..losses import PhotometricLoss, ScaleRegularization
from ..models.anchor_gaussians import AnchorGaussians
from ..models.camera_rig import CameraRig, SingleCameraPose
from ..models.gaussian_model import GaussianModel
from ..rendering.rasterizer import GaussianRasterizer
from ..utils.transforms import get_view_direction
from .adaptive_voxel import compute_adaptive_voxel_size

logger = logging.getLogger(__name__)


class TLCCalibTrainer:
    """Main trainer for TLC-Calib optimization.

    From the paper:
        - 30K iterations with random image sampling
        - AdamW: weight_decay=0.01 for first 15K, then 0
        - Rotation LR: 2e-3, Translation LR: 8e-3
        - Loss: L_photo + λ_scale * L_scale (λ_scale=1.0)
    """

    def __init__(
        self,
        config: TLCCalibConfig,
        device: str = "cuda",
    ):
        """Initialize trainer.

        Args:
            config: TLC-Calib configuration
            device: Device to train on
        """
        self.config = config
        self.device = device

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Initialize components (will be set up in setup())
        self.dataset: Optional[TLCCalibDataset] = None
        self.gaussian_model: Optional[GaussianModel] = None
        self.camera_rig: Optional[CameraRig] = None
        self.rasterizer: Optional[GaussianRasterizer] = None
        self.photo_loss: Optional[PhotometricLoss] = None
        self.scale_reg: Optional[ScaleRegularization] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Training state
        self.iteration = 0
        self.best_loss = float('inf')
        self.loss_history: List[Dict] = []

    def setup(self) -> None:
        """Set up all training components."""
        logger.info("Setting up TLC-Calib trainer...")

        # 1. Create dataset
        logger.info("Loading dataset...")
        self.dataset = TLCCalibDataset(
            config=self.config,
            split="train",
            train_ratio=0.9,  # Default train ratio
        )

        # Get aggregated point cloud
        points, intensities = self.dataset.get_aggregated_pointcloud()
        points = points.to(self.device)
        if intensities is not None:
            intensities = intensities.to(self.device)
        logger.info(f"Aggregated point cloud: {points.shape[0]} points")

        # 2. Compute adaptive voxel size
        logger.info("Computing adaptive voxel size...")
        voxel_size = compute_adaptive_voxel_size(
            points=points,
            beta=self.config.voxel.beta,
            eps_min=self.config.voxel.eps_min,
            eps_max=self.config.voxel.eps_max,
            max_anchors=self.config.voxel.max_anchors,
        )
        logger.info(f"Adaptive voxel size: {voxel_size:.4f}")

        # 3. Create anchor Gaussians
        logger.info("Creating anchor Gaussians...")
        anchor_positions = self._voxelize_points(points, voxel_size)
        logger.info(f"Number of anchors: {anchor_positions.shape[0]}")

        anchors = AnchorGaussians(
            positions=anchor_positions,
            feature_dim=self.config.model.feature_dim,
            initial_scale=voxel_size,
        )

        # 4. Create Gaussian model
        self.gaussian_model = GaussianModel(
            anchors=anchors,
            model_config=self.config.model,
        ).to(self.device)
        logger.info(f"Gaussian model created with {self.gaussian_model.get_gaussian_count()} total Gaussians")

        # 5. Create camera rig
        initial_extrinsics = self.dataset.get_all_extrinsics()
        initial_extrinsics = {k: v.to(self.device) for k, v in initial_extrinsics.items()}

        self.camera_rig = CameraRig(
            camera_ids=self.dataset.get_camera_ids(),
            initial_extrinsics=initial_extrinsics,
            rotation_lr=self.config.training.rotation_lr,
            translation_lr=self.config.training.translation_lr,
        ).to(self.device)
        logger.info(f"Camera rig created with {len(self.camera_rig)} cameras")

        # Store initial extrinsics for comparison
        self.initial_extrinsics = {k: v.clone() for k, v in initial_extrinsics.items()}

        # 6. Create rasterizer
        image_height, image_width = self.dataset.get_image_size()
        self.rasterizer = GaussianRasterizer(
            image_height=image_height,
            image_width=image_width,
            near_plane=self.config.render.near,
            far_plane=self.config.render.far,
            sh_degree=self.config.model.sh_degree,
        ).to(self.device)
        logger.info(f"Rasterizer created for {image_width}x{image_height} images")

        # 7. Create losses
        self.photo_loss = PhotometricLoss(
            lambda_dssim=self.config.training.lambda_ssim,
        )
        self.scale_reg = ScaleRegularization(
            sigma_threshold=self.config.training.sigma,
        )

        # 8. Create optimizer
        self._setup_optimizer()

        # 9. Get intrinsics
        self.intrinsics = self.dataset.get_all_intrinsics()
        self.intrinsics = {k: v.to(self.device) for k, v in self.intrinsics.items()}

        logger.info("Setup complete!")

    def _voxelize_points(self, points: Tensor, voxel_size: float) -> Tensor:
        """Voxelize points and return voxel centers."""
        # Quantize to voxel grid
        voxel_indices = (points / voxel_size).floor().long()

        # Get unique voxels
        unique_voxels = torch.unique(voxel_indices, dim=0)

        # Compute voxel centers
        voxel_centers = (unique_voxels.float() + 0.5) * voxel_size

        return voxel_centers

    def _setup_optimizer(self) -> None:
        """Set up optimizer with separate parameter groups."""
        # Gaussian model parameters
        gaussian_params = self.gaussian_model.get_optimizer_param_groups()

        # Camera rig parameters
        camera_params = self.camera_rig.get_optimizer_param_groups()

        # Combine all parameter groups
        all_params = gaussian_params + camera_params

        # Create optimizer with initial weight decay
        self.optimizer = torch.optim.AdamW(
            all_params,
            weight_decay=self.config.training.weight_decay,
        )

    def _update_weight_decay(self, iteration: int) -> None:
        """Update weight decay according to schedule.

        From paper: weight_decay=0.01 for first 15K iterations, then 0
        """
        if iteration == self.config.training.weight_decay_until:
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = 0.0
            logger.info(f"Iteration {iteration}: Disabled weight decay")

    def train_step(self, batch: Dict) -> Dict[str, Tensor]:
        """Perform a single training step.

        Args:
            batch: Dict containing image, camera_id, intrinsics, etc.

        Returns:
            Dict with loss values
        """
        self.optimizer.zero_grad()

        # Get data
        image = batch['image'].to(self.device)
        camera_id = batch['camera_id']
        if isinstance(camera_id, list):
            camera_id = camera_id[0]  # Batch size 1

        # Remove batch dimension if present
        if image.dim() == 4:
            image = image[0]

        # Convert image from (C, H, W) to (H, W, C)
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)

        # Get camera parameters
        K = self.intrinsics[camera_id]
        T_lidar_to_cam = self.camera_rig.get_extrinsic(camera_id)

        # Compute view direction from camera center
        view_dir = get_view_direction(T_lidar_to_cam)

        # Get all Gaussians
        gaussians = self.gaussian_model(view_dir)

        # Render
        render_result = self.rasterizer(
            means_3d=gaussians['means'],
            scales=gaussians['scales'],
            rotations=gaussians['rotations'],
            colors=gaussians['colors'],
            opacities=gaussians['opacities'],
            K=K,
            T_lidar_to_cam=T_lidar_to_cam,
        )

        rendered = render_result['rendered_image']

        # Compute photometric loss
        photo_result = self.photo_loss(rendered, image)

        # Compute scale regularization
        scale_result = self.scale_reg(gaussians['scales'])

        # Total loss
        total_loss = (
            photo_result['loss'] +
            self.config.training.lambda_scale * scale_result['loss']
        )

        # Backward pass
        total_loss.backward()

        # Optimizer step
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'photo_loss': photo_result['loss'].item(),
            'l1_loss': photo_result['l1_loss'].item(),
            'dssim_loss': photo_result['dssim_loss'].item(),
            'scale_loss': scale_result['loss'].item(),
            'scale_ratio': scale_result['mean_ratio'].item(),
        }

    def train(self, num_iterations: Optional[int] = None) -> Dict[str, List]:
        """Run full training loop.

        Args:
            num_iterations: Number of iterations (default from config)

        Returns:
            Dict with training history
        """
        if num_iterations is None:
            num_iterations = self.config.training.num_iterations

        logger.info(f"Starting training for {num_iterations} iterations...")

        # Log initial pose state BEFORE any training
        logger.info("=" * 60)
        logger.info("INITIAL CALIBRATION STATE (before training)")
        logger.info("=" * 60)
        for cam_id in self.camera_rig.camera_ids:
            delta = self.camera_rig.compute_pose_delta(
                cam_id, self.initial_extrinsics[cam_id]
            )
            logger.info(
                f"  {cam_id}: rot_err={delta['rotation_error_deg']:.6f}°, "
                f"trans_err={delta['translation_error_m']:.6f}m"
            )
        logger.info("=" * 60)

        # Create data loader (batch_size=1 as per paper)
        # Only use pin_memory with CUDA, reduce workers for MPS/CPU
        use_cuda = self.device == "cuda" or (isinstance(self.device, torch.device) and self.device.type == "cuda")
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=2 if use_cuda else 0,  # MPS/CPU works better with fewer workers
            pin_memory=use_cuda,
        )

        # Training loop
        pbar = tqdm(range(num_iterations), desc="Training")

        for self.iteration in pbar:
            # Get random sample
            batch = next(iter(dataloader))

            # Update weight decay
            self._update_weight_decay(self.iteration)

            # Training step
            losses = self.train_step(batch)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'photo': f"{losses['photo_loss']:.4f}",
                'scale': f"{losses['scale_loss']:.4f}",
            })

            # Log periodically (more frequent in early iterations to catch issues)
            log_now = (
                self.iteration % self.config.training.log_interval == 0 or
                self.iteration < 20 or  # Log every iteration for first 20
                (self.iteration < 100 and self.iteration % 10 == 0) or  # Every 10 for first 100
                (self.iteration < 1000 and self.iteration % 50 == 0)  # Every 50 for first 1000
            )
            if log_now:
                self._log_training_state(losses)

            # Save checkpoint periodically
            if self.iteration % self.config.training.save_interval == 0:
                self._save_checkpoint()

            # Store history
            self.loss_history.append(losses)

            # Track best loss
            if losses['total_loss'] < self.best_loss:
                self.best_loss = losses['total_loss']

        logger.info(f"Training complete. Best loss: {self.best_loss:.4f}")

        # Final checkpoint
        self._save_checkpoint(final=True)

        return {'loss_history': self.loss_history}

    def _log_training_state(self, losses: Dict) -> None:
        """Log current training state."""
        logger.info(
            f"Iter {self.iteration}: "
            f"total={losses['total_loss']:.4f}, "
            f"photo={losses['photo_loss']:.4f}, "
            f"l1={losses['l1_loss']:.4f}, "
            f"dssim={losses['dssim_loss']:.4f}, "
            f"scale={losses['scale_loss']:.4f}"
        )

        # Log pose deltas and gradient norms
        for cam_id in self.camera_rig.camera_ids:
            delta = self.camera_rig.compute_pose_delta(
                cam_id, self.initial_extrinsics[cam_id]
            )

            # Get gradient norms for pose parameters
            # CameraRig uses safe_id mapping (replaces '-' with '_')
            safe_id = self.camera_rig._id_map[cam_id]
            rot_param = self.camera_rig.rotations[safe_id]
            trans_param = self.camera_rig.translations[safe_id]

            rot_grad_norm = 0.0
            trans_grad_norm = 0.0
            if rot_param.grad is not None:
                rot_grad_norm = rot_param.grad.norm().item()
            if trans_param.grad is not None:
                trans_grad_norm = trans_param.grad.norm().item()

            # Get actual parameter values
            delta_rot = rot_param.detach()
            delta_trans = trans_param.detach()

            logger.info(
                f"  {cam_id}: rot_err={delta['rotation_error_deg']:.3f}°, "
                f"trans_err={delta['translation_error_m']:.4f}m, "
                f"rot_grad={rot_grad_norm:.6f}, trans_grad={trans_grad_norm:.6f}"
            )
            # Log pose parameters every 1000 iterations or early on
            if self.iteration < 100 or self.iteration % 1000 == 0:
                logger.info(
                    f"    delta_rot=[{delta_rot[0]:.6f}, {delta_rot[1]:.6f}, {delta_rot[2]:.6f}], "
                    f"delta_trans=[{delta_trans[0]:.6f}, {delta_trans[1]:.6f}, {delta_trans[2]:.6f}]"
                )

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        suffix = "final" if final else f"iter_{self.iteration}"
        checkpoint_path = checkpoint_dir / f"checkpoint_{suffix}.pt"

        checkpoint = {
            'iteration': self.iteration,
            'gaussian_model': self.gaussian_model.state_dict(),
            'camera_rig': self.camera_rig.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.iteration = checkpoint['iteration']
        self.gaussian_model.load_state_dict(checkpoint['gaussian_model'])
        self.camera_rig.load_state_dict(checkpoint['camera_rig'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_loss = checkpoint['best_loss']

        logger.info(f"Resumed from iteration {self.iteration}")

    def get_calibration_results(self) -> Dict[str, Dict]:
        """Get final calibration results.

        Returns:
            Dict mapping camera_id to calibration results
        """
        results = {}

        for cam_id in self.camera_rig.camera_ids:
            T = self.camera_rig.get_extrinsic(cam_id)
            T_init = self.initial_extrinsics[cam_id]
            delta = self.camera_rig.compute_pose_delta(cam_id, T_init)

            results[cam_id] = {
                'extrinsic': T.detach().cpu(),
                'initial_extrinsic': T_init.cpu(),
                'rotation_error_deg': delta['rotation_error_deg'],
                'translation_error_m': delta['translation_error_m'],
            }

        return results


class SingleCameraTrainer:
    """Trainer for single-camera calibration.

    Simplified version for calibrating one camera at a time.
    """

    def __init__(
        self,
        dataset: SingleCameraDataset,
        config: TrainingConfig,
        device: str = "cuda",
    ):
        """Initialize single-camera trainer.

        Args:
            dataset: SingleCameraDataset instance
            config: Training configuration
            device: Device to train on
        """
        self.dataset = dataset
        self.config = config
        self.device = device

        self.gaussian_model: Optional[GaussianModel] = None
        self.camera_pose: Optional[SingleCameraPose] = None
        self.rasterizer: Optional[GaussianRasterizer] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.iteration = 0
        self.best_loss = float('inf')

    def setup(
        self,
        anchor_feature_dim: int = 32,
        num_auxiliaries: int = 5,
        mlp_hidden_dim: int = 32,
        avc_beta: float = 5000.0,
    ) -> None:
        """Set up training components."""
        # Get point cloud
        points, _ = self.dataset.get_aggregated_pointcloud()
        points = points.to(self.device)

        # Compute voxel size
        voxel_size = compute_adaptive_voxel_size(points, beta=avc_beta)

        # Voxelize
        voxel_indices = (points / voxel_size).floor().long()
        unique_voxels = torch.unique(voxel_indices, dim=0)
        anchor_positions = (unique_voxels.float() + 0.5) * voxel_size

        # Create models
        anchors = AnchorGaussians(
            positions=anchor_positions,
            feature_dim=anchor_feature_dim,
            initial_scale=voxel_size,
        )

        # Create model config for GaussianModel
        from ..config import ModelConfig
        model_config = ModelConfig(
            feature_dim=anchor_feature_dim,
            num_auxiliary=num_auxiliaries,
            hidden_dim=mlp_hidden_dim,
        )

        self.gaussian_model = GaussianModel(
            anchors=anchors,
            model_config=model_config,
        ).to(self.device)

        # Camera pose
        initial_extrinsic = self.dataset.get_extrinsics().to(self.device)
        self.camera_pose = SingleCameraPose(
            initial_extrinsic=initial_extrinsic,
            rotation_lr=self.config.rotation_lr,
            translation_lr=self.config.translation_lr,
        ).to(self.device)

        self.initial_extrinsic = initial_extrinsic.clone()

        # Rasterizer
        image_height, image_width = self.dataset.get_image_size()
        self.rasterizer = GaussianRasterizer(
            image_height=image_height,
            image_width=image_width,
        ).to(self.device)

        # Losses
        self.photo_loss = PhotometricLoss(lambda_dssim=self.config.lambda_ssim)
        self.scale_reg = ScaleRegularization(sigma_threshold=self.config.sigma)

        # Optimizer
        all_params = (
            self.gaussian_model.get_optimizer_param_groups() +
            self.camera_pose.get_optimizer_param_groups()
        )
        self.optimizer = torch.optim.AdamW(
            all_params,
            weight_decay=self.config.weight_decay,
        )

        # Intrinsics
        self.K = self.dataset.get_intrinsics().to(self.device)

    def train_step(self, image: Tensor) -> Dict[str, float]:
        """Single training step."""
        self.optimizer.zero_grad()

        # Get camera transformation
        T = self.camera_pose.get_extrinsic()
        view_dir = get_view_direction(T)

        # Get Gaussians
        gaussians = self.gaussian_model(view_dir)

        # Render
        result = self.rasterizer(
            means_3d=gaussians['means'],
            scales=gaussians['scales'],
            rotations=gaussians['rotations'],
            colors=gaussians['colors'],
            opacities=gaussians['opacities'],
            K=self.K,
            T_lidar_to_cam=T,
        )

        # Losses
        photo_result = self.photo_loss(result['rendered_image'], image)
        scale_result = self.scale_reg(gaussians['scales'])

        total_loss = photo_result['loss'] + self.config.lambda_scale * scale_result['loss']
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'photo_loss': photo_result['loss'].item(),
            'scale_loss': scale_result['loss'].item(),
        }

    def train(self, num_iterations: int = 30000) -> Dict:
        """Run training loop."""
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        pbar = tqdm(range(num_iterations), desc="Training")

        for self.iteration in pbar:
            batch = next(iter(dataloader))
            image = batch['image'].to(self.device)

            if image.dim() == 4:
                image = image[0]
            if image.shape[0] == 3:
                image = image.permute(1, 2, 0)

            # Update weight decay
            if self.iteration == self.config.weight_decay_until:
                for pg in self.optimizer.param_groups:
                    pg['weight_decay'] = 0.0

            losses = self.train_step(image)
            pbar.set_postfix(loss=f"{losses['total_loss']:.4f}")

        # Get final results
        delta = self.camera_pose.compute_pose_delta()
        return {
            'extrinsic': self.camera_pose.get_extrinsic().detach().cpu(),
            'rotation_error_deg': delta['rotation_error_deg'],
            'translation_error_m': delta['translation_error_m'],
        }
