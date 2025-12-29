"""Camera Rig for TLC-Calib.

This module implements the multi-camera rig with learnable SE(3) extrinsic
parameters. The rig optimization strategy ensures consistent updates across
all images from the same camera.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.lie_groups import (
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_axis_angle,
)


class CameraRig(nn.Module):
    """Multi-camera rig with learnable SE(3) extrinsic parameters.

    From the paper:
        - Parameterize extrinsics as axis-angle rotation + translation
        - Separate learning rates: rotation=2e-3, translation=8e-3
        - Same update applied to all images from the same camera (rig optimization)

    The extrinsic transformation T^e transforms points from LiDAR to camera frame:
        P_cam = R @ P_lidar + t
    """

    def __init__(
        self,
        camera_ids: List[str],
        initial_extrinsics: Dict[str, Tensor],
        rotation_lr: float = 2e-3,
        translation_lr: float = 8e-3,
    ):
        """Initialize Camera Rig.

        Args:
            camera_ids: List of camera identifiers
            initial_extrinsics: Dict mapping camera_id -> 4x4 T^e matrix
            rotation_lr: Learning rate for rotation parameters
            translation_lr: Learning rate for translation parameters
        """
        super().__init__()

        self.camera_ids = camera_ids
        self.rotation_lr = rotation_lr
        self.translation_lr = translation_lr

        # Store parameters as ParameterDicts
        self.rotations = nn.ParameterDict()
        self.translations = nn.ParameterDict()

        for cam_id in camera_ids:
            if cam_id not in initial_extrinsics:
                raise ValueError(f"Missing initial extrinsics for camera: {cam_id}")

            T = initial_extrinsics[cam_id]
            R = T[:3, :3]
            t = T[:3, 3]

            # Convert rotation matrix to axis-angle for optimization
            axis_angle = rotation_matrix_to_axis_angle(R)

            # Store as learnable parameters
            # Note: Use underscores instead of dashes for valid Python identifiers
            safe_id = cam_id.replace('-', '_')
            self.rotations[safe_id] = nn.Parameter(axis_angle.clone())
            self.translations[safe_id] = nn.Parameter(t.clone())

        # Store mapping from camera_id to safe_id
        self._id_map = {cam_id: cam_id.replace('-', '_') for cam_id in camera_ids}

    def get_extrinsic(self, camera_id: str) -> Tensor:
        """Get 4x4 extrinsic matrix for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            4x4 transformation matrix T^e (LiDAR to Camera)
        """
        safe_id = self._id_map[camera_id]

        # Convert axis-angle back to rotation matrix
        R = axis_angle_to_rotation_matrix(self.rotations[safe_id])
        t = self.translations[safe_id]

        # Build 4x4 matrix
        T = torch.eye(4, device=R.device, dtype=R.dtype)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def get_all_extrinsics(self) -> Dict[str, Tensor]:
        """Get extrinsic matrices for all cameras.

        Returns:
            Dict mapping camera_id -> 4x4 transformation matrix
        """
        return {cam_id: self.get_extrinsic(cam_id) for cam_id in self.camera_ids}

    def get_rotation(self, camera_id: str) -> Tensor:
        """Get rotation matrix for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            3x3 rotation matrix
        """
        safe_id = self._id_map[camera_id]
        return axis_angle_to_rotation_matrix(self.rotations[safe_id])

    def get_translation(self, camera_id: str) -> Tensor:
        """Get translation vector for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            Translation vector of shape (3,)
        """
        safe_id = self._id_map[camera_id]
        return self.translations[safe_id]

    def get_camera_center(self, camera_id: str) -> Tensor:
        """Get camera center in LiDAR frame.

        The camera center is: o_c = -R^T @ t

        Args:
            camera_id: Camera identifier

        Returns:
            Camera center of shape (3,)
        """
        R = self.get_rotation(camera_id)
        t = self.get_translation(camera_id)
        return -R.T @ t

    def get_optimizer_param_groups(self) -> List[Dict]:
        """Get parameter groups with separate learning rates.

        Returns:
            List of param group dicts for optimizer
        """
        return [
            {
                'params': list(self.rotations.values()),
                'lr': self.rotation_lr,
                'name': 'rotations',
            },
            {
                'params': list(self.translations.values()),
                'lr': self.translation_lr,
                'name': 'translations',
            },
        ]

    def get_optimizer_param_groups_for_camera(self, camera_id: str) -> List[Dict]:
        """Get parameter groups for a single camera.

        Args:
            camera_id: Camera identifier

        Returns:
            List of param group dicts
        """
        safe_id = self._id_map[camera_id]
        return [
            {
                'params': [self.rotations[safe_id]],
                'lr': self.rotation_lr,
                'name': f'rotation_{camera_id}',
            },
            {
                'params': [self.translations[safe_id]],
                'lr': self.translation_lr,
                'name': f'translation_{camera_id}',
            },
        ]

    def compute_pose_delta(
        self,
        camera_id: str,
        initial_extrinsic: Tensor,
    ) -> Dict[str, float]:
        """Compute pose change from initial calibration.

        Args:
            camera_id: Camera identifier
            initial_extrinsic: Initial 4x4 extrinsic matrix

        Returns:
            Dict with rotation_error (degrees) and translation_error (meters)
        """
        current = self.get_extrinsic(camera_id)

        # Rotation error
        R_init = initial_extrinsic[:3, :3]
        R_curr = current[:3, :3]
        R_diff = R_init.T @ R_curr

        # Angle from trace: trace(R) = 1 + 2*cos(theta)
        trace = R_diff[0, 0] + R_diff[1, 1] + R_diff[2, 2]
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1, 1)
        angle_rad = torch.acos(cos_theta)
        angle_deg = torch.rad2deg(angle_rad).item()

        # Translation error
        t_init = initial_extrinsic[:3, 3]
        t_curr = current[:3, 3]
        t_error = torch.norm(t_curr - t_init).item()

        return {
            'rotation_error_deg': angle_deg,
            'translation_error_m': t_error,
        }

    def __len__(self) -> int:
        return len(self.camera_ids)

    def __repr__(self) -> str:
        return (
            f"CameraRig(\n"
            f"  cameras={self.camera_ids},\n"
            f"  rotation_lr={self.rotation_lr},\n"
            f"  translation_lr={self.translation_lr}\n"
            f")"
        )


class SingleCameraPose(nn.Module):
    """Single camera pose for simpler single-camera calibration."""

    def __init__(
        self,
        initial_extrinsic: Tensor,
        rotation_lr: float = 2e-3,
        translation_lr: float = 8e-3,
    ):
        """Initialize single camera pose.

        Args:
            initial_extrinsic: Initial 4x4 extrinsic matrix
            rotation_lr: Learning rate for rotation
            translation_lr: Learning rate for translation
        """
        super().__init__()

        self.rotation_lr = rotation_lr
        self.translation_lr = translation_lr

        R = initial_extrinsic[:3, :3]
        t = initial_extrinsic[:3, 3]

        axis_angle = rotation_matrix_to_axis_angle(R)

        self.rotation = nn.Parameter(axis_angle.clone())
        self.translation = nn.Parameter(t.clone())

        # Store initial for comparison
        self.register_buffer('initial_extrinsic', initial_extrinsic.clone())

    def get_extrinsic(self) -> Tensor:
        """Get current 4x4 extrinsic matrix."""
        R = axis_angle_to_rotation_matrix(self.rotation)
        T = torch.eye(4, device=R.device, dtype=R.dtype)
        T[:3, :3] = R
        T[:3, 3] = self.translation
        return T

    def get_optimizer_param_groups(self) -> List[Dict]:
        """Get parameter groups with separate learning rates."""
        return [
            {'params': [self.rotation], 'lr': self.rotation_lr, 'name': 'rotation'},
            {'params': [self.translation], 'lr': self.translation_lr, 'name': 'translation'},
        ]

    def compute_pose_delta(self) -> Dict[str, float]:
        """Compute pose change from initial calibration."""
        current = self.get_extrinsic()
        initial = self.initial_extrinsic

        # Rotation error
        R_diff = initial[:3, :3].T @ current[:3, :3]
        trace = R_diff[0, 0] + R_diff[1, 1] + R_diff[2, 2]
        cos_theta = torch.clamp((trace - 1) / 2, -1, 1)
        angle_deg = torch.rad2deg(torch.acos(cos_theta)).item()

        # Translation error
        t_error = torch.norm(current[:3, 3] - initial[:3, 3]).item()

        return {
            'rotation_error_deg': angle_deg,
            'translation_error_m': t_error,
        }
