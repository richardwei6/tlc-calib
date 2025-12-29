"""Image loading and processing for TLC-Calib."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms as T

from ..config import CameraConfig
from ..utils.transforms import (
    compute_camera_intrinsics,
    compute_lidar_to_camera_transform,
)


class ImageLoader:
    """Load images and camera calibration data.

    This loader handles:
    - Loading images with optional downsampling
    - Parsing calibration JSON files
    - Computing intrinsic matrices with wide camera handling
    - Computing extrinsic transformations
    """

    def __init__(
        self,
        data_dir: Path | str,
        camera_config: CameraConfig,
        image_folder: str = "images",
        downsample: int = 1,
        image_extension: str = ".png",
    ):
        """Initialize image loader.

        Args:
            data_dir: Root data directory
            camera_config: Camera configuration
            image_folder: Name of folder containing images within camera folder
            downsample: Downsample factor (1 = full resolution)
            image_extension: Image file extension
        """
        self.data_dir = Path(data_dir)
        self.camera_config = camera_config
        self.downsample = downsample
        self.image_extension = image_extension

        # Build paths
        self.camera_dir = self.data_dir / camera_config.folder
        self.image_dir = self.camera_dir / image_folder
        self.calib_path = self.camera_dir / camera_config.calib_file

        # Validate paths
        if not self.camera_dir.exists():
            raise ValueError(f"Camera directory not found: {self.camera_dir}")
        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not self.calib_path.exists():
            raise ValueError(f"Calibration file not found: {self.calib_path}")

        # Discover images
        self.image_files = sorted(self.image_dir.glob(f"*{image_extension}"))
        if not self.image_files:
            raise ValueError(f"No images found in {self.image_dir}")

        self.num_frames = len(self.image_files)

        # Load calibration
        self._load_calibration()

        # Image transforms
        self._setup_transforms()

    def _load_calibration(self) -> None:
        """Load calibration from JSON file."""
        with open(self.calib_path, 'r') as f:
            calib_data = json.load(f)

        self.intrinsic_params = calib_data['intrinsic_params']
        self.extrinsic_params = calib_data['extrinsic_params']

        # Compute intrinsic matrix
        self.K = compute_camera_intrinsics(
            fx=self.intrinsic_params['fx'],
            fy=self.intrinsic_params['fy'],
            cx=self.intrinsic_params['cx'],
            cy=self.intrinsic_params['cy'],
            is_wide=self.camera_config.is_wide,
            wide_offset_y=self.camera_config.wide_offset_y,
            downsample=self.downsample,
        )

        # Compute extrinsic transformation (LiDAR to Camera)
        self.T_lidar_to_cam = compute_lidar_to_camera_transform(
            roll=self.extrinsic_params['roll'],
            pitch=self.extrinsic_params['pitch'],
            yaw=self.extrinsic_params['yaw'],
            px=self.extrinsic_params['px'],
            py=self.extrinsic_params['py'],
            pz=self.extrinsic_params['pz'],
        )

    def _setup_transforms(self) -> None:
        """Setup image transforms."""
        transform_list = [T.ToTensor()]  # [0, 255] -> [0, 1]

        if self.downsample > 1:
            # Note: Image size will be determined dynamically
            self._resize_transform = True
        else:
            self._resize_transform = False

        self.transform = T.Compose(transform_list)

    def load_image(self, frame_id: int) -> Tensor:
        """Load a single image.

        Args:
            frame_id: Frame index (0-based)

        Returns:
            Image tensor of shape (3, H, W) with values in [0, 1]
        """
        if frame_id < 0 or frame_id >= self.num_frames:
            raise IndexError(f"Frame {frame_id} out of range [0, {self.num_frames})")

        image_path = self.image_files[frame_id]
        image = Image.open(image_path).convert('RGB')

        # Apply downsampling if needed
        if self._resize_transform:
            new_size = (image.width // self.downsample, image.height // self.downsample)
            image = image.resize(new_size, Image.BILINEAR)

        # Convert to tensor
        image_tensor = self.transform(image)

        return image_tensor

    def get_intrinsics(self) -> Tensor:
        """Get camera intrinsic matrix.

        Returns:
            3x3 intrinsic matrix K (already scaled for downsample)
        """
        return self.K.clone()

    def get_extrinsics(self) -> Tensor:
        """Get camera extrinsic transformation (LiDAR to Camera).

        Returns:
            4x4 transformation matrix T_lidar_to_cam
        """
        return self.T_lidar_to_cam.clone()

    def get_image_size(self) -> Tuple[int, int]:
        """Get image dimensions after downsampling.

        Returns:
            Tuple of (height, width)
        """
        # Load first image to get dimensions
        image = Image.open(self.image_files[0])
        width = image.width // self.downsample
        height = image.height // self.downsample
        return height, width

    def get_calibration_data(self) -> Dict:
        """Get raw calibration data.

        Returns:
            Dict with 'intrinsic_params' and 'extrinsic_params'
        """
        return {
            'intrinsic_params': self.intrinsic_params.copy(),
            'extrinsic_params': self.extrinsic_params.copy(),
        }

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> Tensor:
        return self.load_image(idx)


class MultiCameraImageLoader:
    """Load images from multiple cameras.

    This provides a unified interface for multi-camera setups,
    managing multiple ImageLoader instances.
    """

    def __init__(
        self,
        data_dir: Path | str,
        camera_configs: Dict[str, CameraConfig],
        camera_ids: Optional[List[str]] = None,
        image_folder: str = "images",
        downsample: int = 1,
        image_extension: str = ".png",
    ):
        """Initialize multi-camera image loader.

        Args:
            data_dir: Root data directory
            camera_configs: Dict mapping camera_id to CameraConfig
            camera_ids: List of camera IDs to load (None = all cameras)
            image_folder: Name of folder containing images
            downsample: Downsample factor
            image_extension: Image file extension
        """
        self.data_dir = Path(data_dir)
        self.camera_configs = camera_configs
        self.downsample = downsample

        # Select cameras to load
        if camera_ids is None:
            camera_ids = list(camera_configs.keys())
        self.camera_ids = camera_ids

        # Create loaders for each camera
        self.loaders: Dict[str, ImageLoader] = {}
        for cam_id in camera_ids:
            if cam_id not in camera_configs:
                raise ValueError(f"Unknown camera ID: {cam_id}")

            self.loaders[cam_id] = ImageLoader(
                data_dir=data_dir,
                camera_config=camera_configs[cam_id],
                image_folder=image_folder,
                downsample=downsample,
                image_extension=image_extension,
            )

        # Verify all cameras have same number of frames
        num_frames = [len(loader) for loader in self.loaders.values()]
        if len(set(num_frames)) > 1:
            print(f"Warning: Cameras have different number of frames: {dict(zip(camera_ids, num_frames))}")

        self.num_frames = min(num_frames)

    def load_image(self, camera_id: str, frame_id: int) -> Tensor:
        """Load image from specific camera and frame.

        Args:
            camera_id: Camera identifier
            frame_id: Frame index

        Returns:
            Image tensor of shape (3, H, W)
        """
        return self.loaders[camera_id].load_image(frame_id)

    def load_all_cameras(self, frame_id: int) -> Dict[str, Tensor]:
        """Load images from all cameras for a single frame.

        Args:
            frame_id: Frame index

        Returns:
            Dict mapping camera_id to image tensor
        """
        return {cam_id: loader.load_image(frame_id) for cam_id, loader in self.loaders.items()}

    def get_intrinsics(self, camera_id: str) -> Tensor:
        """Get intrinsic matrix for a camera."""
        return self.loaders[camera_id].get_intrinsics()

    def get_extrinsics(self, camera_id: str) -> Tensor:
        """Get extrinsic transformation for a camera."""
        return self.loaders[camera_id].get_extrinsics()

    def get_all_intrinsics(self) -> Dict[str, Tensor]:
        """Get intrinsic matrices for all cameras."""
        return {cam_id: loader.get_intrinsics() for cam_id, loader in self.loaders.items()}

    def get_all_extrinsics(self) -> Dict[str, Tensor]:
        """Get extrinsic transformations for all cameras."""
        return {cam_id: loader.get_extrinsics() for cam_id, loader in self.loaders.items()}

    def get_image_size(self, camera_id: Optional[str] = None) -> Tuple[int, int]:
        """Get image dimensions.

        Args:
            camera_id: Camera to get size for (None = first camera)

        Returns:
            Tuple of (height, width)
        """
        if camera_id is None:
            camera_id = self.camera_ids[0]
        return self.loaders[camera_id].get_image_size()

    def __len__(self) -> int:
        return self.num_frames
