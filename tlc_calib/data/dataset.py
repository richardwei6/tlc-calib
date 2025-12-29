"""Combined dataset for TLC-Calib training."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..config import CameraConfig, DataConfig, TLCCalibConfig
from .image_loader import ImageLoader, MultiCameraImageLoader
from .lidar_loader import LiDARLoader


class TLCCalibDataset(Dataset):
    """Combined dataset for TLC-Calib training.

    This dataset provides:
    - Random sampling of (camera, frame) pairs for training
    - Access to aggregated point cloud
    - Camera intrinsics and extrinsics
    """

    def __init__(
        self,
        config: TLCCalibConfig,
        split: str = "train",
        train_ratio: float = 0.9,
    ):
        """Initialize dataset.

        Args:
            config: TLC-Calib configuration
            split: Data split ("train" or "val")
            train_ratio: Ratio of frames to use for training
        """
        self.config = config
        self.split = split
        self.train_ratio = train_ratio

        # Initialize LiDAR loader
        self.lidar_loader = LiDARLoader(
            pcd_dir=config.data.data_dir / config.data.pc_folder,
        )

        # Initialize image loaders
        self.image_loaders: Dict[str, ImageLoader] = {}
        for cam_id in config.data.cameras:
            if cam_id not in config.camera_configs:
                raise ValueError(f"Unknown camera ID: {cam_id}")

            self.image_loaders[cam_id] = ImageLoader(
                data_dir=config.data.data_dir,
                camera_config=config.camera_configs[cam_id],
                image_folder=config.data.image_folder,
                downsample=config.data.image_downsample,
            )

        # Get number of frames (use minimum across cameras)
        self.num_frames = min(len(loader) for loader in self.image_loaders.values())

        # Split frames
        split_idx = int(self.num_frames * train_ratio)
        if split == "train":
            self.frame_ids = list(range(split_idx))
        else:
            self.frame_ids = list(range(split_idx, self.num_frames))

        # Build index of (camera_id, frame_id) pairs
        self._build_index()

        # Cache aggregated point cloud
        self._points: Optional[Tensor] = None
        self._intensities: Optional[Tensor] = None

    def _build_index(self) -> None:
        """Build index of all (camera_id, frame_id) pairs."""
        self.samples: List[Tuple[str, int]] = []

        for frame_id in self.frame_ids:
            for cam_id in self.image_loaders.keys():
                self.samples.append((cam_id, frame_id))

    def get_aggregated_pointcloud(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Get aggregated point cloud from all LiDAR frames.

        Returns:
            Tuple of (points, intensities) where:
                - points: Tensor of shape (N, 3)
                - intensities: Tensor of shape (N,) or None
        """
        if self._points is None:
            self._points, self._intensities = self.lidar_loader.aggregate_pointcloud()
        return self._points, self._intensities

    def get_intrinsics(self, camera_id: str) -> Tensor:
        """Get camera intrinsic matrix.

        Args:
            camera_id: Camera identifier

        Returns:
            3x3 intrinsic matrix
        """
        return self.image_loaders[camera_id].get_intrinsics()

    def get_extrinsics(self, camera_id: str) -> Tensor:
        """Get initial camera extrinsic transformation.

        Args:
            camera_id: Camera identifier

        Returns:
            4x4 transformation matrix (LiDAR to Camera)
        """
        return self.image_loaders[camera_id].get_extrinsics()

    def get_all_intrinsics(self) -> Dict[str, Tensor]:
        """Get intrinsic matrices for all cameras."""
        return {cam_id: loader.get_intrinsics() for cam_id, loader in self.image_loaders.items()}

    def get_all_extrinsics(self) -> Dict[str, Tensor]:
        """Get initial extrinsic transformations for all cameras."""
        return {cam_id: loader.get_extrinsics() for cam_id, loader in self.image_loaders.items()}

    def get_image_size(self) -> Tuple[int, int]:
        """Get image dimensions (height, width)."""
        first_loader = next(iter(self.image_loaders.values()))
        return first_loader.get_image_size()

    def get_camera_ids(self) -> List[str]:
        """Get list of camera IDs."""
        return list(self.image_loaders.keys())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample.

        Args:
            idx: Sample index

        Returns:
            Dict containing:
                - image: Image tensor of shape (3, H, W)
                - camera_id: Camera identifier string
                - frame_id: Frame index
                - intrinsics: 3x3 intrinsic matrix
                - extrinsics: 4x4 extrinsic matrix (initial estimate)
        """
        camera_id, frame_id = self.samples[idx]
        loader = self.image_loaders[camera_id]

        return {
            'image': loader.load_image(frame_id),
            'camera_id': camera_id,
            'frame_id': frame_id,
            'intrinsics': loader.get_intrinsics(),
            'extrinsics': loader.get_extrinsics(),
        }


class SingleCameraDataset(Dataset):
    """Dataset for single-camera calibration.

    This is a simpler dataset that focuses on one camera at a time.
    """

    def __init__(
        self,
        data_dir: Path | str,
        camera_config: CameraConfig,
        pc_folder: str = "pc",
        image_folder: str = "images",
        downsample: int = 1,
        split: str = "train",
        train_ratio: float = 0.9,
    ):
        """Initialize single-camera dataset.

        Args:
            data_dir: Root data directory
            camera_config: Camera configuration
            pc_folder: Point cloud folder name
            image_folder: Image folder name
            downsample: Image downsample factor
            split: Data split
            train_ratio: Training data ratio
        """
        self.data_dir = Path(data_dir)
        self.camera_config = camera_config

        # Initialize loaders
        self.lidar_loader = LiDARLoader(pcd_dir=self.data_dir / pc_folder)
        self.image_loader = ImageLoader(
            data_dir=data_dir,
            camera_config=camera_config,
            image_folder=image_folder,
            downsample=downsample,
        )

        # Split frames
        num_frames = len(self.image_loader)
        split_idx = int(num_frames * train_ratio)

        if split == "train":
            self.frame_ids = list(range(split_idx))
        else:
            self.frame_ids = list(range(split_idx, num_frames))

        # Cache point cloud
        self._points: Optional[Tensor] = None
        self._intensities: Optional[Tensor] = None

    def get_aggregated_pointcloud(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Get aggregated point cloud."""
        if self._points is None:
            self._points, self._intensities = self.lidar_loader.aggregate_pointcloud()
        return self._points, self._intensities

    def get_intrinsics(self) -> Tensor:
        """Get camera intrinsics."""
        return self.image_loader.get_intrinsics()

    def get_extrinsics(self) -> Tensor:
        """Get initial camera extrinsics."""
        return self.image_loader.get_extrinsics()

    def get_image_size(self) -> Tuple[int, int]:
        """Get image dimensions."""
        return self.image_loader.get_image_size()

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample."""
        frame_id = self.frame_ids[idx]

        return {
            'image': self.image_loader.load_image(frame_id),
            'frame_id': frame_id,
            'intrinsics': self.image_loader.get_intrinsics(),
            'extrinsics': self.image_loader.get_extrinsics(),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for TLC-Calib dataset.

    Note: For TLC-Calib training, we typically use batch_size=1
    since each iteration processes one image at a time.

    Args:
        dataset: TLCCalibDataset or SingleCameraDataset
        batch_size: Batch size (default 1)
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for CUDA

    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
