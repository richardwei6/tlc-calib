"""LiDAR point cloud loading and processing for TLC-Calib."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from torch import Tensor


class LiDARLoader:
    """Load and aggregate LiDAR point clouds from PCD files.

    This loader handles:
    - Loading individual PCD frames
    - Aggregating multiple frames (already in same coordinate frame)
    - Filtering invalid (NaN) points
    - Converting to torch tensors
    """

    def __init__(
        self,
        pcd_dir: Path | str,
        file_pattern: str = "*.pcd",
    ):
        """Initialize LiDAR loader.

        Args:
            pcd_dir: Directory containing PCD files
            file_pattern: Glob pattern for PCD files
        """
        self.pcd_dir = Path(pcd_dir)
        self.file_pattern = file_pattern

        # Discover PCD files
        self.pcd_files = sorted(self.pcd_dir.glob(file_pattern))
        if not self.pcd_files:
            raise ValueError(f"No PCD files found in {pcd_dir} with pattern {file_pattern}")

        self.num_frames = len(self.pcd_files)

    def load_frame(self, frame_id: int) -> o3d.geometry.PointCloud:
        """Load a single PCD frame.

        Args:
            frame_id: Frame index (0-based)

        Returns:
            Open3D PointCloud object
        """
        if frame_id < 0 or frame_id >= self.num_frames:
            raise IndexError(f"Frame {frame_id} out of range [0, {self.num_frames})")

        pcd_path = self.pcd_files[frame_id]
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        return pcd

    def load_frame_as_tensor(self, frame_id: int) -> Tuple[Tensor, Optional[Tensor]]:
        """Load a single PCD frame as torch tensor.

        Args:
            frame_id: Frame index (0-based)

        Returns:
            Tuple of (points, intensities) where:
                - points: Tensor of shape (N, 3)
                - intensities: Tensor of shape (N,) or None if not available
        """
        pcd = self.load_frame(frame_id)
        points = np.asarray(pcd.points)

        # Filter NaN points
        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]

        # Try to load intensities (stored in colors for some PCD files)
        intensities = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)[valid_mask]
            # Assume intensity is stored in first color channel
            intensities = torch.from_numpy(colors[:, 0].astype(np.float32))

        return torch.from_numpy(points.astype(np.float32)), intensities

    def aggregate_pointcloud(
        self,
        frame_ids: Optional[List[int]] = None,
        max_points: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Aggregate multiple PCD frames into a single point cloud.

        Since the PCD files are already in the same coordinate frame,
        this simply concatenates them.

        Args:
            frame_ids: List of frame indices to aggregate. If None, uses all frames.
            max_points: Maximum number of points (random subsample if exceeded)

        Returns:
            Tuple of (points, intensities) where:
                - points: Tensor of shape (N, 3)
                - intensities: Tensor of shape (N,) or None
        """
        if frame_ids is None:
            frame_ids = list(range(self.num_frames))

        all_points = []
        all_intensities = []

        for frame_id in frame_ids:
            points, intensities = self.load_frame_as_tensor(frame_id)
            all_points.append(points)
            if intensities is not None:
                all_intensities.append(intensities)

        # Concatenate all frames
        points = torch.cat(all_points, dim=0)
        intensities = torch.cat(all_intensities, dim=0) if all_intensities else None

        # Random subsample if too many points
        if max_points is not None and points.shape[0] > max_points:
            indices = torch.randperm(points.shape[0])[:max_points]
            points = points[indices]
            if intensities is not None:
                intensities = intensities[indices]

        return points, intensities

    def get_point_cloud_stats(self) -> dict:
        """Get statistics about the aggregated point cloud.

        Returns:
            Dict with statistics including:
                - num_frames: Number of PCD files
                - total_points: Total number of points across all frames
                - bounds_min: Minimum coordinates (3,)
                - bounds_max: Maximum coordinates (3,)
                - centroid: Center of point cloud (3,)
        """
        all_points = []

        for frame_id in range(self.num_frames):
            points, _ = self.load_frame_as_tensor(frame_id)
            all_points.append(points)

        points = torch.cat(all_points, dim=0)

        return {
            "num_frames": self.num_frames,
            "total_points": points.shape[0],
            "bounds_min": points.min(dim=0).values.tolist(),
            "bounds_max": points.max(dim=0).values.tolist(),
            "centroid": points.mean(dim=0).tolist(),
        }

    def compute_trajectory_distance(self) -> float:
        """Compute total trajectory distance (for Adaptive Voxel Control).

        Since points are already in a common frame without per-frame poses,
        this estimates trajectory length from the point cloud extent.

        Returns:
            Estimated trajectory distance in meters
        """
        # Without explicit poses, estimate from point cloud extent
        all_points = []
        for frame_id in range(self.num_frames):
            points, _ = self.load_frame_as_tensor(frame_id)
            all_points.append(points)

        points = torch.cat(all_points, dim=0)

        # Use diagonal of bounding box as rough estimate
        bounds_min = points.min(dim=0).values
        bounds_max = points.max(dim=0).values
        diagonal = torch.norm(bounds_max - bounds_min).item()

        return diagonal

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> Tuple[Tensor, Optional[Tensor]]:
        return self.load_frame_as_tensor(idx)


def load_pcd_ascii(filepath: Path | str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load ASCII PCD file directly (fallback if Open3D fails).

    Args:
        filepath: Path to PCD file

    Returns:
        Tuple of (points, intensities) as numpy arrays
    """
    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse header
    header_end = 0
    num_points = 0
    fields = []

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('FIELDS'):
            fields = line.split()[1:]
        elif line.startswith('POINTS'):
            num_points = int(line.split()[1])
        elif line.startswith('DATA'):
            header_end = i + 1
            break

    # Parse data
    data_lines = lines[header_end:header_end + num_points]

    points = []
    intensities = []

    has_intensity = 'intensity' in fields
    intensity_idx = fields.index('intensity') if has_intensity else None

    for line in data_lines:
        values = line.strip().split()
        if len(values) >= 3:
            x, y, z = float(values[0]), float(values[1]), float(values[2])
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                points.append([x, y, z])
                if has_intensity and intensity_idx is not None:
                    intensities.append(float(values[intensity_idx]))

    points = np.array(points, dtype=np.float32)
    intensities = np.array(intensities, dtype=np.float32) if intensities else None

    return points, intensities
