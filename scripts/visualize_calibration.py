#!/usr/bin/env python3
"""Visualize calibration results by overlaying point cloud onto images.

Usage:
    python scripts/visualize_calibration.py \
        --results outputs/calibration_results.txt \
        --data-dir data \
        --camera cam02 \
        --frame 0 \
        --output overlay.png
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw


def _parse_extrinsic_section(section: str) -> Dict:
    """Parse a calibration section to extract extrinsic matrix."""
    # Extract full extrinsic matrix
    extrinsic_match = re.search(
        r'Full Extrinsic Matrix \(4x4\):\s*'
        r'\[\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]\s*'
        r'\[\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]\s*'
        r'\[\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]\s*'
        r'\[\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]',
        section
    )

    if extrinsic_match:
        extrinsic = np.array([
            [float(extrinsic_match.group(i)) for i in range(1, 5)],
            [float(extrinsic_match.group(i)) for i in range(5, 9)],
            [float(extrinsic_match.group(i)) for i in range(9, 13)],
            [float(extrinsic_match.group(i)) for i in range(13, 17)],
        ])
        return {
            'rotation': extrinsic[:3, :3],
            'translation': extrinsic[:3, 3],
            'extrinsic': extrinsic,
        }
    return None


def parse_calibration_results(txt_path: str) -> Dict[str, Dict]:
    """Parse calibration_results.txt file.

    Returns:
        Dict mapping camera_id to dict with 'extrinsic', 'original_extrinsic', etc.
    """
    results = {}

    with open(txt_path, 'r') as f:
        content = f.read()

    # Split by camera sections
    camera_sections = content.split("=" * 60)

    for section in camera_sections:
        if "Camera:" not in section:
            continue

        # Extract camera ID
        cam_match = re.search(r'Camera:\s*(\w+)', section)
        if not cam_match:
            continue
        cam_id = cam_match.group(1)

        # Check if this has the new format with OPTIMIZED/ORIGINAL sections
        if "=== OPTIMIZED CALIBRATION ===" in section:
            # New format with both optimized and original
            optimized_section = section.split("=== ORIGINAL CALIBRATION ===")[0]
            original_section = section.split("=== ORIGINAL CALIBRATION ===")[1] if "=== ORIGINAL CALIBRATION ===" in section else ""

            optimized = _parse_extrinsic_section(optimized_section)
            original = _parse_extrinsic_section(original_section) if original_section else None

            if optimized:
                results[cam_id] = {
                    'rotation': optimized['rotation'],
                    'translation': optimized['translation'],
                    'extrinsic': optimized['extrinsic'],
                }
                if original:
                    results[cam_id]['original_extrinsic'] = original['extrinsic']
        else:
            # Old format - single extrinsic (backwards compatibility)
            parsed = _parse_extrinsic_section(section)
            if parsed:
                results[cam_id] = parsed

    return results


def load_intrinsics(calib_path: str, is_wide: bool = False, wide_offset_y: float = 0) -> np.ndarray:
    """Load camera intrinsics from JSON calibration file.

    Returns:
        3x3 intrinsic matrix K
    """
    import json

    with open(calib_path, 'r') as f:
        calib = json.load(f)

    intrinsic = calib['intrinsic_params']
    fx = intrinsic['fx']
    fy = intrinsic['fy']
    cx = intrinsic['cx']
    cy = intrinsic['cy']

    # Apply wide camera correction if needed
    if is_wide:
        cx = 2 * cx
        cy = 2 * cy - wide_offset_y

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    return K


def load_point_cloud(pcd_path: str) -> np.ndarray:
    """Load point cloud from PCD file.

    Returns:
        Nx4 array (x, y, z, intensity)
    """
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # Try to get intensity/colors
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        intensity = colors[:, 0]  # Use red channel as intensity
    else:
        intensity = np.ones(len(points))

    return np.column_stack([points, intensity])


def project_points_to_image(
    points: np.ndarray,
    K: np.ndarray,
    T_lidar_to_cam: np.ndarray,
    image_size: Tuple[int, int],
    downsample: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points onto 2D image plane.

    Args:
        points: Nx3 or Nx4 array of 3D points (with optional intensity)
        K: 3x3 intrinsic matrix
        T_lidar_to_cam: 4x4 extrinsic transformation
        image_size: (width, height) of image
        downsample: Downsampling factor for intrinsics

    Returns:
        pixels: Mx2 array of valid pixel coordinates
        depths: M array of depths
        intensities: M array of intensities
    """
    # Extract xyz and intensity
    xyz = points[:, :3]
    if points.shape[1] > 3:
        intensity = points[:, 3]
    else:
        intensity = np.ones(len(points))

    # Transform to camera frame
    ones = np.ones((len(xyz), 1))
    points_homo = np.hstack([xyz, ones])  # Nx4
    points_cam = (T_lidar_to_cam @ points_homo.T).T  # Nx4
    points_cam = points_cam[:, :3]  # Nx3

    # Filter points behind camera
    valid_depth = points_cam[:, 2] > 0.1
    points_cam = points_cam[valid_depth]
    intensity = intensity[valid_depth]

    if len(points_cam) == 0:
        return np.array([]), np.array([]), np.array([])

    # Apply downsampling to intrinsics
    K_scaled = K.copy()
    K_scaled[0, 0] /= downsample  # fx
    K_scaled[1, 1] /= downsample  # fy
    K_scaled[0, 2] /= downsample  # cx
    K_scaled[1, 2] /= downsample  # cy

    # Project to image plane
    depths = points_cam[:, 2]
    points_2d = (K_scaled @ points_cam.T).T  # Nx3
    pixels = points_2d[:, :2] / points_2d[:, 2:3]  # Nx2

    # Filter points outside image
    width, height = image_size[0] // downsample, image_size[1] // downsample
    valid_x = (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
    valid_y = (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    valid = valid_x & valid_y

    return pixels[valid], depths[valid], intensity[valid]


def depth_to_color(depths: np.ndarray, min_depth: float = 1.0, max_depth: float = 50.0) -> np.ndarray:
    """Convert depth values to RGB colors using a colormap.

    Returns:
        Nx3 array of RGB colors (0-255)
    """
    # Normalize depths
    depths_norm = np.clip((depths - min_depth) / (max_depth - min_depth), 0, 1)

    # Use a simple rainbow colormap (red=close, blue=far)
    colors = np.zeros((len(depths), 3), dtype=np.uint8)

    # HSV to RGB conversion (H varies with depth, S=1, V=1)
    h = (1 - depths_norm) * 0.7  # Red (0) to Blue (0.7)

    for i, hue in enumerate(h):
        if hue < 1/6:
            colors[i] = [255, int(hue * 6 * 255), 0]
        elif hue < 2/6:
            colors[i] = [int((2/6 - hue) * 6 * 255), 255, 0]
        elif hue < 3/6:
            colors[i] = [0, 255, int((hue - 2/6) * 6 * 255)]
        elif hue < 4/6:
            colors[i] = [0, int((4/6 - hue) * 6 * 255), 255]
        elif hue < 5/6:
            colors[i] = [int((hue - 4/6) * 6 * 255), 0, 255]
        else:
            colors[i] = [255, 0, int((1 - hue) * 6 * 255)]

    return colors


def create_overlay(
    image: Image.Image,
    pixels: np.ndarray,
    depths: np.ndarray,
    point_size: int = 2,
    alpha: float = 0.8,
) -> Image.Image:
    """Create overlay of projected points on image.

    Args:
        image: PIL Image
        pixels: Mx2 array of pixel coordinates
        depths: M array of depth values
        point_size: Size of points to draw
        alpha: Transparency of overlay

    Returns:
        PIL Image with overlay
    """
    # Create a copy of the image
    overlay = image.copy().convert('RGBA')
    draw = ImageDraw.Draw(overlay)

    # Get colors based on depth
    colors = depth_to_color(depths)

    # Sort by depth (far to near) so closer points are drawn on top
    sort_idx = np.argsort(-depths)
    pixels = pixels[sort_idx]
    colors = colors[sort_idx]

    # Draw points
    for (x, y), color in zip(pixels, colors):
        x, y = int(x), int(y)
        color_rgba = tuple(color) + (int(255 * alpha),)
        draw.ellipse(
            [x - point_size, y - point_size, x + point_size, y + point_size],
            fill=color_rgba
        )

    # Composite with original
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    return result.convert('RGB')


def main():
    parser = argparse.ArgumentParser(
        description="Visualize calibration by overlaying point cloud on image"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to calibration_results.txt",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Camera ID (e.g., cam02)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to visualize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="overlay.png",
        help="Output image path",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Image downsampling factor",
    )
    parser.add_argument(
        "--point-size",
        type=int,
        default=1,
        help="Size of projected points",
    )
    parser.add_argument(
        "--all-pcd",
        action="store_true",
        help="Use all PCD frames (concatenated)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also output original calibration overlay for comparison",
    )

    args = parser.parse_args()

    # Camera configurations (matching config.py)
    CAMERA_CONFIGS = {
        'cam02': {'folder': 'cam02', 'calib': 'fwc_c.json', 'is_wide': True, 'wide_offset_y': 500},
        'cam03': {'folder': 'cam03', 'calib': 'fnc.json', 'is_wide': False, 'wide_offset_y': 0},
        'cam04': {'folder': 'cam04', 'calib': 'rnc_r.json', 'is_wide': False, 'wide_offset_y': 0},
        'cam05': {'folder': 'cam05', 'calib': 'fwc_r.json', 'is_wide': True, 'wide_offset_y': 300},
        'cam06': {'folder': 'cam06', 'calib': 'rnc_c.json', 'is_wide': False, 'wide_offset_y': 0},
        'cam07': {'folder': 'cam07', 'calib': 'fwc_l.json', 'is_wide': True, 'wide_offset_y': 300},
        'cam08': {'folder': 'cam08', 'calib': 'rnc_l.json', 'is_wide': False, 'wide_offset_y': 0},
    }

    if args.camera not in CAMERA_CONFIGS:
        print(f"Unknown camera: {args.camera}")
        print(f"Available cameras: {list(CAMERA_CONFIGS.keys())}")
        return

    cam_config = CAMERA_CONFIGS[args.camera]
    data_dir = Path(args.data_dir)

    # Parse calibration results
    print(f"Loading calibration results from {args.results}...")
    calib_results = parse_calibration_results(args.results)

    if args.camera not in calib_results:
        print(f"Camera {args.camera} not found in calibration results")
        print(f"Available cameras: {list(calib_results.keys())}")
        return

    extrinsic = calib_results[args.camera]['extrinsic']
    print(f"Loaded extrinsic matrix for {args.camera}")

    # Load intrinsics
    calib_path = data_dir / cam_config['folder'] / cam_config['calib']
    print(f"Loading intrinsics from {calib_path}...")
    K = load_intrinsics(
        str(calib_path),
        is_wide=cam_config['is_wide'],
        wide_offset_y=cam_config['wide_offset_y']
    )

    # Load image
    image_dir = data_dir / cam_config['folder'] / 'images'
    image_files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))
    if not image_files:
        print(f"No images found in {image_dir}")
        return

    if args.frame >= len(image_files):
        print(f"Frame {args.frame} out of range (0-{len(image_files)-1})")
        return

    image_path = image_files[args.frame]
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path)
    original_size = image.size

    # Downsample image
    if args.downsample > 1:
        new_size = (image.width // args.downsample, image.height // args.downsample)
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Load point cloud(s)
    pc_dir = data_dir / 'pc'
    if args.all_pcd:
        print("Loading all PCD frames...")
        pcd_files = sorted(pc_dir.glob('*.pcd'))
        all_points = []
        for pcd_file in pcd_files:
            points = load_point_cloud(str(pcd_file))
            all_points.append(points)
        points = np.vstack(all_points)
        print(f"Loaded {len(points)} points from {len(pcd_files)} files")
    else:
        pcd_files = sorted(pc_dir.glob('*.pcd'))
        if not pcd_files:
            print(f"No PCD files found in {pc_dir}")
            return

        pcd_idx = min(args.frame, len(pcd_files) - 1)
        pcd_path = pcd_files[pcd_idx]
        print(f"Loading point cloud from {pcd_path}...")
        points = load_point_cloud(str(pcd_path))

    print(f"Loaded {len(points)} points")

    # Project points with OPTIMIZED calibration
    print("Projecting points to image (optimized calibration)...")
    pixels, depths, intensities = project_points_to_image(
        points, K, extrinsic, original_size, args.downsample
    )
    print(f"Projected {len(pixels)} visible points")

    if len(pixels) == 0:
        print("Warning: No points visible in image!")
        print("This might indicate calibration issues or wrong camera/frame selection")

    # Create overlay
    print("Creating overlay...")
    result = create_overlay(image, pixels, depths, args.point_size)

    # Save result
    result.save(args.output)
    print(f"Saved optimized overlay to {args.output}")

    # Print some stats
    if len(depths) > 0:
        print(f"\nDepth statistics (optimized):")
        print(f"  Min: {depths.min():.2f}m")
        print(f"  Max: {depths.max():.2f}m")
        print(f"  Mean: {depths.mean():.2f}m")

    # Also create original calibration overlay if --compare and original exists
    if args.compare and 'original_extrinsic' in calib_results[args.camera]:
        original_extrinsic = calib_results[args.camera]['original_extrinsic']
        print("\nProjecting points to image (original calibration)...")
        orig_pixels, orig_depths, _ = project_points_to_image(
            points, K, original_extrinsic, original_size, args.downsample
        )
        print(f"Projected {len(orig_pixels)} visible points")

        # Reload image for original overlay
        image_orig = Image.open(image_path)
        if args.downsample > 1:
            new_size = (image_orig.width // args.downsample, image_orig.height // args.downsample)
            image_orig = image_orig.resize(new_size, Image.Resampling.LANCZOS)

        orig_result = create_overlay(image_orig, orig_pixels, orig_depths, args.point_size)

        # Save original overlay with _original suffix
        output_path = Path(args.output)
        orig_output = output_path.parent / f"{output_path.stem}_original{output_path.suffix}"
        orig_result.save(orig_output)
        print(f"Saved original overlay to {orig_output}")

        if len(orig_depths) > 0:
            print(f"\nDepth statistics (original):")
            print(f"  Min: {orig_depths.min():.2f}m")
            print(f"  Max: {orig_depths.max():.2f}m")
            print(f"  Mean: {orig_depths.mean():.2f}m")
    elif args.compare:
        print("\nWarning: --compare specified but no original calibration found in results file")


if __name__ == "__main__":
    main()
