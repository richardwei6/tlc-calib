#!/usr/bin/env python3
"""Training script for TLC-Calib.

Usage:
    # Multi-camera calibration (default)
    python scripts/train.py --config configs/default.yaml

    # Single camera calibration
    python scripts/train.py --config configs/default.yaml --camera cam02

    # With custom options
    python scripts/train.py --data-dir data/ --iterations 30000 --vram-limit 8
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import yaml

from tlc_calib.config import (
    CameraConfig,
    DataConfig,
    ModelConfig,
    TLCCalibConfig,
    TrainingConfig,
    DEFAULT_CAMERAS,
)
from tlc_calib.optimization import TLCCalibTrainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TLC-Calib: Targetless LiDAR-Camera Calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Single camera ID to calibrate (e.g., 'cam02'). If not specified, calibrates all cameras.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help="List of camera IDs to calibrate (e.g., 'cam02 cam03')",
    )

    # Training arguments
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--vram-limit",
        type=int,
        default=8,
        help="VRAM limit in GB (0 for unlimited)",
    )
    parser.add_argument(
        "--image-downsample",
        type=int,
        default=4,
        help="Image downsampling factor",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (typically 1 for this method)",
    )

    # Model arguments
    parser.add_argument(
        "--num-auxiliaries",
        type=int,
        default=5,
        help="Number of auxiliary Gaussians per anchor (K)",
    )
    parser.add_argument(
        "--avc-beta",
        type=float,
        default=5000.0,
        help="AVC proportionality constant (beta)",
    )
    parser.add_argument(
        "--scale-threshold",
        type=float,
        default=10.0,
        help="Scale ratio threshold (sigma)",
    )

    # Optimization arguments
    parser.add_argument(
        "--rotation-lr",
        type=float,
        default=2e-3,
        help="Learning rate for rotation parameters",
    )
    parser.add_argument(
        "--translation-lr",
        type=float,
        default=8e-3,
        help="Learning rate for translation parameters",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for first half of training",
    )
    parser.add_argument(
        "--lambda-dssim",
        type=float,
        default=0.2,
        help="D-SSIM loss weight",
    )
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=1.0,
        help="Scale regularization weight",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and results",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5000,
        help="Checkpoint save interval",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Logging interval",
    )

    # Device arguments
    # Auto-detect best available device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"

    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Device to use for training (cuda, mps, or cpu)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def load_config_from_yaml(yaml_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def build_config(args: argparse.Namespace) -> TLCCalibConfig:
    """Build configuration from arguments and optional YAML config."""
    # Load YAML config if provided
    yaml_config = {}
    if args.config and Path(args.config).exists():
        yaml_config = load_config_from_yaml(args.config)

    # Determine cameras to use
    if args.camera:
        camera_ids = [args.camera]
    elif args.cameras:
        camera_ids = args.cameras
    elif 'cameras' in yaml_config.get('data', {}):
        camera_ids = yaml_config['data']['cameras']
    else:
        camera_ids = list(DEFAULT_CAMERAS.keys())

    # Build camera configs with optional overrides from YAML
    camera_configs = {}
    yaml_cameras = yaml_config.get('cameras', {})

    for cam_id in camera_ids:
        if cam_id not in DEFAULT_CAMERAS:
            raise ValueError(f"Unknown camera ID: {cam_id}")

        base_config = DEFAULT_CAMERAS[cam_id]

        # Apply YAML overrides for this camera
        if cam_id in yaml_cameras:
            cam_overrides = yaml_cameras[cam_id]
            camera_configs[cam_id] = CameraConfig(
                name=base_config.name,
                folder=base_config.folder,
                calib_file=base_config.calib_file,
                is_wide=base_config.is_wide,
                wide_offset_y=cam_overrides.get('wide_offset_y', base_config.wide_offset_y),
            )
        else:
            camera_configs[cam_id] = base_config

    # Data config
    data_config = DataConfig(
        data_dir=Path(args.data_dir),
        cameras=camera_ids,
        image_downsample=args.image_downsample,
        pc_folder=yaml_config.get('data', {}).get('pc_folder', 'pc'),
        image_folder=yaml_config.get('data', {}).get('image_folder', 'images'),
    )

    # Model config
    model_config = ModelConfig(
        num_auxiliary=args.num_auxiliaries,
        feature_dim=yaml_config.get('model', {}).get('anchor_feature_dim', 32),
        hidden_dim=yaml_config.get('model', {}).get('mlp_hidden_dim', 32),
        sh_degree=yaml_config.get('model', {}).get('sh_degree', 0),
    )

    # Training config
    training_config = TrainingConfig(
        num_iterations=args.iterations,
        rotation_lr=args.rotation_lr,
        translation_lr=args.translation_lr,
        weight_decay=args.weight_decay,
        weight_decay_until=args.iterations // 2,
        lambda_ssim=args.lambda_dssim,
        lambda_scale=args.lambda_scale,
        sigma=args.scale_threshold,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        unlimited_vram=args.vram_limit == 0,
    )

    # Create full config
    config = TLCCalibConfig(
        output_dir=Path(args.output_dir),
    )
    config.data = data_config
    config.model = model_config
    config.training = training_config
    config.camera_configs = camera_configs

    # Store avc_beta in voxel config
    config.voxel.beta = args.avc_beta

    return config


def main():
    """Main training entry point."""
    # Parse arguments
    args = parse_args()

    # Create output directory FIRST (before logging setup)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to both console and file
    log_file = output_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")

    # Build config
    logger.info("Building configuration...")
    config = build_config(args)

    # Log configuration
    logger.info(f"Data directory: {config.data.data_dir}")
    logger.info(f"Cameras: {config.data.cameras}")
    logger.info(f"Image downsample: {config.data.image_downsample}")
    logger.info(f"Iterations: {config.training.num_iterations}")
    logger.info(f"Device: {args.device}")

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, trying MPS...")
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        args.device = "cpu"

    if args.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif args.device == "mps":
        logger.info("Using Apple Metal Performance Shaders (MPS)")
        # MPS has limited memory, reduce max anchors
        if config.voxel.max_anchors > 50000:
            logger.warning("Reducing max_anchors to 50000 for MPS memory constraints")
            config.voxel.max_anchors = 50000
    else:
        logger.warning("Using CPU - training will be slow")
        # CPU mode needs very aggressive limits
        if config.voxel.max_anchors > 20000:
            logger.warning("Reducing max_anchors to 20000 for CPU memory constraints")
            config.voxel.max_anchors = 20000

    # Create trainer
    logger.info("Creating trainer...")
    trainer = TLCCalibTrainer(config=config, device=args.device)

    # Setup trainer
    trainer.setup()

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("Starting training...")
    history = trainer.train()

    # Get and save results
    logger.info("Getting calibration results...")
    results = trainer.get_calibration_results()

    # Save results
    results_path = output_dir / "calibration_results.pt"
    torch.save(results, results_path)
    logger.info(f"Saved results to {results_path}")

    # Save human-readable text file with rotation and translation
    txt_path = output_dir / "calibration_results.txt"
    with open(txt_path, 'w') as f:
        f.write("TLC-Calib Calibration Results\n")
        f.write("=" * 60 + "\n\n")

        for cam_id, result in results.items():
            extrinsic = result['extrinsic']  # 4x4 matrix
            rotation = extrinsic[:3, :3]  # 3x3 rotation matrix
            translation = extrinsic[:3, 3]  # 3x1 translation vector

            f.write(f"Camera: {cam_id}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Rotation Error: {result['rotation_error_deg']:.6f} degrees\n")
            f.write(f"Translation Error: {result['translation_error_m']:.6f} meters\n\n")

            f.write("Rotation Matrix (3x3):\n")
            for i in range(3):
                f.write(f"  [{rotation[i, 0]:12.8f}, {rotation[i, 1]:12.8f}, {rotation[i, 2]:12.8f}]\n")

            f.write("\nTranslation Vector (x, y, z):\n")
            f.write(f"  [{translation[0]:12.8f}, {translation[1]:12.8f}, {translation[2]:12.8f}]\n")

            f.write("\nFull Extrinsic Matrix (4x4):\n")
            for i in range(4):
                f.write(f"  [{extrinsic[i, 0]:12.8f}, {extrinsic[i, 1]:12.8f}, {extrinsic[i, 2]:12.8f}, {extrinsic[i, 3]:12.8f}]\n")

            f.write("\n" + "=" * 60 + "\n\n")

    logger.info(f"Saved human-readable results to {txt_path}")

    # Print final results
    logger.info("\n" + "=" * 50)
    logger.info("FINAL CALIBRATION RESULTS")
    logger.info("=" * 50)

    for cam_id, result in results.items():
        logger.info(f"\n{cam_id}:")
        logger.info(f"  Rotation error: {result['rotation_error_deg']:.4f} degrees")
        logger.info(f"  Translation error: {result['translation_error_m']:.4f} meters")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
