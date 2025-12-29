"""Configuration dataclasses for TLC-Calib."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class CameraConfig:
    """Configuration for a single camera."""

    name: str  # e.g., "FWC_C", "FNC"
    folder: str  # e.g., "cam02"
    calib_file: str  # e.g., "fwc_c.json"
    is_wide: bool = False
    wide_offset_y: float = 0.0  # Y offset for wide cameras (configurable per session)

    @classmethod
    def from_dict(cls, d: dict) -> "CameraConfig":
        return cls(**d)


# Default camera configurations
DEFAULT_CAMERAS = {
    "cam02": CameraConfig(
        name="FWC_C",
        folder="cam02",
        calib_file="fwc_c.json",
        is_wide=True,
        wide_offset_y=500.0,
    ),
    "cam03": CameraConfig(
        name="FNC",
        folder="cam03",
        calib_file="fnc.json",
        is_wide=False,
        wide_offset_y=0.0,
    ),
    "cam04": CameraConfig(
        name="RNC_R",
        folder="cam04",
        calib_file="rnc_r.json",
        is_wide=False,
        wide_offset_y=0.0,
    ),
    "cam05": CameraConfig(
        name="FWC_R",
        folder="cam05",
        calib_file="fwc_r.json",
        is_wide=True,
        wide_offset_y=300.0,
    ),
    "cam06": CameraConfig(
        name="RNC_C",
        folder="cam06",
        calib_file="rnc_c.json",
        is_wide=False,
        wide_offset_y=0.0,
    ),
    "cam07": CameraConfig(
        name="FWC_L",
        folder="cam07",
        calib_file="fwc_l.json",
        is_wide=True,
        wide_offset_y=300.0,
    ),
    "cam08": CameraConfig(
        name="RNC_L",
        folder="cam08",
        calib_file="rnc_l.json",
        is_wide=False,
        wide_offset_y=0.0,
    ),
}


@dataclass
class DataConfig:
    """Configuration for data loading."""

    data_dir: Path = Path("data")
    pc_folder: str = "pc"  # Folder containing point clouds relative to data_dir
    image_folder: str = "images"  # Folder containing images relative to camera folder

    # Image settings
    image_downsample: int = 4  # Downsample factor (1 = full resolution)
    original_width: int = 7680
    original_height: int = 2856

    # Camera selection
    cameras: List[str] = field(default_factory=lambda: ["cam02"])  # Which cameras to use
    single_camera_mode: bool = True  # If True, calibrate one camera at a time

    @property
    def image_width(self) -> int:
        return self.original_width // self.image_downsample

    @property
    def image_height(self) -> int:
        return self.original_height // self.image_downsample


@dataclass
class ModelConfig:
    """Configuration for neural Gaussian model."""

    # Anchor Gaussians
    feature_dim: int = 32  # Dimension of anchor features

    # Auxiliary Gaussians
    num_auxiliary: int = 5  # K auxiliary Gaussians per anchor
    hidden_dim: int = 32  # MLP hidden dimension
    num_layers: int = 2  # Number of MLP layers
    sh_degree: int = 3  # Spherical harmonics degree for view-dependent color


@dataclass
class VoxelConfig:
    """Configuration for Adaptive Voxel Control."""

    beta: float = 5000.0  # Proportionality factor for N_target
    tolerance: int = 100  # Binary search tolerance
    eps_min: float = 0.01  # Minimum voxel size
    eps_max: float = 10.0  # Maximum voxel size
    max_iterations: int = 50  # Max binary search iterations

    # Fixed voxel size (if set, overrides adaptive control)
    fixed_voxel_size: Optional[float] = None


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Iterations
    num_iterations: int = 30000
    weight_decay_until: int = 15000  # Apply weight decay until this iteration

    # Learning rates
    scene_lr: float = 0.001
    rotation_lr: float = 0.002  # From paper: 2e-3
    translation_lr: float = 0.008  # From paper: 8e-3

    # Weight decay
    weight_decay: float = 0.01  # From paper: 10^-2

    # Loss weights
    lambda_ssim: float = 0.2  # D-SSIM weight in photometric loss
    lambda_scale: float = 1.0  # Scale regularization weight

    # Scale regularization threshold
    sigma: float = 10.0  # Aspect ratio threshold

    # Logging and checkpointing
    log_interval: int = 100
    save_interval: int = 5000
    eval_interval: int = 1000

    # Memory optimization
    unlimited_vram: bool = False  # If False, use memory optimization for <8GB
    gradient_checkpointing: bool = True  # Use gradient checkpointing for memory efficiency


@dataclass
class RenderConfig:
    """Configuration for rendering."""

    near: float = 0.1
    far: float = 100.0
    background_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class TLCCalibConfig:
    """Main configuration for TLC-Calib."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    voxel: VoxelConfig = field(default_factory=VoxelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    # Per-camera configurations (overrides defaults)
    camera_configs: Dict[str, CameraConfig] = field(default_factory=lambda: DEFAULT_CAMERAS.copy())

    # Output directory
    output_dir: Path = Path("output")
    experiment_name: str = "tlc_calib"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TLCCalibConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        config = cls()

        # Update data config
        if "data" in data:
            for k, v in data["data"].items():
                if k == "data_dir":
                    v = Path(v)
                if hasattr(config.data, k):
                    setattr(config.data, k, v)

        # Update model config
        if "model" in data:
            for k, v in data["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)

        # Update voxel config
        if "voxel" in data:
            for k, v in data["voxel"].items():
                if hasattr(config.voxel, k):
                    setattr(config.voxel, k, v)

        # Update training config
        if "training" in data:
            for k, v in data["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)

        # Update render config
        if "render" in data:
            for k, v in data["render"].items():
                if hasattr(config.render, k):
                    setattr(config.render, k, v)

        # Update camera configs (allows per-session offset overrides)
        if "cameras" in data:
            for cam_id, cam_data in data["cameras"].items():
                if cam_id in config.camera_configs:
                    # Update existing camera config
                    cam_cfg = config.camera_configs[cam_id]
                    for k, v in cam_data.items():
                        if hasattr(cam_cfg, k):
                            setattr(cam_cfg, k, v)
                else:
                    # Create new camera config
                    config.camera_configs[cam_id] = CameraConfig.from_dict(cam_data)

        # Update output settings
        if "output_dir" in data:
            config.output_dir = Path(data["output_dir"])
        if "experiment_name" in data:
            config.experiment_name = data["experiment_name"]

        return config

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "data": {
                "data_dir": str(self.data.data_dir),
                "pc_folder": self.data.pc_folder,
                "image_folder": self.data.image_folder,
                "image_downsample": self.data.image_downsample,
                "original_width": self.data.original_width,
                "original_height": self.data.original_height,
                "cameras": self.data.cameras,
                "single_camera_mode": self.data.single_camera_mode,
            },
            "model": {
                "feature_dim": self.model.feature_dim,
                "num_auxiliary": self.model.num_auxiliary,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "sh_degree": self.model.sh_degree,
            },
            "voxel": {
                "beta": self.voxel.beta,
                "tolerance": self.voxel.tolerance,
                "eps_min": self.voxel.eps_min,
                "eps_max": self.voxel.eps_max,
                "fixed_voxel_size": self.voxel.fixed_voxel_size,
            },
            "training": {
                "num_iterations": self.training.num_iterations,
                "weight_decay_until": self.training.weight_decay_until,
                "scene_lr": self.training.scene_lr,
                "rotation_lr": self.training.rotation_lr,
                "translation_lr": self.training.translation_lr,
                "weight_decay": self.training.weight_decay,
                "lambda_ssim": self.training.lambda_ssim,
                "lambda_scale": self.training.lambda_scale,
                "sigma": self.training.sigma,
                "log_interval": self.training.log_interval,
                "save_interval": self.training.save_interval,
                "unlimited_vram": self.training.unlimited_vram,
            },
            "render": {
                "near": self.render.near,
                "far": self.render.far,
                "background_color": self.render.background_color,
            },
            "cameras": {
                cam_id: {
                    "name": cfg.name,
                    "folder": cfg.folder,
                    "calib_file": cfg.calib_file,
                    "is_wide": cfg.is_wide,
                    "wide_offset_y": cfg.wide_offset_y,
                }
                for cam_id, cfg in self.camera_configs.items()
            },
            "output_dir": str(self.output_dir),
            "experiment_name": self.experiment_name,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
