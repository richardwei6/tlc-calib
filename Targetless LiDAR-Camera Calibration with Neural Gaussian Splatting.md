# TLC-Calib Implementation Plan

## Overview
Implement TLC-Calib (Targetless LiDAR-Camera Calibration with Neural Gaussian Splatting) - a method for jointly optimizing camera extrinsic parameters and neural Gaussian scene representation from time-series image/PCD data.

## Actual Data Structure
```
data/
├── pc/                         # Shared LiDAR point clouds
│   ├── 000000.pcd ... 000009.pcd  (10 frames, 230K points each)
├── cam02/  (FWC_C - Front Wide Center)  ← wide camera
│   ├── fwc_c.json              # Calibration (intrinsics + extrinsics)
│   └── images/                 # 000000.png ... 000097.png (98 frames)
├── cam03/  (FNC - Front Narrow Center)
│   ├── fnc.json
│   └── images/
├── cam04/  (RNC_R - Rear Narrow Right)
├── cam05/  (FWC_R - Front Wide Right)   ← wide camera
├── cam06/  (RNC_C - Rear Narrow Center)
├── cam07/  (FWC_L - Front Wide Left)    ← wide camera
├── cam08/  (RNC_L - Rear Narrow Left)
```

**Image specs:** 7680x2856 RGB (will downsample for training)
**PCD specs:** 230,400 points per frame (x, y, z, intensity), ASCII format
**Note:** 10 PCD frames are pre-registered (same coordinate frame) - simply concatenate for anchor creation

## Calibration JSON Format
```json
{
  "intrinsic_params": {
    "fx": 1903.3, "fy": 1904.73, "cx": 1914, "cy": 1075.27,
    "k1": -0.015, "k2": -0.048, "k3": 0.059, "k4": -0.028,
    "camera_model": "fisheye"  // or "normal"
  },
  "extrinsic_params": {
    "roll": -89.49, "pitch": 0.93, "yaw": -89.84,  // degrees
    "px": 1.96, "py": 0.004, "pz": -0.526,          // meters
    "quaternion": {"x": -0.49, "y": 0.50, "z": -0.50, "w": 0.51},
    "camera_coordinate": "OPTICAL"
  }
}
```

## Wide Camera Intrinsics Handling
Wide cameras require special intrinsics transformation (from convert_lucid_calib.py):

| Camera | Folder | JSON | Default Y Offset |
|--------|--------|------|------------------|
| FWC_C (Front Wide Center) | cam02 | fwc_c.json | 500 |
| FWC_L (Front Wide Left) | cam07 | fwc_l.json | 300 |
| FWC_R (Front Wide Right) | cam05 | fwc_r.json | 300 |

**Y offsets are configurable per camera in config** (may vary across sessions):
```yaml
cameras:
  cam02:
    wide_offset_y: 500  # adjustable per session
  cam05:
    wide_offset_y: 300
  cam07:
    wide_offset_y: 300
```

**Wide camera formula (at full 7680x2856 resolution):**
```python
cx = 2 * cx_original
cy = 2 * cy_original - wide_offset_y
# fx, fy unchanged
```

**When downsampling images by factor S:**
```python
cx_scaled = cx / S
cy_scaled = cy / S  # offset already baked into cy
fx_scaled = fx / S
fy_scaled = fy / S
```

## Configuration Options
- **Single camera mode:** Calibrate one camera at a time
- **Multi-camera mode:** Calibrate all cameras simultaneously (rig optimization)
- **VRAM mode:** 8GB default, unlimited optional

## Project Structure

```
tlc-calib/
├── pyproject.toml
├── configs/
│   └── default.yaml
├── tlc_calib/
│   ├── __init__.py
│   ├── config.py                    # Dataclass configs
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               # TLCCalibDataset
│   │   ├── lidar_loader.py          # PCD loading, aggregation
│   │   └── image_loader.py          # Image sequence loading
│   ├── models/
│   │   ├── __init__.py
│   │   ├── anchor_gaussians.py      # Frozen LiDAR-derived anchors
│   │   ├── auxiliary_mlp.py         # MLP for auxiliary Gaussian attributes
│   │   ├── gaussian_model.py        # Combined scene representation
│   │   └── camera_rig.py            # SE(3) camera poses
│   ├── rendering/
│   │   ├── __init__.py
│   │   ├── rasterizer.py            # Differentiable rasterizer wrapper
│   │   └── projection.py            # Camera projection utilities
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── photometric.py           # L1 + D-SSIM
│   │   └── regularization.py        # Scale regularization
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Main training loop
│   │   └── adaptive_voxel.py        # Voxel size binary search
│   └── utils/
│       ├── __init__.py
│       ├── lie_groups.py            # SO(3)/SE(3) operations
│       └── transforms.py            # Coordinate transforms
└── scripts/
    ├── train.py                     # Entry point
    └── evaluate.py                  # Evaluation script
```

## Key Components

### 1. Anchor Gaussians (Frozen)
- Concatenate all 10 pre-registered PCD files into global point cloud (~2.3M points)
- Filter NaN values and voxelize with adaptive voxel size ε*
- Voxel centers become anchor positions (FROZEN during training)
- Learnable: anchor features f_i, learned scales ℓ_i

### 2. Auxiliary Gaussians (Learnable)
- 2-layer MLP, 32 hidden units, ReLU
- Input: anchor_feature + view_direction + learned_scale
- Output per anchor: K=5 auxiliary Gaussians with offsets δ, scales, rotations, colors (SH), opacities
- Final position: m_{i,k} = v_i + δ_{i,k}

### 3. Camera Rig Optimization
- Parameterize extrinsics as axis-angle rotation + translation
- Separate learning rates: rotation=2e-3, translation=8e-3
- Same update applied to all images from same camera

### 4. Differentiable Rendering
- Use gsplat for CUDA rasterization
- Custom projection layer for camera pose gradients via chain rule
- Gradients flow through: 2D mean, 2D covariance, view-dependent color

### 5. Loss Functions
- **Photometric**: L_photo = (1-λ)×L1 + λ×D-SSIM (λ=0.2)
- **Scale regularization**: L_scale = mean(max(max(s)/min(s) - σ, 0)) with σ=10
- **Total**: L_total = L_photo + λ_scale × L_scale (λ_scale=1.0)

### 6. Training Schedule
- 30K iterations
- AdamW with weight_decay=0.01 for first 15K iterations, then 0
- Random image sampling per iteration

## Implementation Order

### Phase 1: Foundation
1. `pyproject.toml` with dependencies (torch, open3d, gsplat, etc.)
2. `config.py` - configuration dataclasses
3. `utils/lie_groups.py` - axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
4. `utils/transforms.py` - coordinate transformations

### Phase 2: Data Loading
5. `data/lidar_loader.py` - PCD loading, aggregation, trajectory distance
6. `data/image_loader.py` - image sequence loading
7. `data/dataset.py` - TLCCalibDataset combining both

### Phase 3: Core Models
8. `optimization/adaptive_voxel.py` - binary search for voxel size
9. `models/anchor_gaussians.py` - frozen anchor creation from voxelized LiDAR
10. `models/auxiliary_mlp.py` - MLP for auxiliary attributes
11. `models/gaussian_model.py` - combined scene representation
12. `models/camera_rig.py` - SE(3) pose with rig optimization

### Phase 4: Rendering & Training
13. `rendering/projection.py` - camera projection with Jacobians
14. `rendering/rasterizer.py` - differentiable rasterizer wrapper
15. `losses/photometric.py` - L1 + D-SSIM
16. `losses/regularization.py` - scale regularization
17. `optimization/trainer.py` - main training loop

### Phase 5: Scripts & Evaluation
18. `scripts/train.py` - CLI entry point
19. `scripts/evaluate.py` - metrics computation

## Hyperparameters (from paper)
| Parameter | Value |
|-----------|-------|
| K (auxiliaries per anchor) | 5 |
| β (AVC proportionality) | 5000 |
| σ (scale threshold) | 10 |
| hidden_dim | 32 |
| num_layers | 2 |
| iterations | 30K |
| weight_decay | 0.01 (first 15K) |
| rotation_lr | 2e-3 |
| translation_lr | 8e-3 |
| λ_D-SSIM | 0.2 |
| λ_scale | 1.0 |

## Dependencies
```
torch>=2.0.0
gsplat>=1.0.0
open3d>=0.17.0
numpy>=1.24.0
pillow>=9.0.0
pyyaml>=6.0
tqdm>=4.65.0
pytorch-msssim>=1.0.0
```

## Critical Implementation Notes

1. **Anchor positions must be frozen** - use `register_buffer` not `nn.Parameter`
2. **Gradient flow to camera pose** - requires custom backward through projection
3. **Rig optimization** - all images from same camera share extrinsic update
4. **Memory target** - <8GB VRAM via gradient checkpointing and image downscaling
5. **View direction** - computed from camera center: o_c = -R^T × t
6. **Image downscaling** - 7680x2856 → configurable (e.g., 1920x714 = 4x downsample)
7. **Intrinsics scaling** - fx, fy, cx, cy must scale with image size
8. **Undistorted images** - no distortion correction needed (already preprocessed)
9. **Wide camera Y offsets** - apply cy offset for FWC_C(500), FWC_L(300), FWC_R(300)
