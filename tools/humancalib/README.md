# HumanCalib Multi-Camera Adapter (`myosuite-humancalib`)

Standalone CLI for targetless multi-camera extrinsic calibration from walking human pose tracks (`.npz`), writing Pose2Sim-compatible `Calib_scene_calibrated.toml` files and metric Z-up 3D landmarks via `myosuite.biomechanics.camera_calibration`.

## Attribution & Licensing

The calibration solver adapts the pipeline and validation insights of:

- **Florian Delaplace**, *HumanCalib: multi-camera extrinsic calibration from human pose* (v0.5.4, 2026), MIT License, [github.com/flodelaplace/HumanCalib](https://github.com/flodelaplace/HumanCalib), [doi:10.5281/zenodo.23143081](https://doi.org/10.5281/zenodo.23143081).
- **S.-E. Lee, K. Shibata, S. Nonaka, S. Nobuhara, K. Nishino**, *Extrinsic Camera Calibration From a Moving Person*, IEEE Robotics and Automation Letters 7(4), 2022, [doi:10.1109/LRA.2022.3192629](https://doi.org/10.1109/LRA.2022.3192629).

**Upstream pose-estimator weights are NOT bundled and have separate non-commercial terms:**
- `MeTRAbs` (`metrabs_l`): non-commercial research use only.
- `VideoPose3D`: CC BY-NC 4.0.
- For commercial workflows, extract poses with a commercial-permissible estimator (such as `tools/instanthmr --hmr-model medium-commercial`) and pass the `.npz` tracks to `myosuite-humancalib`.

## Usage

```bash
myosuite-humancalib Calib_intrinsics.toml poses.npz -o Calib_scene_calibrated.toml --height 1.78
```
