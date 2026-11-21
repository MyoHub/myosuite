# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for foveated retinal acuity blur and backend-agnostic gaze/posture terms."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from myosuite.biomechanics.foveation import (
    FoveationConfig,
    crop_foveal_patch,
    eccentricity_deg,
    foveate_image,
    intrinsic_from_fovy,
)
from myosuite.core.protocols import PhysicsPath
from myosuite.terms.base_obs import gaze_target_offset_obs
from myosuite.terms.base_reward import gaze_alignment_reward, posture_comfort_reward

pytestmark = pytest.mark.tier1


class _MockGazeAccessor:
    def __init__(self, xp: Any, qpos: Any, sites: Any) -> None:
        self._xp = xp
        self._qpos = qpos
        self._sites = sites

    @property
    def physics_path(self) -> PhysicsPath:
        return (
            PhysicsPath.MJLAB
            if getattr(self._xp, "__name__", "") == "torch"
            else PhysicsPath.CPU
        )

    def array_module(self) -> Any:
        return self._xp

    def joint_pos(self) -> Any:
        return self._qpos

    def joint_vel(self) -> Any:
        return self._xp.zeros_like(self._qpos)

    def muscle_act(self) -> Any:
        return self._qpos[..., :1]

    def site_xpos(self, site_ids: Any) -> Any:
        idx = list(site_ids)
        return self._sites[..., idx, :]

    def time(self) -> Any:
        return 0.0

    def ctrl_range(self) -> Any:
        return self._xp.zeros((1, 2))

    def joint_limits(self) -> tuple[Any, Any, Any]:
        z = self._xp.zeros(self._qpos.shape[-1])
        return z, z, z > 1

    def dt(self) -> float:
        return 0.02

    def site_id(self, name: str) -> int:
        return int(name)


def test_foveation_preserves_fovea_blurs_periphery_and_matches_torch() -> None:
    torch = pytest.importorskip("torch")

    h, w = 128, 128
    k = intrinsic_from_fovy(h, w, fovy_deg=90.0)
    theta = eccentricity_deg(h, w, k, fixation_px=(63.5, 63.5))
    assert theta[63, 63] < 1.0
    assert theta[0, 0] > 45.0

    # Checkerboard high-frequency stimulus: contrast decays monotonically with eccentricity.
    yy, xx = np.indices((h, w))
    checker = (((xx + yy) % 2) * 255.0).astype(np.float64)
    rgb = np.stack([checker, 255.0 - checker, 0.5 * checker], axis=-1)

    cfg = FoveationConfig(fovea_sigma0_deg=0.05, half_resolution_deg=2.0, n_levels=5)
    fov_np = foveate_image(rgb, k, fixation_px=(63.5, 63.5), cfg=cfg)
    assert fov_np.shape == (h, w, 3)

    center_std = float(np.std(fov_np[60:68, 60:68, 0]))
    mid_std = float(np.std(fov_np[32:40, 60:68, 0]))
    periph_std = float(np.std(fov_np[8:16, 8:16, 0]))
    assert center_std > mid_std > periph_std
    assert center_std > 6.0 * periph_std

    # Batched PyTorch (B, C, H, W) matches NumPy output closely.
    rgb_th = torch.as_tensor(np.transpose(rgb, (2, 0, 1))[None].repeat(2, axis=0))
    fov_th = foveate_image(rgb_th, k, fixation_px=(63.5, 63.5), cfg=cfg)
    assert fov_th.shape == (2, 3, h, w)
    fov_th_hwc = np.transpose(fov_th[0].numpy(), (1, 2, 0))
    np.testing.assert_allclose(fov_th_hwc[8:-8, 8:-8], fov_np[8:-8, 8:-8], atol=1.5)

    patch = crop_foveal_patch(rgb, patch_size=(16, 16), fixation_px=(2.0, 126.0))
    assert patch.shape == (16, 16, 3)


def test_importing_foveation_does_not_load_torch() -> None:
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, myosuite.biomechanics.foveation\n"
            "import json; print(json.dumps('torch' in sys.modules))",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(proc.stdout.strip().splitlines()[-1]) is False


def test_gaze_and_posture_terms_cpu_and_batched_torch() -> None:
    torch = pytest.importorskip("torch")

    # CPU single-env: gaze along +X from (0,0,1) through (1,0,1), target at (2,0,1).
    sites_np = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    qpos_np = np.array([0.0, 0.4, -0.2], dtype=np.float32)
    acc_np = _MockGazeAccessor(np, qpos_np, sites_np)

    obs_aligned = gaze_target_offset_obs(
        acc_np, (0, 1), target_pos=np.array([2.0, 0.0, 1.0], dtype=np.float32)
    )
    assert obs_aligned.shape == (4,) and obs_aligned.dtype == np.float32
    np.testing.assert_allclose(obs_aligned, [0.0, 0.0, 0.0, 1.0], atol=1e-6)

    rwd_aligned = gaze_alignment_reward(
        acc_np,
        {"gaze_site_ids": (0, 1), "target_pos": np.array([2.0, 0.0, 1.0])},
        angular_thd_rad=0.1,
    )
    assert rwd_aligned["solved"] is True
    assert rwd_aligned["gaze_alignment"] == pytest.approx(1.0, abs=1e-6)

    comfort = posture_comfort_reward(
        acc_np,
        {},
        neutral_qpos=[0.0, 0.0],
        joint_indices=(1, 2),
        joint_weights=[2.0, 1.0],
    )
    assert 0.0 < comfort["posture_comfort"] < 1.0
    assert comfort["done"] is False

    # Batched Torch (N=2): env 0 aligned (+X), env 1 orthogonal (+Y).
    sites_th = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    qpos_th = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.5, -0.5]], dtype=torch.float32)
    targets_th = torch.tensor([[3.0, 0.0, 1.0], [0.0, 3.0, 1.0]], dtype=torch.float32)
    acc_th = _MockGazeAccessor(torch, qpos_th, sites_th)

    obs_th = gaze_target_offset_obs(acc_th, (0, 1), target_pos=targets_th)
    assert obs_th.shape == (2, 4)
    assert obs_th[0, 3].item() == pytest.approx(1.0, abs=1e-6)
    assert obs_th[1, 3].item() == pytest.approx(0.0, abs=1e-6)

    rwd_th = gaze_alignment_reward(
        acc_th, {"gaze_site_ids": (0, 1), "target_pos": targets_th}, angular_thd_rad=0.1
    )
    assert rwd_th["solved"].tolist() == [True, False]

    comfort_th = posture_comfort_reward(
        acc_th, {}, neutral_qpos=[0.0, 0.0], joint_indices=(1, 2)
    )
    assert comfort_th["posture_comfort"].shape == (2,)
    assert (
        comfort_th["posture_comfort"][0].item()
        > comfort_th["posture_comfort"][1].item()
    )
