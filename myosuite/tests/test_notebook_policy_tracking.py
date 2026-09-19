# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""End-to-end test for notebook-style policy loading, motion tracking, and frame saving.

Mirrors the logic in ``tutorials/MuscleMimic_Fullbody_Policy_Trajectory.ipynb``.

Requires:
    - ``musclemimic_models``: full-body MJCF model package
    - ``orbax.checkpoint``: Orbax checkpoint reader
    - ``imageio``: video writer backend
    - HuggingFace cache hit for ``amathislab/mm-10m-2``
    - Local MuscleMimic motion cache (KIT/314/walking_medium09_poses)

All are skipped gracefully if unavailable.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level skip guards
# ---------------------------------------------------------------------------

pytest.importorskip("musclemimic_models", reason="musclemimic_models package required")
pytest.importorskip("orbax.checkpoint", reason="orbax.checkpoint required")
pytest.importorskip("imageio", reason="imageio required for frame saving")

pytestmark = pytest.mark.tier2

_CHECKPOINT_REF = "hf://amathislab/mm-10m-2"
_MOTION_PATH = "KIT/314/walking_medium09_poses"
_N_STEPS = 30  # short run for CI


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fullbody_model() -> mujoco.MjModel:
    """Compiled MyoFullBody MuJoCo model."""
    from myosuite.integrations.musclemimic.fullbody_model import (
        compile_mimic_fullbody_mjmodel,
        default_mimic_fullbody_config,
    )

    cfg = default_mimic_fullbody_config()
    model, _spec, _xml = compile_mimic_fullbody_mjmodel(cfg)
    return model


@pytest.fixture(scope="module")
def motion_clip(fullbody_model: mujoco.MjModel):
    """Motion clip loaded from local HF cache."""
    from myosuite.core.trajectory_io import load_motion_clip, resolve_motion_path

    try:
        motion_file = resolve_motion_path(_MOTION_PATH, env_name="MyoFullBody")
    except FileNotFoundError:
        pytest.skip("MuscleMimic motion cache not available")
    return load_motion_clip(
        motion_file,
        expected_nq=fullbody_model.nq,
        expected_nv=fullbody_model.nv,
    )


@pytest.fixture(scope="module")
def checkpoint_root() -> Path:
    """Local HuggingFace checkpoint root."""
    from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
        resolve_checkpoint_ref,
    )

    try:
        checkpoint = resolve_checkpoint_ref(_CHECKPOINT_REF)
    except Exception as exc:
        pytest.skip(f"Checkpoint unavailable: {exc}")
    return checkpoint.local_path


@pytest.fixture(scope="module")
def policy_artifacts(checkpoint_root: Path):
    """Policy artifacts loaded from HuggingFace checkpoint."""
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        has_local_policy_artifacts,
        load_local_policy_artifacts,
    )

    if not has_local_policy_artifacts(checkpoint_root):
        pytest.skip("Checkpoint missing train_state / config artifacts")

    return load_local_policy_artifacts(checkpoint_root)


@pytest.fixture(scope="module")
def policy_runner(policy_artifacts, checkpoint_root: Path, fullbody_model, motion_clip):
    """LocalPolicyRunner using the upstream-compatible fullbody obs adapter."""
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        FullbodyObsAdapter,
        LocalPolicyRunner,
        fullbody_history_settings_from_metadata,
        fullbody_obs_adapter_params_from_metadata,
        read_checkpoint_config_metadata,
    )

    metadata = read_checkpoint_config_metadata(checkpoint_root)
    obs_adapter = FullbodyObsAdapter(
        fullbody_model,
        motion_clip,
        fullbody_obs_adapter_params_from_metadata(metadata),
    )
    return LocalPolicyRunner(
        artifacts=policy_artifacts,
        stochastic=False,
        seed=0,
        frame_skip=5,
        obs_adapter=obs_adapter,
        **fullbody_history_settings_from_metadata(metadata),
    )


# ---------------------------------------------------------------------------
# Checkpoint / artifact loading
# ---------------------------------------------------------------------------


def test_policy_artifacts_dimensions(policy_artifacts) -> None:
    """Checkpoint must expose expected obs/action dimensions for mm-10m-2."""
    assert policy_artifacts.obs_dim == 2418
    assert policy_artifacts.action_dim == 354
    assert policy_artifacts.obs_mean.shape == (2418,)
    assert policy_artifacts.obs_var.shape == (2418,)


def test_policy_artifacts_obs_mean_finite(policy_artifacts) -> None:
    """Running-mean statistics must be finite (no NaN/Inf from corrupted checkpoint)."""
    assert np.all(np.isfinite(policy_artifacts.obs_mean))
    assert np.all(np.isfinite(policy_artifacts.obs_var))
    assert np.all(policy_artifacts.obs_var >= 0.0)


# ---------------------------------------------------------------------------
# Model + motion clip
# ---------------------------------------------------------------------------


def test_fullbody_model_dimensions(fullbody_model: mujoco.MjModel) -> None:
    """Compiled full-body model has expected DoF / actuator counts."""
    assert fullbody_model.nq == 89
    assert fullbody_model.nv == 88
    assert fullbody_model.nu == 354


def test_motion_clip_shape(fullbody_model: mujoco.MjModel, motion_clip) -> None:
    """Motion clip has correct trajectory width and non-trivial length."""
    assert motion_clip.qpos.shape[1] == fullbody_model.nq
    assert motion_clip.qpos.shape[0] > 0
    if motion_clip.qvel is not None:
        assert motion_clip.qvel.shape[1] == fullbody_model.nv


# ---------------------------------------------------------------------------
# Policy rollout — action generation
# ---------------------------------------------------------------------------


def test_policy_action_shape(policy_runner, fullbody_model, motion_clip) -> None:
    """action_for must return a vector matching model.nu."""
    data = mujoco.MjData(fullbody_model)
    data.qpos[:] = motion_clip.qpos[0]
    mujoco.mj_forward(fullbody_model, data)
    action = policy_runner.action_for(data, motion_clip, 0)
    assert action.shape == (fullbody_model.nu,)
    assert action.dtype == np.float32


def test_policy_action_within_clamp(policy_runner, fullbody_model, motion_clip) -> None:
    """Policy output must be clipped to [-1, 1] by LocalPolicyRunner."""
    data = mujoco.MjData(fullbody_model)
    data.qpos[:] = motion_clip.qpos[0]
    mujoco.mj_forward(fullbody_model, data)
    action = policy_runner.action_for(data, motion_clip, 0)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_policy_step_advances_simulation(
    policy_runner, fullbody_model, motion_clip
) -> None:
    """step() must advance data.time."""
    data = mujoco.MjData(fullbody_model)
    data.qpos[:] = motion_clip.qpos[0]
    mujoco.mj_forward(fullbody_model, data)
    t0 = float(data.time)
    action = policy_runner.action_for(data, motion_clip, 0)
    policy_runner.step(fullbody_model, data, action)
    assert float(data.time) > t0, "Simulation time must advance after step()"


# ---------------------------------------------------------------------------
# Rollout loop — tracking quality
# ---------------------------------------------------------------------------


def test_rollout_qpos_l2_errors_bounded(
    policy_runner, fullbody_model, motion_clip
) -> None:
    """Over a short rollout the mean qpos L2 error must stay below a loose threshold.

    This is a sanity check, not a performance gate.  The threshold is set
    conservatively so it catches complete policy failure (e.g. wrong obs dims,
    scrambled weights) without rejecting stochastic variation.
    """
    data = mujoco.MjData(fullbody_model)
    data.qpos[:] = motion_clip.qpos[0]
    if motion_clip.qvel is not None:
        data.qvel[:] = motion_clip.qvel[0]
    mujoco.mj_forward(fullbody_model, data)

    errors: list[float] = []
    n = min(_N_STEPS, int(motion_clip.qpos.shape[0]))
    for i in range(n):
        action = policy_runner.action_for(data, motion_clip, i)
        policy_runner.step(fullbody_model, data, action)
        target = motion_clip.qpos[min(i, motion_clip.qpos.shape[0] - 1)]
        errors.append(float(np.linalg.norm(np.asarray(data.qpos) - np.asarray(target))))

    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))
    # Loose sanity bounds: mean < 5.0 rad, max < 20.0 rad for 30 steps.
    assert mean_err < 5.0, f"Mean qpos L2 error too high: {mean_err:.4f}"
    assert max_err < 20.0, f"Max qpos L2 error too high: {max_err:.4f}"


# ---------------------------------------------------------------------------
# Frame rendering and video saving
# ---------------------------------------------------------------------------


def test_offscreen_render_produces_valid_frames(
    fullbody_model: mujoco.MjModel, motion_clip
) -> None:
    """Off-screen rendering must produce uint8 RGB frames with correct shape."""
    data = mujoco.MjData(fullbody_model)
    data.qpos[:] = motion_clip.qpos[0]
    mujoco.mj_forward(fullbody_model, data)

    width, height = 320, 240
    renderer = mujoco.Renderer(fullbody_model, height=height, width=width)
    try:
        renderer.update_scene(data)
        frame = np.asarray(renderer.render(), dtype=np.uint8)
    finally:
        renderer.close()

    assert frame.shape == (height, width, 3), f"Unexpected frame shape: {frame.shape}"
    assert frame.dtype == np.uint8
    # Frame should not be all zeros (model rendered into the scene).
    assert frame.max() > 0, "Frame is completely black — rendering failed"


def test_rollout_saves_video_file(
    tmp_path: Path,
    policy_runner,
    fullbody_model: mujoco.MjModel,
    motion_clip,
) -> None:
    """Full pipeline: rollout + off-screen render + MP4 write -> valid file."""
    from myosuite.utils.video_io import write_video

    data = mujoco.MjData(fullbody_model)
    data.qpos[:] = motion_clip.qpos[0]
    if motion_clip.qvel is not None:
        data.qvel[:] = motion_clip.qvel[0]
    mujoco.mj_forward(fullbody_model, data)

    width, height = 320, 240
    n_steps = min(10, int(motion_clip.qpos.shape[0]))  # very short for speed
    fps = max(1, int(round(1.0 / float(fullbody_model.opt.timestep))))

    renderer = mujoco.Renderer(fullbody_model, height=height, width=width)
    frames: list[np.ndarray] = []
    try:
        for i in range(n_steps):
            action = policy_runner.action_for(data, motion_clip, i)
            policy_runner.step(fullbody_model, data, action)
            renderer.update_scene(data)
            frames.append(np.asarray(renderer.render(), dtype=np.uint8))
    finally:
        renderer.close()

    assert len(frames) == n_steps

    video_path = tmp_path / "tracking_test.mp4"
    write_video(
        str(video_path),
        np.asarray(frames, dtype=np.uint8),
        outputdict={"-r": str(fps), "-pix_fmt": "yuv420p"},
    )

    assert video_path.exists(), "Video file was not created"
    assert video_path.stat().st_size > 1000, "Video file is suspiciously small"
