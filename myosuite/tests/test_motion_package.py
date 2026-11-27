# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for ``myosuite.motion`` contracts, composable EMGTrial, and legacy shims."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from myosuite.motion import (
    EMGTrial,
    KinematicTrial,
    MotionClip,
    PlaybackRequest,
    ReferenceMotion,
)

pytestmark = pytest.mark.tier1


def _sample_clip(
    n_frames: int = 21, nq: int = 4, fps: float | None = 100.0
) -> MotionClip:
    qpos = np.linspace(0.0, 1.0, n_frames * nq, dtype=np.float64).reshape(n_frames, nq)
    qvel = np.zeros_like(qpos)
    return MotionClip(
        qpos=qpos,
        qvel=qvel,
        site_xpos=None,
        site_names=None,
        frequency_hz=fps,
    )


def test_emg_only_trial_without_kinematics() -> None:
    """EMGTrial defaults units to 'unknown' and supports standalone sEMG datasets without kinematics."""
    rng = np.random.default_rng(0)
    signals = rng.uniform(0.0, 1.0, size=(201, 3))
    default_trial = EMGTrial(
        signals=signals,
        channels=("BIClong", "TRIlong", "BRA"),
        sampling_rate_hz=1000.0,
    )
    assert default_trial.units == "unknown"
    assert default_trial.is_emg_only is True

    trial = EMGTrial(
        signals=signals,
        channels=("BIClong", "TRIlong", "BRA"),
        sampling_rate_hz=1000.0,
        trial_id="ninapro_s01_r01",
        kinematics=None,
        units="mvc_normalized",
        muscle_map={"BIClong": "BIClong", "TRIlong": "TRIlong"},
    )
    assert trial.is_emg_only is True
    assert trial.kinematics is None
    assert trial.n_samples == 201
    assert trial.n_channels == 3
    assert trial.duration_s == pytest.approx(0.2)
    assert trial.timestamps_s.shape == (201,)
    assert trial.timestamps_s[-1] == pytest.approx(0.2)

    sub = trial.select_channels(("TRIlong", "BIClong"))
    assert sub.is_emg_only is True
    assert sub.channels == ("TRIlong", "BIClong")
    np.testing.assert_allclose(sub.signals[:, 0], trial.signals[:, 1])
    assert sub.muscle_map == {"TRIlong": "TRIlong", "BIClong": "BIClong"}


def test_synchronized_kinematic_and_emg_trial() -> None:
    """EMGTrial composes cleanly with KinematicTrial when mocap and EMG are paired."""
    kin = KinematicTrial(
        clip=_sample_clip(n_frames=101, fps=100.0),
        trial_id="walk_01",
        valid_mask=np.ones(101, dtype=bool),
    )
    assert kin.n_frames == 101
    assert kin.duration_s == pytest.approx(1.0)

    emg = EMGTrial(
        signals=np.ones((1001, 2), dtype=np.float32),
        channels=("tib_ant_r", "soleus_r"),
        sampling_rate_hz=1000.0,
        trial_id="walk_01",
        kinematics=kin,
        units="mV",
        valid_mask=np.ones((1001, 2), dtype=bool),
        time_offset_s=0.05,
    )
    assert emg.is_emg_only is False
    assert emg.kinematics is kin
    assert emg.duration_s == pytest.approx(1.0)
    assert emg.timestamps_s[0] == pytest.approx(0.05)


def test_trial_validation_guards() -> None:
    clip = _sample_clip(n_frames=10, fps=50.0)
    with pytest.raises(ValueError, match="contradicts"):
        KinematicTrial(clip=clip, frequency_hz=200.0)

    unset_clip = _sample_clip(n_frames=10, fps=None)
    filled = KinematicTrial(clip=unset_clip, frequency_hz=120.0)
    assert filled.frequency_hz == pytest.approx(120.0)

    with pytest.raises(ValueError, match="frequency_hz"):
        KinematicTrial(clip=unset_clip, frequency_hz=-10.0)
    with pytest.raises(ValueError, match="valid_mask"):
        KinematicTrial(clip=clip, valid_mask=np.ones(5, dtype=bool))

    with pytest.raises(ValueError, match="channels"):
        EMGTrial(signals=np.zeros((10, 2)), channels=("ch1",), sampling_rate_hz=100.0)
    with pytest.raises(ValueError, match="unique"):
        EMGTrial(
            signals=np.zeros((10, 2)),
            channels=("ch1", "ch1"),
            sampling_rate_hz=100.0,
        )
    with pytest.raises(ValueError, match="sampling_rate_hz"):
        EMGTrial(
            signals=np.zeros((10, 1)),
            channels=("ch1",),
            sampling_rate_hz=0.0,
        )


def test_motion_and_integrations_shims_preserve_identity() -> None:
    aliases = (
        ("myosuite.core.trajectory_io", "myosuite.motion.io"),
        ("myosuite.core.trajectory_yaw", "myosuite.motion.transforms"),
        ("myosuite.core.playback_contract", "myosuite.motion.playback"),
        ("myosuite.core.hf_dataset_cache", "myosuite.motion.hf_cache"),
        ("myosuite.logger.reference_motion", "myosuite.motion.reference"),
        ("myosuite.core.retargeting", "myosuite.integrations.marker_retarget"),
        (
            "myosuite.biomechanics.opensim_motion",
            "myosuite.integrations.opensim_motion",
        ),
        ("myosuite.biomechanics.smplh", "myosuite.integrations.smplh"),
        (
            "myosuite.biomechanics.retarget_smplh",
            "myosuite.integrations.retarget_smplh",
        ),
        ("myosuite.biomechanics.mano", "myosuite.integrations.mano"),
        ("myosuite.biomechanics.gvhmr", "myosuite.integrations.gvhmr"),
        ("myosuite.biomechanics.chumpy_pickle", "myosuite.integrations.chumpy_pickle"),
        ("myosuite.biomechanics.opensim_grf", "myosuite.integrations.opensim_grf"),
    )
    for legacy, canonical in aliases:
        assert importlib.import_module(legacy) is importlib.import_module(
            canonical
        ), f"{legacy} does not alias {canonical}"

    try:
        legacy_jax = importlib.import_module("myosuite.logger.reference_motion_jax")
        canon_jax = importlib.import_module("myosuite.motion.reference_jax")
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] != "jax":
            raise
    else:
        assert legacy_jax is canon_jax

    from myosuite.biomechanics import external_loads, opensim_grf
    from myosuite.integrations import opensim_grf as int_opensim_grf

    assert callable(external_loads.apply_body_wrenches_to_qfrc)
    assert opensim_grf.opensim_grf_to_qfrc is int_opensim_grf.opensim_grf_to_qfrc
    assert (
        opensim_grf.normalize_opensim_motion is int_opensim_grf.normalize_opensim_motion
    )

    import myosuite.core as core

    assert core.MotionClip is MotionClip
    assert core.PlaybackRequest is PlaybackRequest
    assert ReferenceMotion is not None


def test_core_compat_submodule_attribute_access_in_fresh_process() -> None:
    """``import myosuite.core as c; c.trajectory_io`` continues to resolve lazily."""
    import subprocess
    import sys

    code = (
        "import myosuite.core as c; "
        "assert callable(c.trajectory_io.load_motion_clip); "
        "assert hasattr(c.playback_contract, 'PlaybackRequest'); "
        "assert callable(c.hf_dataset_cache.download_hf_dataset_file)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
