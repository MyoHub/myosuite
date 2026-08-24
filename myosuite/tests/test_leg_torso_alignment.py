# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Rigid-torso / myoLeg pelvis heading must agree at the standing keyframe.

pip myo_sim's ``myolegs_chain`` pelvis is OpenSim-framed
(``euler="1.57 -1.57 0"``, Y-up / X-anterior) while the legacy rigid torso
is a Z-up sibling under ``root``. Without a compensating yaw the character
is twisted 90° at the waist — visible on ``myoLegDirectional*-v0`` and
``myoLegWalk-v0``, which share ``myolegs_with_torso.xml``.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
import pytest

pytestmark = pytest.mark.tier1

_DIRECTIONAL_IDS = (
    "myoLegDirectionalForward-v0",
    "myoLegDirectionalBackward-v0",
    "myoLegDirectionalRandom-v0",
)


def _xy_heading(axis_xyz: np.ndarray) -> np.ndarray:
    """Return the unit XY projection of a body axis."""
    xy = np.asarray(axis_xyz[:2], dtype=np.float64)
    norm = np.linalg.norm(xy)
    assert norm > 1e-6, f"axis has no planar component: {axis_xyz}"
    return xy / norm


def _assert_torso_faces_pelvis(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Torso +X (anterior) must match pelvis +X in the world XY plane."""
    torso = model.body("torso").id
    pelvis = model.body("pelvis").id
    torso_fwd = _xy_heading(data.xmat[torso].reshape(3, 3)[:, 0])
    pelvis_ant = _xy_heading(data.xmat[pelvis].reshape(3, 3)[:, 0])
    np.testing.assert_allclose(
        torso_fwd,
        pelvis_ant,
        atol=0.05,
        err_msg=(
            f"torso +X XY {torso_fwd} vs pelvis +X XY {pelvis_ant} "
            "(90° waist yaw mismatch)"
        ),
    )
    torso_up = data.xmat[torso].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(torso_up, [0.0, 0.0, 1.0], atol=0.05)


@pytest.mark.parametrize("env_id", [*_DIRECTIONAL_IDS, "myoLegWalk-v0"])
def test_standing_torso_faces_same_way_as_legs(env_id: str) -> None:
    """Reset pose must not twist the rigid torso 90° relative to the legs."""
    import myosuite

    myosuite.register_all_envs()
    env = gym.make(env_id)
    try:
        env.reset(seed=0)
        unwrapped = env.unwrapped
        _assert_torso_faces_pelvis(unwrapped.model, unwrapped.data)
        if env_id in _DIRECTIONAL_IDS:
            assert env.observation_space.shape == (153,)
            assert env.action_space.shape == (80,)
    finally:
        env.close()


def test_soccer_torso_faces_same_way_as_legs() -> None:
    """Soccer kicker torso must not be twisted relative to the pelvis.

    The real public competition XML has torso facing +X while pelvis faces
    -Y (verified against commit 8b6635b7~1) — deliberately not replicated;
    see tasks/lessons.md. This asserts the internally-consistent choice.
    """
    import myosuite

    myosuite.register_all_envs()
    env = gym.make("myoChallengeSoccerP1-v0")
    try:
        env.reset(seed=0)
        model = env.unwrapped.model
        data = env.unwrapped.data
        torso = model.body("torso").id
        pelvis = model.body("pelvis").id
        torso_fwd = _xy_heading(data.xmat[torso].reshape(3, 3)[:, 0])
        pelvis_ant = _xy_heading(data.xmat[pelvis].reshape(3, 3)[:, 0])
        np.testing.assert_allclose(
            torso_fwd,
            pelvis_ant,
            atol=0.05,
            err_msg=(
                f"torso +X XY {torso_fwd} vs pelvis +X XY {pelvis_ant} "
                "(90° waist yaw mismatch)"
            ),
        )
        assert env.observation_space.shape == (1273,)
        assert env.action_space.shape == (290,)
    finally:
        env.close()


def test_plane_host_used_by_mjlab_is_also_aligned() -> None:
    """mjlab directional/walk loads ``myolegs_with_torso_plane.xml``."""
    from myosuite.envs.myo.assets._resolve import resolve_leg_xml
    from myosuite.utils.asset_path_resolver import resolve_model_xml_path

    xml = resolve_model_xml_path(resolve_leg_xml("myolegs_with_torso_plane.xml"))
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    _assert_torso_faces_pelvis(model, data)
    assert model.nq == 35
