# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The exosuit of ``myoTorsoExoPoseFixed-v0`` must sit on the trunk it is welded to.

The exo bodies are placed in world coordinates and their weld poses are taken from the
initial configuration, so a torso frame change (the pip ``myo_sim`` torso is rotated
about the vertical axis relative to the original one) silently leaves the brace beside or
in front of the skeleton.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import myosuite  # noqa: F401  (registers the CPU envs)

pytestmark = pytest.mark.tier1


def _mesh_vertices(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> np.ndarray:
    mesh = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start : start + count]
    return (data.geom_xmat[geom].reshape(3, 3) @ vertices.T).T + data.geom_xpos[geom]


def _geom_names(model: mujoco.MjModel, prefix: str) -> list[int]:
    return [
        g
        for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
            prefix
        )
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    ]


def test_exo_shell_covers_the_back_of_the_rib_cage() -> None:
    env = gym.make("myoTorsoExoPoseFixed-v0").unwrapped
    env.reset(seed=0)
    model, data = env.model, env.data
    ribs = np.vstack(
        [_mesh_vertices(model, data, g) for g in _geom_names(model, "torso_geom")]
    )
    spine = np.vstack(
        [
            _mesh_vertices(model, data, g)
            for g in _geom_names(model, "lumbar") + _geom_names(model, "sacrum")
        ]
    )
    shell = _mesh_vertices(model, data, model.geom("upper_exo_geom").id)

    # The ribs sit in front of the spine: that horizontal direction is anterior.
    anterior = (ribs.mean(0) - spine.mean(0))[:2]
    anterior /= np.linalg.norm(anterior)
    lateral = np.array([-anterior[1], anterior[0]])

    def along(points: np.ndarray, direction: np.ndarray) -> np.ndarray:
        return points[:, :2] @ direction

    # centred on the sagittal plane, and reaching behind the whole rib cage
    lateral_offset = along(shell, lateral).mean() - along(spine, lateral).mean()
    assert abs(lateral_offset) < 0.03, f"shell is {lateral_offset:.3f} m off the spine"
    assert (
        along(shell, anterior).min() < along(ribs, anterior).min()
    ), "shell not behind ribs"
    # a vest opens at the front: the shell's centre lies behind the rib cage's centre
    assert along(shell, anterior).mean() < along(ribs, anterior).mean()
    # and it does not float away from the trunk
    assert along(shell, anterior).min() > along(ribs, anterior).min() - 0.1
