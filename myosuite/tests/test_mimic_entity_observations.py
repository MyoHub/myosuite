# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Mimic observations preserve CPU layout and exclude other scene entities."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.tier1
torch = pytest.importorskip("torch")

from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (  # noqa: E402
    _mimic_obs_qpos,
    _mimic_obs_qvel,
)


@pytest.mark.parametrize("fixed_base", [False, True])
def test_mimic_state_is_entity_scoped(fixed_base: bool) -> None:
    origins = torch.tensor([[10.0, 20.0, 0.0], [-10.0, 0.0, 0.0]])
    joint_pos = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    joint_vel = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    root_pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    root_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    entity = SimpleNamespace(
        is_fixed_base=fixed_base,
        data=SimpleNamespace(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            root_link_pos_w=root_pos + origins,
            root_link_quat_w=root_quat,
        ),
        indexing=SimpleNamespace(
            free_joint_v_adr=torch.arange(3, 9),
            joint_v_adr=torch.tensor([9, 10]),
        ),
    )
    scene = type("Scene", (dict,), {})(robot=entity, prop=SimpleNamespace())
    scene.env_origins = origins
    model_qvel = torch.arange(26, dtype=torch.float32).reshape(2, 13)
    model_qvel[:, 9:11] = joint_vel
    env = SimpleNamespace(
        scene=scene,
        sim=SimpleNamespace(data=SimpleNamespace(qvel=model_qvel)),
        physics_dt=0.002,
        cfg=SimpleNamespace(decimation=5),
    )
    expected_qpos = (
        joint_pos if fixed_base else torch.cat([root_pos, root_quat, joint_pos], dim=-1)
    )
    expected_qvel = joint_vel if fixed_base else model_qvel[:, 3:11]
    qpos = _mimic_obs_qpos("robot")(env)
    torch.testing.assert_close(qpos, expected_qpos)
    torch.testing.assert_close(_mimic_obs_qvel("robot")(env), expected_qvel * 0.01)
    qpos.zero_()
    assert torch.count_nonzero(joint_pos) == 4
