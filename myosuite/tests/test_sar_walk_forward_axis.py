# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``myoFullBodyWalkSAR-v0`` rewards walking the way the body faces."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from myosuite.tests.support.optional_deps import (
    require_mjlab,
    require_mujoco_warp,
    require_musclemimic_models,
)

pytestmark = pytest.mark.tier2

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

_TARGET_VEL = 1.2


@pytest.fixture(scope="module", autouse=True)
def _require_stack() -> None:
    require_mjlab()
    require_mujoco_warp()
    require_musclemimic_models()
    pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def model() -> Any:
    from ml_collections import config_dict

    from myosuite.integrations.musclemimic.fullbody_model import (
        build_mimic_fullbody_spec,
        default_mimic_fullbody_config,
    )

    cfg = config_dict.create(**dict(default_mimic_fullbody_config()))
    return build_mimic_fullbody_spec(cfg)[0].compile()


@pytest.fixture(scope="module")
def walk_env(model: Any, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """Two-env CPU ``myoFullBodyWalkSAR-v0``, built by its registration."""
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic
    from myosuite.integrations.musclemimic.sar_extraction import (
        extract_synergies,
        save_synergy_model,
    )

    sar_dir = tmp_path_factory.mktemp("sar")
    activations = np.random.default_rng(0).random((200, model.nu))
    save_synergy_model(extract_synergies(activations, n_synergies=4), sar_dir)
    registered: dict[str, Any] = {}
    mimic.register_directional_walk_sar(
        register_mjlab_task=lambda **task: registered.update({task["task_id"]: task}),
        rl_cfg_fn=lambda: None,
        sar_dir=sar_dir,
        target_vel=_TARGET_VEL,
    )
    env_cfg = registered["myoFullBodyWalkSAR-v0"]["env_cfg"]
    env_cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env.reset()
    yield env
    env.close()


def _heel_to_toe_xy(model: Any, yaw: float = 0.0) -> np.ndarray:
    """Unit horizontal heel-to-toe direction of the standing pose, root yawed by *yaw*."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    turn = np.zeros(4)
    mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), yaw)
    mujoco.mju_mulQuat(data.qpos[3:7], turn, data.qpos[3:7].copy())
    mujoco.mj_kinematics(model, data)
    feet = [
        data.xpos[model.body(f"toes_{s}").id] - data.xpos[model.body(f"calcn_{s}").id]
        for s in "rl"
    ]
    xy = np.mean(feet, axis=0)[:2]
    return xy / np.linalg.norm(xy)


def test_forward_axis_is_the_way_the_feet_point(model: Any) -> None:
    """The anterior axis the reward uses is the heel-to-toe direction of the model."""
    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic

    assert int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
    root_quat = model.key_qpos[0][3:7]
    np.testing.assert_allclose(root_quat, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(
        mimic._anterior_axis_xy(model), _heel_to_toe_xy(model), atol=0.02
    )


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, -2.0])
def test_reward_peaks_for_walking_the_way_the_body_faces(
    walk_env: Any, model: Any, yaw: float
) -> None:
    """Env 0 moves at the target speed along its facing direction, env 1 against it."""
    entity = walk_env.scene["fullbody_walk_robot"]
    facing = torch.tensor(_heel_to_toe_xy(model, yaw), dtype=torch.float32)
    state = torch.zeros(2, 13)
    state[:, :3] = entity.data.root_link_pos_w
    state[:, 3] = math.cos(yaw / 2)
    state[:, 6] = math.sin(yaw / 2)
    state[0, 7:9] = _TARGET_VEL * facing
    state[1, 7:9] = -_TARGET_VEL * facing
    entity.write_root_state_to_sim(state)

    reward = walk_env.reward_manager.get_term_cfg("forward_vel").func(walk_env)

    # The feet point within 0.1 deg of the anterior axis, far inside the tolerance.
    assert float(reward[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(reward[1]) == pytest.approx(
        math.exp(-((2 * _TARGET_VEL) ** 2)), abs=1e-4
    )
