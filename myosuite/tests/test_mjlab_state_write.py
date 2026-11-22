# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Entity API state writes for free-only and fixed entities in real mjlab worlds."""

from __future__ import annotations

from collections.abc import Iterator

import mujoco
import numpy as np
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from mjlab.entity import EntityCfg  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.managers.observation_manager import (  # noqa: E402
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.scene import SceneCfg  # noqa: E402
from mjlab.sim import SimulationCfg  # noqa: E402

from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import (  # noqa: E402
    mujoco_cfg_from_model,
)
from myosuite.envs.myo.backends.mjlab.tasks.mdp.events import write_cpu_state  # noqa: E402

pytestmark = pytest.mark.tier2


def _free_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string("""<mujoco><worldbody><body name="body">
      <freejoint name="root"/>
      <geom type="box" size=".1 .12 .15" mass="1" contype="0" conaffinity="0"/>
    </body></worldbody></mujoco>""")


def _fixed_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string("""<mujoco><worldbody><body name="body">
      <geom type="box" size=".1 .1 .1" contype="0" conaffinity="0"/>
    </body></worldbody></mujoco>""")


def _root_position(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["body"].data.root_link_pos_w


@pytest.fixture(scope="module")
def state_env() -> Iterator[ManagerBasedRlEnv]:
    cfg = ManagerBasedRlEnvCfg(
        decimation=1,
        scene=SceneCfg(
            num_envs=2,
            env_spacing=2.0,
            entities={
                "other": EntityCfg(
                    spec_fn=_free_spec,
                    init_state=EntityCfg.InitialStateCfg(pos=(-0.4, 0.0, 1.0)),
                ),
                "body": EntityCfg(
                    spec_fn=_free_spec,
                    init_state=EntityCfg.InitialStateCfg(pos=(0.4, 0.0, 1.0)),
                ),
                "fixed": EntityCfg(spec_fn=_fixed_spec),
            },
        ),
        sim=SimulationCfg(
            mujoco=mujoco_cfg_from_model(_free_spec().compile()),
            nconmax=16,
            njmax=32,
        ),
        observations={
            "actor": ObservationGroupCfg(
                {"position": ObservationTermCfg(func=_root_position)}
            )
        },
        episode_length_s=1.0,
        seed=7,
    )
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset(seed=7)
    yield env
    env.close()


def test_free_only_partial_state_write_preserves_body_angular_velocity(
    state_env: ManagerBasedRlEnv,
) -> None:
    env = state_env
    entity = env.scene["body"]
    indices = entity.indexing
    assert len(indices.joint_q_adr) == 0
    assert int(indices.free_joint_q_adr[0]) > 0  # Another free entity precedes it.
    qpos = torch.tensor(
        [[0.3, -0.2, 1.2, np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0]], dtype=torch.float32
    )
    qvel = torch.tensor([[0.1, 0.2, -0.3, 0.4, -0.3, 0.2]], dtype=torch.float32)
    qpos_input, qvel_input = qpos.clone(), qvel.clone()
    before_q, before_v = env.sim.data.qpos.clone(), env.sim.data.qvel.clone()
    selected = torch.tensor([1], dtype=torch.long)
    write_cpu_state(env, "body", selected, qpos, qvel)
    env.sim.forward()

    expected_qpos = qpos[0].clone()
    expected_qpos[:3] += env.scene.env_origins[1]
    torch.testing.assert_close(
        env.sim.data.qpos[1, indices.free_joint_q_adr.long()], expected_qpos
    )
    torch.testing.assert_close(
        env.sim.data.qvel[1, indices.free_joint_v_adr.long()],
        qvel[0],
        atol=1e-6,
        rtol=0,
    )
    # A +90 degree rotation about world y maps body (x,y,z) to world (z,y,-x).
    torch.testing.assert_close(
        entity.data.root_link_ang_vel_w[1],
        torch.tensor([0.2, -0.3, -0.4]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(env.sim.data.qpos[0], before_q[0], atol=0, rtol=0)
    torch.testing.assert_close(env.sim.data.qvel[0], before_v[0], atol=0, rtol=0)
    other = env.scene["other"].indexing
    torch.testing.assert_close(
        env.sim.data.qpos[1, other.free_joint_q_adr.long()],
        before_q[1, other.free_joint_q_adr.long()],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        env.sim.data.qvel[1, other.free_joint_v_adr.long()],
        before_v[1, other.free_joint_v_adr.long()],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(qpos, qpos_input, atol=0, rtol=0)
    torch.testing.assert_close(qvel, qvel_input, atol=0, rtol=0)


def test_fixed_entity_state_write_is_a_noop(state_env: ManagerBasedRlEnv) -> None:
    env = state_env
    indices = env.scene["fixed"].indexing
    assert len(indices.free_joint_q_adr) == len(indices.joint_q_adr) == 0
    before_q, before_v = env.sim.data.qpos.clone(), env.sim.data.qvel.clone()
    write_cpu_state(
        env, "fixed", torch.tensor([1]), torch.empty((1, 0)), torch.empty((1, 0))
    )
    torch.testing.assert_close(env.sim.data.qpos[:], before_q, atol=0, rtol=0)
    torch.testing.assert_close(env.sim.data.qvel[:], before_v, atol=0, rtol=0)
