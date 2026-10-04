# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU <-> mjlab parity of the effort data (accessor) and the effort terms.

The mjlab twin is synced to a CPU state; the CPU accessor and the
``MjlabEntityAccessor`` must then expose the same muscle state, joint ranges and
joint-space actuator forces (float32 Warp vs float64 MuJoCo), and the opt-in
mjlab effort rewards must score the state like the shared CPU terms.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.tier2

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

import myosuite  # noqa: E402, F401
from myosuite.envs.myo.backends.mjlab.mjlab_env_base import (  # noqa: E402
    MjlabEntityAccessor,
)
from myosuite.envs.myo.backends.mjlab.tasks import mdp  # noqa: E402
from myosuite.terms import effort  # noqa: E402

_N = 2  # parallel twins, all synced to the same CPU state


def _synced_pair(env_id: str, steps: int = 6) -> tuple[gym.Env, ManagerBasedRlEnv]:
    import myosuite.envs.myo.backends.mjlab  # noqa: F401, PLC0415 (registers twins)

    cpu = gym.make(env_id).unwrapped
    cpu.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(steps):
        cpu.step(rng.uniform(-1, 1, cpu.action_space.shape).astype(np.float32))
    cfg = load_env_cfg(env_id)
    cfg.scene.num_envs = _N
    mj = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    mj.reset()

    def _t(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.repeat(np.asarray(x)[None], _N, 0), dtype=torch.float32
        )

    mdp.write_cpu_state(
        mj, "robot", torch.arange(_N), _t(cpu.data.qpos), _t(cpu.data.qvel)
    )
    mj.sim.data.act[:] = _t(cpu.data.act)
    mj.sim.data.ctrl[:] = _t(cpu.data.ctrl)
    mj.sim.forward()
    return cpu, mj


def _close(mj_value: torch.Tensor, cpu_value: np.ndarray, **tol: float) -> None:
    for i in range(_N):
        np.testing.assert_allclose(mj_value[i].numpy(), cpu_value, **tol)


@pytest.mark.parametrize("env_id", ["myoElbowPose1D6MExoRandom-v0", "myoLegWalk-v0"])
def test_accessor_twin(env_id: str) -> None:
    """Muscle state, muscle parameters, joint ranges and qfrc_actuator agree."""
    cpu_env, mj = _synced_pair(env_id)
    cpu, twin = cpu_env._accessor, MjlabEntityAccessor(mj, "robot")
    _close(twin.muscle_length(), cpu.muscle_length(), atol=1e-6)
    _close(twin.muscle_velocity(), cpu.muscle_velocity(), atol=1e-5)
    _close(twin.muscle_force(), cpu.muscle_force(), rtol=1e-5, atol=1e-2)
    _close(twin.qfrc_actuator(), cpu.qfrc_actuator(), rtol=1e-5, atol=1e-2)
    assert twin.muscle_force().shape == (_N, len(cpu.muscle_force()))

    qids_t, ranges_t = twin.joint_range()
    qids_c, ranges_c = cpu.joint_range()
    assert qids_t.tolist() == qids_c.tolist()
    np.testing.assert_allclose(ranges_t.numpy(), ranges_c, atol=1e-6)
    # The ids index the same joints in both CPU-layout qpos vectors.
    _close(twin.joint_pos()[:, qids_t], cpu.joint_pos()[qids_c], atol=1e-5)

    params_t, params_c = twin.muscle_params(), cpu.muscle_params()
    for name in params_c.__dataclass_fields__:
        np.testing.assert_allclose(
            getattr(params_t, name).numpy(),
            getattr(params_c, name),
            rtol=1e-6,
            err_msg=name,
        )


@pytest.mark.parametrize("env_id", ["myoElbowPose1D6MExoRandom-v0", "myoLegWalk-v0"])
def test_effort_rewards_twin(env_id: str) -> None:
    """The opt-in mjlab effort rewards equal the CPU effort terms on the same state."""
    cpu_env, mj = _synced_pair(env_id)
    cpu = cpu_env._accessor
    dofs = [0] if cpu_env.model.nv == 1 else [6, 7, 8]  # elbow; right hip
    pairs = [
        (
            mdp.muscle_mechanical_power(mj),
            effort.muscle_mechanical_power(cpu, {})["muscle_power_abs"],
        ),
        (
            mdp.muscle_mechanical_power(mj, mode="positive"),
            effort.muscle_mechanical_power(cpu, {}, mode="positive")[
                "muscle_power_positive"
            ],
        ),
        (
            mdp.metabolic_energy_rate(mj),
            effort.metabolic_energy_rate(cpu, {})["metabolic_rate"],
        ),
        (
            mdp.metabolic_energy_rate(mj, version="2010"),
            effort.metabolic_energy_rate(cpu, {}, version="2010")["metabolic_rate"],
        ),
        (
            mdp.consumed_endurance_step(
                mj, shoulder_dof_ids=dofs, max_shoulder_torque=10.0
            ),
            effort.consumed_endurance(
                cpu, {}, shoulder_dof_ids=dofs, max_shoulder_torque=10.0
            )["ce_step"],
        ),
        (
            mdp.joint_limit_discomfort(mj),
            effort.joint_limit_discomfort(cpu, {})["joint_limit_discomfort"],
        ),
    ]
    for mj_value, cpu_value in pairs:
        assert mj_value.shape == (_N,)
        _close(mj_value, cpu_value, rtol=1e-4, atol=1e-4)


def test_fatigue_mf_twin() -> None:
    """mdp.fatigue_mf reads the twin's TorchFatigueState like fatigue_effort on CPU."""
    cpu_env, mj = _synced_pair("myoFatiElbowPose1D6MRandom-v0", steps=30)
    state = mj.action_manager.get_term("muscles").fatigue_state
    fatigue = cpu_env.muscle_fatigue
    for name in ("MA", "MR", "MF"):
        getattr(state, name)[:] = torch.as_tensor(
            getattr(fatigue, name), dtype=torch.float32
        )
    expected = effort.fatigue_effort(cpu_env._accessor, {"fatigue": fatigue})[
        "fatigue_mf"
    ]
    assert expected > 0.0
    _close(mdp.fatigue_mf(mj), expected, rtol=1e-6)
    # Without fatigue the twin has no state and the reward is zero.
    _, plain = _synced_pair("myoElbowPose1D6MRandom-v0", steps=0)
    assert plain.action_manager.get_term("muscles").fatigue_state is None
    assert mdp.fatigue_mf(plain).tolist() == [0.0] * _N
