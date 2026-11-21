# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""A subject's leg walk on mjlab: the twin builds the CPU model, resets and steps alike."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from myosuite import make_env  # noqa: E402
from myosuite.core.config import EnvConfig  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks import cpu_reference as ref  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks.registration import (  # noqa: E402
    rebuild_twin_cfg,
)

pytestmark = pytest.mark.tier2

ENV_ID = "myoLegWalk-v0"
SCALES = {"femur_r": [1.02, 1.07, 0.98], "tibia_l": 1.04}
SUBJECT = {"scales": SCALES, "total_mass": 70.0}


def test_twin_builds_the_cpu_subject_model() -> None:
    cpu = make_env(ENV_ID, **SUBJECT)
    try:
        model = cpu.unwrapped.model
        cfg = rebuild_twin_cfg(ENV_ID, task_kwargs=SUBJECT)
        np.testing.assert_array_equal(
            cfg.events["reset_scene_to_default"].params["qpos"], model.key_qpos[2]
        )
        assert cfg.sim.mujoco == ref.mujoco_cfg_from_model(model)
        entity = cfg.scene.entities["robot"].spec_fn().compile()
        for field in ("body_pos", "body_mass", "body_inertia", "actuator_lengthrange"):
            np.testing.assert_array_equal(
                getattr(entity, field), getattr(model, field), err_msg=field
            )
    finally:
        cpu.close()


def test_nominal_twin_is_unchanged() -> None:
    from mjlab.tasks.registry import load_env_cfg  # noqa: PLC0415

    task = ref.cpu_task_spec(ENV_ID)
    assert not task.is_subject
    assert ref._model_key(task) == (task.kwargs["model_path"], None, None)
    registered = load_env_cfg(ENV_ID).events["reset_scene_to_default"].params
    rebuilt = rebuild_twin_cfg(ENV_ID).events["reset_scene_to_default"].params
    np.testing.assert_array_equal(rebuilt["qpos"], registered["qpos"])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_subject_twin_resets_and_steps_like_the_cpu_env(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    config = EnvConfig(ENV_ID, task_kwargs=SUBJECT, max_episode_steps=50)
    cpu = make_env(config)
    mj = make_env(config, backend="mjlab", device=device, num_envs=2)
    try:
        cpu_obs, _ = cpu.reset(seed=11)
        obs, _ = mj.reset(seed=11)
        np.testing.assert_allclose(obs["actor"][0].cpu().numpy(), cpu_obs, atol=5e-4)
        action = np.full(cpu.action_space.shape, 0.15, dtype=np.float32)
        cpu_next, cpu_reward, cpu_done, _, _ = cpu.step(action)
        next_obs, reward, done, _, _ = mj.step(
            torch.as_tensor(np.stack([action, action]), device=device)
        )
        np.testing.assert_allclose(
            next_obs["actor"][0].cpu().numpy(), cpu_next, atol=1e-2
        )
        assert reward[0].item() == pytest.approx(cpu_reward, abs=1e-2)
        assert bool(done[0].item()) == cpu_done
        # A partial reset leaves the other env alone.
        before = mj.scene["robot"].data.joint_pos[1].clone()
        mj.reset(env_ids=torch.tensor([0], device=device))
        torch.testing.assert_close(
            mj.scene["robot"].data.joint_pos[1], before, atol=0, rtol=0
        )
    finally:
        cpu.close()
        mj.close()


def test_subject_kwargs_pass_through_make_env() -> None:
    by_kwargs = make_env(ENV_ID, backend="mjlab", device="cpu", **SUBJECT)
    try:
        robot = by_kwargs.scene["robot"]
        mass = by_kwargs.sim.mj_model.body_mass[robot.indexing.body_ids.cpu().numpy()]
        assert mass.sum() == pytest.approx(70.0)
    finally:
        by_kwargs.close()


@pytest.mark.parametrize(
    "env_id", ["myoLegRoughTerrainWalk-v0", "myoElbowPose1D6MRandom-v0"]
)
def test_twins_without_a_subject_model_refuse_scales(env_id: str) -> None:
    with pytest.raises(NotImplementedError, match="subject"):
        make_env(env_id, backend="mjlab", device="cpu", scales=SCALES)
