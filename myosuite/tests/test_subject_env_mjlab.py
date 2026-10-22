# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Subject CPU/mjlab construction, step parity and partial reset isolation."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace

import numpy as np
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from myosuite import make_env  # noqa: E402
from myosuite.core.config import EnvConfig  # noqa: E402
from myosuite.core.subject_env import subject_metadata  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks.registration import rebuild_twin_cfg  # noqa: E402
from myosuite.tests.test_subject_env import ENV_ID, subject_config  # noqa: E402
from myosuite.utils.checkpoint_manifest import contract_from_env  # noqa: E402

pytestmark = pytest.mark.tier2


@pytest.mark.parametrize("force_scale", [None, 0.5])
def test_subject_twin_config_copies_and_has_identical_model_identity(
    force_scale: float | None,
) -> None:
    from myosuite.envs.wrappers import SarcopeniaWrapper

    subject = subject_config()
    features = (
        ()
        if force_scale is None
        else ((SarcopeniaWrapper, {"force_scale": force_scale}),)
    )
    config = EnvConfig(ENV_ID, subject=subject, features=features)
    cpu = make_env(config)
    try:
        cfg = rebuild_twin_cfg(
            ENV_ID, features=config.features, task_kwargs={"subject": subject}
        )
        assert subject_metadata(cfg) == subject_metadata(cpu)
        assert subject_metadata(copy.deepcopy(cfg)) == subject_metadata(cpu)
        assert asdict(cfg)["events"]["subject_provenance"]["params"][
            "metadata"
        ] == subject_metadata(cpu)
        reset = cfg.events["reset_scene_to_default"].params
        np.testing.assert_allclose(
            reset["qpos"], cpu.unwrapped.model.key_qpos[subject.reset_keyframe]
        )
        assert cfg.rewards["done"].params["walk"].min_height == subject.min_height
    finally:
        cpu.close()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_subject_twin_reset_step_and_partial_reset(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    config = EnvConfig(ENV_ID, subject=subject_config(), max_episode_steps=50)
    cpu = make_env(config)
    mj = make_env(config, backend="mjlab", device=device, num_envs=2)
    try:
        cpu_obs, info = cpu.reset(seed=11)
        obs, gpu_info = mj.reset(seed=11)
        assert gpu_info["subject_model"] == info["subject_model"]
        assert contract_from_env(cpu) == contract_from_env(mj)
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
        assert done[0].item() == cpu_done
        before = next_obs["actor"][1].clone()
        data_before = mj.scene["robot"].data.joint_pos[1].clone()
        partial, _ = mj.reset(env_ids=torch.tensor([0], device=device))
        torch.testing.assert_close(
            mj.scene["robot"].data.joint_pos[1], data_before, atol=0, rtol=0
        )
        torch.testing.assert_close(partial["actor"][1], before, atol=0, rtol=0)
        first, _ = mj.reset(seed=19)
        first = first["actor"].clone()
        first_state = mj.scene["robot"].data.joint_pos.clone()
        second, _ = mj.reset(seed=19)
        torch.testing.assert_close(
            mj.scene["robot"].data.joint_pos, first_state, atol=0, rtol=0
        )
        # CUDA reductions in derived muscle quantities vary by a few float32 ulps.
        torch.testing.assert_close(second["actor"], first, atol=1e-6, rtol=0)
    finally:
        cpu.close()
        mj.close()


@pytest.mark.parametrize("mesh_soles", [False, True])
def test_subject_noisy_cuda_reset_realigns_soles_and_isolates_other_env(
    mesh_soles: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    config = EnvConfig(
        ENV_ID, subject=subject_config(joint_offsets={"hip_rotation_r": (-0.02, 0.02)})
    )
    if mesh_soles:
        ground = replace(config.subject.model.ground, sole_geoms=("foot_r", "foot_l"))
        morphology = replace(config.subject.model, ground=ground)
        config = replace(config, subject=replace(config.subject, model=morphology))
    mj = make_env(config, backend="mjlab", device="cuda:0", num_envs=2)
    try:
        mj.reset(seed=13)
        entity = mj.scene["robot"]
        first = entity.data.joint_pos.clone()
        root = entity.data.root_link_pos_w.clone()
        mj.reset(seed=13)
        torch.testing.assert_close(entity.data.joint_pos, first, atol=0, rtol=0)
        torch.testing.assert_close(entity.data.root_link_pos_w, root, atol=1e-6, rtol=0)
        from myosuite.core.subject_model import align_root_to_ground
        from myosuite.envs.myo.backends.mjlab.tasks.mdp.observations import (
            qpos as read_qpos,
        )

        qpos = read_qpos(mj)[0].cpu().numpy()
        cpu = make_env(config)
        try:
            _, delta = align_root_to_ground(
                cpu.unwrapped.model, config.subject.model.ground, qpos
            )
            assert delta == pytest.approx(0, abs=1e-6)
        finally:
            cpu.close()
        before = entity.data.joint_pos[1].clone()
        mj.reset(env_ids=torch.tensor([0], device=mj.device))
        torch.testing.assert_close(entity.data.joint_pos[1], before, atol=0, rtol=0)
    finally:
        mj.close()
