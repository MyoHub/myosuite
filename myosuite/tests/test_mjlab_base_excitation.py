# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for step-rate non-inertial base excitation in the mjlab backend."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers import EventTermCfg, SceneEntityCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

from myosuite.envs.myo.backends.mjlab.tasks.mdp.events import (  # noqa: E402
    apply_base_excitation,
)
from myosuite.physics.base_excitation import (  # noqa: E402
    CompositeBaseProfile,
    HarmonicBaseProfile,
    PulseBaseProfile,
    RecordedBaseProfile,
    inertial_body_wrenches,
)
from myosuite.tests.support.host_sync import HostSyncCounter  # noqa: E402

pytestmark = pytest.mark.tier2


def _build_env_with_excitation(
    profile: object,
    *,
    num_envs: int = 3,
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[ManagerBasedRlEnv, str]:
    import myosuite.envs.myo.backends.mjlab  # noqa: F401, PLC0415

    cfg = load_env_cfg("myoElbowPose1D6MFixed-v0")
    cfg.scene.num_envs = num_envs
    entity_name = next(iter(cfg.scene.entities))
    cfg.events["base_excitation"] = EventTermCfg(
        func=apply_base_excitation,
        mode="step",
        params={
            "asset_cfg": SceneEntityCfg(entity_name),
            "profile": profile,
            "pivot": pivot,
        },
    )
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset(seed=0)
    return env, entity_name


def test_apply_base_excitation_matches_cpu_wrenches_and_resets_cleanly() -> None:
    rec = RecordedBaseProfile(
        times=np.array([0.0, 0.5, 1.0]),
        lin_acc=np.array([[1.2, -0.4, 0.8], [2.4, -0.8, 1.6], [0.0, 0.0, 0.0]]),
        ang_vel=np.array([[0.0, 0.2, -0.1], [0.0, 0.4, -0.2], [0.0, 0.0, 0.0]]),
    )
    harm = HarmonicBaseProfile(
        lin_amp=(0.5, -0.3, 0.7),
        freq_hz=2.0,
        phase_rad=(0.2, 0.4, 0.6),
        ang_amp=(0.1, 0.2, -0.15),
    )
    pulse = PulseBaseProfile(
        start_s=0.0,
        duration_s=0.2,
        peak_lin_acc=(0.9, 0.0, -0.5),
        peak_ang_vel=(0.0, 0.3, 0.0),
    )
    profile = CompositeBaseProfile((harm, pulse, rec))
    pivot = (0.05, -0.02, 0.1)

    env, entity_name = _build_env_with_excitation(profile, num_envs=3, pivot=pivot)
    try:
        entity = env.scene[entity_name]
        term = env.event_manager.get_term_cfg("base_excitation").func
        assert isinstance(term, apply_base_excitation)

        # Invoke once at t=0 before stepping dynamics and compare against CPU reference.
        term(
            env,
            None,
            asset_cfg=SceneEntityCfg(entity_name),
            profile=profile,
            pivot=pivot,
        )
        forces = entity.data.body_external_force[0].detach().cpu().numpy()
        torques = entity.data.body_external_torque[0].detach().cpu().numpy()

        mj_model = env.sim.mj_model
        mj_data = mujoco.MjData(mj_model)
        body_ids = entity.indexing.body_ids.cpu().numpy()
        qpos = entity.data.joint_pos[0].detach().cpu().numpy()
        qvel = entity.data.joint_vel[0].detach().cpu().numpy()
        mj_data.qpos[entity.indexing.joint_q_adr.cpu().numpy()] = qpos
        mj_data.qvel[entity.indexing.joint_v_adr.cpu().numpy()] = qvel
        mujoco.mj_forward(mj_model, mj_data)

        rng = np.random.default_rng(0)
        state = profile.reset(rng, float(env.step_dt))
        kin = profile.sample(0.0, state)
        dyn_mask = mj_model.body_weldid != 0
        root_com = mj_data.subtree_com[mj_model.body_rootid]
        cpu_wrenches = inertial_body_wrenches(
            np,
            mj_model.body_mass,
            mj_model.body_inertia,
            mj_data.xipos,
            mj_data.ximat,
            mj_data.cvel,
            root_com,
            kin.lin_acc,
            ang_vel=kin.ang_vel,
            ang_acc=kin.ang_acc,
            pivot=np.asarray(pivot, dtype=np.float64),
            dynamic_mask=dyn_mask,
        )[body_ids]

        np.testing.assert_allclose(forces, cpu_wrenches[:, :3], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(torques, cpu_wrenches[:, 3:], rtol=1e-5, atol=1e-5)

        # Partial reset clears env 0 wrenches and elapsed time while keeping env 1 intact.
        term.reset(torch.tensor([0], dtype=torch.long))
        assert float(term._elapsed[0].item()) == 0.0
        assert float(term._elapsed[1].item()) == pytest.approx(float(env.step_dt))
        np.testing.assert_allclose(
            entity.data.body_external_force[0].detach().cpu().numpy(), 0.0, atol=1e-7
        )
        assert np.any(
            np.abs(entity.data.body_external_force[1].detach().cpu().numpy()) > 0.0
        )
    finally:
        env.close()


def test_apply_base_excitation_steps_without_host_syncs() -> None:
    profile = CompositeBaseProfile(
        (
            HarmonicBaseProfile(
                lin_amp=(1.0, 0.5, 0.2),
                freq_hz=(1.5, 3.0),
                random_phase=True,
            ),
            PulseBaseProfile(
                start_s=0.02, duration_s=0.1, peak_lin_acc=(1.5, 0.0, 0.0)
            ),
            RecordedBaseProfile(
                times=np.array([0.0, 0.1, 0.2]),
                lin_acc=np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0]]),
                loop=True,
            ),
        )
    )
    env, _ = _build_env_with_excitation(profile, num_envs=2)
    try:
        gen = torch.Generator().manual_seed(7)
        dim = int(env.action_manager.total_action_dim)
        env.episode_length_buf[0] = env.max_episode_length - 1
        saw_reset = False
        for _ in range(4):
            action = torch.rand(env.num_envs, dim, generator=gen) * 2.0 - 1.0
            with HostSyncCounter(package_only=True) as syncs:
                env.step(action)
            assert syncs.total == 0, syncs.report()
            saw_reset = saw_reset or bool(env.reset_buf.any())
        assert saw_reset
    finally:
        env.close()
