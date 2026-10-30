# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""SuperMimic world-model components against MuJoCo and the mjlab Mimic env.

Torch FK, the analytic activation step and the joint couplings are compared with
MuJoCo; the analytic heads with the observation, reward and termination terms of a
CPU mjlab ``myoMimicFullbody`` env on a synthetic clip.
"""

from __future__ import annotations

import functools
from typing import Any

import mujoco
import numpy as np
import pytest

from myosuite.tests.support.optional_deps import (
    require_mjlab,
    require_mujoco_warp,
    require_musclemimic_models,
)

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.tier2

_ENTITY = "mimic_fullbody_robot"


@functools.cache
def _variant(variant: str) -> tuple[Any, Any, Any, tuple[str, ...]]:
    """(mimic config, spec builder, compiled model, tracked site names)."""
    require_musclemimic_models()
    from ml_collections import config_dict

    if variant == "fullbody":
        from myosuite.integrations.musclemimic.fullbody_model import (
            FULLBODY_BODY2SITES_FOR_MIMIC as sites,
            build_mimic_fullbody_spec as build,
            default_mimic_fullbody_config as default_cfg,
        )
    else:
        from myosuite.integrations.musclemimic.bimanual_model import (
            BODY2SITES_FOR_MIMIC as sites,
            build_mimic_bimanual_spec as build,
            default_mimic_config as default_cfg,
        )
    cfg = config_dict.create(**dict(default_cfg()))
    return cfg, build, build(cfg)[0].compile(), tuple(sites.values())


def _random_qpos(model: Any, rng: np.random.Generator) -> np.ndarray:
    qpos = model.qpos0 + rng.normal(0.0, 0.3, model.nq)
    if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        qpos[:3] = rng.normal(0.0, 1.0, 3)
        q = rng.normal(size=4)
        qpos[3:7] = q / np.linalg.norm(q)
    return qpos


@pytest.mark.parametrize("variant", ["fullbody", "bimanual"])
def test_torch_fk_matches_mj_kinematics(variant: str) -> None:
    from myosuite.learning.supermimic.kinematics import TorchKinematics

    _, _, model, sites = _variant(variant)
    ids = np.array([model.site(n).id for n in sites])
    kin = TorchKinematics(model, ids).double()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    for _ in range(5):
        data.qpos[:] = _random_qpos(model, rng)
        mujoco.mj_kinematics(model, data)
        qpos = torch.tensor(data.qpos)[None]
        pos, quat = kin.body_poses(qpos)
        np.testing.assert_allclose(pos[0].numpy(), data.xpos, atol=1e-6)
        sign = np.sign((quat[0].numpy() * data.xquat).sum(-1, keepdims=True))
        np.testing.assert_allclose(quat[0].numpy() * sign, data.xquat, atol=1e-6)
        np.testing.assert_allclose(
            kin.site_pos(qpos)[0].numpy(), data.site_xpos[ids], atol=1e-6
        )


def test_activation_step_matches_mj_step() -> None:
    from myosuite.learning.supermimic.muscle import MuscleActivation

    _, _, model, _ = _variant("fullbody")
    act_step = MuscleActivation(model, 5).double()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(1)
    data.act[:] = rng.uniform(0.0, 1.0, model.na)
    for _ in range(5):
        data.ctrl[:] = rng.uniform(0.0, 1.0, model.nu)
        act0 = torch.tensor(data.act.copy())[None]
        for _ in range(5):
            mujoco.mj_step(model, data)
        pred = act_step(act0, torch.tensor(data.ctrl.copy())[None])[0].numpy()
        np.testing.assert_allclose(pred, data.act, atol=1e-10)


def test_joint_coupling_projection() -> None:
    from myosuite.learning.supermimic.kinematics import JointCoupling

    _, _, model, _ = _variant("fullbody")
    jc = JointCoupling(model).double()
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0] if model.nkey else model.qpos0
    rng = np.random.default_rng(2)
    for _ in range(100):
        data.ctrl[:] = rng.uniform(0.0, 0.3, model.nu)
        mujoco.mj_step(model, data)
    qpos, qvel = torch.tensor(data.qpos)[None], torch.tensor(data.qvel)[None]
    qp, qv = jc.project(qpos, qvel)
    # The simulator satisfies the soft couplings up to its constraint slack.
    assert float((qp - qpos).abs().max()) < 1e-2
    qp2, qv2 = jc.project(qp, qv)
    torch.testing.assert_close(qp2, qp)
    torch.testing.assert_close(qv2, qv)
    assert len(jc.independent_dofs) == model.nv - len(jc.q1)


def test_state_spec_constant_velocity_and_heading_invariance() -> None:
    from myosuite.learning.supermimic.state import StateSpec, quat_from_rotvec, yaw_quat
    from myosuite.physics.quat_math_torch import mul_quat, rot_vec_quat

    nq, nv, na, dt = 10, 9, 4, 0.01
    spec = StateSpec(nq, nv, na, dt, torch.tensor([1.0, 0.0, 0.0]))
    g = torch.Generator().manual_seed(0)
    s = torch.randn(3, spec.dim, generator=g)
    s[:, 3:7] = s[:, 3:7] / s[:, 3:7].norm(dim=-1, keepdim=True)
    act_next = torch.rand(3, na, generator=g)
    nxt = spec.integrate(s, torch.zeros(3, spec.delta_dim), act_next)
    torch.testing.assert_close(nxt[:, :3], s[:, :3] + dt * s[:, nq : nq + 3])
    torch.testing.assert_close(nxt[:, 7:nq], s[:, 7:nq] + dt * s[:, nq + 6 : nq + nv])
    torch.testing.assert_close(
        nxt[:, 3:7], mul_quat(s[:, 3:7], quat_from_rotvec(dt * s[:, nq + 3 : nq + 6]))
    )
    torch.testing.assert_close(nxt[:, nq + nv :], act_next)
    assert float(spec.error(s, s).abs().max()) < 1e-5
    # Rotating the world about z (position and linear velocity too) leaves the features unchanged.
    yaw = yaw_quat(torch.tensor([0.7, -1.2, 2.5]))
    r = s.clone()
    r[:, 3:7] = mul_quat(yaw, s[:, 3:7])
    r[:, :3] = rot_vec_quat(s[:, :3], yaw)
    r[:, nq : nq + 3] = rot_vec_quat(s[:, nq : nq + 3], yaw)
    torch.testing.assert_close(
        spec.features(r, act_next), spec.features(s, act_next), atol=1e-5, rtol=0
    )


def test_window_sampler_skips_episode_ends() -> None:
    from myosuite.learning.supermimic.world_model import WindowSampler

    T, N, k = 20, 3, 4
    state = torch.arange(T + 1).float()[:, None, None].expand(T + 1, N, 2).clone()
    done = torch.zeros(T, N, dtype=torch.bool)
    done[5, 0] = True
    sampler = WindowSampler(state, torch.zeros(T, N, 1), done, k)
    t, n = sampler.starts[:, 0], sampler.starts[:, 1]
    crosses = (n == 0) & (t <= 5) & (t + k - 1 >= 5)
    assert not bool(crosses.any())
    assert len(sampler) == N * (T - k + 1) - k


def _synthetic_clip(model: Any, sites: tuple[str, ...], n_frames: int = 80) -> Any:
    """Walking-like clip whose sites are the FK of its own qpos."""
    from myosuite.core.trajectory_io import MotionClip

    data = mujoco.MjData(model)
    pose = (model.key_qpos[0] if model.nkey else model.qpos0).copy()
    rng = np.random.default_rng(0)
    qpos = np.tile(pose, (n_frames, 1))
    qvel = np.zeros((n_frames, model.nv))
    qvel[:, 0] = 1.0
    qvel[:, 6:] = rng.normal(0.0, 0.05, (n_frames, model.nv - 6))
    ids = [model.site(n).id for n in sites]
    site_xpos = np.zeros((n_frames, len(ids), 3))
    for i in range(n_frames):
        qpos[i, 0] = pose[0] + 0.01 * i
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        site_xpos[i] = data.site_xpos[ids]
    return MotionClip(
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        site_xpos=site_xpos.astype(np.float32),
        site_names=list(sites),
        frequency_hz=100.0,
    )


def test_heads_match_the_mjlab_env_terms() -> None:
    """Analytic obs / reward / termination == the env's terms on the same state."""
    require_mjlab()
    require_mujoco_warp()
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic
    from myosuite.learning.supermimic.env_io import MimicEnvIO
    from myosuite.learning.supermimic.heads import MimicHeads
    from myosuite.learning.supermimic.muscle import MuscleActivation

    cfg, build, model, sites = _variant("fullbody")
    env_cfg = mimic._make_mimic_env_cfg(
        _task_id="test-supermimic",
        entity_name=_ENTITY,
        variant="fullbody",
        spec_fn=lambda: build(cfg)[0],
        muscle_actuators=mimic._muscle_actuator_names(model),
        tendon_targets=mimic._muscle_tendon_names(model),
        sim_dt=float(cfg.sim_dt),
        ctrl_dt=float(cfg.ctrl_dt),
        max_episode_steps=int(cfg.max_episode_steps),
        clip=_synthetic_clip(model, sites),
        num_envs=2,
        mj_model=model,
        mimic_reward_weight=5.0,
    )
    env_cfg.seed = 0
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    try:
        env.reset()
        io = MimicEnvIO(env, _ENTITY)
        heads = MimicHeads.from_env(env, _ENTITY)
        act_step = MuscleActivation(model, env.cfg.decimation, env.physics_dt)
        nq, nv = model.nq, model.nv
        weight = env.reward_manager.get_term_cfg("tracking").weight * heads.ctrl_dt
        g = torch.Generator().manual_seed(0)
        for _ in range(4):
            s0 = io.state()
            action = torch.rand(env.num_envs, model.nu, generator=g)
            obs, rew, term, trunc, _ = env.step(action)
            s, f = io.state(), io.frame()
            qpos, qvel, act = s[:, :nq], s[:, nq : nq + nv], s[:, nq + nv :]
            # An env that ended this step was reset: its state no longer follows s0.
            live = ~(term | trunc)
            torch.testing.assert_close(
                act[live],
                act_step(s0[:, nq + nv :], io.ctrl(action))[live],
                atol=1e-6,
                rtol=0,
            )
            torch.testing.assert_close(
                heads.observation(qpos, qvel, act, f), obs["actor"], atol=1e-5, rtol=0
            )
            env_reward = env.reward_manager.get_term_cfg("tracking").func(env) * weight
            torch.testing.assert_close(
                heads.reward(qpos, qvel, f) * weight, env_reward, atol=1e-3, rtol=0
            )
            env_term = env.termination_manager.get_term_cfg("mimic_deviation").func(env)
            assert torch.equal(heads.termination(qpos, f), env_term)
            soft = heads.termination_prob(qpos, f)
            assert bool(((soft > 0.5) == env_term).all())
    finally:
        env.close()


def test_kinematic_tracking_errors() -> None:
    from myosuite.learning.supermimic.metrics import LEG_JOINTS, KinematicTracking

    _, _, model, _ = _variant("fullbody")
    pose = torch.tensor(
        model.key_qpos[0] if model.nkey else model.qpos0, dtype=torch.float64
    )
    t = torch.linspace(0.0, 1.0, 20, dtype=torch.float64)
    clip = pose.repeat(20, 1)
    adr = [int(model.jnt_qposadr[model.joint(n).id]) for n in LEG_JOINTS]
    clip[:, adr] += (
        0.3 * torch.sin(6.0 * t)[:, None]
    )  # moving reference, so correlations are defined
    track = KinematicTracking(model, clip)
    frames = torch.arange(20)
    track.update(clip, frames, torch.ones(20, dtype=torch.bool))
    out = track.summary()
    assert out["leg_rmse_deg"] < 1e-6 and abs(out["leg_corr"] - 1.0) < 1e-6
    assert out["pelvis_tilt_err_deg"] < 1e-6 and out["trunk_tilt_err_deg"] < 1e-6
    # A constant 0.1 rad offset on every leg joint: RMSE 5.73 deg, correlation still 1.
    track.reset()
    shifted = clip.clone()
    shifted[:, adr] += 0.1
    track.update(shifted, frames, torch.ones(20, dtype=torch.bool))
    out = track.summary()
    assert (
        abs(out["leg_rmse_deg"] - np.degrees(0.1)) < 1e-6
        and abs(out["leg_corr"] - 1.0) < 1e-6
    )
    # Invalid envs are ignored.
    track.reset()
    track.update(shifted, frames, torch.zeros(20, dtype=torch.bool))
    assert track.summary() == {}


def test_gradient_agreement_runs() -> None:
    """The gradient check runs end to end; an untrained world model ignores the controls."""
    from myosuite.learning.supermimic.metrics import gradient_agreement
    from myosuite.learning.supermimic.state import StateSpec
    from myosuite.learning.supermimic.world_model import WorldModel, WorldModelCfg

    _, _, model, _ = _variant("fullbody")
    spec = StateSpec(model.nq, model.nv, model.na, 0.01, torch.tensor([1.0, 0.0, 0.0]))
    wm = WorldModel(model, spec, 5, 0.002, WorldModelCfg(hidden=(32,)))
    pose = model.key_qpos[0] if model.nkey else model.qpos0
    s0 = torch.zeros(2, spec.dim)
    s0[:, : model.nq] = torch.tensor(pose, dtype=torch.float32)
    u = torch.full((2, 2, model.nu), 0.3)
    dt = model.opt.timestep  # the model is cached for the whole module
    model.opt.timestep = 0.002
    try:
        out = gradient_agreement(wm, model, s0, u, 5, n_dirs=3)
    finally:
        model.opt.timestep = dt
    assert set(out) == {"grad_cos_median", "grad_cos_mean", "grad_norm_ratio_median"}
    assert all(np.isfinite(v) for v in out.values())
    assert (
        out["grad_norm_ratio_median"] < 1e-6
    )  # zero-initialised head: no control dependence yet


def test_grouped_actor_exports_a_per_muscle_actor(tmp_path: Any) -> None:
    """A grouped actor copies each group's action to its muscles; its export is an ordinary actor."""
    from myosuite.learning.supermimic.policy import Actor

    groups = [[0, 3], [1], [4, 2, 5]]
    actor = Actor(7, 6, (16,), groups)
    obs = torch.randn(4, 7)
    out = actor(obs)
    for g in groups:
        for m in g[1:]:
            torch.testing.assert_close(out[:, m], out[:, g[0]])
    torch.save({"actor_state_dict": actor.rsl_rl_state_dict()}, tmp_path / "ck.pt")
    plain = Actor.from_rsl_rl(tmp_path / "ck.pt")
    assert plain.groups is None
    torch.testing.assert_close(plain(obs), out)


def test_posture_errors_and_smoothed_site_reward() -> None:
    """Posture errors vanish on the clip; the smoothed site reward has a finite gradient there."""
    from myosuite.learning.supermimic.heads import MimicClip, MimicHeads
    from myosuite.learning.supermimic.kinematics import TorchKinematics

    _, _, model, sites = _variant("fullbody")
    ids = np.array([model.site(n).id for n in sites])
    trunk = int(model.body("torso").id)
    kin = TorchKinematics(model, ids, prune=True, extra_bodies=(trunk,))
    clip = _synthetic_clip(model, sites, n_frames=4)
    q = torch.tensor(clip.qpos, dtype=torch.float32)
    mclip = MimicClip(
        sites=kin.site_pos(q),
        qpos=q,
        qvel=torch.tensor(clip.qvel, dtype=torch.float32),
        qpos_cols=torch.arange(model.nq),
        qvel_cols=torch.arange(model.nv),
        root_pos=True,
        root_quat=True,
        root_vel=True,
    )
    heads = MimicHeads(kin, mclip, 0.01, trunk_body=trunk, trunk_err_threshold=0.5)
    frame = torch.arange(4)
    pelvis, tr = heads.posture_errors(q, frame)
    assert float(pelvis.max()) < 1e-3 and float(tr.max()) < 1e-3
    qpos = q.clone().requires_grad_(True)
    heads.reward(qpos, mclip.qvel, frame).sum().backward()
    assert bool(torch.isfinite(qpos.grad).all())
    # Tilting the trunk joints raises the trunk error and the termination probability.
    bent = q.clone()
    bent[:, int(model.jnt_qposadr[model.joint("flex_extension").id])] += 1.0
    assert float(heads.posture_errors(bent, frame)[1].min()) > 0.3
    assert float(heads.termination_prob(bent, frame).min()) > float(
        heads.termination_prob(q, frame).max()
    )
