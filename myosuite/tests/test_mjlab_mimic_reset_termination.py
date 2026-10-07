# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for mjlab Mimic resets, terminations and task configs.

Real mjlab environments run on CPU. They are built with the same
``_make_mimic_env_cfg`` that the task registration uses. The clips are
synthetic: their site positions come from MuJoCo forward kinematics, so a
simulator state equal to a clip frame tracks that frame perfectly.
"""

from __future__ import annotations

import copy
import functools
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from myosuite.core.trajectory_io import MotionClip
from myosuite.tests.support.optional_deps import (
    require_mjlab,
    require_mujoco_warp,
    require_musclemimic_models,
)

pytestmark = pytest.mark.tier2

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

_ENTITY = {"fullbody": "mimic_fullbody_robot", "bimanual": "mimic_bimanual_robot"}
# Clip root: world-frame linear and body-frame angular velocity (MuJoCo qvel).
_ROOT_LIN_VEL = (0.3, 0.1, 0.0)
_ROOT_ANG_VEL_BODY = (0.5, -1.2, 2.0)
# Folded "pike" pose: hips and trunk flexed, arms forward. Its site centroid is
# about 0.36 m from the pelvis, above the 0.3 m root-error tolerance.
_FULLBODY_PIKE = {
    "hip_flexion_r": 2.0,
    "hip_flexion_l": 2.0,
    "flex_extension": -1.3,
    "shoulder_elv_r": 1.57,
    "shoulder_elv_l": 1.57,
}


@pytest.fixture(scope="module", autouse=True)
def _require_mimic_stack() -> None:
    require_mjlab()
    require_mujoco_warp()
    require_musclemimic_models()


@functools.cache
def _variant(variant: str) -> tuple[Any, Any, Any, tuple[str, ...]]:
    """Return (mimic config, spec builder, compiled model, tracked site names)."""
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


def _axis_angle_quat(axis: tuple[float, ...], angle: float) -> np.ndarray:
    unit_axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, unit_axis, angle)
    return quat


def _synthetic_clip(
    variant: str,
    *,
    n_frames: int = 60,
    joint_pos: dict[str, float] | None = None,
) -> MotionClip:
    """Clip whose sites are the forward kinematics of its own qpos.

    A free root (full body) is tilted, turns over the clip and carries a
    body-frame angular velocity, so world and body frames differ.
    """
    _, _, model, site_names = _variant(variant)
    data = mujoco.MjData(model)
    pose = (model.key_qpos[0] if model.nkey else model.qpos0).copy()
    for name, value in (joint_pos or {}).items():
        pose[model.jnt_qposadr[model.joint(name).id]] = value
    free_root = int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
    n_root_dof = 6 if free_root else 0
    rng = np.random.default_rng(0)
    qpos = np.tile(pose, (n_frames, 1))
    qvel = np.zeros((n_frames, model.nv))
    qvel[:, n_root_dof:] = rng.normal(0.0, 0.05, (n_frames, model.nv - n_root_dof))
    site_ids = [model.site(name).id for name in site_names]
    site_xpos = np.zeros((n_frames, len(site_ids), 3))
    tilt = _axis_angle_quat((1.0, 0.3, 0.0), 0.4)
    for i in range(n_frames):
        if free_root:
            yaw = _axis_angle_quat((0.0, 0.0, 1.0), 0.7 + 0.01 * i)
            mujoco.mju_mulQuat(qpos[i, 3:7], yaw, tilt)
            qpos[i, :3] = pose[:3] + 0.01 * i * np.asarray(_ROOT_LIN_VEL)
            qvel[i, :6] = _ROOT_LIN_VEL + _ROOT_ANG_VEL_BODY
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        site_xpos[i] = data.site_xpos[site_ids]
    return MotionClip(
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        site_xpos=site_xpos.astype(np.float32),
        site_names=list(site_names),
        frequency_hz=100.0,
    )


def _make_env(
    variant: str,
    clip: MotionClip | None,
    *,
    num_envs: int = 2,
    mj_model: Any = None,
    **cfg_kwargs: Any,
) -> Any:
    """mjlab env from ``_make_mimic_env_cfg`` with the registration's arguments.

    ``mj_model`` replaces the compiled model the initial state is read from.
    """
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic

    cfg, build, model, _ = _variant(variant)
    model = model if mj_model is None else mj_model
    cfg_kwargs.setdefault("max_episode_steps", int(cfg.max_episode_steps))
    env_cfg = mimic._make_mimic_env_cfg(
        _task_id=f"test-mimic-{variant}",
        entity_name=_ENTITY[variant],
        variant=variant,
        spec_fn=lambda: build(cfg)[0],
        muscle_actuators=mimic._muscle_actuator_names(model),
        tendon_targets=mimic._muscle_tendon_names(model),
        sim_dt=float(cfg.sim_dt),
        ctrl_dt=float(cfg.ctrl_dt),
        clip=clip,
        num_envs=num_envs,
        mj_model=model,
        **cfg_kwargs,
    )
    return ManagerBasedRlEnv(cfg=env_cfg, device="cpu")


def _synergy_model(n_muscles: int, n_syn: int = 4) -> Any:
    """Minimal synergy model with the fields ``SARTorchTransform`` reads."""
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        n_muscles=n_muscles,
        pca=SimpleNamespace(
            components_=rng.standard_normal((n_syn, n_muscles)),
            mean_=rng.random(n_muscles),
        ),
        ica=SimpleNamespace(
            mixing_=rng.standard_normal((n_syn, n_syn)), mean_=rng.random(n_syn)
        ),
        scaler=SimpleNamespace(scale_=np.ones(n_syn), min_=np.zeros(n_syn)),
    )


def _registered_cfgs(register: Callable[..., None], **kwargs: Any) -> dict[str, Any]:
    """Run a registration function and capture ``task_id -> registration kwargs``."""
    registered: dict[str, Any] = {}

    def _capture(**task: Any) -> None:
        registered[task["task_id"]] = task

    register(register_mjlab_task=_capture, **kwargs)
    return registered


def _register_full_body_sar(clip: MotionClip | None) -> dict[str, Any]:
    """Registration kwargs of ``myoMimicFullbody-SAR-v0``."""
    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic

    n_muscles = len(mimic._muscle_actuator_names(_variant("fullbody")[2]))
    return _registered_cfgs(
        mimic._register_mimic_sar_tasks,
        rl_cfg_fn=mimic.default_mimic_clip_on_policy_runner_cfg,
        sar_model=_synergy_model(n_muscles),
        clip=clip,
    )["myoMimicFullbody-SAR-v0"]


def _clip_frames(env: Any, clip: MotionClip) -> list[int]:
    """Clip frame of every env, read back from the ``clip_phase`` observation."""
    phase = env.observation_manager.get_term_cfg("policy", "clip_phase").func(env)
    n_frames = int(clip.site_xpos.shape[0])
    return torch.round(phase.reshape(-1) * n_frames).long().tolist()


@pytest.fixture(scope="module")
def fullbody_env() -> Iterator[tuple[Any, MotionClip]]:
    clip = _synthetic_clip("fullbody", joint_pos=_FULLBODY_PIKE)
    env = _make_env("fullbody", clip)
    yield env, clip
    env.close()


@pytest.fixture(scope="module")
def bimanual_env() -> Iterator[tuple[Any, MotionClip]]:
    clip = _synthetic_clip("bimanual")
    env = _make_env("bimanual", clip)
    yield env, clip
    env.close()


@pytest.fixture(scope="module")
def bimanual_long_env() -> Iterator[tuple[Any, MotionClip]]:
    """Bimanual env whose episodes outlast the longest test rollout."""
    clip = _synthetic_clip("bimanual", n_frames=1500)
    env = _make_env(
        "bimanual", clip, use_early_termination=False, max_episode_steps=1100
    )
    yield env, clip
    env.close()


# ---------------------------------------------------------------------------
# Reference state initialisation
# ---------------------------------------------------------------------------


def test_rsi_writes_reference_world_root_velocity(
    fullbody_env: tuple[Any, MotionClip],
) -> None:
    """RSI must start the root with the clip's world-frame velocity.

    Clip qvel holds the free-joint angular velocity in the body frame, while
    ``write_root_state_to_sim`` expects it in the world frame.
    """
    env, clip = fullbody_env
    env.reset()
    _, _, model, _ = _variant("fullbody")
    data = mujoco.MjData(model)
    expected = []
    for frame in _clip_frames(env, clip):
        data.qpos[:] = clip.qpos[frame]
        data.qvel[:] = clip.qvel[frame]
        mujoco.mj_forward(model, data)
        vel = np.zeros(6)  # [angular, linear] at the body origin, world frame
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_XBODY, int(model.jnt_bodyid[0]), vel, 0
        )
        expected.append(np.concatenate([vel[3:], vel[:3]]))
    root_vel_w = env.scene[_ENTITY["fullbody"]].data.root_link_vel_w
    np.testing.assert_allclose(root_vel_w.cpu().numpy(), np.stack(expected), atol=1e-4)


def test_entity_default_state_is_the_model_keyframe() -> None:
    """The entity's default state (reset without RSI) is the model keyframe.

    mjlab matches ``InitialStateCfg.joint_pos`` keys as regexes, so a bare
    ``knee_angle_r`` also set ``knee_angle_rotation2_r`` & co. to its angle.
    The keyframe root angular velocity is body-frame, ``ang_vel`` world-frame;
    the keyframe gets a tilted root and a root velocity so the frames differ,
    and distinct joint values so a pattern matching the wrong joint shows.
    """
    _, _, model, _ = _variant("fullbody")
    keyed = copy.copy(model)
    rng = np.random.default_rng(0)
    mujoco.mju_mulQuat(
        keyed.key_qpos[0, 3:7],
        _axis_angle_quat((0.0, 0.0, 1.0), 0.7),
        _axis_angle_quat((1.0, 0.3, 0.0), 0.4),
    )
    keyed.key_qpos[0, 7:] += rng.normal(0.0, 0.05, keyed.nq - 7)
    keyed.key_qvel[0, :6] = _ROOT_LIN_VEL + _ROOT_ANG_VEL_BODY
    keyed.key_qvel[0, 6:] = rng.normal(0.0, 0.1, keyed.nv - 6)
    env = _make_env("fullbody", None, num_envs=1, mj_model=keyed)
    env.reset()
    data = env.scene[_ENTITY["fullbody"]].data
    key_qpos = keyed.key_qpos[0].astype(np.float32)
    key_qvel = keyed.key_qvel[0].astype(np.float32)
    np.testing.assert_array_equal(data.default_joint_pos[0].numpy(), key_qpos[7:])
    np.testing.assert_array_equal(data.default_joint_vel[0].numpy(), key_qvel[6:])
    np.testing.assert_array_equal(data.joint_pos[0].numpy(), key_qpos[7:])
    root_vel_b = torch.cat([data.root_link_lin_vel_w, data.root_link_ang_vel_b], -1)
    np.testing.assert_allclose(root_vel_b[0].numpy(), key_qvel[:6], atol=1e-5)
    env.close()


# ---------------------------------------------------------------------------
# Early termination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["fullbody", "bimanual"])
def test_perfect_tracking_does_not_terminate(
    variant: str, request: pytest.FixtureRequest
) -> None:
    """A state on the reference clip must not end the episode.

    The root error used to be measured against the centroid of the target
    sites.  Bimanual has no free joint, so its hinge angles were compared with
    a world position and every step terminated; the full-body pike pose has
    its site centroid 0.36 m from the pelvis, past the 0.3 m tolerance.
    """
    env, _ = request.getfixturevalue(f"{variant}_env")
    torch.manual_seed(0)
    env.reset()
    deviation = env.termination_manager.get_term_cfg("mimic_deviation").func
    assert not deviation(env).any()
    action = torch.zeros(env.num_envs, sum(env.action_manager.action_term_dim))
    _, _, terminated, _, _ = env.step(action)
    assert not terminated.any()


def test_root_drift_from_reference_terminates(
    fullbody_env: tuple[Any, MotionClip],
) -> None:
    """Moving the pelvis 0.5 m off the reference root ends that env only.

    The sites move by 0.5 m as well, below the 1 m site tolerance, so only
    the root check against the clip's reference root can fire.
    """
    env, _ = fullbody_env
    torch.manual_seed(0)
    env.reset()
    entity = env.scene[_ENTITY["fullbody"]]
    pose = torch.cat([entity.data.root_link_pos_w, entity.data.root_link_quat_w], -1)
    pose[0, 0] += 0.5
    entity.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    deviation = env.termination_manager.get_term_cfg("mimic_deviation").func
    assert deviation(env).tolist() == [True, False]


# ---------------------------------------------------------------------------
# Clip frame index
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_steps", [100, pytest.param(1000, marks=pytest.mark.slow)])
def test_clip_frame_follows_cpu_step_counter(
    n_steps: int, bimanual_long_env: tuple[Any, MotionClip]
) -> None:
    """After k control steps the clip frame is start + k, like the CPU twin.

    The frame is held at the last one on the step that truncates at the clip end
    (it no longer wraps to frame 0); the check stops there.

    The CPU env advances one frame per step (``_step_count``).  mjlab's old
    ``floor(float32 time / ctrl_dt)`` index lagged it on 691 of 1000 steps,
    first at step 7, repeating and then skipping frames.
    """
    env, clip = bimanual_long_env
    env.reset()
    n_frames = int(clip.site_xpos.shape[0])
    start = np.asarray(_clip_frames(env, clip))
    target_fn = env.observation_manager.get_term_cfg("policy", "mimic_site_target").func
    action = torch.zeros(env.num_envs, sum(env.action_manager.action_term_dim))
    for k in range(1, n_steps + 1):
        env.step(action)
        frames = np.minimum(start + k, n_frames - 1)
        assert _clip_frames(env, clip) == frames.tolist(), f"step {k}"
        targets = target_fn(env).reshape(env.num_envs, -1, 3).cpu().numpy()
        np.testing.assert_array_equal(targets, clip.site_xpos[frames])
        if (start + k >= n_frames - 1).any():  # a clip end truncates and resets
            break


# ---------------------------------------------------------------------------
# Host syncs per step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["fullbody", "bimanual"])
def test_clip_step_syncs_only_to_detect_resets(
    variant: str, request: pytest.FixtureRequest
) -> None:
    """A step reads no host data, except one reset check on a step that reset an env.

    Every observation, reward and termination term re-synced the clip source
    (a ``reset_mask.any()`` host read) and indexed sites with a NumPy array (a
    host-to-device copy): 34 host syncs per step.
    """
    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic
    from myosuite.tests.support.host_sync import HostSyncCounter

    env, clip = request.getfixturevalue(f"{variant}_env")
    torch.manual_seed(0)
    env.reset()
    source = mimic._resolve_mimic_mjlab_ids(env, _ENTITY[variant], variant)[
        "clip_source"
    ]
    source._start_offsets[0] = int(clip.site_xpos.shape[0]) - 2  # ends at step 2
    action = torch.zeros(env.num_envs, sum(env.action_manager.action_term_dim))
    env.step(action)  # sees the offset write: one reset check
    resets = []
    for _ in range(4):
        with HostSyncCounter(package_only=True) as syncs:
            env.step(action)
        reset = bool(env.reset_buf.any())
        resets.append(reset)
        assert syncs.total <= int(reset), syncs.report()
    assert resets[0], "env 0 should reach its clip end on the second step"


def _reference_lookahead(
    env: Any, cache: dict[str, Any], entity: str, k: int, stride: int
) -> Any:
    """The lookahead observation, one future step at a time (the original loop)."""
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        _clip_has_required_indices,
    )

    source, clip = cache["clip_source"], cache["clip"]
    step = env.episode_length_buf
    n = int(step.shape[0])
    lengths = source.clip_lengths(step)
    frames = source.frame_indices(step)
    free_root = not env.scene[entity].is_fixed_base
    data = env.scene[entity].data.data
    root = data.qpos[:, :3] if free_root else torch.zeros(n, 3)
    has_pos = free_root and _clip_has_required_indices(
        clip.qpos_model_indices, range(3)
    )
    has_vel = free_root and _clip_has_required_indices(
        clip.qvel_model_indices, range(3)
    )
    width = source.n_tracked * 3 + 3 * has_pos + 3 * has_vel + 1
    out = torch.zeros(n, k * width)
    offset = 0
    for i in range(1, k + 1):
        future = (frames + i * stride) % lengths
        sites = (source.site_targets_at_frames(future) - root[:, None]).reshape(n, -1)
        out[:, offset : offset + sites.shape[1]] = sites
        offset += sites.shape[1]
        if has_pos:
            out[:, offset : offset + 3] = (
                source.ref_qpos_at_frames(future)[:, :3] - root
            )
            offset += 3
        if has_vel:
            out[:, offset : offset + 3] = source.ref_qvel_at_frames(future)[:, :3]
            offset += 3
        out[:, offset] = future.float() / torch.clamp(lengths.float() - 1.0, min=1.0)
        offset += 1
    return out


@pytest.mark.parametrize("variant", ["fullbody", "bimanual"])
def test_lookahead_matches_the_reference_loop(
    variant: str, request: pytest.FixtureRequest
) -> None:
    """The batched lookahead (its clip part shared by the obs groups) is bit-exact."""
    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic

    env, clip = request.getfixturevalue(f"{variant}_env")
    torch.manual_seed(1)
    env.reset()
    action = torch.zeros(env.num_envs, sum(env.action_manager.action_term_dim))
    for _ in range(2):
        env.step(action)
    entity = _ENTITY[variant]
    for k, stride in ((2, 3), (5, 20)):
        fn = mimic._mimic_obs_lookahead(
            entity, variant, clip, float(env.step_dt), k, stride
        )
        out = fn(env)
        cache = mimic._resolve_mimic_mjlab_ids(env, entity, variant)
        assert torch.equal(out, _reference_lookahead(env, cache, entity, k, stride))
        assert torch.equal(fn(env), out)  # the shared clip part is reused


# ---------------------------------------------------------------------------
# SAR tasks
# ---------------------------------------------------------------------------


def _runner_obs_groups(rl_cfg: Any) -> dict[str, list[str]]:
    return {name: list(groups) for name, groups in rl_cfg.obs_groups.items()}


def test_sar_mimic_task_resets_standing_and_has_runner_obs_groups() -> None:
    """``myoMimicFullbody-SAR-v0`` starts at the keyframe and trains with rsl_rl.

    Its spec strips keyframes and the cfg set no initial state, so the pelvis
    started at z = 0 with the legs about 1 m through the floor.  It also had
    only a ``policy`` group, while the runner cfg asks for ``actor``/``critic``.
    """
    from mjlab.envs import ManagerBasedRlEnv

    rsl_rl_utils = pytest.importorskip("rsl_rl.utils.utils")
    task = _register_full_body_sar(clip=None)
    env_cfg = task["play_env_cfg"]
    env_cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    try:
        obs, _ = env.reset()
        pelvis_z = env.scene[_ENTITY["fullbody"]].data.root_link_pos_w[:, 2]
        key_z = float(_variant("fullbody")[2].key_qpos[0][2])
        np.testing.assert_allclose(pelvis_z.cpu().numpy(), key_z, atol=1e-5)
        rsl_rl_utils.resolve_obs_groups(
            obs, _runner_obs_groups(task["rl_cfg"]), default_sets=["actor", "critic"]
        )
    finally:
        env.close()


def test_sar_mimic_cfg_is_the_mimic_task_in_synergy_space() -> None:
    """With a clip the SAR cfg equals ``myoMimicFullbody-v0`` but for its action.

    It used to drop RSI, early termination, the DeepMimic reward and its
    ``1 / ctrl_dt`` weight compensation, and the lookahead observation.
    """
    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic

    clip = _synthetic_clip("fullbody", n_frames=30)
    sar = _register_full_body_sar(clip)
    # The weight register_mimic_mjlab_tasks_with_clip passes by default.
    muscle = _registered_cfgs(
        mimic._register_mimic_tasks,
        rl_cfg_fn=lambda: None,
        clip=clip,
        mimic_reward_weight=5.0,
    )["myoMimicFullbody-v0"]
    for key in ("env_cfg", "play_env_cfg"):
        sar_cfg, muscle_cfg = sar[key], muscle[key]
        assert sar_cfg.scene.num_envs == muscle_cfg.scene.num_envs
        entity = _ENTITY["fullbody"]
        assert (
            sar_cfg.scene.entities[entity].init_state
            == muscle_cfg.scene.entities[entity].init_state
        )
        assert {g: list(c.terms) for g, c in sar_cfg.observations.items()} == {
            g: list(c.terms) for g, c in muscle_cfg.observations.items()
        }
        assert list(sar_cfg.events) == list(muscle_cfg.events) == ["rsi"]
        assert list(sar_cfg.terminations) == list(muscle_cfg.terminations)
        assert {
            n: (t.func.__qualname__, t.weight) for n, t in sar_cfg.rewards.items()
        } == {n: (t.func.__qualname__, t.weight) for n, t in muscle_cfg.rewards.items()}
        assert isinstance(
            sar_cfg.actions["muscles"], mimic.SARMuscleActivationActionCfg
        )


def test_directional_sar_task_has_runner_obs_groups(tmp_path: Any) -> None:
    """``myoFullBodyWalkSAR-v0`` defines the groups the default runner cfg uses."""
    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic

    pytest.importorskip("sklearn")
    from myosuite.integrations.musclemimic.sar_extraction import (
        extract_synergies,
        save_synergy_model,
    )

    n_muscles = len(mimic._muscle_actuator_names(_variant("fullbody")[2]))
    activations = np.random.default_rng(0).random((200, n_muscles))
    save_synergy_model(extract_synergies(activations, n_synergies=4), tmp_path)
    task = _registered_cfgs(
        mimic.register_directional_walk_sar,
        rl_cfg_fn=mimic.default_mimic_clip_on_policy_runner_cfg,
        sar_dir=tmp_path,
    )["myoFullBodyWalkSAR-v0"]
    groups = _runner_obs_groups(task["rl_cfg"])
    assert {g for gs in groups.values() for g in gs} <= set(
        task["env_cfg"].observations
    )


# ---------------------------------------------------------------------------
# Reward modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [{"reward_mode": "env"}, {"reward_mode": "augmented"}, {"env_reward_weight": 2.0}],
)
def test_clip_registration_rejects_reward_options_it_cannot_honour(
    options: dict[str, Any],
) -> None:
    """The Mimic tasks have only the clip-tracking objective.

    ``reward_mode="env"``/``"augmented"`` and ``env_reward_weight`` were
    accepted and silently ignored: every mode registered the same
    clip-tracking reward and termination.
    """
    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic

    registered: list[str] = []
    with pytest.raises(NotImplementedError, match="native task reward"):
        mimic.register_mimic_mjlab_tasks_with_clip(
            register_mjlab_task=lambda **task: registered.append(task["task_id"]),
            rl_cfg_fn=lambda: None,
            clip=_synthetic_clip("fullbody", n_frames=10),
            **options,
        )
    assert registered == []


def test_reward_mode_reaches_the_env_cfg_builder() -> None:
    """``reward_mode`` is threaded through the registration to the cfg."""
    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic

    clip = _synthetic_clip("fullbody", n_frames=10)
    cfg = _registered_cfgs(
        mimic._register_mimic_tasks,
        rl_cfg_fn=lambda: None,
        clip=clip,
        reward_mode="MIMIC",
    )["myoMimicFullbody-v0"]["env_cfg"]
    assert list(cfg.rewards) == ["tracking"]
    assert list(cfg.terminations) == [
        "sync_forward",
        "time_out",
        "clip_end",
        "mimic_deviation",
    ]
    with pytest.raises(NotImplementedError, match="native task reward"):
        _registered_cfgs(
            mimic._register_mimic_tasks,
            rl_cfg_fn=lambda: None,
            clip=clip,
            reward_mode="env",
        )


# ---------------------------------------------------------------------------
# Root assumptions: a fixed base (bimanual) has no root, qpos[:7] are hinge angles
# ---------------------------------------------------------------------------


def _lookahead(env: Any, clip: MotionClip, variant: str) -> Any:
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import _mimic_obs_lookahead

    fn = _mimic_obs_lookahead(
        _ENTITY[variant], variant, clip, float(env.step_dt), k=2, stride=3
    )
    return fn(env)


def test_bimanual_lookahead_has_no_root_terms(
    bimanual_env: tuple[Any, MotionClip],
) -> None:
    env, clip = bimanual_env
    env.reset()
    n_sites = len(clip.site_names or [])
    out = _lookahead(env, clip, "bimanual")
    assert out.shape[-1] == 2 * (n_sites * 3 + 1)  # sites + phase per step
    assert env.scene[_ENTITY["bimanual"]].is_fixed_base


def test_fullbody_lookahead_keeps_root_terms(
    fullbody_env: tuple[Any, MotionClip],
) -> None:
    env, clip = fullbody_env
    env.reset()
    n_sites = len(clip.site_names or [])
    out = _lookahead(env, clip, "fullbody")
    assert out.shape[-1] == 2 * (n_sites * 3 + 3 + 3 + 1)


@pytest.mark.parametrize(
    ("variant", "has_root"), [("bimanual", False), ("fullbody", True)]
)
def test_deepmimic_reward_root_terms_only_for_free_root(
    variant: str,
    has_root: bool,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import myosuite.terms.mimic_reward as mimic_reward
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        _mimic_deepmimic_reward,
    )

    env, clip = request.getfixturevalue(f"{variant}_env")
    env.reset()
    seen: dict[str, Any] = {}
    real = mimic_reward.mimic_composite_reward

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen["root"] = args[7:13]  # cur/ref root pos, vel, quat
        return real(*args, **kwargs)

    monkeypatch.setattr(mimic_reward, "mimic_composite_reward", spy)
    _mimic_deepmimic_reward(_ENTITY[variant], variant, clip, float(env.step_dt))(env)
    assert all(arg is not None for arg in seen["root"]) == has_root
    assert all(arg is None for arg in seen["root"]) == (not has_root)


def test_fullbody_viewer_follows_the_pelvis_and_keeps_mjlab_defaults_elsewhere() -> (
    None
):
    """The full body gets the tracking camera; the bimanual env keeps mjlab's default."""
    from mjlab.viewer import ViewerConfig

    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic

    full = _make_env("fullbody", _synthetic_clip("fullbody"), num_envs=1)
    bimanual = _make_env("bimanual", _synthetic_clip("bimanual"), num_envs=1)
    try:
        viewer = full.cfg.viewer
        assert viewer.origin_type == ViewerConfig.OriginType.ASSET_BODY
        assert (viewer.entity_name, viewer.body_name) == (_ENTITY["fullbody"], "pelvis")
        # The image size stays at mjlab's default, so existing renders keep their shape.
        default = ViewerConfig()
        assert (viewer.width, viewer.height) == (default.width, default.height)
        assert bimanual.cfg.viewer.origin_type == default.origin_type
    finally:
        full.close()
        bimanual.close()
    custom = mimic.mimic_viewer_cfg("robot", body_name="head", width=1280, azimuth=10.0)
    assert (custom.body_name, custom.width, custom.azimuth) == ("head", 1280, 10.0)


def test_clip_bank_partial_reset_uses_the_clips_of_the_reset_envs() -> None:
    """Resetting some envs of a multi-clip bank gathers their reference from their own clips."""
    clips = (
        _synthetic_clip("fullbody", n_frames=60),
        _synthetic_clip("fullbody", n_frames=90),
        _synthetic_clip("fullbody", n_frames=75),
    )
    env = _make_env("fullbody", clips, num_envs=6)
    try:
        env.reset()
        action = torch.zeros(env.num_envs, sum(env.action_manager.action_term_dim))
        for ids in ([1, 4], [0], [2, 3, 5], [0, 1, 2, 3, 4, 5]):
            env.step(action)
            env.reset(env_ids=torch.tensor(ids, device=env.device))
        assert env.observation_manager.compute_group("actor").shape[0] == env.num_envs
    finally:
        env.close()


def _assert_envs_read_their_clips(
    env: Any, entity: str, reset_ids: tuple[int, ...] = ()
) -> None:
    """Targets and reference of every env are its own clip's frame (a NumPy gather).

    *reset_ids*: envs just reset by RSI, whose joints must be on that frame.
    """
    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic

    cache = mimic._resolve_mimic_mjlab_ids(env, entity, "fullbody")
    source = cache["clip_source"]
    rows = []
    for step, clip_id, offset in zip(
        env.episode_length_buf.tolist(),
        source._clip_indices.tolist(),
        source._start_offsets.tolist(),
    ):
        clip = source.clips[clip_id]
        rows.append((clip, min(step + offset, int(clip.site_xpos.shape[0]) - 1)))
    sites = np.stack([c.site_xpos[f][source.tracked_site_ids] for c, f in rows])
    np.testing.assert_array_equal(
        cache["target_torch"].numpy(), sites.astype(np.float32)
    )
    for name in ("qpos", "qvel"):
        expected = np.stack([getattr(c, name)[f] for c, f in rows])
        got = mimic._clip_ref(env, cache, f"ref_{name}").numpy()
        np.testing.assert_array_equal(got, expected.astype(np.float32))
    for i in reset_ids:
        clip, frame = rows[i]
        joints = env.scene[entity].data.data.qpos[i, 7:].numpy()
        np.testing.assert_array_equal(joints, clip.qpos[frame, 7:].astype(np.float32))


def test_clip_bank_steps_are_sync_free_and_read_each_envs_clip() -> None:
    """A clip-bank step makes no host sync (one reset check on a step that resets an
    env), and every env reads its own clip, after partial resets inside and outside
    ``env.step``.

    The bank gather looped over the clips with ``mask.any()`` and bool-mask
    indexing: three host syncs per clip for each of the step's gathers.
    """
    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic
    from myosuite.tests.support.host_sync import HostSyncCounter

    clips = (
        _synthetic_clip("fullbody", n_frames=60),
        _synthetic_clip("fullbody", n_frames=90, joint_pos=_FULLBODY_PIKE),
    )
    entity = _ENTITY["fullbody"]
    env = _make_env("fullbody", clips, num_envs=6)
    try:
        torch.manual_seed(0)
        env.reset()
        source = mimic._resolve_mimic_mjlab_ids(env, entity, "fullbody")["clip_source"]
        action = torch.zeros(env.num_envs, sum(env.action_manager.action_term_dim))
        env.step(action)  # uploads the lazily built device constants once
        for ids, ending in (((1, 4), 0), ((0, 2, 3), 5)):
            # Env *ending* reaches its clip end on the second step: a reset in env.step.
            step = env.episode_length_buf
            source._start_offsets[ending] = (
                source.clip_lengths(step)[ending] - step[ending] - 2
            )
            env.reset(env_ids=torch.tensor(ids, device=env.device))
            _assert_envs_read_their_clips(env, entity, ids)
            resets = []
            for _ in range(2):
                with HostSyncCounter(package_only=True) as syncs:
                    env.step(action)
                resets.append(bool(env.reset_buf.any()))
                assert syncs.total <= int(resets[-1]), syncs.report()
                _assert_envs_read_their_clips(env, entity)
            assert resets[-1] and env.episode_length_buf[ending] == 0
    finally:
        env.close()
