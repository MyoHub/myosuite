# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Register MyoSuite tasks with mjlab's task registry so make_env(..., backend="mjlab") works.

When mjlab loads this package via the mjlab.tasks entry point, this module is not
auto-imported; the entry point targets myosuite.envs.myo.backends.mjlab (the parent __init__.py).
We call :func:`bootstrap_myosuite_mjlab_registry` from there so env ids like
``myoElbowPose1D6MFixed-v0`` appear in ``list_tasks()`` and can be created via
``load_env_cfg`` + ``ManagerBasedRlEnv``.  Use the same bootstrap from notebooks
for a single idempotent entry point (optional clip via ``MYOSUITE_MIMIC_CLIP`` /
``MIMIC_CLIP``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mjlab.actuator import XmlActuatorCfg as _XmlWrappedActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import events as mdp_events
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task

from myosuite.core.config import TaskConfig
from myosuite.envs.myo.assets._resolve import resolve_elbow_xml as _resolve_elbow_xml
from myosuite.utils.asset_path_resolver import resolve_model_xml_path
from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
    MyoMuscleActivationActionCfg,
    mjlab_env_cfg_from_task_config,
)
from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
    default_mimic_clip_on_policy_runner_cfg,
)
from myosuite.envs.myo.backends.mjlab.register_mjlab_tabletennis import (
    register_table_tennis_mjlab_tasks,
)

if TYPE_CHECKING:  # pragma: no cover
    import torch


# Resolve model paths — pip package first, submodule fallback.
def _resolve_model_root() -> Path:
    try:
        from etils import epath

        return Path(epath.resource_path("myosuite"))
    except (ImportError, ModuleNotFoundError):
        return Path(__file__).resolve().parents[3]


_MYOSUITE_ROOT = _resolve_model_root()
_ELBOW_XML = resolve_model_xml_path(_resolve_elbow_xml("myoelbow_1dof6muscles.xml"))


def _elbow_tendon_names() -> tuple[str, ...]:
    """Return elbow *muscle* tendon names (one per actuator).

    The packaged elbow MJCF may add passive spatial tendons (e.g. ``error``)
    that are not driven by actuators. ``TendonLengthActionCfg`` must target
    only the ``nu`` muscle tendons so action shape matches Gymnasium CPU.
    """
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(_ELBOW_XML))
    names: list[str] = []
    for i in range(m.nu):
        tid = int(m.actuator_trnid[i, 0])
        if tid < 0:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, tid)
        if name is not None:
            names.append(name)
    return tuple(names)


_ELBOW_ENTITY_NAME = "elbow"
_ELBOW_FIXED_TARGET_RAD = 2.0  # r_elbow_flex fixed target (myoElbowPose1D6MFixed-v0)

# ---------------------------------------------------------------------------
# Elbow obs functions — matching PoseEnvV0 (CPU) obs contract:
#   qpos      (1)  ← accessor.joint_pos()          = r_elbow_flex angle
#   qvel      (1)  ← accessor.joint_vel() * dt      = angle velocity × ctrl_dt
#   pose_err  (1)  ← target - qpos                  = 2.0 - angle (fixed variant)
#   act       (6)  ← mj_data.act                    = muscle activations
# Total: 9-dim, matching CPU env observation_space.shape = (9,)
# ---------------------------------------------------------------------------


def _elbow_obs_qpos(env) -> torch.Tensor:
    """Elbow hinge joint angle. Shape: (N, 1)."""
    return env.scene[_ELBOW_ENTITY_NAME].data.joint_pos[:, 0:1]


def _elbow_obs_qvel(env) -> torch.Tensor:
    """Elbow hinge joint velocity × ctrl_dt. Shape: (N, 1)."""
    ctrl_dt = env.physics_dt * env.cfg.decimation
    return env.scene[_ELBOW_ENTITY_NAME].data.joint_vel[:, 0:1] * ctrl_dt


def _elbow_obs_pose_err(env) -> torch.Tensor:
    """Fixed target minus current angle. Shape: (N, 1).

    Matches PoseEnvV0 pose_error_obs(accessor, target_jnt_value=[2.0]).
    """
    return (
        _ELBOW_FIXED_TARGET_RAD - env.scene[_ELBOW_ENTITY_NAME].data.joint_pos[:, 0:1]
    )


def _elbow_obs_act(env) -> torch.Tensor:
    """Muscle activation state. Shape: (N, 6)."""
    # accepted: no entity.data API for muscle activation — entity.data.data.act
    return env.scene[_ELBOW_ENTITY_NAME].data.data.act


def _elbow_spec_fn():
    import mujoco

    return mujoco.MjSpec.from_file(str(_ELBOW_XML))


def _actor_critic_groups(group: ObservationGroupCfg) -> dict[str, ObservationGroupCfg]:
    """The ``actor`` / ``critic`` observation groups the MyoSuite PPO configs expect."""
    import copy  # noqa: PLC0415

    return {"actor": group, "critic": copy.deepcopy(group)}


def _make_elbow_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Minimal ManagerBasedRlEnvCfg for myoElbowPose1D6MFixed-v0 (1 env, CPU/GPU)."""
    if not _ELBOW_XML.exists():
        raise FileNotFoundError(f"Elbow model not found: {_ELBOW_XML}")
    cfg = TaskConfig(max_episode_steps=200)
    tendon_names = _elbow_tendon_names()
    # Muscle names = tendon names without the "_tendon" suffix, matching
    # MyoMuscleActivationAction.find_actuators() lookup convention.
    muscle_names = tuple(n.replace("_tendon", "") for n in tendon_names)

    # Obs contract: [qpos(1), qvel(1), pose_err(1), act(6)] = 9D
    # Matches myoElbowPose1D6MFixed-v0 CPU observation_space.shape=(9,)
    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "qpos": ObservationTermCfg(func=_elbow_obs_qpos),
                "qvel": ObservationTermCfg(func=_elbow_obs_qvel),
                "pose_err": ObservationTermCfg(func=_elbow_obs_pose_err),
                "act": ObservationTermCfg(func=_elbow_obs_act),
            },
        ),
    }
    # Sigmoid activation matching CPU PoseEnvV0(normalize_act=True)
    actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=_ELBOW_ENTITY_NAME,
            actuator_names=muscle_names,
        ),
    }
    return mjlab_env_cfg_from_task_config(
        cfg=cfg,
        spec_fn=_elbow_spec_fn,
        entity_name=_ELBOW_ENTITY_NAME,
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=tendon_names,
                transmission_type=TransmissionType.TENDON,
            ),
        ),
        observations=observations,
        actions=actions,
        num_envs=1,
        decimation=10,
    )


def _elbow_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Minimal PPO runner config for benchmarking."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(64, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(64, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg=None,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=2,
            num_mini_batches=1,
            learning_rate=3e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="myo_elbow",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=100,
        # Map actor/critic to the env's "policy" group.
        obs_groups={"actor": ("policy",), "critic": ("policy",)},
    )


# ---------------------------------------------------------------------------
# ChaseTag full-body vs. scripted-opponent (GPU match for myoChallengeChaseTagFBP2-v0)
# ---------------------------------------------------------------------------
# Reuses the full-body + mocap-opponent MjSpec (build_fullbody_chasetag_spec)
# as a single mjlab Entity spanning both the agent's kinematic tree and the
# scripted "opponent" mocap body — mirroring how the CPU ChaseTagEnv treats
# both as one MjModel/MjData. Obs order matches
# myosuite/envs/myo/tasks/mimic/chasetag_obs.py::chasetag_obs's 537-dim
# additive composition exactly (qpos_local(82) + qvel_local(82) + act(354) +
# root_vel_body(2) + heading_cmd(2, fixed [1,0] — no external heading command
# in chase-tag) + orientation(6) + opponent_relative(7) + role(2)) so the
# pretrained bc_directional_v2 checkpoint's first 528 input columns warm-start
# meaningfully via ActorCritic.load_expanded.

_CHASETAG_ENTITY_NAME = "chasetag_agent"
# Matches ChaseTagEnv._get_fallen_condition's FLAT-terrain pelvis-height check.
_CHASETAG_FALL_HEIGHT = 0.5
# Matches ChaseTagEnv.__init__'s win_distance / chase_vel_range defaults.
_CHASETAG_WIN_DISTANCE = 0.5
_CHASETAG_CHASE_VEL_RANGE = (1.0, 1.0)
_CHASETAG_MIN_SPAWN_DISTANCE = 2.0
_CHASETAG_ARENA_BOUND = 5.5  # matches ChallengeOpponent.move_opponent's clip range
# ChallengeOpponent.reset_opponent: player_task="CHASE" -> sample_opponent_policy()
# picks among these three (opponent_probabilities=(0.1, 0.45, 0.45) default) --
# "chase_player" is a DIFFERENT opponent policy, only ever selected for
# player_task="EVADE" (where the opponent hunts the agent and the agent's job
# is to evade). FBP2 is CHASE-only, so its opponent must be one of these three,
# never chase_player -- confirmed as a real bug in an earlier version of this
# file, which ported chase_player unconditionally (see git history).
_CHASETAG_OPPONENT_PROBABILITIES = (
    0.1,
    0.45,
    0.45,
)  # static_stationary, stationary, random
_CHASETAG_RANDOM_VEL_RANGE = (-2.0, 2.0)  # ChallengeOpponent's default random_vel_range
_CHASETAG_STATIC_STATIONARY_POSE = (
    0.0,
    -5.0,
    0.0,
)  # ChallengeOpponent.reset_opponent's fixed spot


def _chasetag_spec_fn():
    """Return the full-body + mocap-opponent ``MjSpec`` (no source keyframes)."""
    from myosuite.envs.myo.tasks.challenge.chase_tag_fb_model import (  # noqa: PLC0415
        build_fullbody_chasetag_spec,
    )

    return build_fullbody_chasetag_spec()


def _chasetag_muscle_and_tendon_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (muscle_actuator_names, tendon_target_names) for the full-body model."""
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (  # noqa: PLC0415
        _muscle_actuator_names,
        _muscle_tendon_names,
    )

    mj_model = _chasetag_spec_fn().compile()
    return _muscle_actuator_names(mj_model), _muscle_tendon_names(mj_model)


def _chasetag_init_state():
    """Standing ``InitialStateCfg`` for the full-body chase-tag agent.

    Reuses the same keyframe-extraction helper the full-body Mimic mjlab
    tasks use (``_init_state_from_model``) — the full-body model ships no
    source keyframe, so this returns ``EntityCfg.InitialStateCfg()`` (mjlab's
    own zero default) when ``nkey == 0``, matching Mimic full-body's own
    fallback behavior for this exact host model.
    """
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (  # noqa: PLC0415
        _init_state_from_model,
    )

    mj_model = _chasetag_spec_fn().compile()
    return _init_state_from_model(mj_model)


# ── Obs terms: byte-identical 528-dim directional prefix (torch, batched) ──


def _chasetag_obs_qpos_local(env):
    """``qpos[7:]`` — matches chasetag_obs's ``qpos_local`` block. Shape (N, nq-7)."""
    return env.scene[_CHASETAG_ENTITY_NAME].data.data.qpos[:, 7:]


def _chasetag_obs_qvel_local(env):
    """``qvel[6:]`` — matches chasetag_obs's ``qvel_local`` block. Shape (N, nv-6)."""
    return env.scene[_CHASETAG_ENTITY_NAME].data.data.qvel[:, 6:]


def _chasetag_obs_act(env):
    """Muscle activation state. Shape (N, 354)."""
    return env.scene[_CHASETAG_ENTITY_NAME].data.data.act


def _chasetag_yaw(env):
    """Pelvis yaw from the free-joint quaternion. Shape (N,)."""
    import torch  # noqa: PLC0415

    qpos = env.scene[_CHASETAG_ENTITY_NAME].data.data.qpos
    w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _chasetag_obs_root_vel_body(env):
    """Root planar velocity rotated into the pelvis frame. Shape (N, 2).

    Matches ``_pelvis_yaw_from_qpos`` + the ``root_vel_body`` rotation in
    ``bc_directional_collector._directional_obs``.
    """
    import torch  # noqa: PLC0415

    qvel = env.scene[_CHASETAG_ENTITY_NAME].data.data.qvel
    yaw = _chasetag_yaw(env)
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    vx, vy = qvel[:, 0], qvel[:, 1]
    return torch.stack([c * vx - s * vy, s * vx + c * vy], dim=1)


def _chasetag_obs_heading_cmd(env):
    """Fixed heading command ``[1, 0]`` (chase-tag has no external heading). Shape (N, 2)."""
    import torch  # noqa: PLC0415

    qpos = env.scene[_CHASETAG_ENTITY_NAME].data.data.qpos
    n = qpos.shape[0]
    out = torch.zeros(n, 2, dtype=torch.float32, device=qpos.device)
    out[:, 0] = 1.0
    return out


def _chasetag_obs_orientation(env):
    """``[roll, pitch, wx_b, wy_b, wz_w, vz]`` — matches ``_directional_obs``'s
    ``orientation`` block. Shape (N, 6)."""
    import torch  # noqa: PLC0415

    data = env.scene[_CHASETAG_ENTITY_NAME].data.data
    qpos, qvel = data.qpos, data.qvel
    w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = _chasetag_yaw(env)
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    wx_w, wy_w, wz_w, vz = qvel[:, 3], qvel[:, 4], qvel[:, 5], qvel[:, 2]
    wx_b = c * wx_w - s * wy_w
    wy_b = s * wx_w + c * wy_w
    return torch.stack([roll, pitch, wx_b, wy_b, wz_w, vz], dim=1)


# ── Opponent-relative obs + scripted-opponent (CHASE-only) motion state ────


def _chasetag_opponent_state(env):
    """Lazily create / return the per-env scripted-opponent state buffers.

    Mirrors ``_directional_cmd_buffer``'s lazy-buffer pattern. Holds the
    mocap "opponent" body's control-space pose ``[x, y, theta]`` (N, 3), its
    ``[lin_vel, rot_vel]`` control pair (N, 2) — the same pair
    ``ChallengeOpponent.move_opponent`` stores as ``opponent_vel`` and that
    ``ChaseTagEnv`` feeds into ``relative_pose_obs`` as a 3-D "velocity" — and
    each env's sampled constant chase speed (N,).
    """
    import torch  # noqa: PLC0415

    data = env.scene[_CHASETAG_ENTITY_NAME].data.data
    n = data.qpos.shape[0]
    state = getattr(env, "_chasetag_opponent", None)
    if state is None or state["pose"].shape[0] != n:
        device = data.qpos.device
        state = {
            "pose": torch.zeros(n, 3, dtype=torch.float32, device=device),
            "vel": torch.zeros(n, 2, dtype=torch.float32, device=device),
            "chase_speed": torch.ones(n, dtype=torch.float32, device=device),
            # 0 = static_stationary, 1 = stationary, 2 = random -- sampled
            # per-episode at reset, matching ChallengeOpponent.sample_opponent_policy.
            "policy": torch.ones(n, dtype=torch.long, device=device),
        }
        env._chasetag_opponent = state  # noqa: SLF001
    return state


def _chasetag_write_mocap_pose(env, pose: torch.Tensor) -> None:
    """Write ``[x, y, theta]`` (N, 3) into the scene's mocap_pos/mocap_quat.

    The "opponent" mocap body is grafted onto the *same* MjSpec as the
    full-body agent (single composite entity, see
    ``build_fullbody_chasetag_spec``), not a separate mjlab mocap Entity —
    so there is no ``Entity.write_mocap_pose_to_sim`` target (that API only
    applies when an entity's own root body is the mocap body, see
    ``mjlab.entity.Entity.is_mocap``). Writing ``data.data.mocap_pos`` /
    ``mocap_quat`` directly is therefore a documented, justified exception to
    the "never write wp_data directly" rule (alongside the already-accepted
    ``data.data.act`` reads elsewhere in this file) — there is exactly one
    mocap body in the whole scene (mocap index 0).
    """
    import torch  # noqa: PLC0415

    data = env.scene[_CHASETAG_ENTITY_NAME].data.data
    theta = pose[:, 2]
    half = theta * 0.5
    data.mocap_pos[:, 0, 0] = pose[:, 0]
    data.mocap_pos[:, 0, 1] = pose[:, 1]
    data.mocap_pos[:, 0, 2] = 0.0
    data.mocap_quat[:, 0, 0] = torch.cos(half)
    data.mocap_quat[:, 0, 1] = 0.0
    data.mocap_quat[:, 0, 2] = 0.0
    data.mocap_quat[:, 0, 3] = torch.sin(half)


def _chasetag_reset_opponent(env, env_ids=None, **_):
    """Reset event: place the opponent >= min_spawn_distance from the agent.

    Vectorized port of ``ChallengeOpponent.reset_opponent`` for
    ``task_choice="CHASE"`` only (this pass's scope, per the plan). Samples
    each env's opponent policy (static_stationary / stationary / random,
    matching ``sample_opponent_policy``'s probabilities) and, for
    static_stationary, teleports to the fixed spot the CPU env also uses.
    Uses a bounded number of vectorized resample rounds (not a per-env
    Python loop) to satisfy the minimum-spawn-distance constraint.
    """
    import math  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from myosuite.envs.myo.backends.mjlab.mjlab_env_base import (  # noqa: PLC0415
        normalize_mjlab_env_ids,
    )

    state = _chasetag_opponent_state(env)
    idx = normalize_mjlab_env_ids(env, env_ids)
    if idx.numel() == 0:
        return
    data = env.scene[_CHASETAG_ENTITY_NAME].data.data
    device = data.qpos.device
    agent_xy = data.qpos[idx, :2]

    pose = torch.empty(idx.numel(), 3, dtype=torch.float32, device=device)
    pose[:, 0].uniform_(-5.0, 5.0)
    pose[:, 1].uniform_(-5.0, 5.0)
    pose[:, 2].uniform_(-2.0 * math.pi, 2.0 * math.pi)
    for _attempt in range(8):  # bounded vectorized resample, no per-env loop
        dist = torch.linalg.norm(pose[:, :2] - agent_xy, dim=1)
        bad = dist < _CHASETAG_MIN_SPAWN_DISTANCE
        if not bool(bad.any()):
            break
        n_bad = int(bad.sum())
        pose[bad, 0] = torch.empty(n_bad, device=device).uniform_(-5.0, 5.0)
        pose[bad, 1] = torch.empty(n_bad, device=device).uniform_(-5.0, 5.0)

    # Sample opponent policy per env: 0=static_stationary, 1=stationary,
    # 2=random, matching sample_opponent_policy's probability thresholds.
    r = torch.rand(idx.numel(), device=device)
    p0, p1, _p2 = _CHASETAG_OPPONENT_PROBABILITIES
    policy = torch.full((idx.numel(),), 2, dtype=torch.long, device=device)
    policy[r < p0] = 0
    policy[(r >= p0) & (r < p0 + p1)] = 1
    state["policy"][idx] = policy

    # static_stationary overrides the spawn pose to a fixed spot (matches
    # ChallengeOpponent.reset_opponent: `if self.opponent_policy ==
    # "static_stationary": pose[:] = [0, -5, 0]`).
    static_mask = policy == 0
    if bool(static_mask.any()):
        fixed = torch.tensor(
            _CHASETAG_STATIC_STATIONARY_POSE, dtype=torch.float32, device=device
        )
        pose[static_mask] = fixed

    state["pose"][idx] = pose
    state["vel"][idx] = 0.0
    state["chase_speed"][idx] = torch.empty(idx.numel(), device=device).uniform_(
        *_CHASETAG_CHASE_VEL_RANGE
    )
    _chasetag_write_mocap_pose(env, state["pose"])


def _chasetag_step_opponent(env, dt: float | None = None, **_):
    """Step event (mode="step"): vectorized CHASE-task opponent motion.

    Vectorized port of ``ChallengeOpponent.update_opponent_state`` +
    ``move_opponent`` for the three policies ``sample_opponent_policy``
    actually selects under ``player_task="CHASE"`` (static_stationary /
    stationary / random -- see ``_CHASETAG_OPPONENT_PROBABILITIES``). Each
    env's opponent stays put (stationary/static_stationary) or wanders with
    a random per-step velocity clipped to ``_CHASETAG_RANDOM_VEL_RANGE``
    (an i.i.d.-per-step approximation of the CPU env's colored-noise
    ``random_movement``, not bit-identical but directionally correct: an
    opponent that wanders rather than pursues).

    NOTE: an earlier version of this function ported
    ``ChallengeOpponent.chase_player`` instead -- the opponent *hunting the
    agent*, which is only ever selected for ``player_task="EVADE"``, never
    "CHASE". That was a real, confirmed bug: it trained fbp2_ppo_v3 against
    a fundamentally wrong opponent behavior (the reward function assumes
    the agent should close in on a passive target, while the opponent was
    simultaneously closing in on the agent) -- a self-contradictory task,
    not the intended one.
    """
    import torch  # noqa: PLC0415

    state = _chasetag_opponent_state(env)
    ctrl_dt = dt if dt is not None else (env.physics_dt * env.cfg.decimation)

    pose = state["pose"]
    theta = pose[:, 2]
    n = pose.shape[0]
    device = pose.device

    lin_vel = torch.zeros(n, device=device)
    rot_vel = torch.zeros(n, device=device)
    random_mask = state["policy"] == 2
    if bool(random_mask.any()):
        lo, hi = _CHASETAG_RANDOM_VEL_RANGE
        n_rand = int(random_mask.sum())
        rand_vel = torch.empty(n_rand, 2, device=device).uniform_(lo, hi)
        lin_vel[random_mask] = rand_vel[:, 0]
        rot_vel[random_mask] = rand_vel[:, 1]
    # stationary / static_stationary (policy 0 or 1): lin_vel = rot_vel = 0,
    # already the initialized value -- no motion, matching move_opponent's
    # behavior for opponent_vel == [0, 0].

    vel = torch.stack(
        [lin_vel.abs(), rot_vel], dim=1
    )  # move_opponent: vel[0]=abs(vel[0])
    vel = torch.clamp(vel, -2.0, 2.0)
    lin_vel, rot_vel = vel[:, 0], vel[:, 1]

    x_vel = lin_vel * torch.cos(theta + 0.5 * torch.pi)
    y_vel = lin_vel * torch.sin(theta + 0.5 * torch.pi)
    new_pose = torch.stack(
        [
            torch.clamp(
                pose[:, 0] - ctrl_dt * x_vel,
                -_CHASETAG_ARENA_BOUND,
                _CHASETAG_ARENA_BOUND,
            ),
            torch.clamp(
                pose[:, 1] - ctrl_dt * y_vel,
                -_CHASETAG_ARENA_BOUND,
                _CHASETAG_ARENA_BOUND,
            ),
            theta + ctrl_dt * rot_vel,
        ],
        dim=1,
    )
    state["pose"] = new_pose
    state["vel"] = vel
    _chasetag_write_mocap_pose(env, new_pose)


def _chasetag_opponent_pos3(env) -> torch.Tensor:
    """Opponent mocap world position as ``(N, 3)`` (z fixed at 0, matching CPU)."""
    import torch  # noqa: PLC0415

    state = _chasetag_opponent_state(env)
    pose = state["pose"]
    return torch.stack([pose[:, 0], pose[:, 1], torch.zeros_like(pose[:, 0])], dim=1)


def _chasetag_obs_opponent_relative(env):
    """7-dim ``[rel_pos(3), rel_vel(3), dist(1)]`` block, vectorized equivalent
    of ``relative_pose_obs`` as used by ``chasetag_obs`` (self_pos/self_vel =
    the agent's root ``qpos[:3]``/``qvel[:3]``, matching the CPU formula
    exactly rather than a pelvis-site lookup)."""
    import torch  # noqa: PLC0415

    data = env.scene[_CHASETAG_ENTITY_NAME].data.data
    state = _chasetag_opponent_state(env)
    self_pos = data.qpos[:, :3]
    self_vel = data.qvel[:, :3]
    opp_pos = _chasetag_opponent_pos3(env)
    vel = state["vel"]
    opp_vel = torch.stack([vel[:, 0], vel[:, 1], torch.zeros_like(vel[:, 0])], dim=1)
    rel_pos = opp_pos - self_pos
    rel_vel = opp_vel - self_vel
    dist = torch.linalg.norm(rel_pos, dim=1, keepdim=True)
    return torch.cat([rel_pos, rel_vel, dist], dim=1)


def _chasetag_obs_role(env):
    """Chaser one-hot ``[1, 0]`` (this env always plays the CHASE role). Shape (N, 2)."""
    import torch  # noqa: PLC0415

    data = env.scene[_CHASETAG_ENTITY_NAME].data.data
    n = data.qpos.shape[0]
    out = torch.zeros(n, 2, dtype=torch.float32, device=data.qpos.device)
    out[:, 0] = 1.0
    return out


# ── Rewards / terminations ──────────────────────────────────────────────────


def _chasetag_pelvis_z(env):
    """Pelvis height (free-joint qpos z). Shape (N,)."""
    return env.scene[_CHASETAG_ENTITY_NAME].data.data.qpos[:, 2]


def _chasetag_fallen_bool(env):
    """Bool (N,): pelvis height below ``_CHASETAG_FALL_HEIGHT`` (FLAT-terrain
    fall check, matches ``ChaseTagEnv._get_fallen_condition``)."""
    return _chasetag_pelvis_z(env) < _CHASETAG_FALL_HEIGHT


def _chasetag_distance_to_opponent(env):
    """Planar distance from the agent's root to the opponent. Shape (N,)."""
    import torch  # noqa: PLC0415

    data = env.scene[_CHASETAG_ENTITY_NAME].data.data
    agent_xy = data.qpos[:, :2]
    opp_xy = _chasetag_opponent_pos3(env)[:, :2]
    return torch.linalg.norm(agent_xy - opp_xy, dim=1)


def _chasetag_tagged_bool(env):
    """Bool (N,): agent within ``win_distance`` of the opponent (CHASE win)."""
    return _chasetag_distance_to_opponent(env) <= _CHASETAG_WIN_DISTANCE


def _chasetag_distance_delta_reward(env, weight: float = -0.5):
    """Potential-based distance-closing reward: negative Δdistance since last
    step (closing the gap gives positive reward), matching ``ChaseTagEnv``'s
    CHASE ``distance`` reward term (``b8f61507``'s fix — delta, not raw
    absolute distance) with its default weight folded in via ``RewardTermCfg``
    already, so this returns the raw (unweighted) per-step delta.
    """

    dist = _chasetag_distance_to_opponent(env)
    prev = getattr(env, "_chasetag_prev_distance", None)
    if prev is None or prev.shape[0] != dist.shape[0]:
        prev = dist.clone()
    delta = dist - prev
    env._chasetag_prev_distance = dist.clone()  # noqa: SLF001
    return delta


def _chasetag_alive_reward(env):
    """Continuous upright-ness in [0, 1], matching ``ChaseTagEnv``'s FLAT-terrain
    ``alive`` reward: ``clip((pelvis_z - 0.5) / 0.5, 0, 1)``."""
    import torch  # noqa: PLC0415

    z = _chasetag_pelvis_z(env)
    return torch.clamp((z - 0.5) / 0.5, 0.0, 1.0)


def _chasetag_tag_bonus(env):
    """Sparse +1 the step the agent tags the opponent (see ``solved`` reward)."""
    import torch  # noqa: PLC0415

    return _chasetag_tagged_bool(env).to(dtype=torch.float32)


def _chasetag_fall_penalty(env):
    """Fall penalty term: 1.0 the step the pelvis drops below the fall height."""
    import torch  # noqa: PLC0415

    return _chasetag_fallen_bool(env).to(dtype=torch.float32)


def _chasetag_act_reg(env):
    """Action regularization on muscle activations (mean L2 per env)."""
    import torch  # noqa: PLC0415

    data = env.scene[_CHASETAG_ENTITY_NAME].data.data
    return torch.mean(torch.square(data.act), dim=1)


def _make_chasetag_fbp2_env_cfg(num_envs: int = 128) -> ManagerBasedRlEnvCfg:
    """ManagerBasedRlEnvCfg for ``myoChallengeChaseTagFBP2-v0`` (GPU match).

    Full-body (354-muscle) agent vs. a scripted, mocap-driven CHASE-only
    opponent. Obs = the 537-dim additive chase-tag observation (see module
    docstring above); actions = single muscle-activation term over all 354
    muscles; rewards = distance-closing potential + alive bonus + tag bonus −
    fall penalty − act regularization, mirroring the CPU ``ChaseTagEnv``'s
    current (``b8f61507``) reward shaping.
    """
    muscle_names, tendon_names = _chasetag_muscle_and_tendon_names()
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "qpos_local": ObservationTermCfg(func=_chasetag_obs_qpos_local),
                "qvel_local": ObservationTermCfg(func=_chasetag_obs_qvel_local),
                "act": ObservationTermCfg(func=_chasetag_obs_act),
                "root_vel_body": ObservationTermCfg(func=_chasetag_obs_root_vel_body),
                "heading_cmd": ObservationTermCfg(func=_chasetag_obs_heading_cmd),
                "orientation": ObservationTermCfg(func=_chasetag_obs_orientation),
                "opponent_relative": ObservationTermCfg(
                    func=_chasetag_obs_opponent_relative
                ),
                "role": ObservationTermCfg(func=_chasetag_obs_role),
            },
        ),
    }
    actions = {
        # action_mode="direct": this env hosts policies warm-started from
        # bc_directional_v2 (and PPO fine-tunes thereof), trained on
        # MuscleMimicFullbodyDirectionalEnv, whose action_space passes
        # actuator_ctrlrange straight through with no transform. The default
        # "sigmoid" mode (ctrl = sigmoid(5*(a-0.5)), the WalkEnvV0/CPU
        # myoLeg convention) is wrong here for the same reason the CPU
        # ChaseTagEnv registration's normalize_act=True was wrong (see
        # myosuite/envs/myo/tasks/challenge/__init__.py's
        # myoChallengeChaseTagFBP2-v0 kwargs comment): this full-body
        # model's muscle actuators have ctrlrange=[-1, 1] (not the classic
        # myoLeg [0, 1]), and sigmoid's output is always in (0, 1), so
        # muscles could never receive a negative ctrl value at all under
        # the default mode, regardless of the policy's actual output.
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=_CHASETAG_ENTITY_NAME,
            actuator_names=muscle_names,
            tendon_names=tendon_names,
            action_mode="direct",
        ),
    }
    rewards = {
        "distance_closing": RewardTermCfg(
            func=_chasetag_distance_delta_reward, weight=-0.5
        ),
        "alive_reward": RewardTermCfg(func=_chasetag_alive_reward, weight=0.5),
        "tag_bonus": RewardTermCfg(func=_chasetag_tag_bonus, weight=1000.0),
        "fall_penalty": RewardTermCfg(func=_chasetag_fall_penalty, weight=-100.0),
        "act_reg": RewardTermCfg(func=_chasetag_act_reg, weight=-0.1),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
        "fallen": TerminationTermCfg(func=_chasetag_fallen_bool),
        "tagged": TerminationTermCfg(func=_chasetag_tagged_bool),
    }
    events = {
        # Mandatory for every leg-family full-body GPU env on this branch — without
        # it, resets after the first fall back to mjlab's default init (root at the
        # origin, i.e. in the ground), not a standing pose.
        "reset_scene_to_default": EventTermCfg(
            func=mdp_events.reset_scene_to_default, mode="reset"
        ),
        "reset_opponent": EventTermCfg(func=_chasetag_reset_opponent, mode="reset"),
        "step_opponent": EventTermCfg(func=_chasetag_step_opponent, mode="step"),
    }
    return mjlab_env_cfg_from_task_config(
        cfg=TaskConfig(max_episode_steps=500),
        spec_fn=_chasetag_spec_fn,
        entity_name=_CHASETAG_ENTITY_NAME,
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=tuple(tendon_names),
                transmission_type=TransmissionType.TENDON,
            ),
        ),
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        num_envs=num_envs,
        decimation=5,  # matches the leg walk/directional twins' proven GPU decimation
        sim_cfg=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.002, ccd_iterations=500),
            njmax=1024,
            nconmax=512,
        ),
        episode_length_s=20.0,
        init_state=_chasetag_init_state(),
    )


def _chasetag_ppo_runner_cfg(
    experiment_name: str = "myo_chasetag_fbp2",
) -> RslRlOnPolicyRunnerCfg:
    """PPO runner config for the FBP2 chase-tag GPU env (mirrors the directional runner cfg)."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg=None,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=4,
            num_mini_batches=4,
            learning_rate=3e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=experiment_name,
        save_interval=100,
        num_steps_per_env=48,
        max_iterations=500,
        obs_groups={"actor": ("policy",), "critic": ("policy",)},
    )


def register_mjlab_tasks() -> None:
    """Register MyoSuite env ids with mjlab.tasks.registry. Idempotent."""

    # Basic-suite CPU twins (pose, ...): importing the package registers them.
    import myosuite.envs.myo.backends.mjlab.tasks  # noqa: F401, PLC0415

    # myoLegWalk-v0 (+ Sarc/Fati) are CPU twins registered by the tasks package.

    # myoLegDirectional* are CPU twins registered by the tasks package.

    register_table_tennis_mjlab_tasks()

    # --- ChaseTag full-body vs. scripted opponent (GPU match for CPU FBP2) ---
    try:
        chasetag_env_cfg = _make_chasetag_fbp2_env_cfg()
        chasetag_rl_cfg = _chasetag_ppo_runner_cfg()
        register_mjlab_task(
            task_id="myoChallengeChaseTagFBP2-v0",
            env_cfg=chasetag_env_cfg,
            play_env_cfg=chasetag_env_cfg,
            rl_cfg=chasetag_rl_cfg,
            runner_cls=None,
        )
    except ValueError:
        pass  # already registered
    except Exception:
        logging.getLogger(__name__).warning(
            "mjlab: failed to register myoChallengeChaseTagFBP2-v0", exc_info=True
        )


def bootstrap_myosuite_mjlab_registry(
    *,
    clip_path: str | os.PathLike[str] | None = None,
    rl_cfg_fn: Callable[[], Any] | None = None,
    use_lookahead: bool = True,
) -> None:
    """Register all MyoSuite mjlab tasks in one call (idempotent).

    Runs :func:`register_mjlab_tasks` (static envs). Plain ``import mjlab`` stays
    limited to MyoSuite's own mjlab tasks.

    If *clip_path* is set, or environment variable ``MYOSUITE_MIMIC_CLIP`` or
    ``MIMIC_CLIP`` points to an existing file, also registers clip-mode
    ``myoMimicFullbody-v0`` via
    :func:`~myosuite.envs.myo.backends.mjlab.mimic_mjlab_env.register_mimic_mjlab_tasks_with_clip`.

    Safe to call multiple times per process (notebooks, scripts, tests).  When
    no clip path is available, Mimic task registration is skipped entirely;
    pass *clip_path* here or call
    :func:`~myosuite.envs.myo.backends.mjlab.mimic_mjlab_env.register_mimic_mjlab_tasks_with_clip`
    directly.

    Raises:
        FileNotFoundError: When *clip_path* is given explicitly but does not exist.
        ValueError: From :func:`register_mimic_mjlab_tasks_with_clip` if the
            clip lacks ``site_xpos``.
    """
    register_mjlab_tasks()

    explicit_clip = clip_path is not None
    raw = (
        clip_path
        if explicit_clip
        else os.environ.get("MYOSUITE_MIMIC_CLIP") or os.environ.get("MIMIC_CLIP")
    )
    if not raw:
        return
    p = Path(raw).expanduser()
    if not p.is_file():
        if explicit_clip:
            raise FileNotFoundError(f"Mimic clip path does not exist: {p}")
        return

    from myosuite.core.trajectory_io import load_motion_clip
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        register_mimic_mjlab_tasks_with_clip,
    )

    cfg_fn = rl_cfg_fn or default_mimic_clip_on_policy_runner_cfg
    clip = load_motion_clip(p, expected_nq=89, expected_nv=88)
    try:
        register_mimic_mjlab_tasks_with_clip(
            register_mjlab_task=register_mjlab_task,
            rl_cfg_fn=cfg_fn,
            clip=clip,
            use_lookahead=use_lookahead,
        )
    except Exception as exc:
        if explicit_clip:
            raise
        logging.getLogger(__name__).warning(
            "Skipping MYOSUITE_MIMIC_CLIP / MIMIC_CLIP bootstrap: %s", exc
        )
