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

import functools
import logging
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import torch
from mjlab.actuator import XmlActuatorCfg as _XmlWrappedActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task

from myosuite.core.config import TaskConfig
from myosuite.envs.myo.assets._resolve import resolve_elbow_xml as _resolve_elbow_xml
from myosuite.utils.asset_path_resolver import resolve_model_xml_path
from myosuite.envs.myo.backends.mjlab.mjlab_env_base import normalize_mjlab_env_ids
from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
    MyoMuscleActivationActionCfg,
    mjlab_env_cfg_from_task_config,
)
from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
    _init_state_from_model,
    _muscle_actuator_names,
    _muscle_tendon_names,
    default_mimic_clip_on_policy_runner_cfg,
)
from myosuite.envs.myo.backends.mjlab.register_mjlab_tabletennis import (
    register_table_tennis_mjlab_tasks,
)

if TYPE_CHECKING:  # pragma: no cover
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import CpuTaskSpec


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
        sim_cfg=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)),
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
# ChaseTag full-body vs. scripted opponent (mjlab half of myoChallengeChaseTagFBP2-v0)
# ---------------------------------------------------------------------------
# The CPU registration (tasks/challenge/__init__.py) is the source of truth for
# the model, control step, horizon, reward weights and opponent parameters. The
# terms reproduce ChaseTagEnv(task_choice="CHASE", terrain="FLAT") and its
# 537-dim chasetag_obs layout, one observation term per CHASETAG_OBS_KEYS block.
# Agent and mocap "opponent" share one MjSpec, hence one entity, as on the CPU.

_CHASETAG_ENV_ID = "myoChallengeChaseTagFBP2-v0"
_CHASETAG_ENTITY_NAME = "chasetag_agent"
# ChaseTagEnv: FLAT-terrain fall height and the CHASE out-of-bounds limit.
_CHASETAG_FALL_HEIGHT = 0.5
_CHASETAG_AGENT_BOUND = 6.5
# ChallengeOpponent: arena clip, static_stationary spot and the random policy's
# velocity process ColoredNoiseProcess(beta=2, size=(2, 2000), scale=10).
_CHASETAG_ARENA_BOUND = 5.5
_CHASETAG_STATIC_STATIONARY_POSE = (0.0, -5.0, 0.0)
_CHASETAG_NOISE_BETA = 2.0
_CHASETAG_NOISE_STEPS = 2000
_CHASETAG_NOISE_SCALE = 10.0
# Fixed rejection rounds for the minimum spawn distance (CPU: unbounded loop). A
# draw is rejected with probability <= pi * 2**2 / 10**2 < 0.13, so 16 rounds
# leave fewer than 1e-14 of the resets unresolved.
_CHASETAG_SPAWN_ROUNDS = 16


def _chasetag_cpu_task() -> CpuTaskSpec:
    """The CPU registration of ``myoChallengeChaseTagFBP2-v0``."""
    from myosuite.envs.myo.backends.mjlab.tasks import cpu_reference  # noqa: PLC0415

    return cpu_reference.cpu_task_spec(_CHASETAG_ENV_ID)


@functools.cache
def _chasetag_cpu_model() -> mujoco.MjModel:
    """The compiled CPU model (physics options, keyframe, muscle names)."""
    return _chasetag_cpu_task().kwargs["model_spec_fn"]().compile()


def _chasetag_spec_fn() -> mujoco.MjSpec:
    """The CPU ``MjSpec``, terrain hidden as ``ChaseTagEnv`` does on flat ground."""
    from myosuite.envs.myo.backends.mjlab.tasks.leg.stand_env_cfg import (  # noqa: PLC0415
        hide_terrain,
    )

    spec = _chasetag_cpu_task().kwargs["model_spec_fn"]()
    hide_terrain(spec)
    return spec


def _chasetag_root(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU-layout root ``qpos[:7]`` / ``qvel[:6]`` of the agent, (N, 7) / (N, 6).

    Read from ``qpos`` / ``qvel``: mjlab's ``root_link_*`` API derives from
    ``xpos`` / ``cvel``, which lag one substep until ``forward()`` (terminations
    and rewards run before it). The CPU env reads the pelvis ``xpos`` after
    ``mj_kinematics``; the pelvis sits at the free-joint origin, so that is
    ``qpos[:3]``.
    """
    index = env.scene[_CHASETAG_ENTITY_NAME].indexing
    qpos = env.sim.data.qpos[:, index.free_joint_q_adr.long()]
    qvel = env.sim.data.qvel[:, index.free_joint_v_adr.long()]
    pos = qpos[:, :3] - env.scene.env_origins
    return torch.cat([pos, qpos[:, 3:]], dim=1), qvel


# ── Observation terms: the chasetag_obs blocks, in CHASETAG_OBS_KEYS order ──


def _chasetag_obs_qpos_local(env: ManagerBasedRlEnv) -> torch.Tensor:
    """``qpos[7:]``. Shape (N, nq - 7)."""
    return env.scene[_CHASETAG_ENTITY_NAME].data.joint_pos


def _chasetag_obs_qvel_local(env: ManagerBasedRlEnv) -> torch.Tensor:
    """``qvel[6:]``. Shape (N, nv - 6)."""
    return env.scene[_CHASETAG_ENTITY_NAME].data.joint_vel


def _chasetag_obs_act(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Muscle activation state. Shape (N, na)."""
    # accepted: no entity.data API for muscle activation — entity.data.data.act
    return env.scene[_CHASETAG_ENTITY_NAME].data.data.act


def _chasetag_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Yaw of ``wxyz`` quaternions, as ``_pelvis_yaw_from_qpos``. Shape (N,)."""
    w, x, y, z = quat.unbind(dim=1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _chasetag_rotate_by_neg_yaw(xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate planar vectors ``(N, 2)`` by ``-yaw``, as ``_directional_obs``."""
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    x, y = xy.unbind(dim=1)
    return torch.stack([c * x - s * y, s * x + c * y], dim=1)


def _chasetag_obs_root_vel_body(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Root planar velocity in the pelvis yaw frame. Shape (N, 2)."""
    qpos, qvel = _chasetag_root(env)
    return _chasetag_rotate_by_neg_yaw(qvel[:, :2], _chasetag_yaw(qpos[:, 3:7]))


def _chasetag_obs_heading_cmd(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Fixed heading command ``[1, 0]`` (no heading command in chase-tag). (N, 2)."""
    out = torch.zeros(env.num_envs, 2, device=env.device)
    out[:, 0] = 1.0
    return out


def _chasetag_obs_orientation(env: ManagerBasedRlEnv) -> torch.Tensor:
    """``[roll, pitch, wx_b, wy_b, wz, vz]`` of ``_directional_obs``. Shape (N, 6)."""
    qpos, qvel = _chasetag_root(env)
    w, x, y, z = qpos[:, 3:7].unbind(dim=1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    ang_b = _chasetag_rotate_by_neg_yaw(qvel[:, 3:5], _chasetag_yaw(qpos[:, 3:7]))
    return torch.cat(
        [roll[:, None], pitch[:, None], ang_b, qvel[:, 5:6], qvel[:, 2:3]], dim=1
    )


# ── Scripted CHASE opponent ──────────────────────────────────────────────────


def _powerlaw_psd_gaussian(
    exponent: float,
    shape: tuple[int, ...],
    device: str | torch.device,
    normals: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch port of ``pink.colorednoise.powerlaw_psd_gaussian`` (``fmin=0``).

    Unit-variance Gaussian ``(1/f)**exponent`` noise along the last axis
    (Timmer & Koenig 1995), generated on *device*: pink and colorednoise are
    numpy-only. ``test_chasetag_fbp2_parity`` checks it against pink for the
    same normal draws.

    Args:
        exponent: Power-law exponent (2: Brownian noise).
        shape: Output shape; the last axis is time.
        device: Torch device.
        normals: Standard normals of shape ``(2, *shape[:-1], shape[-1] // 2 + 1)``
            (real and imaginary parts) to use instead of fresh draws.

    Returns:
        Float32 noise of shape *shape*.
    """
    samples = shape[-1]
    freqs = torch.fft.rfftfreq(samples, dtype=torch.float64, device=device)
    freqs[0] = 1.0 / samples  # pink's low-frequency cutoff 1 / samples
    s_scale = freqs ** (-exponent / 2.0)
    w = s_scale[1:].clone()
    w[-1] *= (1 + samples % 2) / 2.0
    sigma = 2.0 * torch.sqrt(torch.sum(w**2)) / samples
    if normals is None:
        normals = torch.randn(
            2, *shape[:-1], freqs.numel(), dtype=torch.float64, device=device
        )
    s_real = normals[0].to(torch.float64) * s_scale
    s_imag = normals[1].to(torch.float64) * s_scale
    if samples % 2 == 0:  # real Nyquist bin
        s_imag[..., -1] = 0.0
        s_real[..., -1] *= math.sqrt(2.0)
    s_imag[..., 0] = 0.0  # real DC bin
    s_real[..., 0] *= math.sqrt(2.0)
    noise = torch.fft.irfft(torch.complex(s_real, s_imag), n=samples, dim=-1)
    return (noise / sigma).float()


class _ChaseTagOpponent(ManagerTermBase):
    """Vectorized ``ChallengeOpponent`` for ``player_task="CHASE"``.

    The ``reset_opponent`` reset event; owns the per-env opponent state. The
    action term's ``pre_physics_fn`` advances it before the physics step, as
    ``ChaseTagEnv.step`` does, so terminations, rewards and observations of a
    step all see the moved opponent. ``policy`` 0 / 1 / 2 = static_stationary /
    stationary / random (the CHASE choices of ``sample_opponent_policy``).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        n, device = env.num_envs, env.device
        self._vel_range = tuple(float(v) for v in cfg.params["random_vel_range"])
        self._static_pose = torch.tensor(
            _CHASETAG_STATIC_STATIONARY_POSE, device=device
        )
        self.pose = torch.zeros(n, 3, device=device)  # [x, y, theta]
        self.vel = torch.zeros(n, 2, device=device)  # [lin_vel, rot_vel]
        self.policy = torch.ones(n, dtype=torch.long, device=device)
        # Unit-variance noise buffer and read index of the colored-noise process.
        self.noise = torch.zeros(n, 2, _CHASETAG_NOISE_STEPS, device=device)
        self.noise_idx = torch.zeros(n, dtype=torch.long, device=device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        min_spawn_distance: float,
        opponent_probabilities: tuple[float, ...],
        random_vel_range: tuple[float, float],
    ) -> None:
        """``reset_opponent``: policy, spawn >= *min_spawn_distance* from the agent."""
        del random_vel_range  # used by advance()
        ids = normalize_mjlab_env_ids(env, env_ids)
        k = ids.numel()
        if k == 0:
            return
        agent_xy = _chasetag_root(env)[0][ids, :2]
        xy = torch.empty(k, 2, device=env.device).uniform_(-5.0, 5.0)
        for _ in range(_CHASETAG_SPAWN_ROUNDS):  # fixed rounds: no host sync
            redraw = torch.linalg.norm(xy - agent_xy, dim=1) < min_spawn_distance
            xy = torch.where(redraw[:, None], torch.empty_like(xy).uniform_(-5, 5), xy)
        theta = torch.empty(k, 1, device=env.device).uniform_(-2 * math.pi, 2 * math.pi)
        r = torch.rand(k, device=env.device)
        p_static, p_stationary = opponent_probabilities[:2]
        policy = (r >= p_static).long() + (r >= p_static + p_stationary).long()
        pose = torch.cat([xy, theta], dim=1)
        self.pose[ids] = torch.where((policy == 0)[:, None], self._static_pose, pose)
        self.vel[ids] = 0.0
        self.policy[ids] = policy
        # reset_noise_process: a fresh noise series per episode.
        self.noise[ids] = _powerlaw_psd_gaussian(
            _CHASETAG_NOISE_BETA, (k, 2, _CHASETAG_NOISE_STEPS), env.device
        )
        self.noise_idx[ids] = 0
        self._write_mocap(env)

    def advance(self, dt: float) -> None:
        """One step of ``update_opponent_state`` + ``move_opponent``, all envs.

        Args:
            dt: Control timestep in seconds.
        """
        lo, hi = self._vel_range
        # The horizon equals the noise length; the index wraps beyond it.
        idx = (self.noise_idx % _CHASETAG_NOISE_STEPS).view(-1, 1, 1).expand(-1, 2, 1)
        sample = _CHASETAG_NOISE_SCALE * torch.gather(self.noise, 2, idx).squeeze(2)
        self.noise_idx += 1
        vel = torch.where(
            (self.policy == 2)[:, None], sample.clamp(lo, hi), torch.zeros_like(sample)
        )
        vel = torch.stack([vel[:, 0].abs(), vel[:, 1]], dim=1)
        step = vel.clamp(-2.0, 2.0) * dt
        heading = self.pose[:, 2] + 0.5 * math.pi
        bound = _CHASETAG_ARENA_BOUND
        self.pose = torch.stack(
            [
                (self.pose[:, 0] - step[:, 0] * torch.cos(heading)).clamp(
                    -bound, bound
                ),
                (self.pose[:, 1] - step[:, 0] * torch.sin(heading)).clamp(
                    -bound, bound
                ),
                self.pose[:, 2] + step[:, 1],
            ],
            dim=1,
        )
        self.vel = vel
        self._write_mocap(self._env)

    def _write_mocap(self, env: ManagerBasedRlEnv) -> None:
        """Write the pose into the opponent mocap body (mocap index 0).

        The mocap body belongs to the agent's composite entity, which has no
        mocap write API (``Entity.write_mocap_pose_to_sim`` needs a mocap root
        body), so this is a documented direct write.
        """
        data = env.scene[_CHASETAG_ENTITY_NAME].data.data
        half = 0.5 * self.pose[:, 2]
        data.mocap_pos[:, 0, :2] = self.pose[:, :2] + env.scene.env_origins[:, :2]
        data.mocap_pos[:, 0, 2] = 0.0
        zeros = torch.zeros_like(half)
        data.mocap_quat[:, 0] = torch.stack(
            [torch.cos(half), zeros, zeros, torch.sin(half)], dim=1
        )


def _chasetag_opponent(env: ManagerBasedRlEnv) -> _ChaseTagOpponent:
    """The opponent state, owned by the ``reset_opponent`` event term."""
    return env.event_manager.get_term_cfg("reset_opponent").func


def _chasetag_advance_opponent(env: ManagerBasedRlEnv) -> None:
    """Action ``pre_physics_fn``: move the opponent before the physics step."""
    _chasetag_opponent(env).advance(env.step_dt)


def _chasetag_obs_opponent_relative(env: ManagerBasedRlEnv) -> torch.Tensor:
    """``[rel_pos, rel_vel, dist]`` of ``chasetag_obs``. Shape (N, 7).

    The opponent sits at its mocap position (z = 0); its "velocity" is the
    ``[lin_vel, rot_vel, 0]`` control pair, as on the CPU.
    """
    qpos, qvel = _chasetag_root(env)
    opponent = _chasetag_opponent(env)
    zeros = torch.zeros_like(opponent.vel[:, :1])
    rel_pos = torch.cat([opponent.pose[:, :2], zeros], dim=1) - qpos[:, :3]
    rel_vel = torch.cat([opponent.vel, zeros], dim=1) - qvel[:, :3]
    dist = torch.linalg.norm(rel_pos, dim=1, keepdim=True)
    return torch.cat([rel_pos, rel_vel, dist], dim=1)


def _chasetag_obs_role(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Chaser one-hot ``[1, 0]`` (this env always plays the CHASE role). (N, 2)."""
    out = torch.zeros(env.num_envs, 2, device=env.device)
    out[:, 0] = 1.0
    return out


# ── Rewards and terminations: ChaseTagEnv.get_reward_dict / _get_done (CHASE) ──


def _chasetag_distance(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Planar agent–opponent distance (``distance_abs``). Shape (N,)."""
    agent_xy = _chasetag_root(env)[0][:, :2]
    return torch.linalg.norm(agent_xy - _chasetag_opponent(env).pose[:, :2], dim=1)


def _chasetag_fallen(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Bool (N,): pelvis below the FLAT-terrain fall height."""
    return _chasetag_root(env)[0][:, 2] < _CHASETAG_FALL_HEIGHT


def _chasetag_out_of_bounds(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Bool (N,): CHASE loses when ``|x|`` or ``|y|`` exceeds the agent bound."""
    agent_xy = _chasetag_root(env)[0][:, :2]
    return (agent_xy.abs() > _CHASETAG_AGENT_BOUND).any(dim=1)


def _chasetag_tagged(env: ManagerBasedRlEnv, win_distance: float) -> torch.Tensor:
    """Bool (N,): the agent is within *win_distance* of the opponent (CHASE win)."""
    return _chasetag_distance(env) <= win_distance


class _ChaseTagDistanceDelta(ManagerTermBase):
    """``distance`` reward: change of the agent–opponent distance since the last step.

    ``reset`` forgets the previous distance, so the first step of an episode
    scores 0, as ``ChaseTagEnv`` (``_prev_distance = None`` on reset).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self._prev = torch.full((env.num_envs,), float("nan"), device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Forget the previous distance of *env_ids* (all envs for ``None``)."""
        self._prev[slice(None) if env_ids is None else env_ids] = float("nan")

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        dist = _chasetag_distance(env)
        delta = torch.where(
            torch.isnan(self._prev), torch.zeros_like(dist), dist - self._prev
        )
        self._prev.copy_(dist)
        return delta


def _chasetag_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """``alive``: FLAT-terrain uprightness ``clip((z - 0.5) / 0.5, 0, 1)``. (N,)."""
    z = _chasetag_root(env)[0][:, 2]
    return torch.clamp((z - 0.5) / 0.5, 0.0, 1.0)


def _chasetag_solved(env: ManagerBasedRlEnv, win_distance: float) -> torch.Tensor:
    """``solved``: 1 on the step the agent tags the opponent. (N,)."""
    return _chasetag_tagged(env, win_distance).float()


def _chasetag_lose(env: ManagerBasedRlEnv) -> torch.Tensor:
    """``lose``: 1 on a fall or out of bounds (the CHASE time limit lies beyond
    the horizon: CPU ``data.time`` after the last step is still below 20 s). (N,)."""
    return (_chasetag_fallen(env) | _chasetag_out_of_bounds(env)).float()


def _chasetag_act_reg(env: ManagerBasedRlEnv) -> torch.Tensor:
    """``act_reg``: ``||act|| / na``, as ``ChaseTagEnv``. Shape (N,)."""
    act = _chasetag_obs_act(env)
    return torch.linalg.norm(act, dim=1) / act.shape[1]


def _make_chasetag_fbp2_env_cfg(num_envs: int = 128) -> ManagerBasedRlEnvCfg:
    """``ManagerBasedRlEnvCfg`` of ``myoChallengeChaseTagFBP2-v0``, from its CPU registration.

    Full-body (354-muscle) agent vs. a scripted mocap opponent, CHASE only:
    the 537-dim chasetag_obs observation, direct muscle controls, the CPU
    reward terms and weights (unscaled by dt), the CPU control step, horizon,
    keyframe reset and physics options.

    Args:
        num_envs: Number of parallel environments.

    Returns:
        The env config.
    """
    from myosuite.envs.myo.backends.mjlab.tasks import cpu_reference as ref  # noqa: PLC0415
    from myosuite.envs.myo.backends.mjlab.tasks import mdp  # noqa: PLC0415

    task = _chasetag_cpu_task()
    kwargs = task.kwargs
    model = _chasetag_cpu_model()
    muscle_names = _muscle_actuator_names(model)
    tendon_names = _muscle_tendon_names(model)
    win = {"win_distance": float(kwargs["win_distance"])}

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
        # action_mode="direct": the CPU env has normalize_act=False (ctrl =
        # action in the [-1, 1] ctrlrange), the convention of the hosted
        # bc_directional_v2-style policies; sigmoid could never output < 0.
        # pre_physics_fn moves the opponent first, as ChaseTagEnv.step.
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=_CHASETAG_ENTITY_NAME,
            actuator_names=muscle_names,
            tendon_names=tendon_names,
            action_mode="direct",
            pre_physics_fn=_chasetag_advance_opponent,
        ),
    }
    reward_terms = {
        "distance": (_ChaseTagDistanceDelta, {}),
        "alive": (_chasetag_alive, {}),
        "solved": (_chasetag_solved, win),
        "lose": (_chasetag_lose, {}),
        "act_reg": (_chasetag_act_reg, {}),
    }
    rewards = {
        key: RewardTermCfg(
            func=reward_terms[key][0], weight=float(weight), params=reward_terms[key][1]
        )
        for key, weight in kwargs["weighted_reward_keys"].items()
    }
    # ChaseTagEnv._get_done for CHASE: lose (fall, out of bounds) or win (tag).
    terminations = {
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
        "fallen": TerminationTermCfg(func=_chasetag_fallen),
        "out_of_bounds": TerminationTermCfg(func=_chasetag_out_of_bounds),
        "tagged": TerminationTermCfg(func=_chasetag_tagged, params=win),
    }
    events = {
        # reset_type="none": keyframe 0, including its body-frame root angular
        # velocity (mjlab's default reset reads it as world-frame).
        "reset_scene_to_default": EventTermCfg(
            func=mdp.reset_to_cpu_state,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(_CHASETAG_ENTITY_NAME),
                "qpos": tuple(float(q) for q in model.key_qpos[0]),
                "qvel": tuple(float(v) for v in model.key_qvel[0]),
            },
        ),
        # After the agent reset: the spawn distance is measured from its new pose.
        "reset_opponent": EventTermCfg(
            func=_ChaseTagOpponent,
            mode="reset",
            params={
                "min_spawn_distance": float(kwargs["min_spawn_distance"]),
                "opponent_probabilities": tuple(kwargs["opponent_probabilities"]),
                "random_vel_range": tuple(kwargs["random_vel_range"]),
            },
        ),
    }
    metrics = {
        "success": MetricsTermCfg(func=_chasetag_tagged, params=win, reduce="last")
    }
    entity = EntityCfg(
        spec_fn=_chasetag_spec_fn,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                _XmlWrappedActuatorCfg(
                    target_names_expr=tuple(tendon_names),
                    transmission_type=TransmissionType.TENDON,
                ),
            )
        ),
        init_state=_init_state_from_model(model),
    )
    step_dt = float(model.opt.timestep) * task.frame_skip
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(num_envs=num_envs, entities={_CHASETAG_ENTITY_NAME: entity}),
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        events=events,
        # CPU <option> (Euler integrator, solver iterations, ccd) of the model.
        sim=SimulationCfg(
            mujoco=ref.mujoco_cfg_from_model(model), njmax=1024, nconmax=512
        ),
        decimation=task.frame_skip,
        episode_length_s=ref.episode_length_s(task.max_episode_steps, step_dt),
        scale_rewards_by_dt=False,
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

    # --- ChaseTag full-body vs. scripted opponent (mjlab half of CPU FBP2) ---
    try:
        chasetag_env_cfg = _make_chasetag_fbp2_env_cfg()
        chasetag_rl_cfg = _chasetag_ppo_runner_cfg()
        register_mjlab_task(
            task_id=_CHASETAG_ENV_ID,
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
