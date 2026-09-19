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
import weakref
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
from myosuite.core.muscle_conditions import apply_sarcopenia_to_spec
from myosuite.envs.myo.assets._resolve import resolve_elbow_xml as _resolve_elbow_xml
from myosuite.envs.myo.assets._resolve import resolve_leg_xml as _resolve_leg_xml
from myosuite.utils.asset_path_resolver import resolve_model_xml_path
from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
    MyoMuscleActivationActionCfg,
    mjlab_env_cfg_from_task_config,
)
from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
    default_mimic_clip_on_policy_runner_cfg,
)
from myosuite.envs.myo.backends.mjlab.configs.walk_cfg import WalkCfg
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


def _resolve_leg_dir() -> Path:
    from myosuite.utils.asset_path_resolver import get_sim_asset_root

    return get_sim_asset_root("myo_sim") / "leg"


_MYOSUITE_ROOT = _resolve_model_root()
_ELBOW_XML = resolve_model_xml_path(_resolve_elbow_xml("myoelbow_1dof6muscles.xml"))
# Prefer the plane-terrain leg model for mjlab: MuJoCo Warp forward on hfield
# terrain has been unreliable (including native crashes) on some platforms.
_LEG_DIR = _resolve_leg_dir()
_WALK_XML_MJX = _LEG_DIR / "myolegs_mjx.xml"
_WALK_XML = resolve_model_xml_path(
    _WALK_XML_MJX
    if _WALK_XML_MJX.is_file()
    else _resolve_leg_xml("myolegs_with_torso_plane.xml")
)


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


def _walk_muscle_names() -> tuple[str, ...]:
    """Return walk muscle actuator names from the compiled model (no hardcoded list)."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(_WALK_XML))
    return tuple(
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)
    )


# ---------------------------------------------------------------------------
# Walk observation functions (matching WalkEnvV0 / MjxWalkEnv obs space)
# ---------------------------------------------------------------------------
# Each function receives `env` (ManagerBasedRlEnv) and returns a batched
# torch.Tensor of shape (N, d) where N = num_envs and d is the component dim.
#
# Mapping to CPU env (walk_v0.py get_obs_dict):
#   qpos_without_xy  (nq-2)   ← mj_data.qpos[2:]
#   qvel             (nv)     ← mj_data.qvel * dt
#   com_vel          (2)      ← mass-weighted COM velocity (x, y)
#   torso_angle      (4)      ← mj_data.xquat[torso_id]
#   feet_heights     (2)      ← talus_l/r z-pos
#   height           (1)      ← mass-weighted COM z-height
#   feet_rel_positions (6)    ← (talus_l, talus_r) pos − pelvis pos
#   phase_var        (1)      ← (steps / hip_period) % 1
#   muscle_length    (80)     ← actuator_length
#   muscle_velocity  (80)     ← clip(actuator_velocity, −100, 100)
#   muscle_force     (80)     ← clip(actuator_force / 1000, −100, 100)
#   act              (80)     ← mj_data.act
# Total: matches CPU/MJX 403-dim observation vector.

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


_WALK_ENTITY_NAME = "walk_robot"
_WALK_HIP_PERIOD = 100  # steps per hip cycle (WalkEnvV0 default)

# Weak-keyed cache: entries are evicted automatically when the env is garbage collected.
_walk_obs_cache: weakref.WeakKeyDictionary[Any, dict] = weakref.WeakKeyDictionary()


def _resolve_walk_obs_ids(env) -> dict:
    """Lazily resolve and cache body IDs, joint qpos addresses, and reward constants.

    Loads the standalone XML model once to extract:
      - Scene-global body IDs (torso, pelvis, talus_l, talus_r)
      - Body mass tensor for mass-weighted COM computations
      - Joint qpos addresses for hip flexion, adduction, rotation (reward computation)
      - Initial quaternion from keyframe 2 (standing pose) as target_rot for ref_rot reward
      - Termination thresholds (min_height, max_rot) matching WalkEnvV0 defaults

    Uses ``entity.find_bodies()`` (mjlab/Isaac-Lab API) for body IDs when available,
    with a fallback to the standalone XML model for single-entity scenes.
    """
    if env in _walk_obs_cache:
        return _walk_obs_cache[env]

    import mujoco  # noqa: PLC0415
    import torch

    entity = env.scene[_WALK_ENTITY_NAME]
    raw_data = entity.data.data
    device = raw_data.qpos.device

    # --- Load standalone model for joint addresses, keyframe poses, and thresholds ---
    # (The mjlab scene strips keyframes, so we must load separately for key_qpos access)
    spec = mujoco.MjSpec.from_file(str(_WALK_XML))
    mj_model = spec.compile()

    def _qadr(name: str) -> int:
        """Return the qpos address of a joint by name."""
        jnt_id = mj_model.joint(name).id
        return int(mj_model.jnt_qposadr[jnt_id])

    # --- Body IDs (scene-global) ---
    try:
        torso_ids, _ = entity.find_bodies(["torso"])
        pelvis_ids, _ = entity.find_bodies(["pelvis"])
        talus_l_ids, _ = entity.find_bodies(["talus_l"])
        talus_r_ids, _ = entity.find_bodies(["talus_r"])
        torso_id = int(torso_ids[0])
        pelvis_id = int(pelvis_ids[0])
        talus_l_id = int(talus_l_ids[0])
        talus_r_id = int(talus_r_ids[0])
    except (AttributeError, TypeError, IndexError):
        # Fallback: for a single-entity scene, scene-global IDs match entity-local IDs.
        torso_id = int(mj_model.body("torso").id)
        pelvis_id = int(mj_model.body("pelvis").id)
        talus_l_id = int(mj_model.body("talus_l").id)
        talus_r_id = int(mj_model.body("talus_r").id)

    # --- Body mass ---
    try:
        body_mass = torch.as_tensor(
            raw_data.model.body_mass, dtype=torch.float32, device=device
        )
    except AttributeError:
        body_mass = torch.tensor(mj_model.body_mass, dtype=torch.float32, device=device)

    # --- Joint qpos addresses for reward computation ---
    hip_flex_l_adr = _qadr("hip_flexion_l")
    hip_flex_r_adr = _qadr("hip_flexion_r")
    hip_adduct_l_adr = _qadr("hip_adduction_l")
    hip_adduct_r_adr = _qadr("hip_adduction_r")
    hip_rot_l_adr = _qadr("hip_rotation_l")
    hip_rot_r_adr = _qadr("hip_rotation_r")

    # --- Target rotation: quaternion from keyframe 2 ("init" standing pose) ---
    # Matches MjxWalkEnv.sample_task() and WalkEnvV0 init_qpos when reset_type="init".
    nq = mj_model.nq
    nkey = mj_model.nkey
    if nkey >= 3:
        init_qpos = mj_model.key_qpos.reshape(nkey, nq)[2]
    else:
        init_qpos = mj_model.qpos0
    target_rot = torch.tensor(
        init_qpos[3:7], dtype=torch.float32, device=device
    )  # (4,)

    # --- Termination thresholds (WalkEnvV0 defaults) ---
    min_height = 0.8  # matches WalkEnvV0(min_height=0.8)
    max_rot = 0.8  # matches WalkEnvV0(max_rot=0.8)

    _walk_obs_cache[env] = dict(
        torso_id=torso_id,
        pelvis_id=pelvis_id,
        talus_l_id=talus_l_id,
        talus_r_id=talus_r_id,
        body_mass=body_mass,
        hip_flex_l_adr=hip_flex_l_adr,
        hip_flex_r_adr=hip_flex_r_adr,
        hip_adduct_l_adr=hip_adduct_l_adr,
        hip_adduct_r_adr=hip_adduct_r_adr,
        hip_rot_l_adr=hip_rot_l_adr,
        hip_rot_r_adr=hip_rot_r_adr,
        target_rot=target_rot,
        min_height=min_height,
        max_rot=max_rot,
    )
    return _walk_obs_cache[env]


def _walk_obs_qpos_without_xy(env) -> torch.Tensor:
    """Joint positions excluding x, y root translation. Shape: (N, nq-2)."""
    data = env.scene[_WALK_ENTITY_NAME].data.data
    return data.qpos[:, 2:]


def _walk_obs_qvel(env) -> torch.Tensor:
    """Joint velocities scaled by ctrl_dt. Shape: (N, nv)."""
    data = env.scene[_WALK_ENTITY_NAME].data.data
    ctrl_dt = env.physics_dt * env.cfg.decimation
    return data.qvel * ctrl_dt


def _walk_obs_com_vel(env) -> torch.Tensor:
    """Mass-weighted COM velocity (x, y). Shape: (N, 2).

    Replicates ``WalkEnvV0._get_com_velocity()`` and the MJX equivalent.
    MuJoCo's ``cvel`` stores spatial velocity in body frame; indices 3:5 are
    the translational components, negated per MuJoCo convention.
    """

    ids = _resolve_walk_obs_ids(env)
    data = env.scene[_WALK_ENTITY_NAME].data.data
    body_mass = ids["body_mass"]  # (nbody,)
    # cvel: (N, nbody, 6) — translational part at [:, :, 3:5]
    cvel_lin = -data.cvel[:, :, 3:5]  # (N, nbody, 2), sign per MJX convention
    total_mass = body_mass.sum()
    com_vel = (body_mass[None, :, None] * cvel_lin).sum(dim=1) / total_mass  # (N, 2)
    return com_vel


def _walk_obs_torso_angle(env) -> torch.Tensor:
    """Torso body quaternion (w, x, y, z). Shape: (N, 4).

    Matches ``WalkEnvV0._get_torso_angle()`` → ``mj_data.xquat[torso_id]``.
    """
    ids = _resolve_walk_obs_ids(env)
    data = env.scene[_WALK_ENTITY_NAME].data.data
    return data.xquat[:, ids["torso_id"], :]  # (N, 4)


def _walk_obs_feet_heights(env) -> torch.Tensor:
    """z-heights of left and right talus bodies. Shape: (N, 2).

    Matches ``WalkEnvV0._get_feet_heights()``.
    """
    import torch

    ids = _resolve_walk_obs_ids(env)
    data = env.scene[_WALK_ENTITY_NAME].data.data
    return torch.stack(
        [data.xpos[:, ids["talus_l_id"], 2], data.xpos[:, ids["talus_r_id"], 2]],
        dim=1,
    )  # (N, 2)


def _walk_obs_height(env) -> torch.Tensor:
    """Mass-weighted COM z-height. Shape: (N, 1).

    Matches ``WalkEnvV0._get_height()`` → ``sum(mass * xipos[:, 2]) / sum(mass)``.
    """
    ids = _resolve_walk_obs_ids(env)
    data = env.scene[_WALK_ENTITY_NAME].data.data
    body_mass = ids["body_mass"]  # (nbody,)
    height = (body_mass[None, :] * data.xipos[:, :, 2]).sum(dim=1) / body_mass.sum()
    return height.unsqueeze(-1)  # (N, 1)


def _walk_obs_feet_rel_positions(env) -> torch.Tensor:
    """Feet positions relative to pelvis. Shape: (N, 6).

    Matches ``WalkEnvV0._get_feet_relative_position()``.
    """
    import torch

    ids = _resolve_walk_obs_ids(env)
    data = env.scene[_WALK_ENTITY_NAME].data.data
    pelvis_pos = data.xpos[:, ids["pelvis_id"], :]  # (N, 3)
    left_rel = data.xpos[:, ids["talus_l_id"], :] - pelvis_pos  # (N, 3)
    right_rel = data.xpos[:, ids["talus_r_id"], :] - pelvis_pos  # (N, 3)
    return torch.cat([left_rel, right_rel], dim=1)  # (N, 6)


def _walk_obs_phase_var(env) -> torch.Tensor:
    """Cyclic gait phase variable in [0, 1). Shape: (N, 1).

    Matches ``WalkEnvV0`` → ``(self.steps / self.hip_period) % 1``.
    Here steps ≈ time / ctrl_dt so phase = (time / (hip_period * ctrl_dt)) % 1.
    """
    data = env.scene[_WALK_ENTITY_NAME].data.data
    ctrl_dt = env.physics_dt * env.cfg.decimation
    hip_period_time = float(_WALK_HIP_PERIOD) * ctrl_dt
    phase = (data.time / hip_period_time) % 1.0  # (N,)
    return phase.unsqueeze(-1)  # (N, 1)


def _walk_obs_muscle_length(env) -> torch.Tensor:
    """Actuator (muscle) lengths. Shape: (N, nu).

    Matches ``WalkEnvV0.muscle_lengths()`` → ``mj_data.actuator_length``.
    """
    data = env.scene[_WALK_ENTITY_NAME].data.data
    return data.actuator_length  # (N, nu)


def _walk_obs_muscle_velocity(env) -> torch.Tensor:
    """Actuator (muscle) velocities, clipped to [-100, 100]. Shape: (N, nu).

    Matches ``WalkEnvV0.muscle_velocities()`` → ``clip(actuator_velocity, -100, 100)``.
    """
    import torch

    data = env.scene[_WALK_ENTITY_NAME].data.data
    return torch.clamp(data.actuator_velocity, -100.0, 100.0)  # (N, nu)


def _walk_obs_muscle_force(env) -> torch.Tensor:
    """Actuator (muscle) forces / 1000, clipped to [-100, 100]. Shape: (N, nu).

    Matches ``WalkEnvV0.muscle_forces()`` → ``clip(actuator_force / 1000, -100, 100)``.
    """
    import torch

    data = env.scene[_WALK_ENTITY_NAME].data.data
    return torch.clamp(data.actuator_force / 1000.0, -100.0, 100.0)  # (N, nu)


def _walk_obs_act(env) -> torch.Tensor:
    """Muscle activation state. Shape: (N, na).

    Matches ``WalkEnvV0.get_obs_dict`` → ``mj_data.act`` (when ``mj_model.na > 0``).
    """
    data = env.scene[_WALK_ENTITY_NAME].data.data
    return data.act  # (N, na)


# ---------------------------------------------------------------------------
# Walk reward functions — matching WalkEnvV0 (CPU) and MjxWalkEnv (MJX)
#
# Reward components and weights (from WalkEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS):
#   vel_reward      ×  5.0  — forward velocity match (exponential decay)
#   done            × -100  — termination penalty (height or rotation exceeded)
#   cyclic_hip      × -10   — hip flexion gait periodicity
#   ref_rot         ×  10.0 — torso quaternion tracking
#   joint_angle_rew ×   5.0 — hip adduction/rotation penalty
# ---------------------------------------------------------------------------


def _walk_vel_reward(env, target_y_vel: float = 1.2, target_x_vel: float = 0.0):
    """Forward/lateral velocity reward using exponential decay, matching WalkEnvV0.

    CPU formula (walk_v0.py:375-380):
        exp(-(target_y - com_vel_y)²) + exp(-(target_x - com_vel_x)²)

    Uses mass-weighted COM velocity (not root qvel) for correct parity with CPU.
    Shape: (N,)
    """
    import torch  # noqa: PLC0415

    ids = _resolve_walk_obs_ids(env)
    data = env.scene[_WALK_ENTITY_NAME].data.data
    body_mass = ids["body_mass"]  # (nbody,)
    cvel_lin = -data.cvel[:, :, 3:5]  # (N, nbody, 2), sign per MuJoCo convention
    com_vel = (body_mass[None, :, None] * cvel_lin).sum(
        dim=1
    ) / body_mass.sum()  # (N, 2)
    return torch.exp(-torch.square(target_y_vel - com_vel[:, 1])) + torch.exp(
        -torch.square(target_x_vel - com_vel[:, 0])
    )  # (N,)


def _walk_forward_vel_reward(
    env,
    target_vel: float = 1.2,
    target_x_vel: float = 0.0,
):
    """Compatibility wrapper for forward-velocity reward term naming.

    Kept as a thin alias so benchmark invariants can assert the canonical
    forward-velocity reward hook while preserving the existing WalkEnvV0-style
    implementation in :func:`_walk_vel_reward`.
    """
    return _walk_vel_reward(env, target_y_vel=target_vel, target_x_vel=target_x_vel)


def _walk_done_signal(env):
    """Termination signal: 1.0 if height < min_height or rotation exceeded, else 0.

    CPU formula (walk_v0.py:346-351):
        done = 1 if height < min_height or |quat2mat(qpos[3:7])[0,0]| > max_rot

    Weight -100 in dense reward.  Shape: (N,)
    """
    import torch  # noqa: PLC0415

    ids = _resolve_walk_obs_ids(env)
    data = env.scene[_WALK_ENTITY_NAME].data.data
    min_height = ids["min_height"]
    max_rot = ids["max_rot"]

    # Height condition: mass-weighted COM z-height < min_height
    body_mass = ids["body_mass"]
    height = (body_mass[None, :] * data.xipos[:, :, 2]).sum(
        dim=1
    ) / body_mass.sum()  # (N,)
    done_height = (height < min_height).to(dtype=torch.float32)

    # Rotation condition: |R[0,0]| > max_rot  where R = quat2mat(qpos[3:7])
    # R[0,0] = 1 - 2*(qy² + qz²)
    qy = data.qpos[:, 5]
    qz = data.qpos[:, 6]
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    done_rot = (torch.abs(r00) > max_rot).to(dtype=torch.float32)

    return torch.maximum(done_height, done_rot)  # (N,)


def _walk_alive_reward(env, fall_height_threshold: float = 0.8):
    """Alive reward term: 1 when COM height is above threshold, else 0.

    This term complements the done penalty and matches the benchmark invariant
    expectation that walk reward shaping includes explicit uprightness signal.
    """
    import torch  # noqa: PLC0415

    ids = _resolve_walk_obs_ids(env)
    data = env.scene[_WALK_ENTITY_NAME].data.data
    body_mass = ids["body_mass"]
    height = (body_mass[None, :] * data.xipos[:, :, 2]).sum(dim=1) / body_mass.sum()
    return (height >= fall_height_threshold).to(dtype=torch.float32)


def _walk_act_reg(env):
    """Action regularization on muscle activations (mean L2 per env)."""
    import torch  # noqa: PLC0415

    data = env.scene[_WALK_ENTITY_NAME].data.data
    return torch.mean(torch.square(data.act), dim=1)


def _walk_cyclic_hip(env):
    """Cyclic hip gait reward: L2 distance from desired sinusoidal hip trajectory.

    CPU formula (walk_v0.py:382-392):
        phase = (steps / hip_period) % 1
        des = [0.8 * cos(phase*2π + π), 0.8 * cos(phase*2π)]
        reward = ‖des − [hip_flex_l, hip_flex_r]‖₂

    Weight -10 (penalises deviation from gait pattern).  Shape: (N,)
    """
    import math

    import torch  # noqa: PLC0415

    ids = _resolve_walk_obs_ids(env)
    data = env.scene[_WALK_ENTITY_NAME].data.data
    ctrl_dt = env.physics_dt * env.cfg.decimation
    hip_period_time = float(_WALK_HIP_PERIOD) * ctrl_dt
    phase = (data.time / hip_period_time) % 1.0  # (N,)

    des_l = 0.8 * torch.cos(phase * 2.0 * math.pi + math.pi)  # (N,)
    des_r = 0.8 * torch.cos(phase * 2.0 * math.pi)  # (N,)

    hip_flex_l = data.qpos[:, ids["hip_flex_l_adr"]]  # (N,)
    hip_flex_r = data.qpos[:, ids["hip_flex_r_adr"]]  # (N,)

    diff = torch.stack([des_l - hip_flex_l, des_r - hip_flex_r], dim=1)  # (N, 2)
    return torch.linalg.norm(diff, dim=1)  # (N,)


def _walk_ref_rot(env):
    """Torso orientation reward: exponential decay from initial standing quaternion.

    CPU formula (walk_v0.py:394-400):
        ref_rot = exp(-‖5 * (qpos[3:7] − target_rot)‖₂)

    target_rot = key_qpos[2][3:7] (keyframe 2 = standing "init" pose),
    matching MjxWalkEnv.sample_task() and WalkEnvV0 with reset_type="init".
    Shape: (N,)
    """
    import torch  # noqa: PLC0415

    ids = _resolve_walk_obs_ids(env)
    data = env.scene[_WALK_ENTITY_NAME].data.data
    target_rot = ids["target_rot"]  # (4,) on correct device

    quat = data.qpos[:, 3:7]  # (N, 4)
    diff = 5.0 * (quat - target_rot[None, :])  # (N, 4)
    return torch.exp(-torch.linalg.norm(diff, dim=1))  # (N,)


def _walk_joint_angle_rew(env):
    """Hip adduction/rotation penalty: exp(-5 * mean(|angles|)).

    CPU formula (walk_v0.py:353-355):
        joint_angle_rew = exp(-5 * mean(|[hip_adduct_l, hip_adduct_r, hip_rot_l, hip_rot_r]|))

    Weight 5.0 (penalises unnatural lateral/rotational hip motion).  Shape: (N,)
    """
    import torch  # noqa: PLC0415

    ids = _resolve_walk_obs_ids(env)
    data = env.scene[_WALK_ENTITY_NAME].data.data

    hip_angles = torch.stack(
        [
            data.qpos[:, ids["hip_adduct_l_adr"]],
            data.qpos[:, ids["hip_adduct_r_adr"]],
            data.qpos[:, ids["hip_rot_l_adr"]],
            data.qpos[:, ids["hip_rot_r_adr"]],
        ],
        dim=1,
    )  # (N, 4)
    return torch.exp(-5.0 * torch.mean(torch.abs(hip_angles), dim=1))  # (N,)


def _elbow_spec_fn():
    import mujoco

    return mujoco.MjSpec.from_file(str(_ELBOW_XML))


def _walk_spec_fn():
    """Load the leg model MJCF for mjlab and strip source keyframes.

    The upstream ``myolegs`` MJCF includes multiple anonymous keyframes. When
    composed into an mjlab ``Scene`` as an ``Entity``, these extra keyframes
    interact poorly with ``Scene._add_entities``'s keyframe merging logic and
    can trigger a MuJoCo ``ValueError`` about repeated names in the keyframe
    table.

    For mjlab we don't rely on those original keyframes: ``Entity`` will create
    its own single ``init_state`` keyframe from ``InitialStateCfg``. To avoid
    the duplicate-name issue, we drop all keyframes from the raw spec before
    handing it to mjlab.
    """
    import mujoco

    spec = mujoco.MjSpec.from_file(str(_WALK_XML))
    # Remove all existing keyframes; Entity will add its own "init_state".
    for k in list(spec.keys):
        spec.delete(k)
    return spec


def _walk_sarc_spec_fn():
    """Build walk spec with sarcopenia transform applied for mjlab variants."""
    return apply_sarcopenia_to_spec(_walk_spec_fn(), force_scale=0.5)


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
        # See _walk_ppo_runner_cfg: map actor/critic to the env's "policy" group.
        obs_groups={"actor": ("policy",), "critic": ("policy",)},
    )


def _make_walk_env_cfg(
    play: bool = False, muscle_condition: str = ""
) -> ManagerBasedRlEnvCfg:
    """ManagerBasedRlEnvCfg for myoLegWalk-v0 (80-muscle bipedal walking).

    Uses the myo_sim leg MJCF (``myolegs_mjx.xml`` when present, else
    ``myolegs.xml``). The plane-terrain MJX variant avoids MuJoCo Warp issues
    with height-field terrain while keeping the same muscle/torso/leg chain as
    the CPU walk task.

    Observations: projected gravity vector (3D) — a minimal but stable obs
    that avoids joint-id indexing issues across mjlab versions.

    Actions: TendonLengthActionCfg for all 80 leg muscles, control in [0, 1].

    Terminations: time-out only (falling is handled by the RL reward shaping).
    """
    if not _WALK_XML.exists():
        raise FileNotFoundError(f"Leg walk model not found: {_WALK_XML}")

    muscle_names = _walk_muscle_names()
    walk_entity_name = "walk_robot"
    walk_spec_fn = (
        _walk_sarc_spec_fn if muscle_condition == "sarcopenia" else _walk_spec_fn
    )

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "qpos_without_xy": ObservationTermCfg(func=_walk_obs_qpos_without_xy),
                "qvel": ObservationTermCfg(func=_walk_obs_qvel),
                "com_vel": ObservationTermCfg(func=_walk_obs_com_vel),
                "torso_angle": ObservationTermCfg(func=_walk_obs_torso_angle),
                "feet_heights": ObservationTermCfg(func=_walk_obs_feet_heights),
                "height": ObservationTermCfg(func=_walk_obs_height),
                "feet_rel_positions": ObservationTermCfg(
                    func=_walk_obs_feet_rel_positions
                ),
                "phase_var": ObservationTermCfg(func=_walk_obs_phase_var),
                "muscle_length": ObservationTermCfg(func=_walk_obs_muscle_length),
                "muscle_velocity": ObservationTermCfg(func=_walk_obs_muscle_velocity),
                "muscle_force": ObservationTermCfg(func=_walk_obs_muscle_force),
                "act": ObservationTermCfg(func=_walk_obs_act),
            },
        ),
    }

    actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=walk_entity_name,
            actuator_names=muscle_names,
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp_terminations.time_out,
            time_out=True,
        ),
    }
    from mjlab.managers.event_manager import EventTermCfg

    events = {
        # Without an explicit reset event, mjlab falls back to its own
        # generic default (root at the origin, all joints at 0) at every
        # episode boundary after the first -- see _directional_init_state()
        # and the "reset_scene_to_default" comment on the directional env
        # config, which hit and fixed the identical bug for the same host
        # model. This mirrors that fix for myoLegWalk-v0.
        "reset_scene_to_default": EventTermCfg(
            func=mdp_events.reset_scene_to_default, mode="reset"
        ),
    }

    from mjlab.managers.reward_manager import RewardTermCfg

    walk_cfg = WalkCfg()

    rewards = {
        "vel_reward": RewardTermCfg(
            func=_walk_forward_vel_reward,
            weight=5.0,
            params={"target_vel": float(walk_cfg.target_vel), "target_x_vel": 0.0},
        ),
        "alive_reward": RewardTermCfg(
            func=_walk_alive_reward,
            weight=float(walk_cfg.alive_bonus),
            params={"fall_height_threshold": float(walk_cfg.fall_height_threshold)},
        ),
        "done": RewardTermCfg(
            func=_walk_done_signal,
            weight=-100.0,
        ),
        "cyclic_hip": RewardTermCfg(
            func=_walk_cyclic_hip,
            weight=-10.0,
        ),
        "ref_rot": RewardTermCfg(
            func=_walk_ref_rot,
            weight=10.0,
        ),
        "joint_angle_rew": RewardTermCfg(
            func=_walk_joint_angle_rew,
            weight=5.0,
        ),
        "act_reg": RewardTermCfg(
            func=_walk_act_reg,
            weight=-float(walk_cfg.act_reg_weight),
        ),
    }

    return mjlab_env_cfg_from_task_config(
        cfg=TaskConfig(max_episode_steps=1000),
        spec_fn=walk_spec_fn,
        entity_name=walk_entity_name,
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=tuple(f"{name}_tendon" for name in muscle_names),
                transmission_type=TransmissionType.TENDON,
            ),
        ),
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        num_envs=1,
        decimation=10,
        sim_cfg=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.002,
                ccd_iterations=500,
            ),
            # MuJoCo-Warp pre-allocates constraint buffers of these sizes. The
            # default is too small for the contact-rich biped under RL
            # exploration (nefc overflow -> NaN obs during training); size them
            # like the other contact-heavy mjlab tasks. A max only, so CPU/mjlab
            # parity is unaffected.
            njmax=512,
            nconmax=256,
        ),
        episode_length_s=20.0,
        init_state=_directional_init_state(),  # stand at ~1.0 m; same host XML as directional
    )


def _walk_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO runner config for the bipedal walk benchmark."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
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
        experiment_name="myo_leg_walk",
        save_interval=100,
        num_steps_per_env=48,
        max_iterations=500,
        # rsl_rl maps the actor/critic obs sets to the env's observation
        # group(s); the walk env exposes a single flat "policy" group. Without
        # this, RslRlOnPolicyRunner raises "Observation 'actor' not found".
        obs_groups={"actor": ("policy",), "critic": ("policy",)},
    )


# ---------------------------------------------------------------------------
# Directional myoLeg locomotion — GPU (mjlab) match for the CPU
# myoLegDirectional{Forward,Backward}-v0 TaskConfig envs. Obs/reward mirror
# myosuite/terms/base_obs.py + base_reward.py (root_planar_vel, heading_cmd,
# heading_reward) so a policy trained on GPU transfers to the CPU env.
# ---------------------------------------------------------------------------
_DIRECTIONAL_FALL_HEIGHT = 0.7  # _HEADING_FALL_HEIGHT in base_reward.py
_DIRECTIONAL_FALL_PENALTY = 1.0  # _HEADING_FALL_PENALTY in base_reward.py


def _directional_obs_joint_pos(env):
    """Full free-joint qpos (N, nq) — matches accessor.joint_pos()."""
    return env.scene[_WALK_ENTITY_NAME].data.data.qpos


def _directional_obs_joint_vel(env):
    """Raw qvel (N, nv) — matches accessor.joint_vel() (unscaled)."""
    return env.scene[_WALK_ENTITY_NAME].data.data.qvel


def _directional_obs_muscle_act(env):
    """Muscle activation state (N, 80) — matches accessor.muscle_act()."""
    return env.scene[_WALK_ENTITY_NAME].data.data.act


def _directional_obs_root_planar_vel(env):
    """Root free-joint planar velocity (N, 2) — matches joint_vel()[:2]."""
    return env.scene[_WALK_ENTITY_NAME].data.data.qvel[:, :2]


def _directional_obs_heading_cmd(env, heading_dir: tuple[float, float] = (0.0, 1.0)):
    """Constant commanded heading direction (N, 2) — matches heading_cmd_obs."""
    import torch  # noqa: PLC0415

    data = env.scene[_WALK_ENTITY_NAME].data.data
    n = data.qpos.shape[0]
    return torch.tensor(
        heading_dir, dtype=torch.float32, device=data.qpos.device
    ).expand(n, 2)


def _directional_heading_tracking(
    env, heading_dir: tuple[float, float] = (0.0, 1.0), target_speed: float = 1.2
):
    """exp(-||target_speed*heading_dir - planar_vel||^2) (N,) — heading_reward tracking."""
    import torch  # noqa: PLC0415

    data = env.scene[_WALK_ENTITY_NAME].data.data
    planar_vel = data.qvel[:, :2]
    direction = torch.tensor(heading_dir, dtype=torch.float32, device=planar_vel.device)
    target_vel = target_speed * direction
    return torch.exp(-torch.sum((target_vel - planar_vel) ** 2, dim=1))


# --- Command-randomized directional variant (GPU match for myoLegDirectionalRandom-v0) ---
# A per-env commanded heading is resampled from the full unit circle at each
# reset and stored on ``env._directional_cmd``. Both the heading_cmd observation
# and the heading-tracking reward read this buffer, so a policy trained here
# learns the command->direction mapping (the fix for the chase-tag steering null
# result). Mirrors the CPU ``LegDirectionalRandomTask`` (randomize_heading=True).


def _directional_cmd_buffer(env):
    """Lazily create / return the per-env commanded-heading buffer ``(N, 2)``."""
    import torch  # noqa: PLC0415

    data = env.scene[_WALK_ENTITY_NAME].data.data
    n = data.qpos.shape[0]
    buf = getattr(env, "_directional_cmd", None)
    if buf is None or buf.shape[0] != n:
        buf = torch.zeros(n, 2, dtype=torch.float32, device=data.qpos.device)
        buf[:, 1] = 1.0  # default forward until the first reset populates it
        env._directional_cmd = buf  # noqa: SLF001
    return env._directional_cmd


def _directional_reset_heading(env, env_ids=None, **_):
    """Reset event: resample a full-circle unit heading for the reset envs."""
    import math  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from myosuite.envs.myo.backends.mjlab.mjlab_env_base import (  # noqa: PLC0415
        normalize_mjlab_env_ids,
    )

    buf = _directional_cmd_buffer(env)
    idx = normalize_mjlab_env_ids(env, env_ids)
    if idx.numel() == 0:
        return
    angles = torch.rand(idx.numel(), device=buf.device) * (2.0 * math.pi)
    buf[idx, 0] = torch.cos(angles)
    buf[idx, 1] = torch.sin(angles)


def _directional_obs_heading_cmd_rand(env):
    """Per-env commanded heading ``(N, 2)`` from the reset-sampled buffer."""
    return _directional_cmd_buffer(env)


def _directional_heading_tracking_rand(env, target_speed: float = 1.2):
    """Heading tracking using the per-env sampled command buffer ``(N,)``."""
    import torch  # noqa: PLC0415

    data = env.scene[_WALK_ENTITY_NAME].data.data
    planar_vel = data.qvel[:, :2]
    target_vel = target_speed * _directional_cmd_buffer(env)
    return torch.exp(-torch.sum((target_vel - planar_vel) ** 2, dim=1))


def _directional_fallen_bool(env, fall_height: float = _DIRECTIONAL_FALL_HEIGHT):
    """Bool (N,): pelvis height (qpos[2]) < fall_height. For the TerminationManager."""
    data = env.scene[_WALK_ENTITY_NAME].data.data
    return data.qpos[:, 2] < fall_height


def _directional_fallen(env, fall_height: float = _DIRECTIONAL_FALL_HEIGHT):
    """1.0 where fallen else 0.0 (N,) float. For the RewardManager fall penalty."""
    import torch  # noqa: PLC0415

    return _directional_fallen_bool(env, fall_height).to(dtype=torch.float32)


def _directional_alive_reward(env, fall_height: float = _DIRECTIONAL_FALL_HEIGHT):
    """Alive reward: 1.0 while the pelvis is above the fall height, else 0.0.

    Mirrors ``_walk_alive_reward`` (myoLegWalk-v0's working pattern for this
    exact class of problem): a small constant per-step bonus for staying
    upright gives the policy an immediate, dense reward signal for surviving
    longer, on top of the sparser heading-tracking term. Without it, an
    early/undertrained policy that can't yet balance the crouched standing
    pose gets no reward difference between falling at step 20 vs step 150,
    so there's nothing pushing episode length to grow during the hardest
    part of exploration (confirmed empirically: episode length was pinned
    flat at ~101/500 steps for hundreds of iterations even with the reset
    pose itself confirmed correct).
    """
    import torch  # noqa: PLC0415

    return (~_directional_fallen_bool(env, fall_height)).to(dtype=torch.float32)


def _directional_init_state():
    """Standing ``InitialStateCfg`` from the leg model's keyframe-0 pose.

    mjlab's default init places a floating-base robot's root at the origin, i.e.
    in the ground (pelvis height 0). ``m.qpos0`` (the XML's raw joint defaults)
    is *not* a standing pose -- it is all zeros, since none of this model's
    joints declare a nonzero ``ref``. The CPU ``myoLegDirectional*-v0`` env
    resets to keyframe 0 (see ``ModularTaskEnv.reset()`` in
    ``myosuite/envs/modular_env.py``), which every bundled leg host XML ships
    as a real crouched-standing pose (pelvis ~0.92 m, nonzero hip/knee
    angles). Reading ``qpos0`` here instead of that keyframe reproduced the
    same "root at the origin" bug on GPU that CPU had before its own fix --
    matching keyframe 0 (not qpos0) on both backends is what actually lets
    the GPU policy learn to walk from a real standing pose instead of
    collapsing/crawling off the floor every episode.
    """
    import mujoco  # noqa: PLC0415
    from mjlab.entity import EntityCfg

    import re  # noqa: PLC0415

    m = mujoco.MjModel.from_xml_path(str(_WALK_XML))
    q0 = m.key_qpos[0] if m.nkey > 0 else m.qpos0
    pos = tuple(float(x) for x in q0[:3])
    rot = tuple(float(x) for x in q0[3:7])
    joint_pos: dict[str, float] = {}
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name:
            continue
        if int(m.jnt_type[j]) in (
            int(mujoco.mjtJoint.mjJNT_SLIDE),
            int(mujoco.mjtJoint.mjJNT_HINGE),
        ):
            # mjlab's Entity.resolve_expr matches these keys as regex
            # patterns via re.match (prefix-anchored, not a full match) --
            # an unescaped literal name like "knee_angle_r" would also match
            # "knee_angle_rotation2_r" as a prefix and silently steal its
            # value (confirmed empirically: without the anchor, several
            # joints sharing a name prefix all resolved to the same wrong
            # value). Anchor with ^...$ so each key matches only its exact
            # joint.
            joint_pos[f"^{re.escape(name)}$"] = float(q0[int(m.jnt_qposadr[j])])
    return EntityCfg.InitialStateCfg(pos=pos, rot=rot, joint_pos=joint_pos)


def _make_directional_env_cfg(
    heading_dir: tuple[float, float],
    target_speed: float,
    randomize_heading: bool = False,
) -> ManagerBasedRlEnvCfg:
    """ManagerBasedRlEnvCfg for directional myoLeg locomotion (GPU match).

    Obs = [joint_pos(nq), joint_vel(nv), muscle_act(80), root_planar_vel(2),
    heading_cmd(2)] matching the CPU ``myoLegDirectional*-v0`` obs. Reward =
    heading velocity tracking − fall penalty − act regularisation.

    Args:
        heading_dir: Fixed commanded direction (used when not randomizing).
        target_speed: Commanded speed (m/s).
        randomize_heading: If True, the commanded heading is resampled from the
            full unit circle per env at each reset (mirrors the CPU
            ``myoLegDirectionalRandom-v0``); the heading_cmd obs and tracking
            reward read the per-env buffer. This teaches the command->direction
            mapping needed for chase-tag steering transfer.
    """
    if not _WALK_XML.exists():
        raise FileNotFoundError(f"Leg model not found: {_WALK_XML}")

    muscle_names = _walk_muscle_names()
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg

    if randomize_heading:
        heading_cmd_term = ObservationTermCfg(func=_directional_obs_heading_cmd_rand)
        heading_reward_term = RewardTermCfg(
            func=_directional_heading_tracking_rand,
            weight=1.0,
            params={"target_speed": float(target_speed)},
        )
    else:
        heading_cmd_term = ObservationTermCfg(
            func=_directional_obs_heading_cmd,
            params={"heading_dir": tuple(heading_dir)},
        )
        heading_reward_term = RewardTermCfg(
            func=_directional_heading_tracking,
            weight=1.0,
            params={
                "heading_dir": tuple(heading_dir),
                "target_speed": float(target_speed),
            },
        )

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "joint_pos": ObservationTermCfg(func=_directional_obs_joint_pos),
                "joint_vel": ObservationTermCfg(func=_directional_obs_joint_vel),
                "muscle_act": ObservationTermCfg(func=_directional_obs_muscle_act),
                "root_planar_vel": ObservationTermCfg(
                    func=_directional_obs_root_planar_vel
                ),
                "heading_cmd": heading_cmd_term,
            },
        ),
    }
    actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=_WALK_ENTITY_NAME,
            actuator_names=muscle_names,
        ),
    }
    rewards = {
        "heading_tracking": heading_reward_term,
        "fall_penalty": RewardTermCfg(
            func=_directional_fallen,
            weight=-_DIRECTIONAL_FALL_PENALTY,
        ),
        # See _directional_alive_reward's docstring: borrowed from
        # myoLegWalk-v0's working alive_bonus pattern (WalkCfg.alive_bonus =
        # 0.2) to give a dense per-step survival signal while the policy is
        # still too undertrained to balance the crouched standing pose.
        "alive_reward": RewardTermCfg(func=_directional_alive_reward, weight=0.2),
        "act_reg": RewardTermCfg(func=_walk_act_reg, weight=-0.1),
    }
    # Fall termination matches the CPU env (heading_reward returns done=fallen).
    # Safe now that init_state starts the pelvis standing at ~0.92 m (well
    # above the 0.7 m fall height), so it only fires on a genuine fall — not
    # step 1.
    terminations = {
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
        "fallen": TerminationTermCfg(func=_directional_fallen_bool),
    }
    events = {
        # init_state (see _directional_init_state()) only seeds the entity's
        # spec-level "init_state" keyframe used to build the compiled model's
        # default_root_state/default_joint_pos; it is NOT automatically
        # reapplied to data.qpos at each episode boundary. Without this
        # event, every reset after the very first falls back to whatever
        # mjlab's own generic default is (confirmed empirically: pelvis
        # z=1.0, all joint angles 0 -- not our standing pose at all), so the
        # leg was silently resetting to the same broken "pedestal" pose this
        # whole fix was meant to solve. This mirrors every mjlab example
        # task (cartpole, velocity, manipulation, tracking), which all
        # register a mode="reset" event to actually apply entity defaults.
        "reset_scene_to_default": EventTermCfg(
            func=mdp_events.reset_scene_to_default, mode="reset"
        ),
    }
    if randomize_heading:
        events["sample_heading"] = EventTermCfg(
            func=_directional_reset_heading, mode="reset"
        )
    return mjlab_env_cfg_from_task_config(
        cfg=TaskConfig(max_episode_steps=500),
        spec_fn=_walk_spec_fn,
        entity_name=_WALK_ENTITY_NAME,
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=tuple(f"{name}_tendon" for name in muscle_names),
                transmission_type=TransmissionType.TENDON,
            ),
        ),
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        num_envs=1,
        decimation=5,  # CPU BackendConfig(n_substeps=5, ctrl_dt=0.01, sim_dt=0.002)
        sim_cfg=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.002, ccd_iterations=500),
            njmax=512,
            nconmax=256,
        ),
        episode_length_s=20.0,  # matches myoLegWalk-v0's proven horizon (was 5.0)
        init_state=_directional_init_state(),  # stand at ~1.0 m, matching CPU reset
    )


def _directional_ppo_runner_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    """PPO runner config for directional myoLeg locomotion."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
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
        # Mandatory for every leg-family full-body GPU env on this branch —
        # without it, resets after the first fall back to mjlab's broken
        # generic default pose (see _directional_init_state's docstring).
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
        decimation=5,  # matches _walk_spec_fn/_directional cfgs' proven GPU decimation
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


# Directional variants: (env_id, heading_dir, target_speed) — mirror the CPU
# LegDirectional{Forward,Backward}Task specs.
_DIRECTIONAL_VARIANTS = (
    ("myoLegDirectionalForward-v0", (0.0, 1.0), 1.2, "myo_leg_directional_fwd", False),
    (
        "myoLegDirectionalBackward-v0",
        (0.0, -1.0),
        1.0,
        "myo_leg_directional_bwd",
        False,
    ),
    ("myoLegDirectionalRandom-v0", (0.0, 1.0), 1.2, "myo_leg_directional_rand", True),
)


def register_mjlab_tasks() -> None:
    """Register MyoSuite env ids with mjlab.tasks.registry. Idempotent."""

    # --- Elbow pose ---
    if _ELBOW_XML.exists():
        try:
            env_cfg = _make_elbow_env_cfg(play=False)
            play_cfg = _make_elbow_env_cfg(play=True)
            rl_cfg = _elbow_ppo_runner_cfg()
            register_mjlab_task(
                task_id="myoElbowPose1D6MFixed-v0",
                env_cfg=env_cfg,
                play_env_cfg=play_cfg,
                rl_cfg=rl_cfg,
                runner_cls=None,
            )
        except ValueError:
            pass  # already registered
        except Exception:
            logging.getLogger(__name__).warning(
                "mjlab: failed to register myoElbowPose1D6MFixed-v0", exc_info=True
            )

    # --- Leg walk ---
    if _WALK_XML.exists():
        try:
            walk_env_cfg = _make_walk_env_cfg(play=False)
            walk_play_cfg = _make_walk_env_cfg(play=True)
            walk_rl_cfg = _walk_ppo_runner_cfg()
            register_mjlab_task(
                task_id="myoLegWalk-v0",
                env_cfg=walk_env_cfg,
                play_env_cfg=walk_play_cfg,
                rl_cfg=walk_rl_cfg,
                runner_cls=None,
            )
        except ValueError:
            pass  # already registered
        except Exception:
            logging.getLogger(__name__).warning(
                "mjlab: failed to register myoLegWalk-v0", exc_info=True
            )

    # --- Directional myoLeg locomotion (GPU match for CPU TaskConfig envs) ---
    if _WALK_XML.exists():
        for (
            env_id,
            heading_dir,
            target_speed,
            exp_name,
            randomize,
        ) in _DIRECTIONAL_VARIANTS:
            try:
                dir_env_cfg = _make_directional_env_cfg(
                    heading_dir, target_speed, randomize_heading=randomize
                )
                dir_rl_cfg = _directional_ppo_runner_cfg(exp_name)
                register_mjlab_task(
                    task_id=env_id,
                    env_cfg=dir_env_cfg,
                    play_env_cfg=dir_env_cfg,
                    rl_cfg=dir_rl_cfg,
                    runner_cls=None,
                )
            except ValueError:
                pass  # already registered
            except Exception:
                logging.getLogger(__name__).warning(
                    "mjlab: failed to register %s", env_id, exc_info=True
                )

    if _WALK_XML.exists():
        try:
            sarc_walk_env_cfg = _make_walk_env_cfg(
                play=False, muscle_condition="sarcopenia"
            )
            sarc_walk_play_cfg = _make_walk_env_cfg(
                play=True, muscle_condition="sarcopenia"
            )
            sarc_walk_rl_cfg = _walk_ppo_runner_cfg()
            register_mjlab_task(
                task_id="myoSarcLegWalk-v0",
                env_cfg=sarc_walk_env_cfg,
                play_env_cfg=sarc_walk_play_cfg,
                rl_cfg=sarc_walk_rl_cfg,
                runner_cls=None,
            )
        except ValueError:
            pass  # already registered
        except Exception:
            logging.getLogger(__name__).warning(
                "mjlab: failed to register myoSarcLegWalk-v0", exc_info=True
            )

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


def _register_optional_myouser_task(myouser_config: Any) -> None:
    """Register the optional myouser mjlab task for an explicit config."""
    from myosuite.envs.myo.backends.mjlab.register_mjlab_myouser_tasks import (
        register_mjlab_myouser_task,
    )

    register_mjlab_myouser_task(myouser_config)


def bootstrap_myosuite_mjlab_registry(
    *,
    clip_path: str | os.PathLike[str] | None = None,
    myouser_config: Any | None = None,
    rl_cfg_fn: Callable[[], Any] | None = None,
    use_lookahead: bool = True,
) -> None:
    """Register all MyoSuite mjlab tasks in one call (idempotent).

    Runs :func:`register_mjlab_tasks` (static envs). Plain ``import mjlab`` stays
    limited to MyoSuite's own mjlab tasks and does not import the optional
    ``myouser`` package anymore.

    To register the optional ``myoUserUniversal-v0`` mjlab task, pass an
    explicit ``myouser_config`` object compatible with
    :func:`register_mjlab_myouser_task`.

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
    if myouser_config is not None:
        _register_optional_myouser_task(myouser_config)

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
