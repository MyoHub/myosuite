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
from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
    MyoMuscleActivationActionCfg,
    mjlab_env_cfg_from_task_config,
)
from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
    default_mimic_clip_on_policy_runner_cfg,
)
from myosuite.envs.myo.backends.mjlab.register_mjlab_saber import (
    register_saber_p0_mjlab_task,
)
from myosuite.envs.myo.backends.mjlab.register_mjlab_boxing import (
    register_boxing_p0_mjlab_task,
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
    try:
        import myo_sim  # type: ignore[import-untyped]

        if hasattr(myo_sim, "MODELS_DIR") and (myo_sim.MODELS_DIR / "leg").exists():
            return myo_sim.MODELS_DIR / "leg"
    except ImportError:
        pass
    root = _resolve_model_root()
    return root / "simhive" / "myo_sim" / "leg"


_MYOSUITE_ROOT = _resolve_model_root()
_ELBOW_XML = _resolve_elbow_xml("myoelbow_1dof6muscles.xml")
# Prefer the plane-terrain leg model for mjlab: MuJoCo Warp forward on hfield
# terrain has been unreliable (including native crashes) on some platforms.
_LEG_DIR = _resolve_leg_dir()
_WALK_XML_MJX = _LEG_DIR / "myolegs_mjx.xml"
_WALK_XML = _WALK_XML_MJX if _WALK_XML_MJX.is_file() else _LEG_DIR / "myolegs.xml"


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
    )


def _make_walk_env_cfg(
    play: bool = False, muscle_condition: str = ""
) -> ManagerBasedRlEnvCfg:
    """ManagerBasedRlEnvCfg for myoLegWalk-v0 (80-muscle bipedal walking).

    Uses the simhive leg MJCF (``myolegs_mjx.xml`` when present, else
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
        num_envs=1,
        decimation=10,
        sim_cfg=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.002,
                ccd_iterations=500,
            ),
        ),
        episode_length_s=20.0,
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

    # --- Saber P0 ---
    try:
        register_saber_p0_mjlab_task(
            register_mjlab_task=register_mjlab_task,
            rl_cfg_fn=default_mimic_clip_on_policy_runner_cfg,
        )
    except ValueError:
        pass  # already registered
    except Exception:
        logging.getLogger(__name__).warning(
            "mjlab: failed to register myoChallengeSaberP0-v0", exc_info=True
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

    # --- Boxing P0 ---
    try:
        register_boxing_p0_mjlab_task(
            register_mjlab_task=register_mjlab_task,
            rl_cfg_fn=default_mimic_clip_on_policy_runner_cfg,
        )
    except ValueError:
        pass  # already registered
    except Exception:
        logging.getLogger(__name__).warning(
            "mjlab: failed to register myoChallengeBoxingP0Mjlab-v0", exc_info=True
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
        default_mimic_clip_on_policy_runner_cfg,
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
