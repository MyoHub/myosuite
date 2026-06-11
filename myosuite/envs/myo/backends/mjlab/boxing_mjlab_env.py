# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab bridge functions for the BoxingP0 task (6-pad static target variant).

All kinematic observations are computed directly from vectorised Warp entity data
so each of the N parallel envs gets its own state.  Contact-based pad observations
use ``env.sim.mj_data`` (env-0 only) until a vectorised contact API is available
from mjlab — see TODO comment in ``_boxing_target_pad_obs``.

No env subclassing — this module provides thin bridge callables and a
``BoxingTaskState`` ManagerTermBase that caches MuJoCo ids at startup.
"""

from __future__ import annotations

from typing import Any

import torch
from mjlab.managers import ManagerTermBase
from mjlab.managers.event_manager import requires_model_fields

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import (
    normalize_mjlab_env_ids as _normalize_env_ids,
)

_BOXING_ENTITY_NAME = "boxing_p0_robot"


# ---------------------------------------------------------------------------
# Task state holder
# ---------------------------------------------------------------------------


class BoxingTaskState(ManagerTermBase):
    """Caches per-scene MuJoCo ids needed by bridge callables.

    Stored on ``env._boxing_state`` so all bridge functions can retrieve it
    without rebuilding the model on every call.  The mannequin (agent_1) is
    static in P0 — its joints are held at zero by position actuators already
    in the model; no extra per-step drive is needed.
    """

    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(env)
        env._boxing_state = self  # noqa: SLF001
        self.low_level_cfg: Any = cfg.params["low_level_cfg"]
        self.meta: Any = cfg.params["meta"]

    def __call__(self, env: Any, **_: Any) -> None:
        """Step event hook — no per-step logic needed for the static P0 mannequin."""


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def _get_boxing_state(env: Any) -> BoxingTaskState:
    state = getattr(env, "_boxing_state", None)
    if state is None:
        raise RuntimeError(
            "BoxingTaskState not initialised. "
            "Ensure the step event is registered with mode='startup'."
        )
    return state


def _entity_data(env: Any) -> Any:
    """Return the mjlab entity data object for the boxing robot."""
    return env.scene[_BOXING_ENTITY_NAME].data


# ---------------------------------------------------------------------------
# Observation bridge functions
# ---------------------------------------------------------------------------


def _boxing_kinematics_obs(env: Any, **_: Any) -> torch.Tensor:
    """Joint qpos, qvel, and muscle activation for all N envs. Shape: (N, nq+nv+na)."""
    data = _entity_data(env)
    # entity.data exposes per-env tensors: joint_pos (N,nq), joint_vel (N,nv), joint_act (N,na)
    qpos = data.joint_pos.to(torch.float32)  # (N, nq)
    qvel = data.joint_vel.to(torch.float32)  # (N, nv)
    act = data.joint_act.to(torch.float32)  # (N, na)
    return torch.cat([qpos, qvel, act], dim=-1)


def _site_xpos(env: Any) -> torch.Tensor:
    return torch.as_tensor(
        env.sim.wp_data.site_xpos, device=env.device, dtype=torch.float32
    )


def _site_xvel(env: Any) -> torch.Tensor:
    """Return world-frame linear site velocity. Shape: (N, nsite, 3).

    ``site_xvelp`` was a mujoco-py artifact and does not exist in modern MuJoCo
    or mujoco-warp.  Site linear velocity is derived from ``cvel`` (com-based
    spatial velocity) following the same formula used by mjlab's
    ``EntityData.site_vel_w``:

        v_site = v_com - \u03c9 × (subtree_com - site_pos)

    where the first three components of the result are the linear velocity.
    """
    data = env.sim.wp_data
    model = env.sim.wp_model
    # site_xpos: (N, nsite, 3)
    site_pos = torch.as_tensor(data.site_xpos, device=env.device, dtype=torch.float32)
    # subtree_com: (N, nbody, 3)
    subtree_com = torch.as_tensor(
        data.subtree_com, device=env.device, dtype=torch.float32
    )
    # cvel: (N, nbody, 6) — spatial velocity (rot:lin) in body-com frame
    cvel = torch.as_tensor(data.cvel, device=env.device, dtype=torch.float32)
    # site_bodyid: (nsite,) — global body index for each site
    site_bodyid = torch.as_tensor(
        model.site_bodyid, device=env.device, dtype=torch.long
    )
    body_subtree_com = subtree_com[:, site_bodyid, :]  # (N, nsite, 3)
    body_cvel = cvel[:, site_bodyid, :]  # (N, nsite, 6)
    lin_vel_c = body_cvel[..., 3:6]  # (N, nsite, 3)
    ang_vel_c = body_cvel[..., 0:3]  # (N, nsite, 3)
    offset = body_subtree_com - site_pos  # (N, nsite, 3)
    return lin_vel_c - torch.linalg.cross(ang_vel_c, offset)


def _boxing_fist_obs(env: Any, **_: Any) -> torch.Tensor:
    """Fist site positions and linear velocities for all N envs. Shape: (N, 12)."""
    state = _get_boxing_state(env)
    sites = state.meta.site_ids["agent_0"]
    xpos = _site_xpos(env)
    xvel = _site_xvel(env)
    return torch.cat(
        [
            xpos[:, sites["fist_r"], :],  # (N, 3)
            xpos[:, sites["fist_l"], :],  # (N, 3)
            xvel[:, sites["fist_r"], :],  # (N, 3)
            xvel[:, sites["fist_l"], :],  # (N, 3)
        ],
        dim=-1,
    )


def _boxing_pelvis_obs(env: Any, **_: Any) -> torch.Tensor:
    """Pelvis site position and linear velocity for all N envs. Shape: (N, 6)."""
    state = _get_boxing_state(env)
    pid = state.meta.site_ids["agent_0"]["pelvis_site"]
    return torch.cat([_site_xpos(env)[:, pid, :], _site_xvel(env)[:, pid, :]], dim=-1)


def _boxing_target_pad_obs(env: Any, **_: Any) -> torch.Tensor:
    """6-pad position, normal, force, and hit-flag observations. Shape: (N, 48).

    Contact detection uses ``env.sim.mj_data`` which reflects env-0 only.
    This is correct when ``num_envs=1`` (play / evaluation).  Training with
    ``num_envs > 1`` uses only env-0 contact data for all envs; use
    ``_boxing_pad_damage_reward`` with caution in that setting and prefer
    single-env evaluation until mjlab exposes a vectorised contact API.
    """
    from myosuite.terms.multiplayer.boxing_mannequin_obs import target_pad_obs

    state = _get_boxing_state(env)
    arr = target_pad_obs(
        env.sim.mj_model,
        env.sim.mj_data,
        state.meta,
        "agent_0",
        hit_force_threshold_n=float(
            state.low_level_cfg.target_hit_normal_force_threshold_n
        ),
    )
    t = torch.as_tensor(arr, device=env.device, dtype=torch.float32).unsqueeze(0)
    if env.num_envs == 1:
        return t
    # Broadcast env-0 result; reward signal is approximate for envs 1..N-1
    return t.expand(env.num_envs, -1)


def _boxing_health_obs(env: Any, **_: Any) -> torch.Tensor:
    """Normalised health pair [agent_0, agent_1]. Shape: (N, 2).

    P0 uses static pad targets — no cumulative health tracking needed.
    Returns a constant (1.0, 1.0) pair.
    """
    return torch.ones(env.num_envs, 2, device=env.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Reward bridge functions
# ---------------------------------------------------------------------------


def _boxing_pad_damage_reward(env: Any, **_: Any) -> torch.Tensor:
    """Pad-contact reward using fist contacts. Shape: (N,).

    Uses env-0 contact data — accurate for ``num_envs=1``, approximate for
    parallel training (envs 1..N-1 receive env-0 reward).
    """
    from myosuite.terms.multiplayer.boxing_mannequin_reward import compute_pad_damage

    state = _get_boxing_state(env)
    val = compute_pad_damage(
        env.sim.mj_model,
        env.sim.mj_data,
        state.meta,
        "agent_0",
        hit_force_threshold_n=float(
            state.low_level_cfg.target_hit_normal_force_threshold_n
        ),
    )
    return torch.full(
        (env.num_envs,), float(val), device=env.device, dtype=torch.float32
    )


def _boxing_upright_posture_reward(env: Any, **_: Any) -> torch.Tensor:
    """Upright posture from pelvis quaternion for all N envs. Shape: (N,).

    Uses ``1 - 2*(qx² + qy²)`` which equals 1 when perfectly upright.
    """
    state = _get_boxing_state(env)
    pelvis_body_id = state.meta.body_ids.get("a0_pelvis")
    if pelvis_body_id is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    # wp_data.xquat: (N, nbody, 4) in (w, x, y, z) order
    xquat = torch.as_tensor(
        env.sim.wp_data.xquat, device=env.device, dtype=torch.float32
    )
    q = xquat[:, pelvis_body_id, :]
    qx, qy = q[:, 1], q[:, 2]
    return torch.clamp(1.0 - 2.0 * (qx * qx + qy * qy), 0.0, 1.0)


def _boxing_act_reg_reward(env: Any, **_: Any) -> torch.Tensor:
    """Mean-squared muscle activation regularization. Shape: (N,).

    Weight should be negative in the env config.
    """
    act = _entity_data(env).joint_act.to(torch.float32)  # (N, na)
    return torch.mean(act * act, dim=1)


def _boxing_survival_reward(env: Any, **_: Any) -> torch.Tensor:
    """Constant survival bonus: 1.0 per step for all N envs. Shape: (N,)."""
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Termination bridge
# ---------------------------------------------------------------------------


def _boxing_fall_termination(env: Any, **_: Any) -> torch.Tensor:
    """Fall termination: pelvis Z below threshold for all N envs. Shape: (N,) bool."""
    state = _get_boxing_state(env)
    pid = state.meta.site_ids["agent_0"]["pelvis_site"]
    threshold = float(state.low_level_cfg.fall_pelvis_z_threshold)
    return _site_xpos(env)[:, pid, 2] < threshold


# ---------------------------------------------------------------------------
# Event bridge functions
# ---------------------------------------------------------------------------


@requires_model_fields("qpos0")
def _boxing_reset_event(env: Any, env_ids: Any, **_: Any) -> None:
    """Reset the boxer entity to its default keyframe state."""
    env_ids_t = _normalize_env_ids(env, env_ids)
    if len(env_ids_t) == 0:
        return
    env.scene[_BOXING_ENTITY_NAME].reset(env_ids=env_ids_t)
    env.sim.forward()


def _boxing_mannequin_pre_physics(env: Any) -> None:
    """Pre-physics no-op: P0 mannequin is static; position actuators hold joints at zero."""
