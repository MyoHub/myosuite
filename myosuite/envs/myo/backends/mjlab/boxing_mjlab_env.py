# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab bridge functions for the BoxingP0 task (6-pad static target variant).

All observations, rewards, and terminations delegate to the existing CPU-parity
term functions in ``boxing_mannequin_obs.py`` and ``boxing_mannequin_reward.py``.
No env subclassing — this module only provides thin bridge callables and a
``BoxingTaskState`` ManagerTermBase that holds cached MuJoCo ids.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers import ManagerTermBase
from mjlab.managers.event_manager import requires_model_fields

if TYPE_CHECKING:
    pass

_BOXING_ENTITY_NAME = "boxing_p0_robot"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wp_tensor(array: Any, device: str) -> torch.Tensor:
    return torch.as_tensor(array, device=device)


try:
    from mjlab.envs.mdp.events import resolve_env_ids as _normalize_env_ids
except ImportError:
    # mjlab<1.4: resolve_env_ids not yet exported
    def _normalize_env_ids(env: Any, env_ids: Any) -> torch.Tensor:  # type: ignore[misc]
        if env_ids is None:
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[
                env_ids
            ]
        return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)


# ---------------------------------------------------------------------------
# Task state holder
# ---------------------------------------------------------------------------


class BoxingTaskState(ManagerTermBase):
    """Holds per-env cached ids for the boxing task.

    Stored on ``env._boxing_state`` so bridge callables can retrieve it.
    The mannequin (agent_1) is static in P0 — its joints are kept at zero
    via position actuators already in the model; no extra drive is needed.
    """

    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(env)
        self.cfg = cfg
        self.env = env
        env._boxing_state = self  # noqa: SLF001

        self.task_cfg = cfg.params["task_cfg"]
        self.meta = cfg.params["meta"]
        self.mj_model = cfg.params["mj_model"]

    def __call__(self, env: Any, env_ids: Any, **_: Any) -> None:
        """Step event hook — no per-step logic needed for static P0 mannequin."""


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def _get_boxing_state(env: Any) -> BoxingTaskState:
    state = getattr(env, "_boxing_state", None)
    if state is None:
        raise RuntimeError(
            "BoxingTaskState was not initialized. "
            "Ensure the step event is registered with mode='step'."
        )
    return state


def _boxing_mj_data(env: Any) -> Any:
    """Return a MjData-compatible view populated from the current Warp simulation."""
    return env.sim.mj_data


def _boxing_mj_model(env: Any) -> Any:
    return env.sim.mj_model


# ---------------------------------------------------------------------------
# Observation bridge functions
# ---------------------------------------------------------------------------


def _boxing_kinematics_obs(env: Any, **_: Any) -> torch.Tensor:
    """Own joint positions, velocities, and muscle activations. Shape: (N, d)."""
    from myosuite.terms.multiplayer.boxing_mannequin_obs import own_kinematics_obs

    state = _get_boxing_state(env)
    arr = own_kinematics_obs(
        _boxing_mj_model(env),
        _boxing_mj_data(env),
        state.meta,
        "agent_0",
    )
    t = torch.as_tensor(arr, device=env.device, dtype=torch.float32).unsqueeze(0)
    return t.expand(env.num_envs, -1)


def _boxing_fist_obs(env: Any, **_: Any) -> torch.Tensor:
    """Fist positions and velocities. Shape: (N, d)."""
    from myosuite.terms.multiplayer.boxing_mannequin_obs import own_fist_obs

    state = _get_boxing_state(env)
    arr = own_fist_obs(_boxing_mj_data(env), state.meta, "agent_0")
    t = torch.as_tensor(arr, device=env.device, dtype=torch.float32).unsqueeze(0)
    return t.expand(env.num_envs, -1)


def _boxing_pelvis_obs(env: Any, **_: Any) -> torch.Tensor:
    """Pelvis position and velocity. Shape: (N, d)."""
    from myosuite.terms.multiplayer.boxing_mannequin_obs import own_pelvis_obs

    state = _get_boxing_state(env)
    arr = own_pelvis_obs(_boxing_mj_data(env), state.meta, "agent_0")
    t = torch.as_tensor(arr, device=env.device, dtype=torch.float32).unsqueeze(0)
    return t.expand(env.num_envs, -1)


def _boxing_target_pad_obs(env: Any, **_: Any) -> torch.Tensor:
    """6-pad position, normal, force, and hit-flag observations. Shape: (N, 48)."""
    from myosuite.terms.multiplayer.boxing_mannequin_obs import target_pad_obs

    state = _get_boxing_state(env)
    arr = target_pad_obs(
        _boxing_mj_model(env),
        _boxing_mj_data(env),
        state.meta,
        "agent_0",
        hit_force_threshold_n=float(state.task_cfg.target_hit_normal_force_threshold_n),
    )
    t = torch.as_tensor(arr, device=env.device, dtype=torch.float32).unsqueeze(0)
    return t.expand(env.num_envs, -1)


def _boxing_health_obs(env: Any, **_: Any) -> torch.Tensor:
    """Normalised health pair [agent_0, agent_1]. Shape: (N, 2)."""
    from myosuite.terms.multiplayer.boxing_mannequin_obs import health_obs

    state = _get_boxing_state(env)
    # P0: static pad variant; no cumulative health tracking in this bridge
    health: dict[str, float] = {"agent_0": 0.0, "agent_1": 0.0}
    arr = health_obs(health, "agent_0", float(state.task_cfg.ko_health_threshold))
    t = torch.as_tensor(arr, device=env.device, dtype=torch.float32).unsqueeze(0)
    return t.expand(env.num_envs, -1)


# ---------------------------------------------------------------------------
# Reward bridge functions
# ---------------------------------------------------------------------------


def _boxing_pad_damage_reward(env: Any, **_: Any) -> torch.Tensor:
    """Pad-contact reward using fist contacts. Shape: (N,)."""
    from myosuite.terms.multiplayer.boxing_mannequin_reward import compute_pad_damage

    state = _get_boxing_state(env)
    val = compute_pad_damage(
        _boxing_mj_model(env),
        _boxing_mj_data(env),
        state.meta,
        "agent_0",
        hit_force_threshold_n=float(state.task_cfg.target_hit_normal_force_threshold_n),
    )
    return torch.full(
        (env.num_envs,), float(val), device=env.device, dtype=torch.float32
    )


def _boxing_upright_posture_reward(env: Any, **_: Any) -> torch.Tensor:
    """Upright posture reward from pelvis quaternion. Shape: (N,).

    Uses ``1 - 2*(qx^2 + qy^2)`` which equals 1 when the body is perfectly upright.
    """
    data = env.sim.wp_data
    state = _get_boxing_state(env)
    pelvis_body_id = state.meta.body_ids.get("a0_pelvis")
    if pelvis_body_id is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    xquat = _wp_tensor(data.xquat, env.device)  # (N, nbody, 4)
    q = xquat[:, pelvis_body_id, :]  # (N, 4) in (w, x, y, z) order
    qx, qy = q[:, 1], q[:, 2]
    upright = torch.clamp(1.0 - 2.0 * (qx * qx + qy * qy), 0.0, 1.0)
    return upright


def _boxing_act_reg_reward(env: Any, **_: Any) -> torch.Tensor:
    """Mean-squared action regularization over muscle activations. Shape: (N,).

    Weight should be negative in the env config.
    """
    data = env.sim.wp_data
    act = _wp_tensor(data.act, env.device)  # (N, na)
    return torch.mean(act * act, dim=1)


def _boxing_survival_reward(env: Any, **_: Any) -> torch.Tensor:
    """Constant survival bonus: 1.0 per step. Shape: (N,)."""
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Termination bridge
# ---------------------------------------------------------------------------


def _boxing_fall_termination(env: Any, **_: Any) -> torch.Tensor:
    """Fall termination based on pelvis Z height. Shape: (N,) bool."""
    from myosuite.terms.multiplayer.boxing_mannequin_reward import is_fallen

    state = _get_boxing_state(env)
    fell = is_fallen(
        _boxing_mj_data(env),
        state.meta,
        "agent_0",
        _boxing_config_from_state(state),
    )
    return torch.full((env.num_envs,), bool(fell), device=env.device, dtype=torch.bool)


def _boxing_config_from_state(state: BoxingTaskState) -> Any:
    """Reconstruct a ``BoxingMannequinConfig`` from the task config fields."""
    from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_config import (
        BoxingMannequinConfig,
    )

    cfg_kwargs = {
        f.name: getattr(state.task_cfg, f.name)
        for f in dataclass_fields(BoxingMannequinConfig)
        if hasattr(state.task_cfg, f.name)
    }
    return BoxingMannequinConfig(**cfg_kwargs)


# ---------------------------------------------------------------------------
# Event bridge functions
# ---------------------------------------------------------------------------


@requires_model_fields("qpos0")
def _boxing_reset_event(env: Any, env_ids: Any, **_: Any) -> None:
    """Reset boxer entity to default initial state."""
    env_ids_t = _normalize_env_ids(env, env_ids)
    if len(env_ids_t) == 0:
        return
    entity = env.scene[_BOXING_ENTITY_NAME]
    entity.reset(env_ids=env_ids_t)
    env.sim.forward()


def _boxing_startup_event(env: Any, env_ids: Any, **_: Any) -> None:
    """Startup event: reset to initial state."""
    _boxing_reset_event(env, env_ids)


def _boxing_mannequin_pre_physics(env: Any) -> None:
    """Pre-physics no-op: P0 mannequin is static; position actuators hold joints at zero."""
