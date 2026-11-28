# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Model-agnostic action and reward wrappers for ``v0`` and ``v1`` musculoskeletal envs."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from myosuite.core.training_recipes import (
    AnatomicalSynergyMap,
    build_anatomical_synergy_map,
)


class SynergyResidualActionWrapper(gym.ActionWrapper):
    """Map low-dimensional anatomical synergy + per-muscle residual actions to excitations.

    Eliminates high-DoF antagonistic co-contraction lockup on ``172-558`` muscle
    whole-body models during standard PPO exploration while preserving full
    per-muscle controllability through the residual channels:

    .. math::

        u = \\sigma(\\operatorname{logit}(u_0) + g_{\\text{syn}} a_{\\text{syn}} M + g_{\\text{res}} a_{\\text{res}})

    Args:
        env: Wrapped Gymnasium environment accepting ``[0, 1]`` or ``[-1, 1]`` muscle actions.
        synergy_gain: Logit gain applied to the ``n_groups`` synergy coordinates.
        residual_gain: Logit gain applied to the ``nu`` per-muscle residual coordinates
            (set ``0.0`` for pure ``n_groups`` synergy control).
        default_tonus: Baseline postural excitation ``u_0`` at zero policy output.
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        synergy_gain: float = 1.8,
        residual_gain: float = 0.8,
        default_tonus: float = 0.16,
    ) -> None:
        super().__init__(env)
        model = getattr(env.unwrapped, "model", None)
        if model is None:
            raise ValueError("SynergyResidualActionWrapper requires env.unwrapped.model")
        self.synergy_map: AnatomicalSynergyMap = build_anatomical_synergy_map(
            model, default_tonus=default_tonus
        )
        self.synergy_gain = float(synergy_gain)
        self.residual_gain = float(residual_gain)
        self._n_syn = len(self.synergy_map.group_names)
        self._nu = int(model.nu)
        u0 = np.clip(self.synergy_map.tonic_excitation, 1e-4, 1.0 - 1e-4)
        self._logit0 = np.log(u0 / (1.0 - u0)).astype(np.float32)

        act_dim = self._n_syn + (self._nu if self.residual_gain > 0.0 else 0)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        inner_low = np.asarray(env.action_space.low, dtype=np.float32)
        inner_high = np.asarray(env.action_space.high, dtype=np.float32)
        self._inner_low = inner_low
        self._inner_span = inner_high - inner_low

    def action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        syn = a[: self._n_syn]
        logits = self._logit0 + self.synergy_gain * (syn @ self.synergy_map.expand_matrix)
        if self.residual_gain > 0.0:
            res = a[self._n_syn :]
            logits = logits + self.residual_gain * res
        exc = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        # Map [0, 1] excitation linearly onto inner env's action_space [low, high]
        return (self._inner_low + exc * self._inner_span).astype(np.float32)


class TaskOnlyRewardWrapper(gym.Wrapper):
    """Strip method-specific reward shaping, returning pure task success/failure outcomes.

    Preserves ``info['rwd_dict']`` and ``info['shaped_reward']`` for diagnostic logging,
    while setting the returned scalar ``reward = float(solved) - float(terminated)``.
    """

    def step(
        self, action: Any
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, shaped_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["shaped_reward"] = float(shaped_reward)
        solved = bool(
            info.get("solved", False)
            or info.get("rwd_dict", {}).get("solved", False)
        )
        pure_reward = float(solved) - (1.0 if terminated and not solved else 0.0)
        return obs, pure_reward, terminated, truncated, info
