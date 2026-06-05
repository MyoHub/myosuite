# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Action term functions and mjlab ActionTerm for MyoSuite environments.

Action Normalisation
--------------------
All MyoSuite muscle environments use a **sigmoid action normalisation** by default
(``normalize_act=True``).  Policy output is expected in ``[-1, 1]`` and is mapped
to muscle excitation in ``(0, 1)`` via::

    excitation = 1 / (1 + exp(-5 * (action - 0.5)))

The sigmoid is centred at 0.5, so a policy output of **0** maps to ≈ 62% excitation
(not 0%), and **−1** maps to ≈ 7%.  This matches Hill-type muscle resting activation.

**Action space vs. ctrl range**: when ``normalize_act=True`` the declared
``action_space`` is ``Box([-1, 1]^n)``, but the underlying ``model.actuator_ctrlrange``
is ``[0, 1]^n``.  If you switch ``normalize_act=False`` at inference time or
load a pre-trained policy trained with a different setting, excitations will be
wrong.  Always check ``env.normalize_act`` before deploying a policy.

``muscle_normalize_action`` is backend-agnostic and uses
``accessor.array_module()`` so it runs identically on CPU (numpy), MJX
(jax.numpy), and mjlab (torch).

``MuscleActionTerm`` is the mjlab (Isaac Lab manager API) integration wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from myosuite.core.protocols import EnvAccessor


def sigmoid_muscle_activation(action: Any, xp: Any) -> Any:
    """Apply the canonical MyoSuite sigmoid muscle mapping.

    Args:
        action: Input activation-like array.
        xp: Array module (`numpy`, `jax.numpy`, or `torch`).

    Returns:
        Array with element-wise ``1 / (1 + exp(-5 * (a - 0.5)))``.
    """
    return 1.0 / (1.0 + xp.exp(-5.0 * (action - 0.5)))


def muscle_normalize_action(accessor: EnvAccessor, action: Any, **kwargs: Any) -> Any:
    """Map policy actions from [-1, 1] to muscle excitation in [0, 1] via sigmoid.

    Uses ``accessor.array_module()`` so the same implementation runs on CPU
    (numpy), MJX (jax.numpy), and mjlab (torch).

    The sigmoid ``σ(5(a − 0.5))`` is the canonical MyoSuite muscle mapping:
    it is centred at 0.5 so that a zero policy action produces ~50% excitation,
    matching the resting state of Hill-type muscle models.

    Args:
        accessor: Environment state accessor (provides array_module).
        action: Policy output array in [-1, 1], any shape.
        **kwargs: Unused; for uniform call signature.

    Returns:
        Muscle excitation array in (0, 1), same shape as *action*.
    """
    xp = accessor.array_module()
    return sigmoid_muscle_activation(action, xp)


@dataclass
class MuscleActionTermCfg:
    """Configuration for MuscleActionTerm.

    Args:
        entity_name: Name of the articulation entity in the mjlab scene.
        normalize: If True, map [-1, 1] → [0, 1]. If False, clamp to [0, 1].
        muscle_fatigue: If True, apply 3-compartment cumulative fatigue dynamics
            to the processed excitations each control step.
        ctrl_dt: Control timestep in seconds; used as the integration step for
            the fatigue model when *muscle_fatigue* is True.
    """

    entity_name: str = "robot"
    normalize: bool = True
    muscle_fatigue: bool = False
    persist_muscle_fatigue: bool = False
    ctrl_dt: float = 0.01


class MuscleActionTerm:
    """mjlab ActionTerm that drives MuJoCo muscle actuators.

    Maps policy output to muscle excitation:
    - normalize=True:  canonical sigmoid mapping.
    - normalize=False: excitation = clamp(action, 0, 1)

    Args:
        cfg: Action term configuration.
        env: mjlab ManagerBasedRlEnv instance (injected by mjlab).

    Example:
        >>> cfg = MuscleActionTermCfg(entity_name="elbow", normalize=True)
        >>> term = MuscleActionTerm(cfg, env)
    """

    def __init__(self, cfg: MuscleActionTermCfg, env: Any) -> None:
        self.cfg = cfg
        self._env = env
        self._entity = env.scene[cfg.entity_name]
        self._processed: Any = None
        self._fatigue: Any = None
        if cfg.muscle_fatigue:
            from myosuite.core.muscle_conditions import TorchFatigueState  # noqa: PLC0415

            num_envs = getattr(env, "num_envs", 1)
            device = str(getattr(env, "device", "cpu"))
            mj_model = getattr(getattr(env, "sim", None), "mj_model", None)
            if mj_model is not None:
                self._fatigue = TorchFatigueState.from_mj_model(
                    mj_model, num_envs=num_envs, device=device
                )
            else:
                self._fatigue = TorchFatigueState(
                    num_envs=num_envs,
                    n_muscles=self._entity.num_actuators,
                    device=device,
                )

    @property
    def action_dim(self) -> int:
        """Number of muscle actuators in the entity."""
        return self._entity.num_actuators

    def process_actions(self, actions: Any) -> None:
        """Convert raw policy actions to muscle excitations.

        Args:
            actions: Policy output tensor, shape (N, action_dim).
        """
        if self.cfg.normalize:
            xp = getattr(actions, "__module__", "")
            if "torch" in xp:
                import torch  # noqa: PLC0415

                self._processed = sigmoid_muscle_activation(actions, torch)
            else:
                self._processed = sigmoid_muscle_activation(actions, np)
        else:
            self._processed = np.clip(actions, 0, 1)
        if self._fatigue is not None:
            self._processed = self._fatigue.step(self._processed, self.cfg.ctrl_dt)

    def reset(self, env_ids: Any = None) -> None:
        """Reset fatigue state for the given environments.

        Args:
            env_ids: Environment indices to reset, or ``None`` for all.
        """
        if self._fatigue is not None and not self.cfg.persist_muscle_fatigue:
            self._fatigue.reset(env_ids)

    def apply_actions(self) -> None:
        """Write processed excitations to the simulation entity."""
        if self._processed is not None:
            self._entity.set_ctrl(self._processed)
