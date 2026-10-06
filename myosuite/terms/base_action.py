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

The sigmoid is centred at 0.5, so a policy output of **−1** maps to ≈ 0.06% excitation,
**0** to ≈ 7.6%, **0.5** to 50% and **+1** to ≈ 92.4%: full (100%) excitation is never
reached.  This is the legacy MyoSuite mapping.

**Action space vs. ctrl range**: when ``normalize_act=True`` the declared
``action_space`` is ``Box([-1, 1]^n)``, but the underlying ``model.actuator_ctrlrange``
is ``[0, 1]^n``.  If you switch ``normalize_act=False`` at inference time or
load a pre-trained policy trained with a different setting, excitations will be
wrong.  Always check ``env.normalize_act`` before deploying a policy.

``muscle_normalize_action`` is backend-agnostic and uses
``accessor.array_module()`` so it runs identically on CPU (numpy), MJX
(jax.numpy), and mjlab (torch).

Motor noise
-----------
``motor_noise`` adds signal-dependent and constant Gaussian noise to muscle
excitations (``MotorNoiseCfg``, off by default). Both backends apply it after
the action -> excitation mapping and before fatigue; see
``docs/wiki/cross-backend-contract.md``.

``MuscleActionTerm`` is the mjlab (Isaac Lab manager API) integration wrapper.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
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
    it is centred at 0.5, so an action of -1 gives ~0.06% excitation, 0 gives ~7.6%
    and +1 gives ~92.4% (the output range is [0.00055, 0.924], not the full [0, 1]).

    Args:
        accessor: Environment state accessor (provides array_module).
        action: Policy output array in [-1, 1], any shape.
        **kwargs: Unused; for uniform call signature.

    Returns:
        Muscle excitation array in (0, 1), same shape as *action*.
    """
    xp = accessor.array_module()
    return sigmoid_muscle_activation(action, xp)


@dataclass(frozen=True)
class MotorNoiseCfg:
    """Gaussian motor noise on muscle excitations (off by default).

    The applied excitation is ``clip(u + signal_dependent_std * u * n1 +
    constant_std * n2, 0, 1)`` with independent standard normals ``n1, n2`` per
    muscle, per control step (and per env on mjlab). Signal-dependent noise
    (Harris & Wolpert 1998) makes the spread grow with the command, the source of
    the speed-accuracy trade-off; constant noise is independent of it.

    Args:
        signal_dependent_std: Std of the multiplicative noise (fraction of ``u``).
        constant_std: Std of the additive noise (excitation units).
    """

    signal_dependent_std: float = 0.0
    constant_std: float = 0.0

    def __post_init__(self) -> None:
        for name in ("signal_dependent_std", "constant_std"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"MotorNoiseCfg.{name} must be finite and >= 0.")
            object.__setattr__(self, name, value)

    @property
    def enabled(self) -> bool:
        """Whether any noise is added."""
        return self.signal_dependent_std > 0.0 or self.constant_std > 0.0

    @classmethod
    def van_beers_2004(cls) -> MotorNoiseCfg:
        """Levels 0.103 (signal-dependent) and 0.185 (constant).

        Fischer et al. (2021, Sci. Rep. 11:14445) take them "following van Beers
        et al." (2004, J. Neurophysiol. 91:1050-1063); User-in-the-Box (Ikkala et
        al., UIST 2022) uses the same defaults.
        """
        return cls(signal_dependent_std=0.103, constant_std=0.185)

    @classmethod
    def from_value(
        cls, value: MotorNoiseCfg | Mapping[str, float] | None
    ) -> MotorNoiseCfg:
        """Coerce a registration kwarg (cfg, field dict or ``None``) to a cfg.

        Args:
            value: ``MotorNoiseCfg``, a dict of its fields, or ``None`` (off).

        Returns:
            The config.
        """
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**value)
        raise TypeError(
            f"motor_noise must be a MotorNoiseCfg, a dict or None, got {type(value)!r}."
        )


def motor_noise(
    excitation: Any,
    normals_sd: Any,
    normals_c: Any,
    signal_dependent_std: float,
    constant_std: float,
    xp: Any,
) -> Any:
    """Add signal-dependent and constant noise to muscle excitations.

    Pure: the standard normals are drawn by the caller, so numpy and torch give
    the same result on the same draws.

    Args:
        excitation: Excitations ``u`` in ``[0, 1]``, any shape.
        normals_sd: Standard normals for the signal-dependent term, same shape.
        normals_c: Standard normals for the constant term, same shape.
        signal_dependent_std: Std of the multiplicative term.
        constant_std: Std of the additive term.
        xp: Array module (``numpy``, ``jax.numpy`` or ``torch``).

    Returns:
        ``clip(u + signal_dependent_std * u * normals_sd + constant_std * normals_c, 0, 1)``.
    """
    noisy = (
        excitation
        + signal_dependent_std * excitation * normals_sd
        + constant_std * normals_c
    )
    return xp.clip(noisy, 0.0, 1.0)


def sample_motor_noise(
    excitation: Any,
    cfg: MotorNoiseCfg,
    standard_normal: Callable[[tuple[int, ...]], Any],
    xp: Any,
) -> Any:
    """Draw fresh normals and apply :func:`motor_noise`; a no-op when disabled.

    A disabled config returns *excitation* unchanged without drawing, so the
    caller's random stream is untouched.

    Args:
        excitation: Muscle excitations, any shape (one draw per element).
        cfg: Noise levels.
        standard_normal: ``f(shape)`` returning standard normals, e.g.
            ``np_random.standard_normal`` or ``partial(torch.randn, device=...)``.
        xp: Array module matching *standard_normal*.

    Returns:
        The noisy (or unchanged) excitations.
    """
    if not cfg.enabled:
        return excitation
    normals = standard_normal((2, *excitation.shape))
    return motor_noise(
        excitation,
        normals[0],
        normals[1],
        cfg.signal_dependent_std,
        cfg.constant_std,
        xp,
    )


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
        is_torch = "torch" in getattr(actions, "__module__", "")
        if is_torch:
            import torch  # noqa: PLC0415

            xp = torch
        else:
            xp = np
        if self.cfg.normalize:
            self._processed = sigmoid_muscle_activation(actions, xp)
        else:
            self._processed = (
                torch.clamp(actions, 0, 1) if is_torch else np.clip(actions, 0, 1)
            )
        if self._fatigue is not None:
            self._processed = self._fatigue.step(self._processed, self.cfg.ctrl_dt)

    def reset(self, env_ids: Any = None) -> None:
        """Reset fatigue state for the given environments.

        Args:
            env_ids: Environment indices to reset, or ``None`` for all.
        """
        if self._fatigue is not None:
            self._fatigue.reset(env_ids)

    def apply_actions(self) -> None:
        """Write processed excitations to the simulation entity."""
        if self._processed is not None:
            self._entity.set_ctrl(self._processed)
