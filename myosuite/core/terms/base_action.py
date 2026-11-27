# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Action term functions and mjlab ActionTerm for MyoSuite environments.

Action Normalisation
--------------------
All MyoSuite muscle environments use **Buchanan exponential twitch-saturation**
by default (``normalize_act=True``, ``muscle_excitation="buchanan"``; Buchanan
et al. 2004, *J. Appl. Biomech.* 20:367-395; Lloyd & Besier 2003, *J. Biomech.*
36:765-776).  Policy muscle output is expected in ``[0, 1]`` (interpretable as
the MVC-normalised linear sEMG / neural command envelope) and is mapped to
muscle excitation in ``[0, 1]`` via::

    excitation = (exp(A * clip(action, 0, 1)) - 1) / (exp(A) - 1),   A = -2.0

An action of **0** maps to exact **0%** excitation (true rest) and **+1** maps
to exact **100%** MVC excitation, with concave twitch-summation saturation on
``(0, 1)``.  Legacy ``muscle_excitation="sigmoid"`` retains the earlier
``[-1, 1] -> (0.00055, 0.924)`` mapping ``1 / (1 + exp(-5 * (action - 0.5)))``
for pre-3.2 checkpoint compatibility, and ``muscle_excitation="linear"`` clips
``[0, 1]`` directly.

``buchanan_excitation``, ``sigmoid_muscle_activation``, and
``muscle_normalize_action`` are backend-agnostic and use
``accessor.array_module()`` so they run identically on CPU (numpy), MJX
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
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from myosuite.core.protocols import EnvAccessor

DEFAULT_BUCHANAN_SHAPE_A: float = -2.0
MuscleExcitationMode = Literal["buchanan", "sigmoid", "linear"]
MUSCLE_EXCITATION_MODES: tuple[str, ...] = ("buchanan", "sigmoid", "linear")


def validate_muscle_excitation(
    mode: str = "buchanan",
    shape_A: float = DEFAULT_BUCHANAN_SHAPE_A,
) -> tuple[MuscleExcitationMode, float]:
    """Validate *mode* and *shape_A* for muscle excitation mapping."""
    if mode not in MUSCLE_EXCITATION_MODES:
        raise ValueError(
            f"muscle_excitation must be one of {MUSCLE_EXCITATION_MODES}, got {mode!r}."
        )
    a = float(shape_A)
    if not (math.isfinite(a) and -5.0 <= a <= 5.0):
        raise ValueError(
            f"buchanan_shape_A must be finite in [-5, 5], got {shape_A!r}."
        )
    return cast(MuscleExcitationMode, mode), a


def buchanan_excitation(
    action: Any,
    xp: Any,
    shape_A: float = DEFAULT_BUCHANAN_SHAPE_A,
) -> Any:
    """Apply Buchanan et al. (2004) exponential twitch-saturation ``[0, 1] -> [0, 1]``.

    Maps a normalised neural command / sEMG envelope ``e = clip(action, 0, 1)``
    to muscle excitation ``(exp(A * e) - 1) / (exp(A) - 1)`` with exact
    endpoints ``0 -> 0`` and ``1 -> 1``.

    Args:
        action: Pre-excitation neural command array in ``[0, 1]``.
        xp: Array module (``numpy``, ``jax.numpy``, or ``torch``).
        shape_A: Nonlinear curvature parameter ``A`` (default ``-2.0``; linear as ``A -> 0``).

    Returns:
        Muscle excitation array in ``[0, 1]``, matching the dtype/shape of *action*.
    """
    _, a = validate_muscle_excitation("buchanan", shape_A)
    e = xp.clip(action, 0.0, 1.0)
    if abs(a) < 1e-6:
        return e
    denom = math.expm1(a)
    return xp.clip(xp.expm1(a * e) / denom, 0.0, 1.0)


def inverse_buchanan_excitation(
    excitation: Any,
    xp: Any,
    shape_A: float = DEFAULT_BUCHANAN_SHAPE_A,
) -> Any:
    """Exact inverse of :func:`buchanan_excitation`, mapping ``u in [0, 1] -> e in [0, 1]``.

    Args:
        excitation: Muscle excitation array in ``[0, 1]``.
        xp: Array module (``numpy``, ``jax.numpy``, or ``torch``).
        shape_A: Nonlinear curvature parameter ``A`` passed to :func:`buchanan_excitation`.

    Returns:
        Pre-saturation neural command envelope ``e in [0, 1]``.
    """
    _, a = validate_muscle_excitation("buchanan", shape_A)
    u = xp.clip(excitation, 0.0, 1.0)
    if abs(a) < 1e-6:
        return u
    return xp.clip(xp.log1p(u * math.expm1(a)) / a, 0.0, 1.0)


def sigmoid_muscle_activation(action: Any, xp: Any) -> Any:
    """Apply the legacy MyoSuite sigmoid muscle mapping ``[-1, 1] -> (0, 1)``.

    Args:
        action: Input activation-like array.
        xp: Array module (`numpy`, `jax.numpy`, or `torch`).

    Returns:
        Array with element-wise ``1 / (1 + exp(-5 * (a - 0.5)))``.
    """
    return 1.0 / (1.0 + xp.exp(-5.0 * (action - 0.5)))


def apply_muscle_excitation(
    action: Any,
    xp: Any,
    mode: str = "buchanan",
    shape_A: float = DEFAULT_BUCHANAN_SHAPE_A,
) -> Any:
    """Map raw policy muscle commands to excitations in ``[0, 1]``."""
    m, a = validate_muscle_excitation(mode, shape_A)
    if m == "buchanan":
        return buchanan_excitation(action, xp, shape_A=a)
    if m == "sigmoid":
        return sigmoid_muscle_activation(action, xp)
    return xp.clip(action, 0.0, 1.0)


def normalized_action_bounds(
    model: Any,
    *,
    normalize_act: bool = True,
    muscle_excitation: str = "buchanan",
    n_actuators: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute float32 ``(act_low, act_high)`` action-space bounds for *model*.

    When ``normalize_act=True``, non-muscle actuators use ``[-1, 1]`` while
    muscle actuators (``mjDYN_MUSCLE``) use ``[0, 1]`` for ``"buchanan"`` /
    ``"linear"`` and ``[-1, 1]`` for legacy ``"sigmoid"``.
    """
    import mujoco  # noqa: PLC0415

    m, _ = validate_muscle_excitation(muscle_excitation)
    n = int(model.nu) if n_actuators is None else int(n_actuators)
    if not normalize_act:
        return (
            model.actuator_ctrlrange[:n, 0].astype(np.float32),
            model.actuator_ctrlrange[:n, 1].astype(np.float32),
        )
    low = -np.ones(n, dtype=np.float32)
    high = np.ones(n, dtype=np.float32)
    if m != "sigmoid" and n > 0:
        is_muscle = model.actuator_dyntype[:n] == int(mujoco.mjtDyn.mjDYN_MUSCLE)
        low[is_muscle] = 0.0
    return low, high


def muscle_normalize_action(
    accessor: EnvAccessor,
    action: Any,
    mode: str = "buchanan",
    shape_A: float = DEFAULT_BUCHANAN_SHAPE_A,
    **kwargs: Any,
) -> Any:
    """Map policy muscle actions to muscle excitation in ``[0, 1]``.

    Uses ``accessor.array_module()`` so the same implementation runs on CPU
    (numpy), MJX (jax.numpy), and mjlab (torch). Defaults to
    :func:`buchanan_excitation` (``[0, 1] -> [0, 1]`` with ``A = -2.0``); pass
    ``mode="sigmoid"`` for the legacy ``1 / (1 + exp(-5 * (a - 0.5)))`` curve.

    Args:
        accessor: Environment state accessor (provides array_module).
        action: Policy output array, any shape.
        mode: ``"buchanan"`` (default), ``"sigmoid"``, or ``"linear"``.
        shape_A: Shape factor ``A`` for ``"buchanan"``.
        **kwargs: Unused; for uniform call signature.

    Returns:
        Muscle excitation array in ``[0, 1]``, same shape as *action*.
    """
    del kwargs
    xp = accessor.array_module()
    return apply_muscle_excitation(action, xp, mode=mode, shape_A=shape_A)


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
        normalize: If True, map raw muscle actions through *muscle_excitation*.
            If False, clamp to ``[0, 1]``.
        muscle_excitation: ``"buchanan"`` (default), ``"sigmoid"``, or ``"linear"``.
        buchanan_shape_A: Curvature parameter ``A`` for ``"buchanan"``.
        muscle_fatigue: If True, apply 3-compartment cumulative fatigue dynamics
            to the processed excitations each control step.
        ctrl_dt: Control timestep in seconds; used as the integration step for
            the fatigue model when *muscle_fatigue* is True.
    """

    entity_name: str = "robot"
    normalize: bool = True
    muscle_excitation: MuscleExcitationMode = "buchanan"
    buchanan_shape_A: float = DEFAULT_BUCHANAN_SHAPE_A
    muscle_fatigue: bool = False
    ctrl_dt: float = 0.01

    def __post_init__(self) -> None:
        self.muscle_excitation, self.buchanan_shape_A = validate_muscle_excitation(
            self.muscle_excitation, self.buchanan_shape_A
        )


class MuscleActionTerm:
    """mjlab ActionTerm that drives MuJoCo muscle actuators.

    Maps policy output to muscle excitation:
    - normalize=True:  :func:`apply_muscle_excitation` (default ``"buchanan"``).
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
            import mujoco  # noqa: PLC0415

            from myosuite.core.muscle_conditions import TorchFatigueState  # noqa: PLC0415

            num_envs = getattr(env, "num_envs", 1)
            device = str(getattr(env, "device", "cpu"))
            mj_model = getattr(getattr(env, "sim", None), "mj_model", None)
            ctrl_ids = getattr(
                getattr(self._entity, "indexing", None), "ctrl_ids", None
            )
            if mj_model is not None and hasattr(mj_model, "actuator_dyntype"):
                if ctrl_ids is not None:
                    ids_np = (
                        ctrl_ids.detach().cpu().numpy()
                        if hasattr(ctrl_ids, "detach")
                        else np.asarray(ctrl_ids, dtype=int)
                    )
                    n_muscles = int(
                        np.sum(
                            mj_model.actuator_dyntype[ids_np]
                            == int(mujoco.mjtDyn.mjDYN_MUSCLE)
                        )
                    )
                    self._fatigue = TorchFatigueState(
                        num_envs=num_envs, n_muscles=n_muscles, device=device
                    )
                else:
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
        mod = getattr(actions, "__module__", "")
        if "torch" in mod:
            import torch  # noqa: PLC0415

            xp: Any = torch
        elif "jax" in mod:
            import jax.numpy as jnp  # noqa: PLC0415

            xp = jnp
        else:
            xp = np
        if self.cfg.normalize:
            self._processed = apply_muscle_excitation(
                actions,
                xp,
                mode=self.cfg.muscle_excitation,
                shape_A=self.cfg.buchanan_shape_A,
            )
        else:
            self._processed = xp.clip(actions, 0.0, 1.0)
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
        if self._processed is None:
            return
        if hasattr(type(self._entity), "write_ctrl_to_sim") or (
            "write_ctrl_to_sim" in getattr(self._entity, "__dict__", {})
        ):
            self._entity.write_ctrl_to_sim(self._processed)
        else:
            self._entity.set_ctrl(self._processed)
