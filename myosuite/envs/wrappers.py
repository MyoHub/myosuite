# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Gymnasium wrappers for MyoSuite environments.

Available wrappers
------------------
:class:`MjInstabilityTerminationWrapper`
    Converts MuJoCo physics instability warnings into episode termination.

:class:`AttributeForwardingWrapper`
    Forwards public attributes (e.g. ``mj_render``) to the wrapped env.

:class:`DictObservationWrapper`
    Exposes structured ``Dict`` observations instead of a flat ``Box`` vector.

:class:`PerturbationWrapper`
    Injects scheduled external forces/torques to named bodies for perturbation
    experiments (motor control, reactive balance, neuroscience lesion studies).

Muscle-command wrappers
-----------------------
None of them is active by default (a plain env runs only its own map): the
``myoFati*``, ``myoReaf*`` and ``myoSarc*`` ids register the matching wrapper, and
noise additionally needs a nonzero level. These install one ordered stage in the
env's action pipeline (see
:mod:`myosuite.envs.muscle_stages`): ``map (env) -> noise -> fatigue -> reroute
-> custom stages -> ctrl``. The built-in order is fixed by the stage, whatever the wrapping
order; custom stages run after them in installation order. Each
stage can be installed **once per env**: a second wrapper of the same kind raises a
``ValueError`` (the ``myoFati*`` and ``myoReaf*`` ids already contain theirs, so
wrap the base id to configure it, e.g. ``FatigueWrapper(make_env(base_id),
fatigue_reset_random=True)``, or change the options with
``env.set_fatigue_reset_random(...)`` / ``env.set_motor_noise(...)``).

:class:`MotorNoiseWrapper`
    Signal-dependent + constant Gaussian noise on the muscle excitations.

:class:`FatigueWrapper`
    3CC-r muscle fatigue (``myoFati*`` envs).

:class:`ReafferentationWrapper`
    EIP command rerouted to EPL (``myoReaf*`` envs).

:class:`SarcopeniaWrapper`
    Muscle force reduction applied to the model (``myoSarc*`` envs).

:class:`ExcitationStageWrapper`
    A portable custom stage on the muscle excitations (filter, cap, gains, ...) at an
    order of your choice; runs on the CPU env and on its mjlab twin.

:class:`CtrlStageWrapper`
    Like it, but env-aware and CPU only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import WrapperSpec, load_env_creator
from gymnasium.utils import RecordConstructorArgs

from myosuite.envs import muscle_stages


class _ForwardPublicAttributes:
    """Forward public attributes the wrapper lacks to the wrapped env.

    Gymnasium 1.0 dropped this from ``Wrapper``, so ``env.mj_render()`` on the env
    returned by ``gym.make`` raised ``AttributeError`` (use ``env.unwrapped`` or
    ``env.get_wrapper_attr(name)``). The outermost wrapper of every registered
    MyoSuite env restores it. Private names are not forwarded.
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "env":
            raise AttributeError(name)
        return self.env.get_wrapper_attr(name)


class AttributeForwardingWrapper(
    _ForwardPublicAttributes, RecordConstructorArgs, gym.Wrapper
):
    """Outermost wrapper of envs registered without an instability wrapper."""

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)


class MjInstabilityTerminationWrapper(
    _ForwardPublicAttributes, RecordConstructorArgs, gym.Wrapper
):
    """Upgrade MuJoCo instability checks into termination after each step."""

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def step(self, action: Any, **kwargs: Any) -> tuple[Any, Any, Any, Any, Any]:
        if kwargs:
            obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)
        else:
            obs, reward, terminated, truncated, info = super().step(action)

        env = self.unwrapped
        check = getattr(env, "_check_mj_instability_termination", None)
        enabled = bool(getattr(env, "mj_instability_termination", False))
        if not enabled or not callable(check) or not bool(check()):
            return obs, reward, terminated, truncated, info

        if isinstance(terminated, dict):
            terminated = {key: True for key in terminated}
        elif isinstance(terminated, np.ndarray):
            terminated = np.ones_like(terminated, dtype=bool)
        else:
            terminated = True

        return obs, reward, terminated, truncated, info


class DictObservationWrapper(RecordConstructorArgs, gym.ObservationWrapper):
    """Expose structured observations as a ``gymnasium.spaces.Dict`` space.

    MyoGymnasiumEnv flattens all observation terms into a single ``Box``
    vector.  This wrapper re-packages the env's observation dict (the
    ``info["obs_dict"]`` of each step; recomputed with ``get_obs_dict()`` at
    reset, whose info carries none) into a proper ``Dict`` observation so that
    modular or hierarchical policies can consume named sub-observations
    directly without manual slicing.  The ``Dict`` space is built at
    construction, so vector envs and SB3 see it before the first reset.

    The action space and physics are unmodified.

    Example::

        env = make_env("myoElbowPose1D6MRandom-v0")
        env = DictObservationWrapper(env)
        obs, info = env.reset()
        print(list(obs.keys()))  # ['qpos', 'qvel', 'pose_err']
        print(obs["qpos"].shape)

    Args:
        env: A :class:`~myosuite.envs.gymnasium_env.MyoGymnasiumEnv` instance.
    """

    def __init__(self, env: gym.Env) -> None:
        RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Dict(
            {
                k: gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=v.shape, dtype=np.float32
                )
                for k, v in self._current_obs().items()
            }
        )

    @staticmethod
    def _to_dict_obs(obs_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        return {
            k: np.atleast_1d(np.asarray(v, dtype=np.float32))
            for k, v in obs_dict.items()
        }

    def _current_obs(self) -> dict[str, np.ndarray]:
        """Dict observation of the env's current state."""
        env = self.env.unwrapped
        # (model, data) also works before an env has built its accessor.
        return self._to_dict_obs(env.get_obs_dict(env.model, env.data))

    def _info_obs(self, info: dict[str, Any]) -> dict[str, np.ndarray]:
        """The info's obs dict, else (reset info is empty) the current one."""
        if "obs_dict" in info:
            return self._to_dict_obs(info["obs_dict"])
        return self._current_obs()

    def observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Dict observation of the current state (*obs* is the flat vector)."""
        return self._current_obs()

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        _, info = self.env.reset(**kwargs)
        return self._info_obs(info), info

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._info_obs(info), reward, terminated, truncated, info


class PerturbationWrapper(RecordConstructorArgs, gym.Wrapper):
    """Apply scheduled external forces/torques to named bodies at runtime.

    Enables perturbation-response experiments — reactive balance, motor
    adaptation, and neuroscience lesion studies — without modifying env
    source code.  Perturbations are injected into MuJoCo's
    ``data.xfrc_applied`` before each physics substep.  Every step the wrapper
    rewrites the rows of the bodies it perturbs as the sum of the active
    perturbations (zero once none is active).

    A perturbation is a dict with the following keys:

    * ``"body"`` *(str)* — MuJoCo body name.
    * ``"force"`` *(array-like, shape (3,), optional)* — world-frame force in N.
    * ``"torque"`` *(array-like, shape (3,), optional)* — world-frame torque in N·m.
    * ``"start"`` *(int)* — step index to begin applying (inclusive).
    * ``"end"`` *(int, optional)* — step index to stop (exclusive); ``-1`` means
      apply forever. Defaults to ``-1``.

    Example::

        env = make_env("myoLegWalk-v0")
        env = PerturbationWrapper(env)

        # Trip: lateral push to pelvis for two steps, starting at step 50
        env.add_perturbation({
            "body": "pelvis",
            "force": [0, 200, 0],
            "start": 50,
            "end": 52,
        })

        obs, info = env.reset()
        for _ in range(200):
            obs, r, term, trunc, info = env.step(env.action_space.sample())

        # Selective muscle "lesion": zero a muscle by disabling its output
        # (set via xfrc_applied; full muscle disable requires env modification).

    Args:
        env: A MyoGymnasiumEnv instance.
    """

    def __init__(self, env: gym.Env) -> None:
        RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        self._perturbations: list[dict[str, Any]] = []
        self._step_count: int = 0
        # Bodies whose xfrc_applied row this wrapper has written since reset.
        self._touched_bodies: set[int] = set()

    def add_perturbation(self, perturbation: dict[str, Any]) -> None:
        """Register a perturbation.

        Args:
            perturbation: Dict with keys ``body``, ``force`` (optional),
                ``torque`` (optional), ``start``, ``end`` (optional, default -1).
        """
        self._perturbations.append(
            {
                "body": str(perturbation["body"]),
                "force": np.asarray(
                    perturbation.get("force", [0.0, 0.0, 0.0]), dtype=np.float64
                ),
                "torque": np.asarray(
                    perturbation.get("torque", [0.0, 0.0, 0.0]), dtype=np.float64
                ),
                "start": int(perturbation["start"]),
                "end": int(perturbation.get("end", -1)),
            }
        )

    def clear_perturbations(self) -> None:
        """Remove all registered perturbations."""
        self._perturbations.clear()

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._step_count = 0
        self._touched_bodies.clear()
        obs, info = self.env.reset(**kwargs)
        # Zero xfrc_applied on reset to prevent carry-over from previous episode.
        env = self.unwrapped
        data = getattr(env, "data", None)
        if data is not None:
            data.xfrc_applied[:] = 0.0
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        env = self.unwrapped
        data = getattr(env, "data", None)
        model = getattr(env, "model", None)

        if data is not None and model is not None:
            import mujoco

            # Rebuild the rows from scratch: perturbations on one body add up,
            # and one closing window does not cancel the others.
            for body_id in self._touched_bodies:
                data.xfrc_applied[body_id] = 0.0
            for p in self._perturbations:
                end = p["end"]
                active = p["start"] <= self._step_count and (
                    end == -1 or self._step_count < end
                )
                if not active:
                    continue
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p["body"])
                if body_id < 0:
                    continue
                data.xfrc_applied[body_id, :3] += p["force"]
                data.xfrc_applied[body_id, 3:] += p["torque"]
                self._touched_bodies.add(body_id)

        result = self.env.step(action)
        self._step_count += 1
        return result


class _PicklableStage:
    """Pickle a muscle wrapper as (env, kwargs) and re-install its stage on restore.

    The CPU envs are rebuilt from their constructor arguments when unpickled, which drops
    the stage the wrapper installed; ``__setstate__`` wraps the restored env again.
    """

    def _pickle_kwargs(self) -> dict[str, Any]:
        return dict(self._saved_kwargs)

    def __getstate__(self) -> dict[str, Any]:
        return {"env": self.env, "kwargs": self._pickle_kwargs()}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["env"], **state["kwargs"])  # type: ignore[misc]


def _stage_host(env: gym.Env, wrapper: str) -> Any:
    """The unwrapped env, which must run wrapper-installed muscle stages."""
    host = env.unwrapped
    if not getattr(host, "supports_ctrl_stages", False):
        raise TypeError(
            f"{wrapper} needs an env whose action pipeline runs muscle stages; "
            f"{type(host).__name__} does not (the basic hand/arm/leg/torso envs "
            "and the MyoChallenge muscle envs do)."
        )
    return host


class MotorNoiseWrapper(
    _PicklableStage, _ForwardPublicAttributes, RecordConstructorArgs, gym.Wrapper
):
    """Gaussian motor noise on the muscle excitations (off by default).

    The applied excitation is ``clip(u + signal_dependent_std * u * n1 +
    constant_std * n2, 0, 1)`` with independent standard normals per muscle and
    control step, drawn from the env's seeded ``np_random`` after the env's
    action-to-excitation map and before fatigue (see
    :class:`~myosuite.terms.base_action.MotorNoiseCfg`). A disabled config draws
    nothing, so the random stream of the env is unchanged.

    Example::

        env = MotorNoiseWrapper(
            make_env("myoElbowPose1D6MRandom-v0"), MotorNoiseCfg.van_beers_2004()
        )

    Args:
        env: Env whose pipeline runs muscle stages.
        motor_noise: A :class:`~myosuite.terms.base_action.MotorNoiseCfg`, a dict
            of its fields or ``None`` (off). Call ``env.set_motor_noise(...)`` to
            change the levels during a run.
    """

    def __init__(self, env: gym.Env, motor_noise: Any = None) -> None:
        from myosuite.terms.base_action import MotorNoiseCfg  # noqa: PLC0415

        RecordConstructorArgs.__init__(self, motor_noise=motor_noise)
        gym.Wrapper.__init__(self, env)
        self.motor_noise = MotorNoiseCfg.from_value(motor_noise)
        _stage_host(env, "MotorNoiseWrapper").add_ctrl_stage(
            "noise", muscle_stages.noise_stage(lambda: self.motor_noise)
        )

    def _pickle_kwargs(self) -> dict[str, Any]:
        return {"motor_noise": self.motor_noise}

    def set_motor_noise(self, motor_noise: Any) -> None:
        """Change the noise levels (a cfg, a dict of its fields or ``None``: off).

        Unlike assigning ``env.motor_noise``, which only reaches this wrapper when it
        is the outermost one, the call is forwarded through the wrappers on top.
        """
        from myosuite.terms.base_action import MotorNoiseCfg  # noqa: PLC0415

        self.motor_noise = MotorNoiseCfg.from_value(motor_noise)


class FatigueWrapper(
    _PicklableStage, _ForwardPublicAttributes, RecordConstructorArgs, gym.Wrapper
):
    """3CC-r muscle fatigue on the muscle excitations.

    The fatigue state (``env.muscle_fatigue``, a
    :class:`~myosuite.physics.fatigue.CumulativeFatigue`) is reset with the env
    from its seeded ``np_random``, at the place the env reset its muscle state.

    Args:
        env: Env whose pipeline runs muscle stages.
        fatigue_reset_vec: Initial fatigue (MF) of every muscle at reset, or ``None``.
        fatigue_reset_random: Draw a random initial fatigue at reset.
    """

    def __init__(
        self,
        env: gym.Env,
        fatigue_reset_vec: Any = None,
        fatigue_reset_random: bool = False,
    ) -> None:
        from myosuite.physics.fatigue import CumulativeFatigue  # noqa: PLC0415

        RecordConstructorArgs.__init__(
            self,
            fatigue_reset_vec=fatigue_reset_vec,
            fatigue_reset_random=fatigue_reset_random,
        )
        gym.Wrapper.__init__(self, env)
        host = _stage_host(env, "FatigueWrapper")
        self.fatigue_reset_vec = fatigue_reset_vec
        self.fatigue_reset_random = fatigue_reset_random
        self.muscle_fatigue = CumulativeFatigue(host.model, host.frame_skip, seed=None)
        apply, reset = muscle_stages.fatigue_stage(
            self.muscle_fatigue,
            lambda: (self.fatigue_reset_vec, self.fatigue_reset_random),
        )
        host.add_ctrl_stage("fatigue", apply, reset)

    def _pickle_kwargs(self) -> dict[str, Any]:
        return {
            "fatigue_reset_vec": self.fatigue_reset_vec,
            "fatigue_reset_random": self.fatigue_reset_random,
        }

    def set_fatigue_reset_random(self, fatigue_reset_random: bool) -> None:
        """Randomise the fatigue state at every reset (or stop doing so)."""
        self.fatigue_reset_random = fatigue_reset_random


class ReafferentationWrapper(
    _PicklableStage, _ForwardPublicAttributes, RecordConstructorArgs, gym.Wrapper
):
    """Reafferentation: the command of EIP drives EPL and EIP is silenced."""

    def __init__(self, env: gym.Env) -> None:
        RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        host = _stage_host(env, "ReafferentationWrapper")
        host.add_ctrl_stage(
            "reroute",
            muscle_stages.reroute_stage(host.model, host._stage_actuator_suffix()),
        )


class SarcopeniaWrapper(
    _PicklableStage, _ForwardPublicAttributes, RecordConstructorArgs, gym.Wrapper
):
    """Sarcopenia: scale the muscle forces of the env's model (applied once, at wrapping).

    Wrap an env once: a second ``SarcopeniaWrapper``, or one on a ``myoSarc*`` id (which
    already has it), raises a ``ValueError`` instead of scaling the forces twice.

    Args:
        env: Env with a compiled ``model``.
        force_scale: Factor on the maximum isometric muscle force.

    Raises:
        ValueError: If sarcopenia is already applied to the env.
    """

    def __init__(self, env: gym.Env, force_scale: float = 0.5) -> None:
        from myosuite.core.muscle_conditions import (
            apply_sarcopenia_to_model,  # noqa: PLC0415
        )

        RecordConstructorArgs.__init__(self, force_scale=force_scale)
        gym.Wrapper.__init__(self, env)
        host = env.unwrapped
        if getattr(host, "_sarcopenia_applied", False):
            raise ValueError(
                "Sarcopenia is already applied to this env's model (a myoSarc* id or "
                "another SarcopeniaWrapper); applying it again would scale the forces twice."
            )
        self.force_scale = force_scale
        apply_sarcopenia_to_model(host.model, force_scale=force_scale)
        host._sarcopenia_applied = True


_CONDITION_WRAPPERS = {
    "sarcopenia": "SarcopeniaWrapper",
    "fatigue": "FatigueWrapper",
    "reafferentation": "ReafferentationWrapper",
}


class ExcitationStageWrapper(
    _PicklableStage, _ForwardPublicAttributes, RecordConstructorArgs, gym.Wrapper
):
    """A portable custom stage on the muscle excitations: CPU env and mjlab twin.

    The stage is an :class:`~myosuite.envs.muscle_stages.ExcitationStage` (its ``name``
    and optional ``order`` say where it runs: by default after the built-in stages, in
    installation order; see :class:`CtrlStageWrapper` for the order scale).
    It sees the muscle excitations only and is written for numpy and torch, so the same
    registration configures both backends: the mjlab twin builds the stage from the same
    factory for every env of the scene. Two stages with the same explicit order raise a
    :class:`~myosuite.envs.muscle_stages.StageOrderWarning`.

    Example::

        env = ExcitationStageWrapper(
            make_env("myoElbowPose1D6MRandom-v0"), functools.partial(LowPassStage, 0.3)
        )

    Args:
        env: Env whose pipeline runs muscle stages.
        make_stage: A zero-argument factory (a class, or ``functools.partial``) that
            returns an :class:`~myosuite.envs.muscle_stages.ExcitationStage`; use a
            module-level callable so that the wrapped env can be pickled.
    """

    def __init__(
        self, env: gym.Env, make_stage: Callable[[], muscle_stages.ExcitationStage]
    ) -> None:
        RecordConstructorArgs.__init__(self, make_stage=make_stage)
        gym.Wrapper.__init__(self, env)
        stage = make_stage()
        if stage.name in muscle_stages.STAGE_ORDER:
            raise ValueError(
                f"{stage.name!r} is a built-in stage; use its wrapper, or pick another name."
            )
        self._make_stage, self.stage = make_stage, stage

        def apply(host: Any, ctrl: np.ndarray) -> np.ndarray:
            idx = host._stage_muscle_index()
            ctrl[idx] = stage(ctrl[idx], np)
            return ctrl

        _stage_host(env, "ExcitationStageWrapper").add_ctrl_stage(
            stage.name, apply, lambda host: stage.reset(None), order=stage.order
        )

    def _pickle_kwargs(self) -> dict[str, Any]:
        return {"make_stage": self._make_stage}


class CtrlStageWrapper(
    _PicklableStage, _ForwardPublicAttributes, RecordConstructorArgs, gym.Wrapper
):
    """An env-aware custom stage on the muscle excitations (CPU only).

    The stage runs after the env's action-to-excitation map and the built-in stages
    (noise 20, fatigue 30, reroute 40), right before ``ctrl`` is written, in the order the
    custom stages were installed. To insert it earlier give it an explicit ``order`` (the
    env's map is 10 and the ``ctrl`` write 100). To act on the raw ``[-1, 1]`` action
    instead, use a plain ``gym.ActionWrapper`` on the outside.

    Two stages with the same explicit order run in installation order and raise a
    :class:`~myosuite.envs.muscle_stages.StageOrderWarning`. This stage gets the host
    env and so runs on the CPU only; for a stage that
    also runs on the mjlab twin, write an
    :class:`~myosuite.envs.muscle_stages.ExcitationStage` and use
    :class:`ExcitationStageWrapper`.

    Example::

        def rate_limit(env, ctrl):
            idx = env._stage_muscle_index()
            ctrl[idx] = np.clip(ctrl[idx], 0.0, 0.8)  # cap the excitation
            return ctrl

        env = CtrlStageWrapper(make_env("myoElbowPose1D6MRandom-v0"), rate_limit,
                               name="cap")  # after the built-in stages; order=25 puts it after noise

    Args:
        env: Env whose pipeline runs muscle stages.
        apply: ``apply(env, ctrl) -> ctrl`` with the host env and the excitation vector
            (edit it in place or return a new one). Use a module-level function so that
            the wrapped env can be pickled.
        name: A unique stage name (not one of the built-in names).
        order: ``None`` (default): after the built-in stages, in installation order; or a
            priority strictly between 10 and 100 to insert it earlier.
        reset: Optional ``reset(env)`` called where the env resets its muscle state.

    Raises:
        ValueError: If the name is built-in or already installed, or an explicit order
            is out of range.
    """

    def __init__(
        self,
        env: gym.Env,
        apply: muscle_stages.CtrlStage,
        name: str,
        order: float | None = None,
        reset: muscle_stages.ResetStage | None = None,
    ) -> None:
        RecordConstructorArgs.__init__(
            self, apply=apply, name=name, order=order, reset=reset
        )
        gym.Wrapper.__init__(self, env)
        if name in muscle_stages.STAGE_ORDER:
            raise ValueError(
                f"{name!r} is a built-in stage; use its wrapper, or pick another name."
            )
        self.stage_name, self.stage_order = name, order
        self._apply, self._reset = apply, reset
        _stage_host(env, "CtrlStageWrapper").add_ctrl_stage(
            name, apply, reset, order=order
        )

    def _pickle_kwargs(self) -> dict[str, Any]:
        return {
            "apply": self._apply,
            "name": self.stage_name,
            "order": self.stage_order,
            "reset": self._reset,
        }


def condition_wrapper_specs(condition: str, **kwargs: Any) -> tuple[WrapperSpec, ...]:
    """Registration wrapper specs of a muscle condition.

    Args:
        condition: ``"sarcopenia"``, ``"fatigue"`` or ``"reafferentation"``.
        **kwargs: Constructor arguments of the wrapper (JSON-serialisable).

    Returns:
        One :class:`~gymnasium.envs.registration.WrapperSpec`, for the
        ``additional_wrappers`` of :func:`myosuite.core.registry.register_env`.

    Raises:
        ValueError: If *condition* is unknown.
    """
    if condition not in _CONDITION_WRAPPERS:
        raise ValueError(
            f"Unknown muscle condition {condition!r}; expected one of "
            f"{tuple(_CONDITION_WRAPPERS)}."
        )
    name = _CONDITION_WRAPPERS[condition]
    return (
        WrapperSpec(
            name=name, entry_point=f"myosuite.envs.wrappers:{name}", kwargs=kwargs
        ),
    )


def wrapper_spec(wrapper: type, **kwargs: Any) -> WrapperSpec:
    """Spec of a muscle-command wrapper, for ``EnvConfig.features`` or a registration.

    Args:
        wrapper: A wrapper class of this module (``MotorNoiseWrapper``, ...).
        **kwargs: Its constructor arguments after the env.

    Returns:
        The :class:`~gymnasium.envs.registration.WrapperSpec`.
    """
    return WrapperSpec(
        name=wrapper.__name__,
        entry_point=f"{wrapper.__module__}:{wrapper.__name__}",
        kwargs=kwargs,
    )


def apply_features(env: gym.Env, features: Iterable[WrapperSpec]) -> gym.Env:
    """Wrap *env* in the wrappers of *features* (the stage order is fixed, not the list's).

    Args:
        env: A CPU env (as made by ``gym.make``).
        features: Wrapper specs, e.g. from :func:`wrapper_spec`.

    Returns:
        The wrapped env.
    """
    for spec in features:
        env = load_env_creator(spec.entry_point)(env, **(spec.kwargs or {}))
    return env
