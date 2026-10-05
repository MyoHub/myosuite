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

:class:`ObservationNormalizeWrapper`
    Online running-mean/std normalisation of observations (no SB3 dependency).

:class:`PerturbationWrapper`
    Injects scheduled external forces/torques to named bodies for perturbation
    experiments (motor control, reactive balance, neuroscience lesion studies).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.utils import RecordConstructorArgs


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

        env = gym.make("myoElbowPose1D6MRandom-v0")
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


class ObservationNormalizeWrapper(RecordConstructorArgs, gym.ObservationWrapper):
    """Online running-mean / running-std observation normalisation.

    Tracks a Welford running mean and variance over observations.
    After *warmup* steps the normalised observation
    ``(obs - mean) / (std + eps)`` is returned; before that the raw
    observation is returned and statistics are still accumulated.

    Matches the behaviour of ``VecNormalize`` from Stable-Baselines3 for a
    single environment without requiring SB3 as a dependency.

    Args:
        env: Wrapped environment.
        warmup: Steps before normalisation is applied (statistics still collected).
        clip: Clip normalised observations to ``[-clip, clip]``.
        eps: Small constant added to std for numerical stability.

    Example::

        env = gym.make("myoElbowPose1D6MRandom-v0")
        env = ObservationNormalizeWrapper(env, warmup=200)
        obs, info = env.reset()
        obs, *_ = env.step(env.action_space.sample())
    """

    def __init__(
        self,
        env: gym.Env,
        warmup: int = 100,
        clip: float = 10.0,
        eps: float = 1e-8,
    ) -> None:
        RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        self._warmup = warmup
        self._clip = clip
        self._eps = eps
        self._n: int = 0
        self._mean: np.ndarray | None = None
        self._M2: np.ndarray | None = None

    def _update(self, obs: np.ndarray) -> None:
        """Welford online update."""
        x = obs.astype(np.float64)
        if self._mean is None:
            self._mean = np.zeros_like(x)
            self._M2 = np.zeros_like(x)
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._M2 += delta * (x - self._mean)

    @property
    def running_mean(self) -> np.ndarray:
        """Current running mean (float32)."""
        if self._mean is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return self._mean.astype(np.float32)

    @property
    def running_std(self) -> np.ndarray:
        """Current running std (float32)."""
        if self._n < 2 or self._M2 is None:
            return np.ones(self.observation_space.shape, dtype=np.float32)
        return np.sqrt(self._M2 / (self._n - 1) + self._eps).astype(np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._update(obs)
        if self._n < self._warmup or self._mean is None:
            return obs
        normalised = (obs - self._mean.astype(np.float32)) / self.running_std
        return np.clip(normalised, -self._clip, self._clip).astype(np.float32)


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

        env = gym.make("myoLegWalk-v0")
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
