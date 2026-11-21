# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU Gymnasium environment adapter for DreamerV3 (embodied.Env)."""

from __future__ import annotations

from collections.abc import Sequence
import functools
from typing import Any

import elements
import embodied
import numpy as np

from myosuite_dreamerv3.compat import import_myosuite


class DreamerEnv(embodied.Env):
    """Wrap a registered MyoSuite CPU environment in DreamerV3's ``embodied.Env`` API.

    Args:
        task: Registered MyoSuite environment ID (with or without ``myo`` prefix).
        repeat: Action-repeat count per agent step.
        proprio: Include proprioceptive vector/dict observations.
        image: Include rendered RGB frames under ``"image"``.
        log_image: Include rendered RGB frames under ``"log/image"`` only.
        size: ``(height, width)`` of offscreen rendered RGB frames.
        camera: MuJoCo camera name or integer ID (``None`` / ``-1`` for default free camera).
        device_id: Offscreen rendering device index for legacy renderers.
        seed: Initial episode reset seed.
        features: Optional sequence of ``WrapperSpec`` or ``--feature`` strings
            (e.g. ``("fatigue",)`` or ``('motor-noise={"constant_std": 0.05}',)``).
        **kwargs: Forwarded to ``myosuite.make_env``.
    """

    def __init__(
        self,
        task: str,
        repeat: int = 1,
        proprio: bool = True,
        image: bool = False,
        log_image: bool = False,
        size: tuple[int, int] = (64, 64),
        camera: str | int | None = None,
        device_id: int = 0,
        seed: int | None = None,
        features: Sequence[Any] = (),
        **kwargs: Any,
    ) -> None:
        if not (proprio or image):
            raise ValueError("At least one of proprio or image must be True.")
        myosuite = import_myosuite()
        registry = myosuite.gym_registry_specs()
        if task not in registry and f"myo{task}" in registry:
            task = f"myo{task}"

        self._task = task
        if features:
            from myosuite.core.config import EnvConfig
            from myosuite.utils.feature_cli import feature_spec

            parsed_features = tuple(
                feature_spec(f) if isinstance(f, str) else f for f in features
            )
            self._env = myosuite.make_env(
                EnvConfig(env_id=task, features=parsed_features), **kwargs
            )
        else:
            self._env = myosuite.make_env(task, **kwargs)

        self._repeat = int(repeat)
        self._proprio = bool(proprio)
        self._image = bool(image)
        self._log_image = bool(log_image)
        self._size = (int(size[0]), int(size[1]))
        self._camera = None if camera in (-1, "") else camera
        self._device_id = int(device_id)
        self._seed = seed
        self._renderer: Any = None
        self._done = True

    @property
    def env(self) -> Any:
        """Underlying MyoSuite Gymnasium environment."""
        return self._env

    @functools.cached_property
    def obs_space(self) -> dict[str, elements.Space]:
        """DreamerV3 observation space dictionary."""
        spaces: dict[str, elements.Space] = {
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            "log/solved": elements.Space(np.float32),
            "log/rwd_dense": elements.Space(np.float32),
            "log/rwd_sparse": elements.Space(np.float32),
        }
        if self._proprio:
            obs_sp = self._env.observation_space
            if hasattr(obs_sp, "spaces"):
                for key, sp in obs_sp.spaces.items():
                    spaces[key.replace("/", "_")] = elements.Space(np.float32, sp.shape)
            else:
                spaces["proprio"] = elements.Space(np.float32, obs_sp.shape)
        if self._image:
            spaces["image"] = elements.Space(np.uint8, (*self._size, 3))
        elif self._log_image:
            spaces["log/image"] = elements.Space(np.uint8, (*self._size, 3))
        return spaces

    @functools.cached_property
    def act_space(self) -> dict[str, elements.Space]:
        """DreamerV3 action space dictionary."""
        act_sp = self._env.action_space
        return {
            "action": elements.Space(
                np.float32,
                act_sp.shape,
                np.asarray(act_sp.low, np.float32),
                np.asarray(act_sp.high, np.float32),
            ),
            "reset": elements.Space(bool),
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """Advance or reset the environment for one DreamerV3 transition."""
        if action["reset"] or self._done:
            reset_kw = {"seed": int(self._seed)} if self._seed is not None else {}
            self._seed = None
            obs, info = self._env.reset(**reset_kw)
            self._done = False
            return self._obs(obs, 0.0, info, is_first=True)

        act = np.asarray(action["action"], self._env.action_space.dtype)
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        for _ in range(self._repeat):
            obs, rew, term, trunc, info = self._env.step(act)
            terminated = terminated or bool(term)
            truncated = truncated or bool(trunc)
            reward += float(rew)
            if terminated or truncated:
                break

        self._done = terminated or truncated
        return self._obs(
            obs,
            reward,
            info,
            is_last=self._done,
            is_terminal=terminated,
        )

    def _obs(
        self,
        obs: Any,
        reward: float,
        info: dict[str, Any] | None,
        is_first: bool = False,
        is_last: bool = False,
        is_terminal: bool = False,
    ) -> dict[str, Any]:
        info = info or {}
        rwd_dict = info.get("rwd_dict", {})
        solved = info.get("solved", rwd_dict.get("solved", 0.0))
        rwd_dense = info.get("rwd_dense", rwd_dict.get("dense", reward))
        rwd_sparse = info.get("rwd_sparse", rwd_dict.get("sparse", 0.0))
        out: dict[str, Any] = {
            "reward": np.float32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            "log/solved": np.float32(np.asarray(solved).mean()),
            "log/rwd_dense": np.float32(np.asarray(rwd_dense).mean()),
            "log/rwd_sparse": np.float32(np.asarray(rwd_sparse).mean()),
        }
        if self._proprio:
            if isinstance(obs, dict):
                for key, val in obs.items():
                    out[key.replace("/", "_")] = np.asarray(val, np.float32)
            else:
                out["proprio"] = np.asarray(obs, np.float32)
        if self._image:
            out["image"] = self._render()
        elif self._log_image:
            out["log/image"] = self._render()
        return out

    def _render(self) -> np.ndarray:
        unwrapped = self._env.unwrapped
        height, width = self._size
        if hasattr(unwrapped, "model") and hasattr(unwrapped, "data"):
            import mujoco

            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    unwrapped.model, height=height, width=width
                )
            cam = -1 if self._camera is None else self._camera
            self._renderer.update_scene(unwrapped.data, camera=cam)
            frame = self._renderer.render()
        elif hasattr(unwrapped, "mj_renderer"):
            frame = unwrapped.mj_renderer.render_offscreen(
                width=width,
                height=height,
                camera_id=self._camera,
                device_id=self._device_id,
            )
        else:
            frame = self._env.render()
        return np.asarray(frame, np.uint8)

    def close(self) -> None:
        """Close offscreen renderer and underlying environment."""
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        try:
            unwrapped = self._env.unwrapped
            if (
                not hasattr(unwrapped, "model")
                and hasattr(unwrapped, "mj_renderer")
                and hasattr(unwrapped.mj_renderer, "close")
            ):
                unwrapped.mj_renderer.close()
        except Exception:
            pass
        self._env.close()


def wrap_dreamer_env(env: embodied.Env) -> embodied.Env:
    """Apply DreamerV3's standard action normalization, dtype, space, and clip wrappers.

    Args:
        env: Unwrapped ``DreamerEnv``.

    Returns:
        Wrapped ``embodied.Env`` accepting ``[-1, 1]`` continuous actions.
    """
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env
