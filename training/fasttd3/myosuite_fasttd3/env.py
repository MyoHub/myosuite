# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Wrap MyoSuite CPU / mjlab envs in the FastTD3 IsaacLab-style vector API.

FastTD3 (https://github.com/younggyoseo/FastTD3) expects:

- ``reset() -> policy_obs`` with shape ``(N, obs)``
- ``step(actions) -> obs, reward, dones, info``
- ``info["time_outs"]`` and ``info["observations"]["raw"]`` for bootstrap

Actor outputs are ``tanh`` in ``[-1, 1]``. Muscle tasks map that to ``[0, 1]``
inside :meth:`step` so the official FastTD3 networks are unchanged.
"""

from __future__ import annotations

from typing import Any

import torch


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _obs_group(obs: Any, *names: str) -> torch.Tensor:
    keys = list(obs.keys()) if hasattr(obs, "keys") else []
    for name in names:
        if name in obs:
            return _as_tensor(obs[name])
    raise KeyError(f"observation groups {keys} do not contain any of {names}")


_MIMIC_IDS = frozenset(
    {
        "myoMimicFullbody-v0",
        "myoMimicBimanual-v0",
        "myoMuscleMimicFullbody-v0",
        "myoMuscleMimicBimanual-v0",
    }
)


def ensure_mjlab_twin(env_id: str, clip_path: str | None = None) -> None:
    """Register the mjlab twin if it is not already in ``list_tasks()``.

    Static twins (elbow, …) come from the backend import. MuscleMimic ids are
    registered on demand: clip-mode when *clip_path* (or ``MYOSUITE_MIMIC_CLIP``)
    is a file, otherwise random-target mode (no Hugging Face clip).
    """
    import os
    from pathlib import Path

    import myosuite.envs.myo.backends.mjlab  # noqa: F401
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    raw = (
        clip_path
        or os.environ.get("MYOSUITE_MIMIC_CLIP")
        or os.environ.get("MIMIC_CLIP")
    )
    if env_id in list_tasks() and not raw:
        return
    if env_id not in _MIMIC_IDS:
        return
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        default_mimic_clip_on_policy_runner_cfg,
        register_mimic_mjlab_tasks,
        register_mimic_mjlab_tasks_with_clip,
    )

    try:
        if raw:
            path = Path(raw).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Mimic clip path does not exist: {path}")
            from myosuite.core.trajectory_io import load_motion_clip

            nq, nv = (69, 68) if "Bimanual" in env_id else (89, 88)
            clip = load_motion_clip(path, expected_nq=nq, expected_nv=nv)
            register_mimic_mjlab_tasks_with_clip(
                register_mjlab_task=register_mjlab_task,
                rl_cfg_fn=default_mimic_clip_on_policy_runner_cfg,
                clip=clip,
            )
            return
        register_mimic_mjlab_tasks(
            register_mjlab_task=register_mjlab_task,
            rl_cfg_fn=default_mimic_clip_on_policy_runner_cfg,
        )
    except ValueError:
        # Already registered in this process.
        if env_id not in list_tasks():
            raise


class FastTD3GymEnv:
    """IsaacLab-shaped wrapper around CPU ``make_env`` (no Warp / no Hugging Face)."""

    def __init__(
        self,
        env_id: str,
        *,
        num_envs: int = 2,
        device: str | torch.device | None = None,
        seed: int = 1,
        unit_interval_actions: bool = True,
        clip_path: str | None = None,
        max_episode_steps: int | None = None,
    ) -> None:
        del clip_path  # CPU random-target mimic does not take a clip.
        from gymnasium.wrappers import TimeLimit
        from myosuite.core.registry import make_env

        if device is None:
            device = "cpu"
        self.device = torch.device(device)
        self.env_id = env_id
        self.unit_interval_actions = unit_interval_actions
        self._envs = []
        for _ in range(int(num_envs)):
            env = make_env(env_id)
            if max_episode_steps is not None:
                env = TimeLimit(env, max_episode_steps=int(max_episode_steps))
            self._envs.append(env)
        obs_rows = []
        for i, env in enumerate(self._envs):
            obs, _ = env.reset(seed=int(seed) + i)
            obs_rows.append(_as_tensor(obs).reshape(-1))
        stacked = torch.stack(obs_rows).to(self.device, dtype=torch.float32)
        self.num_envs = len(self._envs)
        self.num_obs = int(stacked.shape[-1])
        self.num_actions = int(self._envs[0].action_space.shape[0])
        self.asymmetric_obs = False
        self.num_privileged_obs = 0
        spec = getattr(self._envs[0], "spec", None)
        self.max_episode_steps = int(
            max_episode_steps
            or getattr(spec, "max_episode_steps", None)
            or getattr(self._envs[0], "_max_episode_steps", None)
            or 1000
        )

    def reset(self, random_start_init: bool = True) -> torch.Tensor:
        del random_start_init
        rows = []
        for env in self._envs:
            obs, _ = env.reset()
            rows.append(_as_tensor(obs).reshape(-1))
        return torch.stack(rows).to(self.device, dtype=torch.float32)

    def reset_with_critic_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self.reset(random_start_init=False)
        return obs, obs

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        act = actions.detach().to("cpu", dtype=torch.float32).numpy()
        if self.unit_interval_actions:
            act = (act.clip(-1.0, 1.0) + 1.0) * 0.5
        obs_rows, rewards, dones, truncs = [], [], [], []
        for i, env in enumerate(self._envs):
            obs, rew, terminated, truncated, _info = env.step(act[i])
            done = bool(terminated) or bool(truncated)
            if done:
                obs, _ = env.reset()
            obs_rows.append(_as_tensor(obs).reshape(-1))
            rewards.append(float(rew))
            dones.append(float(done))
            truncs.append(float(truncated))
        obs_t = torch.stack(obs_rows).to(self.device, dtype=torch.float32)
        rew_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        done_t = torch.tensor(dones, device=self.device, dtype=torch.long)
        trunc_t = torch.tensor(truncs, device=self.device, dtype=torch.float32)
        info = {
            "time_outs": trunc_t,
            "observations": {
                "critic": None,
                "raw": {"obs": obs_t, "critic_obs": obs_t},
            },
        }
        return obs_t, rew_t, done_t, info

    def close(self) -> None:
        for env in self._envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()


def make_fast_td3_env(
    env_id: str,
    *,
    backend: str = "auto",
    num_envs: int = 128,
    device: str | torch.device | None = None,
    seed: int = 1,
    unit_interval_actions: bool = True,
    clip_path: str | None = None,
    max_episode_steps: int | None = None,
) -> FastTD3GymEnv | FastTD3MjlabEnv:
    """Build a FastTD3 vector env. ``auto`` is mjlab on CUDA, CPU gym otherwise."""
    if backend == "auto":
        backend = "mjlab" if torch.cuda.is_available() else "cpu"
    if backend == "cpu":
        return FastTD3GymEnv(
            env_id,
            num_envs=num_envs,
            device=device or "cpu",
            seed=seed,
            unit_interval_actions=unit_interval_actions,
            clip_path=clip_path,
            max_episode_steps=max_episode_steps,
        )
    if backend != "mjlab":
        raise ValueError(f"backend must be auto, cpu, or mjlab; got {backend!r}")
    return FastTD3MjlabEnv(
        env_id,
        num_envs=num_envs,
        device=device,
        seed=seed,
        unit_interval_actions=unit_interval_actions,
        clip_path=clip_path,
    )


class FastTD3MjlabEnv:
    """IsaacLab-shaped wrapper around ``make_env(..., backend='mjlab')``."""

    def __init__(
        self,
        env_id: str,
        *,
        num_envs: int = 128,
        device: str | torch.device | None = None,
        seed: int = 1,
        unit_interval_actions: bool = True,
        clip_path: str | None = None,
    ) -> None:
        from myosuite.core.registry import make_env

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        ensure_mjlab_twin(env_id, clip_path=clip_path)
        self.env = make_env(
            env_id, backend="mjlab", num_envs=num_envs, device=str(self.device)
        )
        self.env_id = env_id
        self.unit_interval_actions = unit_interval_actions
        self.num_envs = int(self.env.num_envs)
        self.max_episode_steps = int(self.env.max_episode_length)
        self.num_actions = int(self.env.action_manager.total_action_dim)
        obs0, _ = self.env.reset(seed=seed)
        policy = _obs_group(obs0, "policy", "actor")
        self.num_obs = int(policy.shape[-1])
        if "critic" in obs0:
            critic = _as_tensor(obs0["critic"])
            self.asymmetric_obs = int(critic.shape[-1]) != self.num_obs
            self.num_privileged_obs = (
                int(critic.shape[-1]) if self.asymmetric_obs else 0
            )
            self._critic0 = critic
        else:
            self.asymmetric_obs = False
            self.num_privileged_obs = 0
            self._critic0 = None
        self._obs0 = policy

    def _split_obs(self, obs_dict: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        policy = _obs_group(obs_dict, "policy", "actor")
        if self.asymmetric_obs:
            return policy, _obs_group(obs_dict, "critic")
        return policy, None

    def reset(self, random_start_init: bool = True) -> torch.Tensor:
        obs_dict, _ = self.env.reset()
        obs, _ = self._split_obs(obs_dict)
        if random_start_init:
            high = max(int(self.max_episode_steps), 1)
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=high
            )
        return obs

    def reset_with_critic_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs_dict, _ = self.env.reset()
        obs, critic = self._split_obs(obs_dict)
        if critic is None:
            critic = obs
        return obs, critic

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        env_act = actions
        if self.unit_interval_actions:
            env_act = (actions.clamp(-1.0, 1.0) + 1.0) * 0.5
        obs_dict, rew, terminated, truncated, extras = self.env.step(env_act)
        obs, critic = self._split_obs(obs_dict)
        dones = (terminated | truncated).to(dtype=torch.long)
        info = {
            "time_outs": truncated,
            "observations": {
                "critic": critic,
                "raw": {"obs": obs, "critic_obs": critic},
            },
        }
        if extras:
            info["extras"] = extras
        return obs, rew, dones, info

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
