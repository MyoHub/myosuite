# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Read world-model states and clip frames from an mjlab Mimic env, and record rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable

import torch

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor
from myosuite.terms.base_action import sigmoid_muscle_activation


def make_mimic_env(
    clip_path: str | Path,
    num_envs: int,
    device: str = "cuda:0",
    task_id: str = "myoMimicFullbody-v0",
    seed: int | None = None,
) -> Any:
    """Trajectory-mode mjlab Mimic env on *clip_path*."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        bootstrap_myosuite_mjlab_registry,
    )

    bootstrap_myosuite_mjlab_registry(clip_path=str(clip_path))
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = num_envs
    if seed is not None:
        cfg.seed = seed
    return ManagerBasedRlEnv(cfg, device=device)


class MimicEnvIO:
    """World-model state ``[qpos | qvel | act]`` and clip frames of a Mimic env.

    Args:
        env: Constructed mjlab Mimic env (trajectory mode).
        entity_name: Scene entity of the body.
    """

    def __init__(
        self,
        env: Any,
        entity_name: str = "mimic_fullbody_robot",
        variant: str = "fullbody",
    ) -> None:
        from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
            _resolve_mimic_mjlab_ids,
        )

        self.env = env
        self.acc = MjlabEntityAccessor(env, entity_name)
        self.source = _resolve_mimic_mjlab_ids(env, entity_name, variant)["clip_source"]

    def state(self) -> torch.Tensor:
        """Current ``(N, nq + nv + na)`` state."""
        # ``[:]`` unwraps mjlab's TorchArray proxy (torch.cat on it recurses).
        parts = [
            self.acc.joint_pos()[:],
            self.acc.joint_vel()[:],
            self.acc.muscle_act()[:],
        ]
        return torch.cat(parts, dim=-1).float()

    def frame(self) -> torch.Tensor:
        """Current clip frame per env (``n_frames`` once past the clip end)."""
        step = self.env.episode_length_buf
        return self.source.frame_indices(step) + self.source.clip_end(step).long()

    @staticmethod
    def ctrl(action: torch.Tensor) -> torch.Tensor:
        """Muscle control the env's ``sigmoid`` action term applies for *action*."""
        return sigmoid_muscle_activation(action, torch)

    @staticmethod
    def action(ctrl: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Raw action whose control is *ctrl* (inverse of :meth:`ctrl`)."""
        return 0.5 + torch.logit(ctrl.clamp(eps, 1 - eps)) / 5.0


@dataclass
class Rollout:
    """Recorded ``(T, N, ·)`` rollout; ``done[t]`` ends the transition ``t -> t + 1``."""

    state: torch.Tensor  # (T + 1, N, D)
    frame: torch.Tensor  # (T + 1, N)
    ctrl: torch.Tensor  # (T, N, nu)
    reward: torch.Tensor  # (T, N)
    terminated: torch.Tensor  # (T, N)
    truncated: torch.Tensor  # (T, N)

    @property
    def done(self) -> torch.Tensor:
        """Episode ended after step ``t`` (the next state is a reset)."""
        return self.terminated | self.truncated

    def save(self, path: str | Path) -> None:
        """Save as a ``torch`` file (activations and controls in fp16)."""
        d = dict(self.__dict__)
        d["ctrl"] = d["ctrl"].half()
        torch.save({k: v.cpu() for k, v in d.items()}, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> Rollout:
        """Load a file written by :meth:`save` (controls stay fp16)."""
        return cls(**torch.load(path, map_location=device))


@torch.no_grad()
def record(
    io: MimicEnvIO,
    policy: Callable[[torch.Tensor], torch.Tensor],
    steps: int,
    noise_std: float | torch.Tensor = 0.0,
    store_device: str | torch.device = "cpu",
) -> Rollout:
    """Roll out *policy* with Gaussian control noise for *steps* control steps.

    Args:
        io: Env reader.
        policy: Raw-action policy on the ``actor`` observation.
        steps: Control steps to record.
        noise_std: Gaussian noise std on the muscle control (a saturated
            policy is insensitive to raw-action noise), scalar or per env ``(N, 1)``.
        store_device: Where the recorded tensors are kept.
    """
    env = io.env
    obs = env.obs_buf["actor"]
    keep = lambda x: x.to(store_device)  # noqa: E731
    states, frames, ctrls, rews, terms, truncs = (
        [keep(io.state())],
        [keep(io.frame())],
        [],
        [],
        [],
        [],
    )
    for _ in range(steps):
        u = io.ctrl(policy(obs))
        u = (u + torch.randn_like(u) * noise_std).clamp(0.0, 1.0)
        a = io.action(u)
        ctrls.append(keep(io.ctrl(a).half()))
        out, rew, term, trunc, _ = env.step(a)
        obs = out["actor"]
        # mjlab reuses its reward / done buffers across steps.
        rews.append(keep(rew.clone()))
        terms.append(keep(term.clone()))
        truncs.append(keep(trunc.clone()))
        states.append(keep(io.state()))
        frames.append(keep(io.frame()))
    st = lambda x: torch.stack(x, 0)  # noqa: E731
    return Rollout(
        st(states), st(frames), st(ctrls).float(), st(rews), st(terms), st(truncs)
    )
