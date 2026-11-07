# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""On-device ring buffer of simulator transitions, per environment."""

from __future__ import annotations

import torch


class ReplayBuffer:
    """``capacity`` most recent control steps of ``num_envs`` parallel envs.

    Row ``t`` holds the state and clip frame before step ``t``, the control applied
    and whether the episode ended after it, so ``state[t + 1]`` follows ``state[t]``
    unless ``done[t]``.

    Args:
        num_envs: Parallel environments.
        capacity: Steps kept per env.
        state_dim: World-model state width.
        ctrl_dim: Control width.
        device: Storage device.
    """

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        state_dim: int,
        ctrl_dim: int,
        device: torch.device | str,
    ) -> None:
        self.n, self.cap = num_envs, capacity
        self.state = torch.zeros(capacity, num_envs, state_dim, device=device)
        self.frame = torch.zeros(capacity, num_envs, dtype=torch.long, device=device)
        self.ctrl = torch.zeros(
            capacity, num_envs, ctrl_dim, device=device, dtype=torch.float16
        )
        self.done = torch.ones(capacity, num_envs, dtype=torch.bool, device=device)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: torch.Tensor,
        frame: torch.Tensor,
        ctrl: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Append one vectorised step."""
        i = self.ptr
        self.state[i] = state
        self.frame[i] = frame
        self.ctrl[i] = ctrl.half()
        self.done[i] = done
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def _rows(self, t: torch.Tensor) -> torch.Tensor:
        """Physical rows of logical times ``t`` (0 = oldest kept)."""
        start = (self.ptr - self.size) % self.cap
        return (start + t) % self.cap

    def sample_windows(
        self, batch: int, k: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """``s0 (B, D)``, controls ``(k, B, nu)``, targets ``(k, B, D)`` of windows inside one episode."""
        if self.size <= k:
            return None
        dev = self.state.device
        # Oversample and keep windows without an episode end; the latest row has no successor.
        m = batch * 4
        t = torch.randint(0, self.size - k, (m,), device=dev)
        n = torch.randint(0, self.n, (m,), device=dev)
        steps = torch.arange(k, device=dev)[:, None]
        rows = self._rows(t[None] + steps)
        ok = ~self.done[rows, n[None]].any(0)
        idx = torch.nonzero(ok).flatten()[:batch]
        if idx.numel() < batch // 2:
            return None
        t, n = t[idx], n[idx]
        rows = self._rows(t[None] + steps)
        nxt = self._rows(t[None] + steps + 1)
        return (
            self.state[rows[0], n],
            self.ctrl[rows, n[None]].float(),
            self.state[nxt, n[None]],
        )

    def sample_starts(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Start states and frames for imagined rollouts."""
        dev = self.state.device
        t = torch.randint(0, self.size, (batch,), device=dev)
        n = torch.randint(0, self.n, (batch,), device=dev)
        rows = self._rows(t)
        return self.state[rows, n], self.frame[rows, n]
