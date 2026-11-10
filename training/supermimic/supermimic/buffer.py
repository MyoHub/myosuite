# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# training/supermimic/ for the notices of the code it includes.
"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env).

Port of supermimic_vibed's ``buffer.py``. Muscle-sized fields (700 activations, 700 excitations) are
stored in fp16, and the action before a transition is read from the previous row instead of stored.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/rl/supermimic/buffer.py``.
"""

from __future__ import annotations

import torch


class ReplayBuffer:
    def __init__(
        self,
        T: int,
        N: int,
        body_dim: int,
        num_muscles: int,
        num_touch: int,
        device: str,
    ):
        self.T, self.N, self.device = T, N, device
        f16 = torch.float16
        self.fields = {
            "s": ((body_dim,), torch.float32),  # body state before the step
            "act": ((num_muscles,), f16),  # muscle activations before the step
            "touch": ((num_touch,), torch.float32),  # foot touch before the step
            "a": ((num_muscles,), f16),  # excitation applied during the step
            "s1": (
                (body_dim,),
                torch.float32,
            ),  # body state after the step (activations follow analytically)
            "touch1": ((num_touch,), torch.float32),  # foot touch after the step
            "r": ((), torch.float32),
            "f": ((), torch.long),
            "ep_step": ((), torch.int32),
            "done": ((), torch.bool),
        }
        self.data = {
            k: torch.zeros((T, N) + shape, dtype=dt, device=device)
            for k, (shape, dt) in self.fields.items()
        }
        self.ptr, self.size = 0, 0
        # cuda_graph (BlinkMimic): device copies of ptr and size, so that sampling captured in a CUDA graph follows the
        # write head and the fill level (uniform draws scaled by them instead of randint with host bounds: the same
        # distribution, another random stream); use_device_ptr() switches them on
        self.ptr_t = self.size_t = None
        # overlap_collect: sampling reads a snapshot of the write head and fill level (taken by snapshot() while no
        # collection runs), so that updates running concurrently with a collection never see its writes; the fill
        # level is capped at T - reserve, keeping the next `reserve` rows (the ones being overwritten) out of reach
        self.sample_ptr_t = self.sample_size_t = None
        self.reserve = 0

    def use_device_ptr(self):
        self.ptr_t = torch.full((), self.ptr, dtype=torch.long, device=self.device)
        self.size_t = torch.full((), self.size, dtype=torch.long, device=self.device)

    def use_sample_snapshot(self, reserve: int):
        """Sample from snapshot() instead of the live write head (needs use_device_ptr). `reserve`: the most steps a
        collection writes between two snapshots."""
        assert self.ptr_t is not None and 0 < reserve < self.T
        self.sample_ptr_t, self.sample_size_t = self.ptr_t.clone(), self.size_t.clone()
        self.reserve = reserve
        self.snapshot()

    def snapshot(self):
        """The write head and fill level that sampling reads until the next snapshot (on the current stream)."""
        self.sample_ptr_t.copy_(self.ptr_t)
        self.sample_size_t.copy_(torch.clamp(self.size_t, max=self.T - self.reserve))

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)
        if self.ptr_t is not None:
            self.ptr_t.fill_(self.ptr)
            self.size_t.fill_(self.size)

    def add_device(self, tr: dict):
        """add() at the device-side write head (collect_graph: capturable). Advances ptr_t / size_t on the device; the
        host-side ptr / size are advanced by advance_host() after each replay."""
        row = self.ptr_t.view(1)
        for k in self.fields:
            self.data[k].index_copy_(0, row, tr[k].unsqueeze(0).to(self.data[k].dtype))
        self.ptr_t.copy_((self.ptr_t + 1) % self.T)
        self.size_t.copy_(torch.clamp(self.size_t + 1, max=self.T))

    def advance_host(self):
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _size(self):
        if self.sample_size_t is not None:
            return self.sample_size_t
        return self.size_t if self.size_t is not None else self.size

    def _randint(self, lo: int, hi, B: int) -> torch.Tensor:
        """Uniform integers in [lo, hi), hi a host int or (device-side bounds) a 0-dim tensor."""
        if self.size_t is None:
            return torch.randint(lo, hi, (B,), device=self.device)
        n = hi - lo
        return lo + torch.minimum((torch.rand(B, device=self.device) * n).long(), n - 1)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        ptr = (
            self.sample_ptr_t
            if self.sample_ptr_t is not None
            else self.ptr_t
            if self.ptr_t is not None
            else self.ptr
        )
        return (ptr - self._size() + age) % self.T

    def _get(self, age: torch.Tensor, env: torch.Tensor, keys) -> dict:
        rows = self._row(age)
        out = {k: self.data[k][rows, env] for k in keys}
        for k in ("act", "a"):
            if k in out:
                out[k] = out[k].float()
        return out

    def _last_action(self, age: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """The action that led into the state at ``age`` (zero at an episode start); needs age >= 1."""
        first = self.data["ep_step"][self._row(age), env] == 0
        prev = self.data["a"][self._row(age - 1), env].float()
        return torch.where(first.unsqueeze(-1), 0.0, prev)

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = self._randint(0, self._size() - k + 1, B)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        out = self._get(
            ages,
            env.unsqueeze(-1).expand_as(ages),
            ("s", "act", "a", "s1", "touch1", "done"),
        )
        # transition j is valid if no episode ended at transitions 0..j-1
        d = out["done"].float()
        out["mask"] = torch.cumprod(
            torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1
        )
        return out

    def sample_starts(self, B: int, hist: int):
        """Start points for imagination plus their observation history (oldest first, clamped to the episode)."""
        age = self._randint(
            max(hist, 1), self._size(), B
        )  # row 0 has no previous action
        env = torch.randint(0, self.N, (B,), device=self.device)
        keys = ("s", "act", "touch", "f", "ep_step")
        start = self._get(age, env, keys)
        start["last_a"] = self._last_action(age, env)
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(
            back.unsqueeze(0), start["ep_step"].long().unsqueeze(-1)
        )  # stay inside the episode
        hage = age.unsqueeze(-1) - back
        henv = env.unsqueeze(-1).expand_as(hage)
        hist_d = self._get(hage, henv, keys)
        hist_d["last_a"] = self._last_action(hage, henv)
        return start, hist_d
