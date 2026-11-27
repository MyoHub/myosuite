# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Sticky rollout controller and persistent replay buffer for trajectory-swap DAgger.

Design adapted from Firas Al-Hafez et al., LocoMuJoCo (`robfiras/loco-mujoco`,
MIT License; see `LICENSE-loco-mujoco.txt`): specifically the sticky per-environment
mixture horizon of `DaggerRolloutState` and the combined short-term ring + long-term
`ReservoirBuffer` (`min_include_prob`) in `vanilla_dagger_jax.py`, extended in
PyTorch/NumPy to persist across multi-teacher `(teacher, reset_options)` swaps.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


@dataclass
class DaggerRolloutState:
    """Per-environment sticky teacher/student action-selection controller.

    Each vectorized environment commits to either the active teacher
    (`using_teacher[i] == True`) or the student (`False`) for a sticky horizon
    `steps_remaining[i]` sampled uniformly from `[h_min, h_max]`. An environment
    resamples both `using_teacher[i] ~ Bernoulli(beta)` and
    `steps_remaining[i] ~ UniformInt(h_min, h_max)` whenever its countdown hits
    zero or the environment resets (`done` or chunk swap).

    Attributes:
        using_teacher: Boolean mask of shape `(num_envs,)`.
        steps_remaining: Integer countdown of shape `(num_envs,)` in `[h_min, h_max]`.
        h_min: Minimum sticky horizon length (`>= 1`).
        h_max: Maximum sticky horizon length (`>= h_min`).
    """

    using_teacher: np.ndarray | torch.Tensor
    steps_remaining: np.ndarray | torch.Tensor
    h_min: int = 5
    h_max: int = 15
    _np_rng: np.random.Generator | None = None
    _torch_rng: torch.Generator | None = None

    def __post_init__(self) -> None:
        if self.h_min < 1 or self.h_max < self.h_min:
            raise ValueError(
                f"Invalid sticky horizon bounds: h_min={self.h_min}, h_max={self.h_max}."
            )
        if self.using_teacher.ndim != 1 or self.steps_remaining.ndim != 1:
            raise ValueError("using_teacher and steps_remaining must be 1-D vectors.")
        if self.using_teacher.shape[0] != self.steps_remaining.shape[0]:
            raise ValueError(
                "using_teacher and steps_remaining must have matching length."
            )
        if isinstance(self.using_teacher, np.ndarray) and self._np_rng is None:
            self._np_rng = np.random.default_rng(0)

    @property
    def num_envs(self) -> int:
        """Number of parallel environments tracked by this controller."""
        return int(self.using_teacher.shape[0])

    @property
    def is_torch(self) -> bool:
        """Return `True` when backing arrays are PyTorch tensors."""
        return isinstance(self.using_teacher, torch.Tensor)

    @classmethod
    def create(
        cls,
        num_envs: int,
        beta: float,
        h_min: int = 5,
        h_max: int = 15,
        *,
        backend: Literal["numpy", "torch"] = "numpy",
        seed: int = 0,
        np_rng: np.random.Generator | None = None,
        torch_rng: torch.Generator | None = None,
        device: str | torch.device = "cpu",
    ) -> DaggerRolloutState:
        """Initialize a sticky rollout controller for `num_envs` environments.

        Args:
            num_envs: Number of vectorized environments (`>= 1`).
            beta: Initial probability in `[0, 1]` of selecting the teacher.
            h_min: Minimum sticky segment length (`>= 1`).
            h_max: Maximum sticky segment length (`>= h_min`).
            backend: Backing array type (`"numpy"` or `"torch"`).
            seed: Seed used if no explicit generator is supplied.
            np_rng: Optional NumPy generator for `"numpy"` backend.
            torch_rng: Optional Torch generator for `"torch"` backend.
            device: Target Torch device when `backend == "torch"`.

        Returns:
            Initialized `DaggerRolloutState` with sampled `using_teacher` and
            `steps_remaining`.
        """
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
        _validate_beta(beta)
        if backend == "numpy":
            rng = np_rng if np_rng is not None else np.random.default_rng(seed)
            state = cls(
                using_teacher=np.zeros(num_envs, dtype=bool),
                steps_remaining=np.ones(num_envs, dtype=np.int64),
                h_min=h_min,
                h_max=h_max,
                _np_rng=rng,
            )
            state.reset_all(beta)
            return state
        if backend == "torch":
            dev = torch.device(device)
            gen = torch_rng
            if gen is None:
                gen = torch.Generator(device=dev)
                gen.manual_seed(int(seed))
            state = cls(
                using_teacher=torch.zeros(num_envs, dtype=torch.bool, device=dev),
                steps_remaining=torch.ones(num_envs, dtype=torch.long, device=dev),
                h_min=h_min,
                h_max=h_max,
                _torch_rng=gen,
            )
            state.reset_all(beta)
            return state
        raise ValueError(
            f"Unsupported backend {backend!r}; expected 'numpy' or 'torch'."
        )

    def select_action(
        self,
        teacher_action: np.ndarray | torch.Tensor,
        student_action: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """Select per-environment actions according to `using_teacher`.

        If any environment currently has `steps_remaining <= 0`, this method raises
        an error unless resampled via `advance` or `maybe_resample`.

        Args:
            teacher_action: Teacher actions of shape `(num_envs, ...)`
            student_action: Student actions of matching shape.

        Returns:
            Selected execution actions of matching shape and backend type.
        """
        if self.is_torch:
            assert isinstance(self.using_teacher, torch.Tensor)
            t_act = torch.as_tensor(teacher_action, device=self.using_teacher.device)
            s_act = torch.as_tensor(
                student_action, dtype=t_act.dtype, device=self.using_teacher.device
            )
            mask = self.using_teacher
            while mask.ndim < t_act.ndim:
                mask = mask.unsqueeze(-1)
            return torch.where(mask, t_act, s_act)

        assert isinstance(self.using_teacher, np.ndarray)
        t_np = np.asarray(teacher_action)
        s_np = np.asarray(student_action, dtype=t_np.dtype)
        mask_np = self.using_teacher
        while mask_np.ndim < t_np.ndim:
            mask_np = mask_np[..., None]
        return np.where(mask_np, t_np, s_np)

    def resample_where(
        self, mask: np.ndarray | torch.Tensor, beta: float
    ) -> DaggerRolloutState:
        """Resample `using_teacher` and `steps_remaining` where `mask` is `True`.

        Args:
            mask: Boolean mask of shape `(num_envs,)`.
            beta: Teacher selection probability in `[0, 1]`.

        Returns:
            `self` (mutated in place for convenience).
        """
        _validate_beta(beta)
        if self.is_torch:
            assert isinstance(self.using_teacher, torch.Tensor)
            assert isinstance(self.steps_remaining, torch.Tensor)
            dev = self.using_teacher.device
            m = torch.as_tensor(mask, dtype=torch.bool, device=dev)
            if m.shape != (self.num_envs,):
                raise ValueError(
                    f"Expected mask shape {(self.num_envs,)}, got {tuple(m.shape)}."
                )
            count = int(m.sum().item())
            if count == 0:
                return self
            probs = torch.rand((count,), generator=self._torch_rng, device=dev)
            new_teacher = probs < float(beta)
            new_steps = torch.randint(
                self.h_min,
                self.h_max + 1,
                (count,),
                generator=self._torch_rng,
                device=dev,
                dtype=torch.long,
            )
            self.using_teacher[m] = new_teacher
            self.steps_remaining[m] = new_steps
            return self

        assert isinstance(self.using_teacher, np.ndarray)
        assert isinstance(self.steps_remaining, np.ndarray)
        assert self._np_rng is not None
        m_np = np.asarray(mask, dtype=bool)
        if m_np.shape != (self.num_envs,):
            raise ValueError(
                f"Expected mask shape {(self.num_envs,)}, got {m_np.shape}."
            )
        count = int(m_np.sum())
        if count == 0:
            return self
        new_teacher_np = self._np_rng.random(count) < float(beta)
        new_steps_np = self._np_rng.integers(
            self.h_min, self.h_max + 1, size=count, dtype=np.int64
        )
        self.using_teacher[m_np] = new_teacher_np
        self.steps_remaining[m_np] = new_steps_np
        return self

    def maybe_resample(self, beta: float) -> DaggerRolloutState:
        """Resample any environments whose `steps_remaining <= 0`.

        Args:
            beta: Teacher selection probability in `[0, 1]`.

        Returns:
            `self`.
        """
        expired = self.steps_remaining <= 0
        return self.resample_where(expired, beta)

    def advance(
        self, dones: np.ndarray | torch.Tensor, beta: float
    ) -> DaggerRolloutState:
        """Decrement sticky countdowns after an environment step and resample as needed.

        Resamples `using_teacher ~ Bernoulli(beta)` and
        `steps_remaining ~ UniformInt(h_min, h_max)` wherever `steps_remaining`
        reaches `<= 0` or `dones` is `True`.

        Args:
            dones: Episode termination/truncation flags of shape `(num_envs,)`.
            beta: Teacher selection probability in `[0, 1]` for newly sampled segments.

        Returns:
            `self`.
        """
        _validate_beta(beta)
        self.steps_remaining = self.steps_remaining - 1
        if self.is_torch:
            assert isinstance(self.steps_remaining, torch.Tensor)
            done_mask = torch.as_tensor(
                dones, dtype=torch.bool, device=self.steps_remaining.device
            )
            need_resample = (self.steps_remaining <= 0) | done_mask
        else:
            assert isinstance(self.steps_remaining, np.ndarray)
            done_mask_np = np.asarray(dones, dtype=bool)
            need_resample = (self.steps_remaining <= 0) | done_mask_np
        return self.resample_where(need_resample, beta)

    def reset_all(self, beta: float) -> DaggerRolloutState:
        """Resample all environments (called on initial reset and chunk swaps).

        Args:
            beta: Teacher selection probability in `[0, 1]`.

        Returns:
            `self`.
        """
        if self.is_torch:
            assert isinstance(self.using_teacher, torch.Tensor)
            mask: np.ndarray | torch.Tensor = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.using_teacher.device
            )
        else:
            mask = np.ones(self.num_envs, dtype=bool)
        return self.resample_where(mask, beta)


def _validate_beta(beta: float) -> None:
    if not (0.0 <= float(beta) <= 1.0):
        raise ValueError(f"beta must lie in [0, 1], got {beta}.")


@dataclass(frozen=True)
class DaggerBatch:
    """Minibatch drawn from `PersistentDaggerBuffer`.

    Attributes:
        obs: Observations of shape `(B, obs_dim)` (`float32`).
        teacher_action: Target teacher actions of shape `(B, act_dim)` (`float32`).
        teacher_value: Optional teacher state values of shape `(B,)` (`float32`).
        has_value: Boolean mask `(B,)` indicating valid `teacher_value` targets.
        source_id: Integer tag `(B,)` identifying the `(teacher, options)` pair.
    """

    obs: torch.Tensor
    teacher_action: torch.Tensor
    teacher_value: torch.Tensor
    has_value: torch.Tensor
    source_id: torch.Tensor


class PersistentDaggerBuffer:
    """Short-term FIFO ring buffer + optional long-term reservoir memory.

    Stores `(obs, teacher_action, teacher_value)` tuples across multi-teacher
    `(teacher, reset_options)` chunk swaps. The short-term ring buffer retains the
    most recent `ring_buffer_size` transitions, while the optional long-term
    reservoir (`long_term_buffer_size > 0`) maintains cross-chunk coverage using
    LocoMuJoCo's `ReservoirBuffer` inclusion rule
    `p_include = max(long_term_buffer_size / n_seen, min_include_prob)`.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        ring_buffer_size: int = 65_536,
        long_term_buffer_size: int = 16_384,
        min_include_prob: float = 0.05,
        *,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        if obs_dim < 1 or act_dim < 1:
            raise ValueError(
                f"obs_dim ({obs_dim}) and act_dim ({act_dim}) must be >= 1."
            )
        if ring_buffer_size < 1:
            raise ValueError(f"ring_buffer_size must be >= 1, got {ring_buffer_size}.")
        if long_term_buffer_size < 0:
            raise ValueError(
                f"long_term_buffer_size must be >= 0, got {long_term_buffer_size}."
            )
        if not (0.0 <= min_include_prob <= 1.0):
            raise ValueError(
                f"min_include_prob must be in [0, 1], got {min_include_prob}."
            )

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.ring_buffer_size = int(ring_buffer_size)
        self.long_term_buffer_size = int(long_term_buffer_size)
        self.min_include_prob = float(min_include_prob)
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)

        self._ring_obs = torch.zeros(
            (self.ring_buffer_size, self.obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._ring_act = torch.zeros(
            (self.ring_buffer_size, self.act_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._ring_val = torch.zeros(
            (self.ring_buffer_size,), dtype=torch.float32, device=self.device
        )
        self._ring_has_val = torch.zeros(
            (self.ring_buffer_size,), dtype=torch.bool, device=self.device
        )
        self._ring_src = torch.zeros(
            (self.ring_buffer_size,), dtype=torch.long, device=self.device
        )
        self._ring_ptr = 0
        self._ring_size = 0

        self._res_obs = torch.zeros(
            (self.long_term_buffer_size, self.obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._res_act = torch.zeros(
            (self.long_term_buffer_size, self.act_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._res_val = torch.zeros(
            (self.long_term_buffer_size,), dtype=torch.float32, device=self.device
        )
        self._res_has_val = torch.zeros(
            (self.long_term_buffer_size,), dtype=torch.bool, device=self.device
        )
        self._res_src = torch.zeros(
            (self.long_term_buffer_size,), dtype=torch.long, device=self.device
        )
        self._reservoir_size = 0
        self._total_seen = 0

    @property
    def ring_size(self) -> int:
        """Current number of valid entries in the short-term ring buffer."""
        return self._ring_size

    @property
    def reservoir_size(self) -> int:
        """Current number of valid entries in the long-term reservoir buffer."""
        return self._reservoir_size

    @property
    def size(self) -> int:
        """Total active transitions across ring and reservoir buffers."""
        return self._ring_size + self._reservoir_size

    @property
    def total_seen(self) -> int:
        """Cumulative number of transitions added across all chunks and swaps."""
        return self._total_seen

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        """Explicitly empty both ring and reservoir memory."""
        self._ring_ptr = 0
        self._ring_size = 0
        self._reservoir_size = 0
        self._total_seen = 0

    def add(
        self,
        obs: np.ndarray | torch.Tensor,
        teacher_action: np.ndarray | torch.Tensor,
        teacher_value: np.ndarray | torch.Tensor | None = None,
        *,
        source_id: int | np.ndarray | torch.Tensor = 0,
    ) -> None:
        """Insert one transition or a batch of transitions into ring + reservoir.

        Args:
            obs: Observation(s) of shape `(obs_dim,)` or `(B, obs_dim)`.
            teacher_action: Teacher action(s) of shape `(act_dim,)` or `(B, act_dim)`.
            teacher_value: Optional teacher value(s) of shape `()`, `(B,)`, or `(B, 1)`.
            source_id: Integer identifier(s) for the active `(teacher, options)` pair.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(teacher_action, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        if act_t.ndim == 1:
            act_t = act_t.unsqueeze(0)

        batch_n = int(obs_t.shape[0])
        if batch_n == 0:
            return
        if obs_t.shape != (batch_n, self.obs_dim):
            raise ValueError(
                f"Expected obs shape {(batch_n, self.obs_dim)}, got {tuple(obs_t.shape)}."
            )
        if act_t.shape != (batch_n, self.act_dim):
            raise ValueError(
                f"Expected action shape {(batch_n, self.act_dim)}, got {tuple(act_t.shape)}."
            )

        if teacher_value is None:
            val_t = torch.zeros((batch_n,), dtype=torch.float32, device=self.device)
            has_val_t = torch.zeros((batch_n,), dtype=torch.bool, device=self.device)
        else:
            val_t = torch.as_tensor(
                teacher_value, dtype=torch.float32, device=self.device
            ).reshape(-1)
            if val_t.shape != (batch_n,):
                raise ValueError(
                    f"Expected teacher_value length {batch_n}, got {tuple(val_t.shape)}."
                )
            has_val_t = torch.ones((batch_n,), dtype=torch.bool, device=self.device)

        if isinstance(source_id, int):
            src_t = torch.full(
                (batch_n,), int(source_id), dtype=torch.long, device=self.device
            )
        else:
            src_t = torch.as_tensor(
                source_id, dtype=torch.long, device=self.device
            ).reshape(-1)
            if src_t.shape != (batch_n,):
                raise ValueError(
                    f"Expected source_id length {batch_n}, got {tuple(src_t.shape)}."
                )

        self._write_ring(obs_t, act_t, val_t, has_val_t, src_t)
        if self.long_term_buffer_size > 0:
            self._write_reservoir(obs_t, act_t, val_t, has_val_t, src_t)
        else:
            self._total_seen += batch_n

    def _write_ring(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        val_t: torch.Tensor,
        has_val_t: torch.Tensor,
        src_t: torch.Tensor,
    ) -> None:
        batch_n = int(obs_t.shape[0])
        cap = self.ring_buffer_size
        if batch_n >= cap:
            obs_t = obs_t[-cap:]
            act_t = act_t[-cap:]
            val_t = val_t[-cap:]
            has_val_t = has_val_t[-cap:]
            src_t = src_t[-cap:]
            batch_n = cap

        end = self._ring_ptr + batch_n
        if end <= cap:
            sl = slice(self._ring_ptr, end)
            self._ring_obs[sl] = obs_t
            self._ring_act[sl] = act_t
            self._ring_val[sl] = val_t
            self._ring_has_val[sl] = has_val_t
            self._ring_src[sl] = src_t
        else:
            first = cap - self._ring_ptr
            second = end - cap
            self._ring_obs[self._ring_ptr :] = obs_t[:first]
            self._ring_act[self._ring_ptr :] = act_t[:first]
            self._ring_val[self._ring_ptr :] = val_t[:first]
            self._ring_has_val[self._ring_ptr :] = has_val_t[:first]
            self._ring_src[self._ring_ptr :] = src_t[:first]

            self._ring_obs[:second] = obs_t[first:]
            self._ring_act[:second] = act_t[first:]
            self._ring_val[:second] = val_t[first:]
            self._ring_has_val[:second] = has_val_t[first:]
            self._ring_src[:second] = src_t[first:]

        self._ring_ptr = end % cap
        self._ring_size = min(cap, self._ring_size + batch_n)

    def _write_reservoir(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        val_t: torch.Tensor,
        has_val_t: torch.Tensor,
        src_t: torch.Tensor,
    ) -> None:
        batch_n = int(obs_t.shape[0])
        cap = self.long_term_buffer_size
        offset = 0

        if self._reservoir_size < cap:
            take = min(cap - self._reservoir_size, batch_n)
            sl = slice(self._reservoir_size, self._reservoir_size + take)
            self._res_obs[sl] = obs_t[:take]
            self._res_act[sl] = act_t[:take]
            self._res_val[sl] = val_t[:take]
            self._res_has_val[sl] = has_val_t[:take]
            self._res_src[sl] = src_t[:take]
            self._reservoir_size += take
            self._total_seen += take
            offset = take

        rem = batch_n - offset
        if rem <= 0:
            return

        counts = np.arange(
            self._total_seen + 1, self._total_seen + rem + 1, dtype=np.float64
        )
        probs = np.clip(
            np.maximum(float(cap) / counts, self.min_include_prob), 0.0, 1.0
        )
        keep = self._rng.random(rem) < probs
        self._total_seen += rem

        if not np.any(keep):
            return

        keep_indices = np.flatnonzero(keep) + offset
        slots = self._rng.integers(0, cap, size=keep_indices.shape[0], dtype=np.int64)
        for src_idx, dst_slot in zip(keep_indices.tolist(), slots.tolist()):
            self._res_obs[dst_slot] = obs_t[src_idx]
            self._res_act[dst_slot] = act_t[src_idx]
            self._res_val[dst_slot] = val_t[src_idx]
            self._res_has_val[dst_slot] = has_val_t[src_idx]
            self._res_src[dst_slot] = src_t[src_idx]

    def _gather(self, combined_indices: np.ndarray) -> DaggerBatch:
        idx_np = np.asarray(combined_indices, dtype=np.int64)
        in_ring = idx_np < self._ring_size
        if np.all(in_ring):
            idx_t = torch.as_tensor(idx_np, dtype=torch.long, device=self.device)
            return DaggerBatch(
                obs=self._ring_obs[idx_t],
                teacher_action=self._ring_act[idx_t],
                teacher_value=self._ring_val[idx_t],
                has_value=self._ring_has_val[idx_t],
                source_id=self._ring_src[idx_t],
            )
        if not np.any(in_ring):
            res_idx = torch.as_tensor(
                idx_np - self._ring_size, dtype=torch.long, device=self.device
            )
            return DaggerBatch(
                obs=self._res_obs[res_idx],
                teacher_action=self._res_act[res_idx],
                teacher_value=self._res_val[res_idx],
                has_value=self._res_has_val[res_idx],
                source_id=self._res_src[res_idx],
            )

        b = idx_np.shape[0]
        obs_out = torch.empty(
            (b, self.obs_dim), dtype=torch.float32, device=self.device
        )
        act_out = torch.empty(
            (b, self.act_dim), dtype=torch.float32, device=self.device
        )
        val_out = torch.empty((b,), dtype=torch.float32, device=self.device)
        has_out = torch.empty((b,), dtype=torch.bool, device=self.device)
        src_out = torch.empty((b,), dtype=torch.long, device=self.device)

        ring_pos = torch.as_tensor(
            np.flatnonzero(in_ring), dtype=torch.long, device=self.device
        )
        ring_idx = torch.as_tensor(
            idx_np[in_ring], dtype=torch.long, device=self.device
        )
        obs_out[ring_pos] = self._ring_obs[ring_idx]
        act_out[ring_pos] = self._ring_act[ring_idx]
        val_out[ring_pos] = self._ring_val[ring_idx]
        has_out[ring_pos] = self._ring_has_val[ring_idx]
        src_out[ring_pos] = self._ring_src[ring_idx]

        res_mask = ~in_ring
        res_pos = torch.as_tensor(
            np.flatnonzero(res_mask), dtype=torch.long, device=self.device
        )
        res_idx = torch.as_tensor(
            idx_np[res_mask] - self._ring_size, dtype=torch.long, device=self.device
        )
        obs_out[res_pos] = self._res_obs[res_idx]
        act_out[res_pos] = self._res_act[res_idx]
        val_out[res_pos] = self._res_val[res_idx]
        has_out[res_pos] = self._res_has_val[res_idx]
        src_out[res_pos] = self._res_src[res_idx]

        return DaggerBatch(
            obs=obs_out,
            teacher_action=act_out,
            teacher_value=val_out,
            has_value=has_out,
            source_id=src_out,
        )

    def sample(self, batch_size: int) -> DaggerBatch:
        """Sample a uniform minibatch across active ring + reservoir entries.

        Args:
            batch_size: Number of samples (`>= 1`).

        Returns:
            Sampled `DaggerBatch`.
        """
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty PersistentDaggerBuffer.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        replace = batch_size > self.size
        indices = self._rng.choice(self.size, size=batch_size, replace=replace)
        return self._gather(indices)

    def iter_minibatches(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        max_batches: int | None = None,
    ) -> Iterator[DaggerBatch]:
        """Iterate over active ring + reservoir transitions in minibatches.

        Args:
            batch_size: Minibatch size (`>= 1`).
            shuffle: Shuffle indices using the buffer's seeded RNG.
            drop_last: Drop a trailing incomplete batch when `size > batch_size`.
            max_batches: Optional cap on yielded minibatches per epoch.

        Yields:
            `DaggerBatch` instances on `self.device`.
        """
        if self.size == 0:
            return
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

        order = (
            self._rng.permutation(self.size)
            if shuffle
            else np.arange(self.size, dtype=np.int64)
        )
        yielded = 0
        for start in range(0, self.size, batch_size):
            chunk = order[start : start + batch_size]
            if drop_last and chunk.shape[0] < batch_size and self.size >= batch_size:
                break
            yield self._gather(chunk)
            yielded += 1
            if max_batches is not None and yielded >= max_batches:
                break

    def ring_source_ids(self) -> np.ndarray:
        """Return source tags currently stored in the short-term ring buffer."""
        return self._ring_src[: self._ring_size].detach().cpu().numpy().copy()

    def reservoir_source_ids(self) -> np.ndarray:
        """Return source tags currently stored in the long-term reservoir buffer."""
        return self._res_src[: self._reservoir_size].detach().cpu().numpy().copy()

    def all_source_ids(self) -> np.ndarray:
        """Return all source tags across both ring and reservoir memory."""
        return np.concatenate([self.ring_source_ids(), self.reservoir_source_ids()])
