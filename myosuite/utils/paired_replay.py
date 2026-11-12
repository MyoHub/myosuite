# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""In-memory snapshots for paired headless CPU experiments on one fixed model."""

from __future__ import annotations

import copy
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


@dataclass
class ReplayPolicy:
    """Explicit controller state contract, including history and owned RNGs.

    Args:
        action: Consume a raw observation and return the env's action.
        get_state: Return all mutable inference state (not just network weights).
        set_state: Restore an independent copy of that state.
    """

    action: Callable[[Any], np.ndarray]
    get_state: Callable[[], Any]
    set_state: Callable[[Any], None]

    @classmethod
    def stateless(cls, action: Callable[[Any], np.ndarray]) -> ReplayPolicy:
        """Declare that *action* has no mutable inference state."""
        return cls(action, lambda: None, lambda state: None)


def _model_bytes(model: mujoco.MjModel) -> bytes:
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return buffer.tobytes()


class ReplaySnapshot:
    """Save simulator, Python task/wrapper, controller and RNG state for exact replay.

    This is an in-memory runtime object, never a pickle checkpoint. The compiled
    model must stay fixed; model edits are rejected. Use headless envs: active
    renderer resources are unsupported. Env Python state must support deepcopy;
    native model/data/spec handles and the read-only CPU accessor retain identity.
    Controller owners explicitly provide all mutable inference state callbacks.
    """

    @classmethod
    def capture(cls, env: gym.Env, policy: ReplayPolicy) -> ReplaySnapshot:
        """Capture the initialized env and controller before either paired rollout."""
        self = cls()
        self._layers = []
        layer = env
        while True:
            self._layers.append(layer)
            if not isinstance(layer, gym.Wrapper):
                break
            layer = layer.env
        self._model, self._data = layer.model, layer.data
        if not isinstance(self._model, mujoco.MjModel) or not isinstance(
            self._data, mujoco.MjData
        ):
            raise TypeError("Paired replay requires native MuJoCo model/data.")
        self._memo = {id(item): item for item in self._layers}
        for item in self._layers:
            for name, value in vars(item).items():
                if name in ("_renderer", "_mj_renderer_compat") and value is not None:
                    raise ValueError("Paired replay requires a headless env.")
                if (
                    isinstance(value, (mujoco.MjModel, mujoco.MjData, mujoco.MjSpec))
                    or name == "_accessor"
                ):
                    self._memo[id(value)] = value
        self._python = [
            copy.deepcopy(vars(item), self._memo.copy()) for item in self._layers
        ]
        self._native = mujoco.MjData(self._model)
        mujoco.mj_copyData(self._native, self._model, self._data)
        self._model_blob = _model_bytes(self._model)
        self._policy = policy
        self._policy_state = copy.deepcopy(policy.get_state())
        self._random = random.getstate()
        self._numpy = np.random.get_state()
        torch = sys.modules.get("torch")
        self._torch = (
            torch.random.get_rng_state().clone() if torch is not None else None
        )
        self._cuda = (
            torch.cuda.get_rng_state_all()
            if torch is not None and torch.cuda.is_initialized()
            else None
        )
        return self

    @property
    def policy_state(self) -> Any:
        """An independent copy of the captured controller state."""
        return copy.deepcopy(self._policy_state)

    def restore(self) -> None:
        """Restore an independent copy of the captured state without advancing physics."""
        if _model_bytes(self._model) != self._model_blob:
            raise ValueError("Paired replay model changed after the snapshot.")
        torch = sys.modules.get("torch")
        if self._torch is None and torch is not None:
            raise RuntimeError(
                "Initialize the Torch controller before taking a replay snapshot."
            )
        if self._cuda is None and torch is not None and torch.cuda.is_initialized():
            raise RuntimeError(
                "Initialize the CUDA controller before taking a replay snapshot."
            )
        for layer, saved in zip(self._layers, self._python):
            restored = copy.deepcopy(saved, self._memo.copy())
            vars(layer).clear()
            vars(layer).update(restored)
        mujoco.mj_copyData(self._data, self._model, self._native)
        self._policy.set_state(self.policy_state)
        random.setstate(self._random)
        np.random.set_state(self._numpy)
        if self._torch is not None:
            torch.random.set_rng_state(self._torch)
        if self._cuda is not None:
            torch.cuda.set_rng_state_all(self._cuda)

    def integration_state(self) -> np.ndarray:
        """Return the saved native integration-state vector for data-only artifacts."""
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self._model, spec))
        mujoco.mj_getState(self._model, self._native, state, spec)
        return state
