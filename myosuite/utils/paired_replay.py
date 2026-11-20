# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""In-memory snapshots for paired headless CPU experiments on one fixed model."""

from __future__ import annotations

import copy
import dis
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import FunctionType, MethodType
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from myosuite.envs.wrappers import FatigueWrapper


def _restore_attributes(target: Any, state: dict) -> None:
    vars(target).clear()
    vars(target).update(state)


@dataclass
class ReplayStateOwner:
    """Explicit mutable callback owner, restored in place rather than replaced."""

    target: Any
    get_state: Callable[[], Any]
    set_state: Callable[[Any], None]

    @classmethod
    def attributes(cls, target: Any) -> ReplayStateOwner:
        """Declare that all runtime state is in the target's __dict__."""
        vars(target)
        return cls(
            target,
            lambda: vars(target),
            lambda state: _restore_attributes(target, state),
        )


def _check_callback(
    fn: Any, memo: dict, owners: list[ReplayStateOwner], seen: set
) -> None:
    if fn is None or id(fn) in seen:
        return
    seen.add(id(fn))
    if isinstance(fn, MethodType):
        if id(fn.__self__) not in memo:
            raise ValueError("Register the mutable callback's ReplayStateOwner")
        fn = fn.__func__
    if not isinstance(fn, FunctionType):
        if id(fn) not in memo:
            raise ValueError("Register the callable's ReplayStateOwner")
        return
    if any(
        i.opname in ("STORE_GLOBAL", "DELETE_GLOBAL")
        or (
            i.opname in ("STORE_DEREF", "DELETE_DEREF")
            and i.argval in fn.__code__.co_freevars
        )
        for i in dis.get_instructions(fn)
    ):
        raise ValueError(
            "Mutable callback cells/globals need an explicit state owner object"
        )
    values = [c.cell_contents for c in fn.__closure__ or ()]
    values += list(fn.__defaults__ or ()) + list((fn.__kwdefaults__ or {}).values())
    for value in values:
        if id(value) in memo or isinstance(
            value, (type(None), bool, int, float, str, bytes, type, np.generic)
        ):
            continue
        if isinstance(value, (FunctionType, MethodType)):
            _check_callback(value, memo, owners, seen)
        elif (
            isinstance(value, np.ndarray)
            and fn.__module__ == "myosuite.envs.muscle_stages"
        ):
            owners.append(
                ReplayStateOwner(
                    value,
                    value.copy,
                    lambda state, array=value: np.copyto(array, state),
                )
            )
            memo[id(value)] = value
        else:
            raise ValueError(
                "Register mutable closure/default state with ReplayStateOwner"
            )


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
    Built-in fatigue owners retain identity. Custom control-stage callback owners
    must be registered explicitly; closure-cell reassignment is rejected. Callbacks
    must not depend on undeclared mutable global objects. Controller owners
    explicitly provide all mutable inference state callbacks.
    """

    @classmethod
    def capture(
        cls,
        env: gym.Env,
        policy: ReplayPolicy,
        *,
        state_owners: tuple[ReplayStateOwner, ...] = (),
    ) -> ReplaySnapshot:
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
        self._owners = list(state_owners)
        known = {id(owner.target) for owner in self._owners}
        for item in self._layers:
            if (
                isinstance(item, FatigueWrapper)
                and id(item.muscle_fatigue) not in known
            ):
                self._owners.append(ReplayStateOwner.attributes(item.muscle_fatigue))
        self._memo.update({id(owner.target): owner.target for owner in self._owners})
        for item in self._layers:
            for name, value in vars(item).items():
                if name in ("_renderer", "_mj_renderer_compat") and value is not None:
                    raise ValueError("Paired replay requires a headless env.")
                if (
                    isinstance(value, (mujoco.MjModel, mujoco.MjData, mujoco.MjSpec))
                    or name == "_accessor"
                ):
                    self._memo[id(value)] = value
        for _, apply, reset in vars(layer).get("_ctrl_stage_store", {}).values():
            _check_callback(apply, self._memo, self._owners, set())
            _check_callback(reset, self._memo, self._owners, set())
        self._python = copy.deepcopy(
            [vars(item) for item in self._layers]
            + [owner.get_state() for owner in self._owners],
            self._memo.copy(),
        )
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
        restored = copy.deepcopy(self._python, self._memo.copy())
        for layer, state in zip(self._layers, restored):
            vars(layer).clear()
            vars(layer).update(state)
        for owner, state in zip(self._owners, restored[len(self._layers) :]):
            owner.set_state(state)
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

    def rng_state(self) -> dict[str, Any]:
        """Return independent RNG states for data-only experiment artifacts."""
        return {
            "python": copy.deepcopy(self._random),
            "numpy_global": copy.deepcopy(self._numpy),
            "env_and_state_owners": [
                copy.deepcopy(value.bit_generator.state)
                for layer in self._python
                if isinstance(layer, dict)
                for value in layer.values()
                if isinstance(value, np.random.Generator)
            ],
            "torch_cpu": None
            if self._torch is None
            else self._torch.cpu().numpy().copy(),
            "torch_cuda": None
            if self._cuda is None
            else [state.cpu().numpy().copy() for state in self._cuda],
        }
