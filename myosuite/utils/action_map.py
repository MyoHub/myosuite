# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Action maps ``a_muscles = M(z, state)``: play checkpoints whose policy does not command each muscle.

A policy trained through an action map (e.g. DynSyn muscle synergies) outputs ``z``
(``action_dim``); the map turns it into one action per muscle (in the order of the env's
muscle action term). This module holds what is needed to *play* such a checkpoint:

- the :class:`ActionMap` protocol and the map kinds (:class:`DynSyn`);
- :class:`ActionMapWrapper`, which applies a map on a CPU env;
- :func:`trained_action_map` / :func:`load_policy`, which read a checkpoint's map
  (torch, imported on call).

The mjlab action term is
:mod:`myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map`. Learning a map (e.g.
the DynSyn grouping and its trainer) lives in ``training/dynsyn``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.utils import RecordConstructorArgs

from myosuite.utils.checkpoint_manifest import (
    ACTION_MAP_FIELD,
    CfgLoader,
    env_yaml_action_term,
)

_ACTION_MAPS: dict[str, type] = {}
# Checkpoint ``infos`` key of DynSyn runs written before ACTION_MAP_FIELD.
_LEGACY_DYNSYN_KEY = "dynsyn"


@runtime_checkable
class ActionMap(Protocol):
    """A map from policy actions to per-muscle actions.

    Attributes:
        kind: Name under which the map is stored in manifests.
    """

    kind: ClassVar[str]

    @property
    def action_dim(self) -> int:
        """Size of the policy action ``z``."""
        ...

    @property
    def muscle_names(self) -> list[str]:
        """Actuator names of the muscles, in output order."""
        ...

    @property
    def n_muscles(self) -> int:
        """Number of per-muscle outputs."""
        return len(self.muscle_names)

    def forward(self, z: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
        """``(..., action_dim)`` -> ``(..., n_muscles)`` with numpy."""
        ...

    def forward_torch(self, z: Any, state: Mapping[str, Any]) -> Any:
        """Torch version of :meth:`forward` (differentiable where the map is)."""
        raise NotImplementedError

    def to_manifest(self) -> dict[str, Any]:
        """Plain types, with ``"kind"``, that :meth:`from_manifest` reads back."""
        ...

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> ActionMap:
        """Rebuild the map from :meth:`to_manifest`."""
        ...


def register_action_map(cls: type) -> type:
    """Class decorator: make ``cls`` loadable from manifests of its ``kind``."""
    _ACTION_MAPS[cls.kind] = cls
    return cls


def action_map_from_manifest(manifest: Mapping[str, Any]) -> ActionMap:
    """Rebuild any registered action map from its manifest.

    Raises:
        ValueError: No action map of the manifest's kind is registered.
    """
    kind = manifest["kind"]
    if kind not in _ACTION_MAPS:
        raise ValueError(f"Unknown action map {kind!r}; known: {sorted(_ACTION_MAPS)}.")
    return _ACTION_MAPS[kind].from_manifest(manifest)


def forward_torch(action_map: ActionMap, z: Any, state: Mapping[str, Any]) -> Any:
    """Torch forward of a map, falling back to its numpy one (not differentiable)."""
    try:
        return action_map.forward_torch(z, state)
    except NotImplementedError:
        import torch  # noqa: PLC0415

        out = action_map.forward(z.detach().cpu().numpy(), state)
        return torch.as_tensor(out, dtype=z.dtype, device=z.device)


@dataclass
class TrainedActionMap:
    """Action map of a training run at one of its checkpoints.

    Attributes:
        action_map: The run's action map.
        env_steps: Training progress (environment steps, all envs) at the checkpoint.
        saved_counter: The env's step counter stored in the checkpoint.
    """

    action_map: ActionMap
    env_steps: float
    saved_counter: int = 0

    def to_infos(self) -> dict[str, Any]:
        """Plain types for a checkpoint's ``infos[ACTION_MAP_FIELD]``."""
        return {"manifest": self.action_map.to_manifest(), "env_steps": self.env_steps}

    @staticmethod
    def from_infos(
        infos: Mapping[str, Any], saved_counter: int = 0
    ) -> TrainedActionMap:
        """Read back what :meth:`to_infos` wrote."""
        return TrainedActionMap(
            action_map_from_manifest(infos["manifest"]),
            env_steps=infos["env_steps"],
            saved_counter=saved_counter,
        )


class ActionMapWrapper(RecordConstructorArgs, gym.ActionWrapper):
    """Apply an action map on a CPU env (its actions: the model's actuators, in order).

    Args:
        env: Env to wrap.
        action_map: The map; its muscles must be the env's actuators.
        num_envs: Parallel copies of this env, so ``env_steps`` counts all their steps.
        step_offset: Added to this env's steps (continuing a training).
        frozen: Ignore this env's steps: ``env_steps`` stays at ``step_offset``.

    Raises:
        ValueError: If the map was made for other actuators.
    """

    def __init__(
        self,
        env: gym.Env,
        action_map: ActionMap,
        num_envs: int = 1,
        step_offset: float = 0.0,
        frozen: bool = False,
    ) -> None:
        RecordConstructorArgs.__init__(
            self,
            action_map=action_map,
            num_envs=num_envs,
            step_offset=step_offset,
            frozen=frozen,
        )
        gym.ActionWrapper.__init__(self, env)
        model = env.unwrapped.model
        names = [model.actuator(i).name.split("/")[-1] for i in range(model.nu)]
        if names != list(action_map.muscle_names):
            raise ValueError(
                f"The {action_map.kind} action map was made for other actuators than "
                f"this env's ({len(action_map.muscle_names)} vs {len(names)})."
            )
        self.action_map = action_map
        self.num_envs, self.step_offset, self.frozen = num_envs, step_offset, frozen
        self._steps = 0
        self._latched_z: np.ndarray | None = None
        self._hold_counter = 0
        box = env.action_space
        if action_map.action_dim != box.shape[0]:
            self.action_space = gym.spaces.Box(
                low=float(np.min(box.low)),
                high=float(np.max(box.high)),
                shape=(action_map.action_dim,),
                dtype=box.dtype,
            )

    @property
    def env_steps(self) -> float:
        """Training progress the map sees."""
        steps = 0 if self.frozen else self._steps * self.num_envs
        return self.step_offset + steps

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple:
        self._latched_z = None
        self._hold_counter = 0
        return super().reset(seed=seed, options=options)

    def action(self, action: np.ndarray) -> np.ndarray:
        hold = max(1, int(getattr(self.action_map, "hold_steps", 1)))
        if hold > 1:
            if self._latched_z is None or (self._hold_counter % hold) == 0:
                self._latched_z = np.array(action, copy=True)
            z = self._latched_z
        else:
            z = action
        data = self.env.unwrapped.data
        return self.action_map.forward(
            z,
            {
                "env_steps": self.env_steps,
                "actuator_length": data.actuator_length,
                "actuator_velocity": data.actuator_velocity,
            },
        )

    def step(self, action: np.ndarray) -> tuple:
        self._steps += 1
        out = super().step(action)
        self._hold_counter += 1
        return out


# --- DynSyn (He et al., ICML 2024, arXiv:2407.11472) ------------------------------


@dataclass
class DynSynGroups:
    """DynSyn muscle groups of one model, in the order of its muscle action term.

    Attributes:
        groups: Indices into the action term's muscles; each group starts with
            its representative (medoid), whose command is never corrected.
        muscle_names: Names of the muscles the indices refer to.
        meta: How the groups were made (task, steps, seed, inertia).
    """

    groups: list[list[int]]
    muscle_names: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Write the groups as JSON."""
        Path(path).write_text(json.dumps(asdict(self), indent=1) + "\n")

    @staticmethod
    def load(path: str | Path) -> DynSynGroups:
        """Read groups written by :meth:`save`."""
        return DynSynGroups(**json.loads(Path(path).read_text()))


def weight_amplitude(
    env_steps: float, start: float, ramp: float, max_amp: float = 0.1
) -> float:
    """Bound of DynSyn's per-muscle corrections after ``env_steps`` environment steps.

    Zero until ``start``, then linear to ``max_amp`` over ``ramp`` steps:
    ``min(max(k (t - a), 0), kappa)`` with ``a = start``, ``k = max_amp / ramp`` and
    ``kappa = max_amp``. The paper's Eq. 6 prints ``k t + a``, but says the corrections
    start at ``a`` and lists a positive ``a``, so this reads it as ``k (t - a)``. The paper
    gives no value of ``kappa``; the default of 0.1 follows the authors' reference implementation.
    """
    if ramp <= 0:
        return max_amp if env_steps >= start else 0.0
    return float(min(max((env_steps - start) / ramp, 0.0), 1.0) * max_amp)


def _expand_actions(
    actions: Any,
    groups: list[list[int]],
    amplitude: float,
    xp: Any,
    kappa: float = 0.1,
) -> Any:
    """Per-muscle actions from DynSyn actions.

    As in the paper (Eq. 5-7, with the sigmoid of Appendix C.1 applied afterwards), the
    correction scales the policy's action, not the excitation. On tasks with MyoSuite's
    sigmoid, a relative change of the action is therefore not the same relative change
    of the excitation: an action of 0 gets no correction, and on a negative action a
    positive correction lowers the excitation. On the leg-walk tasks, whose actions are
    the excitations, the two are the same.

    Eq. 5 is printed as ``IS(a_G) * clip(kappa w, -c, c)``. Here the correction is relative
    to the group's action (``1 + clip(...)``), as in the authors' reference implementation,
    so a bound of 0 gives the group's action, as the paper's text describes.

    Args:
        actions: ``(..., n_muscles)``: first one action per group, then one correction
            per non-representative muscle (in group order); ``(..., n_groups)``
            when ``amplitude`` is 0 (groups only).
        groups: Groups as in :class:`DynSynGroups`.
        amplitude: Correction bound (:meth:`DynSyn.amplitude`).
        xp: ``numpy`` or ``torch``.
        kappa: Scale of the corrections, the full bound of :func:`weight_amplitude`.

    Returns:
        ``(..., n_muscles)`` actions in muscle order: each muscle takes its group's
        action, scaled by ``1 + clip(kappa tanh(w), -amplitude, amplitude)``. The env
        clips them to its action range afterwards (the paper's Eq. 7).
    """
    n_groups = len(groups)
    group_of = [0] * sum(len(g) for g in groups)
    corrected = []
    for c, g in enumerate(groups):
        for i in g:
            group_of[i] = c
        corrected.extend(g[1:])
    group_actions = actions[..., :n_groups]
    out = group_actions[..., group_of] * 1.0
    if corrected and amplitude > 0:
        w = actions[..., n_groups:]
        scale = 1.0 + xp.clip(kappa * xp.tanh(w), -amplitude, amplitude)
        out[..., corrected] = out[..., corrected] * scale
    return out


@register_action_map
@dataclass
class DynSyn(ActionMap):
    """DynSyn as an action map: one action per group, plus scheduled corrections.

    The groups are learned by ``training/dynsyn``; this class only applies them.

    Attributes:
        groups: The muscle groups.
        start_steps: Environment steps (all envs together) before corrections start.
        ramp_steps: Environment steps from zero to the full correction bound.
        max_amplitude: Full correction bound (0: groups only, one action per group).
    """

    kind: ClassVar[str] = "dynsyn"
    groups: DynSynGroups
    start_steps: float = 5e7
    ramp_steps: float = 1e8
    max_amplitude: float = 0.1

    @property
    def muscle_names(self) -> list[str]:
        return list(self.groups.muscle_names)

    @property
    def action_dim(self) -> int:
        # Groups only: no action dimensions that do nothing (they would still drift
        # and inflate PPO's KL estimate).
        return len(self.groups.groups) if self.max_amplitude == 0 else self.n_muscles

    def amplitude(self, env_steps: float) -> float:
        """Correction bound after ``env_steps`` environment steps."""
        return weight_amplitude(
            env_steps, self.start_steps, self.ramp_steps, self.max_amplitude
        )

    def forward(self, z: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
        z = np.asarray(z)
        return _expand_actions(
            z.astype(np.float64),
            self.groups.groups,
            self.amplitude(state["env_steps"]),
            np,
            self.max_amplitude,
        ).astype(z.dtype)

    def forward_torch(self, z: Any, state: Mapping[str, Any]) -> Any:
        import torch  # noqa: PLC0415

        return _expand_actions(
            z,
            self.groups.groups,
            self.amplitude(state["env_steps"]),
            torch,
            self.max_amplitude,
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "groups": asdict(self.groups),
            "start_steps": self.start_steps,
            "ramp_steps": self.ramp_steps,
            "max_amplitude": self.max_amplitude,
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> DynSyn:
        return cls(
            groups=DynSynGroups(**manifest["groups"]),
            start_steps=manifest["start_steps"],
            ramp_steps=manifest["ramp_steps"],
            max_amplitude=manifest["max_amplitude"],
        )


# --- LambdaHold (Lee et al., arXiv:2608.17030) -------------------------------------

_SIGMOID_LO = 1.0 / (1.0 + math.exp(7.5))
_SIGMOID_HI = 1.0 / (1.0 + math.exp(-2.5))


def _lambda_hold_actions(
    z: Any,
    length: Any,
    velocity: Any,
    l_min: Any,
    l_max: Any,
    kp: float,
    damping_s: float,
    margin_fraction: float,
    muscle_sigmoid: bool,
    xp: Any,
) -> Any:
    span = xp.clip(l_max - l_min, 1e-3, None)
    l_mid = 0.5 * (l_min + l_max)
    threshold = l_mid - (0.5 + margin_fraction) * span * xp.clip(z, -1.0, 1.0)
    stretch = (length - threshold + damping_s * velocity) / span
    u = xp.clip(kp * stretch, 0.0, 1.0)
    if not muscle_sigmoid:
        return u
    u_safe = xp.clip(u, _SIGMOID_LO, _SIGMOID_HI)
    return 0.5 - 0.2 * xp.log(1.0 / u_safe - 1.0)


@register_action_map
@dataclass
class LambdaHold(ActionMap):
    """Equilibrium-point stretch-reflex action map (adapted from Lee et al., 2026).

    Policy actions ``z`` in ``[-1, 1]`` set per-muscle threshold lengths ``lambda_i``
    around each actuator's operating ``lengthrange``; a stretch-reflex law computes
    excitations from live actuator length and velocity.
    """

    kind: ClassVar[str] = "lambda_hold"
    muscles: list[str]
    length_min: tuple[float, ...]
    length_max: tuple[float, ...]
    kp: float = 2.5
    damping_s: float = 0.04
    margin_fraction: float = 0.35
    hold_steps: int = 1
    muscle_sigmoid: bool = True

    def __post_init__(self) -> None:
        n = len(self.muscles)
        if n == 0 or len(self.length_min) != n or len(self.length_max) != n:
            raise ValueError(
                "muscles, length_min and length_max must share non-zero length"
            )
        if (
            self.kp <= 0
            or self.damping_s < 0
            or self.margin_fraction < 0
            or self.hold_steps < 1
        ):
            raise ValueError("Invalid LambdaHold gain, damping, margin or hold_steps")

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        actuator_ids: Sequence[int] | None = None,
        kp: float = 2.5,
        damping_s: float = 0.04,
        margin_fraction: float = 0.35,
        hold_steps: int = 1,
        muscle_sigmoid: bool = True,
    ) -> LambdaHold:
        """Build a :class:`LambdaHold` map from a compiled ``MjModel``."""
        ids = range(model.nu) if actuator_ids is None else actuator_ids
        names = [model.actuator(i).name.split("/")[-1] for i in ids]
        lr = np.asarray(model.actuator_lengthrange, dtype=float)[list(ids)]
        lo = lr[:, 0].copy()
        hi = np.where(lr[:, 1] > lr[:, 0] + 1e-6, lr[:, 1], lr[:, 0] + 1.0)
        return cls(
            muscles=names,
            length_min=tuple(lo.tolist()),
            length_max=tuple(hi.tolist()),
            kp=kp,
            damping_s=damping_s,
            margin_fraction=margin_fraction,
            hold_steps=hold_steps,
            muscle_sigmoid=muscle_sigmoid,
        )

    @property
    def muscle_names(self) -> list[str]:
        return list(self.muscles)

    @property
    def action_dim(self) -> int:
        return len(self.muscles)

    def forward(self, z: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
        z_arr = np.asarray(z)
        l_min = np.asarray(self.length_min, dtype=np.float64)
        l_max = np.asarray(self.length_max, dtype=np.float64)
        length = np.asarray(state["actuator_length"], dtype=np.float64)
        velocity = np.asarray(state["actuator_velocity"], dtype=np.float64)
        return _lambda_hold_actions(
            z_arr.astype(np.float64),
            length,
            velocity,
            l_min,
            l_max,
            self.kp,
            self.damping_s,
            self.margin_fraction,
            self.muscle_sigmoid,
            np,
        ).astype(z_arr.dtype)

    def forward_torch(self, z: Any, state: Mapping[str, Any]) -> Any:
        import torch  # noqa: PLC0415

        l_min = torch.as_tensor(self.length_min, dtype=z.dtype, device=z.device)
        l_max = torch.as_tensor(self.length_max, dtype=z.dtype, device=z.device)
        length = state["actuator_length"].to(dtype=z.dtype, device=z.device)
        velocity = state["actuator_velocity"].to(dtype=z.dtype, device=z.device)
        return _lambda_hold_actions(
            z,
            length,
            velocity,
            l_min,
            l_max,
            self.kp,
            self.damping_s,
            self.margin_fraction,
            self.muscle_sigmoid,
            torch,
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "muscles": list(self.muscles),
            "length_min": list(self.length_min),
            "length_max": list(self.length_max),
            "kp": self.kp,
            "damping_s": self.damping_s,
            "margin_fraction": self.margin_fraction,
            "hold_steps": self.hold_steps,
            "muscle_sigmoid": self.muscle_sigmoid,
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> LambdaHold:
        return cls(
            muscles=list(manifest["muscles"]),
            length_min=tuple(float(v) for v in manifest["length_min"]),
            length_max=tuple(float(v) for v in manifest["length_max"]),
            kp=float(manifest["kp"]),
            damping_s=float(manifest["damping_s"]),
            margin_fraction=float(manifest["margin_fraction"]),
            hold_steps=int(manifest.get("hold_steps", 1)),
            muscle_sigmoid=bool(manifest.get("muscle_sigmoid", True)),
        )


# --- Checkpoints (torch on call) ---------------------------------------------------


def _from_env_yaml(checkpoint: Path, counter: int | None) -> TrainedActionMap | None:
    """The action map in the run's ``params/env.yaml`` (runs without it in ``infos``)."""
    path = checkpoint.parent / "params" / "env.yaml"
    if not path.is_file():
        return None
    env = yaml.load(path.read_text(errors="replace"), Loader=CfgLoader) or {}
    term = env_yaml_action_term(env)
    if term is None:
        return None
    if ACTION_MAP_FIELD in term:
        action_map = action_map_from_manifest(term[ACTION_MAP_FIELD])
    else:  # the earlier DynSyn-only action term
        action_map = DynSyn.from_manifest(term)
    if counter is None:
        print(f"{checkpoint} stores no step count: using the full schedule.")
    steps = (
        float("inf")
        if counter is None
        else counter * env["scene"]["num_envs"] + term.get("step_offset", 0.0)
    )
    return TrainedActionMap(action_map, env_steps=steps, saved_counter=counter or 0)


def trained_action_map(checkpoint: str | Path) -> TrainedActionMap | None:
    """Action map of the run that wrote a checkpoint.

    Trainers store the map in the checkpoint's ``infos`` (``ACTION_MAP_FIELD``), so a
    checkpoint read on its own (a W&B download, a moved file) still has it. Older
    checkpoints fall back to the run's ``params/env.yaml``, which holds the action
    term, and to the env's step counter in the checkpoint.

    Args:
        checkpoint: ``model_*.pt`` of an mjlab training run.

    Returns:
        The map, or ``None`` if the run used none (or is an older run without its
        ``params/env.yaml``).
    """
    import torch  # noqa: PLC0415

    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    infos = state.get("infos") or {}
    counter = infos.get("env_state", {}).get("common_step_counter")
    if ACTION_MAP_FIELD in infos:
        return TrainedActionMap.from_infos(infos[ACTION_MAP_FIELD], counter or 0)
    if _LEGACY_DYNSYN_KEY in infos:
        legacy = infos[_LEGACY_DYNSYN_KEY]
        return TrainedActionMap(
            DynSyn.from_manifest(legacy),
            env_steps=legacy["env_steps"],
            saved_counter=counter or 0,
        )
    return _from_env_yaml(checkpoint, counter)


def load_policy(
    checkpoint: str | Path, action_dim: int
) -> tuple[Any, TrainedActionMap | None]:
    """The deterministic actor of a checkpoint and the action map it acts through.

    Args:
        checkpoint: ``model_*.pt`` of an mjlab training run.
        action_dim: Action dimension of the env (with the map applied) it will drive.

    Returns:
        The :class:`~myosuite.utils.rslrl_policy.RslRlPolicy` and the run's map, or
        ``None`` for one action per muscle. Apply the map with :class:`ActionMapWrapper`
        (CPU) or ``apply_trained_action_map`` of
        :mod:`myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map` (mjlab).
    """
    from myosuite.utils.rslrl_policy import load_rslrl_policy  # noqa: PLC0415

    trained = trained_action_map(checkpoint)
    policy = load_rslrl_policy(
        checkpoint, action_dim, action_map_applied=trained is not None
    )
    return policy, trained
