# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""DynSyn muscle synergies (He et al., ICML 2024, arXiv:2407.11472).

Muscles whose lengths change together under random joint motion form one group.
The policy then commands one action per group plus a small, state-dependent
correction per muscle, whose bound grows during training (see
:func:`weight_amplitude`). Follows the paper; where it leaves something open, the
docstrings say what is used here. The form and scale of the correction and the defaults
of :func:`synergy_groups` follow the authors' reference implementation
(github.com/Beanpow/DynSyn), which was read while writing this.

:class:`DynSyn` is the :class:`~myosuite.methods.synergies.base.ActionMap`; the grouping
runs on a CPU ``MjModel`` (:meth:`DynSyn.fit`), the mapping on ``numpy`` or ``torch``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import mujoco
import numpy as np

from myosuite.methods.synergies.base import ActionMap, register_action_map


@dataclass
class DynSynGroups:
    """Muscle groups of one model, in the order of its muscle action term.

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


def record_muscle_lengths(
    model: mujoco.MjModel,
    actuator_ids: np.ndarray,
    steps: int = 500_000,
    velocity_amplitude: float = 10.0,
    hold_steps: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Muscle lengths under random joint velocities with relaxed muscles.

    Every ``hold_steps`` steps each joint gets a velocity drawn from
    ``U(-velocity_amplitude, velocity_amplitude)``. Gravity and contacts are off and
    free joints are held still, so the body stays in place and every joint moves.

    Args:
        model: CPU model (copied, not changed).
        actuator_ids: Muscle actuators, in action order.
        steps: Simulation steps to record.
        velocity_amplitude: Bound of the random joint velocities (rad/s, m/s).
        hold_steps: Steps between two velocity draws.
        seed: Random seed.

    Returns:
        ``(steps, n_muscles)`` actuator lengths.
    """
    m = copy.deepcopy(model)
    m.opt.gravity[:] = 0.0
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(m)
    free = np.concatenate(
        [
            np.arange(m.jnt_dofadr[j], m.jnt_dofadr[j] + 6)
            for j in range(m.njnt)
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        or [np.zeros(0, dtype=int)]
    )
    rng = np.random.default_rng(seed)
    out = np.empty((steps, len(actuator_ids)))
    for t in range(steps):
        if t % hold_steps == 0:
            d.qvel[:] = rng.uniform(-velocity_amplitude, velocity_amplitude, m.nv)
        d.qvel[free] = 0.0
        d.ctrl[:] = 0.0
        mujoco.mj_step(m, d)
        out[t] = d.actuator_length[actuator_ids]
    return out


def length_similarity(lengths: np.ndarray, segment: int = 100) -> np.ndarray:
    """Mean correlation of the muscle length changes over short segments.

    The paper's Eq. 2 averages the cosine similarity of the trajectory segments, and its
    text calls this the correlation between length changes. Each segment is centred here,
    which makes the cosine that correlation. Without centring, two positive lengths that
    drift slowly within a segment have a cosine near 1 even if they don't change together.
    The paper gives no segment length; this one is this implementation's choice.

    Args:
        lengths: ``(steps, n_muscles)``; ``steps`` is cut to a multiple of ``segment``.
        segment: Steps per segment.

    Returns:
        ``(n_muscles, n_muscles)`` similarity in ``[0, 1]``.
    """
    n = len(lengths) // segment * segment
    x = lengths[:n].reshape(n // segment, segment, -1)
    x = x - x.mean(1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    cos = np.einsum("smi,smj->sij", x, x) / (
        np.einsum("sai,saj->sij", norm, norm) + 1e-12
    )
    similarity = np.clip(cos.mean(0), 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)  # also for a muscle that doesn't move
    return similarity


def k_medoids(
    distance: np.ndarray,
    k: int,
    restarts: int = 20,
    iterations: int = 300,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """k-medoids clustering of a distance matrix (alternating updates, best restart).

    Args:
        distance: ``(n, n)`` symmetric distances.
        k: Number of clusters.
        restarts: Random initialisations (k-means++-style); the lowest inertia wins.
        iterations: Maximum assignment/update rounds per restart.
        seed: Random seed.

    Returns:
        Medoid indices ``(k,)``, labels ``(n,)`` and the inertia (sum of distances
        to the medoids).
    """
    n = len(distance)
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    rng = np.random.default_rng(seed)
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(restarts):
        start = [int(rng.integers(n))]
        while len(start) < k:
            d2 = distance[:, start].min(1) ** 2
            d2[start] = 0.0
            p = d2 / d2.sum() if d2.sum() > 0 else None
            start.append(int(rng.choice(n, p=p)))
        medoids = np.array(start)
        for _ in range(iterations):
            labels = distance[:, medoids].argmin(1)
            labels[medoids] = np.arange(k)
            new = medoids.copy()
            for c in range(k):
                members = np.flatnonzero(labels == c)
                new[c] = members[distance[np.ix_(members, members)].sum(1).argmin()]
            if np.array_equal(new, medoids):
                break
            medoids = new
        labels = distance[:, medoids].argmin(1)
        labels[medoids] = np.arange(k)
        inertia = float(distance[np.arange(n), medoids[labels]].sum())
        if best is None or inertia < best[2]:
            best = (medoids, labels, inertia)
    assert best is not None
    return best


def synergy_groups(
    model: mujoco.MjModel,
    actuator_ids: np.ndarray,
    k: int,
    steps: int = 500_000,
    velocity_amplitude: float = 10.0,
    seed: int = 0,
) -> tuple[list[list[int]], dict[str, Any]]:
    """DynSyn groups: muscles that lengthen and shorten together.

    Args:
        model: CPU model.
        actuator_ids: Muscle actuators, in action order.
        k: Number of groups.
        steps: Simulation steps of random joint motion.
        velocity_amplitude: Bound of the random joint velocities.
        seed: Random seed.

    Returns:
        The groups (indices into ``actuator_ids``, medoid first, groups sorted by
        their first index) and a summary (inertia, sizes).
    """
    lengths = record_muscle_lengths(
        model, actuator_ids, steps, velocity_amplitude, seed=seed
    )
    distance = np.clip(1.0 - length_similarity(lengths), 0.0, None)
    medoids, labels, inertia = k_medoids(distance, k, seed=seed)
    groups = []
    for c, med in enumerate(medoids):
        rest = sorted(int(i) for i in np.flatnonzero(labels == c) if i != med)
        groups.append([int(med), *rest])
    groups.sort(key=lambda g: g[0])
    meta = {"inertia": inertia, "sizes": sorted((len(g) for g in groups), reverse=True)}
    return groups, meta


def weight_amplitude(
    env_steps: float, start: float, ramp: float, max_amp: float = 0.1
) -> float:
    """Bound of the per-muscle corrections after ``env_steps`` environment steps.

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

    @classmethod
    def fit(
        cls,
        model: mujoco.MjModel,
        actuator_ids: np.ndarray,
        k: int,
        steps: int = 500_000,
        velocity_amplitude: float = 10.0,
        seed: int = 0,
        **schedule: float,
    ) -> DynSyn:
        """Group a model's muscles (:func:`synergy_groups`) into a DynSyn map.

        Args:
            model: CPU model.
            actuator_ids: Muscle actuators, in action order.
            k: Number of groups.
            steps: Simulation steps of random joint motion.
            velocity_amplitude: Bound of the random joint velocities.
            seed: Random seed.
            **schedule: ``start_steps``, ``ramp_steps``, ``max_amplitude``.

        Returns:
            The map; ``groups.meta`` holds the grouping summary and its settings.
        """
        groups, meta = synergy_groups(
            model, actuator_ids, k, steps, velocity_amplitude, seed
        )
        meta.update(steps=steps, velocity=velocity_amplitude, seed=seed)
        names = [model.actuator(int(i)).name.split("/")[-1] for i in actuator_ids]
        return cls(DynSynGroups(groups, names, meta), **schedule)
