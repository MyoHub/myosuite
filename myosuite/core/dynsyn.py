# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""DynSyn muscle synergies (He et al., ICML 2024, arXiv:2407.11472).

Muscles whose lengths change together under random joint motion form one group.
The policy then commands one action per group plus a small, state-dependent
correction per muscle, whose bound grows during training (see
:func:`weight_amplitude`). Written from the paper's description.

The grouping runs on a CPU ``MjModel``; the action mapping is backend-agnostic
(``torch`` or ``numpy``).
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


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
    """Mean cosine similarity of the normalised muscle lengths over short segments.

    Args:
        lengths: ``(steps, n_muscles)``; ``steps`` is cut to a multiple of ``segment``.
        segment: Steps per segment.

    Returns:
        ``(n_muscles, n_muscles)`` similarity in ``[0, 1]``.
    """
    lo, hi = lengths.min(0), lengths.max(0)
    x = (lengths - lo) / np.maximum(hi - lo, 1e-12)
    n = len(x) // segment * segment
    x = x[:n].reshape(n // segment, segment, -1)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    cos = np.einsum("smi,smj->sij", x, x) / (
        np.einsum("sai,saj->sij", norm, norm) + 1e-12
    )
    return cos.mean(0)


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

    Zero until ``start``, then linear to ``max_amp`` over ``ramp`` steps (the paper's
    ``min(max(k (t - a), 0), 0.1)`` with ``a = start`` and ``k = max_amp / ramp``).
    """
    if ramp <= 0:
        return max_amp if env_steps >= start else 0.0
    return float(min(max((env_steps - start) / ramp, 0.0), 1.0) * max_amp)


def expand_actions(
    actions: Any, groups: list[list[int]], amplitude: float, xp: Any
) -> Any:
    """Per-muscle actions from DynSyn actions.

    Args:
        actions: ``(..., n_muscles)``: first one action per group, then one correction
            per non-representative muscle (in group order).
        groups: Groups as in :class:`DynSynGroups`.
        amplitude: Correction bound (:func:`weight_amplitude`).
        xp: ``numpy`` or ``torch``.

    Returns:
        ``(..., n_muscles)`` actions in muscle order: each muscle takes its group's
        action, scaled by ``1 + clip(0.1 tanh(w), -amplitude, amplitude)``.
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
        scale = 1.0 + xp.clip(0.1 * xp.tanh(w), -amplitude, amplitude)
        out[..., corrected] = out[..., corrected] * scale
    return out
