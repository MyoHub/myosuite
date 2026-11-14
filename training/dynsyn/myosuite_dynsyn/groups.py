# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""DynSyn muscle groups (He et al., ICML 2024, arXiv:2407.11472).

Muscles whose lengths change together under random joint motion form one group.
Follows the paper; where it leaves something open, the docstrings say what is used
here. The defaults of :func:`synergy_groups` follow the authors' reference
implementation (github.com/kaibohex/DynSyn), which was read, not copied.

Command line (writes the groups for ``python -m myosuite_dynsyn.train --groups``)::

    python -m myosuite_dynsyn.groups myoLegWalk-v0 --groups 40 --out myoLegWalk-v0_k40.json
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.utils.action_map import DynSyn, DynSynGroups


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
    The dependents of joint equalities (knee mechanism, patella) are then put on
    their couplings, in position and velocity, so they follow their masters
    instead of being driven against the equality constraints.

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
    couplings = JointCouplings.from_model(m)
    rng = np.random.default_rng(seed)
    out = np.empty((steps, len(actuator_ids)))
    for t in range(steps):
        if t % hold_steps == 0:
            d.qvel[:] = rng.uniform(-velocity_amplitude, velocity_amplitude, m.nv)
            project_joint_couplings(couplings, d.qpos, d.qvel)
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

    Raises:
        ValueError: ``k`` is not in ``[1, n]``.
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
            if d2.sum() == 0.0:
                d2[:] = 1.0
                d2[start] = 0.0
            start.append(int(rng.choice(n, p=d2 / d2.sum())))
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


def fit_dynsyn(
    model: mujoco.MjModel,
    actuator_ids: np.ndarray,
    k: int,
    steps: int = 500_000,
    velocity_amplitude: float = 10.0,
    seed: int = 0,
    **schedule: float,
) -> DynSyn:
    """Group a model's muscles (:func:`synergy_groups`) into a DynSyn action map.

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
    return DynSyn(DynSynGroups(groups, names, meta), **schedule)


def task_muscles(task: str) -> tuple[mujoco.MjModel, np.ndarray, list[str]]:
    """A registered task's model and its muscle actuators, in the order of its mjlab action term.

    Args:
        task: Env id with an mjlab twin.

    Returns:
        The CPU ``MjModel`` of the twin, the actuator ids and their names.
    """
    from myosuite import make_env
    from myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map import (
        action_actuator_names,
        muscle_action_terms,
    )

    env = make_env(task, backend="mjlab", num_envs=1, device="cpu")
    (name,) = muscle_action_terms(env.cfg)
    model = env.sim.mj_model
    names = action_actuator_names(env.action_manager.get_term(name), env)
    short = [model.actuator(i).name.split("/")[-1] for i in range(model.nu)]
    ids = np.array([short.index(n) for n in names])
    env.close()
    return model, ids, names


def main(argv: list[str] | None = None) -> None:
    """Write the DynSyn groups of a task as JSON."""
    p = argparse.ArgumentParser(description="DynSyn muscle groups of an mjlab task.")
    p.add_argument("task")
    p.add_argument("--groups", type=int, required=True, help="number of muscle groups")
    p.add_argument(
        "--steps", type=int, default=500_000, help="simulation steps of random motion"
    )
    p.add_argument(
        "--velocity",
        type=float,
        default=10.0,
        help="bound of the random joint velocities",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    model, ids, names = task_muscles(args.task)
    groups = fit_dynsyn(
        model, ids, args.groups, args.steps, args.velocity, args.seed
    ).groups
    assert groups.muscle_names == names
    groups.meta["task"] = args.task
    args.out.parent.mkdir(parents=True, exist_ok=True)
    groups.save(args.out)
    print(
        f"{len(groups.groups)} groups of {len(names)} muscles, sizes "
        f"{groups.meta['sizes']} -> {args.out}"
    )


if __name__ == "__main__":
    main()
