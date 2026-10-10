# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache 2 license in the root LICENSE file.
"""Generate tracking goals using pinned upstream functions, without importing TERRA."""

from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from myosuite.integrations.musclemimic import MIMIC_SITES, build_terrain_fullbody_spec

TERRA_SHA = "db9d0d694f776c9de56b8cb3fc3ab48804bd57a1"
MUSCLEMIMIC_SHA = "f1c2dbfd0d8e31d306b2c4e2369aecdc7bc21993"


def load_functions(path: Path, names: set[str], scope: dict[str, Any]) -> None:
    """Execute selected upstream definitions without their heavyweight imports."""
    tree = ast.parse(path.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *selected,
    ]
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), scope)


def generate(terra: Path, musclemimic: Path, output: Path) -> None:
    """Record nonzero tracking errors and bounded horizons from upstream assembly."""
    for root, revision in ((terra, TERRA_SHA), (musclemimic, MUSCLEMIMIC_SHA)):
        actual = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != revision:
            raise ValueError(f"Expected {revision} at {root}, got {actual}")
    scope = {
        "np": np,
        "np_R": Rotation,
        "jnp": object(),
        "GoalTrajMimic": object,
        "DEFAULT_LOOKAHEAD_STEPS": (1, 20, 40, 60, 80),
        "quat_scalarfirst2scalarlast": lambda q: np.roll(q, -1),
    }
    load_functions(
        musclemimic / "loco_mujoco/core/utils/math.py",
        {
            "atleast_3d",
            "calc_rel_positions",
            "transform_motion",
            "calc_site_velocities",
        },
        scope,
    )
    load_functions(
        terra / "src/terra/rl/egocentric.py",
        {
            "rotation_matrix_from_quaternion",
            "heading_rotation_matrix",
            "world_to_heading",
            "root_velocity_in_heading_frame",
            "rotation_error_in_frame",
        },
        scope,
    )
    load_functions(
        terra / "src/terra/rl/tracking.py",
        {
            "_site_kinematics",
            "_root_tracking_error",
            "_site_tracking_error",
            "_future_intent",
            "TerraFullBodyTrackingGoal",
        },
        scope,
    )
    model = build_terrain_fullbody_spec(disabled_contact_pairs=()).compile()
    frames = 100
    qpos = np.repeat(model.qpos0[None], frames, axis=0)
    t = np.arange(frames) * 0.01
    qpos[:, 0] += 0.4 * t
    qpos[:, 1] -= 0.2 * t
    qpos[:, 7:] += 0.08 * np.sin(3 * t[:, None] + np.arange(model.nq - 7)[None] * 0.1)
    qpos[:, 3:7] = np.roll(
        Rotation.from_euler(
            "xyz", np.column_stack([0.04 * np.sin(t), 0.07 * np.cos(t), 0.2 * t])
        ).as_quat(),
        1,
        axis=1,
    )
    step = np.zeros((frames - 1, model.nv))
    qpos = qpos.astype(np.float32).astype(float)
    # Differentiate the rounded positions, matching the supplied reference states.
    for i in range(frames - 1):
        mujoco.mj_differentiatePos(model, step[i], 0.01, qpos[i], qpos[i + 1])
    qvel = (
        np.vstack([step[0], 0.5 * (step[:-1] + step[1:]), step[-1]])
        .astype(np.float32)
        .astype(float)
    )
    refs = []
    for q, v in zip(qpos, qvel):
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = q, v
        mujoco.mj_forward(model, data)
        refs.append(data)
    goal = scope["TerraFullBodyTrackingGoal"].__new__(
        scope["TerraFullBodyTrackingGoal"]
    )
    goal.lookahead_steps = (1, 20, 40, 60, 80)
    goal._root_qpos_indices, goal._root_qvel_indices = np.arange(7), np.arange(6)
    goal._rel_site_ids = np.array([model.site(name).id for name in MIMIC_SITES])
    goal._site_bodyid, goal._body_rootid = model.site_bodyid, model.body_rootid
    goal._site_mapper = SimpleNamespace(requires_mapping=False)
    env = SimpleNamespace(
        preserve_trajectory_root_xy=True,
        dt=0.01,
        th=SimpleNamespace(
            len_trajectory=lambda _: frames,
            get_traj_data_at=lambda _, frame, *args: refs[frame],
        ),
    )
    indices = np.array([0, 7, 20, 79, 98, 99])
    states, velocities, goals = [], [], []
    for i, frame in enumerate(indices):
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = qpos[frame], qvel[frame]
        data.qpos[:3] += [0.2, -0.1, 0.03]
        perturb = np.roll(
            Rotation.from_euler("xyz", [0.06, -0.04, 0.15 + i * 0.02]).as_quat(), 1
        )
        mujoco.mju_mulQuat(data.qpos[3:7], perturb, qpos[frame, 3:7])
        data.qpos[7:] += 0.015 * np.cos(np.arange(model.nq - 7))
        data.qvel[:] += np.linspace(-0.2, 0.3, model.nv)
        mujoco.mj_forward(model, data)
        carry = SimpleNamespace(
            traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=frame)
        )
        goals.append(goal._tracking_observation(env, data, carry, np))
        states.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
    np.savez_compressed(
        output,
        ref_qpos=qpos,
        qpos=states,
        qvel=velocities,
        frames=indices,
        goals=goals,
        terra_sha=TERRA_SHA,
        musclemimic_sha=MUSCLEMIMIC_SHA,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("terra", type=Path)
    parser.add_argument("musclemimic", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.terra, args.musclemimic, args.output)
