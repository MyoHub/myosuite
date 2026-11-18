# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for experimental Newton IndexMapping and NewtonAccessor."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from myosuite.core.protocols import EnvAccessor
from myosuite.terms.base_obs import joint_pos_obs, muscle_act_obs
from myosuite.terms.base_reward import pose_reward, reach_reward

pytestmark = pytest.mark.tier2

_REPO = Path(__file__).resolve().parents[2]


def _require_or_skip_newton() -> tuple[object, str]:
    if os.environ.get("MYOSUITE_REQUIRE_NEWTON") == "1":
        newton = importlib.import_module("newton")
        wp = importlib.import_module("warp")
    else:
        newton = pytest.importorskip("newton")
        wp = pytest.importorskip("warp")
    return newton, ("cuda:0" if wp.is_cuda_available() else "cpu")


def test_newton_package_import_skips_heavy_dependencies() -> None:
    script = (
        "import json, sys\n"
        "import myosuite.integrations.newton as pkg\n"
        "assert 'NewtonAccessor' in dir(pkg)\n"
        "heavy = [m for m in ('newton', 'warp', 'mujoco_warp', 'torch') if m in sys.modules]\n"
        "print(json.dumps(heavy))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == []


@pytest.mark.parametrize(
    "env_id",
    ["myoElbowPose1D6MRandom-v0", "myoHandPoseRandom-v0"],
)
def test_newton_index_mapping_and_accessor_pure_terms(env_id: str) -> None:
    newton, device = _require_or_skip_newton()
    wp = importlib.import_module("warp")
    from myosuite.integrations.newton import (
        IndexMapping,
        NewtonAccessor,
        resolve_env_xml,
    )
    from myosuite.integrations.newton.accessor import _integrator_name

    xml_path, meta = resolve_env_xml(env_id)
    assert xml_path.is_file()
    assert meta["frame_skip"] >= 1
    reference = mujoco.MjModel.from_xml_path(str(xml_path))

    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(template)
    template.default_shape_cfg.gap = 0.0
    template.add_mjcf(str(xml_path), ctrl_direct=True)
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.replicate(template, copy_count=2)
    n_model = builder.finalize(device=device)
    solver = newton.solvers.SolverMuJoCo(
        n_model,
        use_mujoco_cpu=False,
        use_mujoco_contacts=True,
        nconmax=2048,
        njmax=8192,
        strict_capacity=True,
        integrator=_integrator_name(reference),
        iterations=int(reference.opt.iterations),
        ls_iterations=int(reference.opt.ls_iterations),
        kinematic_switchable_roots=False,
    )
    state_in = n_model.state()
    solver.reset(state_in)
    solver._update_mjc_data(solver.mjw_data, n_model, state_in)
    mjw = solver._mujoco_warp
    with wp.ScopedDevice(n_model.device), solver._scoped_mujoco_warp_execution():
        mjw.fwd_position(solver.mjw_model, solver.mjw_data, factorize=False)
        mjw.fwd_velocity(solver.mjw_model, solver.mjw_data)
        mjw.fwd_actuation(solver.mjw_model, solver.mjw_data)

    mapping = IndexMapping.build(reference, n_model, solver)
    assert set(mapping.conv_to_src_qpos.tolist()) == set(range(reference.nq))
    assert set(mapping.conv_to_src_qvel.tolist()) == set(range(reference.nv))
    assert set(mapping.conv_to_src_ctrl.tolist()) == set(range(reference.nu))

    ctx = SimpleNamespace(
        reference_model=reference,
        solver=solver,
        mapping=mapping,
        num_envs=2,
        ctrl_dt=float(reference.opt.timestep) * int(meta["frame_skip"]),
    )
    accessor = NewtonAccessor(ctx)
    assert isinstance(accessor, EnvAccessor)
    assert joint_pos_obs(accessor).shape == (2, reference.nq)
    assert muscle_act_obs(accessor).shape == (2, reference.na)

    target_angles = np.zeros(reference.nq, dtype=np.float32)
    pose_out = pose_reward(accessor, {"target_angles": target_angles})
    assert pose_out["dense"].shape == (2,)
    sid = accessor.site_id(reference.site(0).name)
    reach_out = reach_reward(
        accessor,
        {"target_pos": np.zeros(3, dtype=np.float32), "tip_site_ids": [sid]},
    )
    assert reach_out["dense"].shape == (2,)
