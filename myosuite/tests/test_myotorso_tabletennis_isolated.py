# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Parity: MuscleMimic bimanual *right* arm vs MyoTorso single-arm host."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.tests.support.myotorso_tabletennis_isolated import (
    build_joint_value_map,
    compile_myotorso_arm_host,
    myotorso_tabletennis_isolated_xml_path,
)
from myosuite.tests.support.optional_deps import require_musclemimic_models


def _bimanual_ref_model() -> mujoco.MjModel:
    """Compiled MuscleMimic bimanual model, or skip if dependency missing."""
    require_musclemimic_models()
    from myosuite.integrations.musclemimic.bimanual_model import (
        build_mimic_bimanual_spec,
        default_mimic_config,
    )

    spec, _ = build_mimic_bimanual_spec(default_mimic_config())
    return spec.compile()


def test_myotorso_tabletennis_isolated_xml_loads() -> None:
    """Isolated wrapper MJCF compiles (torso + arm + legs host)."""
    path = myotorso_tabletennis_isolated_xml_path()
    assert path.is_file()
    m = compile_myotorso_arm_host()
    assert m.nq > 0


def test_myotorso_arm_host_right_arm_hinge_parity_with_bimanual() -> None:
    """Right-arm hinges match when copied by name (``{j}`` vs ``{j}_r``)."""
    ref_m = _bimanual_ref_model()
    host_m = compile_myotorso_arm_host()
    rng = np.random.default_rng(42)
    ref_d = mujoco.MjData(ref_m)
    ref_d.qpos[:] = rng.uniform(-0.25, 0.25, size=ref_m.nq)
    mujoco.mj_forward(ref_m, ref_d)
    mapping = build_joint_value_map(ref_m, host_m, ref_d)
    assert len(mapping) >= 10, "expected a non-trivial right-arm hinge overlap"
    host_d = mujoco.MjData(host_m)
    for hj, val in mapping.items():
        adr = int(host_m.jnt_qposadr[hj])
        host_d.qpos[adr] = val
    mujoco.mj_forward(host_m, host_d)
    for hj, val in mapping.items():
        hname = host_m.joint(hj).name
        assert hname is not None
        bname = f"{hname}_r"
        rid = mujoco.mj_name2id(ref_m, mujoco.mjtObj.mjOBJ_JOINT, bname)
        assert rid >= 0
        rq = int(ref_m.jnt_qposadr[rid])
        assert float(ref_d.qpos[rq]) == pytest.approx(val, abs=1e-9)


def test_right_arm_body_names_align_stripped_suffix() -> None:
    """Bimanual ``X_r`` bodies align to host ``X`` for the shared arm chain."""
    ref_m = _bimanual_ref_model()
    host_m = compile_myotorso_arm_host()
    ref_bases: set[str] = set()
    for i in range(ref_m.nbody):
        n = mujoco.mj_id2name(ref_m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if n.endswith("_r") and not n.endswith("_l"):
            ref_bases.add(n[:-2])
    host_names: set[str] = set()
    for i in range(host_m.nbody):
        n = mujoco.mj_id2name(host_m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if n:
            host_names.add(n)
    for key in ("humerus", "ulna", "radius", "scapula", "clavicle"):
        assert key in host_names, f"missing host body {key!r}"
        assert (
            key in ref_bases
        ), f"missing ref counterpart for {key!r} (expected {key!r} as {key}_r)"
