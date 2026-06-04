# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke tests for MyoTorso + MuscleMimic bimanual composite MJCF."""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

import myosuite
from myosuite.integrations.musclemimic.bimanual_model import (
    default_mimic_config,
)
from myosuite.integrations.musclemimic.myotorso_bimanual_model import (
    build_myotorso_bimanual_mimic_spec,
    compile_myotorso_bimanual_mimic_mjmodel,
    save_myotorso_bimanual_mimic_xml,
)


def _config():
    return default_mimic_config()


def test_myotorso_bimanual_mimic_spec_compiles() -> None:
    """Composite spec should load and compile with dual arms + torso + legs."""
    try:
        spec, tag = build_myotorso_bimanual_mimic_spec(_config())
    except ImportError as err:  # pragma: no cover
        pytest.skip(str(err))
    assert tag == "myotorso_bimanual_mimic"
    m = spec.compile()
    assert m.nq >= 48
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "thorax") >= 0
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bimanual_attachment")
    assert bid >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "humerus_r") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "humerus_l") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "L1_L2_FE") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_x") < 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hip_flexion_r") < 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "femur_r") >= 0


def test_myotorso_bimanual_mimic_mjmodel_solver_options() -> None:
    """Compiled model uses Mimic timestep and solver flags from config."""
    try:
        mj, _, _ = compile_myotorso_bimanual_mimic_mjmodel(_config())
    except ImportError as err:  # pragma: no cover
        pytest.skip(str(err))
    cfg = _config()
    assert mj.opt.timestep == float(cfg.sim_dt)
    assert mj.opt.iterations == int(cfg.model_iterations)


def test_myotorso_bimanual_mimic_saved_xml_loads() -> None:
    """Vendored monolithic MJCF under simhive/myo_sim should compile."""
    xml = (
        Path(myosuite.__file__).resolve().parent
        / "simhive"
        / "myo_sim"
        / "myotorso_bimanual_mimic.xml"
    )
    if not xml.is_file():
        try:
            save_myotorso_bimanual_mimic_xml(dest=xml)
        except ImportError as err:  # pragma: no cover
            pytest.skip(f"missing {xml} and cannot generate (musclemimic): {err}")
    assert xml.is_file(), f"missing {xml}"
    m = mujoco.MjModel.from_xml_path(str(xml))
    assert m.nq >= 48
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bimanual_attachment")
    assert bid >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "humerus_r") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "humerus_l") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hip_flexion_r") < 0
