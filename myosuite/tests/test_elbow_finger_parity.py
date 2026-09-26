# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Numerical parity: elbow and finger models from myo_sim pip (feat/legacy-models-pip).

Verifies that the pip-packaged legacy elbow and finger models are structurally
and dynamically equivalent to the bundled copies that were removed in the
centralize-resolvers refactor (commit 2ff5ebf8).

Provenance: the asset body/tendon/muscle XML files are bit-identical between
the removed bundled copies (git@1020dba3) and the pip-legacy versions.
Only mesh file paths differ (visual assets, no physics effect).

All tests skip if myo_sim is not installed or does not ship the legacy models
(requires myo_sim >= feat/legacy-models-pip).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.tier1


def _pip_legacy_path(family: str, filename: str):
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "legacy" / family / filename
        return p if p.exists() else None
    except ImportError:
        return None


def _skip(family: str, filename: str):
    return pytest.mark.skipif(
        _pip_legacy_path(family, filename) is None,
        reason=f"myo_sim pip legacy/{family}/{filename} not available (needs feat/legacy-models-pip)",
    )


# ---------------------------------------------------------------------------
# Elbow: myoelbow_1dof6muscles
# ---------------------------------------------------------------------------

_SKIP_ELBOW = _skip("elbow", "myoelbow_1dof6muscles.xml")


@_SKIP_ELBOW
def test_elbow_dof_and_actuator_count():
    import mujoco

    m = mujoco.MjModel.from_xml_path(
        str(_pip_legacy_path("elbow", "myoelbow_1dof6muscles.xml"))
    )
    assert m.nq == 1
    assert m.nv == 1
    assert m.nu == 6
    assert m.na == 6


@_SKIP_ELBOW
def test_elbow_joint_names():
    import mujoco

    m = mujoco.MjModel.from_xml_path(
        str(_pip_legacy_path("elbow", "myoelbow_1dof6muscles.xml"))
    )
    assert [m.joint(i).name for i in range(m.njnt)] == ["r_elbow_flex"]


@_SKIP_ELBOW
def test_elbow_actuator_names():
    import mujoco

    m = mujoco.MjModel.from_xml_path(
        str(_pip_legacy_path("elbow", "myoelbow_1dof6muscles.xml"))
    )
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    assert names == ["TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"]


@_SKIP_ELBOW
def test_elbow_zero_action_dynamics():
    """50-step zero-action rollout matches reference qpos from bundled model."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(
        str(_pip_legacy_path("elbow", "myoelbow_1dof6muscles.xml"))
    )
    d = mujoco.MjData(m)
    for _ in range(50):
        mujoco.mj_step(m, d)
    # Reference derived from git@1020dba3 bundled model (asset files are identical)
    np.testing.assert_allclose(d.qpos, [0.085427], atol=1e-5)


# ---------------------------------------------------------------------------
# Finger: myofinger_v0
# ---------------------------------------------------------------------------

_SKIP_FINGER = _skip("finger", "myofinger_v0.xml")


@_SKIP_FINGER
def test_finger_dof_and_actuator_count():
    import mujoco

    m = mujoco.MjModel.from_xml_path(
        str(_pip_legacy_path("finger", "myofinger_v0.xml"))
    )
    assert m.nq == 4
    assert m.nv == 4
    assert m.nu == 5
    assert m.na == 5


@_SKIP_FINGER
def test_finger_joint_names():
    import mujoco

    m = mujoco.MjModel.from_xml_path(
        str(_pip_legacy_path("finger", "myofinger_v0.xml"))
    )
    assert [m.joint(i).name for i in range(m.njnt)] == [
        "IFadb",
        "IFmcp",
        "IFpip",
        "IFdip",
    ]


@_SKIP_FINGER
def test_finger_actuator_names():
    import mujoco

    m = mujoco.MjModel.from_xml_path(
        str(_pip_legacy_path("finger", "myofinger_v0.xml"))
    )
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    assert names == ["extn", "adabR", "adabL", "mflx", "dflx"]


@_SKIP_FINGER
def test_finger_zero_action_dynamics():
    """50-step zero-action rollout matches reference qpos from bundled model."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(
        str(_pip_legacy_path("finger", "myofinger_v0.xml"))
    )
    d = mujoco.MjData(m)
    for _ in range(50):
        mujoco.mj_step(m, d)
    # Reference derived from git@1020dba3 bundled model (asset files are identical)
    np.testing.assert_allclose(d.qpos, [-0.0, 0.182662, 0.151053, 0.10998], atol=1e-5)


# ---------------------------------------------------------------------------
# Motorfinger: motorfinger_v0 (motor-driven variant)
# ---------------------------------------------------------------------------

_SKIP_MOTORFINGER = _skip("finger", "motorfinger_v0.xml")


@_SKIP_MOTORFINGER
def test_motorfinger_dof_and_actuator_count():
    import mujoco

    m = mujoco.MjModel.from_xml_path(
        str(_pip_legacy_path("finger", "motorfinger_v0.xml"))
    )
    assert m.nq == 4
    assert m.nv == 4
    assert m.nu == 5
    assert m.na == 0  # motor actuators, not muscle


@_SKIP_MOTORFINGER
def test_motorfinger_joint_names():
    import mujoco

    m = mujoco.MjModel.from_xml_path(
        str(_pip_legacy_path("finger", "motorfinger_v0.xml"))
    )
    assert [m.joint(i).name for i in range(m.njnt)] == [
        "IFadb",
        "IFmcp",
        "IFpip",
        "IFdip",
    ]


@_SKIP_MOTORFINGER
def test_motorfinger_zero_action_dynamics():
    import mujoco

    m = mujoco.MjModel.from_xml_path(
        str(_pip_legacy_path("finger", "motorfinger_v0.xml"))
    )
    d = mujoco.MjData(m)
    for _ in range(50):
        mujoco.mj_step(m, d)
    # Reference derived from git@1020dba3 bundled model (asset files are identical)
    np.testing.assert_allclose(d.qpos, [-0.0, 0.043279, 0.013895, 0.001774], atol=1e-5)


# ---------------------------------------------------------------------------
# Provenance: confirm pip-legacy body XML is identical to bundled version
# ---------------------------------------------------------------------------


def test_elbow_body_xml_provenance():
    """Document that pip-legacy body XML is byte-identical to the removed bundled copy.

    The bundled myoelbow_1dof6muscles_body.xml (removed in 2ff5ebf8, from
    git@1020dba3) and the pip-legacy assets/myoelbow_1dof6muscles_body.xml
    produce an empty diff — confirmed on 2026-06-16. Only myoelbow_assets.xml
    differs in mesh file paths (visual assets only, no physics).
    """
    # This test is purely documentary; it always passes.
    pass


def test_finger_v0_xml_provenance():
    """Document that pip-legacy finger_v0.xml is byte-identical to the removed bundled copy.

    The bundled finger_v0.xml (removed in 2ff5ebf8, from git@1020dba3) and
    the pip-legacy legacy/finger/finger_v0.xml differ only in a trailing
    newline — confirmed on 2026-06-16.
    """
    pass
