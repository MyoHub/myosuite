# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Geared geometry and selected edits retain their source coordinate contracts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap
from typing import Any

import mujoco
import numpy as np
import pytest

from myosuite.integrations.muskemo import export_model, joint_sweep

pytestmark = pytest.mark.tier2


def _export(path: Path) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <site name="origin" pos="0 1 0"/>
      <body name="b">
        <inertial pos=".1 .2 .3" euler="10 20 30" mass="1" diaginertia=".1 .2 .25"/>
        <joint name="q" axis="0 0 1" range="-45 45"/><geom size=".1"/>
        <site name="insert" pos="1 0 0"/><site pos=".1 .2 .3"/>
      </body></worldbody>
      <tendon><spatial name="t"><site site="origin"/><site site="insert"/></spatial></tendon>
      <actuator><muscle name="m" tendon="t" lengthrange=".1 4" gear="2" force="1"/></actuator>
    </mujoco>""")
    return export_model(model, path, qpos=joint_sweep(model, "q", 5))


def test_export_distinguishes_physical_lengths_and_generated_names(
    tmp_path: Path,
) -> None:
    """The physical reference agrees with independent two-site Euclidean geometry."""
    document = _export(tmp_path / "anatomy.json")
    assert np.asarray(document["muscles"][0]["gear"]).reshape(-1)[0] == 2
    assert document["bodies"][1]["source_name"] == "b"
    assert document["sites"][0]["source_name"] == "origin"
    assert document["sites"][2]["source_name"] == ""
    assert document["sites"][2]["name"]
    for frame in document["frames"]:
        sites = np.array(frame["site_pos"])
        physical = np.linalg.norm(sites[0] - sites[1])
        np.testing.assert_allclose(
            frame["muscle_tendon_length"], [physical], atol=1e-12
        )
        np.testing.assert_allclose(frame["muscle_length"], [2 * physical], atol=1e-12)


@pytest.fixture
def native_blender(tmp_path: Path) -> tuple[str, str, Path]:
    blender = os.environ.get("MYOSUITE_BLENDER")
    addon = os.environ.get("MUSKEMO_ADDON")
    if not blender or not addon:
        pytest.skip("Set MYOSUITE_BLENDER and MUSKEMO_ADDON for real Blender tests")
    anatomy = tmp_path / "anatomy.json"
    _export(anatomy)
    return blender, addon, anatomy


def _run_blender(
    native: tuple[str, str, Path],
    tmp_path: Path,
    checks: str,
    *,
    prefix: str = "MyoSuite",
) -> None:
    blender, addon, anatomy = native
    importer = (
        Path(__file__).resolve().parents[1] / "integrations" / "muskemo_blender.py"
    )
    code = f"""
import importlib.util
import json
from pathlib import Path
import re
import bpy
spec = importlib.util.spec_from_file_location("bridge", {str(importer)!r})
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
document = json.loads(Path({str(anatomy)!r}).read_text(encoding="utf-8"))
objects = bridge.import_model({str(anatomy)!r}, addon={addon!r}, prefix={prefix!r})
output = Path({str(tmp_path / "patch.json")!r})
bpy.context.scene.frame_set(1)
bpy.ops.object.select_all(action="DESELECT")

def reject_patch(pattern):
    try:
        bridge.export_patch(output)
    except ValueError as error:
        assert re.search(pattern, str(error), re.IGNORECASE), str(error)
    else:
        raise AssertionError("Invalid/no-op edit exported a patch")
    assert not output.exists(), "Rejected edits must not publish a patch"

{textwrap.dedent(checks)}
print("MUSKEMO_BOUNDARY_VALIDATED")
"""
    script = tmp_path / "verify.py"
    script.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [
            blender,
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(script),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MUSKEMO_BOUNDARY_VALIDATED" in result.stdout


def test_geared_path_has_no_geometry_error(
    native_blender: tuple[str, str, Path], tmp_path: Path
) -> None:
    """A unit-exact physical path does not acquire a gear-induced length/arm error."""
    _run_blender(
        native_blender,
        tmp_path,
        """
        report = bridge.geometry_report(objects, document)
        assert max(report["max_length_error_m"]) < 2e-6, report
        assert report["sweep_joint"] == "q"
        assert max(report["max_effective_moment_arm_error"]) < 1e-5, report
        for actual, frame in zip(report["reference_lengths_m"], document["frames"]):
            assert abs(actual[0] - frame["muscle_tendon_length"][0]) < 1e-12
    """,
    )


def test_unchanged_body_is_not_an_inertia_edit(
    native_blender: tuple[str, str, Path], tmp_path: Path
) -> None:
    """A rotated, non-spherical inertia has a no-op guard without hiding real mass edits."""
    _run_blender(
        native_blender,
        tmp_path,
        """
        body = objects["body"][1]
        body.select_set(True)
        reject_patch("select edited|no changes|no .*edits")
        body["mass"] += .25
        patch = bridge.export_patch(output)
        assert len(patch["inertias"]) == 1 and patch["sites"] == []
        assert patch["inertias"][0]["name"] == "b"
        assert abs(patch["inertias"][0]["mass"] - 1.25) < 1e-12
    """,
    )


@pytest.mark.parametrize("change", ["delete", "add", "spline_type", "extra_spline"])
def test_changed_curve_topology_is_rejected(
    native_blender: tuple[str, str, Path], tmp_path: Path, change: str
) -> None:
    """Point IDs cannot silently shift or truncate after a curve topology edit."""
    _run_blender(
        native_blender,
        tmp_path,
        f"""
        curve = objects["muscle"][0]
        curve.select_set(True)
        bpy.context.view_layer.objects.active = curve
        spline = curve.data.splines[0]
        count = len(spline.points)
        if {change!r} == "delete":
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.curve.select_all(action="DESELECT")
            curve.data.splines[0].points[0].select = True
            bpy.ops.curve.delete(type="VERT")
            bpy.ops.object.mode_set(mode="OBJECT")
            assert len(curve.data.splines[0].points) == count - 1
        elif {change!r} == "add":
            spline.points.add(1)
            spline.points[-1].co = (3, 4, 5, 1)
            assert len(spline.points) == count + 1
        elif {change!r} == "spline_type":
            spline.type = "NURBS"
            assert len(spline.points) == count and spline.type == "NURBS"
        else:
            curve.data.splines.new("POLY")
            assert len(curve.data.splines) == 2
        bpy.context.view_layer.update()
        reject_patch("topology|point count|spline")
    """,
    )


def test_edited_unnamed_site_is_rejected_clearly(
    native_blender: tuple[str, str, Path], tmp_path: Path
) -> None:
    """Generated display labels cannot escape as nonexistent source edit names."""
    _run_blender(
        native_blender,
        tmp_path,
        """
        unnamed = next(s for s in document["sites"] if s["source_name"] == "")
        site = objects["site"][unnamed["id"]]
        site.select_set(True)
        site.location.x += .02
        bpy.context.view_layer.update()
        reject_patch("unnamed|source.name")
    """,
    )


def test_moving_free_root_is_not_labelled_a_scalar_joint_sweep(
    native_blender: tuple[str, str, Path],
    tmp_path: Path,
) -> None:
    """Length changes with multiple independent coordinates are not moment arms."""
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <site name="origin" pos="0 1 0"/><body name="root"><freejoint/>
      <geom size=".1"/><body name="b"><joint name="q" range="-45 45"/>
      <geom size=".1"/><site name="insert" pos="1 0 0"/></body></body></worldbody>
      <tendon><spatial name="t"><site site="origin"/><site site="insert"/></spatial></tendon>
      <actuator><muscle name="m" tendon="t" lengthrange=".1 4" force="1"/></actuator>
      </mujoco>""")
    poses = joint_sweep(model, "q", 5)
    poses[:, 0] = np.linspace(0, 0.1, len(poses))
    export_model(model, native_blender[2], qpos=poses)
    _run_blender(
        native_blender,
        tmp_path,
        """
        report = bridge.geometry_report(objects, document)
        assert "sweep_joint" not in report
        assert "max_effective_moment_arm_error" not in report
    """,
    )


def test_wrapping_survives_blender_modifier_name_truncation(
    native_blender: tuple[str, str, Path],
    tmp_path: Path,
) -> None:
    """A valid long namespace can exceed Blender's modifier-name byte limit."""
    import myo_sim

    export_model(myo_sim.load_spec("myoelbow").compile(), native_blender[2])
    _run_blender(
        native_blender,
        tmp_path,
        """
        assert len(objects["muscle"]) == 6
        assert sum("_wrap_" in m.name for obj in objects["muscle"].values()
                   for m in obj.modifiers) == 6
    """,
        prefix="ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
    )
