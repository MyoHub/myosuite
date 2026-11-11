# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Opt-in tests execute the importer in a real Blender and MuSkeMo installation."""

import os
from pathlib import Path
import subprocess

import myo_sim
import pytest

from myosuite.integrations.muskemo import export_model, joint_sweep

pytestmark = pytest.mark.tier2


@pytest.mark.parametrize(
    "name,count", [("myoelbow", 6), ("myoarm_r", 63), ("myolegs", 80)]
)
def test_real_blender_import(tmp_path, name, count):
    blender, addon = os.environ.get("MYOSUITE_BLENDER"), os.environ.get("MUSKEMO_ADDON")
    if not blender or not addon:
        pytest.skip("Set MYOSUITE_BLENDER and MUSKEMO_ADDON for real Blender tests")
    model = myo_sim.load_spec(name).compile()
    poses = joint_sweep(model, "r_elbow_flex", 5) if name == "myoelbow" else None
    document = export_model(model, tmp_path / "anatomy.json", qpos=poses)
    root = Path(__file__).resolve().parents[2]
    code = f"""
import importlib.util, json, math
from pathlib import Path
import bpy
spec = importlib.util.spec_from_file_location("bridge", {str(root / "myosuite/integrations/muskemo_blender.py")!r})
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
data = json.loads(Path({str(tmp_path / "anatomy.json")!r}).read_text())
objects = bridge.import_model({str(tmp_path / "anatomy.json")!r}, addon={addon!r})
assert len(objects["muscle"]) == {count}
def close(a, b):
    assert max((abs(x-y) for x,y in zip(a,b)), default=0) < 3e-6, (a,b)
for frame, pose in enumerate(data["frames"], start=1):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    for body in data["bodies"]:
        close(objects["body"][body["id"]].matrix_world.translation, pose["body_pos"][body["id"]])
        quat = objects["body"][body["id"]].matrix_world.to_quaternion()
        assert abs(sum(a*b for a,b in zip(quat, pose["body_quat"][body["id"]]))) > 1-1e-6
    for site in data["sites"]:
        close(objects["site"][site["id"]].matrix_world.translation, pose["site_pos"][site["id"]])
    for geom in data["geoms"]:
        close(objects["geom"][geom["id"]].matrix_world.translation, pose["geom_pos"][geom["id"]])
        quat = objects["geom"][geom["id"]].matrix_world.to_quaternion()
        assert abs(sum(a*b for a,b in zip(quat, pose["geom_quat"][geom["id"]]))) > 1-1e-6
for muscle in data["muscles"]:
    obj = objects["muscle"][muscle["id"]]
    elements = data["tendons"][muscle["tendon"]]["elements"]
    wraps = [e for e in elements if e["type"] in ("SPHERE", "CYLINDER")]
    modifiers = [m for m in obj.modifiers if "_wrap_" in m.name]
    assert len(modifiers) == len(wraps)
    assert len([m for m in obj.modifiers if m.type == "HOOK"]) == len(obj["mujoco_site_ids"])
before = set(bpy.data.objects)
try:
    bridge.import_model({str(tmp_path / "anatomy.json")!r}, addon={addon!r})
except ValueError:
    pass
else:
    raise AssertionError("Duplicate namespace accepted")
assert before == set(bpy.data.objects)
bad = dict(data)
bad["muscles"] = [dict(data["muscles"][0], tendon=-100)]
badpath = Path({str(tmp_path / "bad.json")!r})
badpath.write_text(json.dumps(bad))
before = set(bpy.data.objects)
try:
    bridge.import_model(badpath, addon={addon!r}, prefix="Bad")
except (ValueError, IndexError):
    pass
else:
    raise AssertionError("Malformed path accepted")
assert before == set(bpy.data.objects)
bridge.register()
print("MUSKEMO_VALIDATED")
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
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MUSKEMO_VALIDATED" in result.stdout
    assert len(document["muscles"]) == count
