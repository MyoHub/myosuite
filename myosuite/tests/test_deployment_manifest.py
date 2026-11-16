"""Native physical identity and policy provenance boundary regressions."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco
import pytest

from myosuite.utils.deployment_manifest import (
    artifact_record,
    build_deployment_manifest,
    model_identity,
    validate_deployment_manifest,
)

pytestmark = pytest.mark.tier1

XML = """<mujoco><asset><texture name="checker" type="2d" builtin="checker" width="8" height="8"/></asset>
<worldbody><body name="arm"><joint name="hinge"/><geom size=".1"/></body></worldbody>
<actuator><motor name="motor" joint="hinge"/></actuator>
<equality><joint joint1="hinge" polycoef="0 1 0 0 0"/></equality></mujoco>"""


@pytest.mark.parametrize(
    "field",
    [
        "body_mass",
        "body_inertia",
        "actuator_gainprm",
        "geom_friction",
        "eq_data",
        "tex_data",
    ],
)
def test_physical_changes_do_not_reuse_a_kinematic_identity(field: str) -> None:
    model = mujoco.MjModel.from_xml_string(XML)
    before = model_identity(model)
    array = getattr(model, field)
    index = {
        "body_mass": (1,),
        "body_inertia": (1, 0),
        "actuator_gainprm": (0, 0),
        "geom_friction": (0, 0),
        "eq_data": (0, 0),
        "tex_data": (0,),
    }[field]
    array[index] += 1
    after = model_identity(model)
    assert before["physical_sha256"] != after["physical_sha256"]
    # A coupling polynomial is kinematic; other listed changes are physical only.
    if field != "eq_data":
        assert before["kinematic_fingerprint"] == after["kinematic_fingerprint"]


def test_solver_and_timing_changes_affect_identity() -> None:
    model = mujoco.MjModel.from_xml_string(XML)
    before = model_identity(model)
    model.opt.iterations += 1
    assert before != model_identity(model)
    model.opt.iterations -= 1
    assert before == model_identity(model)
    model.opt.timestep *= 2
    assert before["physical_sha256"] != model_identity(model)["physical_sha256"]


def test_compiled_asset_contents_survive_path_relocation(tmp_path: Path) -> None:
    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
    models = []
    for folder in (tmp_path / "a", tmp_path / "longer-machine-local-prefix"):
        folder.mkdir()
        mesh = folder / "tetra.obj"
        mesh.write_text(obj)
        xml = f'<mujoco><asset><mesh name="mesh" file="{mesh.as_posix()}"/></asset><worldbody><geom type="mesh" mesh="mesh"/></worldbody></mujoco>'
        models.append(mujoco.MjModel.from_xml_string(xml))
    assert models[0].paths != models[1].paths
    assert model_identity(models[0]) == model_identity(models[1])
    models[1].mesh_vert[0, 0] += 0.01
    assert model_identity(models[0]) != model_identity(models[1])


def test_native_force_change_is_detected_without_any_space_change() -> None:
    xml = XML.replace(
        '<equality><joint joint1="hinge" polycoef="0 1 0 0 0"/></equality>', ""
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.ctrl[:] = 1
    mujoco.mj_forward(model, data)
    force = data.actuator_force.copy()
    identity = model_identity(model)
    model.actuator_gainprm[0, 0] *= 2
    mujoco.mj_forward(model, data)
    assert data.actuator_force[0] == pytest.approx(2 * force[0])
    assert model_identity(model)["physical_sha256"] != identity["physical_sha256"]
    assert (
        model_identity(model)["kinematic_fingerprint"]
        == identity["kinematic_fingerprint"]
    )


def _manifest(model: mujoco.MjModel) -> dict:
    return build_deployment_manifest(
        model,
        interface={
            "observations": [
                {"name": "angle", "units": "rad"},
                {"name": "speed", "units": "rad/s"},
            ],
            "actions": ["motor"],
            "history": {"length": 2, "order": "oldest-first"},
        },
        control_dt=0.02,
        normalization={"kind": "embedded", "input": "raw"},
    )


@pytest.mark.parametrize(
    "field",
    [
        "interface",
        "normalization",
        "timing",
        "physics_options",
        "references",
        "units",
        "history",
    ],
)
def test_validation_rejects_semantic_changes_and_allows_explicit_override(
    field: str,
) -> None:
    manifest = _manifest(mujoco.MjModel.from_xml_string(XML))
    current = copy.deepcopy(manifest)
    if field == "interface":
        current[field]["observations"].reverse()  # Same width, different meaning.
    elif field == "units":
        current["interface"]["observations"][0]["units"] = "deg"
        field = "interface"
    elif field == "history":
        current["interface"]["history"]["order"] = "newest-first"
        field = "interface"
    else:
        current[field]["changed"] = True
    with pytest.raises(ValueError, match=field):
        validate_deployment_manifest(manifest, current=current)
    with pytest.warns(UserWarning, match="explicit override"):
        assert validate_deployment_manifest(
            manifest, current=current, allow_mismatch=True
        ) == [field]


def test_control_timing_requires_exact_substeps() -> None:
    model = mujoco.MjModel.from_xml_string(XML)
    for dt in (0, -0.02, 0.003, float("nan")):
        with pytest.raises(ValueError, match="integer multiple"):
            build_deployment_manifest(model, interface={}, control_dt=dt)


def test_artifacts_and_pinned_references_are_data_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"local test fixture, not model weights")
    record = artifact_record(
        checkpoint,
        source={
            "repo_id": "MyoHub/baselines",
            "filename": "model.pt",
            "revision": "a" * 40,
        },
    )
    manifest = build_deployment_manifest(
        mujoco.MjModel.from_xml_string(XML),
        interface={},
        control_dt=0.02,
        artifacts={checkpoint.name: checkpoint},
        references={"motion": record},
        checkpoint=record,
    )
    assert manifest["references"]["motion"]["sha256"] == record["sha256"]
    assert "torch" in manifest["runtime"]["dependencies"]
    assert validate_deployment_manifest(manifest, artifact_root=tmp_path) == []
    assert str(tmp_path) not in json.dumps(manifest)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact model.pt"):
        validate_deployment_manifest(manifest, artifact_root=tmp_path)
    for revision in ("main", "HEAD", "master"):
        with pytest.raises(ValueError, match="pinned"):
            artifact_record(
                checkpoint,
                source={"repo_id": "r", "filename": "x", "revision": revision},
            )
    with pytest.raises(ValueError, match="relative POSIX"):
        build_deployment_manifest(
            mujoco.MjModel.from_xml_string(XML),
            interface={},
            control_dt=0.02,
            artifacts={"../escape": checkpoint},
        )
