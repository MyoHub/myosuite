"""Validate source identities without requiring Newton in ordinary CI."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest

# The probes are maintainer scripts next to their workflow, not package modules.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / ".github" / "newton_probe")
)
import newton_dynamics_probe as probe  # noqa: E402

pytestmark = pytest.mark.tier1

_XML = """<mujoco><worldbody><body name="body">
  <joint name="q1" axis="1 0 0"/><joint name="q2" axis="0 1 0"/>
  <geom type="sphere" size="0.1"/><site name="site-with-hyphen"/>
</body></worldbody></mujoco>"""


def _array(value: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(numpy=lambda: np.array(value))


def _models() -> tuple[mujoco.MjModel, SimpleNamespace, SimpleNamespace]:
    reference = mujoco.MjModel.from_xml_string(_XML)
    reordered = _XML.replace('name="q1" axis="1 0 0"', 'name="temporary" axis="0 1 0"')
    reordered = reordered.replace(
        'name="q2" axis="0 1 0"', 'name="export_q1" axis="1 0 0"'
    )
    reordered = reordered.replace('name="temporary"', 'name="export_q2"')
    reordered = reordered.replace('name="body"', 'name="export_body"').replace(
        'name="site-with-hyphen"', 'name="export_site"'
    )
    model = SimpleNamespace(
        body_label=["prefix/body"],
        shape_label=["prefix/body/site-with-hyphen"],
        joint_label=["prefix/body/q1_q2"],
        joint_qd_start=_array([0, 2]),
        mujoco=SimpleNamespace(joint_dof_label=["prefix/body/q1", "prefix/body/q2"]),
    )
    solver = SimpleNamespace(
        mj_model=mujoco.MjModel.from_xml_string(reordered),
        mjc_jnt_to_newton_dof=_array([[1, 0]]),
        mjc_jnt_to_newton_jnt=_array([[0, 0]]),
        mjc_body_to_newton=_array([[-1, 0]]),
        _mjc_site_to_newton_shape=_array([[0]]),
    )
    return reference, model, solver


@pytest.mark.parametrize("labelled", [True, False], ids=["dof-labels", "merged-name"])
def test_combined_joint_reordering_keeps_source_correspondence(labelled: bool) -> None:
    reference, model, solver = _models()
    if not labelled:
        model.mujoco.joint_dof_label = ["", ""]
    mapping, dofs = probe._joint_mapping(reference, model, solver)
    np.testing.assert_array_equal(mapping, [1, 0])
    assert dofs == {0: 0, 1: 1}
    identities = probe._identity_mapping(reference, model, solver, mapping)
    assert [(row["source"], row["converted"]) for row in identities["joint"]] == [
        ("q2", "export_q2"),
        ("q1", "export_q1"),
    ]
    assert identities["site"][0]["source"] == "site-with-hyphen"
    assert identities["body"][1]["source"] == "body"


@pytest.mark.parametrize("labelled", [True, False], ids=["dof-labels", "merged-name"])
def test_duplicate_dof_mapping_is_rejected(labelled: bool) -> None:
    reference, model, solver = _models()
    if not labelled:
        model.mujoco.joint_dof_label = ["", ""]
    solver.mjc_jnt_to_newton_dof = _array([[0, 0]])
    with pytest.raises(ValueError, match="complete correspondence"):
        probe._joint_mapping(reference, model, solver)


def test_merged_joint_dofs_put_slides_first() -> None:
    # Newton orders a merged joint's DoFs linear first, whatever the MJCF order.
    body = """<mujoco><worldbody><body name="body">{}
      <geom type="sphere" size="0.1"/></body></worldbody></mujoco>"""
    reference = mujoco.MjModel.from_xml_string(
        body.format('<joint name="h" axis="1 0 0"/><joint name="s" type="slide"/>')
    )
    model = SimpleNamespace(
        joint_label=["prefix/body/h_s"],
        joint_qd_start=_array([0, 2]),
        mujoco=SimpleNamespace(joint_dof_label=["", ""]),
    )
    solver = SimpleNamespace(
        mj_model=mujoco.MjModel.from_xml_string(
            body.format(
                '<joint name="lin" type="slide"/><joint name="ang" axis="1 0 0"/>'
            )
        ),
        mjc_jnt_to_newton_dof=_array([[0, 1]]),
        mjc_jnt_to_newton_jnt=_array([[0, 0]]),
    )
    mapping, dofs = probe._joint_mapping(reference, model, solver)
    np.testing.assert_array_equal(mapping, [1, 0])
    assert dofs == {1: 0, 0: 1}


def test_unknown_merged_joint_name_is_not_guessed() -> None:
    reference, model, solver = _models()
    model.mujoco.joint_dof_label = ["", ""]
    model.joint_label = ["prefix/body/q2_q1"]
    with pytest.raises(KeyError, match="q2_q1"):
        probe._joint_mapping(reference, model, solver)


def test_free_and_ball_joint_positions_are_mapped_as_blocks() -> None:
    xml = """<mujoco><worldbody><body><freejoint name="root-joint"/>
      <geom type="sphere" size="0.1"/><body pos="0 0 1">
        <joint name="ball-joint" type="ball"/><geom type="sphere" size="0.1"/>
      </body></body></worldbody></mujoco>"""
    reference = mujoco.MjModel.from_xml_string(xml)
    model = SimpleNamespace(
        mujoco=SimpleNamespace(joint_dof_label=[""] * 9),
        joint_label=["prefix/root-joint", "prefix/ball-joint"],
        joint_qd_start=_array([0, 6, 9]),
    )
    solver = SimpleNamespace(
        mj_model=mujoco.MjModel.from_xml_string(xml.replace("-joint", "-export")),
        mjc_jnt_to_newton_dof=_array([[0, 6]]),
        mjc_jnt_to_newton_jnt=_array([[0, 1]]),
    )
    mapping, dofs = probe._joint_mapping(reference, model, solver)
    np.testing.assert_array_equal(mapping, np.arange(11))
    assert not dofs


def test_incomplete_body_mapping_is_rejected() -> None:
    reference, model, solver = _models()
    mapping, _ = probe._joint_mapping(reference, model, solver)
    solver.mjc_body_to_newton = _array([[-1]])
    with pytest.raises(ValueError, match="body identity mapping"):
        probe._identity_mapping(reference, model, solver, mapping)


def test_solver_integrates_like_the_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml = tmp_path / "rk4.xml"
    xml.write_text('<mujoco><option integrator="RK4"/></mujoco>')
    calls: list[dict[str, object]] = []

    def solver(model: object, **options: object) -> None:
        calls.append(options)
        raise RuntimeError("stop after the first solver")

    builder = SimpleNamespace(add_mjcf=lambda *_, **__: None, finalize=lambda **_: None)
    newton = SimpleNamespace(
        ModelBuilder=lambda: builder, solvers=SimpleNamespace(SolverMuJoCo=solver)
    )
    monkeypatch.setitem(sys.modules, "newton", newton)
    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace())
    monkeypatch.setattr(probe, "authored_xml", lambda *_: xml)
    result = probe.run_case("rk4", "cpu", "cpu", 2, 1, tmp_path)
    assert result["integrator"] == "rk4"
    assert [call["integrator"] for call in calls] == ["rk4"]
    assert result["error"]["message"] == "stop after the first solver"


def test_cli_requires_multiple_poses(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        probe.main(["--poses", "1", "--out", str(tmp_path)])
