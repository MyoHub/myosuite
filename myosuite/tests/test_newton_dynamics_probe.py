"""Validate source identities without requiring Newton in ordinary CI."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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


def _array(value: list[list[int]]) -> SimpleNamespace:
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


def test_combined_joint_reordering_keeps_source_correspondence() -> None:
    reference, model, solver = _models()
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


def test_duplicate_dof_mapping_is_rejected() -> None:
    reference, model, solver = _models()
    solver.mjc_jnt_to_newton_dof = _array([[0, 0]])
    with pytest.raises(ValueError, match="complete correspondence"):
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


def test_cli_requires_multiple_poses(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        probe.main(["--poses", "1", "--out", str(tmp_path)])
