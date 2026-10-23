"""Probe regressions that run without Newton or a GPU in the regular test suite."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from scripts import newton_compat_probe as probe

pytestmark = pytest.mark.tier1

_XML = """<mujoco>
  <worldbody><body name="body"><joint name="joint" type="slide"/>
    <geom type="sphere" size="0.1" mass="1"/>
  </body></worldbody>
  <actuator><muscle name="muscle" joint="joint" lengthrange="0.5 1.5" force="5"/></actuator>
</mujoco>"""


def test_compare_identical_model_and_resolved_actuator_names() -> None:
    """Resolved Newton labels participate in both force and name comparison."""
    ref = mujoco.MjModel.from_xml_string(_XML)
    new = mujoco.MjModel.from_xml_string(_XML.replace('name="muscle"', ""))
    result = probe.compare(ref, new, ["muscle"], 1e-6, 1e-8)
    assert not probe._defective(result)
    assert result["names_kept"]["actuator"] == "1/1"
    assert result["forces"]["compared"] == 1


@pytest.mark.parametrize("change", ["lengthrange", "mass", "integrator", "muscle"])
def test_strict_detects_changes_even_when_forces_match(change: str) -> None:
    """A preserved force alone cannot prove that conversion preserved a model."""
    ref = mujoco.MjModel.from_xml_string(_XML)
    new = mujoco.MjModel.from_xml_string(_XML)
    if change == "lengthrange":
        new.actuator_lengthrange[0] *= 1.01
    elif change == "mass":
        new.body_mass[1] *= 2
    elif change == "integrator":
        new.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    else:
        new.actuator_gaintype[0] = mujoco.mjtGain.mjGAIN_FIXED
    result = probe.compare(ref, new, ["muscle"], 1e-6, 1e-8)
    result["forces"] = {"mismatched": 0, "nonfinite": 0}
    assert probe._defective(result)


def test_agreement_nonfinite_reference_is_a_failure_and_valid_json() -> None:
    result = probe.agreement(
        ["finite", "inf", "nan"],
        np.array([1.0, np.inf, np.nan]),
        np.array([1.0, np.inf, 0.0]),
        1e-6,
        1e-8,
    )
    assert result["mismatched"] == result["nonfinite"] == 2
    assert result["max_abs_diff"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("names", [[], ["muscle", "extra"]])
def test_compare_rejects_incomplete_actuator_mapping(names: list[str]) -> None:
    model = mujoco.MjModel.from_xml_string(_XML)
    with pytest.raises(ValueError, match="one name per"):
        probe.compare(model, model, names, 1e-6, 1e-8)


@pytest.mark.parametrize("index", [-1, 1])
def test_converted_actuator_mapping_rejects_invalid_label_index(index: int) -> None:
    model = mujoco.MjModel.from_xml_string(_XML.replace('name="muscle"', ""))
    solver = SimpleNamespace(
        mj_model=model,
        CtrlSource=SimpleNamespace(CTRL_DIRECT=0),
        mjc_actuator_to_newton_idx=SimpleNamespace(numpy=lambda: np.array([index])),
        mjc_actuator_ctrl_source=SimpleNamespace(numpy=lambda: np.array([0])),
    )
    newton_model = SimpleNamespace(mujoco=SimpleNamespace(actuator_label=["muscle"]))
    with pytest.raises(ValueError, match="label index"):
        probe.converted_actuator_names(solver, newton_model)


def test_normalization_preserves_muscle_and_references(tmp_path: Path) -> None:
    xml = _XML.replace('name="joint"', 'name="a-joint"').replace(
        'joint="joint"', 'joint="a-joint"'
    )
    source = tmp_path / "source.xml"
    source.write_text(xml, encoding="utf-8")
    dest = probe.normalized_xml(source, tmp_path / "normalized.xml")
    normalized = mujoco.MjModel.from_xml_path(str(dest))
    assert normalized.joint("a_joint").id == 0
    result = probe.compare(normalized, normalized, ["muscle"], 1e-6, 1e-8)
    assert not probe._defective(result)


def test_normalization_rejects_name_collision(tmp_path: Path) -> None:
    source = tmp_path / "source.xml"
    source.write_text(
        _XML.replace(
            '<geom type="sphere"',
            '<site name="a-b"/><site name="a_b"/><geom type="sphere"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="merge names"):
        probe.normalized_xml(source, tmp_path / "normalized.xml")


def test_probe_records_input_error_without_stopping_other_cases(tmp_path: Path) -> None:
    result = probe.probe("not-a-registered-env", "authored", tmp_path, 1e-6, 1e-8)
    assert result["error"]["stage"] == "input"
    assert probe._defective(result)


def test_capture_collects_newton_logging_and_restores_logger() -> None:
    notes = {}
    logger = logging.getLogger("newton")
    handlers, propagate = list(logger.handlers), logger.propagate
    with probe._captured(notes):
        logging.getLogger("newton.import_mjcf").warning("Unsupported element 'mesh1'")
    assert sum(entry[0] for entry in notes.values()) == 1
    assert logger.handlers == handlers
    assert logger.propagate == propagate


def test_main_records_case_failure_and_strict_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One broken model still produces both reports and cannot look successful."""
    failure = {
        "model": "missing",
        "input": "authored",
        "messages": [],
        "error": {"stage": "input", "type": "ValueError", "message": "model not found"},
    }
    monkeypatch.setattr(probe, "probe", lambda *args: failure)
    args = ["--out", str(tmp_path), "--models", "missing", "--inputs", "authored"]
    assert probe.main(args) == 0
    assert probe.main([*args, "--strict"]) == 1
    assert json.loads((tmp_path / "newton_probe.json").read_text())["cases"] == [
        failure
    ]
    assert "failed at input" in (tmp_path / "newton_probe.md").read_text()


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_main_rejects_invalid_tolerance(value: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        probe.main(["--out", str(tmp_path), "--rtol", value])
    assert error.value.code == 2
