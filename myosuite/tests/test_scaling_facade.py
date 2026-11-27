"""Migration contracts for the canonical model-library scaler."""

from __future__ import annotations

import importlib

import mujoco
import numpy as np
import pytest

from myosuite.core import body_scaling

pytestmark = pytest.mark.tier1


def test_scaled_spec_is_an_independent_copy() -> None:
    pytest.importorskip("myo_sim.scaling")
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="limb"><joint/>'
        '<geom type="capsule" size=".02 .2"/></body></worldbody></mujoco>'
    )
    source = spec.compile()
    before = source.geom_size.copy()
    scaled = body_scaling.scale_spec(spec, {"limb": 1.2})
    assert scaled is not spec
    np.testing.assert_array_equal(spec.compile().geom_size, before)
    np.testing.assert_allclose(scaled.compile().geom_size, before * 1.2)


def test_missing_scaling_api_has_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = importlib.import_module

    def unavailable(name: str) -> object:
        if name == "myo_sim.scaling":
            raise ModuleNotFoundError(name, name=name)
        return original(name)

    monkeypatch.setattr(body_scaling.importlib, "import_module", unavailable)
    with pytest.raises(ImportError, match=r"myo_sim PR #149"):
        body_scaling.scale_model(mujoco.MjSpec(), {})


def test_unrelated_import_error_is_not_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(name: str) -> object:
        raise ModuleNotFoundError("unrelated dependency defect", name="broken_dep")

    monkeypatch.setattr(body_scaling.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError) as exc_info:
        body_scaling.scale_model(mujoco.MjSpec(), {})
    assert exc_info.value.name == "broken_dep"
