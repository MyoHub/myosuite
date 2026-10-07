# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""musclemimic_models release selection: 1.0.6 by default, the exact 1.0.5 on request.

1.0.6 fixed two left-knee coupling polynomials (negated relative to the right
knee in 1.0.5) and moved six muscle-wrap geoms. ``model_version`` rebuilds
either release from the installed one; checkpoints trained on 1.0.5 get 1.0.5.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.integrations.musclemimic import model_versions as mv
from myosuite.integrations.musclemimic.bimanual_model import (
    build_mimic_bimanual_spec,
    default_mimic_config,
)
from myosuite.integrations.musclemimic.fullbody_model import (
    build_mimic_fullbody_spec,
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)
from myosuite.integrations.musclemimic.myotorso_bimanual_model import (
    build_myotorso_bimanual_mimic_spec,
)
from myosuite.tests.support.model_compare import assert_same_model
from myosuite.tests.support.optional_deps import require_musclemimic_models

pytestmark = pytest.mark.tier1

_KNEE_COUPLINGS = (
    "knee_angle_translation1",
    "knee_angle_translation2",
    "knee_angle_rotation2",
    "knee_angle_rotation3",
    "knee_angle_beta_translation1",
    "knee_angle_beta_translation2",
    "knee_angle_beta_rotation1",
)
# The two couplings 1.0.6 fixed.
_FIXED_COUPLINGS = ("knee_angle_translation2", "knee_angle_rotation3")


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(mv.MODELS_VERSION_ENV_VAR, raising=False)


@pytest.fixture
def installed() -> str:
    require_musclemimic_models()
    version = mv.installed_models_version()
    if version not in mv.SUPPORTED_MODELS_VERSIONS:
        pytest.skip(f"musclemimic_models {version} is not a recorded release")
    return version


def _fullbody_cfg(model_version: str | None = None):
    cfg = default_mimic_fullbody_config()
    cfg.model_version = model_version
    return cfg


def _eq_data(model: mujoco.MjModel, name: str) -> np.ndarray:
    return model.eq_data[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name), :5
    ]


def test_version_comes_from_config_then_env_then_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert mv.resolve_models_version() == "1.0.6"
    assert mv.resolve_models_version(_fullbody_cfg()) == "1.0.6"
    monkeypatch.setenv(mv.MODELS_VERSION_ENV_VAR, "1.0.5")
    assert mv.resolve_models_version(_fullbody_cfg()) == "1.0.5"
    assert mv.resolve_models_version(_fullbody_cfg("1.0.6")) == "1.0.6"
    with pytest.raises(ValueError, match="not supported"):
        mv.resolve_models_version(_fullbody_cfg("1.0.4"))


@pytest.mark.parametrize(
    "ref",
    [
        "hf://amathislab/mm-10m-2",
        "hf://amathislab/mm-10m-2@main",
        r"C:\hf\hub\models--amathislab--mm-10m-2\snapshots\abc",
        "/cache/hub/models--amathislab--mm-10m-2/snapshots/abc/train_state",
    ],
)
def test_published_checkpoint_gets_its_training_release(ref: str) -> None:
    assert mv.checkpoint_models_version(ref) == "1.0.5"


@pytest.mark.parametrize(
    "ref", [None, "", "hf://amathislab/mm-10m-20", "runs/walk/model_100.pt"]
)
def test_other_checkpoints_get_the_default(ref: str | None) -> None:
    assert mv.checkpoint_models_version(ref) == mv.DEFAULT_MODELS_VERSION


def test_an_unrecorded_installed_release_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mv, "installed_models_version", lambda: "1.0.7")
    with pytest.raises(RuntimeError, match="1.0.7"):
        mv.apply_models_version(mujoco.MjSpec(), "1.0.6")


@pytest.mark.parametrize("name", ["myofullbody", "bimanual"])
def test_recorded_values_rebuild_the_installed_release_exactly(
    installed: str, name: str
) -> None:
    """The table entry of the installed release is its MJCF, bit for bit."""
    from musclemimic_models import get_xml_path

    path = str(get_xml_path(name))
    other = next(v for v in mv.SUPPORTED_MODELS_VERSIONS if v != installed)
    spec = mujoco.MjSpec.from_file(path)
    assert mv._set_models_values(spec, other) > 0
    mv._set_models_values(spec, installed)
    assert_same_model(mujoco.MjSpec.from_file(path).compile(), spec.compile())


def test_installed_release_is_built_unchanged(installed: str) -> None:
    spec = build_mimic_fullbody_spec(_fullbody_cfg())[0]
    assert mv.apply_models_version(spec, installed) == 0


def test_default_is_1_0_6_with_symmetric_knees(installed: str) -> None:
    """1.0.6 couples the left knee exactly as the right (axes are mirrored)."""
    default = compile_mimic_fullbody_mjmodel(_fullbody_cfg())[0]
    assert_same_model(
        default, compile_mimic_fullbody_mjmodel(_fullbody_cfg("1.0.6"))[0]
    )
    for coupling in _KNEE_COUPLINGS:
        name = f"{coupling}_constraint"
        np.testing.assert_array_equal(
            _eq_data(default, name + "_l"), _eq_data(default, name + "_r"), err_msg=name
        )


def test_1_0_5_has_the_two_negated_left_knee_couplings(installed: str) -> None:
    old = compile_mimic_fullbody_mjmodel(_fullbody_cfg("1.0.5"))[0]
    for coupling in _KNEE_COUPLINGS:
        name = f"{coupling}_constraint"
        left, right = _eq_data(old, name + "_l"), _eq_data(old, name + "_r")
        expected = -right if coupling in _FIXED_COUPLINGS else right
        np.testing.assert_array_equal(left, expected, err_msg=name)


def test_env_var_selects_the_release_without_touching_the_cache(
    installed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rotation3(spec: mujoco.MjSpec) -> float:
        return float(spec.equality("knee_angle_rotation3_constraint_l").data[1])

    before = rotation3(build_mimic_fullbody_spec(_fullbody_cfg())[0])
    monkeypatch.setenv(mv.MODELS_VERSION_ENV_VAR, "1.0.5")
    assert rotation3(build_mimic_fullbody_spec(_fullbody_cfg())[0]) == -0.369499
    monkeypatch.delenv(mv.MODELS_VERSION_ENV_VAR)
    assert (
        rotation3(build_mimic_fullbody_spec(_fullbody_cfg())[0]) == before == 0.369499
    )


def test_explicit_model_path_is_built_as_is(installed: str) -> None:
    from musclemimic_models import get_xml_path

    cfg = _fullbody_cfg("1.0.5" if installed == "1.0.6" else "1.0.6")
    cfg.model_path = str(get_xml_path("myofullbody"))
    spec = build_mimic_fullbody_spec(cfg)[0]
    recorded = mv._MODELS_VALUES[installed]["eq_polycoef"]
    np.testing.assert_array_equal(
        spec.equality("knee_angle_rotation3_constraint_l").data[:5],
        recorded["knee_angle_rotation3_constraint_l"],
    )


@pytest.mark.parametrize("builder", ["bimanual", "myotorso_bimanual"])
def test_arm_and_torso_models_follow_the_release(installed: str, builder: str) -> None:
    build = {
        "bimanual": build_mimic_bimanual_spec,
        "myotorso_bimanual": build_myotorso_bimanual_mimic_spec,
    }[builder]
    for version in mv.SUPPORTED_MODELS_VERSIONS:
        cfg = default_mimic_config()
        cfg.model_version = version
        spec = build(cfg)[0]
        recorded = mv._MODELS_VALUES[version]
        for name, pos in recorded["geom_pos"].items():
            if (geom := spec.geom(name)) is not None:
                np.testing.assert_array_equal(
                    geom.pos, pos, err_msg=f"{version} {name}"
                )
        if (back := spec.geom("back_cylinder_l")) is not None:
            assert back.size[0] == recorded["geom_size0"]["back_cylinder_l"]
        assert spec.geom("DELT1hh_ellipsoid_DELT1") is not None
