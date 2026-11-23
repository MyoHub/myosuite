# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MyoSuite's scaling entry points build exactly the model ``myo_sim.scale_spec`` makes.

That ``make_env(..., scales=...)`` simulates this model is checked in
``test_scaled_env.py``; the scaling rules and their OpenSim comparison are
tested in myo_sim (``tests/test_scaling.py``, ``tests/test_opensim_comparison.py``,
``tests/test_opensim_pose_comparison.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

myo_sim = pytest.importorskip("myo_sim")
pytest.importorskip("myo_sim.scaling", reason="Requires MyoHub/myo_sim#149")

from myosuite.core.body_scaling import scale_model  # noqa: E402
from myosuite.core.model_builder import ModelBuilder  # noqa: E402
from myosuite.utils import gym  # noqa: E402

pytestmark = pytest.mark.tier1

ENV_ID = "myoLegWalk-v0"
FACTORS = {"femur_r": [1.02, 1.07, 0.98], "tibia_l": 1.04}
SEGMENTS = {"femur_r": (1.02, 1.07, 0.98), "tibia_l": (1.04, 1.04, 1.04)}
TOTAL_MASS = 70.0


def _scale_set(path: Path) -> Path:
    rows = "".join(
        f"<Scale><scales>{' '.join(map(repr, values))}</scales>"
        f"<segment>{name}</segment><apply>true</apply></Scale>"
        for name, values in SEGMENTS.items()
    )
    path.write_text(
        "<OpenSimDocument Version='40000'><ScaleSet><objects>"
        f"{rows}</objects></ScaleSet></OpenSimDocument>"
    )
    return path


@pytest.fixture(scope="module")
def spec():
    return ModelBuilder.from_xml_file(
        gym.spec(ENV_ID).kwargs["model_path"]
    ).build_spec()


@pytest.fixture(params=["factors", "scale_set"])
def scales(request: pytest.FixtureRequest, tmp_path: Path):
    return FACTORS if request.param == "factors" else _scale_set(tmp_path / "s.xml")


def test_public_entry_points_give_the_library_model(spec, scales) -> None:
    expected = myo_sim.scale_spec(spec, scales, total_mass=TOTAL_MASS).compile()
    model = scale_model(spec, scales, total_mass=TOTAL_MASS)
    assert myo_sim.model_digest(model) == myo_sim.model_digest(expected)
    assert myo_sim.model_digest(model) != myo_sim.model_digest(spec.copy().compile())
    assert model.body_mass.sum() == pytest.approx(TOTAL_MASS)
