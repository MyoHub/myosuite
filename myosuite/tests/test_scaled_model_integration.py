# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MyoSuite's scaling entry points build ``myo_sim.scale_spec``'s model and replay clips.

That ``make_env(..., scales=...)`` simulates this model is checked in
``test_scaled_env.py``; the scaling rules and their OpenSim comparison are
tested in myo_sim (``tests/test_scaling.py``, ``tests/test_scale_groups.py``,
``tests/test_opensim_comparison.py``, ``tests/test_opensim_pose_comparison.py``).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

myo_sim = pytest.importorskip("myo_sim")
pytest.importorskip("myo_sim.scaling", reason="Requires MyoHub/myo_sim#149")

from myosuite.core.body_scaling import scale_model, scale_spec  # noqa: E402
from myosuite.core.model_builder import ModelBuilder  # noqa: E402
from myosuite.core.trajectory_io import (  # noqa: E402
    MotionClip,
    expand_motion_clip_to_model,
)
from myosuite.tests.support.optional_deps import (  # noqa: E402
    build_fullbody_spec,
)
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
def spec() -> mujoco.MjSpec:
    return ModelBuilder.from_xml_file(
        gym.spec(ENV_ID).kwargs["model_path"]
    ).build_spec()


@pytest.fixture(params=["factors", "scale_set"])
def scales(request: pytest.FixtureRequest, tmp_path: Path) -> dict[str, object] | Path:
    return FACTORS if request.param == "factors" else _scale_set(tmp_path / "s.xml")


def test_public_entry_points_give_the_library_model(
    spec: mujoco.MjSpec, scales: dict[str, object] | Path
) -> None:
    expected = myo_sim.scale_spec(spec, scales, total_mass=TOTAL_MASS).compile()
    from_spec = scale_spec(spec, scales, total_mass=TOTAL_MASS).compile()
    model = scale_model(spec, scales, total_mass=TOTAL_MASS)
    digest = myo_sim.model_digest(expected)
    assert myo_sim.model_digest(from_spec) == digest
    assert myo_sim.model_digest(model) == digest
    assert digest != myo_sim.model_digest(spec.copy().compile())
    assert model.body_mass.sum() == pytest.approx(TOTAL_MASS)


def test_named_motion_replays_on_scaled_fullbody() -> None:
    source = build_fullbody_spec()
    nominal = source.copy().compile()
    model = scale_model(source, {"femur_r": 1.08})
    assert (model.nq, model.nv, model.nu, model.ntendon) == (
        nominal.nq,
        nominal.nv,
        nominal.nu,
        nominal.ntendon,
    )
    names = [model.joint(i).name for i in range(model.njnt)]
    assert names == [nominal.joint(i).name for i in range(nominal.njnt)]
    # A clip authored against the scaled target, stored in reversed named-joint order.
    qpos = np.repeat(model.qpos0[None], 3, axis=0)
    hinge = next(
        i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    )
    qpos[:, model.jnt_qposadr[hinge]] += [0, 0.01, 0.02]
    indices: list[int] = []
    for i in reversed(range(model.njnt)):
        start = int(model.jnt_qposadr[i])
        stop = int(model.jnt_qposadr[i + 1]) if i + 1 < model.njnt else model.nq
        indices.extend(range(start, stop))
    clip = MotionClip(
        qpos=qpos[:, indices],
        qvel=None,
        site_xpos=None,
        site_names=None,
        qpos_joint_names=list(reversed(names)),
        frequency_hz=100,
    )
    replay = expand_motion_clip_to_model(clip, model)
    np.testing.assert_allclose(replay.qpos, qpos)
    data = mujoco.MjData(model)
    for frame in replay.qpos:
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        assert np.all(np.isfinite(data.site_xpos))
        assert np.all(np.isfinite(data.actuator_length))
