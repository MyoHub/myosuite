# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Optional refit measurements stay unavailable rather than becoming zero quality errors."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics import retarget as rt

pytestmark = pytest.mark.tier1


def _case(
    feet: tuple[tuple[str, ...], tuple[str, ...]] = ((), ()),
    orientations: bool = False,
    contact: bool = True,
) -> tuple[rt.LandmarkIK, rt.ClipTargets, np.ndarray]:
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body name="segment">
        <joint name="slide" type="slide" axis="1 0 0" range="-1 1"/>
        <geom size=".1"/><site name="tip" pos=".2 0 0"/>
        <body name="left" pos="0 .2 .08"><geom size=".1"/></body>
        <body name="right" pos="0 -.2 .12"><geom size=".1"/></body>
      </body></worldbody></mujoco>
    """)
    lmap = rt.LandmarkMap(
        "positions",
        "1",
        (rt.Landmark("tip", "segment", "tip"),),
        (rt.OrientationTarget("segment", "segment"),) if orientations else (),
        feet,
    )
    ik = rt.LandmarkIK(model, lmap)
    qpos = np.arange(6, dtype=float)[:, None] * 0.01
    pos = np.stack([ik.landmarks(q) for q in qpos])
    quat = np.tile([[[1.0, 0, 0, 0]]], (len(qpos), len(lmap.orientations), 1))
    targets = rt.ClipTargets(
        100.0, pos, quat, qpos[0], np.full((len(qpos), 2), contact)
    )
    return ik, targets, qpos


@pytest.mark.parametrize(
    "feet", [((), ()), (("left",), ()), ((), ("right",)), (("left",), ("right",))]
)
@pytest.mark.parametrize("orientations", [False, True])
@pytest.mark.parametrize("contact", [False, True])
def test_optional_measurements_follow_available_targets(
    feet: tuple[tuple[str, ...], tuple[str, ...]],
    orientations: bool,
    contact: bool,
) -> None:
    ik, targets, qpos = _case(feet, orientations, contact)
    metrics = rt.validate_refit(ik, targets, qpos)
    assert metrics["landmark_max_mm"] == pytest.approx(0.0)
    assert metrics["orientation_mean_deg"] == (0.0 if orientations else None)
    assert metrics["orientation_max_deg"] == (0.0 if orientations else None)
    if any(feet):
        assert metrics["penetration_mm"] == pytest.approx(20.0 if feet[0] else 0.0)
        assert ("penetration" in metrics["flags"]) is bool(feet[0])
    else:
        assert metrics["penetration_mm"] is None
        assert "penetration" not in metrics["flags"]
    if any(feet) and contact:
        assert metrics["sliding_mps"] == pytest.approx(1.0)
        assert "sliding" in metrics["flags"]
        expected_float = 10.0 if all(feet) else 0.0 if feet[0] else 20.0
        assert metrics["contact_float_mm_median"] == pytest.approx(expected_float)
    else:
        for field in ("sliding_mps", "contact_float_mm_median", "contact_float_mm_p95"):
            assert metrics[field] is None
        assert "sliding" not in metrics["flags"]
    json.dumps(metrics, allow_nan=False)


def test_foot_without_collision_geometry_has_no_sole_metrics() -> None:
    ik, targets, qpos = _case((("left",), ()))
    geom = ik.model.body("left").geomadr[0]
    ik.model.geom_contype[geom] = ik.model.geom_conaffinity[geom] = 0
    metrics = rt.validate_refit(ik, targets, qpos)
    assert metrics["penetration_mm"] is None
    assert metrics["contact_float_mm_median"] is None
    assert metrics["contact_float_mm_p95"] is None
    assert metrics["sliding_mps"] == pytest.approx(1.0)


def test_position_only_fit_metadata_and_summaries_remain_json_serializable(
    tmp_path,
) -> None:
    ik, targets, _ = _case()
    fitted, _ = ik.fit(targets.pos, targets.quat, targets.seed)
    metrics = rt.validate_refit(ik, targets, fitted)
    assert metrics["flags"] == []
    assert metrics["landmark_max_mm"] < 1.0
    meta = {"metrics": metrics, "source": "positions", "fit_seconds": 0.01}
    path = tmp_path / "fitted.npz"
    rt.save_refit(path, ik.model, fitted, targets.frequency, meta)
    assert rt.read_refit_meta(path) == meta
    summary = rt.summarize([meta])
    assert summary["stats"]["orientation_mean_deg"] == {
        "count": 0,
        "median": None,
        "p95": None,
        "max": None,
    }
    full_ik, full_targets, full_qpos = _case((("left",), ("right",)), True)
    measured = rt.validate_refit(full_ik, full_targets, full_qpos)
    mixed = rt.summarize([meta, {**meta, "metrics": measured}])
    assert mixed["stats"]["orientation_mean_deg"] == {
        "count": 1,
        "median": 0.0,
        "p95": 0.0,
        "max": 0.0,
    }
    assert mixed["stats"]["sliding_mps"]["median"] == pytest.approx(1.0)
    assert mixed["stats"]["landmark_rmse_mm"]["count"] == 2
    json.dumps(summary, allow_nan=False)
    json.dumps(mixed, allow_nan=False)


def test_free_only_model_has_no_scalar_joint_speed_measurement() -> None:
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body name="point"><freejoint/>
      <geom size=".1"/></body></worldbody></mujoco>
    """)
    ik = rt.LandmarkIK(
        model, rt.LandmarkMap("free", "1", (rt.Landmark("point", "point"),))
    )
    qpos = np.tile(model.qpos0, (6, 1))
    targets = rt.ClipTargets(
        100.0,
        np.stack([ik.landmarks(q) for q in qpos]),
        np.empty((6, 0, 4)),
        qpos[0],
        np.zeros((6, 2), bool),
    )
    metrics = rt.validate_refit(ik, targets, qpos)
    assert metrics["joint_speed_max_rps"] is None
    assert metrics["flags"] == []
    json.dumps(metrics, allow_nan=False)
