# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Landmark IK on a model's couplings (``myosuite.biomechanics.retarget``) and the
metadata check of ``myoMimic-v0``'s motion sources.

Targets are made on myofullbody itself (round trip): body origins and orientations of a
synthetic motion projected onto the couplings.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.envs.myo.tasks.mimic.tracking.models import compiled_body, model_version
from myosuite.envs.myo.tasks.mimic.tracking.motion import (
    load_motion_lib,
    write_synthetic_clip,
)
from myosuite.biomechanics import retarget as rt
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

pytestmark = pytest.mark.tier2

BODIES = ("pelvis", "lumbar1", "head", "humerus_l", "ulna_l", "lunate_l", "humerus_r",
          "ulna_r", "lunate_r", "femur_l", "tibia_l", "talus_l", "toes_l", "femur_r",
          "tibia_r", "talus_r", "toes_r")  # fmt: skip
LMAP = rt.LandmarkMap(
    name="test", version="0",
    landmarks=tuple(rt.Landmark(b, b) for b in BODIES),
    orientations=tuple(rt.OrientationTarget(b, b) for b in
                       ("pelvis", "torso", "lunate_l", "lunate_r", "calcn_l", "calcn_r")),
    feet=(("calcn_l", "toes_l"), ("calcn_r", "toes_r")),
)  # fmt: skip


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return compiled_body("myofullbody")


@pytest.fixture(scope="module")
def walk(model: mujoco.MjModel, tmp_path_factory: pytest.TempPathFactory) -> np.ndarray:
    """0.6 s of synthetic motion (every independent joint oscillating), on the couplings."""
    path = tmp_path_factory.mktemp("src") / "walk.npz"
    write_synthetic_clip(path, model, 0.6, 1.0, 0.4, 3)
    q = np.load(path)["qpos"]
    project_joint_couplings(JointCouplings.from_model(model), q)
    return q


def _targets(m: mujoco.MjModel, qpos: np.ndarray) -> rt.ClipTargets:
    d = mujoco.MjData(m)
    ids = [m.body(b).id for b in BODIES]
    oid = [m.body(o.body).id for o in LMAP.orientations]
    pos, quat = np.empty((len(qpos), len(ids), 3)), np.empty((len(qpos), len(oid), 4))
    for t, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        pos[t], quat[t] = d.xpos[ids], d.xquat[oid]
    contact = np.zeros((len(qpos), 2), bool)
    return rt.ClipTargets(100.0, pos, quat, qpos[0], contact)


def test_round_trip_on_the_model(model, walk) -> None:
    tg = _targets(model, walk)
    # without smoothing the fit is exact; the default smoothing costs little
    exact = rt.LandmarkIK(model, LMAP, rt.RefitConfig(smooth_weight=0.0))
    met = rt.validate_refit(exact, tg, exact.fit(tg.pos, tg.quat, tg.seed)[0])
    assert met["landmark_max_mm"] < 0.01
    assert met["coupling_residual"] == 0.0
    assert met["limit_violations"] == 0
    ik = rt.LandmarkIK(model, LMAP)
    met = rt.validate_refit(ik, tg, ik.fit(tg.pos, tg.quat, tg.seed)[0])
    assert met["landmark_rmse_mm"] < 2.0


def test_posture_prior_holds_an_open_joint(model, walk) -> None:
    # without hand targets the wrist deviation is left to the prior
    lmap = rt.LandmarkMap(
        "test", "0", tuple(lm for lm in LMAP.landmarks if lm.body != "lunate_l"),
        tuple(o for o in LMAP.orientations if o.body != "lunate_l"), LMAP.feet,
        posture=(("deviation_l", 0.1, 0.05),),
    )  # fmt: skip
    keep = [i for i, b in enumerate(BODIES) if b != "lunate_l"]
    tg = _targets(model, walk[:5])
    ik = rt.LandmarkIK(model, lmap)
    q, _ = ik.fit(tg.pos[:, keep], tg.quat[:, [0, 1, 3, 4, 5]], tg.seed)
    adr = model.jnt_qposadr[model.joint("deviation_l").id]
    np.testing.assert_allclose(q[:, adr], 0.1, atol=1e-3)


def test_per_frame_posture_and_axial_weight(model, walk) -> None:
    lmap = rt.LandmarkMap(
        "test", "0", tuple(lm for lm in LMAP.landmarks if lm.body != "lunate_l"),
        tuple(o for o in LMAP.orientations if o.body != "lunate_l"), LMAP.feet,
        posture=(("deviation_l", 0.1, 0.05),),
    )  # fmt: skip
    keep = [i for i, b in enumerate(BODIES) if b != "lunate_l"]
    tg = _targets(model, walk[:5])
    ik = rt.LandmarkIK(model, lmap, rt.RefitConfig(smooth_weight=0.0))
    ref = np.linspace(-0.1, 0.1, 5)[:, None]
    q, _ = ik.fit(tg.pos[:, keep], tg.quat[:, [0, 1, 3, 4, 5]], tg.seed, ref)
    adr = model.jnt_qposadr[model.joint("deviation_l").id]
    np.testing.assert_allclose(q[:, adr], ref[:, 0], atol=1e-3)
    # an orientation target's axial component weighted by 0: rotating the target about
    # the axis costs nothing
    axial = rt.LandmarkMap(
        "test", "0", (), (rt.OrientationTarget("torso", "torso", 1.0, (0.0, 0.0, 1.0), 0.0),)
    )  # fmt: skip
    ik = rt.LandmarkIK(model, axial)
    d = mujoco.MjData(model)
    d.qpos[:] = walk[0]
    mujoco.mj_kinematics(model, d)
    turn, target = np.empty(4), np.empty(4)
    mujoco.mju_axisAngle2Quat(
        turn, d.xmat[model.body("torso").id].reshape(3, 3)[:, 2], 0.3
    )
    mujoco.mju_mulQuat(target, turn, d.xquat[model.body("torso").id])
    r, _ = ik._evaluate(walk[0], np.zeros((0, 3)), target[None], None, False)
    np.testing.assert_allclose(r, 0.0, atol=1e-9)


def test_elbow_angle_table(model) -> None:
    from myosuite.biomechanics import retarget_smplh as rs

    q = model.qpos0.copy()
    q[model.jnt_qposadr[model.joint("shoulder_elv_l").id]] = np.pi / 2
    hinge = next(h for h in rs.SMPLH_MYOFULLBODY.hinges if h.joint == "elbow_flexion_l")
    phi, theta = rs._hinge_table(model, mujoco.MjData(model), q, hinge)
    # myofullbody's carrying angle at zero flexion; the angle then follows the hinge
    assert np.degrees(theta[0]) == pytest.approx(12.8, abs=0.2)
    assert np.all(np.diff(theta) > 0) and abs(theta[-1] - phi[-1]) < np.radians(5)


def test_validation_flags_a_spike(model, walk) -> None:
    tg = _targets(model, walk)
    ik = rt.LandmarkIK(model, LMAP)
    bad = walk.copy()
    bad[30, model.jnt_qposadr[model.joint("knee_angle_r").id]] += 0.05
    assert "velocity_spikes" in rt.validate_refit(ik, tg, ik.project(bad))["flags"]


def test_motion_sources_check_metadata(model, walk, tmp_path) -> None:
    meta = {"model": "myofullbody", "model_version": model_version("myofullbody")}
    good, other = tmp_path / "good", tmp_path / "other"
    rt.save_refit(good / "c.npz", model, walk, 100.0, meta)
    rt.save_refit(
        other / "c.npz", model, walk, 100.0, {**meta, "model_version": "0.0.0"}
    )
    raw = tmp_path / "raw.npz"
    with np.load(good / "c.npz") as d:
        np.savez(raw, **{k: d[k] for k in d.files if k != rt.REFIT_META_KEY})
    args = ("myofullbody", True, 0.02, 1.0)
    assert load_motion_lib(str(good), "", *args).num_frames > 0
    with pytest.raises(ValueError, match="refitted on myofullbody 0.0.0"):
        load_motion_lib(str(other), "", *args)
    with pytest.raises(ValueError, match="not a refitted clip"):
        load_motion_lib(str(raw), "", *args)


def test_kit_lists_and_root_override(monkeypatch, tmp_path) -> None:
    from myosuite.envs.myo.tasks.mimic import tracking as tr

    def clips(path: str) -> list[str]:
        lines = open(path).read().splitlines()  # noqa: SIM115
        return [ln.split("/KIT/")[-1].removeprefix("KIT/") for ln in lines
                if ln.strip() and not ln.startswith("#")]  # fmt: skip

    monkeypatch.setenv(tr.KIT_ROOT_ENV, str(tmp_path))
    for name, n in (("kit_all", 1080), ("kit_train", 972), ("kit_test", 108)):
        path, root = tr.resolve_clips(name)
        assert root == str(tmp_path) and len(clips(path)) == n
