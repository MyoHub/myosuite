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

from myosuite.envs.myo.tasks.mimic.tracking.models import (
    compiled_body,
    kinematic_fingerprint,
    model_cfg,
    model_version,
)
from myosuite.envs.myo.tasks.mimic.tracking.motion import (
    load_motion_lib,
    write_synthetic_clip,
)
from myosuite.biomechanics import retarget as rt
from myosuite.biomechanics import retarget_smplh as rs
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


def test_independent_joint_limits() -> None:
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body><freejoint/><geom size="0.1"/>
      <body><joint name="hinge" range="-30 60"/><geom size="0.1"/>
        <body><joint name="slide" type="slide" range="-0.2 0.3"/>
          <geom size="0.1"/>
          <body><joint name="dependent"/><geom size="0.1"/></body>
        </body>
      </body>
    </body></worldbody><equality>
      <joint joint1="dependent" joint2="hinge"/>
    </equality></mujoco>
    """)
    ik = rt.LandmarkIK(model, rt.LandmarkMap("test", "0", ()))
    joints = [model.joint(name).id for name in ("hinge", "slide")]
    np.testing.assert_array_equal(ik.indep_qadr, model.jnt_qposadr[joints])
    np.testing.assert_allclose(ik.lo[ik.bounded], model.jnt_range[joints, 0])
    np.testing.assert_allclose(ik.hi[ik.bounded], model.jnt_range[joints, 1])


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
    with pytest.raises(ValueError, match="refitted on myofullbody from myo_sim 0.0.0"):
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
    path, root = tr.resolve_clips("kit_train")
    assert root == str(tmp_path) and len(clips(path)) == 972
    from myosuite.envs.myo.tasks.mimic.tracking.config import MotionCfg

    assert (
        MotionCfg().clips == "kit_train"
    )  # the task's default: MuscleMimic's training motions
    with pytest.raises(ValueError, match="only the 'train' list"):
        tr.kit_split("test")


# ------------------------------------------------------------- kinematic fingerprint

SMAP = rs.SMPLH_MAPS["myofullbody"]
ARGS = ("myofullbody", True, 0.02, 1.0)


def _other_version() -> str:
    """A myo_sim version other than the installed one (a release with the same body)."""
    return "0.2.5" if model_version("myofullbody") != "0.2.5" else "0.2.6"


def _fingerprint(edit=None) -> str:
    """Fingerprint of a fresh myofullbody build, after ``edit(spec)``."""
    spec = model_cfg("myofullbody").build_spec(True)
    if edit is not None:
        edit(spec)
    return rt.kinematic_fingerprint(spec.compile(), rs.landmark_bodies(SMAP), SMAP.feet)


def _coupling(spec):
    return next(e for e in spec.equalities if e.type == mujoco.mjtEq.mjEQ_JOINT)


def _set(obj, attr: str, fn) -> None:
    setattr(obj, attr, fn(np.array(getattr(obj, attr), np.float64)))


def test_fingerprint_is_stable_and_kinematic() -> None:
    ref = _fingerprint()
    assert ref == _fingerprint() == kinematic_fingerprint("myofullbody")
    assert len(ref) == 16 and int(ref, 16) >= 0
    kinematic = {
        "joint range": lambda s: _set(s.joint("knee_angle_r"), "range", lambda r: r + [0, 0.01]),
        "body offset": lambda s: _set(s.body("tibia_r"), "pos", lambda x: x + [0, 0.001, 0]),
        "coupling": lambda s: _set(_coupling(s), "data", lambda d: d + np.eye(len(d))[1] * 1e-3),
    }  # fmt: skip
    for what, edit in kinematic.items():
        assert _fingerprint(edit) != ref, what
    dynamic = {
        "muscle": lambda s: _set(s.actuators[0], "gainprm", lambda g: g + np.eye(len(g))[2]),
        "mass": lambda s: setattr(s.body("femur_r"), "mass", s.body("femur_r").mass * 1.5),
    }  # fmt: skip
    for what, edit in dynamic.items():
        assert _fingerprint(edit) == ref, what


def _clip(path, model, walk, version: str, fingerprint: str | None) -> str:
    meta = {"model": "myofullbody", "model_version": version}
    if fingerprint is not None:
        meta["model_fingerprint"] = fingerprint
    rt.save_refit(path, model, walk, 100.0, meta)
    return str(path)


def test_fingerprint_decides_across_versions(
    model, walk, tmp_path, monkeypatch
) -> None:
    import myo_sim

    fitted = model_version("myofullbody")
    fp = kinematic_fingerprint("myofullbody")
    same = _clip(tmp_path / "same.npz", model, walk, fitted, fp)
    differs = _clip(tmp_path / "differs.npz", model, walk, fitted, "0" * 16)
    old = _clip(tmp_path / "old.npz", model, walk, fitted, None)
    assert load_motion_lib(old, "", *ARGS).num_frames > 0  # same version: accepted
    old = _clip(tmp_path / "old2.npz", model, walk, fitted, None)  # (lib is cached)
    monkeypatch.setattr(myo_sim, "__version__", other := _other_version())
    assert load_motion_lib(same, "", *ARGS).num_frames > 0
    with pytest.raises(
        ValueError, match=f"myo_sim {fitted}, but myo_sim {other}"
    ) as err:
        load_motion_lib(differs, "", *ARGS)
    assert (
        f"fingerprint 0000000000000000) differs from the installed one ({fp})"
        in str(err.value)
    )
    assert f"myo_sim=={fitted}" in str(err.value)
    with pytest.raises(
        ValueError,
        match=f"myo_sim {fitted}, but myo_sim {other}.*no kinematic fingerprint",
    ):
        load_motion_lib(old, "", *ARGS)
    with pytest.raises(ValueError, match="refitted on other"):
        rt.check_refit_meta(_clip_of(tmp_path, model, walk), "myofullbody", other, fp)


def _clip_of(tmp_path, model, walk) -> str:
    path = tmp_path / "model.npz"
    rt.save_refit(path, model, walk, 100.0, {"model": "other", "model_version": "1"})
    return str(path)


def test_cache_lookup_finds_a_sibling_version(
    model, walk, tmp_path, monkeypatch
) -> None:
    import myo_sim

    from myosuite.envs.myo.tasks.mimic import tracking as tr

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(tr.KIT_ROOT_ENV, raising=False)
    monkeypatch.delenv(tr.DATASET_ENV, raising=False)
    rel = next(ln for ln in tr.kit_split("train").read_text().splitlines()
               if ln.strip() and not ln.startswith("#"))  # fmt: skip
    fitted = model_version("myofullbody")
    old_root = rt.default_cache_root("myofullbody", fitted)
    _clip(old_root / rel, model, walk[:2], fitted, kinematic_fingerprint("myofullbody"))
    assert tr.kit_root() == old_root
    monkeypatch.setattr(myo_sim, "__version__", other := _other_version())
    assert tr.kit_root() == old_root  # same body under another version
    _clip(old_root / rel, model, walk[:2], fitted, "0" * 16)
    with pytest.raises(tr.MissingMotionData) as err:
        tr.kit_root()
    assert str(rt.default_cache_root("myofullbody", other)) in str(err.value)
    assert str(old_root) in str(err.value)


def test_stamping_keeps_qpos(model, walk, tmp_path) -> None:
    version, fp = model_version("myofullbody"), kinematic_fingerprint("myofullbody")
    path = tmp_path / "c.npz"
    _clip(path, model, walk, version, None)
    with np.load(path) as d:
        before = {k: d[k].tobytes() for k in d.files if k != rt.REFIT_META_KEY}
    assert rt.stamp_fingerprint(path, "myofullbody", version, fp)
    with np.load(path) as d:
        assert {k: d[k].tobytes() for k in d.files if k != rt.REFIT_META_KEY} == before
    meta = rt.read_refit_meta(path)
    assert meta["model_fingerprint"] == fp and meta["model_version"] == version
    assert not rt.stamp_fingerprint(path, "myofullbody", version, fp)  # idempotent
    other = _other_version()
    with pytest.raises(ValueError, match=f"not on the installed myofullbody {other}"):
        rt.stamp_fingerprint(path, "myofullbody", other, fp)
