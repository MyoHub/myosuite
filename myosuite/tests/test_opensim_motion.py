# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""OpenSim motion files and their mapping onto MyoLeg's free root and joints."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from myosuite import make_env
from myosuite.biomechanics.opensim_motion import (
    GROUND_Y_UP,
    opensim_motion_to_qpos,
    read_opensim_motion,
)
from myosuite.tests.support.joint_couplings import joint_equality_residuals

pytestmark = pytest.mark.tier1

_SUBJECT01 = (
    Path(__file__).parents[2] / "tutorials" / "files" / "3.5" / "subject01_walk1_ik.mot"
)
_PELVIS = ["pelvis_tilt", "pelvis_list", "pelvis_rotation"]
_LABELS = _PELVIS + ["pelvis_tx", "pelvis_ty", "pelvis_tz"]


@pytest.fixture(scope="module")
def myoleg() -> mujoco.MjModel:
    env = make_env("myoLegWalk-v0")
    yield env.unwrapped.model
    env.close()


def _write(path: Path, header: str, labels: list[str], rows: np.ndarray) -> Path:
    body = "\n".join("\t".join(f"{x:.8f}" for x in row) for row in rows)
    path.write_text(f"{header}endheader\n" + "\t".join(labels) + "\n" + body + "\n")
    return path


def test_reads_opensim_and_simm_headers(tmp_path: Path) -> None:
    rows = np.array([[0.0, 1.0, 2.0], [0.01, 3.0, 4.0]])
    sto = _write(
        tmp_path / "a.sto",
        "Coordinates\nversion=1\nnRows=2\nnColumns=3\ninDegrees=no\n",
        ["time", "knee_angle_r", "pelvis_ty"],
        rows,
    )
    motion = read_opensim_motion(sto)
    assert motion.labels == ("knee_angle_r", "pelvis_ty") and not motion.in_degrees
    np.testing.assert_array_equal(motion.time, [0.0, 0.01])
    np.testing.assert_array_equal(motion.column("pelvis_ty"), [2.0, 4.0])
    # A SIMM header has no inDegrees: degrees, unless the caller says otherwise.
    mot = _write(
        tmp_path / "b.mot", "name trial\ndatacolumns 3\n", ["time", "a", "b"], rows
    )
    assert read_opensim_motion(mot).in_degrees
    assert not read_opensim_motion(mot, in_degrees=False).in_degrees
    with pytest.raises(ValueError, match="time"):
        read_opensim_motion(_write(tmp_path / "c.mot", "", ["t", "a", "b"], rows))


def test_pelvis_pose_follows_opensims_ground_pelvis_joint(
    myoleg: mujoco.MjModel, tmp_path: Path
) -> None:
    rng = np.random.default_rng(0)
    angles = rng.uniform(-40, 40, (5, 3))  # degrees
    origin = rng.uniform(-1, 1, (5, 3)) + [0, 1, 0]
    rows = np.column_stack([0.01 * np.arange(5), angles, origin])
    motion = read_opensim_motion(
        _write(tmp_path / "p.mot", "", ["time", *_LABELS], rows)
    )
    clip = opensim_motion_to_qpos(myoleg, motion, ground_quat=GROUND_Y_UP)
    assert clip.dt == pytest.approx(0.01) and set(clip.coordinates) == set(_LABELS)
    ground = Rotation.from_quat(np.roll(GROUND_Y_UP, -1))  # scalar last
    data = mujoco.MjData(myoleg)
    pelvis = myoleg.body("pelvis").id
    for t in range(5):
        data.qpos[:] = clip.qpos[t]
        mujoco.mj_kinematics(myoleg, data)
        # gait2392's ground_pelvis: body-fixed rotations about z, x, y (tilt, list,
        # rotation), then the origin in the ground frame.
        expected = ground * Rotation.from_euler("ZXY", angles[t], degrees=True)
        np.testing.assert_allclose(
            data.xmat[pelvis].reshape(3, 3), expected.as_matrix(), atol=1e-12
        )
        np.testing.assert_allclose(
            data.xpos[pelvis], ground.apply(origin[t]), atol=1e-12
        )


def test_default_ground_keeps_the_models_heading(
    myoleg: mujoco.MjModel, tmp_path: Path
) -> None:
    rows = np.zeros((2, 7))
    rows[1, 0] = 0.01
    motion = read_opensim_motion(
        _write(tmp_path / "z.mot", "", ["time", *_LABELS], rows)
    )
    clip = opensim_motion_to_qpos(myoleg, motion)
    data = mujoco.MjData(myoleg)
    data.qpos[:] = clip.qpos[0]
    mujoco.mj_kinematics(myoleg, data)
    # Zero pelvis angles give MyoLeg's default orientation: OpenSim's x (forward),
    # y (up) and z (right) along the world's -y, z and -x, as tutorial 3.3 maps them.
    expected = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(
        data.xmat[myoleg.body("pelvis").id].reshape(3, 3), expected, atol=1e-3
    )
    np.testing.assert_allclose(clip.qpos[0, 3:7], myoleg.qpos0[3:7], atol=1e-3)


def test_joints_map_by_name_with_knee_signs_and_couplings(
    myoleg: mujoco.MjModel, tmp_path: Path
) -> None:
    labels = [
        "time",
        "hip_flexion_r",
        "knee_angle_r",
        "knee_angle_l",
        "lumbar_extension",
    ]
    rows = np.array(
        [[0.0, 30.0, -60.0, -10.0, -15.0], [0.01, 20.0, -50.0, -5.0, -15.0]]
    )
    motion = read_opensim_motion(_write(tmp_path / "j.mot", "", labels, rows))
    clip = opensim_motion_to_qpos(myoleg, motion, root_body=None)

    def qpos(name):
        return clip.qpos[:, myoleg.jnt_qposadr[myoleg.joint(name).id]]

    np.testing.assert_allclose(qpos("hip_flexion_r"), np.radians([30.0, 20.0]))
    np.testing.assert_allclose(qpos("knee_angle_r"), np.radians([60.0, 50.0]))
    np.testing.assert_allclose(qpos("knee_angle_l"), np.radians([10.0, 5.0]))
    assert clip.unmapped == ("lumbar_extension",)
    # The coupled knee dofs follow the knee angles; the root stays at qpos0.
    assert np.abs(qpos("knee_angle_beta_rotation1_r")).min() > 0.1
    for q in clip.qpos:
        assert np.abs(joint_equality_residuals(myoleg, q)[0]).max() < 1e-9
    np.testing.assert_array_equal(clip.qpos[:, :7], np.tile(myoleg.qpos0[:7], (2, 1)))
    with pytest.raises(ValueError, match="pelvis coordinates missing"):
        opensim_motion_to_qpos(myoleg, motion)
    uneven = np.vstack([rows, rows[1]])
    uneven[2, 0] = 0.05  # a dropped stretch of frames
    motion = read_opensim_motion(_write(tmp_path / "u.mot", "", labels, uneven))
    with pytest.raises(ValueError, match="evenly spaced"):
        opensim_motion_to_qpos(myoleg, motion, root_body=None)


@pytest.mark.skipif(not _SUBJECT01.exists(), reason="tutorial files not present")
def test_subject01_walk_lands_on_its_own_markers(myoleg: mujoco.MjModel) -> None:
    """gait2392 IK of OpenSim's subject01 on MyoLeg: the pelvis matches the ASIS and
    sacral markers, and the heels move with the heel markers."""
    motion = read_opensim_motion(_SUBJECT01)
    clip = opensim_motion_to_qpos(myoleg, motion, ground_quat=GROUND_Y_UP)
    assert clip.dt == pytest.approx(1 / 60) and len(clip.qpos) == 211
    # Resampled to 100 Hz, the frames every 50 ms are the file's own.
    fast = opensim_motion_to_qpos(myoleg, motion, rate=100.0, ground_quat=GROUND_Y_UP)
    assert fast.dt == 0.01 and len(fast.qpos) == 351
    np.testing.assert_allclose(fast.qpos[::5], clip.qpos[::3], atol=1e-9)
    ground = Rotation.from_quat(np.roll(GROUND_Y_UP, -1))

    def marker(name):
        return ground.apply(
            np.stack([motion.column(f"{name}_t{a}") for a in "xyz"], axis=1)
        )

    right, left, sacrum = marker("R.ASIS"), marker("L.ASIS"), marker("V.Sacral")
    heel = marker("R.Heel")
    data = mujoco.MjData(myoleg)
    pelvis, calcn = myoleg.body("pelvis").id, myoleg.body("calcn_r").id
    lateral, heading, offset = [], [], []
    for t, q in enumerate(clip.qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(myoleg, data)
        axes = data.xmat[pelvis].reshape(3, 3)
        across = right[t] - left[t]
        ahead = 0.5 * (right[t] + left[t]) - sacrum[t]
        ahead -= (ahead @ axes[:, 1]) * axes[:, 1]
        lateral.append(
            np.degrees(np.arccos(axes[:, 2] @ across / np.linalg.norm(across)))
        )
        heading.append(
            np.degrees(np.arccos(axes[:, 0] @ ahead / np.linalg.norm(ahead)))
        )
        offset.append(data.xpos[calcn] - heel[t])
    # Measured: 2.8 / 6.2 deg, 2.2 / 3.5 deg, offsets varying by 4-18 mm.
    assert np.mean(lateral) < 4 and np.max(lateral) < 8
    assert np.mean(heading) < 3 and np.max(heading) < 5
    assert np.std(offset, axis=0).max() < 0.025


def test_a_wrong_knee_convention_is_reported(
    myoleg: mujoco.MjModel, tmp_path: Path
) -> None:
    """A Rajagopal-style file (knee flexion positive) read with gait2392's signs."""
    labels = ["time", "knee_angle_r", "knee_angle_l"]
    rows = np.array([[0.0, 60.0, 10.0], [0.01, 50.0, 5.0]])
    motion = read_opensim_motion(_write(tmp_path / "k.mot", "", labels, rows))
    with pytest.warns(UserWarning, match=r"knee_angle_r \(100% of frames\)"):
        opensim_motion_to_qpos(myoleg, motion, root_body=None)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clip = opensim_motion_to_qpos(myoleg, motion, root_body=None, signs={})
    knee = clip.qpos[:, myoleg.jnt_qposadr[myoleg.joint("knee_angle_r").id]]
    np.testing.assert_allclose(knee, np.radians([60.0, 50.0]))
