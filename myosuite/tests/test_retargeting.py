# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""retarget: closed-form recovery, marker files, gaps, site maps and ground placement."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.core.retargeting import retarget
from myosuite.core.trajectory_io import expand_motion_clip_to_model
from myosuite.integrations.capture.markers import _read_markers

pytestmark = pytest.mark.tier1

# Planar arm on a fixed base: upper 0.3 m, lower 0.25 m.
PLANAR = """<mujoco><compiler angle="radian"/><worldbody>
<body name="upper"><joint name="shoulder" axis="0 0 1"/>
  <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/><site name="elbow" pos=".3 0 0"/>
  <body name="lower" pos=".3 0 0"><joint name="elbow" axis="0 0 1" range="0 2.5"/>
    <geom type="capsule" fromto="0 0 0 .25 0 0" size=".02"/><site name="wrist" pos=".25 0 0"/>
  </body>
</body></worldbody></mujoco>"""

# One leg on a free pelvis; the box under the foot is the sole.
LEG = """<mujoco><compiler angle="radian"/><worldbody>
<geom name="floor" type="plane" size="3 3 .1"/>
<body name="pelvis" pos="0 0 1"><freejoint name="root"/><geom type="sphere" size=".08"/>
  <site name="PEL" pos="0 0 .08"/><site name="ASI" pos=".08 .03 0"/><site name="PSI" pos="-.08 .03 0"/>
  <body name="thigh"><joint name="hip" axis="0 1 0" range="-1.5 1.5"/>
    <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".04"/><site name="THI" pos=".05 0 -.2"/>
    <body name="shank" pos="0 0 -.4"><joint name="knee" axis="0 1 0" range="-2.5 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".035"/>
      <site name="KNE" pos=".05 0 0"/><site name="SHA" pos=".05 0 -.2"/>
      <body name="foot" pos="0 0 -.4">
        <geom name="sole" type="box" size=".1 .04 .02" pos=".05 0 -.02"/>
        <site name="ANK" pos="0 .05 0"/><site name="TOE" pos=".15 0 -.04"/>
        <site name="HEEL" pos="-.05 0 -.04"/>
      </body>
    </body>
  </body>
</body></worldbody></mujoco>"""

RATE = 100.0


def _planar_motion(frames: int = 150) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Closed-form angles and the elbow and wrist markers they produce."""
    t = np.arange(frames) / RATE
    q1, q2 = 0.8 * np.sin(np.pi * t), 1.2 + 0.8 * np.sin(2 * np.pi * t)
    elbow = np.stack([0.3 * np.cos(q1), 0.3 * np.sin(q1), np.zeros_like(t)], axis=1)
    wrist = elbow + np.stack(
        [0.25 * np.cos(q1 + q2), 0.25 * np.sin(q1 + q2), 0 * t], axis=1
    )
    return np.stack([q1, q2], axis=1), {"elbow": elbow, "wrist": wrist}


def _leg_motion(
    frames: int = 120,
) -> tuple[mujoco.MjModel, np.ndarray, dict[str, np.ndarray]]:
    """A free-root stride (truth qpos) and every site's trajectory as markers."""
    model = mujoco.MjModel.from_xml_string(LEG)
    t = np.arange(frames) / RATE
    qpos = np.zeros((frames, model.nq))
    qpos[:, 0] = 0.8 * t + 0.3
    qpos[:, 1] = -0.2
    qpos[:, 2] = 0.9 + 0.03 * np.sin(4 * np.pi * t)  # the sole stays above the floor
    pitch = 0.1 * np.sin(2 * np.pi * t)
    qpos[:, 3], qpos[:, 5] = np.cos(pitch / 2), np.sin(pitch / 2)
    qpos[:, 7] = 0.5 * np.sin(2 * np.pi * t)
    qpos[:, 8] = -0.6 - 0.4 * np.sin(2 * np.pi * t)
    data = mujoco.MjData(model)
    markers = {model.site(i).name: np.zeros((frames, 3)) for i in range(model.nsite)}
    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        mujoco.mj_kinematics(model, data)
        for name in markers:
            markers[name][frame] = data.site(name).xpos
    return model, qpos, markers


def _box_bottom(model: mujoco.MjModel, qpos: np.ndarray, geom: str) -> float:
    """Lowest corner of a box geom, computed from its eight corners."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    g = data.geom(geom)
    corners = (
        np.array(np.meshgrid(*[[-1, 1]] * 3)).reshape(3, -1).T * model.geom(geom).size
    )
    return float(np.min(g.xpos[2] + corners @ g.xmat.reshape(3, 3)[2]))


def _write_trc(
    path: Path, markers: dict[str, np.ndarray], rate: float, scale: float = 1000.0
) -> None:
    """OpenSim TRC in millimetres; NaN cells are written empty."""
    names, frames = list(markers), len(next(iter(markers.values())))
    lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{rate:g}\t{rate:g}\t{frames}\t{len(names)}\tmm\t{rate:g}\t1\t{frames}",
        "Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t",
        "\t\t" + "\t".join(f"{a}{i + 1}" for i in range(len(names)) for a in "XYZ"),
        "",
    ]
    for frame in range(frames):
        cells = [str(frame + 1), f"{frame / rate:.6f}"]
        for name in names:
            cells += [
                "" if np.isnan(v) else f"{v * scale:.9f}" for v in markers[name][frame]
            ]
        lines.append("\t".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_recovers_closed_form_angles() -> None:
    model = mujoco.MjModel.from_xml_string(PLANAR)
    angles, markers = _planar_motion()
    clip = retarget(model, markers, fps=RATE)
    np.testing.assert_allclose(clip.qpos, angles, atol=1e-8)
    np.testing.assert_allclose(
        clip.qvel, np.gradient(angles, 1 / RATE, axis=0), atol=1e-6
    )
    assert clip.frequency_hz == RATE and clip.site_names == ["elbow", "wrist"]
    np.testing.assert_allclose(
        clip.site_xpos, np.stack([markers["elbow"], markers["wrist"]], 1), atol=1e-9
    )
    meta = clip.metadata
    assert meta["markers"] == ["elbow", "wrist"] and meta["marker_error"].shape == (
        150,
        2,
    )
    assert np.nanmax(meta["marker_error"]) < 1e-9 and meta["solved"].all()
    assert meta["sole_height"] is None and meta["ground_offset"] == 0.0  # fixed base
    assert expand_motion_clip_to_model(clip, model).metadata is meta


def test_free_root_motion_and_ground_placement() -> None:
    model, truth, markers = _leg_motion()
    kept = retarget(model, markers, fps=RATE, ground_sites=[])
    np.testing.assert_allclose(kept.qpos[:, :3], truth[:, :3], atol=1e-7)
    np.testing.assert_allclose(
        np.abs(kept.qpos[:, 3:7] @ truth[:, 3:7].T).diagonal(), 1, atol=1e-10
    )
    np.testing.assert_allclose(kept.qpos[:, 7:], truth[:, 7:], atol=1e-7)
    soles = np.array([_box_bottom(model, q, "sole") for q in truth])
    np.testing.assert_allclose(kept.metadata["sole_height"], soles, atol=1e-7)
    assert kept.metadata["ground_offset"] == 0.0
    # Default: the lowest sole point of the clip lands on z = 0, one offset for all frames.
    placed = retarget(model, markers, fps=RATE)
    offset = placed.metadata["ground_offset"]
    assert offset == pytest.approx(-soles.min(), abs=1e-7)
    assert placed.metadata["sole_height"].min() == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(placed.qpos[:, 2], kept.qpos[:, 2] + offset)
    np.testing.assert_allclose(placed.qpos[:, 7:], kept.qpos[:, 7:])
    np.testing.assert_allclose(placed.metadata["marker_error"], abs(offset), atol=1e-7)
    np.testing.assert_allclose(
        placed.site_xpos[..., 2], kept.site_xpos[..., 2] + offset
    )
    # Ground sites instead of the geometry.
    heel = retarget(model, markers, fps=RATE, ground_sites=["HEEL"])
    assert heel.site_xpos[:, heel.site_names.index("HEEL"), 2].min() == pytest.approx(
        0, abs=1e-9
    )


def test_fixed_base_is_not_shifted() -> None:
    model = mujoco.MjModel.from_xml_string(PLANAR)
    _, markers = _planar_motion(10)
    with pytest.raises(ValueError, match="free root"):
        retarget(model, markers, fps=RATE, ground_sites=["wrist"])


def test_nan_markers_are_skipped() -> None:
    model = mujoco.MjModel.from_xml_string(PLANAR)
    angles, markers = _planar_motion(60)
    markers["wrist"][10:20] = np.nan  # the elbow marker still fixes the shoulder
    markers["elbow"][30:35] = markers["wrist"][30:35] = np.nan  # nothing seen
    markers["elbow"][:3] = markers["wrist"][:3] = np.nan
    clip = retarget(model, markers, fps=RATE)
    solved = clip.metadata["solved"]
    assert not solved[:3].any() and not solved[30:35].any() and solved[3:30].all()
    np.testing.assert_allclose(
        clip.qpos[:3], clip.qpos[[3, 3, 3]]
    )  # leading gap: first pose
    np.testing.assert_allclose(
        clip.qpos[30:35], clip.qpos[[29] * 5]
    )  # later gaps hold the pose
    np.testing.assert_allclose(clip.qpos[10:20, 0], angles[10:20, 0], atol=1e-8)
    np.testing.assert_allclose(clip.qpos[10:20, 1], angles[9, 1], atol=1e-8)
    seen = np.isfinite(clip.metadata["marker_error"])
    assert not seen[10:20, 1].any() and seen[10:20, 0].all()
    np.testing.assert_allclose(clip.qpos[35:], angles[35:], atol=1e-8)


def test_trc_file_round_trips(tmp_path: Path) -> None:
    model = mujoco.MjModel.from_xml_string(PLANAR)
    _, markers = _planar_motion(40)
    markers["wrist"][5] = np.nan
    path = tmp_path / "trial.trc"
    _write_trc(path, markers, RATE)
    data = _read_markers(path)
    assert data.names == ("elbow", "wrist") and data.rate == RATE
    np.testing.assert_allclose(
        data.positions, np.stack(list(markers.values()), 1), atol=1e-11
    )
    assert np.isnan(data.positions[5, 1]).all()
    from_file = retarget(model, path)
    from_arrays = retarget(model, markers, fps=RATE)
    np.testing.assert_allclose(from_file.qpos, from_arrays.qpos, atol=1e-9)
    assert (
        from_file.metadata["source"] == str(path.resolve())
        and from_file.frequency_hz == RATE
    )
    bad = tmp_path / "bad.trc"
    bad.write_text(path.read_text().replace("\tmm\t", "\tin\t"))
    with pytest.raises(ValueError, match="units"):
        retarget(model, bad)


def test_site_mapping() -> None:
    model = mujoco.MjModel.from_xml_string(PLANAR)
    angles, markers = _planar_motion(20)
    renamed = {"E": markers["elbow"], "W": markers["wrist"], "extra": markers["wrist"]}
    clip = retarget(model, renamed, fps=RATE, sites={"E": "elbow", "W": "wrist"})
    np.testing.assert_allclose(clip.qpos, angles, atol=1e-8)
    assert clip.metadata["marker_sites"] == {"E": "elbow", "W": "wrist"}
    # An explicit entry wins over a marker that shares the site's name.
    both = retarget(
        model, {**markers, "W": markers["wrist"]}, fps=RATE, sites={"W": "wrist"}
    )
    assert both.metadata["marker_sites"] == {"elbow": "elbow", "W": "wrist"}
    with pytest.raises(KeyError):
        retarget(model, renamed, fps=RATE, sites={"missing": "elbow"})
    with pytest.raises(KeyError):
        retarget(model, renamed, fps=RATE, sites={"E": "no_such_site"})
    with pytest.raises(ValueError, match="only one marker"):
        retarget(model, renamed, fps=RATE, sites={"E": "elbow", "W": "elbow"})
    with pytest.raises(ValueError, match="No marker"):
        retarget(model, {"E": markers["elbow"]}, fps=RATE)


def test_lab_frame_is_rotated_and_translated() -> None:
    model = mujoco.MjModel.from_xml_string(PLANAR)
    angles, markers = _planar_motion(20)
    y_up = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # Y-up lab -> Z-up model
    offset = np.array([0.5, -0.2, 1.0])
    lab = {name: (xyz - offset) @ y_up for name, xyz in markers.items()}  # inverse map
    clip = retarget(model, lab, fps=RATE, rotation=y_up, translation=offset)
    np.testing.assert_allclose(clip.qpos, angles, atol=1e-8)
    with pytest.raises(ValueError, match="rotation"):
        retarget(model, markers, fps=RATE, rotation=np.diag([1.0, 1.0, -1.0]))


def test_array_input_is_validated() -> None:
    model = mujoco.MjModel.from_xml_string(PLANAR)
    _, markers = _planar_motion(5)
    with pytest.raises(ValueError, match="fps"):
        retarget(model, markers)
    with pytest.raises(ValueError, match="fps"):
        retarget(model, markers, fps=0)
    with pytest.raises(ValueError, match="shape"):
        retarget(
            model, {"elbow": markers["elbow"], "wrist": markers["wrist"][:3]}, fps=RATE
        )
    with pytest.raises(ValueError, match="No frame"):
        retarget(model, {"elbow": np.full((5, 3), np.nan)}, fps=RATE)


def test_landmark_csv_gives_a_static_pose(tmp_path: Path) -> None:
    model = mujoco.MjModel.from_xml_string(PLANAR)
    angles, markers = _planar_motion(1)
    path = tmp_path / "static.csv"
    rows = ["label,x,y,z"] + [
        n + "," + ",".join(f"{v:.17g}" for v in p[0]) for n, p in markers.items()
    ]
    path.write_text("\n".join(rows) + "\n")
    clip = retarget(model, path)
    np.testing.assert_allclose(clip.qpos, angles, atol=1e-8)
    assert clip.frequency_hz is None and not clip.qvel.any()


def test_c3d_needs_ezc3d(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = mujoco.MjModel.from_xml_string(PLANAR)
    monkeypatch.setitem(sys.modules, "ezc3d", None)
    with pytest.raises(ImportError, match=r"myosuite\[capture\]"):
        retarget(model, tmp_path / "trial.c3d")


def test_c3d_points_are_read_in_metres(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    angles, markers = _planar_motion(30)
    xyz = (
        np.stack([markers["elbow"], markers["wrist"]], axis=1) * 1000.0
    )  # (T, M, 3) mm
    residuals = np.zeros((1, 2, 30))
    residuals[0, 1, 7] = -1.0  # an invalid wrist sample

    def c3d(path: str) -> dict:
        return {
            "parameters": {
                "POINT": {
                    "LABELS": {"value": ["elbow "]},
                    "LABELS2": {"value": ["wrist"]},
                    "UNITS": {"value": ["mm"]},
                    "RATE": {"value": [RATE]},
                }
            },
            "data": {
                "points": np.concatenate([xyz.transpose(2, 1, 0), np.ones((1, 2, 30))]),
                "meta_points": {"residuals": residuals},
            },
        }

    monkeypatch.setitem(sys.modules, "ezc3d", types.SimpleNamespace(c3d=c3d))
    data = _read_markers(tmp_path / "trial.c3d")
    assert data.names == ("elbow", "wrist") and data.rate == RATE
    assert np.isnan(data.positions[7, 1]).all()
    np.testing.assert_allclose(data.positions[:7], xyz[:7] / 1000.0)
    clip = retarget(mujoco.MjModel.from_xml_string(PLANAR), tmp_path / "trial.c3d")
    np.testing.assert_allclose(clip.qpos[8:], angles[8:], atol=1e-8)
