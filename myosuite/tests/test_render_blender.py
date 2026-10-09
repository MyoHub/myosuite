# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache 2 license in the repository root.
"""Regression checks for USD rollout timing, identifiers and policy failures."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from scripts import render_blender

pytestmark = pytest.mark.tier1


class _Env(gym.Env):
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string("""
        <mujoco><option timestep="0.002" gravity="0 0 0"/>
          <asset><mesh name="fixture" vertex="0 0 0 .2 0 0 0 .2 0 0 0 .2"/></asset>
          <worldbody><geom name="goal" type="mesh" mesh="fixture" pos="0 2 0"/>
          <geom name="fence" type="mesh" mesh="fixture" pos="0 -2 0"/>
          <geom name="room" type="mesh" mesh="fixture" pos="0 0 3" contype="0" conaffinity="0"/>
          <body><joint name="slide" type="slide" axis="1 0 0"/>
            <geom name="moving-box" type="box" size=".1 .1 .1"/>
            <geom name="bone" type="mesh" mesh="fixture" contype="0" conaffinity="0"/>
            <site name="anchor" pos="0 0 0"/>
          </body><site name="end" pos="0 1 0"/></worldbody>
          <tendon><spatial name="test-path"><site site="anchor"/><site site="end"/></spatial></tendon>
          <actuator><motor joint="slide"/><muscle name="flexor" tendon="test-path" lengthrange="0.9 1.3" force="10"/></actuator>
        </mujoco>""")
        self.data = mujoco.MjData(self.model)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        self.closed = False

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.data.qpos.astype(np.float32), {}

    def step(self, action: np.ndarray) -> tuple:
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=5)
        mujoco.mj_forward(self.model, self.data)
        return self.data.qpos.astype(np.float32), 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True


def _args(output: Path, checkpoint: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        env="test",
        seed=0,
        checkpoint=checkpoint,
        output=output,
        seconds=0.04,
        fps=50,
        resolution=[128, 128],
        preview=True,
        muscles="volumetric",
        muscle_scale=1.0,
        scene="studio",
        samples=16,
    )


def test_export_preserves_metres_timing_motion_and_valid_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 100 Hz episode must not inherit USD's centimetres/60 Hz defaults."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    import myosuite

    env = _Env()
    monkeypatch.setattr(myosuite, "make_env", lambda _id: env)
    meta = render_blender.export_rollout(_args(tmp_path))
    stage = Usd.Stage.Open(meta["usd"])
    reference = np.load(tmp_path / "reference.npz")
    assert env.closed
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetTimeCodesPerSecond() == pytest.approx(100)
    assert stage.GetEndTimeCode() == 4
    assert meta["duration"] == pytest.approx(0.04)
    assert not np.allclose(reference["geom_xpos"][0], reference["geom_xpos"][-1])
    box_id = env.model.geom("moving-box").id
    prim = stage.GetPrimAtPath(f"/World/Mesh_Xform_moving_box_id{box_id}_geom")
    assert prim.IsValid()
    for index in [0, 2, 4]:
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(index)
        np.testing.assert_allclose(
            transform.ExtractTranslation(),
            reference["geom_xpos"][index, env.model.geom("moving-box").id],
            atol=1e-7,
        )
    for name in ["goal", "fence"]:
        geom_id = env.model.geom(name).id
        prim = stage.GetPrimAtPath(f"/World/Mesh_Xform_{name}_id{geom_id}_geom")
        assert prim.IsValid()
        assert UsdGeom.Imageable(prim).ComputeVisibility(0) == "inherited"
        assert f"{name}_id{geom_id}_geom" not in meta["scenery_objects"]
        assert f"{name}_id{geom_id}_geom" in meta["static_objects"]
    room = f"room_id{env.model.geom('room').id}_geom"
    assert meta["scenery_objects"] == [room]
    assert meta["tendon_objects"]
    assert all("test_path" in name for name in meta["tendon_objects"])
    bone_id = env.model.geom("bone").id
    assert meta["bone_objects"] == [f"bone_id{bone_id}_geom"]
    assert meta["muscle_objects"] == ["Muscle_flexor"]
    tube = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Muscles/Muscle_flexor"))
    assert tube.GetPointsAttr().GetNumTimeSamples() == 5
    assert np.load(tmp_path / "muscles.npz")["activation"].shape == (5, 1)


def test_muscle_tube_keeps_its_volume_as_the_path_shortens() -> None:
    """Shortening a path thickens the belly; the enclosed volume is unchanged."""
    profile = render_blender.radius_profile()
    assert profile.max() == pytest.approx(1.0, abs=1e-2)
    assert profile[0] == pytest.approx(render_blender.TENDON_RADIUS_RATIO)
    assert np.argmax(profile) < len(profile) // 2  # belly biased to the origin
    lengths = np.array([0.3, 0.2])
    radii = render_blender.belly_radius(4e-5, lengths, profile)
    assert radii[1] > radii[0]
    np.testing.assert_allclose(
        render_blender.belly_volume(radii, lengths, profile), 4e-5, rtol=1e-9
    )
    for length, radius in zip(lengths, radii):
        centre = np.zeros((len(profile), 3))
        centre[:, 2] = np.linspace(0, length, len(profile))
        points, counts, indices = render_blender.tube_mesh(centre, radius * profile)
        assert counts.sum() == len(indices)
        assert indices.max() == len(points) - 1
        ring = points[:-2].reshape(len(profile), render_blender.RING_POINTS, 3)
        np.testing.assert_allclose(
            np.linalg.norm(ring[..., :2], axis=-1),
            np.broadcast_to(radius * profile[:, None], ring.shape[:2]),
            rtol=1e-6,
        )


def test_resample_path_is_even_in_arc_length() -> None:
    """Wrap points of uneven spacing become evenly spaced centreline samples."""
    path = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.3, 0.0]])
    points, length = render_blender.resample_path(path, 9)
    assert length == pytest.approx(0.4)
    np.testing.assert_allclose(np.linalg.norm(np.diff(points, axis=0), axis=1), 0.05)


def test_checkpoint_symlink_keeps_format_and_rejects_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HF-style blob symlinks retain .zip dispatch and never become random rollouts."""
    pytest.importorskip("pxr")
    import myosuite
    from myosuite.utils import checkpoint_utils

    blob = tmp_path / "blob_without_extension"
    blob.write_bytes(b"fixture")
    checkpoint = tmp_path / "policy.zip"
    checkpoint.symlink_to(blob)
    env = _Env()
    monkeypatch.setattr(myosuite, "make_env", lambda _id: env)
    model = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(2,)), action_space=env.action_space
    )
    monkeypatch.setattr(checkpoint_utils, "load_sb3_model", lambda path: model)
    with pytest.raises(ValueError, match="observation/action shapes"):
        render_blender.export_rollout(_args(tmp_path / "output", checkpoint))
    assert env.closed
    assert not (tmp_path / "output" / "render.json").exists()


@pytest.mark.parametrize("seconds", ["0", "-1", "nan", "inf"])
def test_invalid_duration_rejected_before_export(
    seconds: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-finite durations must not start an unbounded recording."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "render_blender.py",
            "--env",
            "test",
            "--random",
            "--seconds",
            seconds,
            "--output",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        render_blender.main()
    assert exc.value.code == 2
