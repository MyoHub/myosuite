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
          <body><joint name="slide" type="slide" axis="1 0 0"/>
            <geom name="moving-box" type="box" size=".1 .1 .1"/>
            <site name="anchor" pos="0 0 0"/>
          </body><site name="end" pos="0 1 0"/></worldbody>
          <tendon><spatial name="test-path"><site site="anchor"/><site site="end"/></spatial></tendon>
          <actuator><motor joint="slide"/></actuator>
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
    prim = stage.GetPrimAtPath("/World/Mesh_Xform_moving_box_id2_geom")
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
        assert f"{name}_id{geom_id}_geom" not in meta.get("background_objects", [])
    assert meta["tendon_objects"]
    assert all("test_path" in name for name in meta["tendon_objects"])


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
