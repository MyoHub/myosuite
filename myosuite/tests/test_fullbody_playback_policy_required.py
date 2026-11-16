# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The full-body eval CLI plays a policy or fails; it replays the clip only on request."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest

from myosuite.core.playback_contract import PlaybackResult
from myosuite.core.trajectory_io import MotionClip
from myosuite.integrations.musclemimic import fullbody_eval_cli as cli
from myosuite.integrations.musclemimic import fullbody_native_playback as native

pytestmark = pytest.mark.tier2

_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body pos="0 0 1"><freejoint/><geom size="0.1"/></body>
  </worldbody>
</mujoco>
"""


def _clip(model: mujoco.MjModel, n: int = 4) -> MotionClip:
    qpos = np.tile(model.qpos0, (n, 1)).astype(np.float32)
    return MotionClip(
        qpos=qpos,
        qvel=np.zeros((n, model.nv), np.float32),
        site_xpos=np.zeros((n, 0, 3), np.float32),
        site_names=[],
        frequency_hz=100.0,
    )


@pytest.fixture()
def playback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Native playback on a tiny model; records which runner runs."""
    model = mujoco.MjModel.from_xml_string(_XML)
    ckpt = tmp_path / "ckpt"
    (ckpt / "train_state").mkdir(parents=True)
    (ckpt / "config").mkdir()
    (ckpt / "config" / "metadata").write_text("{}")
    ran: list[str] = []

    def runner(name: str):
        def _run(self, request, artifacts) -> PlaybackResult:
            ran.append(name)
            return PlaybackResult(frames_executed=1, mean_tracking_error=None)

        return _run

    def no_orbax(*args: Any, **kwargs: Any):
        raise ImportError("orbax.checkpoint is required for local policy inference.")

    monkeypatch.setattr(native, "_build_model", lambda ref: (model, "tiny.xml"))
    monkeypatch.setattr(native, "resolve_motion_path", lambda ref, **kw: ref)
    monkeypatch.setattr(native, "load_motion_clip", lambda *a, **kw: _clip(model))
    monkeypatch.setattr(native, "expand_motion_clip_to_model", lambda c, m: c)
    monkeypatch.setattr(
        native, "resolve_checkpoint_ref", lambda ref: SimpleNamespace(local_path=ckpt)
    )
    monkeypatch.setattr(native, "describe_checkpoint_dir", lambda path: "")
    monkeypatch.setattr(native, "load_local_policy_artifacts", no_orbax)
    monkeypatch.setattr(native.FullbodyNativePlaybackRunner, "run", runner("replay"))
    monkeypatch.setattr(
        native.FullbodyLocalPolicyPlaybackRunner, "run", runner("policy")
    )
    monkeypatch.delenv("MYOSUITE_MUSCLEMIMIC_DISABLE_LOCAL_POLICY", raising=False)
    return {"ckpt": ckpt, "ran": ran}


def _argv(ckpt: Path, *extra: str) -> list[str]:
    return [
        "--path",
        str(ckpt),
        "--motion_path",
        "clip.npz",
        "--use_mujoco",
        "--record",
        *extra,
    ]


def test_a_policy_that_cannot_load_fails(playback: dict[str, Any]) -> None:
    """Orbax missing (Windows, macOS): non-zero exit, no silent replay."""
    assert cli.main(_argv(playback["ckpt"])) != 0
    assert playback["ran"] == []


def test_a_checkpoint_without_a_policy_fails(playback: dict[str, Any]) -> None:
    for name in ("train_state", "config"):
        for path in sorted((playback["ckpt"] / name).rglob("*"), reverse=True):
            path.unlink()
        (playback["ckpt"] / name).rmdir()
    assert cli.main(_argv(playback["ckpt"])) != 0
    assert playback["ran"] == []


def test_replay_runs_only_on_request(
    playback: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli.main(_argv(playback["ckpt"], "--replay")) == 0
    monkeypatch.setenv("MYOSUITE_MUSCLEMIMIC_DISABLE_LOCAL_POLICY", "1")
    assert cli.main(_argv(playback["ckpt"])) == 0
    assert playback["ran"] == ["replay", "replay"]


def test_onnx_policy_video_plays_at_the_control_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One frame per control step: the video rate is 1 / (timestep * frame_skip)."""
    from myosuite.core import trajectory_io
    from myosuite.integrations.musclemimic import (
        fullbody_local_policy,
        fullbody_model,
    )

    model = mujoco.MjModel.from_xml_string(_XML)

    class _Runner:
        frame_skip = 5

        def __init__(self, **kwargs: Any) -> None:
            pass

        def action_for(self, data, clip, frame_idx) -> np.ndarray:
            return np.zeros(model.nu)

        def step(self, m, d, action) -> None:
            for _ in range(self.frame_skip):
                mujoco.mj_step(m, d)

    class _Renderer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def update_scene(self, *args: Any, **kwargs: Any) -> None:
            pass

        def render(self) -> np.ndarray:
            return np.zeros((4, 4, 3), np.uint8)

        def close(self) -> None:
            pass

    graph_input = SimpleNamespace(
        type=SimpleNamespace(
            tensor_type=SimpleNamespace(
                shape=SimpleNamespace(
                    dim=[SimpleNamespace(dim_value=1), SimpleNamespace(dim_value=3)]
                )
            )
        )
    )
    onnx_stub = SimpleNamespace(
        load=lambda *a, **kw: SimpleNamespace(
            graph=SimpleNamespace(input=[graph_input])
        )
    )
    written: list[dict[str, str]] = []
    monkeypatch.setitem(sys.modules, "onnx", onnx_stub)
    monkeypatch.setattr(
        fullbody_model,
        "compile_mimic_fullbody_mjmodel",
        lambda cfg: (model, None, None),
    )
    monkeypatch.setattr(trajectory_io, "resolve_motion_path", lambda ref: ref)
    monkeypatch.setattr(
        trajectory_io, "load_motion_clip", lambda *a, **kw: _clip(model)
    )
    monkeypatch.setattr(trajectory_io, "expand_motion_clip_to_model", lambda c, m: c)
    monkeypatch.setattr(fullbody_local_policy, "FullbodyObsAdapter", lambda **kw: None)
    monkeypatch.setattr(fullbody_local_policy, "OnnxPolicyRunner", _Runner)
    monkeypatch.setattr(native, "side_tracking_camera", lambda m, c: None)
    monkeypatch.setattr(mujoco, "Renderer", _Renderer)
    monkeypatch.setattr(
        cli, "write_video", lambda path, frames, outputdict: written.append(outputdict)
    )
    onnx_file = tmp_path / "policy.onnx"
    onnx_file.write_bytes(b"")

    code = cli._mjlab_onnx_main(
        onnx_path=str(onnx_file),
        motion_path="clip.npz",
        path_ref=None,
        record=True,
        record_path=str(tmp_path / "out.mp4"),
        n_steps=3,
    )

    assert code == 0
    assert written == [{"-r": "100"}]
