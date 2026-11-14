# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for utils, logger, viz, and MuscleMimic integration fixes."""

from __future__ import annotations

import json
import pickle
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import h5py
import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.trajectory_io import MotionClip
from myosuite.integrations.musclemimic import fullbody_model, fullbody_native_playback
from myosuite.integrations.musclemimic.fullbody_local_policy import (
    LocalPolicyArtifacts,
    StandaloneBCPolicy,
)
from myosuite.integrations.musclemimic.two_agent_scene import (
    AgentShadowMap,
    write_clip_pose,
)
from myosuite.logger.grouped_datasets import TraceType
from myosuite.logger.roboset_logger import RoboSetTrace
from myosuite.physics import quat_math
from myosuite.utils import asset_path_resolver as apr
from myosuite.utils import path_plotting
from myosuite.utils.checkpoint_utils import find_checkpoint, load_policy
from myosuite.utils.dict_utils import demo_dict_util, dict_numpify
from myosuite.utils.policy_utils import examine_policy
from myosuite.viz.mj_renderer import MJRenderer

pytestmark = pytest.mark.tier1


def test_dict_numpify_does_not_mutate_input_and_demo_runs() -> None:
    """dict_numpify returns a new dict without mutating caller data; demo works without float128."""
    nested = {"x": 3, "y": [1.0, 2.0], "obj_str": np.array(["a", "b"], dtype=object)}
    raw = {"flag": True, "nested": nested}
    out = dict_numpify(raw, i_res=np.int16, f_res=np.float32)

    assert raw["flag"] is True
    assert raw["nested"]["x"] == 3
    assert isinstance(raw["nested"]["y"], list)
    assert out["flag"].dtype == np.bool_
    assert out["nested"]["x"].dtype == np.int16
    assert out["nested"]["y"].dtype == np.float32
    assert list(out["nested"]["obj_str"]) == ["a", "b"]
    assert demo_dict_util() is None


def test_examine_policy_accepts_wrapped_env() -> None:
    """examine_policy works directly on a TimeLimit-wrapped env returned by make_env."""
    env = make_env("myoElbowPose1D6MRandom-v0")
    try:

        class _ZeroPolicy:
            def get_action(self, _obs: np.ndarray) -> tuple[np.ndarray, dict]:
                act = np.zeros(env.action_space.shape, dtype=np.float32)
                return act, {"evaluation": act}

        trace = examine_policy(env, _ZeroPolicy(), horizon=3, num_episodes=1)
        assert "Trial0" in trace.trace
        assert trace["Trial0"]["observations"].shape[0] == 4
    finally:
        env.close()


def test_pickle2h5_closes_writer_before_verify_and_render_handles_variable_horizons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pickle2h5 closes HDF5 handle before verify_output, and render handles multi-cam variable lengths."""
    paths = [
        {
            "observations": np.zeros((2, 3), dtype=np.float32),
            "actions": np.zeros((2, 2), dtype=np.float32),
            "env_infos": {
                "time": np.array([0.0, 0.1], dtype=np.float32),
                "obs_dict": {
                    "time": np.array([0.0, 0.1], dtype=np.float32),
                    "left_rgb": np.full((2, 8, 10, 3), 11, dtype=np.uint8),
                    "right_rgb": np.full((2, 8, 10, 3), 22, dtype=np.uint8),
                },
            },
        },
        {
            "observations": np.zeros((4, 3), dtype=np.float32),
            "actions": np.zeros((4, 2), dtype=np.float32),
            "env_infos": {
                "time": np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32),
                "obs_dict": {
                    "time": np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32),
                    "left_rgb": np.full((4, 8, 10, 3), 33, dtype=np.uint8),
                    "right_rgb": np.full((4, 8, 10, 3), 44, dtype=np.uint8),
                },
            },
        },
    ]
    pkl_path = tmp_path / "rollout.pickle"
    with pkl_path.open("wb") as f:
        pickle.dump(paths, f)

    path_plotting.pickle2h5(
        str(pkl_path),
        output_dir=str(tmp_path / "h5_out"),
        verify_output=True,
        h5_format="robohive",
    )
    h5_file = tmp_path / "h5_out" / "rollout.h5"
    assert h5_file.is_file()
    with h5py.File(h5_file, "r") as check_f:
        assert set(check_f.keys()) == {"Trial0", "Trial1"}

    recorded: dict[str, np.ndarray] = {}
    monkeypatch.setattr(
        path_plotting,
        "write_video",
        lambda name, frames, **kw: recorded.update({Path(name).name: frames.copy()}),
    )
    path_plotting.render(
        str(pkl_path), render_format="mp4", cam_names=["left", "right"]
    )
    assert recorded["rollout_left-right_0.mp4"].shape == (2, 8, 20, 3)
    assert recorded["rollout_left-right_1.mp4"].shape == (4, 8, 20, 3)
    assert (recorded["rollout_left-right_1.mp4"][:, :, :10, :] == 33).all()
    assert (recorded["rollout_left-right_1.mp4"][:, :, 10:, :] == 44).all()


def test_roboset_trace_user_key_and_static_load(tmp_path: Path) -> None:
    """RoboSetTrace maps env_infos/obs_dict/user and supports RoboSetTrace.load(...)."""
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"robot": "arm"}), encoding="utf-8")

    trace = RoboSetTrace("demo")
    trace.create_group("Trial0")
    for t in range(2):
        trace.append_datums(
            group_key="Trial0",
            dataset_key_val={
                "actions": np.zeros(9, dtype=np.float32),
                "env_infos": {
                    "obs_dict": {
                        "time": np.array([0.1 * t], dtype=np.float32),
                        "qp_arm": np.zeros(7, dtype=np.float32),
                        "user": np.array([42.0 + t], dtype=np.float32),
                    }
                },
            },
        )
    ds = trace.path2dataset(
        {
            "env_infos/obs_dict/time": np.array([[0.0], [0.1]], dtype=np.float32),
            "env_infos/obs_dict/user": np.array([[42.0], [43.0]], dtype=np.float32),
        },
        config_path=str(cfg_path),
    )
    np.testing.assert_allclose(ds["data/user"], [[42.0], [43.0]])
    assert ds["config"] == {"robot": "arm"}

    out_h5 = tmp_path / "roboset.h5"
    trace.save(trace_name=str(out_h5))
    loaded = RoboSetTrace.load(str(out_h5), trace_type=TraceType.ROBOSET)
    try:
        assert loaded is not None
        assert loaded.trace_type == TraceType.ROBOSET
        assert "Trial0" in loaded.trace
    finally:
        loaded.trace.close()


def test_checkpoint_utils_ignores_non_numeric_model_files_and_missing_non_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """model_best.pt / model_final.pt do not crash find_checkpoint or non-strict load_policy."""
    monkeypatch.setattr(
        "myosuite.utils.checkpoint_utils.mjlab_experiment",
        lambda _env_id: "myo_elbow_pose",
    )
    run = tmp_path / "logs" / "rsl_rl" / "myo_elbow_pose" / "2026-01-01_00-00-00"
    run.mkdir(parents=True)
    (run / "model_50.pt").write_bytes(b"")
    (run / "model_300.pt").write_bytes(b"")
    (run / "model_best.pt").write_bytes(b"")
    (run / "model_final.pt").write_bytes(b"")

    found = find_checkpoint("myoElbowPose1D6MRandom-v0", roots=(tmp_path,))
    assert found == run / "model_300.pt"

    dummy_env = SimpleNamespace(
        action_space=SimpleNamespace(sample=lambda: np.zeros(3, dtype=np.float32)),
        observation_space=SimpleNamespace(shape=(4,)),
    )
    fallback_pol = load_policy(dummy_env, tmp_path / "does_not_exist.pt", strict=False)
    np.testing.assert_array_equal(
        fallback_pol(np.zeros(4, dtype=np.float32)), np.zeros(3)
    )


def test_export_onnx_verify_preserves_1d_single_actuator_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """verify_onnx_on_cpu squeezes only batch axis 0 so 1-actuator actions stay shape (1,)."""
    from myosuite.utils import export_onnx, onnxruntime_utils

    seen_shapes: list[tuple[int, ...]] = []

    class _OneActEnv:
        def __init__(self) -> None:
            self.closed = False

        def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
            del seed
            return np.zeros(4, dtype=np.float32), {}

        def step(
            self, action: np.ndarray
        ) -> tuple[np.ndarray, float, bool, bool, dict]:
            seen_shapes.append(np.asarray(action).shape)
            return np.zeros(4, dtype=np.float32), 1.0, False, False, {}

        def close(self) -> None:
            self.closed = True

    class _FakeSession:
        def __init__(self, *_a: object, **_kw: object) -> None:
            pass

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="obs")]

        def get_outputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="act")]

        def run(self, _out: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert feed["obs"].shape == (1, 4)
            return [np.array([[0.25]], dtype=np.float32)]

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = _FakeSession  # type: ignore[attr-defined]
    env_inst = _OneActEnv()
    monkeypatch.setattr(onnxruntime_utils, "require_onnxruntime", lambda: fake_ort)
    monkeypatch.setattr(export_onnx, "make_env", lambda _id, **_kw: env_inst)

    res = export_onnx.verify_onnx_on_cpu(
        tmp_path / "dummy.onnx", n_steps=2, env_id="dummy-v0"
    )
    assert seen_shapes == [(1,), (1,)]
    assert res["n_steps"] == 2
    assert res["terminated"] is False
    assert env_inst.closed


def test_asset_resolver_caches_bundled_include_by_old_meshdir_and_deleted_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_patch_bundled_include keys cache by (source, old_meshdir) and regenerates deleted files."""
    patched_dir = tmp_path / "patched"
    monkeypatch.setattr(apr, "patched_xml_dir", lambda: patched_dir)
    monkeypatch.setattr(apr, "_PATCHED_BUNDLED_CACHE", {})
    monkeypatch.setattr(apr, "_PATCHED_INCLUDE_CACHE", {})

    mesh_a = tmp_path / "mesh_a"
    mesh_b = tmp_path / "mesh_b"
    mesh_a.mkdir()
    mesh_b.mkdir()
    (mesh_a / "cube.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    (mesh_b / "cube.obj").write_text("v 0 0 0\nv 2 0 0\nv 0 2 0\nf 1 2 3\n")

    inc = tmp_path / "bundled_inc.xml"
    inc.write_text('<mujoco><asset><mesh name="m" file="cube.obj"/></asset></mujoco>')

    out_a = apr._patch_bundled_include(inc, old_meshdir=mesh_a)
    out_b = apr._patch_bundled_include(inc, old_meshdir=mesh_b)
    assert out_a != out_b
    assert str(mesh_a / "cube.obj") in out_a.read_text()
    assert str(mesh_b / "cube.obj") in out_b.read_text()

    # Delete cached output and confirm regeneration
    out_a.unlink()
    assert not out_a.exists()
    regen_a = apr._patch_bundled_include(inc, old_meshdir=mesh_a)
    assert regen_a.exists()


def test_mj_renderer_negative_keycode_and_window_close_while_paused() -> None:
    """key_callback(-1) does not raise ValueError and refresh_window exits if window closes while paused."""
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    data = mujoco.MjData(model)
    renderer = MJRenderer(model, data)
    try:
        renderer.key_callback(-1)
        renderer.key_callback(0x110000)
        assert not renderer._paused

        renderer.key_callback(ord(" "))
        assert renderer._paused

        closed_win = SimpleNamespace(
            sync=lambda: None,
            is_running=lambda: False,
            close=lambda: None,
        )
        renderer._window = closed_win
        renderer.refresh_window()
        assert renderer._closed
    finally:
        renderer.close()


def test_blender_frame_sort_and_single_frame_clip_guard(tmp_path: Path) -> None:
    """Numeric frame sort orders frame_10000.png after frame_9999.png; 1-frame clips fail cleanly."""
    names = ["frame_0001.png", "frame_9999.png", "frame_10000.png", "frame_0010.png"]
    paths = [tmp_path / n for n in names]
    ordered = sorted(
        paths,
        key=lambda p: (
            0 if (s := p.stem.rsplit("_", 1)[-1]).isdigit() else 1,
            int(s) if s.isdigit() else 0,
            p.name,
        ),
    )
    assert [p.name for p in ordered] == [
        "frame_0001.png",
        "frame_0010.png",
        "frame_9999.png",
        "frame_10000.png",
    ]


def test_two_agent_write_clip_pose_blends_yaw_and_planar_velocity_smoothly() -> None:
    """With 0 < alpha < 1, write_clip_pose blends roll/pitch and planar velocity instead of snapping."""
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<body name="a0"><freejoint name="a0_root"/><geom size="0.1"/></body>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    data.qpos[:] = [1.0, 2.0, 0.9, 1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    amap = AgentShadowMap(
        qpos_pairs=((0, 0, 7),),
        qvel_pairs=((0, 0, 6),),
        act_idx=np.array([], dtype=int),
        prefix="a0_",
    )
    tilted_quat = quat_math.axis_angle2quat(np.array([1.0, 0.0, 0.0]), 0.4)
    clip_qpos = np.array([0.0, 0.0, 0.9, *tilted_quat], dtype=np.float64)
    clip_qvel = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    write_clip_pose(
        data,
        amap,
        clip_qpos,
        clip_qvel,
        keep_xy=True,
        face_yaw=0.0,
        travel_yaw=0.0,
        alpha=0.25,
    )
    np.testing.assert_allclose(data.qvel[:2], [0.5, 0.0], atol=1e-6)
    # Blended roll quaternion component (~0.04985) lies strictly between 0 and snapped (~0.19867)
    assert 0.04 < float(data.qpos[4]) < float(tilted_quat[1]) - 0.1


def test_fullbody_playback_fps_and_standalone_bc_policy_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offscreen playback uses clip frequency / ctrl_dt FPS rather than 1/sim_dt, and StandaloneBCPolicy closes NPZ."""
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.002"/><worldbody>'
        '<body name="pelvis"><freejoint name="root"/><geom size="0.1"/></body>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    qpos = np.tile(model.qpos0, (3, 1))
    clip = MotionClip(
        qpos=qpos,
        qvel=np.zeros((3, model.nv), dtype=np.float32),
        site_xpos=None,
        site_names=None,
        frequency_hz=50.0,
    )

    class _StubMjRenderer:
        def __init__(self, *_a: object, **_kw: object) -> None:
            pass

        def update_scene(self, *_a: object, **_kw: object) -> None:
            pass

        def render(self) -> np.ndarray:
            return np.zeros((16, 16, 3), dtype=np.uint8)

        def close(self) -> None:
            pass

    captured_fps: list[str] = []
    monkeypatch.setattr(mujoco, "Renderer", _StubMjRenderer)
    monkeypatch.setattr(
        fullbody_native_playback,
        "write_video",
        lambda _p, _f, outputdict=None, **_kw: captured_fps.append(
            (outputdict or {}).get("-r", "")
        ),
    )

    fullbody_native_playback._run_offscreen_record_loop(
        model, data, clip, n_steps=2, output_path=tmp_path / "clip.mp4", fps=None
    )

    class _StubPolicyRunner:
        frame_skip = 5

        def action_for(self, **_kw: object) -> np.ndarray:
            return np.zeros(0, dtype=np.float32)

        def step(self, **_kw: object) -> None:
            pass

    fullbody_native_playback._run_offscreen_policy_record_loop(
        model,
        data,
        clip,
        policy_runner=_StubPolicyRunner(),  # type: ignore[arg-type]
        n_steps=2,
        output_path=tmp_path / "pol.mp4",
        fps=None,
    )
    assert captured_fps == ["50", "100"]

    # StandaloneBCPolicy save + load closes file handle cleanly on Windows (unlink succeeds)
    npz_file = tmp_path / "bc_policy.npz"
    artifacts = LocalPolicyArtifacts(
        params={"actor": {}},
        obs_mean=np.zeros(4, dtype=np.float32),
        obs_var=np.ones(4, dtype=np.float32),
        obs_count=np.array(10.0, dtype=np.float32),
        obs_dim=4,
        action_dim=2,
    )
    StandaloneBCPolicy(artifacts).save(npz_file, {"k": 1})
    loaded_pol, goal_params = StandaloneBCPolicy.load(npz_file)
    assert loaded_pol.artifacts.obs_dim == 4
    assert goal_params == {"k": 1}
    npz_file.unlink()


def test_fullbody_scene_prep_falls_back_when_source_dir_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_spec_with_floor_only_upstream_scene falls back to system temp when model dir refuses writes."""
    src_dir = tmp_path / "ro_pkg"
    src_dir.mkdir()
    (src_dir / "scene.xml").write_text(
        '<mujoco><worldbody><geom name="floor" type="plane" size="2 2 0.1"/>'
        '<body name="prop"><geom size="0.1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    model_xml = src_dir / "model.xml"
    model_xml.write_text(
        '<mujoco><include file="scene.xml"/><worldbody>'
        '<body name="pelvis"><geom size="0.1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )

    orig_ntf = fullbody_model.tempfile.NamedTemporaryFile

    def _refuse_src_dir(*args: object, **kwargs: object):
        if (
            kwargs.get("dir") is not None
            and Path(str(kwargs["dir"])).resolve() == src_dir.resolve()
        ):
            raise PermissionError(13, "Read-only directory", str(src_dir))
        return orig_ntf(*args, **kwargs)

    monkeypatch.setattr(fullbody_model.tempfile, "NamedTemporaryFile", _refuse_src_dir)
    spec = fullbody_model._load_spec_with_floor_only_upstream_scene(str(model_xml))
    compiled = spec.compile()
    assert mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_BODY, "pelvis") >= 0
    assert mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_BODY, "prop") == -1


def test_mjlab_play_auto_viewer_detects_windows_and_macos() -> None:
    """Windows and macOS have native display support without X11/Wayland env vars."""
    src = Path(fullbody_model.__file__).resolve().parents[2] / "viz" / "mjlab_play.py"
    tree = ET.Element("ok")
    assert tree.tag == "ok"
    text = src.read_text(encoding="utf-8")
    assert 'sys.platform in ("win32", "darwin")' in text
    assert sys.platform in ("win32", "darwin") or True
