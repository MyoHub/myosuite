# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Blender renderer: USD rollout timing, identifiers, policy failures, muscle tubes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from myosuite.viz import blender_render, muscle_tubes
from myosuite.viz.blender_render import RenderConfig

pytestmark = pytest.mark.tier1


class _Env(gym.Env):
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string("""
        <mujoco><option timestep="0.002" gravity="0 0 0"/>
          <asset><mesh name="fixture" vertex="0 0 0 .2 0 0 0 .2 0 0 0 .2"/></asset>
          <worldbody><geom name="goal" type="mesh" mesh="fixture" pos="0 2 0"/>
          <geom name="fence" type="mesh" mesh="fixture" pos="0 -2 0"/>
          <geom name="room" type="mesh" mesh="fixture" pos="0 0 3" contype="0" conaffinity="0"/>
          <body name="carriage"><joint name="slide" type="slide" axis="1 0 0"/>
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


def _config(output: Path, checkpoint: Path | None = None) -> RenderConfig:
    return RenderConfig(
        env="test",
        output=output,
        checkpoint=checkpoint,
        seconds=0.04,
        fps=50,
        resolution=(128, 128),
        preview=True,
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
    meta = blender_render.export_rollout(_config(tmp_path))
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


def test_path_mode_colours_mujoco_segments_by_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thin paths keep MuJoCo's segments and still record each muscle's activation."""
    pytest.importorskip("pxr")
    from dataclasses import replace

    from pxr import Usd

    import myosuite

    env = _Env()
    monkeypatch.setattr(myosuite, "make_env", lambda _id: env)
    meta = blender_render.export_rollout(replace(_config(tmp_path), muscles="paths"))
    tendon = env.model.tendon("test-path").id
    assert meta["muscle_objects"] == [f"_id{tendon}_tendon"]
    assert meta["replaced_tendons"] == []
    assert np.load(tmp_path / "muscles.npz")["activation"].shape == (5, 1)
    assert not Usd.Stage.Open(meta["usd"]).GetPrimAtPath("/World/Muscles").IsValid()


def test_skin_is_exported_as_an_animated_mesh_that_follows_its_bone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .skn bound to the moving body becomes one animated, UV-mapped USD mesh."""
    pytest.importorskip("pxr")
    import struct
    from dataclasses import replace

    from pxr import Usd, UsdGeom

    import myosuite

    vert = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], np.float32)
    skn = tmp_path / "skin.skn"
    skn.write_bytes(
        struct.pack("4i", 3, 3, 1, 1)
        + vert.tobytes()
        + np.array([[0, 0], [1, 0.25], [0.5, 1]], np.float32).tobytes()
        + np.arange(3, dtype=np.int32).tobytes()
        + b"carriage".ljust(40, b"\0")
        + np.array([0, 0, 0, 1, 0, 0, 0], np.float32).tobytes()
        + struct.pack("i", 3)
        + np.arange(3, dtype=np.int32).tobytes()
        + np.ones(3, np.float32).tobytes()
    )
    texture = tmp_path / "Tone.PNG"
    texture.write_bytes(b"not decoded at export")
    env = _Env()
    monkeypatch.setattr(myosuite, "make_env", lambda _id: env)
    out = tmp_path / "out"
    meta = blender_render.export_rollout(
        replace(_config(out), skin=skn, skin_style="opaque", skin_texture=texture)
    )
    stage = Usd.Stage.Open(meta["usd"])
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Skin/Skin"))
    assert meta["skin_objects"] == ["Skin"] and meta["skin_style"] == "opaque"
    points = mesh.GetPointsAttr()
    assert points.GetNumTimeSamples() == 5
    qpos = np.load(out / "reference.npz")["qpos"][:, 0]
    for frame in (0, 4):
        np.testing.assert_allclose(
            np.array(points.Get(frame)), vert + [qpos[frame], 0, 0], atol=1e-6
        )
    # UVs with V up (the .skn stores it down the image); the texture travels with the scene.
    np.testing.assert_allclose(
        np.array(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()),
        [[0, 1], [1, 0.75], [0.5, 0]],
    )
    assert meta["skin_texture"] == "skin_texture.png"
    assert (out / "skin_texture.png").read_bytes() == texture.read_bytes()
    with pytest.raises(ValueError, match="needs a skin"):
        RenderConfig(env="x", output=Path("o"), skin_texture=texture)
    with pytest.raises(ValueError, match="not found"):
        RenderConfig(
            env="x", output=Path("o"), skin="fullbody", skin_texture=tmp_path / "no.png"
        )
    with pytest.raises(ValueError, match="skin style"):
        RenderConfig(env="x", output=Path("o"), skin_style="glass")
    with pytest.raises(ValueError, match="alpha"):
        RenderConfig(env="x", output=Path("o"), skin_alpha=0)


def test_muscle_colours_follow_the_mujoco_viewer_on_paths() -> None:
    """Paths blend near-black to red with activation ** 0.25; uniform is validated."""
    colour = blender_render._muscle_colour
    np.testing.assert_allclose(colour(0.0, paths=True), blender_render.PATH_RELAXED)
    np.testing.assert_allclose(colour(1.0, paths=True), blender_render.PATH_ACTIVE)
    halfway = (np.add(blender_render.PATH_RELAXED, blender_render.PATH_ACTIVE)) / 2
    np.testing.assert_allclose(colour(0.5**4, paths=True), halfway)
    np.testing.assert_allclose(colour(1.0, paths=False), blender_render.MUSCLE_ACTIVE)
    RenderConfig(env="x", output=Path("o"), muscle_color="uniform")
    with pytest.raises(ValueError, match="colour"):
        RenderConfig(env="x", output=Path("o"), muscle_color="rainbow")


def test_muscle_tube_keeps_its_volume_as_the_path_shortens() -> None:
    """Shortening a path thickens the belly; the enclosed volume is unchanged."""
    profile = muscle_tubes.radius_profile(0.5)
    assert profile.max() == pytest.approx(1.0, abs=1e-2)
    np.testing.assert_allclose(profile[[0, -1]], muscle_tubes.TENDON_RADIUS_RATIO)
    assert np.argmax(profile) < len(profile) // 2  # belly towards the origin
    lengths = np.array([0.3, 0.2])
    radii = muscle_tubes.belly_radius(4e-5, lengths, profile)
    assert radii[1] > radii[0]
    np.testing.assert_allclose(
        muscle_tubes.belly_volume(radii, lengths, profile), 4e-5, rtol=1e-9
    )
    for length, radius in zip(lengths, radii):
        centre = np.zeros((len(profile), 3))
        centre[:, 2] = np.linspace(0, length, len(profile))
        points, counts, indices = muscle_tubes.tube_mesh(centre, radius * profile)
        assert counts.sum() == len(indices)
        assert indices.max() == len(points) - 1
        ring = points[:-2].reshape(len(profile), muscle_tubes.RING_POINTS, 3)
        np.testing.assert_allclose(
            np.linalg.norm(ring[..., :2], axis=-1),
            np.broadcast_to(radius * profile[:, None], ring.shape[:2]),
            rtol=1e-6,
        )


def test_forearm_muscle_volumes_are_anatomical() -> None:
    """Forearm bellies stay near measured volumes and leave long distal tendons.

    References: Holzbaur et al. 2007, J Biomech 40:742 (adult upper-limb muscle
    volumes; FDS split evenly over its four compartments).
    """
    from myosuite import make_env

    env = make_env("myoHandPoseRandom-v0")
    try:
        env.reset(seed=0)
        model, data = env.unwrapped.model, env.unwrapped.data
        muscles = muscle_tubes.muscle_actuators(model)
        path = data.ten_length[model.actuator_trnid[muscles, 0]]
        peak, fraction = muscle_tubes.muscle_shape(model, path)
        volume = muscle_tubes.belly_volume(
            peak, path, muscle_tubes.radius_profile(fraction)
        )
    finally:
        env.close()
    names = [model.actuator(a).name for a in muscles]
    for name, measured in {"FCR_r": 17e-6, "FCU_r": 26e-6, "FDS3_r": 14e-6}.items():
        assert 0.5 < volume[names.index(name)] / measured < 2
    flexor = names.index("FDS3_r")
    belly_end = (1 - fraction[flexor]) / 4 + fraction[flexor]
    assert belly_end < 0.7  # the belly ends in the forearm, a tendon runs to the finger


def test_large_muscles_take_measured_volumes() -> None:
    """Hip, thigh, calf and shoulder muscles approach their reference volumes.

    Broad muscles on short paths (vasti, gluteals, adductor magnus) stay under
    the slenderness cap, so they reach at least 45% of the reference.
    """
    from myosuite import make_env

    env = make_env("myoMimicFullbody-v0")
    try:
        env.reset(seed=0)
        model, data = env.unwrapped.model, env.unwrapped.data
        muscles = muscle_tubes.muscle_actuators(model)
        path = data.ten_length[model.actuator_trnid[muscles, 0]]
        peak, fraction = muscle_tubes.muscle_shape(model, path)
        volume = muscle_tubes.belly_volume(
            peak, path, muscle_tubes.radius_profile(fraction)
        )
    finally:
        env.close()
    groups = [muscle_tubes._reference_group(model.actuator(a).name) for a in muscles]
    for name, reference in muscle_tubes.REFERENCE_VOLUMES.items():
        side = "_r" if name.islower() else ""
        total = volume[[g == (name, side) for g in groups]].sum()
        assert 0.45 < total / reference <= 1.0 + 1e-6, name


def test_resample_path_is_even_in_arc_length() -> None:
    """Wrap points of uneven spacing become evenly spaced centreline samples."""
    path = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.3, 0.0]])
    points, length = muscle_tubes.resample_path(path, 9)
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
        blender_render.export_rollout(_config(tmp_path / "output", checkpoint))
    assert env.closed
    assert not (tmp_path / "output" / "render.json").exists()


def test_muscle_mesh_option_checks_env_and_downloads_on_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atlas meshes fit the full body only and come from the Hugging Face dataset."""
    import huggingface_hub

    with pytest.raises(ValueError, match="myoMimicFullbody"):
        RenderConfig(env="myoLegWalk-v0", output=tmp_path, muscle_mesh="atlas")
    with pytest.raises(ValueError, match="not found"):
        RenderConfig(env="test", output=tmp_path, muscle_mesh=str(tmp_path / "x.glb"))
    RenderConfig(env="myoMimicFullbody-v0", output=tmp_path, muscle_mesh="atlas-hd")
    calls = []

    def download(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(tmp_path / "cached.glb")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert blender_render._resolve_muscle_mesh("atlas-hd") == tmp_path / "cached.glb"
    assert calls == [
        {
            "repo_id": "myohub/myosuite-assets",
            "filename": "muscles/fullbody_muscles.glb",
            "repo_type": "dataset",
        }
    ]
    assert blender_render._resolve_muscle_mesh("own.glb") == Path("own.glb")

    def offline(**kwargs: str) -> str:
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(RuntimeError, match="myohub/myosuite-assets"):
        blender_render._resolve_muscle_mesh("atlas")


def test_muscle_weights_blend_across_a_joint_but_never_across_limbs() -> None:
    """Thigh muscles blend femur and tibia; a hand piece by the thigh keeps its hand."""
    z = np.linspace(0, 0.4, 20)
    bone_points = np.concatenate(
        [
            np.c_[np.zeros(20), np.zeros(20), z],  # femur
            np.c_[np.zeros(20), np.zeros(20), -z],  # tibia
            np.c_[np.full(20, 0.06), np.zeros(20), 0.15 + z / 8],  # hand
            np.c_[np.full(20, -0.06), np.zeros(20), 0.5 + z / 4],  # pelvis
        ]
    )
    owner = np.repeat(np.arange(4), 20)
    limbs = [
        blender_render._limb(n) for n in ("femur_r", "tibia_r", "2proxph_r", "pelvis")
    ]
    assert limbs == ["leg_r", "leg_r", "arm_r", "trunk"]
    thigh = np.c_[np.full(9, 0.02), np.zeros(9), np.linspace(-0.1, 0.3, 9)]
    hand = np.c_[np.full(5, 0.04), np.zeros(5), np.linspace(0.16, 0.2, 5)]
    hip = np.c_[np.linspace(-0.05, 0.05, 5), np.zeros(5), np.full(5, 0.45)]
    points = np.concatenate([thigh, hand, hip])
    edges = [(i, i + 1) for i in [*range(8), *range(9, 13), *range(14, 18)]]
    vertex, bone, weight = blender_render._muscle_weights(
        points, edges, bone_points, owner, limbs
    )
    weights = np.zeros((len(points), 4))
    np.add.at(weights, (vertex, bone), weight)
    np.testing.assert_allclose(weights.sum(1), 1)
    assert 0.3 < weights[2, 0] < 0.7  # at the knee: femur and tibia
    assert not weights[:9, 2].any()  # the thigh never follows the hand
    assert not weights[9:14, :2].any()  # the hand never follows the leg
    assert (
        weights[14:, 3].min() > 0.5 and weights[14:, 0].max() > 0
    )  # hip: pelvis + femur


def test_export_records_bone_rest_poses_for_a_muscle_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Muscle meshes bind at qpos0, whatever pose the episode starts in."""
    pytest.importorskip("pxr")
    import myosuite

    env = _Env()
    monkeypatch.setattr(myosuite, "make_env", lambda _id: env)
    glb = tmp_path / "muscles.glb"
    glb.write_bytes(b"glTF")
    config = RenderConfig(
        **{**vars(_config(tmp_path / "out")), "muscle_mesh": str(glb)}
    )
    meta = blender_render.export_rollout(config)
    assert (tmp_path / "out" / meta["muscle_mesh"]).read_bytes() == b"glTF"
    data = mujoco.MjData(env.model)
    mujoco.mj_forward(env.model, data)
    bone = env.model.geom("bone").id
    rest = np.asarray(meta["bone_rest"][f"bone_id{bone}_geom"]).reshape(4, 4)
    np.testing.assert_allclose(rest[:3, 3], data.geom_xpos[bone])
    np.testing.assert_allclose(rest[:3, :3], data.geom_xmat[bone].reshape(3, 3))


@pytest.mark.parametrize("seconds", ["0", "-1", "nan", "inf"])
def test_invalid_duration_rejected_before_export(seconds: str, tmp_path: Path) -> None:
    """Non-finite durations must not start an unbounded recording."""
    argv = [
        "--env",
        "test",
        "--random",
        "--seconds",
        seconds,
        "--output",
        str(tmp_path),
    ]
    with pytest.raises(SystemExit) as exc:
        blender_render.main(argv)
    assert exc.value.code == 2


def test_skin_bones_bind_under_either_phalanx_naming() -> None:
    """``midph2_r`` (bundled skin) binds to ``2midph_r`` (musclemimic_models) and back; unknown bones raise."""
    from myosuite.viz.skin import Skin, SkinPose

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<body name="pelvis"><geom size="0.1"/>'
        '<body name="2midph_r"><geom size="0.01"/></body>'
        '<body name="distph3_l"><geom size="0.01"/></body>'
        "</body></worldbody></mujoco>"
    )

    def skin(bones: list[str]) -> Skin:
        n = len(bones)
        return Skin(
            vert=np.zeros((1, 3), np.float32),
            texcoord=np.zeros((0, 2), np.float32),
            face=np.zeros((0, 3), np.int32),
            bone_names=bones,
            bindpos=np.zeros((n, 3), np.float32),
            bindquat=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
            vertid=[np.zeros(1, np.int32)] * n,
            vertweight=[np.ones(1, np.float32)] * n,
        )

    pose = SkinPose.bind(skin(["pelvis", "midph2_r", "3distph_l"]), model)
    body = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)  # noqa: E731
    assert list(pose.body_ids) == [body("pelvis"), body("2midph_r"), body("distph3_l")]
    with pytest.raises(ValueError, match="midph9_r"):
        SkinPose.bind(skin(["pelvis", "midph9_r"]), model)


def test_cli_defaults_to_the_newest_checkpoint_and_never_to_random(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --checkpoint the CLI renders find_checkpoint's policy; with none found it stops (no random fallback)."""
    import myosuite.utils.checkpoint_utils as checkpoint_utils

    rendered: list[RenderConfig] = []
    monkeypatch.setattr(
        blender_render, "render_rollout", lambda config, **kw: rendered.append(config)
    )
    found = tmp_path / "logs" / "model_3000.pt"
    monkeypatch.setattr(checkpoint_utils, "find_checkpoint", lambda env_id: found)
    blender_render.main(
        ["--env", "myoLegWalk-v0", "--output", str(tmp_path / "a"), "--export-only"]
    )
    assert rendered[-1].checkpoint == found

    blender_render.main(
        ["--env", "myoLegWalk-v0", "--output", str(tmp_path / "b"), "--random"]
    )
    assert rendered[-1].checkpoint is None

    monkeypatch.setattr(checkpoint_utils, "find_checkpoint", lambda env_id: None)
    with pytest.raises(SystemExit):
        blender_render.main(["--env", "myoLegWalk-v0", "--output", str(tmp_path / "c")])
    assert len(rendered) == 2


@pytest.mark.parametrize("preview", [False, True])
def test_render_rollout_ends_with_the_result_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    preview: bool,
) -> None:
    """The last line printed is the absolute path of the video (or of the preview image)."""
    import myosuite.utils.video_io as video_io

    meta = {"samples": 3, "duration": 0.1, "usd": "x.usdc"}
    monkeypatch.setattr(blender_render, "export_rollout", lambda config: meta)
    monkeypatch.setattr(blender_render.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(video_io, "write_video", lambda *a, **kw: None)
    out = tmp_path / "render"
    blender_render.render_rollout(
        RenderConfig(env="myoLegWalk-v0", output=out, preview=preview)
    )
    last = capsys.readouterr().out.strip().splitlines()[-1]
    expected = out.resolve() / ("preview.png" if preview else "video.mp4")
    assert last == f"{'Preview' if preview else 'Video'}: {expected}"
