# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for trajectory_io helpers."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.core.trajectory_io import (
    MotionClip,
    expand_motion_clip_to_model,
    load_motion_clip,
    resolve_motion_path,
)

pytestmark = pytest.mark.tier2

_UNPICKLED: list[str] = []


def _record_unpickling(tag: str) -> str:
    _UNPICKLED.append(tag)
    return tag


class _Tripwire:
    """Records a call when unpickled, where a malicious pickle would run code."""

    def __reduce__(self):
        return (_record_unpickling, ("unpickled",))


def _write_tripwire_clip(path: Path, key: str) -> None:
    np.savez(
        path,
        qpos=np.zeros((2, 1)),
        site_xpos=np.zeros((2, 1, 3)),
        **{key: np.array([_Tripwire()], dtype=object)},
    )


def _write_motion(path: Path, nq: int, nv: int) -> None:
    frames = 4
    np.savez(
        path,
        qpos=np.zeros((frames, nq), dtype=np.float32),
        qvel=np.zeros((frames, nv), dtype=np.float32),
        frequency=np.array(100.0, dtype=np.float32),
    )


def test_resolve_motion_path_absolute(tmp_path: Path) -> None:
    """Absolute NPZ path should resolve as-is."""
    p = tmp_path / "m.npz"
    _write_motion(p, nq=3, nv=2)
    got = resolve_motion_path(str(p))
    assert got == p.resolve()


def test_load_motion_clip_validates_dimensions(tmp_path: Path) -> None:
    """Loader validates qpos/qvel widths against model nq/nv."""
    p = tmp_path / "m.npz"
    _write_motion(p, nq=5, nv=4)
    clip = load_motion_clip(p, expected_nq=5, expected_nv=4)
    assert clip.qpos.shape == (4, 5)
    assert clip.qvel is not None
    assert clip.frequency_hz == pytest.approx(100.0)


def test_load_motion_clip_rejects_bad_qpos(tmp_path: Path) -> None:
    """Wrong qpos width should raise ValueError."""
    p = tmp_path / "bad.npz"
    np.savez(p, qpos=np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="qpos width mismatch"):
        load_motion_clip(p, expected_nq=4, expected_nv=3)


def test_load_motion_clip_rejects_empty_frames(tmp_path: Path) -> None:
    path = tmp_path / "empty.npz"
    np.savez(path, qpos=np.empty((0, 2)), qvel=np.empty((0, 2)))
    with pytest.raises(ValueError, match="qpos must contain at least one frame"):
        load_motion_clip(path, expected_nq=2, expected_nv=2)


@pytest.mark.parametrize("field", ["qvel", "site_xpos"])
@pytest.mark.parametrize("frames", [0, 1, 4])
def test_load_motion_clip_rejects_mismatched_frame_counts(
    tmp_path: Path, field: str, frames: int
) -> None:
    path = tmp_path / "mismatched.npz"
    values = np.zeros((frames, 1) if field == "qvel" else (frames, 2, 3))
    np.savez(path, qpos=np.zeros((3, 1)), **{field: values})
    with pytest.raises(
        ValueError, match=f"{field} frame count mismatch: expected 3, got {frames}"
    ):
        load_motion_clip(path, expected_nq=1, expected_nv=1)


@pytest.mark.parametrize(
    "shape", [(), (3,), (3, 3), (3, 2, 2), (3, 2, 4), (3, 2, 3, 1)]
)
def test_load_motion_clip_rejects_malformed_site_positions(
    tmp_path: Path, shape: tuple[int, ...]
) -> None:
    path = tmp_path / "bad_sites.npz"
    np.savez(path, qpos=np.zeros((3, 1)), site_xpos=np.zeros(shape))
    with pytest.raises(ValueError, match=r"site_xpos must have shape \(T, S, 3\)"):
        load_motion_clip(path, expected_nq=1, expected_nv=1)


@pytest.mark.parametrize("with_qvel", [False, True])
def test_load_motion_clip_keeps_single_frame_partial_clips(
    tmp_path: Path, with_qvel: bool
) -> None:
    """One-frame named subsets and zero sites remain valid, with optional velocities."""
    path = tmp_path / "single.npz"
    velocities = {"qvel": np.array([[0.3]])} if with_qvel else {}
    np.savez(
        path,
        qpos=np.array([[0.2]]),
        site_xpos=np.empty((1, 0, 3)),
        joint_names=np.array(["joint_a"]),
        **velocities,
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="joint_a"/>'
        '<joint name="joint_b" axis="0 1 0"/><geom size="0.1"/>'
        "</body></worldbody></mujoco>"
    )
    clip = load_motion_clip(path, expected_nq=model.nq, expected_nv=model.nv)
    assert clip.num_frames == 1 and clip.site_xpos.shape == (1, 0, 3)
    assert clip.qpos.shape == (1, 1) and clip.qpos_joint_names == ["joint_a"]
    assert (clip.qvel is not None) == with_qvel
    data = expand_motion_clip_to_model(clip, model).to_mjdata(model)
    np.testing.assert_array_equal(data.qpos, [0.2, 0.0])
    np.testing.assert_array_equal(data.qvel, [0.3 if with_qvel else 0.0, 0.0])


def test_load_motion_clip_accepts_partial_joint_subset(tmp_path: Path) -> None:
    """Named joint subsets should load without full-width qpos/qvel arrays."""
    p = tmp_path / "partial.npz"
    np.savez(
        p,
        qpos=np.array([[0.1], [0.2]], dtype=np.float32),
        qvel=np.array([[0.0], [0.1]], dtype=np.float32),
        joint_names=np.array(["joint_a"]),
    )

    clip = load_motion_clip(p, expected_nq=2, expected_nv=2)

    assert clip.qpos.shape == (2, 1)
    assert clip.qvel is not None and clip.qvel.shape == (2, 1)
    assert clip.qpos_joint_names == ["joint_a"]
    assert clip.qvel_joint_names == ["joint_a"]
    assert clip.qpos_model_indices is None
    assert clip.qvel_model_indices is None


def test_expand_motion_clip_to_model_expands_named_subset() -> None:
    """Partial clips should expand to model width using joint-name metadata."""
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <joint name="joint_a" type="hinge"/>
              <joint name="joint_b" type="hinge"/>
              <geom type="sphere" size="0.01"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    clip = MotionClip(
        qpos=np.array([[0.1], [0.2]], dtype=np.float64),
        qvel=np.array([[0.0], [0.3]], dtype=np.float64),
        site_xpos=None,
        site_names=None,
        qpos_joint_names=["joint_a"],
        qvel_joint_names=["joint_a"],
    )
    expanded = expand_motion_clip_to_model(clip, model)

    assert expanded.qpos is not None and expanded.qpos.shape == (2, 2)
    assert expanded.qvel is not None and expanded.qvel.shape == (2, 2)
    np.testing.assert_allclose(expanded.qpos[:, 0], [0.1, 0.2])
    np.testing.assert_allclose(expanded.qvel[:, 0], [0.0, 0.3])
    np.testing.assert_allclose(expanded.qpos[:, 1], 0.0)
    np.testing.assert_allclose(expanded.qvel[:, 1], 0.0)
    np.testing.assert_array_equal(
        expanded.qpos_model_indices, np.array([0], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        expanded.qvel_model_indices, np.array([0], dtype=np.int32)
    )


def test_motion_clip_weights_default_to_none(tmp_path: Path) -> None:
    """Backward compatible: clips without a weights entry have ``weights is None``."""
    p = tmp_path / "m.npz"
    _write_motion(p, nq=5, nv=4)
    assert load_motion_clip(p, expected_nq=5, expected_nv=4).weights is None


def test_motion_clip_weights_load_and_survive_expansion(tmp_path: Path) -> None:
    """Per-frame weights are loaded from the NPZ and kept when a clip is expanded."""
    p = tmp_path / "w.npz"
    np.savez(
        p,
        qpos=np.zeros((4, 1)),
        qpos_joint_names=np.array(["joint_a"]),
        weights=np.array([1.0, 0.5, 0.25, 0.0], dtype=np.float32),
    )
    clip = load_motion_clip(p, expected_nq=2, expected_nv=2)
    assert clip.weights is not None
    np.testing.assert_allclose(clip.weights, [1.0, 0.5, 0.25, 0.0])
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="joint_a" type="hinge"/>'
        '<joint name="joint_b" type="hinge"/><geom size="0.01"/></body></worldbody></mujoco>'
    )
    expanded = expand_motion_clip_to_model(clip, model)
    assert expanded.qpos.shape == (4, 2)
    np.testing.assert_allclose(expanded.weights, clip.weights)


@pytest.mark.parametrize("key", ["joint_names", "site_names"])
def test_load_motion_clip_never_unpickles(tmp_path: Path, key: str) -> None:
    """A downloaded clip with a pickled name array is refused, not executed."""
    p = tmp_path / "pickled.npz"
    _write_tripwire_clip(p, key)
    _UNPICKLED.clear()
    if key == "joint_names":
        with pytest.raises(ValueError, match="Re-save the names as strings"):
            load_motion_clip(p, expected_nq=1, expected_nv=1)
    else:
        with pytest.warns(UserWarning, match="pickled object array"):
            clip = load_motion_clip(p, expected_nq=1, expected_nv=1)
        assert clip.site_names is None
    assert _UNPICKLED == []
    np.load(p, allow_pickle=True)[key]  # what the loader used to do
    assert _UNPICKLED == ["unpickled"]


def test_mimic_clip_env_names_never_unpickle(tmp_path: Path) -> None:
    from myosuite.envs.myo.tasks.mimic.clip_env import _decode_joint_names

    p = tmp_path / "pickled.npz"
    _write_tripwire_clip(p, "joint_names")
    _UNPICKLED.clear()
    with pytest.raises(ValueError, match="Re-save the names as strings"):
        _decode_joint_names(np.load(p, allow_pickle=False), ("joint_names",))
    assert _UNPICKLED == []


def test_names_stored_as_bytes_are_decoded(tmp_path: Path) -> None:
    """Bytes names read as text (str() used to give "b'joint_a'")."""
    p = tmp_path / "bytes.npz"
    np.savez(
        p,
        qpos=np.zeros((2, 1)),
        qpos_joint_names=np.array([b"joint_a"]),
        site_xpos=np.zeros((2, 1, 3)),
        site_names=np.array([b"site_a"]),
    )
    clip = load_motion_clip(p, expected_nq=2, expected_nv=2)
    assert clip.qpos_joint_names == ["joint_a"]
    assert clip.site_names == ["site_a"]


def test_expand_motion_clip_maps_full_width_clip_by_name(tmp_path: Path) -> None:
    """A full-width clip stored in another joint order is reordered by name.

    It used to be applied column by column (names ignored at full width).
    """
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="pelvis"><freejoint name="root"/>'
        '<geom size="0.1"/><body name="shank"><joint name="knee" type="hinge"/>'
        '<geom size="0.05"/></body></body></worldbody></mujoco>'
    )
    root_qpos = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    root_qvel = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    p = tmp_path / "reordered.npz"
    np.savez(  # columns: knee, then the 7 / 6 root coordinates
        p,
        qpos=np.array([[0.5, *root_qpos]]),
        qvel=np.array([[-0.5, *root_qvel]]),
        joint_names=np.array(["knee", "root"]),
    )
    clip = load_motion_clip(p, expected_nq=model.nq, expected_nv=model.nv)
    expanded = expand_motion_clip_to_model(clip, model)

    np.testing.assert_allclose(expanded.qpos, [[*root_qpos, 0.5]])
    np.testing.assert_allclose(expanded.qvel, [[*root_qvel, -0.5]])
    np.testing.assert_allclose(expanded.qpos[:, expanded.qpos_model_indices], clip.qpos)
    # Idempotent, and positional full-width clips (no names) are left as they are.
    again = expand_motion_clip_to_model(expanded, model)
    np.testing.assert_array_equal(again.qpos, expanded.qpos)
    positional = MotionClip(
        qpos=expanded.qpos, qvel=expanded.qvel, site_xpos=None, site_names=None
    )
    np.testing.assert_array_equal(
        expand_motion_clip_to_model(positional, model).qpos, expanded.qpos
    )


def test_motion_clip_weights_must_match_the_frame_count() -> None:
    with pytest.raises(ValueError, match="weights"):
        MotionClip(
            qpos=np.zeros((3, 2)),
            qvel=None,
            site_xpos=None,
            site_names=None,
            weights=np.ones(5),
        )


def test_supplied_reference_velocity_preserves_states_and_site_kinematics() -> None:
    """A procedural reference must keep its velocities and unrounded poses."""
    from myosuite.core.trajectory_io import motion_clip_from_states

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint type="hinge"/>'
        '<geom size="0.1"/><site name="tip" pos="1 0 0"/>'
        "</body></worldbody></mujoco>"
    )
    poses = np.array([[0.123456789012], [0.223456789012]])
    velocities = np.array([[2.0], [3.0]])
    clip = motion_clip_from_states(model, poses, 0.01, ("tip",), qvel=velocities)
    np.testing.assert_array_equal(clip.qpos, poses)
    np.testing.assert_array_equal(clip.qvel, velocities)
    data = mujoco.MjData(model)
    for frame in range(2):
        data.qpos[:] = poses[frame]
        data.qvel[:] = velocities[frame]
        mujoco.mj_forward(model, data)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, 0, velocity, 0)
        np.testing.assert_array_equal(clip.site_xpos[frame, 0], data.site_xpos[0])
        np.testing.assert_array_equal(clip.site_velocity[frame, 0], velocity)
    with pytest.raises(ValueError, match="qvel"):
        motion_clip_from_states(model, poses, 0.01, ("tip",), qvel=np.zeros((1, 1)))


@pytest.mark.parametrize("prefix", ["", "a0_", "a1_"])
@pytest.mark.parametrize("reverse", [False, True])
def test_motion_clip_elbow_aliases(prefix: str, reverse: bool) -> None:
    source, target = ("elbow_flex", "elbow_flexion")
    if reverse:
        source, target = target, source
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><body><freejoint name="{prefix}root"/>'
        f'<geom size="0.1"/><body><joint name="{prefix}{target}_r"/>'
        f'<joint name="{prefix}{target}_l" axis="0 1 0"/>'
        '<geom size="0.1"/></body></body></worldbody></mujoco>'
    )
    clip = MotionClip(
        qpos=np.array([[0.3, 0.8]]),
        qvel=np.array([[0.4, -0.2]]),
        site_xpos=None,
        site_names=None,
        qpos_joint_names=[f"{prefix}{source}_l", f"{prefix}{source}_r"],
        qvel_joint_names=[f"{prefix}{source}_l", f"{prefix}{source}_r"],
    )
    expanded = expand_motion_clip_to_model(clip, model)
    np.testing.assert_array_equal(expanded.qpos[:, 7:], [[0.8, 0.3]])
    np.testing.assert_array_equal(expanded.qvel[:, 6:], [[-0.2, 0.4]])
    np.testing.assert_array_equal(expanded.qpos_model_indices, [8, 7])
    np.testing.assert_array_equal(expanded.qvel_model_indices, [7, 6])
    again = expand_motion_clip_to_model(expanded, model)
    np.testing.assert_array_equal(again.qpos, expanded.qpos)
    np.testing.assert_array_equal(clip.qpos, [[0.3, 0.8]])


@pytest.mark.parametrize("field", ["qpos", "qvel"])
def test_motion_clip_rejects_alias_collision(field: str) -> None:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="elbow_flexion_r"/>'
        '<geom size="0.1"/></body></worldbody></mujoco>'
    )
    kwargs = {
        "qpos": np.zeros((1, 1)),
        "qvel": None,
        "site_xpos": None,
        "site_names": None,
    }
    kwargs[field] = np.zeros((1, 2))
    kwargs[field + "_joint_names"] = ["elbow_flexion_r", "elbow_flex_r"]
    with pytest.raises(ValueError, match="both map to model joint"):
        expand_motion_clip_to_model(MotionClip(**kwargs), model)


def test_motion_clip_exact_joint_name_wins_over_alias() -> None:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="elbow_flexion_r"/>'
        '<joint name="elbow_flex_r" axis="0 1 0"/><geom size="0.1"/>'
        "</body></worldbody></mujoco>"
    )
    clip = MotionClip(
        qpos=np.array([[0.2, 0.7]]),
        qvel=None,
        site_xpos=None,
        site_names=None,
        qpos_joint_names=["elbow_flex_r", "elbow_flexion_r"],
    )
    np.testing.assert_array_equal(
        expand_motion_clip_to_model(clip, model).qpos, [[0.7, 0.2]]
    )


def test_native_chase_tag_accepts_musclemimic_joint_names(tmp_path: Path) -> None:
    from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_config import (
        ChaseTagVsConfig,
    )
    from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_model import (
        build_chase_tag_vs_fullbody_model,
    )

    model, _, _ = build_chase_tag_vs_fullbody_model(ChaseTagVsConfig())
    names = [
        model.joint(i).name.replace("elbow_flexion_", "elbow_flex_")
        for i in range(model.njnt)
    ]
    assert sum("elbow_flex_" in name for name in names) == 4
    qpos = np.repeat(model.qpos0[None, :], 2, axis=0)
    for i in range(model.njnt):
        if "elbow_flex_" in names[i]:
            qpos[:, model.jnt_qposadr[i]] = [0.3, 0.7]
    qvel = np.arange(2 * model.nv, dtype=float).reshape(2, model.nv) / 100
    path = tmp_path / "pose.npz"
    np.savez(path, qpos=qpos, qvel=qvel, joint_names=names)
    clip = load_motion_clip(path, model.nq, model.nv)
    expanded = expand_motion_clip_to_model(clip, model)
    np.testing.assert_array_equal(expanded.qpos, qpos)
    np.testing.assert_array_equal(expanded.qvel, qvel)


@pytest.mark.parametrize("name", ["unknown_joint", "a1_elbow_flex_r"])
def test_motion_clip_aliases_do_not_hide_unknown_joints(name: str) -> None:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="a0_elbow_flexion_r"/>'
        '<geom size="0.1"/></body></worldbody></mujoco>'
    )
    clip = MotionClip(
        qpos=np.array([[0.2]]),
        qvel=None,
        site_xpos=None,
        site_names=None,
        qpos_joint_names=[name],
    )
    with pytest.raises(ValueError, match="unknown model joints"):
        expand_motion_clip_to_model(clip, model)
