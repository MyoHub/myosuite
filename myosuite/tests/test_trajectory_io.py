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


# Free-root pelvis with a site above the knee; the knee (master) drives the shank's
# rotation (nonzero ref) and slide through quartic couplings, and the toes site is
# below them.
_KNEE_XML = """
<mujoco>
  <worldbody>
    <body name="pelvis">
      <freejoint name="root"/>
      <geom size="0.1" mass="1"/>
      <site name="hip" pos="0 0.1 0"/>
      <body name="thigh" pos="0 0 -0.1">
        <joint name="knee" axis="0 1 0"/>
        <geom size="0.05" mass="1"/>
        <body name="shank" pos="0 0 -0.4">
          <joint name="shank_rot" axis="0 0 1" ref="0.05"/>
          <joint name="shank_slide" type="slide" axis="1 0 0"/>
          <geom size="0.05" mass="1"/>
          <site name="toes" pos="0.15 0 -0.4"/>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="shank_rot" joint2="knee" polycoef="0 0.37 -0.17 0.025 0"/>
    <joint joint1="shank_slide" joint2="knee" polycoef="0 0.006 -0.001 0 0"/>
  </equality>
</mujoco>
"""
_SITES = ("hip", "toes")


def _knee_clip(model: mujoco.MjModel) -> MotionClip:
    """Knee sweep whose shank joints follow the negated polynomials, as the
    MuscleMimic clips the 1.0.5 left knee; sites and velocities are its kinematics."""
    from myosuite.core.trajectory_io import motion_clip_from_states

    rng = np.random.default_rng(0)
    knee = np.linspace(0.0, 1.2, 25)
    qpos = np.tile(model.qpos0, (len(knee), 1))
    qpos[:, :3] += rng.normal(0.0, 0.1, (len(knee), 3))
    qpos[:, 7] = knee
    qpos[:, 8] = 0.05 - (0.37 * knee - 0.17 * knee**2 + 0.025 * knee**3)
    qpos[:, 9] = -(0.006 * knee - 0.001 * knee**2)
    qvel = rng.normal(0.0, 1.0, (len(knee), model.nv))
    return motion_clip_from_states(model, qpos, 0.01, _SITES, qvel=qvel)


def test_project_motion_clip_couplings_serves_one_reachable_state_per_frame() -> None:
    from myosuite.core.trajectory_io import (
        motion_clip_from_states,
        project_motion_clip_couplings,
    )
    from myosuite.tests.support.joint_couplings import joint_equality_residuals

    model = mujoco.MjModel.from_xml_string(_KNEE_XML)
    clip = _knee_clip(model)
    stored = clip.qpos.copy()
    assert np.abs(joint_equality_residuals(model, clip.qpos[-1])[0]).max() > 0.4

    projected = project_motion_clip_couplings(clip, model)

    # MuJoCo's own residuals: positions on the couplings, velocities along them.
    for qpos, qvel in zip(projected.qpos, projected.qvel):
        for residual in joint_equality_residuals(model, qpos, qvel):
            np.testing.assert_allclose(residual, 0.0, rtol=0, atol=1e-12)
    # The site tracks are the kinematics of the projected states.
    expected = motion_clip_from_states(
        model, projected.qpos, 0.01, _SITES, qvel=projected.qvel
    )
    for name in ("site_xpos", "site_xmat", "site_velocity"):
        np.testing.assert_allclose(
            getattr(projected, name), getattr(expected, name), rtol=0, atol=1e-12
        )
    # Root, master and the site above the coupling keep their values; toes move.
    np.testing.assert_array_equal(projected.qpos[:, :8], stored[:, :8])
    np.testing.assert_array_equal(projected.site_xpos[:, 0], clip.site_xpos[:, 0])
    assert (
        np.linalg.norm(projected.site_xpos[:, 1] - clip.site_xpos[:, 1], -1).max()
        > 0.05
    )
    np.testing.assert_array_equal(clip.qpos, stored)
    again = project_motion_clip_couplings(projected, model)
    for name in ("qpos", "qvel", "site_xpos", "site_xmat", "site_velocity"):
        np.testing.assert_allclose(
            getattr(again, name), getattr(projected, name), rtol=0, atol=1e-14
        )


def test_project_motion_clip_couplings_moves_sites_with_their_bodies() -> None:
    """Stored sites off the kinematics (e.g. markers) keep their offset to the bodies."""
    import dataclasses

    from myosuite.core.trajectory_io import project_motion_clip_couplings

    model = mujoco.MjModel.from_xml_string(_KNEE_XML)
    clip = _knee_clip(model)
    offset = np.array([0.0, 0.01, 0.02])
    marked = dataclasses.replace(clip, site_xpos=clip.site_xpos + offset)
    np.testing.assert_allclose(
        project_motion_clip_couplings(marked, model).site_xpos,
        project_motion_clip_couplings(clip, model).site_xpos + offset,
        rtol=0,
        atol=1e-12,
    )


def test_project_motion_clip_couplings_needs_the_model_layout() -> None:
    import dataclasses

    from myosuite.core.trajectory_io import project_motion_clip_couplings

    model = mujoco.MjModel.from_xml_string(_KNEE_XML)
    clip = _knee_clip(model)
    narrow = MotionClip(
        qpos=clip.qpos[:, :8], qvel=None, site_xpos=None, site_names=None
    )
    with pytest.raises(ValueError, match="expand_motion_clip_to_model"):
        project_motion_clip_couplings(narrow, model)
    for names in (None, ["hip", "nose"]):
        with pytest.raises(ValueError, match="site name"):
            project_motion_clip_couplings(
                dataclasses.replace(clip, site_names=names), model
            )
    uncoupled = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size="0.1"/>'
        '<body><joint name="knee"/><geom size="0.1"/></body></body>'
        "</worldbody></mujoco>"
    )
    plain = MotionClip(
        qpos=clip.qpos[:, :8], qvel=None, site_xpos=None, site_names=None
    )
    assert project_motion_clip_couplings(plain, uncoupled) is plain
