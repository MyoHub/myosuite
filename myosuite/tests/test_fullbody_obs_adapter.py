# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""FullbodyObsAdapter.build: exact upstream layout, vectorized hot path."""

from __future__ import annotations

import gc
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from myosuite.core.trajectory_io import MotionClip, load_motion_clip
from myosuite.integrations.musclemimic import fullbody_local_policy as fbl
from myosuite.integrations.musclemimic.fullbody_local_policy import FullbodyObsAdapter

pytestmark = pytest.mark.tier1

_SITES = ["pelvis_mimic", "knee_mimic", "ankle_mimic", "hand_mimic"]
_ACTUATED_JOINTS = ("hip", "knee", "shoulder")
_TOUCH = ("r_foot", "r_toes", "l_foot", "l_toes")
_MUSCLE_FLAGS = (
    "enable_muscle_length_observations",
    "enable_muscle_velocity_observations",
    "enable_muscle_force_observations",
    "enable_muscle_excitation_observations",
    "enable_muscle_activation_observations",
)

# Free "root" first, then hinge / ball joints, so non-root qpos = qpos[7:].
_XML = """
<mujoco model="fullbody_obs_toy">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="pelvis" pos="0 0 0.72">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.15 0.05" mass="5"/>
      <site name="pelvis_mimic"/>
      <body name="thigh" pos="0 0.1 -0.1">
        <joint name="hip" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05" mass="2"/>
        <site name="knee_mimic" pos="0 0 -0.4"/>
        <body name="shank" pos="0 0 -0.4">
          <joint name="knee" axis="0 1 0" range="-2 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="1.5"/>
          <body name="foot" pos="0 0 -0.4">
            <joint name="ankle" type="ball" range="0 0.6"/>
            <geom type="box" size="0.1 0.05 0.02" mass="0.5"/>
            <site name="ankle_mimic" pos="0.05 0 0"/>
            <site name="r_foot_site" type="box" size="0.11 0.06 0.04"/>
            <site name="r_toes_site" type="sphere" pos="0.1 0 0" size="0.04"/>
          </body>
        </body>
      </body>
      <body name="arm" pos="0 -0.2 0.05">
        <joint name="shoulder" axis="1 0 0" range="-1.5 1.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="1"/>
        <site name="hand_mimic" pos="0 0 -0.3"/>
        <site name="l_foot_site" type="sphere" pos="0 0 -0.3" size="0.05"/>
        <site name="l_toes_site" type="sphere" pos="0 0 -0.15" size="0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>__ACTUATORS__</actuator>
  <sensor>
    <touch name="r_foot" site="r_foot_site"/>
    <touch name="r_toes" site="r_toes_site"/>
    <touch name="l_foot" site="l_foot_site"/>
    <touch name="l_toes" site="l_toes_site"/>
  </sensor>
</mujoco>
"""


def _toy_model(n_actuators: int) -> mujoco.MjModel:
    """Toy full-body-like model with *n_actuators* activation-dynamics actuators."""
    actuators = "".join(
        f'<general name="m{i}" joint="{_ACTUATED_JOINTS[i % 3]}" dyntype="filter" '
        f'dynprm="0.05" gainprm="{5 + i}" ctrlrange="0 1"/>'
        for i in range(n_actuators)
    )
    return mujoco.MjModel.from_xml_string(_XML.replace("__ACTUATORS__", actuators))


def _write_clip(
    model: mujoco.MjModel, path: Path, *, n_frames: int = 24, with_qvel: bool = True
) -> MotionClip:
    """Write a smooth clip NPZ in the MuscleMimic trajectory layout and load it."""
    t = np.arange(n_frames) * 0.01
    qpos = np.repeat(model.qpos0[None], n_frames, axis=0)
    yaw = 0.3 * np.sin(2.0 * t)
    qpos[:, 0] = 0.8 * t
    qpos[:, 2] = 0.72 + 0.02 * np.sin(8.0 * t)
    qpos[:, 3:7] = np.stack([np.cos(yaw / 2), 0 * t, 0 * t, np.sin(yaw / 2)], axis=1)
    adr = {
        n: int(model.joint(n).qposadr[0]) for n in ("hip", "knee", "ankle", "shoulder")
    }
    qpos[:, adr["hip"]] = 0.5 * np.sin(6.0 * t)
    qpos[:, adr["knee"]] = -0.8 + 0.6 * np.sin(6.0 * t + 1.0)
    ang = 0.3 * np.sin(6.0 * t + 2.0)
    qpos[:, adr["ankle"] : adr["ankle"] + 4] = np.stack(
        [np.cos(ang / 2), np.sin(ang / 2), 0 * t, 0 * t], axis=1
    )
    qpos[:, adr["shoulder"]] = 0.7 * np.sin(6.0 * t + 3.0)
    qvel = np.zeros((n_frames, model.nv))
    for i in range(1, n_frames):
        mujoco.mj_differentiatePos(model, qvel[i], 0.01, qpos[i - 1], qpos[i])

    data = mujoco.MjData(model)
    rec: dict[str, list[np.ndarray]] = {
        k: [] for k in ("site_xpos", "site_xmat", "cvel", "subtree_com")
    }
    for qp, qv in zip(qpos, qvel):
        data.qpos[:] = qp
        data.qvel[:] = qv
        mujoco.mj_forward(model, data)
        for key, rows in rec.items():
            rows.append(getattr(data, key).copy())
    arrays: dict[str, np.ndarray] = {k: np.asarray(v) for k, v in rec.items()}
    arrays.update(
        qpos=qpos,
        site_bodyid=np.asarray(model.site_bodyid),
        body_rootid=np.asarray(model.body_rootid),
        site_names=np.asarray([model.site(i).name for i in range(model.nsite)]),
    )
    if with_qvel:
        arrays["qvel"] = qvel
    np.savez(path, **arrays)
    return load_motion_clip(path, expected_nq=model.nq, expected_nv=model.nv)


def _set_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: MotionClip,
    frame: int,
    seed: int,
    *,
    root_z: float | None = None,
) -> None:
    """Clip pose with perturbed hinges, random velocities, excitations, activations."""
    rng = np.random.default_rng(seed)
    data.qpos[:] = clip.qpos[frame]
    if root_z is not None:
        data.qpos[2] = root_z
    for name in _ACTUATED_JOINTS:
        data.qpos[model.joint(name).qposadr[0]] += rng.normal(0.0, 0.1)
    data.qvel[:] = rng.normal(0.0, 0.5, model.nv)
    data.ctrl[:] = rng.uniform(0.0, 1.0, model.nu)
    data.act[:] = rng.uniform(0.0, 1.0, model.na)
    mujoco.mj_forward(model, data)


def _reference_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: MotionClip,
    goal_params: dict[str, Any],
    frame_idx: int,
) -> np.ndarray:
    """Straightforward upstream layout: per-actuator and per-lookahead-step loops."""
    f32 = np.float32

    def flag(key: str) -> bool:
        return bool(goal_params.get(key, True))

    concise = bool(goal_params.get("use_concise_lookahead", False))
    n_look = int(goal_params.get("n_step_lookahead", 5))
    stride = int(goal_params.get("n_step_stride", 1))

    parts: list[Any] = []
    if flag("enable_joint_pos_observations"):
        parts += [data.qpos[2:7], data.qpos[7:]]
    if flag("enable_joint_vel_observations"):
        parts += [data.qvel[:6], data.qvel[6:]]
    muscle = (
        data.actuator_length,
        data.actuator_velocity,
        data.actuator_force,
        data.ctrl,
        data.act,
    )
    fields = dict(zip(_MUSCLE_FLAGS, muscle))
    for act_idx in range(model.nu):
        for key in _MUSCLE_FLAGS:
            if flag(key):
                parts.append([fields[key][act_idx]])
    if flag("enable_touch_sensor_observations"):
        sens = np.asarray(data.sensordata, dtype=f32)
        for name in _TOUCH:
            adr, dim = int(model.sensor(name).adr[0]), int(model.sensor(name).dim[0])
            parts.append([float(np.sum(sens[adr : adr + dim]))])

    site_ids = np.asarray([model.site(n).id for n in _SITES])
    rpos, rangles, rvel = fbl._relative_site_quantities(
        site_ids=site_ids,
        site_xpos=np.asarray(data.site_xpos, dtype=f32),
        site_xmat=np.asarray(data.site_xmat, dtype=f32),
        cvel_parent=np.asarray(data.cvel, dtype=f32),
        subtree_com_root=np.asarray(data.subtree_com, dtype=f32),
        site_bodyid=model.site_bodyid,
        body_rootid=model.body_rootid,
    )
    goal: list[Any] = [rpos] if flag("enable_mimic_site_rpos_observations") else []
    goal += [rangles, rvel]

    with np.load(clip.source_path) as npz:
        traj = {key: npz[key] for key in npz.files}
    names = [str(n) for n in traj["site_names"]]
    traj_ids = np.asarray([names.index(n) for n in _SITES])
    n_frames = int(clip.qpos.shape[0])
    qpos = np.asarray(clip.qpos, dtype=f32)
    qvel = (
        np.asarray(clip.qvel, dtype=f32)
        if clip.qvel is not None
        else np.zeros((n_frames, model.nv), dtype=f32)
    )
    futures = [min(n_frames - 1, frame_idx + k * stride) for k in range(n_look)]
    per_frame = [
        fbl._relative_site_quantities(
            site_ids=traj_ids,
            site_xpos=np.asarray(traj["site_xpos"], dtype=f32)[f],
            site_xmat=np.asarray(traj["site_xmat"], dtype=f32)[f],
            cvel_parent=np.asarray(traj["cvel"], dtype=f32)[f],
            subtree_com_root=np.asarray(traj["subtree_com"], dtype=f32)[f],
            site_bodyid=traj["site_bodyid"],
            body_rootid=traj["body_rootid"],
        )
        for f in futures
    ]
    if concise:
        goal.append(per_frame[0][0])
        for k in range(1, n_look):
            goal += [
                qpos[futures[k], :3] - qpos[frame_idx, :3],
                qvel[futures[k], :6] - qvel[frame_idx, :6],
                per_frame[k][0],
            ]
    else:
        goal += [qpos[futures, 2:], qvel[futures]]
        goal += [np.stack([q[j] for q in per_frame]) for j in range(3)]
    if flag("enable_motion_phase"):
        goal.append([float(frame_idx) / float(max(n_frames, 1))])
    return np.concatenate([np.asarray(p, dtype=f32).reshape(-1) for p in parts + goal])


def _count_python_calls(fn: Callable[[], object]) -> int:
    """Number of Python and builtin function calls made while running *fn*."""
    count = 0

    def _profile(_frame: Any, event: str, _arg: Any) -> None:
        nonlocal count
        if event in ("call", "c_call"):
            count += 1

    previous = sys.getprofile()
    gc_enabled = gc.isenabled()
    gc.disable()
    sys.setprofile(_profile)
    try:
        fn()
    finally:
        sys.setprofile(previous)
        if gc_enabled:
            gc.enable()
    return count


_LAYOUTS = {
    "concise": {"use_concise_lookahead": True},
    "concise_stride_no_phase": {
        "use_concise_lookahead": True,
        "n_step_lookahead": 3,
        "n_step_stride": 2,
        "enable_motion_phase": False,
    },
    "full": {"n_step_lookahead": 3},
    "full_subset": {
        "n_step_lookahead": 2,
        "n_step_stride": 3,
        "enable_mimic_site_rpos_observations": False,
        "enable_muscle_force_observations": False,
        "enable_muscle_excitation_observations": False,
        "enable_touch_sensor_observations": False,
    },
}


@pytest.mark.parametrize("with_qvel", [True, False], ids=["qvel", "no_qvel"])
@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_build_is_bit_identical_to_reference_layout(
    tmp_path: Path, layout: str, with_qvel: bool
) -> None:
    """Cached + vectorized build equals the loop-form layout bit for bit."""
    model = _toy_model(7)
    clip = _write_clip(model, tmp_path / "clip.npz", with_qvel=with_qvel)
    goal_params = {"sites_for_mimic": _SITES, **_LAYOUTS[layout]}
    adapter = FullbodyObsAdapter(model, clip, goal_params)
    data = mujoco.MjData(model)
    n = int(clip.qpos.shape[0])
    touched = False
    # Clamped end frames and revisits (cache hits); fill outputs to catch aliasing.
    for seed, frame in enumerate((0, 3, n - 4, n - 2, n - 1, 3, 0, n - 1)):
        # One state with the feet pushed into the floor: non-zero touch sums.
        _set_state(model, data, clip, frame, seed, root_z=0.45 if seed == 2 else None)
        touched |= bool(np.any(data.sensordata > 0.0))
        obs = adapter.build(data, frame)
        ref = _reference_obs(model, data, clip, goal_params, frame)
        assert obs.dtype == np.float32
        np.testing.assert_array_equal(obs.view(np.uint32), ref.view(np.uint32))
        obs.fill(np.nan)
    assert touched


def test_with_clip_matches_reference_on_new_clip(tmp_path: Path) -> None:
    """``with_clip`` serves the new trajectory, not the original clip's caches."""
    model = _toy_model(5)
    clip_a = _write_clip(model, tmp_path / "a.npz", n_frames=24)
    clip_b = _write_clip(model, tmp_path / "b.npz", n_frames=17, with_qvel=False)
    goal_params = {"sites_for_mimic": _SITES, "use_concise_lookahead": True}
    adapter = FullbodyObsAdapter(model, clip_a, goal_params)
    data = mujoco.MjData(model)
    _set_state(model, data, clip_b, 6, 1)
    adapter.build(data, 6)
    obs = adapter.with_clip(clip_b).build(data, 6)
    ref = _reference_obs(model, data, clip_b, goal_params, 6)
    np.testing.assert_array_equal(obs.view(np.uint32), ref.view(np.uint32))


@pytest.mark.parametrize("layout", ["concise", "full"])
def test_build_does_no_per_actuator_python_work(tmp_path: Path, layout: str) -> None:
    """Python-level call count of ``build`` is independent of the actuator count."""
    counts = []
    for n_actuators in (3, 30):
        model = _toy_model(n_actuators)
        clip = _write_clip(model, tmp_path / f"clip_{n_actuators}.npz")
        goal_params = {"sites_for_mimic": _SITES, **_LAYOUTS[layout]}
        adapter = FullbodyObsAdapter(model, clip, goal_params)
        data = mujoco.MjData(model)
        _set_state(model, data, clip, 4, 0)
        adapter.build(data, 4)
        counts.append(_count_python_calls(lambda: adapter.build(data, 4)))
    assert counts[0] == counts[1], counts


def test_lookahead_frames_are_not_recomputed_per_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trajectory-frame goal features are cached; only the live state is recomputed."""
    model = _toy_model(4)
    clip = _write_clip(model, tmp_path / "clip.npz")
    calls: list[int] = []
    relative_site_quantities = fbl._relative_site_quantities

    def _counting(**kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calls.append(1)
        return relative_site_quantities(**kwargs)

    monkeypatch.setattr(fbl, "_relative_site_quantities", _counting)
    concise = FullbodyObsAdapter(
        model, clip, {"sites_for_mimic": _SITES, "use_concise_lookahead": True}
    )
    full = FullbodyObsAdapter(
        model, clip, {"sites_for_mimic": _SITES, "n_step_lookahead": 3}
    )
    data = mujoco.MjData(model)
    _set_state(model, data, clip, 5, 0)

    def _calls_of(fn: Callable[[], object]) -> int:
        calls.clear()
        fn()
        return len(calls)

    assert _calls_of(lambda: concise.build(data, 5)) == 1
    assert _calls_of(lambda: concise.build(data, 6)) == 1
    assert _calls_of(lambda: full.build(data, 5)) == 1 + 3
    assert _calls_of(lambda: full.build(data, 6)) == 1 + 1
    assert _calls_of(lambda: full.build(data, 5)) == 1
