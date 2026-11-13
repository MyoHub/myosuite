# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Steering a myoSuperMimic tracker in MyoSuite's CPU envs (directional walking, ChaseTag).

The tracker observes MuscleMimic's observation in world axes: the root quaternion, the site offsets from the pelvis
and the reference's displacements. It is not invariant to heading, so a reference turned away from the headings of
the training clips is out of its distribution. Steering here keeps the reference clip as trained and turns the
robot's *observed* state by ``-delta`` about the vertical axis instead: the policy sees itself in the clip's frame.
On flat ground the physics is invariant to that turn, so the robot walks the clip's path turned by ``delta`` in the
world, and a change of ``delta`` shows the policy a heading error, which it corrects.

* :class:`TrackerPolicy`: a checkpoint's deployed policy with the task's own observation terms, on states;
* :func:`body_index` / :func:`read_state`: the tracker's state of one body in a host model (a single body, or one
  prefixed ChaseTag player), with the foot touch the task's contact sensor reads (ground contacts only);
* :class:`HeadingSteer`: ``delta`` from a world heading command, turned at a bounded rate;
* :class:`TrackerAgent`: one body of a host env, driven by a tracker and steered by a world heading;
* :func:`loop_clip`: a walking clip extended by repeating one gait cycle, as a canonical clip file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from myosuite.biomechanics.retarget import REFIT_META_KEY, read_refit_meta
from myosuite.core.model_names import resolve_model_names
from myosuite.envs.myo.tasks.mimic.tracking import resolve_clips
from myosuite.envs.myo.tasks.mimic.tracking.config import resolve_task
from myosuite.envs.myo.tasks.mimic.tracking.models import (
    DEFAULT_MODEL,
    ModelInfo,
    compiled_body,
    kinematic_fingerprint,
    model_cfg,
    model_version,
    root_offset,
)
from myosuite.envs.myo.tasks.mimic.tracking.motion import (
    MotionLib,
    build_motion_lib,
    load_clip,
    motion_files,
)

from .env import TIMESTEP, settings_cfg, task_overrides
from .heads import StepCtx, TrackingHeads, _goal, _observation
from .kinematics import SkeletonKinematics
from .muscle_geometry import MuscleGeometry
from .networks import ActionMap, Policy
from .rotations import mat_to_quat, mm3, mv3, quat_to_mat

BODY = model_cfg(DEFAULT_MODEL)
TRACKED_BODIES = BODY.tracked_bodies
FOOT_BODIES = BODY.foot_bodies

State = tuple[torch.Tensor, ...]
"""``(p, R, v, w, q, qd, act, touch)`` with a leading batch dimension (see :mod:`.heads`)."""


def _wrap(x: torch.Tensor | np.ndarray | float):
    """Angle(s) wrapped to [-pi, pi)."""
    return (x + math.pi) % (2 * math.pi) - math.pi


def _yaw(R: torch.Tensor) -> torch.Tensor:
    """Heading of the canonical root frame's x (forward) axis."""
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def _rz(yaw: torch.Tensor) -> torch.Tensor:
    """Rotations about z, (..., 3, 3)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(yaw.shape + (3, 3))


def turn_state(S: State, delta: torch.Tensor) -> State:
    """The state turned by ``-delta`` about the vertical axis: world quantities (position, rotation, linear and angular
    velocity) rotate, joint coordinates, activations and touch stay. ``delta`` is (B,)."""
    Rz = _rz(-delta.to(S[0].dtype))
    p, R, v, w = S[:4]
    return (mv3(Rz, p), mm3(Rz, R), mv3(Rz, v), mv3(Rz, w)) + tuple(S[4:])


def stack_states(states: list[State]) -> State:
    """Batch states of single bodies (each with batch 1) along the batch dimension."""
    return tuple(torch.cat(parts, 0) for parts in zip(*states, strict=True))


class TrackerPolicy:
    """A myoSuperMimic checkpoint's deployed policy, observing states the way its mjlab task does.

    The kinematics, the reference library and the observation terms are the task's (``myoMimic-v0`` ``supermimic``
    preset with the checkpoint's settings), built on the body the tracker was trained on, so the observation equals the
    runner's analytic observation (and the mjlab task's up to the coupled joints' constraint violation).

    Args:
        checkpoint: A ``model_*.pt`` of the SuperMimic runner.
        motion_file: The reference clip(s) to track: a clip, a directory or a ``.txt`` list (see
            :func:`myosuite.envs.myo.tasks.mimic.tracking.motion.motion_files`).
        device: Torch device.
        model: The tracker's body model (default: ``compiled_body("myofullbody")``).
        motion_root: The directory a list's paths start from.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        motion_file: str | Path,
        device: str = "cpu",
        model: mujoco.MjModel | None = None,
        motion_root: str = "",
    ) -> None:
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        env, cfg = ck["env"], ck["cfg"]
        self.device = device
        self.model = model if model is not None else compiled_body(DEFAULT_MODEL)
        self.kin = SkeletonKinematics(TRACKED_BODIES, device, self.model)
        self.info: ModelInfo = self.kin.info
        self.decimation = int(env["decimation"])
        self.dt = round(TIMESTEP * self.decimation, 9)
        files = motion_files(
            *resolve_clips(str(motion_file), motion_root, DEFAULT_MODEL)
        )
        self.motion = build_motion_lib(files, self.model, TRACKED_BODIES, self.dt).map(
            lambda x: torch.as_tensor(x, device=device)
        )
        muscle_obs = bool(env["muscle_obs"])
        geo = MuscleGeometry(self.model, device) if muscle_obs else None
        self.heads = TrackingHeads(
            self.kin, self.motion, geo, int(env["lookahead"]), int(env["stride"])
        )
        self.task = resolve_task(
            "supermimic",
            task_overrides(
                settings_cfg(env, motion_file=str(motion_file), motion_root=motion_root)
            ),
        )
        saved_obs = (ck.get("contract") or {}).get("obs", {}).get("policy")
        self._policy_terms = (
            tuple(str(n) for n, _ in saved_obs) if saved_obs else self.task.obs.policy
        )
        nm = self.info.num_muscles
        self.pi = Policy(
            int(env["obs_dim"]),
            int(env["goal_dim"]),
            nm,
            tuple(cfg["enc_hidden"]),
            tuple(cfg["dec_hidden"]),
            int(cfg["latent_dim"]),
            float(cfg["init_excitation"]),
            action_map=ActionMap.from_spec(ck.get("action_map"), nm),
        ).to(device)
        self.pi.load_state_dict(ck["pi"])
        self.pi.eval()
        self.iteration = int(ck.get("iteration", 0))
        self.wall_time = float(ck.get("wall_time", 0.0))
        self.sim_steps = int(ck.get("sim_steps", 0))

    def observe(
        self, S: State, f: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The policy and goal observations of state ``S`` against frame ``f``; ``a`` is the action that led to it."""
        ctx = StepCtx(S, f, a)
        obs = torch.cat(
            [_observation(self.heads, ctx, name=n) for n in self._policy_terms], -1
        )
        goal = _goal(self.heads, ctx)
        return obs, goal

    @torch.no_grad()
    def act(self, S: State, f: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Excitations in [0, 1], in the tracker's muscle (actuator) order, (B, num_muscles)."""
        return self.pi(*self.observe(S, f, a))


@dataclass(frozen=True)
class BodyIndex:
    """Where one tracker body lives in a host model (matched by name, with an optional prefix)."""

    root_q: int
    root_v: int
    q_adr: np.ndarray
    v_adr: np.ndarray
    joint_q_adr: np.ndarray
    """Every non-root joint (coupled ones too), in :attr:`ModelInfo.joint_names` order."""
    joint_v_adr: np.ndarray
    actuators: np.ndarray
    """Host actuator ids, in the tracker's muscle order."""
    act_adr: np.ndarray
    feet: tuple[int, ...]
    """Host body ids of the foot touch bodies, in the task's sensor order (:data:`FOOT_BODIES`)."""
    tracked: tuple[int, ...]
    """Host body ids of the tracked bodies (:data:`TRACKED_BODIES`)."""
    ground: frozenset[int]
    """Host geom ids of the ground: planes and height fields fixed to the world."""


def body_index(host: mujoco.MjModel, info: ModelInfo, prefix: str = "") -> BodyIndex:
    """Match the tracker's joints, muscles and feet to a host model by name.

    The musclemimic_models and myo_sim full bodies name a few joints and muscles differently (``elbow_flex_r`` /
    ``elbow_flexion_r``, ``DELT1_left`` / ``DELT1_l``); :func:`myosuite.core.model_names.resolve_model_names`
    resolves those.

    Args:
        host: The host model (e.g. a directional env's body, or the two-player ChaseTag scene).
        info: The tracker body's :class:`ModelInfo`.
        prefix: Name prefix of the body in the host (``"a0_"`` / ``"a1_"`` in ChaseTag).

    Returns:
        The addresses.

    Raises:
        KeyError: If a joint, actuator or foot body is missing in the host.
    """

    def ids(kind: mujoco.mjtObj, names: list[str] | tuple[str, ...]) -> np.ndarray:
        count = {
            mujoco.mjtObj.mjOBJ_JOINT: host.njnt,
            mujoco.mjtObj.mjOBJ_ACTUATOR: host.nu,
            mujoco.mjtObj.mjOBJ_BODY: host.nbody,
        }[kind]
        pool = {
            n[len(prefix) :]
            for n in (mujoco.mj_id2name(host, kind, i) for i in range(count))
            if n and n.startswith(prefix)
        }
        resolved = resolve_model_names(names, pool, field_name=kind.name)
        out = np.array([mujoco.mj_name2id(host, kind, prefix + n) for n in resolved])
        missing = [n for n, i in zip(resolved, out, strict=True) if i < 0]
        if missing:
            raise KeyError(
                f"host model lacks {len(missing)} {kind.name} names, e.g. {prefix}{missing[0]}"
            )
        return out

    root = int(ids(mujoco.mjtObj.mjOBJ_JOINT, [info.root_joint_name])[0])
    joints = ids(mujoco.mjtObj.mjOBJ_JOINT, info.indep_names)
    every = ids(mujoco.mjtObj.mjOBJ_JOINT, info.joint_names)
    actuators = ids(mujoco.mjtObj.mjOBJ_ACTUATOR, info.muscle_names)
    feet = ids(mujoco.mjtObj.mjOBJ_BODY, FOOT_BODIES)
    tracked = ids(mujoco.mjtObj.mjOBJ_BODY, TRACKED_BODIES)
    ground = frozenset(
        g
        for g in range(host.ngeom)
        if host.body_weldid[host.geom_bodyid[g]] == 0
        and host.geom_type[g]
        in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)
    )
    return BodyIndex(
        root_q=int(host.jnt_qposadr[root]),
        root_v=int(host.jnt_dofadr[root]),
        q_adr=host.jnt_qposadr[joints].astype(int),
        v_adr=host.jnt_dofadr[joints].astype(int),
        joint_q_adr=host.jnt_qposadr[every].astype(int),
        joint_v_adr=host.jnt_dofadr[every].astype(int),
        actuators=actuators.astype(int),
        act_adr=host.actuator_actadr[actuators].astype(int),
        feet=tuple(int(b) for b in feet),
        tracked=tuple(int(b) for b in tracked),
        ground=ground,
    )


def foot_touch(host: mujoco.MjModel, data: mujoco.MjData, idx: BodyIndex) -> np.ndarray:
    """Normal force of the ground on each foot body (the task's contact sensor: net force, its |z|). Needs the
    constraint forces of ``data`` (after ``mj_forward`` or ``mj_step``)."""
    net = np.zeros((len(idx.feet), 3))
    slot = {b: k for k, b in enumerate(idx.feet)}
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(host.geom_bodyid[c.geom1]), int(host.geom_bodyid[c.geom2])
        if c.geom1 in idx.ground and b2 in slot:
            k, sign = slot[b2], 1.0
        elif c.geom2 in idx.ground and b1 in slot:
            k, sign = slot[b1], -1.0
        else:
            continue
        mujoco.mj_contactForce(host, data, i, f6)
        net[k] += sign * (c.frame.reshape(3, 3).T @ f6[:3])
    return np.abs(net[:, 2])


def read_state(
    host: mujoco.MjModel,
    data: mujoco.MjData,
    idx: BodyIndex,
    R_off: np.ndarray,
    device: str = "cpu",
) -> State:
    """The tracker's state ``S`` (batch 1) of one host body, as ``TaskView.read_state``."""
    q, v = data.qpos, data.qvel
    Rb = np.zeros(9)
    mujoco.mju_quat2Mat(Rb, q[idx.root_q + 3 : idx.root_q + 7])
    Rb = Rb.reshape(3, 3)
    parts = (
        q[idx.root_q : idx.root_q + 3],
        Rb @ R_off,
        v[idx.root_v : idx.root_v + 3],
        Rb @ v[idx.root_v + 3 : idx.root_v + 6],
        q[idx.q_adr],
        v[idx.v_adr],
        data.act[idx.act_adr],
        foot_touch(host, data, idx),
    )
    return tuple(
        torch.tensor(np.asarray(x), dtype=torch.float32, device=device)[None]
        for x in parts
    )


def write_reference_state(
    host: mujoco.MjModel,
    data: mujoco.MjData,
    idx: BodyIndex,
    tracker: TrackerPolicy,
    frame: int,
    activation: float = 0.0,
) -> None:
    """Put one host body on reference frame ``frame`` of the tracker's library (pose, velocities, every joint with
    the coupled ones on their polynomials, activations), as the task's reset does, then ``mj_forward``."""
    fr, kin = tracker.motion.frames, tracker.kin
    f = torch.tensor([frame], device=kin.R_off.device)
    cols = torch.as_tensor(kin.info.indep_cols, device=f.device)
    q = torch.clamp(fr["joint_pos"][f][:, cols], kin.q_lo, kin.q_hi)
    qd = fr["joint_vel"][f][:, cols]
    q_full, qd_full = kin.full_joint_state(q, qd)
    Rb = fr["root_rot"][frame].double().cpu().numpy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, Rb.ravel())
    data.qpos[idx.root_q : idx.root_q + 7] = np.concatenate(
        (fr["root_pos"][frame].double().cpu().numpy(), quat)
    )
    data.qvel[idx.root_v : idx.root_v + 3] = (
        fr["root_qvel"][frame, :3].double().cpu().numpy()
    )
    data.qvel[idx.root_v + 3 : idx.root_v + 6] = (
        fr["root_qvel"][frame, 3:6].double().cpu().numpy()
    )
    data.qpos[idx.joint_q_adr] = q_full[0].double().cpu().numpy()
    data.qvel[idx.joint_v_adr] = qd_full[0].double().cpu().numpy()
    data.act[idx.act_adr] = activation
    mujoco.mj_forward(host, data)


class HeadingSteer:
    """``delta``, the yaw from the clip's frame to the world, steered toward a world heading command.

    The robot follows the clip, so its world travel heading is the clip's travel heading at frame f plus ``delta``.
    The clip's travel heading is its root's facing yaw averaged over ``window_s`` (one gait cycle cancels the
    pelvis's sway), plus the mean angle between travel and facing over the frames where it moves.

    Args:
        motion: The tracker's reference library (one clip).
        max_rate: Largest turn of ``delta`` per second (rad/s).
        window_s: Averaging window of the facing yaw.
        moving_speed: Root speed above which a frame counts as moving.
        R_off: Root canonical-frame offset (default: ``DEFAULT_MODEL``'s).
    """

    def __init__(
        self,
        motion: MotionLib,
        max_rate: float = 1.0,
        window_s: float = 1.0,
        moving_speed: float = 0.3,
        R_off: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        self.max_step = max_rate * motion.dt
        fr = motion.frames
        dev = fr["root_pos"].device
        off = torch.as_tensor(
            root_offset(compiled_body(DEFAULT_MODEL)) if R_off is None else R_off,
            dtype=fr["root_rot"].dtype,
            device=dev,
        )
        root_R = fr["root_rot"] @ off
        yaw = np.unwrap(_yaw(root_R).double().cpu().numpy())
        half = max(1, round(0.5 * window_s / motion.dt))
        pad = np.pad(yaw, half, mode="edge")
        kernel = np.ones(2 * half + 1) / (2 * half + 1)
        facing = np.convolve(pad, kernel, mode="valid")
        v = fr["root_qvel"][:, :2].double().cpu().numpy()
        moving = np.linalg.norm(v, axis=1) > moving_speed
        offset = 0.0
        if moving.any():
            d = _wrap(np.arctan2(v[moving, 1], v[moving, 0]) - facing[moving])
            offset = float(np.arctan2(np.sin(d).mean(), np.cos(d).mean()))
        self.travel = torch.tensor(facing + offset, dtype=torch.float32, device=dev)

    def target(self, f: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        """The ``delta`` that points the clip's travel heading at frame f along ``heading``."""
        return _wrap(heading - self.travel[f])

    def step(
        self, delta: torch.Tensor, f: torch.Tensor, heading: torch.Tensor
    ) -> torch.Tensor:
        """``delta`` turned toward :meth:`target` by at most ``max_rate * dt``."""
        turn = _wrap(self.target(f, heading) - delta).clamp(
            -self.max_step, self.max_step
        )
        return _wrap(delta + turn)


def _inverse_sigmoid_activation(e: np.ndarray) -> np.ndarray:
    """The action whose ``sigmoid(5 (a - 0.5))`` (MyoSuite's muscle action mapping) is the excitation ``e``."""
    e = np.clip(e, 1e-6, 1.0 - 1e-6)
    return 0.5 + np.log(e / (1.0 - e)) / 5.0


class TrackerAgent:
    """One body of a host env driven by a :class:`TrackerPolicy` and steered by a world heading.

    The tracker acts every ``hold`` host control steps (its own control step over the host's) and holds its
    excitations in between; its reference frame advances once per action.

    Args:
        tracker: The policy.
        host: The host model.
        host_dt: The host env's control step (s).
        prefix: The body's name prefix in the host.
        max_rate: The steer's largest turn rate (rad/s).
    """

    def __init__(
        self,
        tracker: TrackerPolicy,
        host: mujoco.MjModel,
        host_dt: float,
        prefix: str = "",
        max_rate: float = 1.0,
    ) -> None:
        if len(tracker.motion.names) != 1:
            raise ValueError(
                "an agent steers along one clip: build its TrackerPolicy on a single clip"
            )
        self.tracker, self.host = tracker, host
        self.idx = body_index(host, tracker.info, prefix)
        self.R_off = tracker.kin.R_off.double().cpu().numpy()
        self.hold = round(tracker.dt / host_dt)
        if self.hold < 1 or abs(self.hold * host_dt - tracker.dt) > 1e-9:
            raise ValueError(
                f"the tracker's step {tracker.dt} s is not a multiple of the host's {host_dt} s"
            )
        self.steer = HeadingSteer(
            tracker.motion, max_rate=max_rate, R_off=tracker.kin.R_off
        )
        dev = tracker.device
        self.reset_frame = 0
        self.f = torch.zeros(1, dtype=torch.long, device=dev)
        self.delta = torch.zeros(1, device=dev)
        self.a = torch.zeros(1, tracker.info.num_muscles, device=dev)
        self._t = 0

    def state(self, data: mujoco.MjData) -> State:
        return read_state(self.host, data, self.idx, self.R_off, self.tracker.device)

    def match_frame(self, data: mujoco.MjData, first: int = 0) -> int:
        """The reference frame (from ``first``, keeping the lookahead inside the clip) nearest the body's joint
        positions and velocities: a phase match independent of position and heading."""
        S = self.state(data)
        m, h = self.tracker.motion, self.tracker.heads
        last = int(m.clip_end[0]) - h.stride * (h.lookahead - 1)
        cols = torch.as_tensor(self.tracker.info.indep_cols, device=h.q.device)
        q = h.q[first:last]
        qd = m.frames["joint_vel"][first:last][:, cols]
        cost = ((q - S[4]) ** 2).sum(-1) + 0.01 * ((qd - S[5]) ** 2).sum(-1)
        return first + int(cost.argmin())

    def reset(
        self, data: mujoco.MjData, frame: int | None = None, first: int = 0
    ) -> None:
        """Start an episode from the body's current state: reference frame ``frame`` (default: the phase match from
        ``first``), and ``delta`` so that the body faces the clip's heading at that frame."""
        f = self.match_frame(data, first) if frame is None else int(frame)
        S = self.state(data)
        dev = self.tracker.device
        self.reset_frame = f
        self.f = torch.tensor([f], dtype=torch.long, device=dev)
        self.delta = _wrap(_yaw(S[1]) - _yaw(self.tracker.heads.root_R[self.f]))
        self.a = torch.zeros(1, self.tracker.info.num_muscles, device=dev)
        self._t = 0

    def excitation(self, data: mujoco.MjData, heading: float) -> np.ndarray:
        """The excitations for this host step (tracker muscle order), acting on a new state every ``hold`` steps."""
        if self._t % self.hold == 0:
            if self._t > 0:
                self.f = torch.minimum(self.f + 1, self.tracker.motion.clip_end[0])
            self.delta = self.steer.step(
                self.delta,
                self.f,
                torch.tensor(
                    [heading], dtype=torch.float32, device=self.tracker.device
                ),
            )
            S = turn_state(self.state(data), self.delta)
            self.a = self.tracker.act(S, self.f, self.a)
        self._t += 1
        return self.a[0].cpu().numpy().astype(np.float64)

    def fk_mismatch_mm(self, data: mujoco.MjData) -> float:
        """Mean distance (mm) between the tracked bodies' pelvis-relative positions in the host (``data.xpos``, after
        ``mj_forward``) and in the tracker's kinematics at the same joint angles: 0 on the training body, the body
        mismatch on another one."""
        S = self.state(data)
        with torch.no_grad():
            P = self.tracker.kin.forward(*S[:6])[0][0].double().cpu().numpy()
        H = data.xpos[list(self.idx.tracked)]
        return float(1e3 * np.linalg.norm((H - H[0]) - (P - P[0]), axis=-1).mean())

    def frames_left(self) -> int:
        """Reference frames before the lookahead reaches the clip's end."""
        h = self.tracker.heads
        return (
            int(self.tracker.motion.clip_end[0])
            - h.stride * (h.lookahead - 1)
            - int(self.f[0])
        )

    def host_action(
        self, data: mujoco.MjData, heading: float, sigmoid: bool = False
    ) -> np.ndarray:
        """A full host action vector (every host actuator; the others 0). ``sigmoid``: the host maps actions through
        MyoSuite's ``sigmoid(5 (a - 0.5))`` (the MuscleMimic CPU envs), so the inverse is applied to the excitations."""
        e = self.excitation(data, heading)
        out = np.zeros(self.host.nu)
        out[self.idx.actuators] = _inverse_sigmoid_activation(e) if sigmoid else e
        return out


def _heading_frame_features(
    p: np.ndarray, R: np.ndarray, q: np.ndarray, fps: float
) -> np.ndarray:
    """Per-frame features independent of position and heading, for matching gait phases: joints, joint velocities,
    root height and tilt, and the root velocity in the heading frame."""
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    v = np.gradient(p, 1.0 / fps, axis=0)
    v_h = np.stack((c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]), -1)
    tilt = np.stack((R[:, 2, 0], R[:, 2, 1], R[:, 0, 2], R[:, 1, 2]), -1)
    qd = np.gradient(q, 1.0 / fps, axis=0)
    return np.concatenate((q, 0.1 * qd, p[:, 2:3], tilt, 0.3 * v_h), -1)


def loop_clip(
    src: str | Path,
    dst: str | Path,
    info: ModelInfo,
    R_off: np.ndarray,
    seconds: float = 15.0,
    search_from_s: float = 0.5,
    cycle_s: tuple[float, float] = (0.8, 1.6),
    blend_s: float = 0.1,
) -> dict:
    """Extend a walking clip by repeating one gait cycle, and write it as a canonical clip.

    The cycle is the pair of frames ``i < j`` (``j - i`` within ``cycle_s``, ``i`` after ``search_from_s``) whose
    joints, joint velocities, root height, tilt and heading-frame root velocity match best. The output keeps frames
    ``[0, i)`` (the clip's start, e.g. standing) and then repeats ``[i, j)``, each repetition moved by the planar
    rigid transform that takes frame ``i``'s root onto frame ``j``'s (so the walk continues where the last cycle
    ended, turning by the cycle's net yaw). The joint and height mismatch at each seam is blended in over
    ``blend_s`` before it.

    Args:
        src: A clip file (MuscleMimic trajectory or canonical, see :mod:`myosuite.envs.myo.tasks.mimic.tracking.motion`).
        dst: The ``.npz`` to write (canonical format).
        info: The tracker body's :class:`ModelInfo`.
        R_off: The tracker body's root offset (``SkeletonKinematics.R_off``).
        seconds: Least duration of the output.
        search_from_s: Earliest start of the cycle.
        cycle_s: Shortest and longest cycle.
        blend_s: Seam blend duration.

    Returns:
        The cycle: its frames ``i`` and ``j``, ``period_s``, the match ``cost``, the net ``turn_deg`` per cycle, the
        mean planar ``speed`` over it, and the output ``frames`` and ``fps``.
    """
    fps, p, quat, q = load_clip(str(src), info, R_off)
    R = quat_to_mat(torch.tensor(quat, dtype=torch.float64)).numpy() @ R_off
    feats = _heading_frame_features(p, R, q, fps)
    T = len(p)
    lo, hi = round(cycle_s[0] * fps), round(cycle_s[1] * fps)
    best = (math.inf, 0, 0)
    for i in range(round(search_from_s * fps), T - lo):
        js = np.arange(i + lo, min(i + hi, T - 1) + 1)
        if len(js) == 0:
            continue
        cost = ((feats[js] - feats[i]) ** 2).mean(-1)
        k = int(cost.argmin())
        if cost[k] < best[0]:
            best = (float(cost[k]), i, int(js[k]))
    cost, i, j = best
    if not math.isfinite(cost):
        raise ValueError(
            f"{src}: too short for a gait cycle of {cycle_s} s after {search_from_s} s"
        )
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    turn = float(_wrap(yaw[j] - yaw[i]))
    G = _rz(torch.tensor(turn, dtype=torch.float64)).numpy()
    t = p[j, :2] - G[:2, :2] @ p[i, :2]
    dq, dz = q[i] - q[j], p[i, 2] - p[j, 2]
    w = max(1, round(blend_s * fps))
    ramp = np.zeros(j - i)
    ramp[-w:] = np.arange(1, w + 1) / (w + 1)
    cyc_p, cyc_R, cyc_q = p[i:j].copy(), R[i:j].copy(), q[i:j] + ramp[:, None] * dq
    cyc_p[:, 2] += ramp * dz
    out_p, out_R, out_q = [p[:i]], [R[:i]], [q[:i]]
    Gk, tk = np.eye(3), np.zeros(2)
    n = i
    while n < seconds * fps:
        pk = cyc_p.copy()
        pk[:, :2] = cyc_p[:, :2] @ Gk[:2, :2].T + tk
        out_p.append(pk), out_R.append(Gk @ cyc_R), out_q.append(cyc_q)
        n += j - i
        Gk, tk = G @ Gk, G[:2, :2] @ tk + t
    out_p, out_R, out_q = (np.concatenate(x) for x in (out_p, out_R, out_q))
    refit = read_refit_meta(str(src)) or {
        "source": "loop_clip",
        "model": DEFAULT_MODEL,
        "model_version": model_version(DEFAULT_MODEL),
        "model_fingerprint": kinematic_fingerprint(DEFAULT_MODEL),
    }
    np.savez(
        dst,
        fps=fps,
        root_pos=out_p,
        root_quat=mat_to_quat(torch.from_numpy(out_R)).numpy(),
        joint_pos=out_q,
        joint_names=np.array(info.indep_names),
        **{REFIT_META_KEY: np.array(json.dumps(refit))},
    )
    speed = float(np.linalg.norm(p[j, :2] - p[i, :2]) / ((j - i) / fps))
    return {
        "i": i,
        "j": j,
        "period_s": (j - i) / fps,
        "cost": cost,
        "turn_deg": math.degrees(turn),
        "speed": speed,
        "frames": len(out_p),
        "fps": fps,
    }
