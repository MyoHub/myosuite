# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Directional walking on ``myofullbody`` (tutorial 5.4): scene, observation, steered teacher, student, evaluation.

The walker reads a 528-d **directional observation** of one ``myofullbody`` body:

==========  ====  ===========================================================================
slice       size  content
==========  ====  ===========================================================================
0:82        82    joint positions (``qpos[7:]``)
82:164      82    joint velocities (``qvel[6:]``)
164:518     354   muscle activations
518:520     2     root xy velocity in the yaw frame
520:522     2     commanded heading in the yaw frame, ``[cos(theta - yaw), sin(theta - yaw)]``
522:528     6     roll, pitch, root angular velocity (body frame), root z velocity
==========  ====  ===========================================================================

Every entry is unchanged when the scene is rotated about the vertical axis, so the walker steers from the relative
heading alone. :class:`ChaseTagHeadingAdapter` builds the same observation from a ChaseTag FBVs player's 546-d
observation, with the heading toward the opponent (chaser) or away from it (runner) and turned off the fence, so a
walker exports as a scored-game player (:func:`export_chasetag_onnx`).

This module needs only torch, numpy, MuJoCo and MyoSuite at import time; the teacher (:class:`HeadingTeacher`) and
the scene need the rest of the project (myoSuperMimic) when used.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from myosuite.envs.myo.tasks.mimic.policy import ActorCritic

DIR_OBS_DIM, ACT_DIM, FBVS_OBS_DIM = 528, 354, 546
HEADING_SLICE = slice(520, 522)
"""The commanded heading in the directional observation."""
COMPASS = {
    "E": 0,
    "NE": 45,
    "N": 90,
    "NW": 135,
    "W": 180,
    "SW": 225,
    "S": 270,
    "SE": 315,
}
"""The eight headings of tutorial 5.4's evaluation (degrees, world frame)."""
WALK_SPEED = 1.2
"""Speed (m/s) the teacher walks at."""
FALL_Z = 0.6
"""Root height (m) below which the walker has fallen."""
ARENA_LIM, WALL_MARGIN, WALL_WEIGHT = 6.0, 2.0, 2.5
"""ChaseTag fence (m from the centre), its repulsion band (m) and weight, as the steered teacher's planner."""


# ---------------------------------------------------------------- observation


def _quat_yaw_roll_pitch(q: torch.Tensor) -> tuple[torch.Tensor, ...]:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    return yaw, roll, pitch


def directional_obs_torch(qpos: torch.Tensor, qvel: torch.Tensor, act: torch.Tensor,
                          heading: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """The directional observation of a batch of bodies.

    Args:
        qpos: (B, 89) body joint positions (free root first).
        qvel: (B, 88) body joint velocities.
        act: (B, 354) muscle activations.
        heading: (B, 2) world-frame unit heading.

    Returns:
        (B, 528) observations.
    """
    yaw, roll, pitch = _quat_yaw_roll_pitch(qpos[:, 3:7])
    c, s = torch.cos(yaw), torch.sin(yaw)
    vx, vy = qvel[:, 0], qvel[:, 1]
    hx, hy = heading[:, 0], heading[:, 1]
    tail = torch.stack((c * vx + s * vy, -s * vx + c * vy, c * hx + s * hy, -s * hx + c * hy, roll, pitch,
                        qvel[:, 3], qvel[:, 4], qvel[:, 5], qvel[:, 2]), -1)  # fmt: skip
    return torch.cat((qpos[:, 7:89], qvel[:, 6:88], act, tail), -1)


def directional_obs(
    qpos: np.ndarray, qvel: np.ndarray, act: np.ndarray, theta: float
) -> np.ndarray:
    """The directional observation (528,) of one body for the world heading ``theta`` (rad), float32."""
    t = lambda x: torch.as_tensor(np.asarray(x, np.float64))[None]  # noqa: E731
    h = t([math.cos(theta), math.sin(theta)])
    return (
        directional_obs_torch(t(qpos), t(qvel), t(act), h)[0].numpy().astype(np.float32)
    )


def _wall_push(xy: torch.Tensor) -> torch.Tensor:
    edge = ARENA_LIM - WALL_MARGIN
    return (
        torch.clamp(-edge - xy, min=0.0) - torch.clamp(xy - edge, min=0.0)
    ) / WALL_MARGIN


def _drop_outward(xy: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    edge = ARENA_LIM - WALL_MARGIN
    v = torch.where(xy > edge, torch.clamp(v, max=0.0), v)
    return torch.where(xy < -edge, torch.clamp(v, min=0.0), v)


def _unit_or(v: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    n = torch.linalg.norm(v, dim=-1, keepdim=True)
    return torch.where(n > 1e-6, v / n.clamp_min(1e-6), fallback)


def role_heading_torch(
    me_xy: torch.Tensor, rel_xy: torch.Tensor, runner: torch.Tensor
) -> torch.Tensor:
    """Unit heading toward the opponent (chaser) or away (runner), turned off the fence.

    The batched form of :func:`myosuite.integrations.musclemimic.steering.aim_heading_xy` (same arena constants).

    Args:
        me_xy: (B, 2) own position.
        rel_xy: (B, 2) opponent position minus own.
        runner: (B, 1) 1 for the runner, 0 for the chaser.
    """
    role = torch.where(runner > 0.5, -rel_xy, rel_xy)
    east = torch.stack(
        (torch.ones_like(role[:, 0]), torch.zeros_like(role[:, 0])), -1
    )  # opponent on top of us
    role = _unit_or(role, east)
    inward = _wall_push(me_xy)
    v = _drop_outward(me_xy, role + WALL_WEIGHT * inward)
    along = _drop_outward(me_xy, role)
    return _unit_or(v, _unit_or(along, _unit_or(inward, role)))


class ChaseTagHeadingAdapter(nn.Module):
    """FBVs player observation (546) -> directional observation (528), heading from the role and the opponent.

    FBVs layout: ``qpos`` 0:89, ``qvel`` 89:177, ``act`` 177:531, own pelvis position and velocity 531:537,
    opponent pelvis relative position and velocity and distance 537:544, role 544:546 (``[1, 0]`` chaser,
    ``[0, 1]`` runner).
    """

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        heading = role_heading_torch(obs[:, 531:533], obs[:, 537:539], obs[:, 545:546])
        return directional_obs_torch(
            obs[:, 0:89], obs[:, 89:177], obs[:, 177:531], heading
        )


class WalkerNet(nn.Module):
    """A walker's mean action in [0, 1]: ``obs (B, 528) -> excitations (B, 354)``."""

    def __init__(self, net: ActorCritic):
        super().__init__()
        self.trunk, self.actor_mean = net.trunk, net.actor_mean

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Excitations (B, 354)."""
        return torch.clamp(self.actor_mean(self.trunk(obs)), 0.0, 1.0)


def load_walker(path: str | Path) -> WalkerNet:
    """A walker state dict (``ActorCritic`` 528 -> 354, normalisation folded in) as a :class:`WalkerNet`."""
    return WalkerNet(
        ActorCritic.load(path, obs_dim=DIR_OBS_DIM, act_dim=ACT_DIM)
    ).eval()


def export_chasetag_onnx(walker: str | Path, path: str | Path) -> Path:
    """ONNX ``obs [batch, 546] -> action [batch, 354]``: the walker behind :class:`ChaseTagHeadingAdapter`.

    Args:
        walker: The walker state dict (``ActorCritic`` 528 -> 354).
        path: The ONNX file to write.

    Returns:
        ``path``.
    """
    net = nn.Sequential(ChaseTagHeadingAdapter(), load_walker(walker)).eval()
    torch.onnx.export(net, torch.zeros(1, FBVS_OBS_DIM), str(path), input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17,
                      do_constant_folding=True, dynamo=False)  # fmt: skip
    return Path(path)


# ---------------------------------------------------------------- scene


class DirectionalScene:
    """One ``myofullbody`` on flat ground (``myoMimic-v0``'s scene), stepped at 100 Hz like the FBVs players.

    Args:
        frame_skip: Physics steps per control step (0.002 s each).
    """

    def __init__(self, frame_skip: int = 5):
        import mujoco  # noqa: PLC0415

        from myosuite.envs.myo.tasks.mimic.tracking.models import compiled_scene  # noqa: PLC0415
        from myosuite.physics.joint_coupling import JointCouplings  # noqa: PLC0415

        self.mj = mujoco
        self.model = compiled_scene("myofullbody")
        self.data = mujoco.MjData(self.model)
        self._couplings = JointCouplings.from_model(self.model)
        self.frame_skip = frame_skip
        self.dt = frame_skip * float(self.model.opt.timestep)
        self.nq, self.nv = self.model.nq, self.model.nv

    def reset(
        self, yaw: float, rng: np.random.Generator, joint_noise: float = 0.02
    ) -> None:
        """Standing pose (``qpos0``) turned to the facing direction ``yaw`` (rad, world), small joint noise."""
        from myosuite.physics.joint_coupling import project_joint_couplings  # noqa: PLC0415

        m, d = self.model, self.data
        self.mj.mj_resetData(m, d)
        q0 = m.qpos0.copy()
        rot = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        out = np.zeros(4)
        self.mj.mju_mulQuat(out, rot, q0[3:7])
        q0[3:7] = out
        q0[7:] += rng.normal(0.0, joint_noise, self.nq - 7)
        d.qpos[:] = q0
        project_joint_couplings(self._couplings, d.qpos, d.qvel)
        self.mj.mj_forward(m, d)

    def step(self, action: np.ndarray) -> None:
        """Apply excitations for one control step."""
        self.data.ctrl[:] = np.clip(action, 0.0, 1.0)
        self.mj.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.mj.mj_forward(self.model, self.data)

    def obs(self, theta: float) -> np.ndarray:
        """The directional observation for the world heading ``theta``."""
        d = self.data
        return directional_obs(d.qpos, d.qvel, d.act, theta)

    @property
    def fallen(self) -> bool:
        """The root dropped below :data:`FALL_Z`."""
        return float(self.data.qpos[2]) < FALL_Z

    @property
    def facing(self) -> float:
        """World yaw of the body's forward direction (the canonical root frame's x axis)."""
        from myosuite.envs.myo.tasks.mimic.tracking.models import root_offset  # noqa: PLC0415

        R = np.zeros(9)
        self.mj.mju_quat2Mat(R, self.data.qpos[3:7])
        Rc = R.reshape(3, 3) @ root_offset(self.model)
        return float(math.atan2(Rc[1, 0], Rc[0, 0]))


# ---------------------------------------------------------------- teacher


class HeadingTeacher:
    """The steered myoSuperMimic player (:class:`~chasetag_teacher.player.SuperMimicPlayer`) asked for a world
    heading and a speed instead of a ChaseTag role.

    Args:
        checkpoint: The tracker checkpoint.
        lib: The steering clip library (:func:`~chasetag_teacher.library.load_library`).
        scene: The scene it plays in.
        speed: Requested speed (m/s).
    """

    def __init__(
        self,
        checkpoint: str,
        lib: Any,
        scene: DirectionalScene,
        speed: float = WALK_SPEED,
    ):
        from .player import SuperMimicPlayer  # noqa: PLC0415
        from .reference import Steer  # noqa: PLC0415

        self._steer = Steer
        self.p = SuperMimicPlayer(checkpoint, lib, scene.model, "", scene.dt)
        self.p.steer = lambda me, opp, pursue: self._steer(self.theta, self.speed)
        self.scene, self.speed, self.theta = scene, speed, 0.0

    def reset(self, theta: float) -> None:
        """Start for the world heading ``theta`` (rad) from the scene's current state."""
        self.theta = theta
        self.p.reset(self.scene.data, None, None, True)

    def act(self) -> np.ndarray:
        """Excitations (354,) for the scene's current state and :attr:`theta`."""
        return self.p.act(self.scene.data, None, None, True)

    def set_prev_action(self, action: np.ndarray) -> None:
        """The excitation actually executed (the tracker observes its previous action)."""
        self.p.tracker.prev_action = np.asarray(action, np.float32)


# ---------------------------------------------------------------- DAgger episodes


@dataclass
class EpisodeCfg:
    """One DAgger episode: random start facing, heading and heading changes, executed by teacher or student."""

    steps: int = 1000
    changes: tuple[int, int] = (1, 2)
    noise: float = 0.03
    every: int = 2
    grace: int = 20
    takeover_z: float = 0.0


def heading_schedule(
    rng: np.random.Generator, cfg: EpisodeCfg
) -> list[tuple[int, float]]:
    """``[(start step, world heading)]``: a heading at step 0, then 1-2 random changes at least 2 s apart."""
    n = int(rng.integers(cfg.changes[0], cfg.changes[1] + 1))
    times = sorted(
        rng.choice(np.arange(200, cfg.steps - 100, 50), size=n, replace=False).tolist()
    )
    return [(0, float(rng.uniform(-math.pi, math.pi)))] + [(int(t), float(rng.uniform(-math.pi, math.pi)))
                                                           for t in times]  # fmt: skip


def collect_episode(scene: DirectionalScene, teacher: HeadingTeacher, student: Callable[[np.ndarray], np.ndarray] | None,
                    beta: float, seed: int, cfg: EpisodeCfg) -> dict:  # fmt: skip
    """One episode of (directional observation, teacher excitation) pairs.

    The student (``obs -> excitations``) executes with probability 1 - beta (per episode); in that case the teacher
    reads the executed excitation as its previous action, and takes over once the root drops below
    ``cfg.takeover_z``. Pairs are stored every ``cfg.every`` steps until ``cfg.grace`` steps after a fall.

    Returns:
        ``obs`` (N, 528), ``act`` (N, 354), ``fell`` (s or None), ``took`` (s or None), ``student`` (bool).
    """
    rng = np.random.default_rng(seed)
    use_student = student is not None and rng.random() > beta
    scene.reset(float(rng.uniform(-math.pi, math.pi)), rng)
    plan = heading_schedule(rng, cfg)
    teacher.reset(plan[0][1])
    X, Y = [], []
    fell = took = None
    down = 0
    k = 0
    for t in range(cfg.steps):
        while k + 1 < len(plan) and plan[k + 1][0] <= t:
            k += 1
        teacher.theta = plan[k][1]
        o = scene.obs(teacher.theta)
        y = teacher.act()
        if t % cfg.every == 0 and down <= cfg.grace:
            X.append(o)
            Y.append(y)
        x = student(o) if use_student and took is None else y
        x = np.clip(x + rng.normal(0.0, cfg.noise, x.shape), 0.0, 1.0)
        if use_student and took is None:
            teacher.set_prev_action(x)
        scene.step(x)
        if use_student and took is None and float(scene.data.qpos[2]) < cfg.takeover_z:
            took = t * scene.dt
        if fell is None and scene.fallen:
            fell = t * scene.dt
        down += fell is not None
        if down > cfg.grace:
            break
    return {"obs": np.array(X, np.float32), "act": np.array(Y, np.float32), "fell": fell, "took": took,
            "student": use_student}  # fmt: skip


# ---------------------------------------------------------------- evaluation


@dataclass
class DirectionalResult:
    """Tutorial 5.4's numbers for one heading: survival, mean steps alive, alignment of the displacement."""

    survival: float
    steps_alive: float
    alignment: float
    runs: list = field(default_factory=list)


def rollout(scene: DirectionalScene, policy: Callable[[np.ndarray], np.ndarray], theta: float, seed: int,
            steps: int, start_yaw: float = 0.0,
            record: Callable[[DirectionalScene], None] | None = None) -> tuple[bool, int, float]:  # fmt: skip
    """One rollout from standing, facing ``start_yaw``, commanded the world heading ``theta``.

    Returns:
        (survived, steps alive, cosine between the displacement and the heading).
    """
    rng = np.random.default_rng(seed)
    scene.reset(start_yaw, rng)
    start = scene.data.qpos[:2].copy()
    n = 0
    for _ in range(steps):
        scene.step(policy(scene.obs(theta)))
        n += 1
        if record is not None:
            record(scene)
        if scene.fallen:
            break
    disp = scene.data.qpos[:2] - start
    norm = float(np.linalg.norm(disp))
    align = (
        float(disp @ np.array([math.cos(theta), math.sin(theta)]) / norm)
        if norm > 1e-4
        else 0.0
    )
    return not scene.fallen, n, align


def evaluate(policy: Callable[[np.ndarray], np.ndarray], labels: list[str] | None = None, seeds: int = 10,
             steps: int = 200, scene: DirectionalScene | None = None) -> dict[str, DirectionalResult]:  # fmt: skip
    """Tutorial 5.4's protocol: every heading x ``seeds`` rollouts of ``steps`` control steps (100 Hz), facing east.

    Args:
        policy: ``obs (528,) -> excitations (354,)``.
        labels: Compass labels (default: all eight).
        seeds: Rollouts per heading (seed = joint noise of the start).
        steps: Control steps per rollout.
        scene: Reused scene (default: a new one).

    Returns:
        Per label, a :class:`DirectionalResult`.
    """
    scene = scene or DirectionalScene()
    out = {}
    for lab in labels or list(COMPASS):
        theta = math.radians(COMPASS[lab])
        runs = [rollout(scene, policy, theta, s, steps) for s in range(seeds)]
        out[lab] = DirectionalResult(float(np.mean([r[0] for r in runs])), float(np.mean([r[1] for r in runs])),
                                     float(np.mean([r[2] for r in runs])), runs)  # fmt: skip
    return out


def torch_policy(net: nn.Module) -> Callable[[np.ndarray], np.ndarray]:
    """``obs (528,) -> excitations (354,)`` of a :class:`WalkerNet`."""

    @torch.no_grad()
    def f(o: np.ndarray) -> np.ndarray:
        return net(torch.as_tensor(np.asarray(o, np.float32))[None])[0].numpy()

    return f
