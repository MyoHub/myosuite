# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""A steerable reference motion for the tracker: KIT clips played under a yaw transform, with motion-matching
transitions and a small yaw warp, producing the tracker's goal (5 frames 0.2 s apart + phase) in world axes.

Only the reference's heading enters the goal (site offsets and root displacements are relative), so steering
needs no position anchoring: the clip frame f is shown rotated by ``off`` about z, and ``off`` grows at the warp
rate ``omega``.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from myosuite.envs.myo.tasks.mimic.tracking.motion import MotionLib, motion_files

LOCOMOTION = re.compile(
    r"(walking_(slow|medium|fast|run)|turn_(left|right)|(Left|Right)Turn|WalkIn(Counter)?ClockwiseCircle|WalkingStraight(Forwards?|Backwards))\d*_poses"
)
"""KIT clip names of the steering library (no handrail, support or stylised walks)."""

FUT = (25, 50)
"""Frames ahead (0.5 s, 1.0 s at 50 Hz) of the motion-matching trajectory features."""


def wrap(a: float) -> float:
    """Angle wrapped to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def rotz(a: float) -> torch.Tensor:
    """Rotation by ``a`` about z (3, 3)."""
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def locomotion_files(
    motion_list: str, root: str, limit: int | None = None
) -> list[str]:
    """The clips of a clip list whose names match :data:`LOCOMOTION` (the first ``limit`` of them)."""
    files = [
        f
        for f in motion_files(motion_list, root)
        if LOCOMOTION.search(os.path.basename(f))
    ]
    return files[:limit] if limit else files


class ClipLibrary:
    """Reference quantities of every frame of the steering clips, plus motion-matching features.

    Args:
        motion: The task's motion library of the clips (NumPy, at the tracker's control step).
        R_off: Free-joint rotation -> canonical root frame (x forward) of the body.
        stride: Control steps between the goal frames.
        lookahead: Goal frames (the current one included).
    """

    def __init__(
        self, motion: MotionLib, R_off: np.ndarray, stride: int = 10, lookahead: int = 5
    ):
        self.dt, self.stride, self.lookahead = motion.dt, stride, lookahead
        fr = {k: torch.as_tensor(v) for k, v in motion.frames.items()}
        self.m = SimpleNamespace(root_p=fr["root_pos"], num_frames=motion.num_frames)
        Rp = fr["root_rot"]
        self.Rp = Rp
        self.v, self.w_body = fr["root_qvel"][:, :3], fr["root_qvel"][:, 3:]
        self.ez_body = Rp[..., 2, :]  # Rp^T e_z
        self.rpos = fr["site_rpos"]
        Rc = Rp @ torch.as_tensor(R_off, dtype=Rp.dtype)
        self.yaw = torch.atan2(Rc[:, 1, 0], Rc[:, 0, 0])
        self.clip_id, self.clip_start, self.clip_end, self.clip_len = (
            torch.as_tensor(motion.clip_id), torch.as_tensor(motion.clip_start),
            torch.as_tensor(motion.clip_end), torch.as_tensor(motion.clip_len))  # fmt: skip
        self.names = motion.names
        self.feat_pose, self.feat_traj = self._features()
        # frames a transition may jump to: at least the goal's span plus 1 s before the clip end
        span = (lookahead - 1) * stride + FUT[-1]
        self.valid = (torch.arange(self.m.num_frames) + span) <= self.clip_end[
            self.clip_id
        ]

    def heading_frame(self, f: torch.Tensor) -> torch.Tensor:
        """Rotations into the heading frame (x along the reference heading) of frames ``f``."""
        return rotz_batch(-self.yaw[f])

    def _features(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pose (sites, root velocity in the heading frame, joint velocities of the legs via site velocities) and
        trajectory (speed and heading change at 0.5 s / 1.0 s) features of every frame."""
        f = torch.arange(self.m.num_frames)
        H = self.heading_frame(f)
        pose = pose_features(H, self.rpos, self.v)
        end = self.clip_end[self.clip_id]
        traj = []
        for k in FUT:
            g = torch.minimum(f + k, end)
            d = (H @ (self.m.root_p[g] - self.m.root_p[f]).unsqueeze(-1)).squeeze(-1)
            traj += [
                d[:, :2] / (k * self.dt),
                wrap(self.yaw[g] - self.yaw[f]).unsqueeze(-1),
            ]
        return pose, torch.cat(traj, -1)


def rotz_batch(a: torch.Tensor) -> torch.Tensor:
    """Rotations about z by the angles ``a`` (..., 3, 3)."""
    c, s, z, o = a.cos(), a.sin(), torch.zeros_like(a), torch.ones_like(a)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(a.shape + (3, 3))


def pose_features(H: torch.Tensor, rpos: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Feet, hands and head offsets from the pelvis in the heading frame, and the root velocity in it."""
    sel = [
        1,
        4,
        7,
        10,
        11,
        14,
        15,
    ]  # head, both hands, both ankles, both toes (rpos skips the pelvis)
    p = (H.unsqueeze(-3) @ rpos[..., sel, :].unsqueeze(-1)).squeeze(-1)
    vh = (H @ v.unsqueeze(-1)).squeeze(-1)
    return torch.cat((p.flatten(-2), 0.3 * vh), -1)


@dataclass
class Steer:
    """What the planner asks of the reference: a world travel heading and a speed."""

    heading: float
    speed: float


class SteeredReference:
    """Plays the library under a yaw transform and produces the tracker goal.

    Args:
        lib: The clip library.
        kp: Yaw-warp gain (1/s) on the heading error left after the clip's own turning.
        omega_max: Warp-rate limit (rad/s).
        mm_every: Frames between motion-matching searches.
        switch_margin: Cost the best candidate must beat the current continuation by.
        w_traj: Weight of the trajectory features against the pose features.
    """

    def __init__(self, lib: ClipLibrary, kp: float = 2.0, omega_max: float = 1.2, mm_every: int = 10,
                 switch_margin: float = 0.05, w_traj: float = 1.0):  # fmt: skip
        self.lib, self.kp, self.omega_max = lib, kp, omega_max
        self.mm_every, self.switch_margin, self.w_traj = mm_every, switch_margin, w_traj
        self.f, self.off, self.omega, self.t = 0, 0.0, 0.0, 0
        self.switches = 0

    # ---------------- planning
    def ref_heading(self) -> float:
        """World heading of the current reference frame."""
        return float(self.off + self.lib.yaw[self.f])

    def desired_traj(self, steer: Steer, heading: float) -> torch.Tensor:
        """Trajectory features of a path from ``heading`` that turns toward the requested heading (at most 1.5 rad
        per 0.5 s) at the requested speed."""
        e = wrap(steer.heading - heading)
        out = []
        for k in FUT:
            turn = max(-1.5 * k / 25, min(1.5 * k / 25, e))
            out += [
                steer.speed * torch.tensor([math.cos(turn / 2), math.sin(turn / 2)]),
                torch.tensor([turn]),
            ]
        return torch.cat(out)

    def _cost(
        self, actual_pose: torch.Tensor, steer: Steer, heading: float
    ) -> torch.Tensor:
        lib = self.lib
        wt = torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0, 2.0])
        traj = (((lib.feat_traj - self.desired_traj(steer, heading)) * wt) ** 2).sum(-1)
        cost = ((lib.feat_pose - actual_pose) ** 2).sum(-1) + self.w_traj * traj
        return torch.where(lib.valid, cost, torch.inf)

    def match(self, actual_pose: torch.Tensor, steer: Steer) -> None:
        """Jump to the frame whose pose matches the body's and whose future matches the steering request."""
        heading = self.ref_heading()
        cost = self._cost(actual_pose, steer, heading)
        best = int(cost.argmin())
        if (
            best != self.f
            and float(cost[best]) < float(cost[self.f]) - self.switch_margin
        ):
            self.f = best
            self.off = heading - float(self.lib.yaw[best])
            self.switches += 1

    def reset(
        self, actual_pose: torch.Tensor, body_heading: float, steer: Steer
    ) -> None:
        """Start at the frame best matching the body, rotated to the body's heading."""
        self.omega, self.t, self.switches = 0.0, 0, 0
        self.f = int(self._cost(actual_pose, steer, body_heading).argmin())
        self.off = body_heading - float(self.lib.yaw[self.f])

    def update(self, actual_pose: torch.Tensor, steer: Steer) -> None:
        """Advance one tracker step: maybe transition, then set the warp rate for the residual heading error."""
        if self.t % self.mm_every == 0 or not bool(self.lib.valid[self.f]):
            self.match(actual_pose, steer)
        lib = self.lib
        # residual error after the clip's own turning over the next 0.5 s
        g = min(self.f + FUT[0], int(lib.clip_end[lib.clip_id[self.f]]))
        own = wrap(float(lib.yaw[g] - lib.yaw[self.f]))
        e = wrap(steer.heading - self.ref_heading() - own)
        self.omega = max(-self.omega_max, min(self.omega_max, self.kp * e))

    def step(self) -> None:
        """Advance the reference by one tracker step."""
        self.f = min(self.f + 1, int(self.lib.clip_end[self.lib.clip_id[self.f]]))
        self.off += self.omega * self.lib.dt
        self.t += 1

    # ---------------- goal
    def goal(self) -> torch.Tensor:
        """The tracker goal (1, goal_dim) of the current frame, in world axes."""
        lib, f = self.lib, self.f
        s = lib.stride
        end = int(lib.clip_end[lib.clip_id[f]])
        R0 = rotz(self.off)
        qv0 = torch.cat((R0 @ lib.v[f], lib.w_body[f] + self.omega * lib.ez_body[f]))
        parts = [(R0 @ lib.rpos[f].T).T.flatten()]
        p0 = lib.m.root_p[f]
        for k in range(1, lib.lookahead):
            g = min(f + k * s, end)
            tau = k * s * lib.dt
            Rk, Rmid = (
                rotz(self.off + self.omega * tau),
                rotz(self.off + self.omega * tau / 2),
            )
            qv = torch.cat((Rk @ lib.v[g], lib.w_body[g] + self.omega * lib.ez_body[g]))
            parts += [
                Rmid @ (lib.m.root_p[g] - p0),
                qv - qv0,
                (Rk @ lib.rpos[g].T).T.flatten(),
            ]
        c = int(lib.clip_id[f])
        parts.append(
            torch.tensor([(f - int(lib.clip_start[c])) / float(lib.clip_len[c])])
        )
        return torch.cat(parts)[None].float()
