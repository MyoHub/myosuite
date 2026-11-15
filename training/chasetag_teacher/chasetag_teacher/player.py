# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""A ChaseTag FBVs player: planner (role heading + speed) -> steered reference -> myoSuperMimic tracker."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from supermimic.heads import rel_sites

from myosuite.integrations.musclemimic.steering import aim_heading_xy

from .reference import ClipLibrary, Steer, SteeredReference, pose_features, rotz_batch
from .tracker import TRACKER_DT, CpuTracker


@dataclass
class PlannerCfg:
    """Speeds (m/s) the planner asks for."""

    chase_speed: float = 1.6
    chase_near_speed: float = 1.2
    near_m: float = 1.2
    evade_speed: float = 1.6


class SuperMimicPlayer:
    """One player of the two-player CPU scene, acting at the env's control step (the tracker at 50 Hz, held).

    Args:
        checkpoint: The myoSuperMimic tracker checkpoint.
        lib: The steering clip library.
        model: The FBVs scene.
        prefix: The player's name prefix (``"a0_"`` / ``"a1_"``).
        env_dt: The env's control step (s).
        planner: Requested speeds.
        **ref_kwargs: Options of :class:`SteeredReference`.
    """

    def __init__(self, checkpoint: str, lib: ClipLibrary, model: mujoco.MjModel, prefix: str, env_dt: float,
                 planner: PlannerCfg | None = None, **ref_kwargs: float):  # fmt: skip
        self.tracker = CpuTracker(checkpoint, model, prefix)
        self.ref = SteeredReference(lib, **ref_kwargs)
        self.planner = planner or PlannerCfg()
        self.hold = max(1, round(TRACKER_DT / env_dt))
        self.k, self.action = 0, None
        self.last_goal = None

    def _pose(self, S: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, float]:
        p, R, v, w, q, qd = S[:6]
        heading = float(torch.atan2(R[0, 1, 0], R[0, 0, 0]))
        rpos = rel_sites(*self.tracker.kin.forward(p, R, v, w, q, qd))[0]
        H = rotz_batch(torch.tensor([-heading]))
        return pose_features(H, rpos, v)[0], heading

    def steer(self, me_xy: np.ndarray, opp_xy: np.ndarray, pursue: bool) -> Steer:
        """Travel heading toward (chaser) or away from (runner) the opponent, turned off the fence, and a speed."""
        travel = aim_heading_xy(me_xy, opp_xy, pursue=pursue)
        dist = float(np.linalg.norm(np.asarray(opp_xy) - np.asarray(me_xy)))
        c = self.planner
        speed = (
            (c.chase_speed if dist > c.near_m else c.chase_near_speed)
            if pursue
            else c.evade_speed
        )
        return Steer(math.atan2(travel[1], travel[0]), speed)

    def reset(
        self, data: mujoco.MjData, me_xy: np.ndarray, opp_xy: np.ndarray, pursue: bool
    ) -> None:
        """Start the reference at the clip frame that best matches the player's pose and the steering request."""
        self.tracker.reset()
        pose, heading = self._pose(self.tracker.state(data))
        self.ref.reset(pose, heading, self.steer(me_xy, opp_xy, pursue))
        self.k, self.action = 0, None

    def act(
        self, data: mujoco.MjData, me_xy: np.ndarray, opp_xy: np.ndarray, pursue: bool
    ) -> np.ndarray:
        """Excitations (354,) in [0, 1]; the tracker acts every 0.02 s and its action is held in between."""
        if self.k % self.hold == 0:
            S = self.tracker.state(data)
            if self.k > 0:
                self.ref.step()
            pose, _ = self._pose(S)
            self.ref.update(pose, self.steer(me_xy, opp_xy, pursue))
            self.last_goal = self.ref.goal()
            self.action = self.tracker.act(data, self.last_goal)
        self.k += 1
        return self.action
