# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""The method's handle on the tracking task of an env built with ``make_env``.

:class:`TaskView` wraps the task's motion command (reference frames, reset sampling)
and adds what SuperMimic needs on top: the body state ``S`` read straight from the
simulator, the differentiable kinematics, muscle geometry and activation dynamics, and
:class:`~.heads.TrackingHeads`, which evaluate the task's pure terms on ``S``.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at
commit 8a6244b, ``src/muscle_supermimic/tasks/tracking/mdp/commands.py``.
"""

from __future__ import annotations

from typing import Any

import torch

from myosuite.envs.myo.tasks.mimic.tracking.models import model_cfg

from .activation import MuscleActivation
from .heads import MUSCLE_OBSERVATIONS, StepCtx, TrackingHeads
from .kinematics import SkeletonKinematics
from .muscle_geometry import MuscleGeometry
from .rotations import quat_to_mat


class TaskView:
    """SuperMimic's view of a ``myoMimic-v0`` mjlab env; unknown attributes are the motion command's.

    Args:
        env: The ``ManagerBasedRlEnv``.
        command_name: The task's motion command.
    """

    def __init__(self, env: Any, command_name: str = "motion"):
        self._env = env
        self.cmd = cmd = env.command_manager.get_term(command_name)
        dev = cmd.device
        self.robot = cmd.robot
        self.info = cmd.info
        m = self.info.m
        self.kin = SkeletonKinematics(model_cfg(cmd.cfg.model).tracked_bodies, dev, m)
        self.muscle = MuscleActivation(m, env.cfg.decimation, env.physics_dt, dev)
        self.geo = MuscleGeometry(m, dev)
        task = cmd.task
        self.lookahead, self.stride = task.obs.goal_frames, cmd.goal_stride
        self.heads = TrackingHeads(
            self.kin, cmd.motion, self.geo, self.lookahead, self.stride
        )
        self.observes_muscles = any(
            t.params.get("name") in MUSCLE_OBSERVATIONS
            for g in env.cfg.observations.values()
            if g is not None
            for t in g.terms.values()
            if t is not None
        )
        # the shared pieces of each step's terms in one compiled call (muscle geometry only if observed); on sm_120
        # Triton fails to compile the tendon-wrap kernels, so a precompute with them stays eager there
        cuda = str(dev).startswith("cuda")
        sm120 = cuda and torch.cuda.get_device_capability(dev) >= (12, 0)
        comp = torch.compile if cuda else (lambda fn: fn)
        eager = lambda fn: fn  # noqa: E731
        self._shared_obs = (eager if sm120 and self.observes_muscles else comp)(
            lambda S: self.heads.precompute(S, muscles=self.observes_muscles))  # fmt: skip
        self._shared_transition = comp(
            lambda S: self.heads.precompute(S, muscles=False)
        )
        idx = self.robot.indexing
        self._free_q = idx.free_joint_q_adr.long()
        self._free_v = idx.free_joint_v_adr.long()
        cols = torch.tensor(self.info.indep_cols, device=dev)
        self._q_adr = idx.joint_q_adr.long()[cols]
        self._v_adr = idx.joint_v_adr.long()[cols]
        self._act_adr = torch.tensor(
            env.sim.mj_model.actuator_actadr[idx.ctrl_ids.cpu().numpy()], device=dev
        ).long()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.cmd, name)

    @property
    def gamma(self) -> float:
        """Discount of the task's reset statistics."""
        return self.cmd.task.reset.gamma

    @property
    def bin_fix(self) -> bool:
        """The task's reset statistics average the episodes of a bin."""
        return self.cmd.task.reset.bin_fix

    def read_state(self, fresh_touch: bool = False) -> tuple:
        """``S = (p, R, v, w, q, qd, act, touch)`` straight from qpos/qvel/act and the foot contact sensor.

        Args:
            fresh_touch: Read the sensor itself, not its per-step cache (the collector steps the simulator
                without mjlab's scene updates).
        """
        d = self._env.sim.data
        root_q, root_v = d.qpos[:, self._free_q], d.qvel[:, self._free_v]
        Rb = quat_to_mat(root_q[:, 3:7])
        p = root_q[:, :3] - self._env.scene.env_origins
        w = (Rb @ root_v[:, 3:6].unsqueeze(-1)).squeeze(-1)  # free joint: body-local
        sensor = self.cmd.touch_sensor
        force = sensor._compute_data().force if fresh_touch else sensor.data.force
        touch = force[..., 2].abs().reshape(self._env.num_envs, -1)
        return (p, Rb @ self.kin.R_off, root_v[:, :3], w, d.qpos[:, self._q_adr], d.qvel[:, self._v_adr],
                d.act[:, self._act_adr], touch)  # fmt: skip

    def obs_ctx(self) -> StepCtx:
        """The current state against frame f, with its shared pieces."""
        am = self._env.action_manager
        ctx = StepCtx(self.read_state(), self.cmd.f.clone(), am.action, am.prev_action)
        ctx.cache.update(self._shared_obs(ctx.S))
        return ctx
