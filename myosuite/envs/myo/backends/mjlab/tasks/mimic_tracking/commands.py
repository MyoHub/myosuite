# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# myosuite/envs/myo/tasks/mimic/tracking/ for the notices of the code it includes.
"""Motion command of the ``myoMimic-v0`` twin: reference frame, reset sampling and the
tracking state the terms share.

Per env it holds the reference frame ``f`` the current state is compared with, the
start bin of the episode and the reset statistics (SuperMimic's return-based bins,
BlinkMimic's bin fix, InstantTrack's fail resets). The ``*_masked`` methods do the same
updates as fixed-size operations without host synchronization, for a learner that
steps the simulator itself inside a CUDA graph.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at
commit 8a6244b, ``src/muscle_supermimic/tasks/tracking/mdp/commands.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from myosuite.envs.myo.tasks.mimic.tracking.config import MimicTaskCfg
from myosuite.envs.myo.tasks.mimic.tracking.models import ModelInfo, model_cfg
from myosuite.envs.myo.tasks.mimic.tracking.motion import MotionLib, load_motion_lib
from myosuite.terms import mimic_track as mt

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def bin_ema(
    stat: torch.Tensor,
    bins: torch.Tensor,
    x: torch.Tensor,
    wt: torch.Tensor,
    old: float,
) -> torch.Tensor:
    """EMA update of per-bin statistics with repeated bin indices: the samples of a bin
    (weight 1; others 0) are averaged, then one EMA step for every bin that got one."""
    sums = torch.zeros_like(stat).index_add_(0, bins, x * wt)
    cnt = torch.zeros_like(stat).index_add_(0, bins, wt)
    return torch.where(cnt > 0, old * stat + (1 - old) * sums / cnt.clamp(min=1), stat)


class MotionCommand(CommandTerm):
    """Reference frames, resets and the shared tracking state of every env."""

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        task = cfg.task
        self.robot: Entity = env.scene[cfg.entity_name]
        mc = model_cfg(cfg.model)
        m = self.robot.cfg.spec_fn().compile()
        self.info = ModelInfo(m, mc.motor_expr or None)
        self.fixed_root = self.info.fixed_root
        dev = self.device
        lib = load_motion_lib(
            task.motion.clips, task.motion.root, cfg.model, task.self_collisions,
            round(env.step_dt, 9), task.motion.bin_s,
        )  # fmt: skip
        self.motion: MotionLib = lib.map(lambda x: torch.as_tensor(x, device=dev))
        self.touch_sensor = (
            None if cfg.touch_sensor_name is None else env.scene[cfg.touch_sensor_name]
        )
        self._body_ids = torch.tensor(
            self.robot.find_bodies(mc.tracked_bodies, preserve_order=True)[0],
            device=dev,
        )
        idx = self.robot.indexing
        mus = torch.as_tensor(self.info.muscle_ids, device=dev)
        self._act_adr = torch.tensor(
            env.sim.mj_model.actuator_actadr[idx.ctrl_ids.cpu().numpy()], device=dev
        ).long()[mus]
        self._ctrl_ids = idx.ctrl_ids.long()
        self._muscle_ctrl_ids = self._ctrl_ids[mus]
        self._joint_qadr = torch.as_tensor(self.info.joint_qadr, device=dev)
        self._joint_dadr = torch.as_tensor(self.info.joint_dadr, device=dev)
        self.goal_stride = max(1, round(task.obs.goal_spacing_s / env.step_dt))
        self._goal_offsets = self.goal_stride * torch.arange(
            1, task.obs.goal_frames, device=dev
        )

        self.f = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self.start_bin = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._just_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self.ep_disc_ret = torch.zeros(self.num_envs, device=dev)
        self.ep_ret = torch.zeros(self.num_envs, device=dev)
        # the proposed reward's largest step reward (also with a method's reward_fn)
        weights = (
            [r.weight * env.step_dt for r in env.cfg.rewards.values() if r is not None]
            if task.reward_fn is None
            else list(task.reward.step_weights(env.step_dt).values())
        )
        self.max_step_reward = sum(w for w in weights if w > 0)
        self.task_reward = torch.zeros(self.num_envs, device=dev)
        """The proposed reward of the last step (the env's reward unless ``reward_fn`` replaces it)."""
        self.metrics["task_reward"] = torch.zeros(self.num_envs, device=dev)
        self.G_max = self.max_step_reward / (1 - task.reset.gamma)
        nb = self.motion.num_bins
        self.bin_G = torch.zeros(nb, device=dev)
        self.bin_w_avg = torch.full((nb,), 1.0 / nb, device=dev)
        self.bin_fail = torch.zeros(nb, device=dev)
        prev = torch.arange(nb, device=dev) - 1
        self.bin_prev = torch.where(
            (prev >= 0)
            & (self.motion.bin_clip[prev.clamp(min=0)] == self.motion.bin_clip),
            prev,
            prev + 1,
        )
        self._cache: dict[str, tuple[int, object]] = {}

    @property
    def task(self) -> MimicTaskCfg:
        """The task config."""
        return self.cfg.task

    # ------------------------------------------------------------------ state
    @property
    def command(self) -> torch.Tensor:
        return self.goal()

    def track_state(self, fresh_touch: bool = False) -> mt.TrackState:
        """The tracking state of every env from the entity data (derived data must be fresh).

        Args:
            fresh_touch: Read the contact sensor itself, not its per-step cache (for a
                caller that steps the simulator without mjlab's scene updates).
        """
        d, sim = self.robot.data, self._env.sim.data
        ids = self._body_ids
        quat = d.body_link_quat_w[:, ids]
        if self.fixed_root:  # a fixed base stays where its model puts it in every world
            origins = torch.zeros_like(self._env.scene.env_origins)
            root_pos = root_quat = root_qvel = origins[:, :0]
        else:
            origins = self._env.scene.env_origins
            R = mt.quat_to_mat(torch, d.root_link_quat_w)
            ang_b = (R.transpose(-1, -2) @ d.root_link_ang_vel_w[..., None])[..., 0]
            root_pos = d.root_link_pos_w - origins
            root_quat = d.root_link_quat_w
            root_qvel = torch.cat((d.root_link_lin_vel_w, ang_b), -1)
        return mt.TrackState(
            root_pos=root_pos,
            root_quat=root_quat,
            root_qvel=root_qvel,
            joint_pos=d.joint_pos,
            joint_vel=d.joint_vel,
            body_pos=d.body_link_pos_w[:, ids] - origins[:, None],
            body_rot=mt.quat_to_mat(torch, quat),
            body_lin_vel=d.body_link_lin_vel_w[:, ids],
            body_ang_vel=d.body_link_ang_vel_w[:, ids],
            act=sim.act[:, self._act_adr],
            excitation=self._env.action_manager.action,
            touch=self.read_touch(fresh_touch),
            muscle_length=sim.actuator_length[:, self._muscle_ctrl_ids],
            muscle_velocity=sim.actuator_velocity[:, self._muscle_ctrl_ids],
            muscle_force=sim.actuator_force[:, self._muscle_ctrl_ids],
        )

    def read_touch(self, fresh: bool = False) -> torch.Tensor:
        """Normal ground force on the contact bodies ``(N, C)`` (``C = 0``: none).

        Args:
            fresh: Read the contact sensor itself, not its per-step cache.
        """
        sensor = self.touch_sensor
        if sensor is None:
            return torch.zeros(self.num_envs, 0, device=self.device)
        force = sensor._compute_data().force if fresh else sensor.data.force
        return force[..., 2].abs().reshape(self.num_envs, -1)

    def _cached(self, key: str, build) -> object:
        step = self._env.common_step_counter
        hit = self._cache.get(key)
        if hit is None or hit[0] != step:
            hit = (step, build())
            self._cache[key] = hit
        return hit[1]

    def obs_state(self) -> tuple[mt.TrackState, mt.TrackFeatures]:
        """The current state and its features (shared by the observation terms of a step)."""

        def build():
            s = self.track_state()
            return s, mt.features(torch, s)

        return self._cached("obs", build)

    def transition(self) -> tuple[mt.TrackState, mt.TrackFeatures, mt.TrackRef]:
        """The post-step state against frame f + 1 (shared by rewards and terminations)."""

        def build():
            s = self.track_state()
            m = self.motion
            f1 = torch.minimum(self.f + 1, m.clip_end[m.clip_id[self.f]])
            return s, mt.features(torch, s), m.ref(f1)

        return self._cached("transition", build)

    def goal(self) -> torch.Tensor:
        """The reference-only goal at the current frame."""
        # an ended episode not yet reset (auto_reset off) can sit one past the last frame
        m = self.motion
        f = self.f.clamp(max=m.num_frames - 1)
        fut = torch.minimum(
            f[:, None] + self._goal_offsets, m.clip_end[m.clip_id[f]][:, None]
        )
        return mt.goal_observation(torch, m.ref(f), m.ref(fut), m.phase(f).float())

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            self._finish_episodes(env_ids)
        return super().reset(env_ids)

    def _finish_episodes(self, ids: torch.Tensor) -> None:
        """Fold finished episodes into their start bin's return estimate (truncated ones
        bootstrap with their mean reward) and, with ``fail_reset``, into the failure
        rate of the bin they ended in."""
        env, rc, m = self._env, self.task.reset, self.motion
        g = rc.gamma
        T = env.episode_length_buf[ids].float()
        term = getattr(
            env,
            "reset_terminated",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )[ids]
        valid = T > 0
        boot = torch.where(
            term, 0.0, (g**T) * (self.ep_ret[ids] / T.clamp(min=1)) / (1 - g)
        )
        G = self.ep_disc_ret[ids] + boot
        if rc.bin_fix:
            self.bin_G.copy_(
                bin_ema(self.bin_G, self.start_bin[ids], G, valid.float(), 0.8)
            )
        else:
            b = self.start_bin[ids][valid]
            self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G[valid])
        if rc.fail_reset > 0:
            fb = m.frame_bin[self.f[ids].clamp(max=m.num_frames - 1)]
            self.bin_fail.mul_(0.999)
            if rc.bin_fix:
                self.bin_fail.copy_(
                    bin_ema(self.bin_fail, fb, term.float(), valid.float(), 0.9)
                )
            else:
                fb, t = fb[valid], term[valid].float()
                self.bin_fail.index_put_((fb,), 0.9 * self.bin_fail[fb] + 0.1 * t)
        self.ep_disc_ret[ids] = 0.0
        self.ep_ret[ids] = 0.0

    def reset_probs(self, gate: torch.Tensor | None = None) -> torch.Tensor:
        """The reset distribution over bins. ``gate`` (masked resets, called every step):
        the running average only advances where it is true."""
        m, rc = self.motion, self.task.reset
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        if gate is None:
            self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        else:
            self.bin_w_avg.copy_(
                torch.where(gate, self.bin_w_avg * 0.999 + 0.001 * soft, self.bin_w_avg)
            )
        if rc.mode == "uniform":
            probs = torch.full((m.num_bins,), 1.0 / m.num_bins, device=self.device)
        else:
            probs = (
                0.5 / m.num_bins
                + 0.4 * soft
                + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
            )
        if rc.fail_reset > 0:
            fail = torch.zeros_like(self.bin_fail).index_add_(
                0, self.bin_prev, self.bin_fail
            )
            fail = fail / fail.sum().clamp(min=1e-6)
            probs = (1 - rc.fail_reset) * probs + rc.fail_reset * torch.where(
                fail.sum() > 0, fail, probs
            )
        return probs

    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset frames and their bins (at least one step before the clip end)."""
        m = self.motion
        bins = torch.multinomial(self.reset_probs(), n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + m.bin_frames, m.clip_end[m.bin_clip[bins]])
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self.task.reset.mode == "start":
            frames = self.motion.clip_start[env_ids % len(self.motion.names)]
            bins = self.motion.frame_bin[frames]
        else:
            frames, bins = self.sample_frames(len(env_ids))
        self.write_reference_state(env_ids, frames)
        self.start_bin[env_ids] = bins

    def _reference_state(self, frames: torch.Tensor):
        info, fr = self.info, self.motion.frames
        qpos, qvel = fr["reset_qpos"][frames], fr["reset_qvel"][frames]
        if info.fixed_root:
            return None, qpos[:, self._joint_qadr], qvel[:, self._joint_dadr]
        a, b = info.root_qadr, info.root_dadr
        quat = qpos[:, a + 3 : a + 7]
        w_body = qvel[:, b + 3 : b + 6]
        w = (mt.quat_to_mat(torch, quat) @ w_body[..., None])[..., 0]
        root = (qpos[:, a : a + 3], quat, qvel[:, b : b + 3], w)
        return root, qpos[:, self._joint_qadr], qvel[:, self._joint_dadr]

    def write_reference_state(
        self, env_ids: torch.Tensor, frames: torch.Tensor
    ) -> None:
        """Put envs on the reference at *frames* (joints clamped to their ranges)."""
        root, q, qd = self._reference_state(frames)
        if root is not None:
            p, quat, v, w = root
            origins = self._env.scene.env_origins[env_ids]
            self.robot.write_root_state_to_sim(
                torch.cat((p + origins, quat, v, w), -1), env_ids=env_ids
            )
        self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
        self._env.sim.data.act[env_ids[:, None], self._act_adr] = (
            self.task.reset.init_activation
        )
        self.f[env_ids] = frames
        self._just_reset[env_ids] = True
        self._cache.clear()

    # ------------------------------------------------------------------ masked episode ends
    def finish_episodes_masked(
        self, done: torch.Tensor, terminated: torch.Tensor, T: torch.Tensor
    ) -> None:
        """:meth:`_finish_episodes` for the envs where ``done`` (needs ``bin_fix``)."""
        rc, m = self.task.reset, self.motion
        g = rc.gamma
        Tf = T.float()
        wt = (done & (Tf > 0)).float()
        boot = torch.where(
            terminated, 0.0, (g**Tf) * (self.ep_ret / Tf.clamp(min=1)) / (1 - g)
        )
        self.bin_G.copy_(
            bin_ema(self.bin_G, self.start_bin, self.ep_disc_ret + boot, wt, 0.8)
        )
        if rc.fail_reset > 0:
            fb = m.frame_bin[self.f.clamp(max=m.num_frames - 1)]
            self.bin_fail.copy_(
                torch.where(done.any(), self.bin_fail * 0.999, self.bin_fail)
            )
            self.bin_fail.copy_(bin_ema(self.bin_fail, fb, terminated.float(), wt, 0.9))
        self.ep_disc_ret.masked_fill_(done, 0.0)
        self.ep_ret.masked_fill_(done, 0.0)

    def sample_frames_masked(
        self, gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A reset frame and bin for every env (inverse-CDF draw, no host sync)."""
        m, n = self.motion, self.num_envs
        if self.task.reset.mode == "start":
            frames = m.clip_start[torch.arange(n, device=self.device) % len(m.names)]
            return frames, m.frame_bin[frames]
        cdf = self.reset_probs(gate).cumsum(0)
        bins = torch.searchsorted(
            cdf, torch.rand(n, device=self.device) * cdf[-1], right=True
        ).clamp(max=m.num_bins - 1)
        start = m.bin_start[bins]
        end = torch.minimum(start + m.bin_frames, m.clip_end[m.bin_clip[bins]])
        return start + (
            torch.rand(n, device=self.device) * (end - start).float()
        ).long(), bins

    def write_reference_state_masked(
        self, done: torch.Tensor, frames: torch.Tensor, bins: torch.Tensor,
        qpos: torch.Tensor, qvel: torch.Tensor, act: torch.Tensor,
    ) -> None:  # fmt: skip
        """:meth:`write_reference_state` for the envs where ``done``, into the simulator's
        qpos, qvel and act tensors (MuJoCo-Warp's reset of those worlds runs first)."""
        root, q, qd = self._reference_state(frames)
        idx, d = self.robot.indexing, done.unsqueeze(-1)
        jq, jv = idx.joint_q_adr, idx.joint_v_adr
        if root is not None:
            p, quat, v, _ = root
            w_body = self.motion.frames["reset_qvel"][frames][
                :, self.info.root_dadr + 3 : self.info.root_dadr + 6
            ]
            fq, fv = idx.free_joint_q_adr, idx.free_joint_v_adr
            qpos[:, fq] = torch.where(
                d, torch.cat((p + self._env.scene.env_origins, quat), -1), qpos[:, fq]
            )
            qvel[:, fv] = torch.where(d, torch.cat((v, w_body), -1), qvel[:, fv])
        qpos[:, jq] = torch.where(d, q, qpos[:, jq])
        qvel[:, jv] = torch.where(d, qd, qvel[:, jv])
        act[:, self._act_adr] = torch.where(
            d, torch.full_like(act[:, self._act_adr], self.task.reset.init_activation),
            act[:, self._act_adr],
        )  # fmt: skip
        self.f.copy_(torch.where(done, frames, self.f))
        self.start_bin.copy_(torch.where(done, bins, self.start_bin))

    # ------------------------------------------------------------------ step
    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is not None:  # reset path: the reset frame stays
            self._just_reset[env_ids] = False
            return
        stepped = ~self._just_reset
        env = self._env
        if self.task.reward_fn is None:
            self.task_reward = env.reward_buf.clone()
        self.metrics["task_reward"] += torch.where(stepped, self.task_reward, 0.0)
        r = torch.where(stepped, env.reward_buf, 0.0)
        T = env.episode_length_buf.float()
        self.ep_disc_ret += (self.task.reset.gamma ** (T - 1).clamp(min=0)) * r
        self.ep_ret += r
        self.f += stepped.long()
        self._just_reset[:] = False
        self._cache.clear()

    def _update_metrics(self) -> None:
        pass


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
    """Config of :class:`MotionCommand`."""

    task: MimicTaskCfg
    model: str
    entity_name: str = "robot"
    touch_sensor_name: str | None = "foot_touch"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
        return MotionCommand(self, env)
