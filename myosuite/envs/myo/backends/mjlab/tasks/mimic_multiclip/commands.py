# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""Motion command: the reference library, the analytic MDP heads and SuperMimic's reset sampling.

The term keeps ``f``, the reference frame of each env's *current* state (supermimic_vibed's
``TrackingSim.f``): reward and termination terms, which run between physics and the command update,
compare the post-step state with frame ``f + 1``; observations, which run after the update, use ``f``.

Episodes start at reference states drawn by return-based bin sampling (paper Appendix D.1: 50%
uniform over 1 s bins, 40% softmax(-25 G_b / G_max), 10% a slow average of that softmax). The reset
writes the reference state exactly -- no pose, velocity or joint noise.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/tasks/tracking/mdp/commands.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers.command_manager import CommandTerm
from mjlab.tasks.tracking.mdp import MotionCommandCfg as MjlabMotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .model_info import ModelInfo
from .rotations import mat_to_quat, quat_to_mat

from .heads import StepCtx, TrackingHeads
from .kinematics import SkeletonKinematics
from .motion import MotionLib
from .activation import MuscleActivation
from .muscle_geometry import MuscleGeometry

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
    """EMA update of per-bin statistics with possibly repeated bin indices: the samples of a bin (those with weight
    ``wt`` = 1; the others weigh 0) are averaged, then one EMA step is taken for every bin that received at least one.
    Fixed size, no host-side selection (BlinkMimic's ``_bin_ema``)."""
    sums = torch.zeros_like(stat).index_add_(0, bins, x * wt)
    cnt = torch.zeros_like(stat).index_add_(0, bins, wt)
    return torch.where(cnt > 0, old * stat + (1 - old) * sums / cnt.clamp(min=1), stat)


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if not cfg.motion_file:
            raise ValueError(
                "motion_file is empty: pass --env.commands.motion.motion-file <clip.npz | dir> (play: --motion-file)"
            )
        self.robot: Entity = env.scene[cfg.entity_name]
        m = (
            self.robot.cfg.spec_fn().compile()
        )  # the entity's own variant (e.g. locked hands), unprefixed
        self.info = ModelInfo(m)
        if m.body_weldid[m.body(cfg.anchor_body_name).id] != self.info.root_id:
            raise ValueError(
                f"the reward re-anchors on the root: anchor_body_name must be welded to {self.info.root_name!r}"
            )
        assert tuple(self.robot.joint_names) == tuple(
            self.info.joint_names
        ), "entity joints differ from the model's"
        dev = self.device
        self.kin = SkeletonKinematics(cfg.body_names, dev, m)
        bin_frames = max(1, round(cfg.bin_seconds / env.step_dt))
        self.motion = MotionLib.from_path(cfg.motion_file, self.kin, env.step_dt, dev, bin_frames=bin_frames,
                                          root=cfg.motion_root)  # fmt: skip
        self.muscle = MuscleActivation(m, env.cfg.decimation, env.physics_dt, dev)
        self.geo = MuscleGeometry(m, dev)
        self.heads = TrackingHeads(
            self.kin, self.motion, self.geo, cfg.lookahead, cfg.stride
        )
        # the shared pieces of each step's terms in one compiled call: observations with muscle geometry (if a term
        # observes muscle length, velocity or force), rewards and terminations without (eager, they launch tens of
        # thousands of small kernels per step). On sm_120 (RTX 50xx) Triton 3.8 hits an LLVM assertion compiling the
        # no-grad tendon-wrap kernels, so a precompute with the muscle geometry stays eager there.
        muscle_terms = (
            TrackingHeads.muscle_length,
            TrackingHeads.muscle_velocity,
            TrackingHeads.muscle_force,
        )
        self.observes_muscles = any(getattr(t.func, "analytic", None) in muscle_terms
                                    for g in env.cfg.observations.values() if g is not None
                                    for t in g.terms.values() if t is not None)  # fmt: skip
        cuda = str(self.device).startswith("cuda")
        sm120 = cuda and torch.cuda.get_device_capability(self.device) >= (12, 0)
        comp = torch.compile if cfg.compile and cuda else (lambda fn: fn)
        eager = lambda fn: fn  # noqa: E731
        # on sm_120 only the muscle geometry (the tendon-wrap kernels) fails to compile: the precompute without it is
        # compiled there too (the No-Muscle-Obs tasks' observation and every transition)
        self._shared_obs = (eager if sm120 and self.observes_muscles else comp)(
            lambda S: self.heads.precompute(S, muscles=self.observes_muscles))  # fmt: skip
        self._shared_transition = comp(
            lambda S: self.heads.precompute(S, muscles=False)
        )
        self.touch_sensor = env.scene[cfg.touch_sensor_name]

        # simulator addresses (global, from the scene's model)
        idx = self.robot.indexing
        self._free_q = idx.free_joint_q_adr.long()
        self._free_v = idx.free_joint_v_adr.long()
        cols = torch.tensor(self.info.indep_cols, device=dev)
        self._q_adr = idx.joint_q_adr.long()[cols]
        self._v_adr = idx.joint_v_adr.long()[cols]
        self._act_adr = torch.tensor(
            env.sim.mj_model.actuator_actadr[idx.ctrl_ids.cpu().numpy()], device=dev
        ).long()
        assert len(self._act_adr) == self.info.num_muscles

        # episode bookkeeping
        self.f = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self.start_bin = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._just_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self.ep_disc_ret = torch.zeros(self.num_envs, device=dev)
        self.ep_ret = torch.zeros(self.num_envs, device=dev)
        self.max_step_reward = (
            sum(
                r.weight
                for r in env.cfg.rewards.values()
                if r is not None and r.weight > 0
            )
            * env.step_dt
        )
        self.G_max = self.max_step_reward / (1 - cfg.gamma)
        nb = self.motion.num_bins
        self.bin_G = torch.zeros(nb, device=dev)
        self.bin_w_avg = torch.full((nb,), 1.0 / nb, device=dev)
        # fail_reset (InstantTrack): the termination rate of the bin each episode ends in, and every bin's predecessor
        # in its clip (itself at a clip start), where the resets before recent failure points go
        self.bin_fail = torch.zeros(nb, device=dev)
        prev = torch.arange(nb, device=dev) - 1
        self.bin_prev = torch.where(
            (prev >= 0)
            & (self.motion.bin_clip[prev.clamp(min=0)] == self.motion.bin_clip),
            prev,
            prev + 1,
        )
        self._ctx_step, self._ctx = -1, None
        self._obs_ctx_step, self._obs_ctx = -1, None

    # ------------------------------------------------------------------ state
    @property
    def command(self) -> torch.Tensor:
        return self.heads.goal(StepCtx((), self.f))

    def read_state(self, fresh_touch: bool = False):
        """(p, R, v, w, q, qd, act, touch) straight from qpos/qvel/act and the foot contact sensor (no derived
        quantities, so never stale). Touch is the normal force of the ground on each foot body, which on flat
        ground is what a MuJoCo touch sensor around the foot reads. ``fresh_touch``: read the sensor data itself
        instead of the sensor's per-step cache (the runner's collector steps the simulator without mjlab's scene
        updates, which invalidate that cache)."""
        d = self._env.sim.data
        root_q, root_v = d.qpos[:, self._free_q], d.qvel[:, self._free_v]
        Rb = quat_to_mat(root_q[:, 3:7])
        p = root_q[:, :3] - self._env.scene.env_origins
        w = (Rb @ root_v[:, 3:6].unsqueeze(-1)).squeeze(
            -1
        )  # free joint: angular velocity is body-local
        force = (
            self.touch_sensor._compute_data().force
            if fresh_touch
            else self.touch_sensor.data.force
        )
        touch = force[..., 2].abs().reshape(self.num_envs, -1)
        return (p, Rb @ self.kin.R_off, root_v[:, :3], w, d.qpos[:, self._q_adr], d.qvel[:, self._v_adr], d.act[:, self._act_adr],
                touch)  # fmt: skip

    def obs_ctx(self) -> StepCtx:
        """The current state against frame f, cached for all observation terms of one step: they share its
        kinematics and muscle geometry. A reset or the command update (which move f or the state) drop it."""
        step = self._env.common_step_counter
        if self._obs_ctx is None or self._obs_ctx_step != step:
            am = self._env.action_manager
            self._obs_ctx, self._obs_ctx_step = (
                StepCtx(self.read_state(), self.f.clone(), am.action, am.prev_action),
                step,
            )
            self._obs_ctx.cache.update(self._shared_obs(self._obs_ctx.S))
        return self._obs_ctx

    def transition_ctx(self) -> StepCtx:
        """The post-step state against frame f + 1, cached for all reward/termination terms of one step."""
        step = self._env.common_step_counter
        if self._ctx_step != step:
            m = self.motion
            f1 = torch.minimum(self.f + 1, m.clip_end[m.clip_id[self.f]])
            am = self._env.action_manager
            self._ctx, self._ctx_step = (
                StepCtx(self.read_state(), f1, am.action, am.prev_action),
                step,
            )
            self._ctx.cache.update(self._shared_transition(self._ctx.S))
        return self._ctx

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            self._finish_episodes(env_ids)
        return super().reset(env_ids)

    def _finish_episodes(self, ids: torch.Tensor) -> None:
        """Fold finished episodes into their start bin's return estimate; truncated ones bootstrap with
        their mean reward. With ``fail_reset``, also into the termination rate of the bin they ended in."""
        env, g, m = self._env, self.cfg.gamma, self.motion
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
        if self.cfg.bin_fix:  # the episodes of a bin averaged, then one EMA step per bin that got any (no host sync)
            self.bin_G.copy_(
                bin_ema(self.bin_G, self.start_bin[ids], G, valid.float(), 0.8)
            )
        else:  # index_put_ with repeated bins keeps an arbitrary one of their episodes (undefined behaviour)
            b = self.start_bin[ids][valid]
            self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G[valid])
        if self.cfg.fail_reset > 0:
            # f is already the post-step frame the termination was evaluated against (the command update ran first)
            fb = m.frame_bin[self.f[ids].clamp(max=m.num_frames - 1)]
            self.bin_fail.mul_(0.999)  # bins no episode ends in any more fade out
            if self.cfg.bin_fix:
                self.bin_fail.copy_(
                    bin_ema(self.bin_fail, fb, term.float(), valid.float(), 0.9)
                )
            else:
                fb, t = fb[valid], term[valid].float()
                self.bin_fail.index_put_((fb,), 0.9 * self.bin_fail[fb] + 0.1 * t)
        self.ep_disc_ret[ids] = 0.0
        self.ep_ret[ids] = 0.0

    def reset_probs(self, gate: torch.Tensor | None = None) -> torch.Tensor:
        """The reset distribution over bins (``sample_frames`` draws from it): SuperMimic's return-based mixture, with
        ``fail_reset`` of it moved to the bins before recent termination points. ``gate`` (the collector's masked
        resets, called every step): a 0-dim bool; the running average bin_w_avg only advances where it is true, as
        the eager reset samples only when an episode ended."""
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        if gate is None:
            self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        else:
            self.bin_w_avg.copy_(
                torch.where(gate, self.bin_w_avg * 0.999 + 0.001 * soft, self.bin_w_avg)
            )
        if self.cfg.sampling_mode == "uniform":
            probs = torch.full((m.num_bins,), 1.0 / m.num_bins, device=self.device)
        else:
            probs = (
                0.5 / m.num_bins
                + 0.4 * soft
                + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
            )
        if self.cfg.fail_reset > 0:
            fail = torch.zeros_like(self.bin_fail).index_add_(
                0, self.bin_prev, self.bin_fail
            )
            fail = fail / fail.sum().clamp(min=1e-6)
            probs = (
                1 - self.cfg.fail_reset
            ) * probs + self.cfg.fail_reset * torch.where(fail.sum() > 0, fail, probs)
        return probs

    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        bins = torch.multinomial(self.reset_probs(), n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(
            start + m.bin_frames, m.clip_end[m.bin_clip[bins]]
        )  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self.cfg.sampling_mode == "start":
            frames = self.motion.clip_start[env_ids % len(self.motion.names)]
            bins = self.motion.frame_bin[frames]
        else:
            frames, bins = self.sample_frames(len(env_ids))
        self.write_reference_state(env_ids, frames)
        self.start_bin[env_ids] = bins

    def write_reference_state(
        self, env_ids: torch.Tensor, frames: torch.Tensor
    ) -> None:
        p, R, v, w, q, qd = (x.clone() for x in self.motion.ref_state(frames))
        q = torch.clamp(q, self.kin.q_lo, self.kin.q_hi)
        q_full, qd_full = self.kin.full_joint_state(q, qd)
        quat = mat_to_quat(R @ self.kin.R_off.T)
        root = torch.cat((p + self._env.scene.env_origins[env_ids], quat, v, w), -1)
        self.robot.write_root_state_to_sim(root, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q_full, qd_full, env_ids=env_ids)
        self._env.sim.data.act[env_ids[:, None], self._act_adr] = (
            self.cfg.init_activation
        )
        self.f[env_ids] = frames
        self._just_reset[env_ids] = True
        self._obs_ctx = None

    # ------------------------------------------------------------------ masked episode ends (the runner's collector)
    # The same updates as reset(ids) for the envs where ``done`` is true, as fixed-size operations on every env with no
    # host synchronization, so that a collection step can be captured as one CUDA graph (BlinkMimic's --reset_masked).
    def finish_episodes_masked(
        self, done: torch.Tensor, terminated: torch.Tensor, T: torch.Tensor
    ) -> None:
        """_finish_episodes for the envs where ``done``; ``T`` is the episode length (the env's episode_length_buf).
        Needs bin_fix (the masked bin statistics are bin_ema with weight 0 for the other envs)."""
        g, m = self.cfg.gamma, self.motion
        Tf = T.float()
        wt = (done & (Tf > 0)).float()
        boot = torch.where(
            terminated, 0.0, (g**Tf) * (self.ep_ret / Tf.clamp(min=1)) / (1 - g)
        )
        self.bin_G.copy_(
            bin_ema(self.bin_G, self.start_bin, self.ep_disc_ret + boot, wt, 0.8)
        )
        if self.cfg.fail_reset > 0:
            fb = m.frame_bin[self.f.clamp(max=m.num_frames - 1)]
            self.bin_fail.copy_(
                torch.where(done.any(), self.bin_fail * 0.999, self.bin_fail)
            )  # as eager: once per reset
            self.bin_fail.copy_(bin_ema(self.bin_fail, fb, terminated.float(), wt, 0.9))
        self.ep_disc_ret.masked_fill_(done, 0.0)
        self.ep_ret.masked_fill_(done, 0.0)

    def sample_frames_masked(
        self, gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A reset frame and its bin for every env (the caller keeps those of the ended episodes): sample_frames with
        an inverse-CDF draw in place of multinomial (the same distribution, another random stream). The running average
        of the reset distribution advances only if ``gate`` (some episode ended), as in the eager reset."""
        m, n = self.motion, self.num_envs
        mode = self.cfg.sampling_mode
        if mode == "start":
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

    def write_reference_state_masked(self, done: torch.Tensor, frames: torch.Tensor, bins: torch.Tensor, qpos: torch.Tensor,
                                     qvel: torch.Tensor, act: torch.Tensor) -> None:  # fmt: skip
        """write_reference_state (and the start bin) for the envs where ``done``, into the simulator's qpos, qvel and
        act tensors; MuJoCo-Warp's reset of those worlds runs first, as in the eager reset."""
        p, R, v, w, q, qd = self.motion.ref_state(frames)
        q = torch.clamp(q, self.kin.q_lo, self.kin.q_hi)
        q_full, qd_full = self.kin.full_joint_state(q, qd)
        quat = mat_to_quat(R @ self.kin.R_off.T)
        idx, d = self.robot.indexing, done.unsqueeze(-1)
        fq, fv, jq, jv = (
            idx.free_joint_q_adr,
            idx.free_joint_v_adr,
            idx.joint_q_adr,
            idx.joint_v_adr,
        )
        qpos[:, fq] = torch.where(
            d, torch.cat((p + self._env.scene.env_origins, quat), -1), qpos[:, fq]
        )
        qvel[:, fv] = torch.where(
            d, torch.cat((v, quat_apply_inverse(quat, w)), -1), qvel[:, fv]
        )  # body-frame angular
        qpos[:, jq] = torch.where(d, q_full, qpos[:, jq])
        qvel[:, jv] = torch.where(d, qd_full, qvel[:, jv])
        act[:, self._act_adr] = torch.where(
            d,
            torch.full_like(act[:, self._act_adr], self.cfg.init_activation),
            act[:, self._act_adr],
        )
        self.f.copy_(torch.where(done, frames, self.f))
        self.start_bin.copy_(torch.where(done, bins, self.start_bin))

    # ------------------------------------------------------------------ step
    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is not None:  # reset path: the reset frame stays
            self._just_reset[env_ids] = False
            return
        stepped = ~self._just_reset
        env = self._env
        r = torch.where(stepped, env.reward_buf, 0.0)
        T = env.episode_length_buf.float()
        self.ep_disc_ret += (self.cfg.gamma ** (T - 1).clamp(min=0)) * r
        self.ep_ret += r
        self.f += stepped.long()
        self._just_reset[:] = False
        self._obs_ctx = None

    def _update_metrics(self) -> None:
        pass


@dataclass(kw_only=True)
class MotionCommandCfg(MjlabMotionCommandCfg):
    """Subclasses mjlab's tracking command cfg so that mjlab's ``train`` (``--env.commands.motion.motion-file``)
    and ``play`` (``--motion-file``) CLIs hand it the motion. Of the inherited fields only ``motion_file``,
    ``anchor_body_name``, ``body_names`` and ``entity_name`` are used; the inherited reset randomization
    ranges must stay empty (this task does not randomize anything)."""

    motion_file: str = ""
    """A clip (.npz) or a directory of clips; see :mod:`.motion` for the formats."""
    anchor_body_name: str = "pelvis"
    """The root's rigid body (the root body itself, or one welded to it)."""
    body_names: tuple[str, ...] = ()
    """Tracked bodies of the reward (the root's rigid body first)."""
    entity_name: str = "robot"
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    motion_root: str = ""
    """The directory the paths of a ``.txt`` clip list start from (default: the list's own directory), e.g. the
    download directory of ``amathislab/musclemimic-retargeted`` for the packaged KIT lists."""
    sampling_mode: Literal["return", "uniform", "start"] = "return"
    """Reset frames: SuperMimic's return-based bins, uniform bins, or each env at the start of clip env_id % C."""
    lookahead: int = 5
    """Reference frames in the goal (MuscleMimic's n_step_lookahead), the current one included."""
    stride: int = 20
    """Control steps between them (MuscleMimic's n_step_stride: 0.2 s at 100 Hz)."""
    touch_sensor_name: str = "foot_touch"
    """Scene contact sensor of the four foot bodies against the terrain (net force per body)."""
    gamma: float = 0.99
    bin_seconds: float = 1.0
    init_activation: float = 0.0
    """Muscle activation written at reset (MuJoCo's reset value is 0)."""
    bin_fix: bool = False
    """Reset statistics (BlinkMimic): average the episodes that end in one step with the same start bin before the
    EMA step, instead of index_put_, which keeps an arbitrary one of them."""
    fail_reset: float = 0.0
    """Share of the resets sent one bin before recent termination points (InstantTrack's anti-forgetting of hard
    clips): the per-bin termination rate, which decays as bins are mastered, spread to each bin's predecessor. 0: off."""
    compile: bool = True
    """torch.compile the kinematics and muscle geometry the terms share each step (CUDA only; on sm_120, where Triton
    3.8 fails to compile the muscle geometry, only the precompute without it)."""
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def __post_init__(self):
        if self.pose_range or self.velocity_range or any(self.joint_position_range):
            raise ValueError(
                "reset randomization (pose_range / velocity_range / joint_position_range) is not supported"
            )

    def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
        return MotionCommand(self, env)
