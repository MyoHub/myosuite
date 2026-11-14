# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU half of ``myoMimic-v0``: multi-clip full-body motion tracking."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.utils import EzPickle

from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.envs.muscle_stages import CtrlStageHost
from myosuite.terms import mimic_track as mt

from .config import DEFAULT_PRESET, MimicTaskCfg, resolve_task
from .models import DEFAULT_MODEL, TOUCH_SENSOR, ModelInfo, compiled_scene, model_cfg
from .motion import MotionLib, load_motion_lib
from .sampling import ResetSampler


def observation_dims(
    task: MimicTaskCfg, info: ModelInfo, num_tracked: int
) -> dict[str, int]:
    """Size of every policy term and of the goal."""
    n_j, n_m, n_s = len(info.joint_ids), info.num_muscles, num_tracked - 1
    sizes = {
        "root_pos": 5, "joint_pos": n_j, "root_vel": 6, "joint_vel": n_j,
        "muscle_length": n_m, "muscle_velocity": n_m, "muscle_force": n_m,
        "muscle_excitation": n_m, "muscle_activation": n_m, "touch": 4,
        "site_rpos": 3 * n_s, "site_rangles": 3 * n_s, "site_rvel": 6 * n_s,
    }  # fmt: skip
    dims = {name: sizes[name] for name in task.obs.policy}
    dims["goal"] = 3 * n_s + (task.obs.goal_frames - 1) * (9 + 3 * n_s) + 1
    return dims


class MimicEnv(CtrlStageHost, MyoGymnasiumEnv, EzPickle):
    """Track reference motion clips with a full-body muscle model.

    Args:
        frame_skip: Physics steps per control step (``EnvConfig.ctrl_dt``).
        preset: Task preset (:data:`~.config.PRESETS`).
        overrides: Dotted-key overrides of the preset (``{"motion.clips": ...}``).
        model: Body model (:data:`~.models.MODELS`, ``EnvConfig.model``).
        render_mode: Standard render mode.
        seed: Seed of ``np_random``.

    Actions are muscle excitations, clipped to [0, 1]. The observation is the
    ``policy`` terms followed by the ``goal`` (the mjlab twin's two groups, in order).
    An episode ends at a failure (``terminated``), at the clip end or the time limit
    (``truncated``). ``reset(options={"clip": c})`` starts at the first frame of clip
    ``c``, otherwise the preset's reset distribution draws the frame. The info holds
    the reward components and the deviations.
    """

    def __init__(
        self,
        frame_skip: int = 10,
        preset: str = DEFAULT_PRESET,
        overrides: Mapping[str, Any] | None = None,
        model: str = DEFAULT_MODEL,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.task = resolve_task(preset, overrides)
        self.model_name = model
        self._model_cfg = model_cfg(model)
        super().__init__(frame_skip=int(frame_skip), render_mode=render_mode)
        EzPickle.__init__(
            self,
            frame_skip=frame_skip,
            preset=preset,
            overrides=overrides,
            model=model,
            render_mode=render_mode,
            seed=seed,
        )
        self.seed(seed)
        self.model = copy.deepcopy(compiled_scene(model, self.task.self_collisions))
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep) * self.frame_skip
        self.metadata = {
            **self.metadata,
            "render_fps": max(1, round(1 / self._ctrl_dt)),
        }
        self.info = ModelInfo(self.model)
        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        m = self.model
        self._tracked = np.array([m.body(n).id for n in self._model_cfg.tracked_bodies])
        self._tracked_root = m.body_rootid[self._tracked]
        # mjlab orders the foot sensors as its find_bodies does: model order
        feet = sorted(self._model_cfg.foot_bodies, key=lambda n: m.body(n).id)
        self._touch_adr = np.array(
            [m.sensor(f"{TOUCH_SENSOR}_{n}_force").adr[0] + 2 for n in feet]
        )
        self._step_weights = self.task.reward.step_weights(self._ctrl_dt)
        self.max_step_reward = sum(w for w in self._step_weights.values() if w > 0)
        self._kernels = {t.name: t.kernel for t in self.task.reward.terms}
        self._goal_stride = max(1, round(self.task.obs.goal_spacing_s / self._ctrl_dt))
        self.obs_dims = observation_dims(self.task, self.info, len(self._tracked))
        self.obs_groups = {"policy": tuple(self.task.obs.policy), "goal": ("goal",)}
        self.obs_keys = [*self.task.obs.policy, "goal"]
        self.action_space = gym.spaces.Box(0.0, 1.0, (m.nu,), dtype=np.float32)
        self.observation_space = self._unbounded_obs_space(sum(self.obs_dims.values()))
        self._motion: MotionLib | None = None
        self._sampler: ResetSampler | None = None
        self._f = 0
        self._start_bin = 0
        self._steps = 0
        self._ret = self._disc_ret = 0.0
        self._failed = False
        self._excitation = np.zeros(m.nu)
        self._reset_clip: int | None = None

    # ------------------------------------------------------------------ motion
    @property
    def motion(self) -> MotionLib:
        """The reference clips at this env's control step (loaded on first use)."""
        if self._motion is None:
            mc = self.task.motion
            self._motion = load_motion_lib(
                mc.clips, mc.root, self.model_name, self.task.self_collisions,
                round(self._ctrl_dt, 9), mc.bin_s,
                workers=mc.workers, cache_dir=mc.cache_dir,
            )  # fmt: skip
            self._sampler = ResetSampler(
                self._motion, self.task.reset, self.max_step_reward
            )
        return self._motion

    @property
    def frame(self) -> int:
        """The reference frame the current state is compared with."""
        return self._f

    # ------------------------------------------------------------------ state
    def track_state(self) -> mt.TrackState:
        """The body state of :mod:`myosuite.terms.mimic_track` (needs fresh derived data)."""
        _m, d, info = self.model, self.data, self.info
        root_q = d.qpos[info.root_qadr : info.root_qadr + 7]
        ids = self._tracked
        w = d.cvel[ids, :3]
        com = d.subtree_com[self._tracked_root]
        return mt.TrackState(
            root_pos=root_q[:3].copy(),
            root_quat=root_q[3:7] / np.linalg.norm(root_q[3:7]),
            root_qvel=d.qvel[info.root_dadr : info.root_dadr + 6].copy(),
            joint_pos=d.qpos[info.joint_qadr].copy(),
            joint_vel=d.qvel[info.joint_dadr].copy(),
            body_pos=d.xpos[ids].copy(),
            body_rot=d.xmat[ids].reshape(-1, 3, 3).copy(),
            body_lin_vel=d.cvel[ids, 3:] + np.cross(w, d.xpos[ids] - com),
            body_ang_vel=w.copy(),
            act=d.act[info.act_adr].copy(),
            excitation=self._excitation.copy(),
            touch=np.abs(d.sensordata[self._touch_adr]),
            muscle_length=d.actuator_length.copy(),
            muscle_velocity=d.actuator_velocity.copy(),
            muscle_force=d.actuator_force.copy(),
        )

    def _goal(self) -> np.ndarray:
        m, f = self.motion, self._f
        end = int(m.clip_end[m.clip_id[f]])
        fut = np.minimum(
            f + self._goal_stride * np.arange(1, self.task.obs.goal_frames), end
        )
        return mt.goal_observation(np, m.ref(f), m.ref(fut), np.asarray(m.phase(f)))

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        del accessor
        s = self.track_state()
        feats = mt.features(np, s)
        obs = {n: mt.observation(np, s, feats, n) for n in self.task.obs.policy}
        obs["goal"] = self._goal()
        return obs

    def transition_terms(
        self,
    ) -> tuple[dict[str, float], dict[str, float], float | None]:
        """Reward components, deviations and the custom ``reward_fn`` (if any) of the
        current state against frame f + 1."""
        m = self.motion
        f1 = min(self._f + 1, int(m.clip_end[m.clip_id[self._f]]))
        s = self.track_state()
        feats, ref = mt.features(np, s), m.ref(f1)
        rewards = {
            n: float(mt.reward(np, s, feats, ref, n, k))
            for n, k in self._kernels.items()
        }
        devs = {n: float(mt.deviation(np, s, feats, ref, n)) for n in mt.DEVIATIONS}
        fn = self.task.reward_fn
        custom = None if fn is None else float(fn(s, ref, self._ctrl_dt))
        return rewards, devs, custom

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        """Reward of the last transition (state vs frame f + 1) and the failure rule.

        ``task_reward`` is the proposed reward, also when ``reward_fn`` replaces it.
        """
        del obs_dict
        rewards, devs, custom = self.transition_terms()
        task_reward = sum(self._step_weights[n] * r for n, r in rewards.items())
        failed = any(devs[n] > thr for n, thr in self.task.failure.thresholds().items())
        return {
            "dense": task_reward if custom is None else custom,
            "task_reward": task_reward,
            "solved": False,
            "done": failed,
            "failed": failed,
            **{f"r_{n}": r for n, r in rewards.items()},
            **{f"dev_{n}": v for n, v in devs.items()},
        }

    # ------------------------------------------------------------------ episodes
    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode; ``options={"clip": c}`` plays clip ``c`` from its start."""
        self._reset_clip = None if options is None else options.get("clip")
        return super().reset(seed=seed, options=options)

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Fold the last episode into the reset statistics and write a reference state."""
        m = self.motion
        assert self._sampler is not None
        self._sampler.finish_episode(
            self._start_bin,
            self._f,
            self._steps,
            self._disc_ret,
            self._ret,
            self._failed,
        )
        clip = self._reset_clip
        if clip is None and self.task.reset.mode == "start":
            clip = int(np_random.integers(len(m.names)))
        if clip is not None:
            frame = int(m.clip_start[clip])
            b = int(m.frame_bin[frame])
        else:
            frame, b = self._sampler.sample(np_random)
        self.data.qpos[:] = m.frames["reset_qpos"][frame]
        self.data.qvel[:] = m.frames["reset_qvel"][frame]
        self.data.act[self.info.act_adr] = self.task.reset.init_activation
        self._f, self._start_bin = frame, b
        self._steps, self._ret, self._disc_ret, self._failed = 0, 0.0, 0.0, False
        self._excitation = np.zeros(self.model.nu)
        self._run_reset_stages()
        return {"frame": frame, "clip": int(m.clip_id[frame])}

    def step(
        self, action: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Excite the muscles for one control step, then score it against frame f + 1."""
        del kwargs
        m = self.motion
        action = np.asarray(action, dtype=float)
        self._excitation = action.copy()
        ctrl = np.clip(action, 0.0, 1.0)
        self.data.ctrl[:] = self._run_ctrl_stages(ctrl)
        self._step_physics()
        clip_end = bool(self._f + 1 >= int(m.clip_end[m.clip_id[self._f]]))
        rwd = self.get_reward_dict({})
        self._f += 1
        self._steps += 1
        self._disc_ret += self.task.reset.gamma ** (self._steps - 1) * rwd["dense"]
        self._ret += rwd["dense"]
        self._failed = bool(rwd["failed"])
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs, reward, terminated, _, info = self._finalize_step(
            self._get_obs_dict(self._accessor), rwd
        )
        limit = self.task.episode_length_s
        timeout = limit is not None and self._steps * self._ctrl_dt >= limit - 1e-9
        info.update(frame=self._f, clip_end=clip_end)
        return obs, reward, terminated, (clip_end or timeout) and not terminated, info
