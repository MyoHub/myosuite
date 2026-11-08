# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""The multi-clip full-body tracking task (``myoMimicFullbodyMulticlip-v0``, mjlab only) and its registration.

MuscleMimic's tracking MDP on MyoSuite's full body: its observation (muscle length, velocity and force included
unless ``muscle_obs=False``, muscle-supermimic's ``-No-Muscle-Obs`` variant), its concise goal (5 reference frames
0.2 s apart and the motion phase), its MimicReward with the full-body weights scaled by 0.1, and its deviation
terminations (1 m / 1 m / 1 rad). No randomization of any kind. The registered defaults are MuscleMimic's: 100 Hz
control (2 ms physics x 5), resets uniform over 1 s bins of the clips. Training methods choose other reset
statistics, control rates or muscle observations through the options of :func:`mimic_multiclip_env_cfg`, e.g.
``make_env(EnvConfig(ENV_ID, ctrl_dt=0.02, task_kwargs={"motion_file": ..., "sampling_mode": "return"}),
backend="mjlab")``.

The reward weights are those of the 100 Hz task at every control rate, so at 50 Hz a step earns twice the 100 Hz
step's reward; the normalized return divides by the maximum step reward and is unaffected.

``myoMimicFullbody-v0`` is a different MDP (DeepMimic-style composite reward on world site positions, its own
observation and terminations) and is unchanged.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit 8a6244b,
``src/muscle_supermimic/tasks/tracking/{tracking_env_cfg.py,config/myofullbody/*}``.
"""

from __future__ import annotations

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from . import ENV_ID, mdp
from .body import BODY_MODEL, FOOT_BODIES, TRACKED_BODIES, body_cfg
from .commands import MotionCommandCfg

CMD = {"command_name": "motion"}
TIMESTEP = 0.002
BASE_DECIMATION = 5
"""MuscleMimic's 100 Hz control step (the default): the reward weights are set for it."""
GOAL_SPACING_S = 0.2
"""Time between the goal's reference frames (MuscleMimic's n_step_stride at 100 Hz)."""
SamplingMode = Literal["uniform", "return", "start"]
REWARD_SCALE = 0.1
MIMIC_REWARD: dict[str, tuple[float, float]] = {
    "qpos": (0.1, 10.0), "qvel": (0.1, 2.0), "root_pos": (0.1, 10.0), "root_vel": (0.1, 10.0),
    "rpos": (0.6, 100.0), "rquat": (0.01, 10.0), "rvel_rot": (0.1, 0.1), "rvel_lin": (0.1, 0.1),
}  # fmt: skip
"""MimicReward: weight (w_sum) and kernel (w_exp) per component."""


def mimic_multiclip_env_cfg(
    motion_file: str = "",
    motion_root: str = "",
    num_envs: int = 1024,
    ctrl_dt: float = TIMESTEP * BASE_DECIMATION,
    muscle_obs: bool = True,
    sampling_mode: SamplingMode = "uniform",
    bin_fix: bool = False,
    fail_reset: float = 0.0,
    lookahead: int = 5,
    goal_stride: int | None = None,
    time_limit: bool = True,
    episode_length_s: float = 300.0,
    auto_reset: bool = True,
    self_collisions: bool = True,
    model_name: str = BODY_MODEL,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """The env config of ``myoMimicFullbodyMulticlip-v0`` (``make_env``'s ``EnvConfig.task_kwargs`` are these options).

    Args:
        motion_file: A clip, a directory of clips or a ``.txt`` clip list (see :func:`.motion.motion_files`).
        motion_root: The directory a list's paths start from.
        num_envs: Parallel envs.
        ctrl_dt: Control step in seconds, a whole multiple of the 2 ms physics step.
        muscle_obs: Observe muscle length, velocity and force, as MuscleMimic does. Off: the muscle geometry, the
            costliest part of an analytic step, is not evaluated.
        sampling_mode: Reset frames: ``"uniform"`` over 1 s bins, ``"return"`` (SuperMimic's return-based bins) or
            ``"start"`` (env i at the start of clip i % C).
        bin_fix: Average the episodes ending in one step with the same start bin before the bin statistics' EMA step
            (BlinkMimic's reset-statistics fix).
        fail_reset: Share of the resets sent one bin before recent termination points (InstantTrack); 0 is off.
        lookahead: Reference frames in the goal, the current one included.
        goal_stride: Control steps between them; default :data:`GOAL_SPACING_S` / ``ctrl_dt``.
        time_limit: End episodes after ``episode_length_s`` (a time-out); off, clips run to their end.
        episode_length_s: Episode time limit in seconds.
        auto_reset: mjlab resets ended episodes in ``step`` (off: the caller resets them).
        self_collisions: Keep the body's explicit self-contact pairs.
        model_name: The myo_sim full-body model (see :mod:`.body`).
        play: One env with auto-reset, every clip from its first frame, no time limit (mjlab's ``play``).

    Returns:
        The env config.
    """
    # imported here: importing the mjlab backend registers this task
    from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import (  # noqa: PLC0415
        musclemimic_mujoco_cfg,
    )

    decimation = round(ctrl_dt / TIMESTEP)
    if decimation < 1 or abs(decimation * TIMESTEP - ctrl_dt) > 1e-9:
        raise ValueError(f"ctrl_dt={ctrl_dt} is not a whole multiple of {TIMESTEP} s")
    if goal_stride is None:
        goal_stride = max(1, round(GOAL_SPACING_S / ctrl_dt))
    decoder = ("root_pos", "joint_pos", "root_vel", "joint_vel", "muscle_length", "muscle_velocity", "muscle_force",
               "muscle_excitation", "muscle_activation", "touch", "site_rpos", "site_rangles", "site_rvel")  # fmt: skip
    if not muscle_obs:
        decoder = tuple(
            n
            for n in decoder
            if n not in ("muscle_length", "muscle_velocity", "muscle_force")
        )
    observations = {
        "policy": ObservationGroupCfg(
            terms={
                n: ObservationTermCfg(func=getattr(mdp, n), params=dict(CMD))
                for n in decoder
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "goal": ObservationGroupCfg(
            terms={
                "motion_goal": ObservationTermCfg(
                    func=mdp.motion_goal, params=dict(CMD)
                )
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }
    actions: dict[str, ActionTermCfg] = {
        "muscles": mdp.MuscleExcitationActionCfg(entity_name="robot")
    }
    motion_cmd = MotionCommandCfg(
        entity_name="robot",
        motion_file=motion_file,
        motion_root=motion_root,
        body_names=TRACKED_BODIES,
        sampling_mode=sampling_mode,
        lookahead=lookahead,
        stride=goal_stride,
        bin_fix=bin_fix,
        fail_reset=fail_reset,
    )
    commands: dict[str, CommandTermCfg] = {"motion": motion_cmd}
    # mjlab multiplies the weights by the step dt
    scale = REWARD_SCALE / (TIMESTEP * BASE_DECIMATION)
    rewards: dict[str, RewardTermCfg] = {
        f"mimic_{k}": RewardTermCfg(
            func=getattr(mdp, f"mimic_{k}"),
            weight=w * scale,
            params={**CMD, "w_exp": e},
        )
        for k, (w, e) in MIMIC_REWARD.items()
    }
    dev = {"threshold": 1.0, "soft_width": 0.05}
    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "motion_end": TerminationTermCfg(
            func=mdp.motion_end, params=dict(CMD), time_out=True
        ),
        "body_pos_dev": TerminationTermCfg(
            func=mdp.bad_body_pos_dev, params={**CMD, **dev}
        ),
        "root_pos_dev": TerminationTermCfg(
            func=mdp.bad_root_pos_dev, params={**CMD, **dev}
        ),
        "root_ori_dev": TerminationTermCfg(
            func=mdp.bad_root_ori_dev, params={**CMD, **dev}
        ),
    }
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=num_envs,
            entities={
                "robot": body_cfg(
                    self_collisions=self_collisions, model_name=model_name
                )
            },
            # foot touch: the ground's net force on the bodies of the touch sensor sites (what they read on flat ground)
            sensors=(
                ContactSensorCfg(
                    name=motion_cmd.touch_sensor_name,
                    primary=ContactMatch(
                        mode="body", pattern=FOOT_BODIES, entity="robot"
                    ),
                    secondary=ContactMatch(mode="body", pattern="terrain"),
                    fields=("force",),
                    reduce="netforce",
                ),
            ),
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events={},
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="pelvis",
            distance=3.0,
            fovy=55.0,
            elevation=-5.0,
            azimuth=120.0,
        ),
        # MuscleMimic's per-env MuJoCo-Warp budget for this model (self-contact pairs + 51 equality rows)
        sim=SimulationCfg(
            nconmax=96,
            njmax=768,
            mujoco=musclemimic_mujoco_cfg("fullbody", timestep=TIMESTEP),
        ),
        decimation=decimation,
        episode_length_s=episode_length_s,
        auto_reset=auto_reset,
    )
    if play:
        cfg.scene.num_envs = 1
        cfg.auto_reset = True
        motion_cmd.sampling_mode = "start"
        time_limit = False
    if not time_limit:
        cfg.episode_length_s = 1e9
        cfg.terminations.pop("time_out")
    return cfg


def register() -> None:
    """Register ``myoMimicFullbodyMulticlip-v0`` with mjlab's task registry (MyoSuite's PPO defaults as its runner)
    and its config factory with ``make_env``, which builds it from ``EnvConfig.task_kwargs`` and ``ctrl_dt``."""
    from mjlab.tasks.registry import register_mjlab_task  # noqa: PLC0415

    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (  # noqa: PLC0415
        default_mimic_clip_on_policy_runner_cfg,
    )
    from myosuite.envs.myo.backends.mjlab.tasks.registration import (  # noqa: PLC0415
        register_task_cfg_factory,
    )

    groups = ("policy", "goal")
    register_mjlab_task(
        task_id=ENV_ID,
        env_cfg=mimic_multiclip_env_cfg(),
        play_env_cfg=mimic_multiclip_env_cfg(play=True),
        rl_cfg=default_mimic_clip_on_policy_runner_cfg(
            experiment_name="mimic_fullbody_multiclip",
            obs_groups={"actor": groups, "critic": groups},
        ),
    )
    register_task_cfg_factory(ENV_ID, mimic_multiclip_env_cfg)
