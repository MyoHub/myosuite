# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The training env: MyoSuite's ``myoMimic-v0`` through ``make_env``, with the recipe's options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from myosuite.envs.myo.tasks.mimic.tracking import MIMIC_ENV_ID


@dataclass
class TaskEnvCfg:
    """Options of the task (overrides of its ``supermimic`` preset, and ``ctrl_dt``) the fast recipe trains with.

    The defaults are muscle-supermimic's ``scripts/train_blink.sh`` env flags: 50 Hz control, no muscle observations
    (``-No-Muscle-Obs``), return-based resets with the reset-statistics fix and 30 % of the resets before recent failure
    points, and resets driven by the runner (``auto_reset`` off: it records each terminal state first).
    """

    motion_file: str = "kit_all"
    """A clip, a directory of clips, a ``.txt`` clip list or a KIT set name (``"kit_all"``: all 1080 clips, for a
    policy to release or reuse; ``"kit_train"`` / ``"kit_test"``: MuscleMimic's split, for benchmarks)."""
    motion_root: str = ""
    """The directory a list's paths start from (the dataset root of the packaged KIT lists)."""
    num_envs: int = 1024
    ctrl_dt: float = 0.02
    muscle_obs: bool = False
    sampling_mode: Literal["uniform", "return", "start"] = "return"
    bin_fix: bool = True
    fail_reset: float = 0.3
    lookahead: int = 5
    goal_stride: int | None = None
    """Control steps between goal frames (default: 0.2 s)."""
    time_limit: bool = True
    auto_reset: bool = False


TIMESTEP = 0.002
"""The body's physics step (the task's models)."""
_MUSCLE_OBS = ("muscle_length", "muscle_velocity", "muscle_force")


def task_overrides(cfg: TaskEnvCfg) -> dict[str, Any]:
    """The task-config overrides (``MimicTaskCfg`` dotted keys) of the recipe's options."""
    from myosuite.envs.myo.tasks.mimic.tracking.config import PRESETS  # noqa: PLC0415

    policy = tuple(n for n in PRESETS["supermimic"].obs.policy if n not in _MUSCLE_OBS)
    if cfg.muscle_obs:  # MuscleMimic's order: after the joint velocities
        i = policy.index("joint_vel") + 1
        policy = policy[:i] + _MUSCLE_OBS + policy[i:]
    out: dict[str, Any] = {
        "motion.clips": cfg.motion_file,
        "motion.root": cfg.motion_root,
        "obs.policy": policy,
        "obs.goal_frames": cfg.lookahead,
        "reset.mode": cfg.sampling_mode,
        "reset.bin_fix": cfg.bin_fix,
        "reset.fail_reset": cfg.fail_reset,
        "reset.auto_reset": cfg.auto_reset,
    }
    if cfg.goal_stride is not None:
        out["obs.goal_spacing_s"] = cfg.goal_stride * cfg.ctrl_dt
    if not cfg.time_limit:
        out["episode_length_s"] = None
    return out


def make_task_env(
    cfg: TaskEnvCfg, device: str, seed: int | None = None, wrap: bool = True
) -> Any:
    """Build the task with ``make_env(..., backend="mjlab")``.

    Args:
        cfg: The task options.
        device: Torch device of the simulation.
        seed: Env seed (``None``: mjlab draws one).
        wrap: Return mjlab's RSL-RL wrapper (what the runner takes) instead of the manager env.

    Returns:
        The env.
    """
    from mjlab.rl import RslRlVecEnvWrapper  # noqa: PLC0415

    from myosuite import make_env  # noqa: PLC0415
    from myosuite.core.config import EnvConfig  # noqa: PLC0415

    extra = {} if seed is None else {"seed": seed}
    env = make_env(
        EnvConfig(
            MIMIC_ENV_ID,
            backend="mjlab",
            num_envs=cfg.num_envs,
            ctrl_dt=cfg.ctrl_dt,
            task_kwargs={"preset": "supermimic", "overrides": task_overrides(cfg)},
        ),
        device=device,
        **extra,
    )
    return RslRlVecEnvWrapper(env) if wrap else env


def settings_cfg(settings: dict, **overrides: Any) -> TaskEnvCfg:
    """The task options a checkpoint's ``"env"`` entry (control rate, goal stride, muscle observations) asks for."""
    cfg = TaskEnvCfg(**overrides)
    if "decimation" in settings:
        cfg.ctrl_dt = settings["decimation"] * TIMESTEP
    for key in ("lookahead", "muscle_obs"):
        if key in settings:
            setattr(cfg, key, settings[key])
    if "stride" in settings:
        cfg.goal_stride = settings["stride"]
    return cfg
