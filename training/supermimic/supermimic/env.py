# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The training env: MyoSuite's ``myoMimicFullbodyMulticlip-v0`` through ``make_env``, with the recipe's options."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip import ENV_ID


@dataclass
class TaskEnvCfg:
    """Options of the task (``EnvConfig.task_kwargs`` and ``ctrl_dt``) the fast recipe trains with.

    The defaults are muscle-supermimic's ``scripts/train_blink.sh`` env flags: 50 Hz control, no muscle observations
    (``-No-Muscle-Obs``), return-based resets with the reset-statistics fix and 30 % of the resets before recent failure
    points, and resets driven by the runner (``auto_reset`` off: it records each terminal state first).
    """

    motion_file: str = ""
    """A clip, a directory of clips or a ``.txt`` clip list (e.g. ``kit_split("train")``)."""
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

    options = asdict(cfg)
    num_envs, ctrl_dt = options.pop("num_envs"), options.pop("ctrl_dt")
    extra = {} if seed is None else {"seed": seed}
    env = make_env(
        EnvConfig(
            ENV_ID,
            backend="mjlab",
            num_envs=num_envs,
            ctrl_dt=ctrl_dt,
            task_kwargs=options,
        ),
        device=device,
        **extra,
    )
    return RslRlVecEnvWrapper(env) if wrap else env


def settings_cfg(settings: dict, **overrides: Any) -> TaskEnvCfg:
    """The task options a checkpoint's ``"env"`` entry (control rate, goal stride, muscle observations) asks for."""
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.env_cfg import TIMESTEP  # noqa: PLC0415

    cfg = TaskEnvCfg(**overrides)
    if "decimation" in settings:
        cfg.ctrl_dt = settings["decimation"] * TIMESTEP
    for key in ("lookahead", "muscle_obs"):
        if key in settings:
            setattr(cfg, key, settings[key])
    if "stride" in settings:
        cfg.goal_stride = settings["stride"]
    return cfg
