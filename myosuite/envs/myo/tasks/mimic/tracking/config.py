# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Typed task config of ``myoMimic-v0`` and its presets.

A preset is a :class:`MimicTaskCfg`; ``EnvConfig.task_kwargs`` picks one (``preset``)
and overrides any field with dotted keys (``overrides``)::

    make_env(EnvConfig(MIMIC_ENV_ID, task_kwargs={"overrides": {"motion.clips": "walk.npz"}}))

The control step is not part of the task: it comes from ``EnvConfig.ctrl_dt``. Pure
Python (no simulator import), so checkpoints and methods can read it anywhere.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

ResetMode = Literal["uniform", "return", "start"]


@dataclass(frozen=True)
class MotionCfg:
    """Reference motion.

    Attributes:
        clips: A clip (``.npz``), a directory, a ``.txt`` list, or ``"kit_train"``
            (the default: MuscleMimic's 972 KIT training motions, the clips its
            released checkpoints were trained on).
        root: Directory the paths of a list start from (default for the KIT names:
            ``$MYOSUITE_KIT_ROOT``, the local retarget cache or the download, see
            :func:`~myosuite.envs.myo.tasks.mimic.tracking.kit_root`; else the
            list's own directory).
        bin_s: Length of the reset bins in seconds.
        workers: Threads preprocessing clips (one by default; benchmark your clips).
        cache_dir: Optional directory for validated, persistent NumPy motion data.
    """

    clips: str = "kit_train"
    root: str = ""
    bin_s: float = 1.0
    workers: int = 1
    cache_dir: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or self.workers < 1
        ):
            raise ValueError("motion.workers must be a positive integer")


@dataclass(frozen=True)
class ObsCfg:
    """Observation groups.

    Attributes:
        policy: Body-state terms (:data:`myosuite.terms.mimic_track.OBSERVATIONS`), in order.
        goal_frames: Reference frames in the goal, the current one included.
        goal_spacing_s: Time between them.
        goal_frame: A tracked body (not the anchor): the goal is then ``goal_bodies``'
            reference positions in its frame, now and at each future frame, and its own
            future motion in its current frame (``mimic_track.goal_observation``), e.g.
            fingertips in a palm frame; ``""``: the lookahead goal of every tracked body,
            root motion and phase.
        goal_bodies: The tracked bodies of a ``goal_frame`` goal.
    """

    policy: tuple[str, ...] = (
        "root_pos", "joint_pos", "root_vel", "joint_vel", "muscle_excitation",
        "muscle_activation", "touch", "site_rpos", "site_rangles", "site_rvel",
    )  # fmt: skip
    goal_frames: int = 5
    goal_spacing_s: float = 0.2
    goal_frame: str = ""
    goal_bodies: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewardTermCfg:
    """One MimicReward component: ``weight * exp(-kernel * mse)``."""

    name: str
    weight: float
    kernel: float


@dataclass(frozen=True)
class RewardCfg:
    """The proposed dense reward (methods may replace it).

    A step earns ``scale * (ctrl_dt / ref_dt) * sum_k weight_k * c_k``: the weights
    are per ``ref_dt`` of motion, so the return per second does not depend on the
    control step.

    Attributes:
        terms: The components.
        scale: Overall factor (MuscleMimic's full-body weights x 0.1).
        ref_dt: Control step the weights are set for.
    """

    terms: tuple[RewardTermCfg, ...] = (
        RewardTermCfg("qpos", 0.1, 10.0),
        RewardTermCfg("qvel", 0.1, 2.0),
        RewardTermCfg("root_pos", 0.1, 10.0),
        RewardTermCfg("root_vel", 0.1, 10.0),
        RewardTermCfg("rpos", 0.6, 100.0),
        RewardTermCfg("rquat", 0.01, 10.0),
        RewardTermCfg("rvel_rot", 0.1, 0.1),
        RewardTermCfg("rvel_lin", 0.1, 0.1),
    )
    scale: float = 0.1
    ref_dt: float = 0.01

    def step_weights(self, ctrl_dt: float) -> dict[str, float]:
        """Per-step weight of each component at *ctrl_dt*."""
        return {
            t.name: t.weight * self.scale * ctrl_dt / self.ref_dt for t in self.terms
        }


@dataclass(frozen=True)
class FailureCfg:
    """The failure rule (a terminal state): any deviation above its threshold.

    Attributes:
        rel_site: Mean distance of the relative sites (m).
        root_pos: Root position distance (m); ``None``: not checked (a fixed root).
        root_ori: Root geodesic angle (rad); ``None``: not checked (a fixed root).
    """

    rel_site: float = 1.0
    root_pos: float | None = 1.0
    root_ori: float | None = 1.0

    def thresholds(self) -> dict[str, float]:
        """``deviation name -> threshold`` of the checked deviations."""
        out = {
            "rel_site": self.rel_site,
            "root_pos": self.root_pos,
            "root_ori": self.root_ori,
        }
        return {k: v for k, v in out.items() if v is not None}


@dataclass(frozen=True)
class ResetCfg:
    """Episode starts (a training choice; evaluation plays every clip from its start).

    Attributes:
        mode: ``"uniform"`` over the bins, ``"return"`` (SuperMimic's return-based bins)
            or ``"start"`` (env i at the start of clip ``i % C``).
        bin_fix: Average the episodes of one bin before the EMA step (BlinkMimic).
        fail_reset: Share of resets sent one bin before recent failures (InstantTrack).
        gamma: Discount of the bins' return estimates.
        init_activation: Muscle activation at reset.
        auto_reset: mjlab resets ended episodes in ``step`` (off: the caller does).
    """

    mode: ResetMode = "uniform"
    bin_fix: bool = False
    fail_reset: float = 0.0
    gamma: float = 0.99
    init_activation: float = 0.0
    auto_reset: bool = True


@dataclass(frozen=True)
class MimicTaskCfg:
    """The full MDP of ``myoMimic-v0`` apart from the control step and the body model.

    Attributes:
        motion: Reference clips.
        obs: Observation groups.
        reward: Dense reward.
        failure: Failure rule.
        reset: Episode starts.
        episode_length_s: Time limit (a truncation); ``None``: clips run to their end.
        self_collisions: Keep the body's self-contact pairs.
        reward_fn: A method's own step reward ``reward_fn(state, ref, ctrl_dt)`` replacing
            :attr:`reward` (``None``: the proposed reward). It gets the
            :class:`~myosuite.terms.mimic_track.TrackState` of the post-step state and the
            :class:`~myosuite.terms.mimic_track.TrackRef` of frame f + 1, batched NumPy or
            torch arrays (:func:`~myosuite.terms.mimic_track.array_module`), and returns the
            reward per step, ``(...,)``. The proposed reward is still reported (CPU
            ``info["task_reward"]``, the twin's ``task_reward``). A whole replacement
            ``RewardCfg`` of the library's components is an override of ``"reward"``.
    """

    motion: MotionCfg = field(default_factory=MotionCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    failure: FailureCfg = field(default_factory=FailureCfg)
    reset: ResetCfg = field(default_factory=ResetCfg)
    episode_length_s: float | None = 300.0
    self_collisions: bool = True
    reward_fn: Callable[[Any, Any, float], Any] | None = None

    def fingerprint(self) -> str:
        """Stable hash of the config (store it with checkpoints)."""
        data = dataclasses.asdict(self)
        for name in ("workers", "cache_dir"):
            data["motion"].pop(name)
        blob = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


SUPERMIMIC = MimicTaskCfg(
    reset=ResetCfg(mode="return", bin_fix=True, fail_reset=0.3),
)
"""SuperMimic's MDP (the default): no muscle length/velocity/force observations,
return-based resets with the bin fix and 30 % of the resets before recent failures."""

PRESETS: dict[str, MimicTaskCfg] = {"supermimic": SUPERMIMIC}
DEFAULT_PRESET = "supermimic"


ROOT_OBSERVATIONS = ("root_pos", "root_vel")
ROOT_REWARDS = ("root_pos", "root_vel")


def for_body(task: MimicTaskCfg, fixed_root: bool, touch: bool) -> MimicTaskCfg:
    """The task on a body: with a fixed root, no root observations, root reward
    components or root failure checks; without contact bodies, no touch observation.
    Unchanged for a free root with contact bodies (the full body).

    Args:
        task: The resolved task.
        fixed_root: The body has no free joint.
        touch: The body has contact bodies.

    Returns:
        The config.
    """
    drop = (ROOT_OBSERVATIONS if fixed_root else ()) + (() if touch else ("touch",))
    if drop:
        policy = tuple(n for n in task.obs.policy if n not in drop)
        task = dataclasses.replace(
            task, obs=dataclasses.replace(task.obs, policy=policy)
        )
    if fixed_root:
        terms = tuple(t for t in task.reward.terms if t.name not in ROOT_REWARDS)
        task = dataclasses.replace(
            task,
            reward=dataclasses.replace(task.reward, terms=terms),
            failure=dataclasses.replace(task.failure, root_pos=None, root_ori=None),
        )
    return task


def _replace_path(obj: Any, path: list[str], value: Any) -> Any:
    name = path[0]
    if not dataclasses.is_dataclass(obj) or name not in {
        f.name for f in dataclasses.fields(obj)
    }:
        raise KeyError(f"{type(obj).__name__} has no field {name!r}")
    if len(path) > 1:
        value = _replace_path(getattr(obj, name), path[1:], value)
    elif isinstance(value, list):
        value = tuple(value)
    return dataclasses.replace(obj, **{name: value})


def resolve_task(
    preset: str | MimicTaskCfg = DEFAULT_PRESET,
    overrides: Mapping[str, Any] | None = None,
) -> MimicTaskCfg:
    """The task config of a preset with dotted-key overrides.

    Args:
        preset: A key of :data:`PRESETS`, or a config.
        overrides: ``{"motion.clips": ..., "reset.mode": ...}``.

    Returns:
        The resolved config.

    Raises:
        KeyError: For an unknown preset or field.
    """
    if isinstance(preset, MimicTaskCfg):
        cfg = preset
    elif preset in PRESETS:
        cfg = PRESETS[preset]
    else:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    for key, value in (overrides or {}).items():
        cfg = _replace_path(cfg, key.split("."), value)
    return cfg
