# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Collect muscle activations from MuscleMimic policy rollouts.

Rolls a :class:`~...LocalPolicyRunner` through a list of motion clips and
collects ``data.act`` (post-dynamics muscle activations, shape ``(nu,)`` per
step) from fixed-length episodes whose mean per-step tracking reward clears a
percentile threshold. Episodes that fall are rejected, and the activation
ramp after the ``act = 0`` reset is left out.
The filtered activations from all clips are concatenated into a single
``(T, n_muscles)`` array that feeds the SAR extraction pipeline.

Typical usage::

    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        load_local_policy_artifacts, LocalPolicyRunner, FullbodyObsAdapter,
        fullbody_obs_adapter_params_from_metadata,
        read_checkpoint_config_metadata,
    )
    from myosuite.core.trajectory_io import (
        load_motion_clip, resolve_motion_path,
    )
    from myosuite.integrations.musclemimic.activation_collector import (
        CollectionConfig, collect_activations,
    )

    artifacts = load_local_policy_artifacts(checkpoint_root)
    goal_params = fullbody_obs_adapter_params_from_metadata(
        read_checkpoint_config_metadata(checkpoint_root)
    )
    obs_adapter = FullbodyObsAdapter(model, clips[0], goal_params)
    policy = LocalPolicyRunner(artifacts=artifacts, stochastic=False, seed=0,
                               obs_adapter=obs_adapter)

    acts = collect_activations(policy, model, clips,
                               CollectionConfig(n_episodes_per_clip=100))
    # acts: (T, n_muscles), values in [0, 1]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from myosuite.integrations.musclemimic.fullbody_local_policy import FullbodyObsAdapter

if TYPE_CHECKING:
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        LocalPolicyRunner,
        OnnxPolicyRunner,
    )
    from myosuite.core.trajectory_io import MotionClip

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectionConfig:
    """Hyperparameters for activation rollout collection.

    Args:
        n_episodes_per_clip: Number of rollout episodes per motion clip.
        reward_percentile: Only episodes whose mean per-step tracking reward
            reaches this percentile of the preview episodes contribute
            activations. As in the original SAR filter, every episode has the
            same length, so the tracking error of all of them has grown for
            the same time since the start on the reference.
        max_steps_per_episode: Control steps of every episode. ``None`` means
            half the clip; random starts are drawn so that the episode ends
            inside the clip.
        use_random_start: Whether to initialise each episode at a random
            clip frame, improving activation diversity.
        warmup_steps: Leading steps of each episode whose activations are left
            out: every episode starts from ``act = 0``.
        min_root_height: An episode whose free root drops below this height (m)
            has fallen and is rejected. Ignored for models without a free root.
        seed: RNG seed for reproducible random starts.
    """

    n_episodes_per_clip: int = 100
    reward_percentile: int = 80
    max_steps_per_episode: int | None = None
    use_random_start: bool = True
    warmup_steps: int = 10
    min_root_height: float = 0.5
    seed: int = 0


@dataclass
class _ClipRolloutState:
    """Mutable state for one episode within a clip."""

    frame_idx: int = 0
    step_count: int = 0
    activations: list[np.ndarray] = field(default_factory=list)
    total_reward: float = 0.0


def _init_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: MotionClip,
    *,
    start_frame: int = 0,
) -> None:
    """Reset MjData to the clip state at *start_frame*.

    ``mj_resetData`` runs first, so act, ctrl, qvel, time and warm-starts from
    the previous episode never carry over. qvel stays zero when the clip has
    none.
    """
    mujoco.mj_resetData(model, data)
    data.qpos[:] = clip.qpos[start_frame]
    if clip.qvel is not None and clip.qvel.shape[0] > start_frame:
        data.qvel[:] = clip.qvel[start_frame]
    mujoco.mj_forward(model, data)


def _tracking_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: MotionClip,
    frame_idx: int,
) -> float:
    """Negative mean site-tracking error used as episode reward signal.

    Returns 0.0 when site positions are unavailable in the clip.
    """
    from myosuite.integrations.musclemimic.fullbody_native_playback import (
        _compute_site_tracking_error,
    )

    err = _compute_site_tracking_error(model, data, clip, frame_idx)
    return 0.0 if err is None else -float(err)


def _has_fallen(model: mujoco.MjModel, data: mujoco.MjData, min_height: float) -> bool:
    """True when the model has a free root and it is below *min_height*."""
    if not int(model.njnt) or int(model.jnt_type[0]) != mujoco.mjtJoint.mjJNT_FREE:
        return False
    return bool(data.qpos[int(model.jnt_qposadr[0]) + 2] < min_height)


def _run_episode(
    policy_runner: LocalPolicyRunner | OnnxPolicyRunner,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: MotionClip,
    start_frame: int,
    max_steps: int,
    *,
    obs_adapter: FullbodyObsAdapter | None = None,
    warmup_steps: int = 0,
    min_root_height: float | None = None,
) -> tuple[np.ndarray, float]:
    """Run one episode; return ``(activations (T, nu), mean per-step reward)``.

    After each step the data is forwarded and scored against the clip frame
    the step has reached.

    Each episode starts from a fresh MjData state and a reset runner (running
    normaliser and observation history), so it does not depend on the episodes
    run before it.

    Args:
        policy_runner: Loaded policy runner.
        model: Compiled MuJoCo model.
        data: MjData buffer (mutated in-place).
        clip: Motion clip providing reference targets.
        start_frame: Clip frame to initialise from.
        max_steps: Maximum number of control steps.
        obs_adapter: Optional :class:`FullbodyObsAdapter` for this *clip* (must
            match ``clip`` trajectory length when the runner's default adapter
            was built from a different clip).
        warmup_steps: Leading steps whose activations are not returned.
        min_root_height: Fall threshold of the free root (m); ``None`` disables it.

    Returns:
        Tuple of activation array ``(steps - warmup_steps, nu)`` and mean
        per-step tracking reward; a fallen episode returns no activations and
        ``-inf``.
    """
    _init_episode(model, data, clip, start_frame=start_frame)
    policy_runner.reset()

    state = _ClipRolloutState(frame_idx=start_frame)
    n_frames = int(clip.qpos.shape[0])

    while state.step_count < max_steps:
        action = policy_runner.action_for(
            data, clip, state.frame_idx, obs_adapter=obs_adapter
        )
        policy_runner.step(model, data, action)
        mujoco.mj_forward(model, data)
        state.frame_idx = min(state.frame_idx + 1, n_frames - 1)
        state.step_count += 1
        if min_root_height is not None and _has_fallen(model, data, min_root_height):
            return np.empty((0, model.nu), dtype=np.float32), -np.inf
        if state.step_count > warmup_steps:
            state.activations.append(np.asarray(data.act, dtype=np.float32).copy())
        state.total_reward += _tracking_reward(model, data, clip, state.frame_idx)
        if state.frame_idx >= n_frames - 1:
            break

    acts = (
        np.stack(state.activations)
        if state.activations
        else np.empty((0, model.nu), dtype=np.float32)
    )
    return acts, state.total_reward / max(state.step_count, 1)


def collect_activations_from_clip(
    policy_runner: LocalPolicyRunner | OnnxPolicyRunner,
    model: mujoco.MjModel,
    clip: MotionClip,
    config: CollectionConfig,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Collect high-reward muscle activations from one motion clip.

    Args:
        policy_runner: Loaded policy runner.
        model: Compiled MuJoCo model.
        clip: Motion clip to track.
        config: Collection hyperparameters.
        rng: NumPy random generator (for reproducible random starts).

    Returns:
        Filtered activation array ``(T_kept, nu)``, dtype ``float32``.
        Empty array with shape ``(0, nu)`` if no episode clears the threshold.
    """
    n_frames = int(clip.qpos.shape[0])
    max_steps = min(
        config.max_steps_per_episode or max(1, (n_frames - 1) // 2),
        max(1, n_frames - 1),
    )
    if config.warmup_steps >= max_steps:
        raise ValueError(
            f"warmup_steps={config.warmup_steps} leaves no activations of a "
            f"{max_steps}-step episode"
        )
    last_start = n_frames - 1 - max_steps
    episode = {
        "max_steps": max_steps,
        "warmup_steps": config.warmup_steps,
        "min_root_height": config.min_root_height,
    }
    data = mujoco.MjData(model)
    clip_obs_adapter: FullbodyObsAdapter | None = None
    if policy_runner.obs_adapter is not None:
        clip_obs_adapter = policy_runner.obs_adapter.with_clip(clip)

    # ------------------------------------------------------------------
    # Pass 1: estimate the mean-reward threshold from preview episodes.
    # ------------------------------------------------------------------
    n_preview = max(10, config.n_episodes_per_clip // 10)
    preview_rewards: list[float] = []
    for _ in range(n_preview):
        start = int(rng.integers(0, last_start + 1)) if config.use_random_start else 0
        _, r = _run_episode(
            policy_runner,
            model,
            data,
            clip,
            start,
            obs_adapter=clip_obs_adapter,
            **episode,
        )
        preview_rewards.append(r)

    standing = [r for r in preview_rewards if np.isfinite(r)]
    threshold = (
        float(np.percentile(standing, config.reward_percentile))
        if standing
        else -np.inf
    )
    logger.debug(
        "Clip %s: mean-reward threshold (p%d over %d preview) = %.4f",
        getattr(clip, "source_path", "?"),
        config.reward_percentile,
        n_preview,
        threshold,
    )

    # ------------------------------------------------------------------
    # Pass 2: full collection, keep episodes above threshold.
    # ------------------------------------------------------------------
    kept: list[np.ndarray] = []
    for _ in range(config.n_episodes_per_clip):
        start = int(rng.integers(0, last_start + 1)) if config.use_random_start else 0
        acts, r = _run_episode(
            policy_runner,
            model,
            data,
            clip,
            start,
            obs_adapter=clip_obs_adapter,
            **episode,
        )
        if np.isfinite(r) and r >= threshold and acts.shape[0] > 0:
            kept.append(acts)

    if not kept:
        logger.warning(
            "Clip %s: no episodes cleared the reward threshold — "
            "returning empty activation array.",
            getattr(clip, "source_path", "?"),
        )
        return np.empty((0, model.nu), dtype=np.float32)

    result = np.concatenate(kept, axis=0)
    logger.info(
        "Clip %s: kept %d/%d episodes → %d activation frames",
        getattr(clip, "source_path", "?"),
        len(kept),
        config.n_episodes_per_clip,
        result.shape[0],
    )
    return result


def collect_activations(
    policy_runner: LocalPolicyRunner | OnnxPolicyRunner,
    model: mujoco.MjModel,
    clips: list[MotionClip],
    config: CollectionConfig | None = None,
    *,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Collect muscle activations across multiple motion clips.

    Iterates over all clips, filters per-clip activations by reward
    percentile, and concatenates into one array.  When *cache_path* is
    given and the file exists, it is loaded directly; otherwise the result
    is saved there after collection.

    Args:
        policy_runner: Loaded policy runner.
        model: Compiled MuJoCo model.
        clips: List of motion clips to collect from.
        config: Collection hyperparameters.  Defaults to
            :class:`CollectionConfig` with all defaults.
        cache_path: Optional ``.npy`` path to cache / load results.

    Returns:
        Concatenated activation array ``(T_total, n_muscles)``, dtype
        ``float32``, values in ``[0, 1]``.

    Raises:
        ValueError: If no activations were collected from any clip.
    """
    cfg = config or CollectionConfig()

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            acts = np.load(cache_path)
            logger.info(
                "Loaded cached activations from %s (%s)", cache_path, acts.shape
            )
            return acts

    rng = np.random.default_rng(cfg.seed)
    all_acts: list[np.ndarray] = []

    for i, clip in enumerate(clips):
        logger.info(
            "Collecting activations from clip %d/%d: %s",
            i + 1,
            len(clips),
            getattr(clip, "source_path", "?"),
        )
        clip_acts = collect_activations_from_clip(
            policy_runner, model, clip, cfg, rng=rng
        )
        if clip_acts.shape[0] > 0:
            all_acts.append(clip_acts)

    if not all_acts:
        raise ValueError(
            "No activations collected from any clip. "
            "Check that the policy produces above-threshold rewards."
        )

    result = np.concatenate(all_acts, axis=0).astype(np.float32)
    logger.info(
        "Total activations collected: %s from %d clips",
        result.shape,
        len(all_acts),
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, result)
        logger.info("Saved activations to %s", cache_path)

    return result


__all__ = [
    "CollectionConfig",
    "collect_activations",
    "collect_activations_from_clip",
]
