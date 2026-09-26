# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Find and load trained-policy checkpoints (mjlab / RSL-RL or Stable-Baselines3)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

Policy = Callable[[np.ndarray], np.ndarray]


def mjlab_experiment(env_id: str) -> str | None:
    """Log-directory name of the env's default mjlab training run.

    Args:
        env_id: Registered env id.

    Returns:
        ``experiment_name`` of the mjlab runner config, or ``None`` when mjlab is not
        installed or the env has no mjlab twin.
    """
    try:
        import myosuite.envs.myo.backends.mjlab  # noqa: F401  (registers the twins)
        from mjlab.tasks.registry import load_rl_cfg

        return load_rl_cfg(env_id).experiment_name
    except Exception as err:  # noqa: BLE001  (mjlab missing, or no twin for env_id)
        print(f"No mjlab run lookup for {env_id}: {err}")
        return None


def find_checkpoint(
    env_id: str,
    checkpoint: str | Path | None = None,
    roots: Sequence[Path] = (Path.cwd(),),
    sb3_zip: str | None = None,
) -> Path | None:
    """Locate a trained policy for *env_id*.

    Args:
        env_id: Registered env id.
        checkpoint: Explicit checkpoint (returned as is when given).
        roots: Directories searched for ``logs/rsl_rl/<experiment>/<run>/model_*.pt``
            (newest run and iteration wins), then for the repository's default policy
            ``baselines/checkpoints/<env_id>/model_*.pt``, then for *sb3_zip*.
        sb3_zip: Optional file name of a Stable-Baselines3 checkpoint to fall back to.

    Returns:
        Path of the checkpoint, or ``None`` when none was found.
    """
    if checkpoint is not None:
        return Path(checkpoint)
    experiment = mjlab_experiment(env_id)
    for root in roots:
        runs = sorted(
            (root / "logs" / "rsl_rl" / experiment).glob("*/model_*.pt")
            if experiment
            else [],
            key=lambda c: (c.parent.name, int(c.stem.split("_")[-1])),
        )
        if runs:
            return runs[-1]
    for root in roots:
        baseline = sorted(
            (root / "baselines" / "checkpoints" / env_id).glob("model_*.pt"),
            key=lambda c: int(c.stem.split("_")[-1]),
        )
        if baseline:
            return baseline[-1]
    for root in roots:
        if sb3_zip and (root / sb3_zip).is_file():
            return root / sb3_zip
    return None


def load_policy(env: Any, checkpoint: Path | None) -> Policy:
    """Return ``act(obs) -> action`` for a checkpoint, driving *env* with raw observations.

    Args:
        env: Gymnasium env the policy will act in (its spaces are used).
        checkpoint: mjlab ``model_*.pt`` or run directory, an SB3 ``.zip``, or ``None``.

    Returns:
        A deterministic policy; a random one when there is no usable checkpoint (none
        given, SB3 missing, or an SB3 policy trained on a different observation space).
    """

    def random_policy(_obs: np.ndarray) -> np.ndarray:
        return env.action_space.sample()

    if checkpoint is None:
        print("No checkpoint found; using a random policy.")
        return random_policy
    if checkpoint.suffix == ".zip":  # Stable-Baselines3
        try:
            from stable_baselines3 import PPO
        except ImportError:
            print("stable-baselines3 is not installed; using a random policy.")
            return random_policy
        model = PPO.load(checkpoint)
        if model.observation_space.shape != env.observation_space.shape:
            print(
                f"{checkpoint} was trained on another env (obs "
                f"{model.observation_space.shape} vs {env.observation_space.shape}); "
                "using a random policy."
            )
            return random_policy
        print(f"SB3 policy: {checkpoint}")
        return lambda obs: model.predict(obs, deterministic=True)[0]
    from myosuite.utils.rslrl_policy import load_rslrl_policy  # mjlab / RSL-RL

    if checkpoint.is_dir():
        checkpoint = max(
            checkpoint.glob("model_*.pt"), key=lambda c: int(c.stem.split("_")[-1])
        )
    policy = load_rslrl_policy(checkpoint, env.action_space.shape[0])
    print(f"mjlab policy: {checkpoint}")
    return lambda obs: policy.act(np.atleast_2d(obs))[0]  # the policy takes batches
