# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Find and load trained-policy checkpoints (mjlab / RSL-RL or Stable-Baselines3)."""

from __future__ import annotations

import pickle
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

Policy = Callable[[np.ndarray], np.ndarray]

# ``env_id: <id>`` line of a run's ``params/env.yaml`` (the task's ``CpuTaskSpec``). The
# dump holds ``!!python/...`` tags, so it is matched as text rather than YAML-loaded.
_ENV_ID_LINE = re.compile(r"^[ \t-]*env_id:[ \t]*['\"]?([^'\"\s]+)", re.MULTILINE)


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


def _run_is_for(run_dir: Path, env_id: str) -> bool:
    """Whether a ``logs/rsl_rl/<experiment>/<run>`` directory was trained on *env_id*.

    Several env ids share one ``experiment_name`` (e.g. all ``myo_elbow_pose`` envs,
    whose Exo members have 7 actuators instead of 6). A run that records no env id in
    its ``params/env.yaml`` (e.g. Table Tennis) is kept.
    """
    params = run_dir / "params" / "env.yaml"
    if not params.is_file():
        return True
    recorded = set(_ENV_ID_LINE.findall(params.read_text(errors="replace")))
    return not recorded or env_id in recorded


def resume_checkpoint(
    experiment_dir: Path,
    env_id: str,
    load_run: str = ".*",
    load_checkpoint: str = "model_.*.pt",
) -> Path:
    """Checkpoint that ``scripts/train_mjlab.py <env_id> --agent.resume`` continues from.

    mjlab's ``get_checkpoint_path`` picks the newest run of the experiment matching
    *load_run*, whatever env id it was trained on. Here only runs trained on *env_id*
    qualify (see :func:`_run_is_for`), unless *load_run* names a run exactly, and runs
    without a checkpoint matching *load_checkpoint* (e.g. a run that crashed before
    its first save) are skipped. The newest checkpoint of the newest such run wins.

    Args:
        experiment_dir: ``logs/rsl_rl/<experiment_name>``.
        env_id: Task being trained.
        load_run: Regex of run directory names (``--agent.load-run``).
        load_checkpoint: Regex of checkpoint file names (``--agent.load-checkpoint``).

    Returns:
        Path of the checkpoint.

    Raises:
        ValueError: No run qualifies.
    """
    from mjlab.utils.os import get_checkpoint_path  # noqa: PLC0415  (optional dep)

    runs = sorted(
        run.name
        for run in (experiment_dir.iterdir() if experiment_dir.is_dir() else ())
        if run.is_dir()
        and run.name != "wandb_checkpoints"  # mjlab's W&B download cache
        and re.match(load_run, run.name)
        and (run.name == load_run or _run_is_for(run, env_id))
        and any(re.match(load_checkpoint, f.name) for f in run.iterdir())
    )
    if not runs:
        raise ValueError(
            f"No run of {env_id} in {experiment_dir} matches {load_run!r} and holds a "
            f"checkpoint matching {load_checkpoint!r} (runs whose params/env.yaml "
            "records another env id are skipped; pass --agent.load-run <run> to pick "
            "one explicitly)."
        )
    return get_checkpoint_path(
        experiment_dir, re.escape(runs[-1]) + "$", load_checkpoint
    )


def default_roots(max_levels: int = 4) -> tuple[Path, ...]:
    """Directories searched for local runs: the working directory and its parents.

    The parents are included up to the MyoSuite checkout (the directory holding
    ``myosuite/`` or ``.git``), so a notebook started in ``tutorials/`` still finds the
    ``logs/`` of the checkout. Outside a checkout (a pip install, Colab) it is the working
    directory only.

    Args:
        max_levels: Most parent levels to look at.

    Returns:
        The working directory first, then its parents.
    """
    cwd = Path.cwd().resolve()
    for level, parent in enumerate(cwd.parents[:max_levels], start=1):
        if (parent / "myosuite").is_dir() or (parent / ".git").exists():
            return (cwd, *cwd.parents[:level])
    return (cwd,)


def find_checkpoint(
    env_id: str,
    checkpoint: str | Path | None = None,
    roots: Sequence[Path] | None = None,
    sb3_zip: str | None = None,
) -> Path | None:
    """Locate a trained policy for *env_id*.

    Args:
        env_id: Registered env id.
        checkpoint: Explicit checkpoint (returned as is when given).
        roots: Directories searched (default: :func:`default_roots`) for
            ``logs/rsl_rl/<experiment>/<run>/model_*.pt``
            (newest run and iteration wins; runs whose ``params/env.yaml`` records
            another env id are skipped), then for the repository's default policy
            ``baselines/checkpoints/<env_id>/model_*.pt``, then for *sb3_zip*.
        sb3_zip: Optional file name of a Stable-Baselines3 checkpoint to fall back to.

    Returns:
        Path of the checkpoint, or ``None`` when none was found. When no local baseline
        exists in *roots* either, this downloads it from the Hugging Face baselines repo
        (:func:`myosuite.core.hf_io.download_baseline_checkpoint`) before giving up.
    """
    if checkpoint is not None:
        return Path(checkpoint)
    roots = default_roots() if roots is None else roots
    experiment = mjlab_experiment(env_id)
    for root in roots:
        runs = sorted(
            (
                c
                for c in (root / "logs" / "rsl_rl" / experiment).glob("*/model_*.pt")
                if _run_is_for(c.parent, env_id)
            )
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
    from myosuite.core.hf_io import download_baseline_checkpoint  # noqa: PLC0415

    hf_dir = download_baseline_checkpoint(env_id)
    if hf_dir is not None:
        from myosuite.utils.checkpoint_manifest import check_contract  # noqa: PLC0415

        check_contract(env_id, hf_dir)
        hf_ckpts = sorted(
            hf_dir.glob("model_*.pt"), key=lambda c: int(c.stem.split("_")[-1])
        )
        if hf_ckpts:
            return hf_ckpts[-1]
    for root in roots:
        if sb3_zip and (root / sb3_zip).is_file():
            return root / sb3_zip
    return None


def load_sb3_model(checkpoint: str | Path, device: str = "cpu") -> Any:
    """Load a Stable-Baselines3 ``.zip`` saved by SAC, TD3 or PPO.

    Args:
        checkpoint: File written by ``model.save()``.
        device: Torch device for the policy.

    Returns:
        The loaded model (the first of SAC, TD3, PPO that loads it).
    """
    from stable_baselines3 import PPO, SAC, TD3

    errors = []
    for cls in (SAC, TD3, PPO):
        try:
            return cls.load(checkpoint, device=device)
        except Exception as err:  # noqa: BLE001 — saved by another algorithm
            errors.append(f"{cls.__name__}: {type(err).__name__}: {err}")
    raise ValueError(
        f"Could not load {checkpoint} as SAC, TD3 or PPO:\n" + "\n".join(errors)
    )


# ``VecNormalize.save()`` files of an SB3 checkpoint, looked up in its directory:
# ``<stem>_vecnormalize.pkl``, then the RL Zoo and earlier MyoSuite names.
_VEC_NORMALIZE_NAMES = (
    "{stem}_vecnormalize.pkl",
    "vecnormalize.pkl",
    "vec_normalize.pkl",
)


def find_vec_normalize(checkpoint: str | Path) -> Path | None:
    """Locate the ``VecNormalize`` statistics saved next to an SB3 checkpoint.

    Args:
        checkpoint: SB3 ``.zip`` checkpoint.

    Returns:
        ``<stem>_vecnormalize.pkl``, ``vecnormalize.pkl`` or ``vec_normalize.pkl``
        in the checkpoint's directory (the first that exists), or ``None``.
    """
    checkpoint = Path(checkpoint)
    for name in _VEC_NORMALIZE_NAMES:
        candidate = checkpoint.with_name(name.format(stem=checkpoint.stem))
        if candidate.is_file():
            return candidate
    return None


def load_vec_normalize(path: str | Path) -> Any:
    """Load the ``VecNormalize`` written by its ``save()`` for inference.

    ``VecNormalize.load`` would also need a VecEnv to wrap, but normalizing
    observations does not. The statistics are frozen (``training=False``) and
    rewards are left unnormalized, as for SB3 evaluation.

    Args:
        path: File written by ``VecNormalize.save()`` (a pickle: load trusted files only).

    Returns:
        The ``VecNormalize``; use its ``normalize_obs`` before ``model.predict``.
    """
    with Path(path).open("rb") as f:
        vec_normalize = pickle.load(f)  # the format of VecNormalize.save
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    return vec_normalize


def sb3_policy(model: Any, vec_normalize: Any | None = None) -> Policy:
    """Deterministic ``act(raw_obs)`` of an SB3 model trained behind *vec_normalize*.

    Args:
        model: Loaded SB3 model.
        vec_normalize: Its ``VecNormalize`` (see :func:`load_vec_normalize`), if any.

    Returns:
        ``model.predict(vec_normalize.normalize_obs(obs), deterministic=True)``.
    """

    def act(obs: np.ndarray) -> np.ndarray:
        if vec_normalize is not None:
            obs = vec_normalize.normalize_obs(obs)
        return model.predict(obs, deterministic=True)[0]

    return act


def load_policy(env: Any, checkpoint: Path | None) -> Policy:
    """Return ``act(obs) -> action`` for a checkpoint, driving *env* with raw observations.

    Args:
        env: Gymnasium env the policy will act in (its spaces are used).
        checkpoint: mjlab ``model_*.pt`` or run directory, an SB3 (SAC/TD3/PPO)
            ``.zip``, or ``None``. The ``VecNormalize`` statistics saved next to an
            SB3 checkpoint (:func:`find_vec_normalize`) normalize its observations.

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
            model = load_sb3_model(checkpoint)
        except ImportError:
            print("stable-baselines3 is not installed; using a random policy.")
            return random_policy
        if model.observation_space.shape != env.observation_space.shape:
            print(
                f"{checkpoint} was trained on another env (obs "
                f"{model.observation_space.shape} vs {env.observation_space.shape}); "
                "using a random policy."
            )
            return random_policy
        stats = find_vec_normalize(checkpoint)
        print(f"SB3 policy: {checkpoint} (VecNormalize: {stats})")
        return sb3_policy(model, load_vec_normalize(stats) if stats else None)
    from myosuite.utils.rslrl_policy import load_rslrl_policy  # mjlab / RSL-RL

    if checkpoint.is_dir():
        checkpoint = max(
            checkpoint.glob("model_*.pt"), key=lambda c: int(c.stem.split("_")[-1])
        )
    policy = load_rslrl_policy(checkpoint, env.action_space.shape[0])
    print(f"mjlab policy: {checkpoint}")
    return lambda obs: policy.act(np.atleast_2d(obs))[0]  # the policy takes batches
