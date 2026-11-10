# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""PPO (mjlab / RSL-RL) on a registered MyoSuite task through DynSyn muscle groups.

The task's env and runner configs come from mjlab's task registry (MyoSuite registers
its twins there); DynSyn only puts an action term in front of the task's muscle term
(:func:`~myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map.apply_action_map`).
Every checkpoint stores the map and its training progress, so
``scripts/eval_mjlab_policy.py`` plays it without this package::

    python -m myosuite_dynsyn.train myoLegWalk-v0 --groups myoLegWalk-v0_k40.json \\
        --env.scene.num-envs 1024
    python -m myosuite_dynsyn.train myoLegWalk-v0 --resume logs/rsl_rl/.../model_500.pt

Muscle-command features: ``--feature NAME[=JSON]`` as in ``scripts/train_mjlab.py``.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map import (
    action_map_term,
    apply_action_map,
    apply_trained_action_map,
)
from myosuite.utils.action_map import DynSyn, DynSynGroups, trained_action_map
from myosuite.utils.checkpoint_manifest import ACTION_MAP_FIELD
from myosuite.utils.checkpoint_utils import resume_checkpoint
from myosuite.utils.feature_cli import parse_feature_args


@dataclass(frozen=True)
class DynSynTrainConfig:
    """Task configs (``--env.*``, ``--agent.*``) plus the DynSyn options."""

    env: ManagerBasedRlEnvCfg
    agent: RslRlOnPolicyRunnerCfg
    groups: str | None = None
    """Muscle groups from ``python -m myosuite_dynsyn.groups``. The policy then commands
    one action per group plus a bounded correction per muscle. Not needed to resume."""
    start_steps: float = 5e7
    """Environment steps (all envs together) before the per-muscle corrections start."""
    ramp_steps: float = 1e8
    """Environment steps over which the correction bound grows to ``max_amplitude``."""
    max_amplitude: float = 0.1
    """Largest relative correction of a muscle's group action, before the task's
    sigmoid (0: groups only, one action per group)."""
    resume: str | None = None
    """``model_*.pt`` to continue (a local run or a W&B download); it brings its own
    groups and schedule. ``--agent.resume`` picks the newest local run instead."""
    device: str | None = None
    """Torch device; default ``cuda:0`` if available, else ``cpu``."""
    log_root: str = "logs/rsl_rl"

    @staticmethod
    def from_task(task_id: str, features: tuple = ()) -> DynSynTrainConfig:
        """The task's registered configs; *features* rebuild the twin with them."""
        if features:
            from myosuite.envs.myo.backends.mjlab.tasks.registration import (
                rebuild_twin_cfg,
            )

            env_cfg = rebuild_twin_cfg(task_id, features)
        else:
            env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
        return DynSynTrainConfig(env=env_cfg, agent=agent_cfg)


def resolve_dynsyn(cfg: Any, resume_path: Path | None) -> str:
    """Apply DynSyn to ``cfg.env``: a resumed run's own map, or the ``--groups`` file.

    Args:
        cfg: A :class:`DynSynTrainConfig` (``env``, ``groups`` and the schedule fields).
        resume_path: Checkpoint that is continued, if any.

    Returns:
        What was applied, for the log.

    Raises:
        ValueError: Neither groups nor a DynSyn checkpoint is given, ``--groups``
            contradicts the resumed checkpoint, or that checkpoint is not a DynSyn run.
    """
    trained = trained_action_map(resume_path) if resume_path is not None else None
    groups = DynSynGroups.load(cfg.groups) if cfg.groups else None
    if trained is not None:
        trained_map = trained.action_map
        if not isinstance(trained_map, DynSyn):
            raise ValueError(f"{resume_path} was trained through {trained_map.kind!r}.")
        if groups is not None and groups.groups != trained_map.groups.groups:
            raise ValueError(f"--groups differs from the groups of {resume_path}.")
        apply_trained_action_map(cfg.env, trained)
        return (
            f"DynSyn from {resume_path}: correction bound "
            f"{trained_map.amplitude(trained.env_steps):.4g} after "
            f"{trained.env_steps:.4g} env steps"
        )
    if resume_path is not None:
        raise ValueError(
            f"{resume_path} was trained without DynSyn: its actions are per muscle, "
            "so DynSyn cannot continue it."
        )
    if groups is None:
        raise ValueError("Give --groups (from python -m myosuite_dynsyn.groups).")
    apply_action_map(
        cfg.env, DynSyn(groups, cfg.start_steps, cfg.ramp_steps, cfg.max_amplitude)
    )
    return f"DynSyn groups from {cfg.groups}"


def track_action_map_in_checkpoints(runner: Any) -> None:
    """Keep a runner's checkpoints and its env's action map together (in place).

    Every checkpoint the runner saves stores the map in ``infos[ACTION_MAP_FIELD]``,
    so a checkpoint without its run directory (a W&B download) can be resumed and
    evaluated as trained. Loading a checkpoint with an action map into an env
    without one raises instead of reading its actions as per-muscle actions.

    Args:
        runner: An RSL-RL runner whose ``save`` takes ``infos`` (mjlab's runners).
    """
    term = action_map_term(runner.env.unwrapped)
    save, load = runner.save, runner.load

    def save_with_map(path: str, infos: dict | None = None) -> None:
        if term is not None:
            infos = {**(infos or {}), ACTION_MAP_FIELD: term.trained().to_infos()}
        save(path, infos)

    def load_checked(path: str, *args: Any, **kwargs: Any) -> Any:
        if term is None and (trained := trained_action_map(path)) is not None:
            raise ValueError(
                f"{path} was trained through the {trained.action_map.kind!r} action map, "
                "but this env has per-muscle actions."
            )
        return load(path, *args, **kwargs)

    runner.save, runner.load = save_with_map, load_checked


def train(task_id: str, cfg: DynSynTrainConfig) -> Path:
    """Train ``task_id`` through DynSyn; returns the run directory.

    Args:
        task_id: Registered env id with an mjlab twin.
        cfg: Configs and DynSyn options.

    Returns:
        The run directory (``params/`` and ``model_*.pt``).
    """
    import myosuite.envs.myo.backends.mjlab  # noqa: F401  (registers the twins)

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    configure_torch_backends()
    cfg.env.seed = cfg.agent.seed

    experiment_dir = Path(cfg.log_root) / cfg.agent.experiment_name
    resume_path = Path(cfg.resume) if cfg.resume else None
    if resume_path is None and cfg.agent.resume:
        resume_path = resume_checkpoint(
            experiment_dir, task_id, cfg.agent.load_run, cfg.agent.load_checkpoint
        )
    print(f"[INFO] {resolve_dynsyn(cfg, resume_path)}.")

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.agent.run_name:
        run_name += f"_{cfg.agent.run_name}"
    log_dir = experiment_dir / run_name

    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg.env, device=device),
        clip_actions=cfg.agent.clip_actions,
    )
    agent_cfg = asdict(cfg.agent)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, agent_cfg, str(log_dir), device)
    track_action_map_in_checkpoints(runner)
    if resume_path is not None:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))
    dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    runner.learn(
        num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
    )
    env.close()
    return log_dir


def main() -> None:
    """Command line: ``python -m myosuite_dynsyn.train <task> [options]``."""
    import mjlab
    import mjlab.tasks  # noqa: F401

    import myosuite.envs.myo.backends.mjlab  # noqa: F401  (registers the twins)

    task, rest = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    features, rest = parse_feature_args(rest)
    cfg = tyro.cli(
        DynSynTrainConfig,
        args=rest,
        default=DynSynTrainConfig.from_task(task, features),
        prog=f"{sys.argv[0]} {task}",
        config=mjlab.TYRO_FLAGS,
    )
    train(task, cfg)


if __name__ == "__main__":
    main()
