# Copyright (c) MyoSuite Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PPO (mjlab / RSL-RL) on a registered MyoSuite task through LambdaHold stretch-reflex control."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from myosuite.utils.action_map import LambdaHold, trained_action_map
from myosuite.utils.checkpoint_manifest import ACTION_MAP_FIELD
from myosuite.utils.checkpoint_utils import resume_checkpoint
from myosuite.utils.feature_cli import parse_feature_args


@dataclass(frozen=True)
class LambdaHoldOptions:
    """Hyperparameters of the :class:`~myosuite.utils.action_map.LambdaHold` map."""

    kp: float = 2.5
    damping_s: float = 0.04
    margin_fraction: float = 0.35
    hold_steps: int = 2


def build_lambda_hold_for_mjlab_cfg(
    env_cfg: Any, opts: LambdaHoldOptions
) -> LambdaHold:
    """Construct a :class:`LambdaHold` map matching an mjlab env config's muscle term."""
    from myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map import (
        muscle_action_terms,
    )

    terms = muscle_action_terms(env_cfg)
    if len(terms) != 1:
        raise ValueError(f"Expected one muscle action term, found {sorted(terms)}")
    ((_name, base),) = terms.items()
    entity_cfg = env_cfg.scene.entities[base.entity_name]
    spec = entity_cfg.build()
    model = spec.compile() if hasattr(spec, "compile") else spec
    sigmoid = bool(getattr(base, "muscle_sigmoid", True))
    return LambdaHold.from_model(
        model,
        kp=opts.kp,
        damping_s=opts.damping_s,
        margin_fraction=opts.margin_fraction,
        hold_steps=opts.hold_steps,
        muscle_sigmoid=sigmoid,
    )


def main(argv: list[str] | None = None) -> None:
    """Train a registered MyoSuite mjlab task with PPO through LambdaHold."""
    import tyro
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from mjlab.tasks.registry import (
        list_tasks,
        load_env_cfg,
        load_rl_cfg,
        load_runner_cls,
    )
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    from myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map import (
        action_map_term,
        apply_action_map,
        apply_trained_action_map,
    )
    import myosuite.envs.myo.backends.mjlab  # noqa: F401

    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args or raw_args[0].startswith("-"):
        raise SystemExit(
            f"Give a registered task id first; known: {len(list_tasks())} tasks"
        )
    task_id, rest = raw_args[0], raw_args[1:]
    rest, features = parse_feature_args(rest)

    @dataclass(frozen=True)
    class TrainCfg:
        env: ManagerBasedRlEnvCfg
        agent: RslRlOnPolicyRunnerCfg
        kp: float = 2.5
        damping_s: float = 0.04
        margin_fraction: float = 0.35
        hold_steps: int = 2
        resume: str | None = None
        device: str | None = None
        log_root: str = "logs/rsl_rl"

    if features:
        from myosuite.envs.myo.backends.mjlab.tasks.registration import rebuild_twin_cfg

        env_cfg = rebuild_twin_cfg(task_id, features)
    else:
        env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    cfg = tyro.cli(TrainCfg, args=rest, default=TrainCfg(env=env_cfg, agent=agent_cfg))

    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = (
        Path(cfg.log_root)
        / cfg.agent.experiment_name
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    resume_path = (
        Path(cfg.resume)
        if cfg.resume
        else resume_checkpoint(
            Path(cfg.log_root) / cfg.agent.experiment_name, cfg.agent
        )
    )

    if resume_path is not None:
        trained = trained_action_map(resume_path)
        if trained is None or not isinstance(trained.action_map, LambdaHold):
            raise ValueError(f"{resume_path} is not a LambdaHold checkpoint")
        apply_trained_action_map(cfg.env, trained)
    else:
        amap = build_lambda_hold_for_mjlab_cfg(
            cfg.env,
            LambdaHoldOptions(
                cfg.kp, cfg.damping_s, cfg.margin_fraction, cfg.hold_steps
            ),
        )
        apply_action_map(cfg.env, amap)

    env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
    vec = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(vec, asdict(cfg.agent), str(log_dir), device=device)
    term = action_map_term(env)
    orig_save = runner.save

    def _save_with_map(path: str, infos: dict[str, Any] | None = None) -> None:
        extra = dict(infos or {})
        if term is not None:
            extra[ACTION_MAP_FIELD] = term.trained().to_infos()
        orig_save(path, infos=extra)

    runner.save = _save_with_map  # type: ignore[method-assign]
    dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))
    if resume_path is not None:
        runner.load(str(resume_path))
    runner.learn(
        num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
    )
    env.close()


if __name__ == "__main__":  # pragma: no cover
    main()
