# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Training: the task through ``make_env`` with the recipe's options, and the SuperMimic runner.

The run layout is mjlab's ``train`` command's: ``<log_root>/<experiment_name>/<date>_<run name>/`` with
``params/{env,agent}.yaml`` and ``model_<iteration>.pt`` checkpoints, and ``agent.resume`` / ``load_run`` /
``load_checkpoint`` continue a run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from supermimic.config import SuperMimicRunnerCfg, fast_runner_cfg
from supermimic.env import TaskEnvCfg


@dataclass
class TrainArgs:
    """A training run."""

    env: TaskEnvCfg = field(default_factory=TaskEnvCfg)
    agent: SuperMimicRunnerCfg = field(default_factory=fast_runner_cfg)
    log_root: str = "logs/rsl_rl"
    device: str | None = None
    """Torch device (default: ``cuda:0`` if available)."""


def train(args: TrainArgs) -> Path:
    """Train a SuperMimic policy.

    Args:
        args: The run.

    Returns:
        The run directory.
    """
    import torch  # noqa: PLC0415
    from mjlab.utils.os import dump_yaml, get_checkpoint_path  # noqa: PLC0415
    from mjlab.utils.torch import configure_torch_backends  # noqa: PLC0415

    from supermimic.env import make_task_env  # noqa: PLC0415
    from supermimic.runner import SuperMimicRunner  # noqa: PLC0415

    if not args.env.motion_file:
        raise ValueError(
            "--env.motion-file is empty: a clip, a directory or a .txt clip list"
        )
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    configure_torch_backends()
    root = (Path(args.log_root) / args.agent.experiment_name).resolve()
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (
        f"_{args.agent.run_name}" if args.agent.run_name else ""
    )
    log_dir = root / name
    resume = (
        get_checkpoint_path(root, args.agent.load_run, args.agent.load_checkpoint)
        if args.agent.resume
        else None
    )
    env = make_task_env(args.env, device, seed=args.agent.seed)
    agent = asdict(args.agent)
    dump_yaml(
        log_dir / "params" / "env.yaml",
        {"task": asdict(args.env), "env": asdict(env.unwrapped.cfg)},
    )
    dump_yaml(log_dir / "params" / "agent.yaml", agent)
    runner = SuperMimicRunner(env, agent, str(log_dir), device)
    if resume is not None:
        print(f"[INFO] resuming from {resume}")
        runner.load(str(resume))
    runner.learn(
        num_learning_iterations=args.agent.max_iterations, init_at_random_ep_len=True
    )
    env.close()
    return log_dir
