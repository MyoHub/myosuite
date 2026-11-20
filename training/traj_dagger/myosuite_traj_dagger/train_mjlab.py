# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""GPU `mjlab` multi-teacher trajectory-swap DAgger distillation entrypoint.

Distills one or more RSL-RL, PyTorch, TorchScript, or ONNX teacher checkpoints
into a `StudentActorCritic` on a registered MyoSuite `ManagerBasedRlEnv` twin:

    traj-dagger-mjlab myoElbowPose1D6MRandom-v0 \\
        --teachers logs/rsl_rl/run_a/model_500.pt logs/rsl_rl/run_b/model_500.pt \\
        --env.scene.num-envs 1024 --total-steps 1000000

Teacher entries accept `<path>`, `<weight>*<path>`, or `<weight>*<path>::<json>`
so repeating or weighting a `(teacher, reset_options)` pair scales chunk sampling
frequency without reloading duplicate teacher checkpoints.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from myosuite_traj_dagger.buffer import DaggerRolloutState, PersistentDaggerBuffer
from myosuite_traj_dagger.dagger import (
    ChunkMetrics,
    DaggerConfig,
    RunningMeanStd,
    StudentActorCritic,
    SwapPair,
    TeacherRegistry,
    TeacherSpec,
    compute_beta,
    repeat_pair,
    update_student_on_buffer,
)


def extract_actor_obs(obs: Any) -> torch.Tensor:
    """Extract the policy observation tensor from a `ManagerBasedRlEnv` obs output.

    MyoSuite mjlab twins expose an `"actor"` group, other mjlab tasks expose
    `"policy"`, and standalone tensor wrappers may return a `torch.Tensor` directly.
    """
    if isinstance(obs, torch.Tensor):
        return obs
    if isinstance(obs, Mapping):
        if "actor" in obs:
            return obs["actor"]
        if "policy" in obs:
            return obs["policy"]
    raise TypeError(
        f"Cannot extract actor observation tensor from {type(obs).__name__}."
    )


def parse_teacher_cli_entry(
    entry: str,
    *,
    act_dim: int,
    device: str = "cpu",
) -> list[SwapPair]:
    """Parse `<path>`, `<weight>*<path>`, or `<weight>*<path>::<json_options>`.

    Args:
        entry: CLI teacher specification string.
        act_dim: Environment action dimension.
        device: Inference device string.

    Returns:
        List of `SwapPair` instances (`len == weight`, default `1`).
    """
    raw = entry.strip()
    if not raw:
        raise ValueError("Teacher CLI entry cannot be empty.")

    options: dict[str, Any] | None = None
    if "::" in raw:
        raw, json_part = raw.split("::", 1)
        parsed = json.loads(json_part)
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Teacher reset_options JSON must be an object, got {type(parsed).__name__}."
            )
        options = parsed

    weight = 1
    if "*" in raw:
        prefix, rest = raw.split("*", 1)
        if prefix.strip().isdigit():
            weight = int(prefix.strip())
            raw = rest

    spec = TeacherSpec.from_path(raw.strip(), act_dim=act_dim, device=device)
    return repeat_pair(SwapPair(teacher=spec, reset_options=options), weight)


def parse_teacher_cli_specs(
    entries: Sequence[str],
    *,
    act_dim: int,
    device: str = "cpu",
) -> list[SwapPair]:
    """Expand a sequence of CLI teacher specification strings into `SwapPair`s."""
    pairs: list[SwapPair] = []
    for item in entries:
        pairs.extend(parse_teacher_cli_entry(item, act_dim=act_dim, device=device))
    if not pairs:
        raise ValueError(
            "Provide at least one teacher checkpoint via --teachers <path> [...]."
        )
    return pairs


def distill_on_vec_env(
    env: Any,
    student: StudentActorCritic,
    pairs: Sequence[SwapPair],
    cfg: DaggerConfig,
    *,
    total_steps: int,
    registry: TeacherRegistry | None = None,
    clip_actions: float | None = 1.0,
) -> list[ChunkMetrics]:
    """Run vectorized GPU/Torch trajectory-swap DAgger on a `ManagerBasedRlEnv`.

    Args:
        env: `ManagerBasedRlEnv` (or compatible Torch vectorized env with `num_envs`,
            `reset()`, and 5-tuple `step(actions)`).
        student: `StudentActorCritic` to train.
        pairs: Weighted `(teacher, reset_options)` chunk schedule.
        cfg: `DaggerConfig` hyperparameters.
        total_steps: Total interactive environment steps across all parallel envs.
        registry: Optional `TeacherRegistry` cache.
        clip_actions: Optional symmetric action clamp before `env.step`.

    Returns:
        Per-chunk `ChunkMetrics` log.
    """
    if not pairs:
        raise ValueError("pairs schedule must be non-empty.")
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}.")

    dev = torch.device(cfg.device)
    student = student.to(dev)
    reg = registry if registry is not None else TeacherRegistry()
    num_envs = int(env.num_envs)

    rng = np.random.default_rng(cfg.seed)
    torch_rng = torch.Generator(device=dev)
    torch_rng.manual_seed(int(cfg.seed))

    optimizer = torch.optim.Adam(
        student.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    buffer = PersistentDaggerBuffer(
        obs_dim=student.obs_dim,
        act_dim=student.act_dim,
        ring_buffer_size=cfg.ring_buffer_size,
        long_term_buffer_size=cfg.long_term_buffer_size,
        min_include_prob=cfg.min_include_prob,
        seed=cfg.seed + 17,
        device=dev,
    )
    rollout_state = DaggerRolloutState.create(
        num_envs=num_envs,
        beta=cfg.beta_start,
        h_min=cfg.sticky_h_min,
        h_max=cfg.sticky_h_max,
        backend="torch",
        torch_rng=torch_rng,
        device=dev,
    )

    anneal_steps = (
        cfg.beta_anneal_steps if cfg.beta_anneal_steps is not None else total_steps
    )
    env_steps = 0
    chunk_idx = 0
    metrics_log: list[ChunkMetrics] = []

    while env_steps < total_steps:
        p_idx = (
            chunk_idx % len(pairs)
            if cfg.pair_sampling == "round_robin"
            else int(rng.integers(0, len(pairs)))
        )
        pair = pairs[p_idx]
        teacher = reg.get_or_load(pair.teacher)

        beta = compute_beta(
            env_steps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            anneal_steps=anneal_steps,
            schedule=cfg.beta_schedule,
        )
        raw_obs = _reset_vec_env(env, pair.reset_options)
        obs = extract_actor_obs(raw_obs).to(device=dev, dtype=torch.float32)
        rollout_state.reset_all(beta)

        chunk_budget = min(cfg.steps_per_chunk, total_steps - env_steps)
        horizon = max(1, math.ceil(chunk_budget / num_envs))
        collected = 0
        teacher_fracs: list[float] = []
        student.eval()

        with torch.no_grad():
            for _ in range(horizon):
                if collected >= chunk_budget:
                    break
                beta = compute_beta(
                    env_steps,
                    beta_start=cfg.beta_start,
                    beta_end=cfg.beta_end,
                    anneal_steps=anneal_steps,
                    schedule=cfg.beta_schedule,
                )
                t_act_raw, t_val_raw = teacher.predict_with_value(obs)
                t_act = torch.as_tensor(t_act_raw, dtype=torch.float32, device=dev)
                t_val = (
                    torch.as_tensor(t_val_raw, dtype=torch.float32, device=dev).reshape(
                        -1
                    )
                    if t_val_raw is not None
                    else None
                )
                s_act = student.act(
                    obs,
                    deterministic=not cfg.student_rollout_stochastic,
                    generator=torch_rng,
                )
                assert isinstance(s_act, torch.Tensor)
                exec_act = rollout_state.select_action(t_act, s_act)
                assert isinstance(exec_act, torch.Tensor)
                if clip_actions is not None:
                    exec_act = torch.clamp(exec_act, -clip_actions, clip_actions)

                assert isinstance(rollout_state.using_teacher, torch.Tensor)
                teacher_fracs.append(
                    float(rollout_state.using_teacher.float().mean().item())
                )

                take = min(num_envs, chunk_budget - collected)
                obs_slice = obs[:take]
                act_slice = t_act[:take]
                val_slice = t_val[:take] if t_val is not None else None
                if cfg.update_obs_rms and isinstance(student.obs_rms, RunningMeanStd):
                    student.obs_rms.update(obs_slice)
                buffer.add(obs_slice, act_slice, val_slice, source_id=p_idx)

                next_raw, _, terminated, truncated, _ = env.step(exec_act)
                obs = extract_actor_obs(next_raw).to(device=dev, dtype=torch.float32)
                dones = (
                    torch.as_tensor(terminated, device=dev).bool()
                    | torch.as_tensor(truncated, device=dev).bool()
                )
                rollout_state.advance(dones, beta)

                env_steps += take
                collected += take

        nll, mse, v_loss = update_student_on_buffer(
            student,
            optimizer,
            buffer,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs_per_chunk,
            value_loss_coef=cfg.value_loss_coef,
            max_grad_norm=cfg.max_grad_norm,
            max_batches_per_epoch=cfg.max_batches_per_epoch,
        )
        metrics_log.append(
            ChunkMetrics(
                chunk_idx=chunk_idx,
                env_steps=env_steps,
                pair_idx=p_idx,
                teacher_key=pair.teacher.key,
                beta=beta,
                teacher_step_fraction=float(np.mean(teacher_fracs))
                if teacher_fracs
                else 0.0,
                policy_nll=nll,
                policy_mse=mse,
                value_loss=v_loss,
                buffer_size=buffer.size,
                ring_size=buffer.ring_size,
                reservoir_size=buffer.reservoir_size,
            )
        )
        chunk_idx += 1

    return metrics_log


def _reset_vec_env(env: Any, reset_options: dict[str, Any] | None) -> Any:
    if reset_options is not None:
        try:
            out = env.reset(options=reset_options)
        except TypeError:
            out = env.reset()
    else:
        out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def save_student_artifacts(
    student: StudentActorCritic,
    log_dir: Path,
    *,
    metrics: Sequence[ChunkMetrics],
    export_torchscript_file: bool = True,
    export_onnx_file: bool = True,
) -> dict[str, Path]:
    """Save student PyTorch checkpoint and optional TorchScript / ONNX exports."""
    log_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    ckpt_path = log_dir / "student_final.pt"
    payload = {
        "student_config": {
            "obs_dim": student.obs_dim,
            "act_dim": student.act_dim,
            "actor_hidden_dims": list(student.actor_hidden_dims),
            "critic_hidden_dims": list(student.critic_hidden_dims),
            "activation": student.activation,
            "log_std_bounds": (student.log_std_min, student.log_std_max),
            "normalize_obs": student.normalize_obs,
            "use_value_head": student.use_value_head,
            "action_limit": student.action_limit,
        },
        "state_dict": student.cpu().state_dict(),
        "metrics": [asdict(m) for m in metrics],
    }
    torch.save(payload, ckpt_path)
    written["checkpoint"] = ckpt_path

    if export_torchscript_file:
        ts_path = log_dir / "student_final.ts"
        student.export_torchscript(ts_path)
        written["torchscript"] = ts_path

    if export_onnx_file:
        try:
            onnx_path = log_dir / "student_final.onnx"
            student.export_onnx(onnx_path)
            written["onnx"] = onnx_path
        except ModuleNotFoundError:
            pass

    return written


@dataclass(frozen=True)
class MjlabDistillCliConfig:
    """CLI options for `traj-dagger-mjlab <env_id> ...`."""

    env: Any
    teachers: tuple[str, ...] = ()
    total_steps: int = 1_000_000
    steps_per_chunk: int = 16_384
    beta_start: float = 1.0
    beta_end: float = 0.05
    sticky_h_min: int = 5
    sticky_h_max: int = 15
    ring_buffer_size: int = 262_144
    long_term_buffer_size: int = 65_536
    min_include_prob: float = 0.05
    actor_hidden_dims: tuple[int, ...] = (256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "elu"
    batch_size: int = 1024
    epochs_per_chunk: int = 4
    learning_rate: float = 3e-4
    value_loss_coef: float = 0.5
    clip_actions: float = 1.0
    seed: int = 0
    device: str | None = None
    log_root: Path = Path("logs/traj_dagger")
    run_name: str | None = None
    export_torchscript_file: bool = True
    export_onnx_file: bool = True


def _require_mjlab() -> tuple[Any, Any]:
    try:
        import mjlab
        import tyro
    except ImportError as exc:
        raise ImportError(
            "GPU mjlab distillation requires `myosuite[mjlab]` and `tyro`. "
            "Install with: pip install -e 'training/traj_dagger[mjlab]'"
        ) from exc
    return mjlab, tyro


def run_mjlab_distill(task_id: str, cfg: MjlabDistillCliConfig) -> dict[str, Path]:
    """Instantiate `ManagerBasedRlEnv` for `task_id` and run trajectory-swap DAgger."""
    _require_mjlab()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    import myosuite.envs.myo.backends.mjlab  # noqa: F401

    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.env.seed = cfg.seed

    env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
    try:
        init_obs, _ = env.reset()
        obs_dim = int(extract_actor_obs(init_obs).shape[-1])
        act_dim = int(env.action_manager.total_action_dim)

        pairs = parse_teacher_cli_specs(cfg.teachers, act_dim=act_dim, device=device)
        student = StudentActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            actor_hidden_dims=cfg.actor_hidden_dims,
            critic_hidden_dims=cfg.critic_hidden_dims,
            activation=cfg.activation,
            action_limit=cfg.clip_actions,
        ).to(device)
        dagger_cfg = DaggerConfig(
            steps_per_chunk=cfg.steps_per_chunk,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            sticky_h_min=cfg.sticky_h_min,
            sticky_h_max=cfg.sticky_h_max,
            ring_buffer_size=cfg.ring_buffer_size,
            long_term_buffer_size=cfg.long_term_buffer_size,
            min_include_prob=cfg.min_include_prob,
            batch_size=cfg.batch_size,
            epochs_per_chunk=cfg.epochs_per_chunk,
            learning_rate=cfg.learning_rate,
            value_loss_coef=cfg.value_loss_coef,
            seed=cfg.seed,
            device=device,
        )
        metrics = distill_on_vec_env(
            env,
            student,
            pairs,
            dagger_cfg,
            total_steps=cfg.total_steps,
            clip_actions=cfg.clip_actions,
        )
    finally:
        env.close()

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"{stamp}_{cfg.run_name}" if cfg.run_name else stamp
    log_dir = Path(cfg.log_root) / task_id / dir_name
    return save_student_artifacts(
        student,
        log_dir,
        metrics=metrics,
        export_torchscript_file=cfg.export_torchscript_file,
        export_onnx_file=cfg.export_onnx_file,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for `traj-dagger-mjlab`."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    mjlab, tyro = _require_mjlab()
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.tasks.registry import list_tasks, load_env_cfg

    import myosuite.envs.myo.backends.mjlab  # noqa: F401
    from myosuite.utils.feature_cli import parse_feature_args

    raw_args = list(sys.argv[1:] if argv is None else argv)
    all_tasks = list_tasks()
    chosen_task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        args=raw_args,
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    features, remaining = parse_feature_args(remaining)
    if features:
        from myosuite.envs.myo.backends.mjlab.tasks.registration import (
            rebuild_twin_cfg,
        )

        env_cfg: ManagerBasedRlEnvCfg = rebuild_twin_cfg(chosen_task, features)
    else:
        env_cfg = load_env_cfg(chosen_task)

    @dataclass(frozen=True)
    class _TypedCliConfig(MjlabDistillCliConfig):
        env: ManagerBasedRlEnvCfg  # type: ignore[assignment]

    default_cfg = _TypedCliConfig(env=env_cfg)
    parsed_cfg = tyro.cli(
        _TypedCliConfig,
        args=remaining,
        default=default_cfg,
        prog=f"{sys.argv[0]} {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    written = run_mjlab_distill(chosen_task, parsed_cfg)
    for label, path in written.items():
        print(f"[traj-dagger-mjlab] Wrote {label}: {path}")


if __name__ == "__main__":
    main()
