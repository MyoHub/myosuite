# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Script to play RL agent with RSL-RL."""

import os
import sys
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

from myosuite.utils.onnx_checkpoint import (
    OnnxPolicy,
    get_wandb_onnx_checkpoint_path,
    is_onnx_checkpoint_name,
    normalize_onnx_checkpoint_name,
    onnx_checkpoint_sort_key,
)


def _parse_wandb_dt(value: str | datetime) -> datetime:
    """Parse a W&B datetime string (or pass through a datetime object)."""
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value


@dataclass(frozen=True)
class PlayConfig:
    agent: Literal["zero", "random", "trained"] = "trained"
    registry_name: str | None = None
    wandb_run_path: str | None = None
    wandb_checkpoint_name: str | None = None
    """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.onnx')."""
    checkpoint_file: str | None = None
    motion_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    camera: int | str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = False
    """Disable all termination conditions (useful for viewing motions with dummy agents)."""

    # Internal flag used by demo script.
    _demo_mode: tyro.conf.Suppress[bool] = False


def _resolve_local_onnx_checkpoint(checkpoint_file: str) -> Path:
    checkpoint_path = Path(checkpoint_file).expanduser()
    if checkpoint_path.suffix == ".pt":
        onnx_candidate = checkpoint_path.with_suffix(".onnx")
        if onnx_candidate.exists():
            return onnx_candidate
        raise FileNotFoundError(
            "Expected an ONNX checkpoint bundle for mjlab play, but only received "
            f"{checkpoint_path}. Looked for {onnx_candidate}."
        )
    return checkpoint_path


def run_play(task_id: str, cfg: PlayConfig) -> None:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

    # Disable terminations if requested (useful for viewing motions).
    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO]: Terminations disabled")

    # Check if this is a tracking task by checking for motion command.
    is_tracking_task = "motion" in env_cfg.commands and isinstance(
        env_cfg.commands["motion"], MotionCommandCfg
    )

    if is_tracking_task and cfg._demo_mode:
        # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        motion_cmd.sampling_mode = "uniform"

    if is_tracking_task:
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)

        # Check for local motion file first (works for both dummy and trained modes).
        if cfg.motion_file is not None and Path(cfg.motion_file).exists():
            print(f"[INFO]: Using local motion file: {cfg.motion_file}")
            motion_cmd.motion_file = cfg.motion_file
        elif DUMMY_MODE:
            if not cfg.registry_name:
                raise ValueError(
                    "Tracking tasks require either:\n"
                    "  --motion-file /path/to/motion.npz (local file)\n"
                    "  --registry-name your-org/motions/motion-name (download from WandB)"
                )
            # Check if the registry name includes alias, if not, append ":latest".
            registry_name = cfg.registry_name
            if ":" not in registry_name:
                registry_name = registry_name + ":latest"
            import wandb

            api = wandb.Api()
            artifact = api.artifact(registry_name)
            motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
        else:
            if cfg.motion_file is not None:
                print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
                motion_cmd.motion_file = cfg.motion_file
            else:
                import wandb

                api = wandb.Api()
                if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
                    raise ValueError(
                        "Tracking tasks require `motion_file` when using `checkpoint_file`, "
                        "or provide `wandb_run_path` so the motion artifact can be resolved."
                    )
                if cfg.wandb_run_path is not None:
                    wandb_run = api.run(str(cfg.wandb_run_path))
                    art = next(
                        (a for a in wandb_run.used_artifacts() if a.type == "motions"),
                        None,
                    )
                    if art is None:
                        raise RuntimeError("No motion artifact found in the run.")
                    motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

    log_dir: Path | None = None
    resume_path: Path | None = None
    if TRAINED_MODE:
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        if cfg.checkpoint_file is not None:
            resume_path = _resolve_local_onnx_checkpoint(cfg.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
            print(f"[INFO]: Loading checkpoint: {resume_path.name}")
        else:
            if cfg.wandb_run_path is None:
                raise ValueError(
                    "`wandb_run_path` is required when `checkpoint_file` is not provided."
                )
            checkpoint_name = (
                normalize_onnx_checkpoint_name(cfg.wandb_checkpoint_name)
                if cfg.wandb_checkpoint_name is not None
                else None
            )
            resume_path, was_cached = get_wandb_onnx_checkpoint_path(
                log_root_path, Path(cfg.wandb_run_path), checkpoint_name
            )
            # Extract run_id and checkpoint name from path for display.
            run_id = resume_path.parent.name
            checkpoint_name = resume_path.name
            cached_str = "cached" if was_cached else "downloaded"
            print(
                f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
            )
        log_dir = resume_path.parent

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    if cfg.video and DUMMY_MODE:
        print(
            "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
        )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if TRAINED_MODE and cfg.video:
        print("[INFO] Recording videos during play")
        assert log_dir is not None  # log_dir is set in TRAINED_MODE block
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos" / "play",
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if DUMMY_MODE:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
        if cfg.agent == "zero":

            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return torch.zeros(action_shape, device=env.unwrapped.device)

            policy = PolicyZero()
        else:

            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

            policy = PolicyRandom()
    else:
        policy = OnnxPolicy(resume_path, device=device)

    # Build checkpoint manager for hot-swapping checkpoints in the viewer.
    ckpt_manager: CheckpointManager | None = None
    if TRAINED_MODE and resume_path is not None:

        def _reload_policy(path: str) -> OnnxPolicy:
            return OnnxPolicy(Path(path), device=device)

        if cfg.wandb_run_path is None:
            ckpt_dir = resume_path.parent

            def fetch_available_local() -> list[tuple[str, str]]:
                now = _time.time()
                entries: list[tuple[str, str]] = []
                for f in ckpt_dir.glob("*.onnx"):
                    if not is_onnx_checkpoint_name(f.name):
                        continue
                    ago = format_time_ago(int(now - f.stat().st_mtime))
                    entries.append((f.name, ago))
                entries.sort(key=lambda x: onnx_checkpoint_sort_key(x[0]))
                return entries

            ckpt_manager = CheckpointManager(
                current_name=resume_path.name,
                fetch_available=fetch_available_local,
                load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
            )
        else:
            import wandb

            api = wandb.Api()
            run_path = str(cfg.wandb_run_path)
            wandb_run = api.run(run_path)
            _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

            def fetch_available_wandb() -> list[tuple[str, str]]:
                wandb_run.load()
                now = datetime.now(tz=timezone.utc)
                entries: list[tuple[str, str]] = []
                for f in wandb_run.files():
                    if not is_onnx_checkpoint_name(f.name):
                        continue
                    ago = format_time_ago(
                        int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
                    )
                    entries.append((f.name, ago))
                entries.sort(key=lambda x: onnx_checkpoint_sort_key(x[0]))
                return entries

            ckpt_manager = CheckpointManager(
                current_name=resume_path.name,
                fetch_available=fetch_available_wandb,
                load_checkpoint=lambda name: _reload_policy(
                    str(
                        get_wandb_onnx_checkpoint_path(_log_root, Path(run_path), name)[
                            0
                        ]
                    )
                ),
                run_name=_parse_wandb_dt(wandb_run.created_at).strftime(
                    "%Y-%m-%d_%H-%M-%S"
                ),
                run_url=wandb_run.url,
                run_status=wandb_run.state,
            )

    # Handle "auto" viewer selection.
    if cfg.viewer == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        resolved_viewer = "native" if has_display else "viser"
        del has_display
    else:
        resolved_viewer = cfg.viewer

    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager).run()
    else:
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()


def main(argv: list[str] | None = None) -> int:
    maybe_print_top_level_help("play")

    # Parse first argument to choose the task.
    # Import tasks to populate the registry.
    import mjlab.tasks  # noqa: F401

    cli_args = sys.argv[1:] if argv is None else list(argv)
    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        args=cli_args,
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    del remaining_args

    run_play(chosen_task, args)
    return 0


# if __name__ == "__main__":
#   main()
