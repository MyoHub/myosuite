"""LOCAL-ONLY Modal launcher: cross-backend trajectory verification (CPU-loadable
motion clip -> mjlab GPU).

Confirms that a trajectory/motion clip loadable on the local CPU dev machine
(via ``myosuite.core.trajectory_io.load_motion_clip``, the same loader code
used by CPU-side BC data collection) also registers and runs correctly under
mjlab's GPU backend -- i.e. that CPU-collected trajectory data is genuinely
cross-backend compatible, not just structurally similar. Reusable for any
future clip/task, not tied to one specific dataset file.

Much lighter than a full training smoke test (scripts/modal_verify_11c_mjlab_gpu.py):
no PPO loop, just clip-load -> task registration -> env construction -> a
short zero-action rollout. Typical runtime: a few minutes (mostly mujoco-warp
JIT compilation on first GPU use), not tied to notebook execution.

Usage (always ``--detach`` for anything that might run long; this one
usually finishes within the foreground timeout)::

    modal run scripts/modal_verify_trajectory_gpu.py
    modal run scripts/modal_verify_trajectory_gpu.py \\
        --clip-repo amathislab/musclemimic-retargeted \\
        --clip-file MyoFullBody/gmr/KIT/4/WalkInCounterClockwiseCircle08_poses.npz \\
        --num-envs 8 --num-steps 10
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP = modal.App("myo-verify-trajectory-gpu")

_IGNORE = [
    ".git",
    ".git/**",
    ".venv",
    ".venv/**",
    ".venv_install_check/**",
    "**/__pycache__",
    "**/*.pyc",
    ".claude/**",
    "docs/**",
    "renders/**",
    "runs/**",
    "tasks/**",
    "**/*.mp4",
    "**/*.gif",
    "logs/**",
    "wandb",
    "wandb/**",
    ".pytest_cache/**",
    "**/.myosuite_resolved_*.xml",
    "myosuite/../worktree-*",
    "tutorials/**/iterations/**",
    "**/*.zip",
    "tutorials/*.ipynb",
]

IMAGE = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.10")
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
        "libegl1",
        "libgles2",
        "libosmesa6",
        "build-essential",
    )
    .add_local_dir(str(REPO), "/root/myosuite4", ignore=_IGNORE, copy=True)
    .run_commands(
        "cd /root/myosuite4 && pip install -e '.[mjlab,musclemimic]'",
        # See modal_train_directional_gpu.py -- tensordict/torch._dynamo
        # probing triton at import time segfaults; not needed here.
        "pip uninstall -y triton || true",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "MUJOCO_GL": "osmesa"})
)

SECRETS = [modal.Secret.from_name("hf-token")]
TEN_MINUTES_S = 60 * 10


@APP.function(image=IMAGE, gpu="L4", timeout=TEN_MINUTES_S, secrets=SECRETS)
def verify_trajectory_on_gpu(
    clip_repo: str = "amathislab/musclemimic-retargeted",
    clip_file: str = "MyoFullBody/gmr/KIT/4/WalkInClockwiseCircle01_poses.npz",
    expected_nq: int = 89,
    expected_nv: int = 88,
    num_envs: int = 4,
    num_steps: int = 5,
) -> str:
    """Load a clip (CPU-loader code path) and roll it out under mjlab GPU.

    Args:
        clip_repo: HuggingFace dataset repo id containing the clip.
        clip_file: Path within that repo to the retargeted ``.npz`` clip.
        expected_nq: Expected ``qpos`` width for the target model (full-body
            default: 89).
        expected_nv: Expected ``qvel`` width (full-body default: 88).
        num_envs: Parallel GPU environments to construct.
        num_steps: Zero-action physics steps to run after reset.

    Returns:
        A short human-readable summary string (also printed).
    """
    import sys

    sys.path.insert(0, "/root/myosuite4")
    import os

    os.chdir("/root/myosuite4")

    from huggingface_hub import hf_hub_download
    from myosuite.core.trajectory_io import load_motion_clip

    # Same loader code path used by local CPU BC data collection.
    clip_path = Path(
        hf_hub_download(repo_id=clip_repo, filename=clip_file, repo_type="dataset")
    )
    clip = load_motion_clip(clip_path, expected_nq=expected_nq, expected_nv=expected_nv)
    print(
        f"Loaded clip on GPU container: T={clip.qpos.shape[0]} frames, nq={clip.qpos.shape[1]}"
    )

    from mjlab.tasks.registry import register_mjlab_task
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        default_mimic_clip_on_policy_runner_cfg,
        register_mimic_mjlab_tasks_with_clip,
    )

    register_mimic_mjlab_tasks_with_clip(
        register_mjlab_task=register_mjlab_task,
        rl_cfg_fn=default_mimic_clip_on_policy_runner_cfg,
        clip=clip,
        use_lookahead=True,
    )
    print("Registered myoMimicFullbody-v0 with locally-loadable trajectory on GPU.")

    from mjlab.tasks.registry import load_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    import torch

    cfg = load_env_cfg("myoMimicFullbody-v0")
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg, device="cuda")
    obs, _ = env.reset()
    action = torch.zeros(env.action_manager.total_action_dim, device=env.device).repeat(
        num_envs, 1
    )
    for _ in range(num_steps):
        obs, rew, term, trunc, info = env.step(action)
    result = (
        f"OK: trajectory loaded locally-compatible code path, ran on GPU. "
        f"obs shape={obs['policy'].shape if isinstance(obs, dict) else obs.shape}, "
        f"reward sample={float(rew[0]):.4f}"
    )
    print(result)
    return result


@APP.local_entrypoint()
def main(
    clip_repo: str = "amathislab/musclemimic-retargeted",
    clip_file: str = "MyoFullBody/gmr/KIT/4/WalkInClockwiseCircle01_poses.npz",
    expected_nq: int = 89,
    expected_nv: int = 88,
    num_envs: int = 4,
    num_steps: int = 5,
) -> None:
    print(
        verify_trajectory_on_gpu.remote(
            clip_repo=clip_repo,
            clip_file=clip_file,
            expected_nq=expected_nq,
            expected_nv=expected_nv,
            num_envs=num_envs,
            num_steps=num_steps,
        )
    )
