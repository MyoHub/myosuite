"""LOCAL-ONLY Modal launcher: BC directional-locomotion data collection + training.

Runs on Linux (Modal) because loading the ``amathislab/mm-10m-2`` teacher
checkpoint requires ``orbax-checkpoint``, which this repo's ``pyproject.toml``
excludes on macOS/darwin (see the ``sys_platform != 'darwin'`` marker) --- so
data collection cannot run on a local Mac dev machine. Training itself is
plain CPU torch and could run anywhere, but is bundled into the same
container run to avoid a second multi-GB checkpoint download.

Mirrors the image/install pattern used by
``scripts/modal_train_directional_gpu.py`` (ubuntu:22.04 base, not
nvidia/cuda -- avoids the torch-bundled-CUDA-runtime SIGSEGV; strips triton
post-install to avoid the tensordict/torch._dynamo heap-corruption segfault),
but this job needs no GPU: CPU MuJoCo rollouts + a small MLP BC training loop.

Usage (data collection + training can run several minutes to an hour
depending on ``--samples-per-clip``; always ``--detach`` for anything that
might run long)::

    modal run --detach scripts/modal_bc_directional_pipeline.py \\
        --samples-per-clip 30000 --epochs 200
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP = modal.App("myo-bc-directional")
VOL = modal.Volume.from_name("myosuite-bc-directional-out", create_if_missing=True)

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
        # orbax-checkpoint (needed to load the teacher's Orbax artifacts)
        # lives under the `mjx` extra, but that extra also pulls in
        # brax/flax/jax with a jax<0.5.3 pin that conflicts with brax's own
        # floor -- and none of that jax/brax/flax stack is actually needed
        # here (only orbax-checkpoint itself, for its NumPy-independent
        # Orbax artifact loader). Install orbax directly instead of the
        # whole `mjx` extra.
        "cd /root/myosuite4 && pip install -e '.[musclemimic]' 'orbax-checkpoint>=0.11.33'",
        # CPU-only torch wheel (BC training is a small MLP -- no GPU needed).
        "pip install torch --index-url https://download.pytorch.org/whl/cpu",
        # See modal_train_directional_gpu.py -- tensordict/torch._dynamo
        # probing triton at import time segfaults; not needed here (no
        # torch.compile), so strip it defensively.
        "pip uninstall -y triton || true",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "MUJOCO_GL": "osmesa"})
)

FOUR_HOURS_S = 60 * 60 * 4
SECRETS = [modal.Secret.from_name("hf-token")]


@APP.function(
    image=IMAGE,
    cpu=4.0,
    memory=8192,
    volumes={"/root/out": VOL},
    secrets=SECRETS,
    timeout=FOUR_HOURS_S,
)
def collect_and_train(
    samples_per_clip: int = 30_000,
    episode_len: int = 400,
    epochs: int = 200,
    seed: int = 0,
    run_name: str = "bc_directional_v1",
) -> str:
    """Collect a real teacher-rollout dataset, then BC-train on it.

    Args:
        samples_per_clip: Transitions collected per circular clip (2 clips
            total, so total dataset size = ``2 * samples_per_clip``).
        episode_len: Steps per collection episode before resetting.
        epochs: BC training epochs.
        seed: RNG seed for both collection and training.
        run_name: Output subdirectory name under the Modal volume.

    Returns:
        Path (inside the volume) where outputs were written.
    """
    import subprocess
    import sys

    sys.path.insert(0, "/root/myosuite4")
    import os

    os.chdir("/root/myosuite4")

    out_dir = Path("/root/out") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "bc_directional.npz"
    ckpt_path = out_dir / "policy_bc_best.pt"

    subprocess.run(
        [
            sys.executable,
            "scripts/bc_directional_collect.py",
            "--out",
            str(npz_path),
            "--samples-per-clip",
            str(samples_per_clip),
            "--episode-len",
            str(episode_len),
            "--seed",
            str(seed),
        ],
        check=True,
    )
    VOL.commit()

    subprocess.run(
        [
            sys.executable,
            "scripts/bc_directional_train.py",
            "--data",
            str(npz_path),
            "--out",
            str(ckpt_path),
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
        ],
        check=True,
    )
    VOL.commit()

    return str(out_dir)


@APP.local_entrypoint()
def main(
    samples_per_clip: int = 30_000,
    episode_len: int = 400,
    epochs: int = 200,
    seed: int = 0,
    run_name: str = "bc_directional_v1",
) -> None:
    result = collect_and_train.remote(
        samples_per_clip=samples_per_clip,
        episode_len=episode_len,
        epochs=epochs,
        seed=seed,
        run_name=run_name,
    )
    print(result)
